"""
V3 projectors for hidden-state driven Cache-to-Cache fusion.

The V3 projector skips the original KV concat/flatten/down-projection input path.
It consumes the sharer model hidden state directly and keeps the later C2C
projection, fusion, dynamic weighting, and residual cache update path.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from rosetta.model.projector import (
    AllInOneProjector,
    C2CProjector,
    Projector,
    RegularMLP,
)
from rosetta.utils.registry import (
    capture_init_args,
    get_projector_class,
    load_object,
    register_model,
    save_object,
)


def _nvtx_push(name: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def _nvtx_pop() -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


@register_model
@capture_init_args
class C2CHiddenStateProjector(Projector):
    """
    C2C V3 projector.

    Input:
        source_hidden_state: (B, N, hidden_size) from the mapped sharer layer.
        target_kv: receiver cache tuple, each (B, H_t, N, D_t).

    Difference from C2CProjector:
        - Removes source/target KV flatten + concat + Linear(hidden_dim) input path.
        - Uses sharer hidden state directly as the common hidden representation.
        - Keeps separate key/value MLP branches, scalar dynamic weights, gates,
          and receiver residual cache update.
    """

    uses_source_hidden_states = True

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        source_num_heads: int = 1,
        target_num_heads: int = 1,
        source_hidden_dim: Optional[int] = None,
        intermediate_dim: int = 1024,
        hidden_dim: Optional[int] = None,
        num_layers: int = 3,
        dropout: float = 0.1,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.001,
        anneal_steps: int = 1929,
        dtype: torch.dtype = torch.float32,
        zero_init: bool = False,
    ):
        super().__init__()

        assert num_layers >= 3, "num_layers must be >= 3"

        if source_hidden_dim is not None and source_hidden_dim != source_dim:
            raise ValueError(
                "source_hidden_dim is kept only for backward-compatible configs "
                f"and must equal source_dim ({source_dim}); got {source_hidden_dim}."
            )

        hidden_dim = source_dim if hidden_dim is None else hidden_dim
        if hidden_dim != source_dim:
            raise ValueError(
                "C2CHiddenStateProjector skips the input down-projection, so "
                f"hidden_dim must equal source hidden size ({source_dim}); got {hidden_dim}."
            )

        self.source_dim = source_dim
        self.target_dim = target_dim
        self.source_num_heads = source_num_heads
        self.target_num_heads = target_num_heads
        self.hidden_dim = hidden_dim

        out_dim = target_dim * target_num_heads

        # Direct hidden-state input: no concat(source KV, target KV), no input Linear.
        self.key_mlp1 = RegularMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=1,
            dropout=dropout,
            dtype=dtype,
        )
        self.value_mlp1 = RegularMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=1,
            dropout=dropout,
            dtype=dtype,
        )

        self.key_scalar_mlp2 = RegularMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=hidden_dim,
            num_layers=1,
            dropout=dropout,
            dtype=dtype,
        )
        self.value_scalar_mlp2 = RegularMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=hidden_dim,
            num_layers=1,
            dropout=dropout,
            dtype=dtype,
        )
        self.key_scalar_head = nn.Linear(hidden_dim, target_num_heads, dtype=dtype)
        self.value_scalar_head = nn.Linear(hidden_dim, target_num_heads, dtype=dtype)

        self.key_proj_mlp2 = RegularMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers - 2,
            dropout=dropout,
            dtype=dtype,
        )
        self.value_proj_mlp2 = RegularMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers - 2,
            dropout=dropout,
            dtype=dtype,
        )
        self.key_proj_out = nn.Linear(hidden_dim, out_dim, bias=True, dtype=dtype)
        self.value_proj_out = nn.Linear(hidden_dim, out_dim, bias=True, dtype=dtype)

        if zero_init:
            nn.init.zeros_(self.key_proj_out.weight)
            nn.init.zeros_(self.key_proj_out.bias)
            nn.init.zeros_(self.value_proj_out.weight)
            nn.init.zeros_(self.value_proj_out.bias)

        self.key_gate_logit = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.value_gate_logit = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.use_gumbel = True
        self.register_buffer("gate_temperature", torch.tensor(initial_temperature, dtype=dtype))
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.anneal_steps = anneal_steps
        self.scalar_temperature = 1.0

    def update_temperature(self, step: int):
        ratio = min(step / self.anneal_steps, 1.0)
        temp = self.initial_temperature * (self.final_temperature / self.initial_temperature) ** ratio
        self.gate_temperature.fill_(temp)

    def forward(
        self,
        source_hidden_state: Tensor,
        target_kv: Tuple[Tensor, Tensor],
        position_ids: Optional[Tensor] = None,
        max_pos: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        target_key, target_value = target_kv

        if source_hidden_state.dim() != 3:
            raise ValueError(
                "source_hidden_state must have shape (B, N, hidden_size); "
                f"got {tuple(source_hidden_state.shape)}"
            )

        B, Ht, N, Dt = target_key.shape
        if source_hidden_state.size(0) != B:
            raise ValueError(
                "source_hidden_state batch size must match target KV batch size; "
                f"got {source_hidden_state.size(0)} and {B}"
            )
        if source_hidden_state.size(1) != N:
            if source_hidden_state.size(1) < N:
                raise ValueError(
                    "source_hidden_state sequence length is shorter than target KV length; "
                    f"got {source_hidden_state.size(1)} and {N}"
                )
            source_hidden_state = source_hidden_state[:, -N:, :]
        if source_hidden_state.size(-1) != self.hidden_dim:
            raise ValueError(
                "source_hidden_state hidden size must match projector hidden_dim; "
                f"got {source_hidden_state.size(-1)} and {self.hidden_dim}"
            )

        _nvtx_push("fuser_v3.hidden_input")
        key_hidden = source_hidden_state.to(dtype=target_key.dtype)
        value_hidden = source_hidden_state.to(dtype=target_value.dtype)
        _nvtx_pop()

        _nvtx_push("fuser_v3.projection")
        key_hidden = self.key_mlp1(key_hidden)
        value_hidden = self.value_mlp1(value_hidden)
        _nvtx_pop()

        _nvtx_push("fuser_v3.feature_fusion")
        key_proj_hidden = self.key_proj_out(self.key_proj_mlp2(key_hidden))
        value_proj_hidden = self.value_proj_out(self.value_proj_mlp2(value_hidden))
        projected_key = key_proj_hidden.view(B, N, Ht, Dt).transpose(1, 2)
        projected_value = value_proj_hidden.view(B, N, Ht, Dt).transpose(1, 2)
        _nvtx_pop()

        _nvtx_push("fuser_v3.dynamic_weight")
        key_scalar = self.key_scalar_head(self.key_scalar_mlp2(key_hidden))
        value_scalar = self.value_scalar_head(self.value_scalar_mlp2(value_hidden))
        key_scalar = key_scalar.permute(0, 2, 1).unsqueeze(-1)
        value_scalar = value_scalar.permute(0, 2, 1).unsqueeze(-1)
        _nvtx_pop()

        _nvtx_push("fuser_v3.gate")
        key_gate_logit = self.key_gate_logit.view(1, 1, 1, 1)
        value_gate_logit = self.value_gate_logit.view(1, 1, 1, 1)
        if self.training and self.use_gumbel:
            u1 = torch.rand(B, Ht, N, 1, device=key_gate_logit.device, dtype=key_gate_logit.dtype)
            u2 = torch.rand(B, Ht, N, 1, device=value_gate_logit.device, dtype=value_gate_logit.dtype)
            g1 = -torch.log(-torch.log(u1 + 1e-20) + 1e-20)
            g2 = -torch.log(-torch.log(u2 + 1e-20) + 1e-20)
            key_gate = torch.sigmoid((key_gate_logit + g1) / self.gate_temperature)
            value_gate = torch.sigmoid((value_gate_logit + g2) / self.gate_temperature)
        else:
            key_gate = (key_gate_logit > 0).float()
            value_gate = (value_gate_logit > 0).float()
        _nvtx_pop()

        _nvtx_push("fuser_v3.dynamic_weight_norm")
        norm_key_scalar = torch.sigmoid(key_scalar / self.scalar_temperature)
        norm_value_scalar = torch.sigmoid(value_scalar / self.scalar_temperature)
        _nvtx_pop()

        _nvtx_push("fuser_v3.write_cache")
        output_key = target_key + key_gate * norm_key_scalar * projected_key
        output_value = target_value + value_gate * norm_value_scalar * projected_value
        _nvtx_pop()

        try:
            self.last_norm_key_scalar = norm_key_scalar.detach().cpu()
            self.last_norm_value_scalar = norm_value_scalar.detach().cpu()
            self.last_key_gate_logit = float(self.key_gate_logit.detach().cpu().item())
            self.last_value_gate_logit = float(self.value_gate_logit.detach().cpu().item())
        except Exception:
            pass

        return output_key, output_value


register_model("C2CProjectorV3")(C2CHiddenStateProjector)
register_model("HiddenStateC2CProjector")(C2CHiddenStateProjector)


def save_projector(obj: Projector, file_path: str) -> None:
    save_object(obj, file_path)


def load_projector(file_path: str, override_args: Optional[dict] = None) -> Projector:
    return load_object(file_path, get_projector_class, override_args)


def create_projector(projector_type: str, **kwargs) -> Projector:
    cls = get_projector_class(projector_type)
    return cls(**kwargs)
