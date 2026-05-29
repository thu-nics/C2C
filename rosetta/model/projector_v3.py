"""
Projector v3: feed sharer hidden states directly into the C2C fusion path.

Unlike C2CProjector, this skips source/target KV flatten + concat + input
projection. The mapped sharer layer hidden state is used as the key/value MLP
input, and the output is still written as a residual into receiver KV cache.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from rosetta.model.projector import (
    Projector,
    RegularMLP,
    create_projector as _base_create_projector,
    save_projector,
)
from rosetta.utils.registry import (
    capture_init_args,
    get_projector_class,
    load_object,
    register_model,
)


@register_model
@capture_init_args
class C2CProjectorV3(Projector):
    """
    C2C projector that consumes sharer hidden states instead of sharer/receiver KV.

    Input:
        source_hidden_state: (B, N, source_hidden_dim), specifically the hidden
            state used as input to the mapped sharer layer.
        target_kv: receiver target layer KV slice, each (B, Ht, N, Dt).

    Output:
        receiver KV slice with source-derived residual added:
            target + gate * dynamic_weight * projected_hidden
    """

    uses_source_hidden_state = True

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

        self.source_dim = source_dim
        self.target_dim = target_dim
        self.source_num_heads = source_num_heads
        self.target_num_heads = target_num_heads
        self.source_hidden_dim = source_hidden_dim or (source_dim * source_num_heads)
        self.hidden_dim = hidden_dim or self.source_hidden_dim

        if self.hidden_dim != self.source_hidden_dim:
            raise ValueError(
                "C2CProjectorV3 skips the input projection, so hidden_dim must "
                "match source_hidden_dim. Omit hidden_dim or set it to the "
                "sharer model hidden size."
            )

        out_dim = target_dim * target_num_heads

        # The source hidden state enters here directly. There is no KV flatten,
        # source/target concat, or input down projection in v3.
        self.key_mlp1 = RegularMLP(
            hidden_dim=self.hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=1,
            dropout=dropout,
            dtype=dtype,
        )
        self.value_mlp1 = RegularMLP(
            hidden_dim=self.hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=1,
            dropout=dropout,
            dtype=dtype,
        )

        self.key_scalar_mlp2 = RegularMLP(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.hidden_dim,
            num_layers=1,
            dropout=dropout,
            dtype=dtype,
        )
        self.value_scalar_mlp2 = RegularMLP(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.hidden_dim,
            num_layers=1,
            dropout=dropout,
            dtype=dtype,
        )
        self.key_scalar_head = nn.Linear(self.hidden_dim, target_num_heads, dtype=dtype)
        self.value_scalar_head = nn.Linear(self.hidden_dim, target_num_heads, dtype=dtype)

        self.key_proj_mlp2 = RegularMLP(
            hidden_dim=self.hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers - 2,
            dropout=dropout,
            dtype=dtype,
        )
        self.value_proj_mlp2 = RegularMLP(
            hidden_dim=self.hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers - 2,
            dropout=dropout,
            dtype=dtype,
        )
        self.key_proj_out = nn.Linear(self.hidden_dim, out_dim, bias=True, dtype=dtype)
        self.value_proj_out = nn.Linear(self.hidden_dim, out_dim, bias=True, dtype=dtype)

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
                "C2CProjectorV3 expects source_hidden_state with shape "
                f"(B, N, D), got {tuple(source_hidden_state.shape)}"
            )

        B, N, D = source_hidden_state.shape
        _, Ht, Nt, Dt = target_key.shape
        if N != Nt:
            raise ValueError(
                f"source hidden length ({N}) must match target KV length ({Nt})"
            )
        if D != self.source_hidden_dim:
            raise ValueError(
                f"source hidden dim ({D}) does not match configured "
                f"source_hidden_dim ({self.source_hidden_dim})"
            )

        source_hidden_state = source_hidden_state.to(dtype=self.key_proj_out.weight.dtype)
        key_hidden = self.key_mlp1(source_hidden_state)
        value_hidden = self.value_mlp1(source_hidden_state)

        key_proj_hidden = self.key_proj_out(self.key_proj_mlp2(key_hidden))
        value_proj_hidden = self.value_proj_out(self.value_proj_mlp2(value_hidden))
        projected_key = key_proj_hidden.view(B, N, Ht, Dt).transpose(1, 2)
        projected_value = value_proj_hidden.view(B, N, Ht, Dt).transpose(1, 2)

        key_scalar = self.key_scalar_head(self.key_scalar_mlp2(key_hidden))
        value_scalar = self.value_scalar_head(self.value_scalar_mlp2(value_hidden))
        key_scalar = key_scalar.permute(0, 2, 1).unsqueeze(-1)
        value_scalar = value_scalar.permute(0, 2, 1).unsqueeze(-1)

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

        norm_key_scalar = torch.sigmoid(key_scalar / self.scalar_temperature)
        norm_value_scalar = torch.sigmoid(value_scalar / self.scalar_temperature)

        output_key = target_key + key_gate * norm_key_scalar * projected_key
        output_value = target_value + value_gate * norm_value_scalar * projected_value

        try:
            self.last_norm_key_scalar = norm_key_scalar.detach().cpu()
            self.last_norm_value_scalar = norm_value_scalar.detach().cpu()
            self.last_key_gate_logit = float(self.key_gate_logit.detach().cpu().item())
            self.last_value_gate_logit = float(self.value_gate_logit.detach().cpu().item())
        except Exception:
            pass

        return output_key, output_value


def load_projector(file_path: str, override_args: Optional[dict] = None) -> Projector:
    return load_object(file_path, get_projector_class, override_args)


def create_projector(projector_type: str, **kwargs) -> Projector:
    if projector_type == "C2CProjectorV3":
        return C2CProjectorV3(**kwargs)
    return _base_create_projector(projector_type, **kwargs)


__all__ = [
    "C2CProjectorV3",
    "Projector",
    "create_projector",
    "load_projector",
    "save_projector",
]
