"""
Ablation Projector: A configurable projector for ablation studies based on C2CProjector.
Allows gradual removal of components to study their individual contributions.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Literal

from rosetta.utils.registry import register_model, capture_init_args
from rosetta.model.projector import Projector
from rosetta.model.projector import RegularMLP


@register_model
@capture_init_args
class AblationProjector(Projector):
    """
    Ablation study projector based on C2CProjector with configurable component removal.
    
    Ablation levels:
    0. Full C2C (baseline)
    1. Remove scalar weights (set to 1.0)
    2. Remove gates (set to 1.0) 
    3. Remove target contribution (only use source)
    4. Remove gates only (gates=1.0), keep scalars and target
    
    Each level builds on the previous one, allowing gradual degradation study.
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        source_num_heads: int = 1,
        target_num_heads: int = 1,
        intermediate_dim: int = 1024,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.1,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.001,
        anneal_steps: int = 1929,
        dtype: torch.dtype = torch.float32,
        
        # Ablation configuration
        ablation_level: int = 0,  # 0=full, 1=no_scalar, 2=no_gate+no_scalar, 3=no_target, 4=no_gate_only
        use_scalar_weights: bool = True,  # Can be overridden by ablation_level
        use_gates: bool = True,          # Can be overridden by ablation_level  
        use_target: bool = True,         # Can be overridden by ablation_level
    ):
        super().__init__()

        assert 0 <= ablation_level <= 4, "ablation_level must be 0, 1, 2, 3, or 4"

        # Dimensions
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.source_num_heads = source_num_heads
        self.target_num_heads = target_num_heads
        self.ablation_level = ablation_level

        # Override component usage based on ablation level
        if ablation_level == 4:
            # Special case: disable gates only, keep scalars and target
            use_scalar_weights = True
            use_gates = False
            use_target = True
        else:
            if ablation_level >= 1:
                use_scalar_weights = False
            if ablation_level >= 2: 
                use_gates = False
            if ablation_level >= 3:
                use_target = False
            
        self.use_scalar_weights = use_scalar_weights
        self.use_gates = use_gates
        self.use_target = use_target

        # Sizes
        in_dim = source_dim * source_num_heads
        out_dim = target_dim * target_num_heads

        # 1) concat(source_X, target_X) then project to hidden_dim
        # If not using target, only use source features
        if self.use_target:
            self.key_in = nn.Linear(in_dim + out_dim, hidden_dim, bias=True, dtype=dtype)
            self.value_in = nn.Linear(in_dim + out_dim, hidden_dim, bias=True, dtype=dtype)
        else:
            # Only use source features
            self.key_in = nn.Linear(in_dim, hidden_dim, bias=True, dtype=dtype)
            self.value_in = nn.Linear(in_dim, hidden_dim, bias=True, dtype=dtype)

        # 2) one-layer common embedding MLP to get intermediate representation (at hidden_dim)
        self.key_mlp1 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, num_layers=1, dropout=dropout, dtype=dtype)
        self.value_mlp1 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, num_layers=1, dropout=dropout, dtype=dtype)

        # 3a) intermediate representation → (L-2)-layer MLP for weights → project to head dim
        # Only build if using scalar weights
        if self.use_scalar_weights:
            self.key_scalar_mlp2 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=hidden_dim, num_layers=1, dropout=dropout, dtype=dtype)
            self.value_scalar_mlp2 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=hidden_dim, num_layers=1, dropout=dropout, dtype=dtype)
            self.key_scalar_head = nn.Linear(hidden_dim, target_num_heads, dtype=dtype)
            self.value_scalar_head = nn.Linear(hidden_dim, target_num_heads, dtype=dtype)

        # 3b) intermediate representation → (L-2)-layer MLP for projected_X → finally project hidden_dim → out_dim
        self.key_proj_mlp2 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, num_layers=num_layers-2, dropout=dropout, dtype=dtype)
        self.value_proj_mlp2 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, num_layers=num_layers-2, dropout=dropout, dtype=dtype)
        self.key_proj_out = nn.Linear(hidden_dim, out_dim, bias=True, dtype=dtype)
        self.value_proj_out = nn.Linear(hidden_dim, out_dim, bias=True, dtype=dtype)

        # Scalar key/value gate parameters and temperature schedule
        # Only build if using gates
        if self.use_gates:
            self.key_gate_logit = nn.Parameter(torch.tensor(0.0, dtype=dtype))
            self.value_gate_logit = nn.Parameter(torch.tensor(0.0, dtype=dtype))
            self.use_gumbel = True
            self.register_buffer("gate_temperature", torch.tensor(initial_temperature, dtype=dtype))
            self.initial_temperature = initial_temperature
            self.final_temperature = final_temperature
            self.anneal_steps = anneal_steps

        # Temperature for weight normalization
        self.scalar_temperature = 1.0

    def update_temperature(self, step: int):
        """Update temperature using exponential annealing schedule for gates."""
        if self.use_gates:
            ratio = min(step / self.anneal_steps, 1.0)
            temp = self.initial_temperature * (self.final_temperature / self.initial_temperature) ** ratio
            self.gate_temperature.fill_(temp)

    def forward(
        self,
        source_kv: Tuple[Tensor, Tensor],
        target_kv: Tuple[Tensor, Tensor],
        position_ids: Optional[Tensor] = None,
        max_pos: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        source_key, source_value = source_kv
        target_key, target_value = target_kv

        B, Hs, N, Ds = source_key.shape
        _, Ht, _, Dt = target_key.shape

        # Flatten heads
        source_key_flat = source_key.transpose(1, 2).contiguous().view(B, N, Hs * Ds)
        source_value_flat = source_value.transpose(1, 2).contiguous().view(B, N, Hs * Ds)
        target_key_flat = target_key.transpose(1, 2).contiguous().view(B, N, Ht * Dt)
        target_value_flat = target_value.transpose(1, 2).contiguous().view(B, N, Ht * Dt)

        # 1) Prepare input features based on ablation level
        if self.use_target:
            # Full C2C: concat source and target features
            key_cat = torch.cat([source_key_flat, target_key_flat], dim=-1)
            value_cat = torch.cat([source_value_flat, target_value_flat], dim=-1)
        else:
            # Ablation level 3: only use source features
            key_cat = source_key_flat
            value_cat = source_value_flat

        # 2) project to hidden dim
        key_hidden = self.key_in(key_cat)
        value_hidden = self.value_in(value_cat)

        # 3) one-layer common embedding MLP to get intermediate representation (at hidden_dim)
        key_hidden = self.key_mlp1(key_hidden)
        value_hidden = self.value_mlp1(value_hidden)

        # 4b) intermediate representation -> projected feature path
        key_proj_hidden = self.key_proj_out(self.key_proj_mlp2(key_hidden)) # (B, N, Ht * Dt)
        value_proj_hidden = self.value_proj_out(self.value_proj_mlp2(value_hidden)) # (B, N, Ht * Dt)
        projected_key = key_proj_hidden.view(B, N, Ht, Dt).transpose(1, 2) # (B, Ht, N, Dt)
        projected_value = value_proj_hidden.view(B, N, Ht, Dt).transpose(1, 2) # (B, Ht, N, Dt)

        # 4a) intermediate representation -> scalar path (if using scalar weights)
        if self.use_scalar_weights:
            key_scalar = self.key_scalar_head(self.key_scalar_mlp2(key_hidden))       # (B, N, Ht)
            value_scalar = self.value_scalar_head(self.value_scalar_mlp2(value_hidden)) # (B, N, Ht)
            key_scalar = key_scalar.permute(0, 2, 1).unsqueeze(-1)   # (B, Ht, N, 1)
            value_scalar = value_scalar.permute(0, 2, 1).unsqueeze(-1)  # (B, Ht, N, 1)
            # Normalize scalars
            norm_key_scalar = torch.sigmoid(key_scalar)
            norm_value_scalar = torch.sigmoid(value_scalar)
        else:
            # Ablation level 1+: set scalar weights to 1.0
            norm_key_scalar = torch.ones(B, Ht, N, 1, device=projected_key.device, dtype=projected_key.dtype)
            norm_value_scalar = torch.ones(B, Ht, N, 1, device=projected_value.device, dtype=projected_value.dtype)

        # Key/value gates (if using gates)
        if self.use_gates:
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
        else:
            # Gates disabled: set gates to 1.0 (always open)
            key_gate = torch.ones(B, Ht, N, 1, device=projected_key.device, dtype=projected_key.dtype)
            value_gate = torch.ones(B, Ht, N, 1, device=projected_value.device, dtype=projected_value.dtype)

        # Compute projected contribution
        projected_key_term = key_gate * norm_key_scalar * projected_key
        projected_value_term = value_gate * norm_value_scalar * projected_value

        # Compute target contribution (if using target)
        if self.use_target:
            # Full C2C: add target with projected
            output_key = target_key + projected_key_term
            output_value = target_value + projected_value_term
        else:
            # Ablation level 3: only use projected (no target)
            output_key = projected_key_term
            output_value = projected_value_term

        return output_key, output_value

    def get_ablation_info(self) -> dict:
        """Return information about current ablation configuration."""
        return {
            'ablation_level': self.ablation_level,
            'use_scalar_weights': self.use_scalar_weights,
            'use_gates': self.use_gates,
            'use_target': self.use_target,
            'description': self._get_ablation_description()
        }
    
    def _get_ablation_description(self) -> str:
        """Get human-readable description of current ablation level."""
        descriptions = {
            0: "Full C2C (baseline)",
            1: "No scalar weights (scalars=1.0)",
            2: "No gates (gates=1.0) + No scalar weights",
            3: "No target (source-only) + No gates + No scalar weights",
            4: "No gates (gates=1.0), keep scalars and target"
        }
        return descriptions.get(self.ablation_level, "Unknown ablation level")


# Convenience functions for creating specific ablation levels
def create_ablation_projector(
    source_dim: int,
    target_dim: int,
    source_num_heads: int = 1,
    target_num_heads: int = 1,
    ablation_level: int = 0,
    **kwargs
) -> AblationProjector:
    """Create an AblationProjector with specified ablation level."""
    return AblationProjector(
        source_dim=source_dim,
        target_dim=target_dim,
        source_num_heads=source_num_heads,
        target_num_heads=target_num_heads,
        ablation_level=ablation_level,
        **kwargs
    )


def create_full_c2c_projector(**kwargs) -> AblationProjector:
    """Create full C2C projector (ablation level 0)."""
    return create_ablation_projector(ablation_level=0, **kwargs)


def create_no_scalar_projector(**kwargs) -> AblationProjector:
    """Create projector without scalar weights (ablation level 1)."""
    return create_ablation_projector(ablation_level=1, **kwargs)


def create_no_gate_projector(**kwargs) -> AblationProjector:
    """Create projector without gates (ablation level 2)."""
    return create_ablation_projector(ablation_level=2, **kwargs)


def create_source_only_projector(**kwargs) -> AblationProjector:
    """Create source-only projector (ablation level 3)."""
    return create_ablation_projector(ablation_level=3, **kwargs)


def create_no_gate_only_projector(**kwargs) -> AblationProjector:
    """Create projector without gates but with scalar weights and target (ablation level 4)."""
    return create_ablation_projector(ablation_level=4, **kwargs)
