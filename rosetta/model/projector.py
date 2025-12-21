"""
Projector nn module for the unified memory
"""

import torch
import torch.nn as nn
from torch import Tensor
from transformers import Cache, DynamicCache
from typing import Optional, Tuple, Literal, Union
import copy
import math

from rosetta.utils.registry import register_model, get_projector_class, PROJECTOR_REGISTRY, capture_init_args, save_object, load_object

class Projector(nn.Module):
    """Base projector class for unified memory"""
    
    def forward(self, source_kv: Tuple[Tensor, Tensor], target_kv: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Project and combine the source key-value tensors to the target key-value tensors
        Args:
            source_kv: Tuple of (key, value) tensors, each (..., D_s) where ... are arbitrary leading dimensions
            target_kv: Tuple of (key, value) tensors, each (..., D_t) where ... are arbitrary leading dimensions
        Returns:
            Tuple of (key, value) tensors, each (..., D_t) with same leading dimensions as input
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def cache_project(self, source_kv_cache: Cache, target_kv_cache: Cache) -> Cache:
        """
        Project the source kv cache to the target kv cache
        """
        if not isinstance(source_kv_cache, DynamicCache) or not isinstance(target_kv_cache, DynamicCache):
            raise ValueError("Only DynamicCache is supported")
        
        projected_cache = DynamicCache()
        
        # Process each layer
        for layer_idx in range(len(source_kv_cache.key_cache)):
            source_key = source_kv_cache.key_cache[layer_idx]  # (B, H, N, D_s)
            source_value = source_kv_cache.value_cache[layer_idx]  # (B, H, N, D_s)
            
            # Get corresponding target tensors (for reference/combination)
            if layer_idx < len(target_kv_cache.key_cache):
                target_key = target_kv_cache.key_cache[layer_idx]  # (B, H, N, D_t)
                target_value = target_kv_cache.value_cache[layer_idx]  # (B, H, N, D_t)
            else:
                # If target cache doesn't have this layer, create dummy tensors
                B, H, N, D_s = source_key.shape
                D_t = source_key.shape[-1]  # Assume same dimension for simplicity
                target_key = torch.zeros(B, H, N, D_t, device=source_key.device, dtype=source_key.dtype)
                target_value = torch.zeros(B, H, N, D_t, device=source_value.device, dtype=source_value.dtype)
            
            # Reshape for forward pass: DynamicCache format (B, H, N, D) -> projector format (B, N, H, D)
            source_key_reshaped = source_key.transpose(1, 2)
            source_value_reshaped = source_value.transpose(1, 2)
            target_key_reshaped = target_key.transpose(1, 2)
            target_value_reshaped = target_value.transpose(1, 2)
            
            # Project using forward method with tuple input/output
            source_kv = (source_key_reshaped, source_value_reshaped)
            target_kv = (target_key_reshaped, target_value_reshaped)
            projected_key, projected_value = self.forward(source_kv, target_kv)
            
            # Reshape back: projector format (B, N, H, D) -> DynamicCache format (B, H, N, D)
            projected_key = projected_key.transpose(1, 2)
            projected_value = projected_value.transpose(1, 2)
            
            # Update cache
            projected_cache.update(projected_key, projected_value, layer_idx)
        
        return projected_cache

class ModernMLP(nn.Module):
    """
    Modern MLP with residual connections, layer normalization, and configurable architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        activation: str = "gelu",
        use_layer_norm: bool = True,
        use_residual: bool = True,
        dropout: float = 0.1,
        use_swiglu: bool = False,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_residual = use_residual and (input_dim == output_dim)
        self.use_swiglu = use_swiglu
        
        # Activation function
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            layer_output_dim = output_dim if i == num_layers - 1 else hidden_dim
            
            if self.use_swiglu and i < num_layers - 1:  # Don't use SwiGLU on output layer
                layer = SwiGLUBlock(layer_input_dim, layer_output_dim, dtype=dtype)
            else:
                layer = nn.Linear(layer_input_dim, layer_output_dim, dtype=dtype)
            
            self.layers.append(layer)
            
            # Add layer norm after each layer except the last one
            if use_layer_norm and i < num_layers - 1:
                self.layers.append(nn.LayerNorm(layer_output_dim, dtype=dtype))
            
            # Add activation after each layer except the last one
            if i < num_layers - 1 and not self.use_swiglu:
                self.layers.append(copy.deepcopy(self.activation))
            
            # Add dropout after activation
            if dropout > 0 and i < num_layers - 1:
                self.layers.append(nn.Dropout(dropout))
        
        # Residual projection if dimensions don't match
        if self.use_residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim, dtype=dtype)
        else:
            self.residual_proj = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with optional residual connection."""
        residual = x
        
        for layer in self.layers:
            x = layer(x)
        
        # Add residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            x = x + residual
        
        return x


class SwiGLUBlock(nn.Module):
    """SwiGLU activation block for modern transformer architectures."""
    
    def __init__(self, input_dim: int, output_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, output_dim, dtype=dtype)
        self.up_proj = nn.Linear(input_dim, output_dim, dtype=dtype)
        self.activation = nn.SiLU()
    
    def forward(self, x: Tensor) -> Tensor:
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return gate * up


@register_model
@capture_init_args
class AllInOneProjector(Projector):
    """
    Unified projector that consolidates all projection functionalities with modern patterns.
    
    Features:
    1. Gate logit granularity: scalar, token-wise, head-wise, or value-wise
    2. (DISABLED) Selector logit granularity: scalar, token-wise, head-wise, or value-wise
    3. Key/Value weight granularity: scalar, token-wise, head-wise, or value-wise
    4. Input-dependent gates and weights via MLP or parameters
    5. Optional concatenation with combiner networks
    6. Modern MLP architecture with residual connections and SwiGLU
    7. Configurable target preservation: choose between traditional blending or simplified projection
    8. Optional adding of target (self) signal to outputs via add_self
    
    Target Preservation Modes:
    - preserve_target_weight=True (default): output = (1-weight)*target + gate*selector*weight*projected
    - preserve_target_weight=False: output = target + gate*selector*weight*projected (no weight coefficient on target)
    
    Note: Selector functionality has been disabled/commented out.
    """
    
    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        source_num_heads: int = 1,
        target_num_heads: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layer_norm: bool = True,
        use_residual: bool = True,
        use_swiglu: bool = False,
        
        # Gate configuration
        gate_granularity: Literal["scalar", "token", "head", "head_merged", "value"] = "scalar",
        gate_depends_on_input: bool = False,
        gate_input_features: Optional[str] = "target_key",  # "target_key", "target_value", "both", "target_projected_key", "target_projected_value", "target_projected_both"
        gate_init_value: float = 0.0,
        
        # Weight configuration
        weight_granularity: Literal["scalar", "token", "head", "head_merged", "value"] = "scalar",
        weight_depends_on_input: bool = False,
        weight_input_features: Optional[str] = "target_key",  # "target_key", "target_value", "both", "target_projected_key", "target_projected_value", "target_projected_both"
        weight_init_value: float = 0.0,
        
        # Target preservation configuration
        preserve_target_weight: bool = True,  # If False, target won't be multiplied by (1 - normalized_weight)
        add_self: bool = True,  # If False, target (self) won't be added to outputs
        
        # Concat configuration
        use_concat: bool = False,
        # combiner_hidden_dim: int = 128,
        weight_hidden_dim: int = 1024,
        
        # Temperature and gumbel
        use_gumbel: bool = True,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.01,
        anneal_steps: int = 1360,
        scalar_temperature: float = 0.005,
        
        # Sequence length configuration
        max_sequence_length: int = 8192,  # Maximum sequence length for token-level parameters

        pos_emb: bool = False,

        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.source_num_heads = source_num_heads
        self.target_num_heads = target_num_heads
        self.hidden_dim = hidden_dim
        self.weight_hidden_dim = weight_hidden_dim
        self.max_sequence_length = max_sequence_length
        
        # Configuration
        self.gate_granularity = gate_granularity
        self.gate_depends_on_input = gate_depends_on_input
        self.gate_input_features = gate_input_features
        self.weight_granularity = weight_granularity
        self.weight_depends_on_input = weight_depends_on_input
        self.weight_input_features = weight_input_features
        self.preserve_target_weight = preserve_target_weight
        self.add_self = add_self
        self.use_concat = use_concat
        self.use_gumbel = use_gumbel
        self.scalar_temperature = scalar_temperature

        # Temperature annealing for gate only (removed selector temperature)
        self.register_buffer("gate_temperature", torch.tensor(initial_temperature, dtype=dtype))
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.anneal_steps = anneal_steps
        
        # Build projection networks
        self.key_projection = self._build_projection_mlp(
            source_dim * source_num_heads, 
            target_dim * target_num_heads,
            hidden_dim, num_layers, activation, use_layer_norm, 
            use_residual, dropout, use_swiglu, dtype
        )
        self.value_projection = self._build_projection_mlp(
            source_dim * source_num_heads,
            target_dim * target_num_heads,
            hidden_dim, num_layers, activation, use_layer_norm,
            use_residual, dropout, use_swiglu, dtype
        )
        
        # Build gate components
        self._build_gate_components(dtype)
        
        # Build weight components  
        self._build_weight_components(weight_init_value, dtype)

        # Build concat components if needed
        if self.use_concat:
            in_dim = target_dim * target_num_heads * 2
            out_dim = target_dim * target_num_heads
            self.key_combiner = nn.Linear(in_dim, out_dim, dtype=dtype)
            self.value_combiner = nn.Linear(in_dim, out_dim, dtype=dtype)
        
    def _build_projection_mlp(
        self, input_dim: int, output_dim: int, hidden_dim: int, 
        num_layers: int, activation: str, use_layer_norm: bool,
        use_residual: bool, dropout: float, use_swiglu: bool, dtype: torch.dtype
    ) -> ModernMLP:
        """Build modern MLP for projection."""
        return ModernMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            dropout=dropout,
            use_swiglu=use_swiglu,
            dtype=dtype
        )
    
    def _build_gate_components(self, dtype: torch.dtype):
        """Build gate logit components based on configuration."""
        if not self.gate_depends_on_input:
            # Parameter-based gate
            gate_shape = self._get_parameter_shape(self.gate_granularity)
            self.gate_logit = nn.Parameter(torch.zeros(gate_shape, dtype=dtype))
        else:
            # Input-dependent gate via MLP
            input_dim = self._get_gate_input_dim()
            output_dim = self._get_gate_output_dim()
            
            self.gate_generator = ModernMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2,
                activation="gelu",
                use_layer_norm=True,
                use_residual=False,
                dropout=0.1,
                dtype=dtype
            )
    
    def _build_weight_components(self, weight_init_value: float, dtype: torch.dtype):
        """Build weight components based on configuration."""
        if not self.weight_depends_on_input:
            # Parameter-based weights
            weight_shape = self._get_parameter_shape(self.weight_granularity)
            self.key_weight = nn.Parameter(torch.full(weight_shape, weight_init_value, dtype=dtype))
            self.value_weight = nn.Parameter(torch.full(weight_shape, weight_init_value, dtype=dtype))
        else:
            # Input-dependent weights via MLP
            input_dim = self._get_weight_input_dim()
            output_dim = self._get_weight_output_dim()
            
            # Shared hidden layer for efficiency
            self.weight_hidden = ModernMLP(
                input_dim=input_dim,
                output_dim=self.weight_hidden_dim,
                hidden_dim=self.weight_hidden_dim,
                num_layers=2,
                activation="gelu",
                use_layer_norm=True,
                use_residual=False,
                dropout=0.1,
                dtype=dtype
            )
            
            # Separate heads for key and value weights
            self.key_weight_head = nn.Linear(self.weight_hidden_dim, output_dim, dtype=dtype)
            self.value_weight_head = nn.Linear(self.weight_hidden_dim, output_dim, dtype=dtype)
    
    def _get_parameter_shape(self, granularity: str) -> tuple:
        """Get parameter shape based on granularity."""
        if granularity == "scalar":
            return ()  # Scalar
        elif granularity == "token":
            return (self.max_sequence_length,)  # Token-level parameters with max sequence length
        elif granularity == "head":
            return (self.max_sequence_length, self.target_num_heads)  # Token and head level parameters
        elif granularity == "head_merged":
            return (self.max_sequence_length, self.target_num_heads)  # Token and head level parameters
        elif granularity == "value":
            return (self.max_sequence_length, self.target_num_heads, self.target_dim)  # Token, head and value level parameters
        else:
            raise ValueError(f"Invalid granularity: {granularity}")
    
    def _get_gate_input_dim(self) -> int:
        """Get input dimension for gate generator."""
        base_dim = 0
        if self.gate_input_features == "target_key":
            base_dim = self.target_dim
        elif self.gate_input_features == "target_value":
            base_dim = self.target_dim
        elif self.gate_input_features == "both":
            base_dim = self.target_dim * 2
        elif self.gate_input_features == "target_projected_key":
            base_dim = self.target_dim * 2  # target_key + projected_key
        elif self.gate_input_features == "target_projected_value":
            base_dim = self.target_dim * 2  # target_value + projected_value
        elif self.gate_input_features == "target_projected_both":
            base_dim = self.target_dim * 4  # target_key + target_value + projected_key + projected_value
        else:
            raise ValueError(f"Invalid gate input features: {self.gate_input_features}")
        
        # Adjust for granularity processing strategy
        if self.gate_granularity == "scalar":
            # Scalar: process aggregated features across all heads
            return base_dim  # Use pooled features
        elif self.gate_granularity == "token":
            # Token: process merged head dimensions 
            return base_dim * self.target_num_heads  # Flatten (H, D) to (H*D)
        elif self.gate_granularity == "head_merged":
            # Head-merged: similar to token granularity, merge H and D
            return base_dim * self.target_num_heads  # (B, N, H*D)
        elif self.gate_granularity == "head":
            # Head-local: per head processing, do not merge heads
            return base_dim  # (B, H, N, D)
        else:  # value
            # Value: process per-head features
            return base_dim  # Keep per-head processing (B, H, N, D)
    
    def _get_gate_output_dim(self) -> int:
        """Get output dimension for gate generator."""
        if self.gate_granularity == "scalar":
            return 1
        elif self.gate_granularity == "token":
            return 1  # Per token
        elif self.gate_granularity == "head_merged":
            # Per token per head after merge: output one value per head
            return self.target_num_heads
        elif self.gate_granularity == "head":
            # Per token per head: scalar per head
            return 1
        elif self.gate_granularity == "value":
            return self.target_dim  # Per token per head per value (but processed per-head, so output D per head)
        else:
            raise ValueError(f"Invalid gate granularity: {self.gate_granularity}")
    
    def _get_weight_input_dim(self) -> int:
        """Get input dimension for weight generator."""
        base_dim = 0
        if self.weight_input_features == "target_key":
            base_dim = self.target_dim
        elif self.weight_input_features == "target_value":
            base_dim = self.target_dim
        elif self.weight_input_features == "both":
            base_dim = self.target_dim * 2
        elif self.weight_input_features == "target_projected_key":
            base_dim = self.target_dim * 2  # target_key + projected_key
        elif self.weight_input_features == "target_projected_value":
            base_dim = self.target_dim * 2  # target_value + projected_value
        elif self.weight_input_features == "target_projected_both":
            base_dim = self.target_dim * 4  # target_key + target_value + projected_key + projected_value
        else:
            raise ValueError(f"Invalid weight input features: {self.weight_input_features}")
        
        # Adjust for granularity processing strategy
        if self.weight_granularity == "scalar":
            # Scalar: process aggregated features across all heads
            return base_dim  # Use pooled features
        elif self.weight_granularity == "token":
            # Token: process merged head dimensions 
            return base_dim * self.target_num_heads  # Flatten (H, D) to (H*D)
        elif self.weight_granularity == "head_merged":
            # Head-merged: similar to token granularity, merge H and D
            return base_dim * self.target_num_heads  # (B, N, H*D)
        elif self.weight_granularity == "head":
            # Head-local: per head processing, do not merge heads
            return base_dim  # (B, H, N, D)
        else:  # value
            # Value: process per-head features
            return base_dim  # Keep per-head processing (B, H, N, D)
    
    def _get_weight_output_dim(self) -> int:
        """Get output dimension for weight generator."""
        if self.weight_granularity == "scalar":
            return 1
        elif self.weight_granularity == "token":
            return 1  # Per token
        elif self.weight_granularity == "head_merged":
            # Per token per head after merge: output one value per head
            return self.target_num_heads
        elif self.weight_granularity == "head":
            # Per token per head: scalar per head
            return 1
        elif self.weight_granularity == "value":
            return self.target_dim  # Per token per head per value (but processed per-head, so output D per head)
        else:
            raise ValueError(f"Invalid weight granularity: {self.weight_granularity}")
    
    def _generate_gates(self, target_key: Tensor, target_value: Tensor, projected_key: Tensor = None, projected_value: Tensor = None) -> Tensor:
        """Generate gate logits based on configuration."""
        if not self.gate_depends_on_input:
            # Use parameter-based gate
            return self.gate_logit
        else:
            # Generate input-dependent gate
            # First, prepare the base input features
            if self.gate_input_features == "target_key":
                base_input = target_key
            elif self.gate_input_features == "target_value":
                base_input = target_value
            elif self.gate_input_features == "both":
                base_input = torch.cat([target_key, target_value], dim=-1)
            elif self.gate_input_features == "target_projected_key":
                if projected_key is None:
                    raise ValueError("projected_key is required for target_projected_key input features")
                base_input = torch.cat([target_key, projected_key], dim=-1)
            elif self.gate_input_features == "target_projected_value":
                if projected_value is None:
                    raise ValueError("projected_value is required for target_projected_value input features")
                base_input = torch.cat([target_value, projected_value], dim=-1)
            elif self.gate_input_features == "target_projected_both":
                if projected_key is None or projected_value is None:
                    raise ValueError("Both projected_key and projected_value are required for target_projected_both input features")
                base_input = torch.cat([target_key, target_value, projected_key, projected_value], dim=-1)
            
            # Now process based on granularity
            # base_input shape: (B, H, N, D_input)
            B, H, N, D_input = base_input.shape
            
            if self.gate_granularity == "scalar":
                # For scalar granularity, aggregate all dimensions: (B, H, N, D_input) -> (B, D_input)
                gate_input = base_input.mean(dim=(1, 2))  # Average over heads and tokens
            elif self.gate_granularity == "token":
                # For token granularity, merge H and D_input dimensions: (B, H, N, D_input) -> (B, N, H*D_input)
                gate_input = base_input.transpose(1, 2).contiguous().view(B, N, H * D_input)
            elif self.gate_granularity == "head_merged":
                # For head granularity, merge H and D like token: (B, H, N, D_in) -> (B, N, H*D_in)
                gate_input = base_input.transpose(1, 2).contiguous().view(B, N, H * D_input)
            elif self.gate_granularity == "head":
                # For head granularity, keep per-head processing: (B, H, N, D_input)
                gate_input = base_input
            elif self.gate_granularity == "value":
                # For value granularity, keep per-head processing: (B, H, N, D_input)
                gate_input = base_input
            
            return self.gate_generator(gate_input)
    
    def _generate_weights(self, target_key: Tensor, target_value: Tensor, projected_key: Tensor = None, projected_value: Tensor = None) -> Tuple[Tensor, Tensor]:
        """Generate weights based on configuration."""
        if not self.weight_depends_on_input:
            # Use parameter-based weights
            return self.key_weight, self.value_weight
        else:
            # Generate input-dependent weights
            # First, prepare the base input features
            if self.weight_input_features == "target_key":
                base_input = target_key
            elif self.weight_input_features == "target_value":
                base_input = target_value
            elif self.weight_input_features == "both":
                base_input = torch.cat([target_key, target_value], dim=-1)
            elif self.weight_input_features == "target_projected_key":
                if projected_key is None:
                    raise ValueError("projected_key is required for target_projected_key input features")
                base_input = torch.cat([target_key, projected_key], dim=-1)
            elif self.weight_input_features == "target_projected_value":
                if projected_value is None:
                    raise ValueError("projected_value is required for target_projected_value input features")
                base_input = torch.cat([target_value, projected_value], dim=-1)
            elif self.weight_input_features == "target_projected_both":
                if projected_key is None or projected_value is None:
                    raise ValueError("Both projected_key and projected_value are required for target_projected_both input features")
                base_input = torch.cat([target_key, target_value, projected_key, projected_value], dim=-1)
            
            # Now process based on granularity
            # base_input shape: (B, H, N, D_input)
            B, H, N, D_input = base_input.shape
            
            if self.weight_granularity == "scalar":
                # For scalar granularity, aggregate all dimensions: (B, H, N, D_input) -> (B, D_input)
                weight_input = base_input.mean(dim=(1, 2))  # Average over heads and tokens
            elif self.weight_granularity == "token":
                # For token granularity, merge H and D_input dimensions: (B, H, N, D_input) -> (B, N, H*D_input)
                weight_input = base_input.transpose(1, 2).contiguous().view(B, N, H * D_input)
            elif self.weight_granularity == "head_merged":
                # For head granularity, merge H and D like token: (B, H, N, D_in) -> (B, N, H*D_in)
                weight_input = base_input.transpose(1, 2).contiguous().view(B, N, H * D_input)
            elif self.weight_granularity == "head":
                # For head granularity, keep per-head processing: (B, H, N, D_input)
                weight_input = base_input
            elif self.weight_granularity == "value":
                # For value granularity, keep per-head processing: (B, H, N, D_input)
                weight_input = base_input
    
            weight_hidden = self.weight_hidden(weight_input)
            key_weight = self.key_weight_head(weight_hidden)
            value_weight = self.value_weight_head(weight_hidden)

            return key_weight, value_weight
    
    def _apply_gumbel_sigmoid(self, gate_logit: Tensor) -> Tensor:
        """Apply Gumbel sigmoid trick for training."""
        if self.training and self.use_gumbel:
            gumbel_noise = self._sample_gumbel(gate_logit.shape, gate_logit.device, gate_logit.dtype)
            return torch.sigmoid((gate_logit + gumbel_noise) / self.gate_temperature)
        else:
            return (gate_logit > 0).float()
    
    @staticmethod
    def _sample_gumbel(shape: tuple, device: torch.device, dtype: torch.dtype, eps: float = 1e-20) -> Tensor:
        """Sample from Gumbel distribution."""
        u = torch.rand(shape, device=device, dtype=dtype)
        return -torch.log(-torch.log(u + eps) + eps)
    
    def _reshape_for_granularity(self, tensor: Tensor, granularity: str, target_shape: tuple) -> Tensor:
        """Reshape tensor to match target shape based on granularity."""
        B, H, N, D = target_shape
        
        if granularity == "scalar":
            # Scalar -> (B, H, N, D)
            return tensor.view(1, 1, 1, 1).expand(B, H, N, D)
        elif granularity == "token":
            # (max_seq_len,) -> (B, H, N, D) - slice to actual sequence length
            token_params = tensor[:N]  # Take first N tokens
            return token_params.view(1, 1, N, 1).expand(B, H, N, D)
        elif granularity == "head":
            # (max_seq_len, H) -> (B, H, N, D) - slice to actual sequence length, each token each head independent
            head_params = tensor[:N, :]  # Take first N tokens, all heads: (N, H)
            return head_params.view(1, N, H, 1).transpose(1, 2).expand(B, H, N, D)  # (1, N, H, 1) -> (1, H, N, 1) -> (B, H, N, D)
        elif granularity == "head_merged":
            raise NotImplementedError
        elif granularity == "value":
            # (max_seq_len, H, D) -> (B, H, N, D) - slice to actual sequence length, each token each head each value independent
            value_params = tensor[:N, :, :]  # Take first N tokens: (N, H, D)
            return value_params.view(1, N, H, D).transpose(1, 2).expand(B, H, N, D)  # (1, N, H, D) -> (1, H, N, D) -> (B, H, N, D)
        else:
            raise ValueError(f"Invalid granularity: {granularity}")
    
    def update_temperature(self, step: int):
        """Update temperature using exponential annealing schedule for gate only."""
        # Update gate temperature
        gate_ratio = min(step / self.anneal_steps, 1.0)
        gate_temp = self.initial_temperature * (self.final_temperature / self.initial_temperature) ** gate_ratio
        self.gate_temperature.fill_(gate_temp)
    
    
    def forward(self, source_kv: Tuple[Tensor, Tensor], target_kv: Tuple[Tensor, Tensor], position_ids: Optional[Tensor] = None, max_pos: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with unified projection logic.
        
        Args:
            source_kv: Tuple of (key, value) tensors, each (B, H_s, N, D_s)
            target_kv: Tuple of (key, value) tensors, each (B, H_t, N, D_t)
            position_ids: Position ids tensor (B, N), optional, required if pos_emb=True
        Returns:
            Tuple of (key, value) tensors, each (B, H_t, N, D_t)
        """
        source_key, source_value = source_kv
        target_key, target_value = target_kv
        
        # Get shapes
        B, H_s, N, D_s = source_key.shape
        _, H_t, _, D_t = target_key.shape
        
        # Reshape for projection: (B, H, N, D) -> (B, N, H*D)
        source_key_flat = source_key.transpose(1, 2).contiguous().view(B, N, H_s * D_s)
        source_value_flat = source_value.transpose(1, 2).contiguous().view(B, N, H_s * D_s)
        
        # Project source to target dimension
        projected_key_flat = self.key_projection(source_key_flat)  # (B, N, H_t * D_t)
        projected_value_flat = self.value_projection(source_value_flat)  # (B, N, H_t * D_t)
        
        # Handle concatenation if enabled
        if self.use_concat:
            target_key_flat = target_key.transpose(1, 2).contiguous().view(B, N, H_t * D_t)
            target_value_flat = target_value.transpose(1, 2).contiguous().view(B, N, H_t * D_t)
            
            # Concatenate and combine
            combined_key = torch.cat([projected_key_flat, target_key_flat], dim=-1)
            combined_value = torch.cat([projected_value_flat, target_value_flat], dim=-1)
            
            final_projected_key_flat = self.key_combiner(combined_key)
            final_projected_value_flat = self.value_combiner(combined_value)
        else:
            final_projected_key_flat = projected_key_flat
            final_projected_value_flat = projected_value_flat
        
        # Reshape back: (B, N, H_t * D_t) -> (B, H_t, N, D_t)
        projected_key = final_projected_key_flat.view(B, N, H_t, D_t).transpose(1, 2)
        projected_value = final_projected_value_flat.view(B, N, H_t, D_t).transpose(1, 2)
        
        # Generate gates, selectors and weights (may need projected tensors for input features)
        needs_projected_for_gate = self.gate_depends_on_input and self.gate_input_features in [
            "target_projected_key", "target_projected_value", "target_projected_both"
        ]
        needs_projected_for_weight = self.weight_depends_on_input and self.weight_input_features in [
            "target_projected_key", "target_projected_value", "target_projected_both"
        ]
        
        if needs_projected_for_gate or needs_projected_for_weight:
            gate_logit = self._generate_gates(target_key, target_value, projected_key, projected_value)
            key_weight, value_weight = self._generate_weights(target_key, target_value, projected_key, projected_value)
        else:
            gate_logit = self._generate_gates(target_key, target_value)
            key_weight, value_weight = self._generate_weights(target_key, target_value)
        
        # Reshape gates and weights to match target shape
        target_shape = (B, H_t, N, D_t)
        if self.gate_depends_on_input:
            # Reshape based on gate granularity - all preserve token dimension N
            if self.gate_granularity == "scalar":
                # For scalar, gate_logit is already (B, 1) from MLP, just expand
                gate_logit = gate_logit.view(B, 1, 1, 1).expand(target_shape)
            elif self.gate_granularity == "token":
                gate_logit = gate_logit.unsqueeze(1).unsqueeze(-1).expand(target_shape)  # (B, N, 1) -> (B, H, N, D)
            elif self.gate_granularity == "head_merged":
                # (B, N, H) -> (B, H, N, D) - per token per head, broadcast over D
                gate_logit = gate_logit.permute(0, 2, 1).unsqueeze(-1).expand(B, H_t, N, D_t)
            elif self.gate_granularity == "head":
                # (B, H, N, 1) -> (B, H, N, D) - per token per head scalar, broadcast over D
                gate_logit = gate_logit.expand(B, H_t, N, D_t)
            elif self.gate_granularity == "value":
                # (B, H, N, D) -> (B, H, N, D) - each token each head each value has one value
                pass  # Already in correct shape
        else:
            gate_logit = self._reshape_for_granularity(gate_logit, self.gate_granularity, target_shape)

        if self.weight_depends_on_input:
            # Reshape weights based on granularity - all preserve token dimension N
            if self.weight_granularity == "scalar":
                # For scalar, weights are already (B, 1) from MLP, just expand
                key_weight = key_weight.view(B, 1, 1, 1).expand(target_shape)
                value_weight = value_weight.view(B, 1, 1, 1).expand(target_shape)
            elif self.weight_granularity == "token":
                key_weight = key_weight.unsqueeze(1).expand(target_shape)  # (B, N, 1) -> (B, H, N, D)
                value_weight = value_weight.unsqueeze(1).expand(target_shape)
            elif self.weight_granularity == "head_merged":
                # (B, N, H) -> (B, H, N, D) - per token per head, broadcast over D
                key_weight = key_weight.permute(0, 2, 1).unsqueeze(-1).expand(B, H_t, N, D_t)
                value_weight = value_weight.permute(0, 2, 1).unsqueeze(-1).expand(B, H_t, N, D_t)
            elif self.weight_granularity == "head":
                # (B, H, N, 1) -> (B, H, N, D) - per token per head scalar, broadcast over D
                key_weight = key_weight.expand(B, H_t, N, D_t)
                value_weight = value_weight.expand(B, H_t, N, D_t)
            elif self.weight_granularity == "value":
                # (B, H, N, D) -> (B, H, N, D) - each token each head each value has one value
                pass  # Already in correct shape
        else:
            key_weight = self._reshape_for_granularity(key_weight, self.weight_granularity, target_shape)
            value_weight = self._reshape_for_granularity(value_weight, self.weight_granularity, target_shape)
        
        # Apply gating and selection
        gate = self._apply_gumbel_sigmoid(gate_logit)
        
        # Normalize weights using dynamic temperature
        normalized_key_weight = torch.sigmoid(key_weight / self.scalar_temperature)
        normalized_value_weight = torch.sigmoid(value_weight / self.scalar_temperature)
        
        # Final combination
        # Compute projected contribution (always present)
        projected_key_term = gate * normalized_key_weight * projected_key
        projected_value_term = gate * normalized_value_weight * projected_value

        # Compute target (self) contribution depending on flags
        if self.add_self:
            if self.preserve_target_weight:
                target_key_term = (1 - normalized_key_weight) * target_key
                target_value_term = (1 - normalized_value_weight) * target_value
            else:
                target_key_term = target_key
                target_value_term = target_value
        else:
            target_key_term = torch.zeros_like(target_key)
            target_value_term = torch.zeros_like(target_value)

        # Final outputs
        output_key = target_key_term + projected_key_term
        output_value = target_value_term + projected_value_term
        
        return (output_key, output_value)

class QwenStyleLayer(nn.Module):
    """
    One Qwen3-style MLP sublayer:
      y = x + Dropout( down( SiLU(gate(LN(x))) * up(LN(x)) ) )
    - Pre-norm with RMSNorm
    - Bias-free linears
    """
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6, dtype=dtype)
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up   = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.act  = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h = self.act(self.gate(h)) * self.up(h)  # SwiGLU
        h = self.down(h)
        h = self.drop(h)
        return x + h

class StandardFFNLayer(nn.Module):
    """
    Pre-norm RMSNorm, classic MLP:
      y = x + Dropout( W2( Act( W1( RMSNorm(x) ) ) ) )
    - No SwiGLU: single hidden nonlinearity (GELU/ReLU/SiLU)
    - Bias-free linears (common in modern LLM FFNs)
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
        activation: str = "gelu",
    ):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6, dtype=dtype)
        self.w1   = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.w2   = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        act = activation.lower()
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h = self.act(self.w1(h))
        h = self.w2(h)
        h = self.drop(h)
        return x + h

class RegularMLP(nn.Module):
    """
    Qwen3-style stacked MLP operating at a fixed hidden size.
    - No input/output projections; caller is responsible for projections.
    - num_layers repeats of Qwen-style FFN sublayer (pre-RMSNorm, SwiGLU, bias-free)
    """
    def __init__(
        self,
        hidden_dim: int = 1024,
        intermediate_dim: int = 3072,
        num_layers: int = 3,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.blocks = nn.ModuleList([
            StandardFFNLayer(hidden_size=hidden_dim, intermediate_size=intermediate_dim, dropout=dropout, dtype=dtype)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x

@register_model
@capture_init_args
class C2CProjector(Projector):
    """
    Concise projector specialized to a fixed C2C configuration using StandardMLP.
    - Projections: StandardMLP (pre-RMSNorm, SwiGLU, residual per sublayer)
    - Concat: enabled, followed by linear combiner to target size
    - Gate: scalar parameter with Gumbel-sigmoid during training
    - Weights: input-dependent, head_merged granularity using target and projected key
    - Target preservation: add_self=True, preserve_target_weight=False
    - Temperatures: annealed gate temperature (1.0 -> 0.001 over 1929 steps), scalar_temperature=1.0
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
        zero_init: bool = False
    ):
        super().__init__()

        assert num_layers >= 3, "num_layers must be >= 3"

        # Dimensions
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.source_num_heads = source_num_heads
        self.target_num_heads = target_num_heads

        # Sizes
        in_dim = source_dim * source_num_heads
        out_dim = target_dim * target_num_heads

        # 1) concat(source_X, target_X) then project to hidden_dim
        self.key_in = nn.Linear(in_dim + out_dim, hidden_dim, bias=True, dtype=dtype)
        self.value_in = nn.Linear(in_dim + out_dim, hidden_dim, bias=True, dtype=dtype)

        # 2) one-layer common embedding MLP to get intermediate representation (at hidden_dim)
        self.key_mlp1 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, num_layers=1, dropout=dropout, dtype=dtype)
        self.value_mlp1 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, num_layers=1, dropout=dropout, dtype=dtype)

        # 3a) intermediate representation → (L-2)-layer MLP for weights → project to head dim
        self.key_scalar_mlp2 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=hidden_dim, num_layers=1, dropout=dropout, dtype=dtype)
        self.value_scalar_mlp2 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=hidden_dim, num_layers=1, dropout=dropout, dtype=dtype)
        self.key_scalar_head = nn.Linear(hidden_dim, target_num_heads, dtype=dtype)
        self.value_scalar_head = nn.Linear(hidden_dim, target_num_heads, dtype=dtype)

        # 3b) intermediate representation → (L-2)-layer MLP for projected_X → finally project hidden_dim → out_dim
        self.key_proj_mlp2 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, num_layers=num_layers-2, dropout=dropout, dtype=dtype)
        self.value_proj_mlp2 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, num_layers=num_layers-2, dropout=dropout, dtype=dtype)
        self.key_proj_out = nn.Linear(hidden_dim, out_dim, bias=True, dtype=dtype)
        self.value_proj_out = nn.Linear(hidden_dim, out_dim, bias=True, dtype=dtype)
        
        if zero_init:
            print("Initializing projector weights to zero")
            nn.init.zeros_(self.key_proj_out.weight)
            nn.init.zeros_(self.key_proj_out.bias)
            nn.init.zeros_(self.value_proj_out.weight)
            nn.init.zeros_(self.value_proj_out.bias)
        
        # Scalar key/value gate parameters and temperature schedule
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

        # 1) concat source and target features along channel
        key_cat = torch.cat([source_key_flat, target_key_flat], dim=-1)
        value_cat = torch.cat([source_value_flat, target_value_flat], dim=-1)

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
    
        # 4a) intermediate representation -> scalar path
        key_scalar = self.key_scalar_head(self.key_scalar_mlp2(key_hidden))       # (B, N, Ht)
        value_scalar = self.value_scalar_head(self.value_scalar_mlp2(value_hidden)) # (B, N, Ht)
        key_scalar = key_scalar.permute(0, 2, 1).unsqueeze(-1)   # (B, Ht, N, 1)
        value_scalar = value_scalar.permute(0, 2, 1).unsqueeze(-1)  # (B, Ht, N, 1)

        # Key/value gates: element-wise Gumbel noise with scalar logits (broadcast over channels)
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

        # Normalize scalars (scalar_temperature=1.0)
        norm_key_scalar = torch.sigmoid(key_scalar)
        norm_value_scalar = torch.sigmoid(value_scalar)

        # Combine (preserve_target_weight=False, add_self=True)
        output_key = target_key + key_gate * norm_key_scalar * projected_key
        output_value = target_value + value_gate * norm_value_scalar * projected_value

        # Expose capture attributes for downstream analysis scripts
        try:
            # Store normalized scalars (detach to avoid autograd, keep device-agnostic via CPU)
            self.last_norm_key_scalar = norm_key_scalar.detach().cpu()
            self.last_norm_value_scalar = norm_value_scalar.detach().cpu()
            # Store gate logits as python floats (parameters are scalar)
            self.last_key_gate_logit = float(self.key_gate_logit.detach().cpu().item())
            self.last_value_gate_logit = float(self.value_gate_logit.detach().cpu().item())
        except Exception:
            # Best-effort capture; never break forward path
            pass

        return output_key, output_value

def save_projector(obj: Projector, file_path: str) -> None:
    save_object(obj, file_path)

def load_projector(file_path: str, override_args: Optional[dict] = None) -> Projector:
    return load_object(file_path, get_projector_class, override_args)

def create_projector(projector_type: str, **kwargs) -> Projector:
    """
    Factory function to create a projector based on type.
    
    Args:
        projector_type: String indicating the type of projector
        **kwargs: Additional arguments to pass to the projector constructor
        
    Returns:
        An instance of the appropriate projector
    """
    # Prefer using the unified registry getter (handles case-insensitive keys)
    try:
        cls = get_projector_class(projector_type)
    except ValueError as e:
        raise e
    return cls(**kwargs)