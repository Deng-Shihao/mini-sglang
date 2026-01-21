"""Quantization module for mini-sglang.

Currently supports:
- AWQ (Activation-aware Weight Quantization)
"""

from .awq import AWQConfig, AWQLinearMethod
from .base_config import LinearMethodBase, QuantizationConfig, set_weight_attrs
from .unquant import UnquantizedLinearMethod

# Map of quantization method names to their config classes
QUANTIZATION_METHODS = {
    "awq": AWQConfig,
}


def get_quantization_config(method_name: str) -> type:
    """Get the quantization config class for a given method name."""
    method_name = method_name.lower()
    if method_name not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Unknown quantization method: {method_name}. "
            f"Supported methods: {list(QUANTIZATION_METHODS.keys())}"
        )
    return QUANTIZATION_METHODS[method_name]


__all__ = [
    "AWQConfig",
    "AWQLinearMethod",
    "LinearMethodBase",
    "QuantizationConfig",
    "UnquantizedLinearMethod",
    "set_weight_attrs",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
