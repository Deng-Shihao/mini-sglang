from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from transformers import LlamaConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, float] | None


def load_quantization_config(model_path: str) -> Optional[Any]:
    """Load quantization config from model directory if present.
    
    Looks for AWQ config files: quant_config.json or quantize_config.json
    Returns the appropriate QuantizationConfig instance or None.
    """
    # Resolve HuggingFace model ID to local cache path if needed
    local_path = model_path
    if not os.path.isdir(model_path):
        try:
            from huggingface_hub import snapshot_download
            # This will return the cached path if already downloaded, or download if not
            local_path = snapshot_download(repo_id=model_path, local_files_only=True)
        except Exception:
            # If it fails (not cached or not a valid repo), return None
            return None
        
    # Try different AWQ config file names
    for config_name in ["quant_config.json", "quantize_config.json"]:
        config_path = os.path.join(local_path, config_name)
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            
            # Detect quantization method
            quant_method = config_dict.get("quant_method", "").lower()
            if quant_method == "awq" or "w_bit" in config_dict or "bits" in config_dict:
                from minisgl.layers.quantization import AWQConfig
                return AWQConfig.from_config(config_dict)
    
    return None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    quantization_config: Optional[Any] = None

    @classmethod
    def from_hf(
        cls,
        config: LlamaConfig,
        model_path: Optional[str] = None,
    ) -> ModelConfig:
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        
        # Try to load quantization config if model_path is provided
        quant_config = None
        if model_path:
            quant_config = load_quantization_config(model_path)
        
        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                scaling=getattr(config, "rope_scaling", None),
            ),
            quantization_config=quant_config,
        )
    
    @property
    def is_quantized(self) -> bool:
        """Check if the model is quantized."""
        return self.quantization_config is not None
    
    @property
    def quant_method_name(self) -> Optional[str]:
        """Get the quantization method name if quantized."""
        if self.quantization_config:
            return self.quantization_config.get_name()
        return None

