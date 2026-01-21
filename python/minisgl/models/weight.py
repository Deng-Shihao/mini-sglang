from __future__ import annotations

import glob
import os
from typing import Dict, Optional

import safetensors
import torch
from huggingface_hub import snapshot_download
from minisgl.distributed import get_tp_info
from minisgl.utils import divide_up
from tqdm.asyncio import tqdm


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def _is_awq_weight(key: str) -> bool:
    """Check if a weight key is from AWQ quantization."""
    return any(suffix in key for suffix in [".qweight", ".qzeros", ".scales"])


def _get_weight_suffix(key: str) -> Optional[str]:
    """Get the AWQ weight suffix if present."""
    for suffix in [".qweight", ".qzeros", ".scales"]:
        if key.endswith(suffix):
            return suffix
    return None


def _shard_tensor(tensor: torch.Tensor, dim: int, rank: int, world_size: int, pack_factor: int = 1) -> torch.Tensor:
    """Shard a tensor along a dimension, accounting for pack factor on that dimension.
    
    For AWQ qweight and qzeros, the output dimension is packed by pack_factor=8.
    When sharding along the packed dimension, we need to divide by (world_size * pack_factor)
    and keep the tensor together.
    """
    if pack_factor > 1:
        # For packed dimensions, we shard the logical size
        # The tensor shape already reflects packing, so we shard normally
        pass
    return tensor.chunk(world_size, dim=dim)[rank]


def _shard_state_dict(
    state_dict: Dict[str, torch.Tensor],
    pack_factor: int = 1,
) -> Dict[str, torch.Tensor]:
    """Shard state dict for tensor parallelism.
    
    Args:
        state_dict: The state dict to shard
        pack_factor: AWQ pack factor (8 for 4-bit quantization), used for packed dims
    """
    shard_state_dict: Dict[str, torch.Tensor] = {}
    tp_info = get_tp_info()
    r = tp_info.rank
    n = tp_info.size

    # Column-parallel layers: split output dim (dim 0 for weight, dim 1 for AWQ qweight)
    SPLIT_DIM_0_LIST = [
        ".q_proj",
        ".k_proj",
        ".v_proj",
        ".gate_proj",
        ".up_proj",
    ]
    # Row-parallel layers: split input dim (dim 1 for weight, dim 0 for AWQ qweight)
    SPLIT_DIM_1_LIST = [
        ".o_proj",
        ".down_proj",
    ]

    for key, value in state_dict.items():
        is_awq = _is_awq_weight(key)
        suffix = _get_weight_suffix(key) if is_awq else None
        
        # Determine base key for matching
        base_key = key
        if suffix:
            base_key = key[:-len(suffix)]

        if any(base_key.count(sub) for sub in SPLIT_DIM_0_LIST):
            if is_awq:
                # AWQ: qweight is [K, N/pack], scales is [K/group, N], qzeros is [K/group, N/pack]
                # For column parallel (split output), split dim 1
                if suffix in [".qweight", ".qzeros"]:
                    # Packed dimension - shard normally, pack factor is already accounted for in shape
                    shard_state_dict[key] = value.chunk(n, dim=1)[r]
                else:  # .scales
                    shard_state_dict[key] = value.chunk(n, dim=1)[r]
            else:
                # Standard weight: [output, input], split dim 0
                shard_state_dict[key] = value.chunk(n, dim=0)[r]

        elif any(base_key.count(sub) for sub in SPLIT_DIM_1_LIST):
            if is_awq:
                # AWQ: For row parallel (split input), split dim 0
                # qweight is [K, N/pack], scales is [K/group, N], qzeros is [K/group, N/pack]
                if suffix in [".qweight"]:
                    shard_state_dict[key] = value.chunk(n, dim=0)[r]
                elif suffix in [".qzeros", ".scales"]:
                    # These depend on K/group_size, need to shard along dim 0
                    shard_state_dict[key] = value.chunk(n, dim=0)[r]
            else:
                # Standard weight: [output, input], split dim 1
                shard_state_dict[key] = value.chunk(n, dim=1)[r]

        elif base_key.count("lm_head") or base_key.count("embed_tokens"):
            # Embeddings are not quantized
            num_embeddings = value.shape[0]
            num_embeddings_per_partition = divide_up(num_embeddings, n)
            vocab_start_idx = r * num_embeddings_per_partition
            vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
            shard_state_dict[key] = value[vocab_start_idx:vocab_end_idx, :]
        else:
            shard_state_dict[key] = value

    return shard_state_dict


def _merge_awq_weights(
    state_dict: Dict[str, torch.Tensor],
    keys: list,
    new_base_key: str,
    dim: int,
) -> None:
    """Merge AWQ weight components (qweight, qzeros, scales) for multiple projections."""
    for suffix in [".qweight", ".qzeros", ".scales"]:
        weight_keys = [k + suffix for k in keys if k + suffix in state_dict]
        if weight_keys:
            tensors = [state_dict[k] for k in weight_keys]
            merged = torch.cat(tensors, dim=dim)
            state_dict[new_base_key + suffix] = merged
            for k in weight_keys:
                del state_dict[k]


def _merge_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Merge separate projection weights into combined weights.
    
    Handles both standard weights and AWQ quantized weights.
    """
    filtered_state_dict: Dict[str, torch.Tensor] = {}
    processed_keys = set()
    
    for key in list(state_dict.keys()):
        if key in processed_keys:
            continue
            
        # Check for AWQ weight suffixes
        is_awq = _is_awq_weight(key)
        suffix = _get_weight_suffix(key) if is_awq else None
        base_key = key[:-len(suffix)] if suffix else key
        
        # Handle Q/K/V projection merging
        if base_key.endswith(".q_proj"):
            prefix = base_key[:-len(".q_proj")]
            
            if is_awq:
                # Merge AWQ weights for Q, K, V
                q_base = prefix + ".q_proj"
                k_base = prefix + ".k_proj"
                v_base = prefix + ".v_proj"
                new_base = prefix + ".qkv_proj"
                
                for awq_suffix in [".qweight", ".qzeros", ".scales"]:
                    q_key = q_base + awq_suffix
                    k_key = k_base + awq_suffix
                    v_key = v_base + awq_suffix
                    
                    if q_key in state_dict and k_key in state_dict and v_key in state_dict:
                        # AWQ weights: concat along output dim (dim 1 for qweight/qzeros/scales)
                        merged = torch.cat([
                            state_dict[q_key],
                            state_dict[k_key],
                            state_dict[v_key]
                        ], dim=1)
                        filtered_state_dict[new_base + awq_suffix] = merged
                        processed_keys.update([q_key, k_key, v_key])
            else:
                # Standard weight merging
                q_proj = state_dict[key]
                k_key = key.replace(".q_proj", ".k_proj")
                v_key = key.replace(".q_proj", ".v_proj")
                k_proj = state_dict[k_key]
                v_proj = state_dict[v_key]
                new_key = key.replace(".q_proj", ".qkv_proj")
                filtered_state_dict[new_key] = torch.cat([q_proj, k_proj, v_proj], dim=0)
                processed_keys.update([key, k_key, v_key])
                
        # Handle gate/up projection merging
        elif base_key.endswith(".gate_proj"):
            prefix = base_key[:-len(".gate_proj")]
            
            if is_awq:
                gate_base = prefix + ".gate_proj"
                up_base = prefix + ".up_proj"
                new_base = prefix + ".gate_up_proj"
                
                for awq_suffix in [".qweight", ".qzeros", ".scales"]:
                    gate_key = gate_base + awq_suffix
                    up_key = up_base + awq_suffix
                    
                    if gate_key in state_dict and up_key in state_dict:
                        # AWQ weights: concat along output dim (dim 1)
                        merged = torch.cat([
                            state_dict[gate_key],
                            state_dict[up_key]
                        ], dim=1)
                        filtered_state_dict[new_base + awq_suffix] = merged
                        processed_keys.update([gate_key, up_key])
            else:
                gate_proj = state_dict[key]
                up_key = key.replace(".gate_proj", ".up_proj")
                up_proj = state_dict[up_key]
                new_key = key.replace(".gate_proj", ".gate_up_proj")
                filtered_state_dict[new_key] = torch.cat([gate_proj, up_proj], dim=0)
                processed_keys.update([key, up_key])
                
        # Skip already merged keys
        elif any(base_key.endswith(s) for s in [".k_proj", ".v_proj", ".up_proj"]):
            if key not in processed_keys:
                processed_keys.add(key)
            continue
            
        else:
            if key not in processed_keys:
                filtered_state_dict[key] = state_dict[key]
                processed_keys.add(key)
                
    return filtered_state_dict


def load_hf_weight(
    model_path: str,
    device: torch.device,
    pack_factor: int = 1,
) -> Dict[str, torch.Tensor]:
    """Load HuggingFace weights with optional AWQ support.
    
    Args:
        model_path: Path to model directory or HuggingFace model ID
        device: Target device for weights
        pack_factor: AWQ pack factor (8 for 4-bit), used for proper sharding
    """
    if os.path.isdir(model_path):
        hf_folder = model_path
    else:
        try:
            hf_folder = snapshot_download(
                model_path,
                allow_patterns=["*.safetensors"],
                tqdm_class=DisabledTqdm,
            )
        except Exception:
            raise ValueError(
                f"Model path '{model_path}' is neither a local directory nor a valid HuggingFace repository ID"
            )

    # find the all *.pt files in the hf_folder
    files = glob.glob(f"{hf_folder}/*.safetensors")
    state_dict: Dict[str, torch.Tensor] = {}
    for file in sorted(files):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)

    if get_tp_info().size > 1:
        state_dict = _shard_state_dict(state_dict, pack_factor=pack_factor)

    state_dict = {k: v.to(device) for k, v in state_dict.items()}
    return _merge_state_dict(state_dict)

