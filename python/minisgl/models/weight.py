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


def _shard_tensor(
    tensor: torch.Tensor, dim: int, rank: int, world_size: int, pack_factor: int = 1
) -> torch.Tensor:
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
    state_dict: Dict[str, torch.Tensor], pack_factor: int = 1
) -> Dict[str, torch.Tensor]:
    shard_state_dict: Dict[str, torch.Tensor] = {}
    tp_info = get_tp_info()
    r = tp_info.rank
    n = tp_info.size

    SPLIT_DIM_0_LIST = [
        ".q_proj",
        ".k_proj",
        ".v_proj",
        ".gate_proj",
        ".up_proj",
    ]

    SPLIT_DIM_1_LIST = [
        ".o_proj",
        ".down_proj",
    ]

    for key, value in state_dict.items():
        if any(key.count(sub) for sub in SPLIT_DIM_0_LIST):
            shard_state_dict[key] = value.chunk(n, dim=0)[r]

        elif any(key.count(sub) for sub in SPLIT_DIM_1_LIST):
            shard_state_dict[key] = value.chunk(n, dim=1)[r]

        elif key.count("lm_head") or key.count("embed_tokens"):
            num_embeddings = value.shape[0]
            num_embeddings_per_partition = divide_up(num_embeddings, n)
            vocab_start_idx = r * num_embeddings_per_partition
            vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
            shard_state_dict[key] = value[vocab_start_idx:vocab_end_idx, :]

        else:
            shard_state_dict[key] = value

    return shard_state_dict


def _merge_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Merge separate projection weights into combined weights.

    Handles both standard weights and AWQ quantized weights.

    unquantized:
        model.layers.0.self_attn.q_proj.weight
        model.layers.0.self_attn.k_proj.weight
        model.layers.0.self_attn.v_proj.weight

    awq:
        model.layers.0.self_attn.q_proj.qweight
        model.layers.0.self_attn.q_proj.qzeros
        model.layers.0.self_attn.q_proj.scales

    """
    filtered_state_dict: Dict[str, torch.Tensor] = {}

    for key in list(state_dict.keys()):
        # model.layers.0.self_attn.q_proj.weight -> unquantize

        # model.layers.0.self_attn.q_proj.qweight -> awq
        # model.layers.0.self_attn.q_proj.qzeros
        # model.layers.0.self_attn.q_proj.scales
        if key.count(".q_proj"):
            q_proj = state_dict[key]  # q_proj.qweight
            k_proj = state_dict[key.replace(".q_proj", ".k_proj")]  # k_proj.
            v_proj = state_dict[key.replace(".q_proj", ".v_proj")]
            new_key = key.replace(".q_proj", ".qkv_proj")  # new_key = ".qkv_proj"

            # save q, k, v -> qkv_proj : cat qkv
            # q_proj.qweight, k_proj.qweight, v_proj.qweight -> qkv.qweight
            # q_proj.qzero, k_proj.qzero, v_proj.qzero -> qkv.qzero
            # q_proj.scales, k_proj.scales, v_proj.scales -> qkv.scales
            cat_dim = 1 if _is_awq_weight(key) else 0
            filtered_state_dict[new_key] = torch.cat([q_proj, k_proj, v_proj], dim=cat_dim)

            del state_dict[key]
            del state_dict[key.replace(".q_proj", ".k_proj")]
            del state_dict[key.replace(".q_proj", ".v_proj")]

        elif key.count(".gate_proj"):
            # gate_proj + up_proj -> gate_up_proj
            gate_proj = state_dict[key]
            up_proj = state_dict[key.replace(".gate_proj", ".up_proj")]

            # .gate_proj -> .gate_up_proj
            new_key = key.replace(".gate_proj", ".gate_up_proj")

            # filtered_state_dict[.gate_up_proj]
            cat_dim = 1 if _is_awq_weight(key) else 0
            filtered_state_dict[new_key] = torch.cat([gate_proj, up_proj], dim=cat_dim)

            del state_dict[key]
            del state_dict[key.replace(".gate_proj", ".up_proj")]

        elif key.count(".k_proj") or key.count(".v_proj") or key.count("up_proj"):
            continue

        else:
            filtered_state_dict[key] = state_dict[key]

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

    # this is state_dict in safetensor
    state_dict = {k: v.to(device) for k, v in state_dict.items()}
    return _merge_state_dict(state_dict)
