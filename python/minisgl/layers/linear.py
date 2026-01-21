from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import divide_even

from .base import BaseOP
from .quantization import LinearMethodBase, UnquantizedLinearMethod


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism and quantization support."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
        params_dtype: torch.dtype = torch.float16,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self._linear_method = linear_method or UnquantizedLinearMethod()
        self._weights: Dict[str, torch.Tensor] = self._linear_method.create_weights(
            local_isize, local_osize, params_dtype
        )
        # Expose weights as attributes for state_dict compatibility
        for name, tensor in self._weights.items():
            setattr(self, name, tensor)
        self.bias = torch.empty(local_osize, dtype=params_dtype) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sync weight dict with potentially reloaded attributes
        for name in list(self._weights.keys()):
            self._weights[name] = getattr(self, name)
        return self._linear_method.apply_weights(self._weights, x, self.bias)


class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
        params_dtype: torch.dtype = torch.float16,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [divide_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(
            input_size,
            output_size,
            input_size,
            tp_output_size,
            has_bias,
            linear_method,
            params_dtype,
        )


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
        params_dtype: torch.dtype = torch.float16,
    ):
        tp_info = get_tp_info()

        GQA_ratio = divide_even(num_qo_heads, num_kv_heads)
        local_num_kv = divide_even(num_kv_heads, tp_info.size)
        full_isize = hidden_size
        full_osize = (GQA_ratio + 2) * num_kv_heads * head_dim
        local_isize = hidden_size
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
        super().__init__(
            full_isize,
            full_osize,
            local_isize,
            local_osize,
            has_bias,
            linear_method,
            params_dtype,
        )


class LinearOProj(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
        params_dtype: torch.dtype = torch.float16,
    ):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = divide_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(
            full_isize,
            full_osize,
            local_isize,
            local_osize,
            has_bias,
            linear_method,
            params_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sync weight dict with potentially reloaded attributes
        for name in list(self._weights.keys()):
            self._weights[name] = getattr(self, name)
        y = self._linear_method.apply_weights(self._weights, x, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
        params_dtype: torch.dtype = torch.float16,
    ):
        tp_info = get_tp_info()
        local_input_size = divide_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(
            input_size,
            output_size,
            local_input_size,
            local_output_size,
            has_bias,
            linear_method,
            params_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sync weight dict with potentially reloaded attributes
        for name in list(self._weights.keys()):
            self._weights[name] = getattr(self, name)
        y = self._linear_method.apply_weights(self._weights, x, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y

