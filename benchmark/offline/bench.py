# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py

import time
import torch
from random import randint, seed

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    # align the hyperparameters
    # num_seqs = 1 306.70tok/s
    # num_seqs = 256 256 5244.43tok/s
    llm = LLM(
        "Qwen/Qwen3-0.6B", max_seq_len_override=4096, max_extend_tokens=16384, cuda_graph_max_bs=256
    )

    # num_seqs = 1   80.87tok/s
    # num_seqs = 256 1963.13tok/s
    # llm = LLM(
    #     "Qwen/Qwen3-4B", max_seq_len_override=4096, max_extend_tokens=16384, cuda_graph_max_bs=256
    # )

    # num_serqs = 1 63tok/s
    # num_seqs = 10 281.31tok/s
    # llm = LLM(
    #     "Qwen/Qwen3-4B-AWQ",
    #     dtype=torch.float16,
    #     max_seq_len_override=4096,
    #     max_extend_tokens=16384,
    #     cuda_graph_max_bs=256,
    # )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len))
        for _ in range(num_seqs)
    ]
    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1))  # to warm up flashinfer
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
