import time
from vllm import LLM, SamplingParams
"""
vLLM Throughput Benchmark with Warmup
Includes warmup phase to stabilize GPU performance before measurement
"""

"""
# Qwen/Qwen3-4B-AWQ (Marlin Kernel)
- tp = 1
Total time: 109.76 seconds
Total generated tokens: 262144
Throughput: 2338.43tok/s
- tp = 2
Total time: 95.84 seconds
Total generated tokens: 262144
Throughput: 2735.30tok/s

# Qwen/Qwen3-4B
"""
model_path = "Qwen/Qwen3-4B-AWQ"
# model_path = "Qwen/Qwen3-4B"

# Test configuration
num_seqs = 256
max_input_len = 1024 
max_output_len = 1024

# Warmup configuration
warmup_seqs = 32  # Fewer sequences for warmup
warmup_tokens = 128  # Shorter outputs for warmup

print("Initializing model...")
llm = LLM(
    model=model_path,
    max_model_len=4096,
    max_num_seqs=num_seqs,
    enable_chunked_prefill=True,
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
)

sampling_params = SamplingParams(
    max_tokens=max_output_len,
    temperature=0.6,
    top_p=0.95,
    presence_penalty=1.5,
)

# Warmup phase
print(f"\nRunning warmup with {warmup_seqs} sequences...")
warmup_sampling_params = SamplingParams(
    max_tokens=warmup_tokens,
    temperature=0.6,
    top_p=0.95,
    presence_penalty=1.5,
)
warmup_prompt_token_ids = [100] * max_input_len
warmup_prompts = [{"prompt_token_ids": warmup_prompt_token_ids} for _ in range(warmup_seqs)]

warmup_start = time.perf_counter()
_ = llm.generate(warmup_prompts, warmup_sampling_params)
warmup_end = time.perf_counter()
print(f"Warmup completed in {warmup_end - warmup_start:.2f} seconds")

# Main benchmark
dummy_prompt_token_ids = [100] * max_input_len
prompts = [{"prompt_token_ids": dummy_prompt_token_ids} for _ in range(num_seqs)]

print(f"\nStarting throughput test with {num_seqs} sequences...")
start_time = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
end_time = time.perf_counter()

# Calculate metrics
total_time = end_time - start_time
total_generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
tokens_per_sec = total_generated_tokens / total_time

# Results
print("=" * 50)
print("BENCHMARK RESULTS")
print("=" * 50)
print(f"Total time: {total_time:.2f} seconds")
print(f"Total generated tokens: {total_generated_tokens}")
print(f"Throughput: {tokens_per_sec:.2f} tokens/s")
print(f"Average tokens per sequence: {total_generated_tokens / num_seqs:.2f}")
print("=" * 50)