import time
from vllm import LLM, SamplingParams


model_path = "Qwen/Qwen3-4B-AWQ"
num_seqs = 256
max_input_len = 1024 
max_output_len = 1024


llm = LLM(
    model=model_path,
    max_model_len=4096,
    max_num_seqs=num_seqs,
    cuda_graph_max_bs=num_seqs,

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

dummy_prompt_token_ids = [100] * max_input_len
prompts = [{"prompt_token_ids": dummy_prompt_token_ids} for _ in range(num_seqs)]

print(f"Starting throughput test with {num_seqs} sequences...")

start_time = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
end_time = time.perf_counter()

total_time = end_time - start_time
total_generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
tokens_per_sec = total_generated_tokens / total_time

print("-" * 30)
print(f"Total time: {total_time:.2f} seconds")
print(f"Total generated tokens: {total_generated_tokens}")
print(f"Throughput: {tokens_per_sec:.2f} tokens/s")
print("-" * 30)
