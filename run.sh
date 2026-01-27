#!/bin/bash

# Deploy Qwen/Qwen3-0.6B on a single GPU
# python -m minisgl --model "Qwen/Qwen3-0.6B"

# Deploy Qwen/Qwen3-4B on a single GPU
python -m minisgl --model "Qwen/Qwen3-4B"

# Deploy Qwen/Qwen3-4B with TP
# python -m minisgl --model "Qwen/Qwen3-4B" --tp 2