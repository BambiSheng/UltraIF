python -m vllm.entrypoints.openai.run_batch \
--model meta-llama/Meta-Llama-3.1-70B-Instruct \
-i "input.jsonl" \
-o "output.jsonl"  \
--dtype auto \
--tensor-parallel-size 4 \
--trust-remote-code \
--disable-custom-all-reduce