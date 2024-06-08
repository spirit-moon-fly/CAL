python -m vllm.entrypoints.openai.api_server --served-model-name Qwen1.5-72B-Chat --model ../../../share/models/Qwen1.5-72B-Chat --tensor-parallel-size 4 --max-model-len 8000 --dtype float16

