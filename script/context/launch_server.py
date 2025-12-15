"""script.context.launch_server

Launcher for the contextual OpenAI-compatible server.

All server implementation lives in `rosetta/context/server.py`.

Example:
  CUDA_VISIBLE_DEVICES=0,1 python script/context/launch_server.py --model Qwen/Qwen3-1.7B --port 30000
  CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server --model-path Qwen/Qwen3-1.7B --host 0.0.0.0 --tp-size 2 --tool-call-parser qwen --port 30000
"""

from rosetta.context.server import main


if __name__ == "__main__":
    main()
