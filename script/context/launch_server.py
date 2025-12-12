"""script.context.launch_server

Launcher for the contextual OpenAI-compatible server.

All server implementation lives in `rosetta/context/server.py`.

Example:
  python script/context/launch_server.py --model /share/public/public_models/Qwen3-1.7B --port 30000
"""

from rosetta.context.server import main


if __name__ == "__main__":
    main()
