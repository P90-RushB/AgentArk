import argparse

import uvicorn

from agent_ark.ark_env.serving.env_server import app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
