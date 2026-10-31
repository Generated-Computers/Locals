import sys
from locals.server import Server, serve
srv = Server("mlx-community/Qwen3.5-4B-MLX-4bit", engines=("locals",), preload=("locals",))
serve(srv, "127.0.0.1", int(sys.argv[1]))
