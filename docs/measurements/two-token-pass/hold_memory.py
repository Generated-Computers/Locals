"""A neighbour that holds GPU memory and runs no GPU work: GB gigabytes of MLX arrays, made and touched once, with
the wired limit Runtime.load sets (as an idle locals server holds its weights and prompt cache). It exits after
SECONDS, or when stopped.
Usage: hold_memory.py GB SECONDS"""
import sys
import time

import mlx.core as mx

gb, secs = float(sys.argv[1]), float(sys.argv[2])
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
chunks = [mx.ones((1 << 28,), mx.float16) for _ in range(int(gb * 2))]  # 512 MB each
mx.eval(chunks)
print(f"holding {sum(c.nbytes for c in chunks) / 2**30:.1f} GB (active {mx.get_active_memory() / 2**30:.1f} GB)", flush=True)
time.sleep(secs)
