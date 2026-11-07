#!/bin/bash
# Issue #18, step 4: one request through the server, MTP drafts off and on. Run from this folder:
#   bash step4.sh <label> [LOCALS_MTP_DRAFTS]     (no drafts value: MTP off)
# It starts its own server from this checkout on PORT, records its PID, checks that the server reports this
# checkout's build, runs the decode sweep at 1 request and `locals bench --agents --n 1`, saves the server's
# /stats, and stops only that PID. One GPU job at a time: nothing else may run on the GPU meanwhile.
set -u
LABEL=$1
DRAFTS=${2:-}
PORT=${PORT:-8399}
MODEL=${MODEL:-Kaziko/Qwen3.5-4B-Zwyrol}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
PY="$REPO/.venv/bin/python"
OUT="$HERE/results/step4"
mkdir -p "$OUT"

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port $PORT is taken: not ours to use" >&2
  exit 1
fi
ENVS=(LOCALS_DISK_CACHE=0)
if [ -n "$DRAFTS" ]; then
  ENVS+=(LOCALS_MTP="$HERE/mtp_bf16.safetensors" LOCALS_MTP_DRAFTS="$DRAFTS")
fi
cd "$REPO" || exit 1
env -u LOCALS_MTP -u LOCALS_MTP_DRAFTS "${ENVS[@]}" "$PY" -m locals.server --model "$MODEL" --port "$PORT" \
  --engines locals --preload locals > "$OUT/${LABEL}_server.log" 2>&1 &
PID=$!
echo "server pid $PID ($LABEL, ${ENVS[*]})"
stop() { kill "$PID" 2>/dev/null && wait "$PID" 2>/dev/null; }
trap stop EXIT
for _ in $(seq 1 600); do
  curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$PID" 2>/dev/null || { echo "the server stopped:" >&2; tail -20 "$OUT/${LABEL}_server.log" >&2; exit 1; }
  sleep 1
done
LISTEN=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1)
[ "$LISTEN" = "$PID" ] || { echo "port $PORT is held by $LISTEN, not by our server $PID" >&2; exit 1; }
SERVED=$(curl -s "http://127.0.0.1:$PORT/health" | "$PY" -c "import json,sys; print(json.load(sys.stdin)['build'])")
CHECKOUT=$("$PY" -c "from locals.defaults import build_id; print(build_id())")
[ "$SERVED" = "$CHECKOUT" ] || { echo "the server runs build $SERVED, the checkout is $CHECKOUT" >&2; exit 1; }
{
  echo "label $LABEL; env ${ENVS[*]}; model $MODEL; build $SERVED; commit $(git rev-parse --short HEAD)$(git diff --quiet -- locals || echo '+changes')"
  "$PY" -c "import mlx.core as mx, mlx_lm, platform; d = mx.device_info(); print('mlx', mx.__version__, '| mlx-lm', mlx_lm.__version__, '|', d.get('device_name'), d.get('memory_size', 0) // 2**30, 'GB')"
  date
} > "$OUT/${LABEL}_setup.txt"
grep -i "mtp" "$OUT/${LABEL}_server.log" >> "$OUT/${LABEL}_setup.txt"
"$PY" "$HERE/decode_sweep.py" qwen3.5-4b-zwyrol locals "$PORT" 1 > "$OUT/${LABEL}_sweep.txt" 2>&1
"$REPO/.venv/bin/locals" bench --agents --n 1 --engines locals --port "$PORT" --model "$MODEL" > "$OUT/${LABEL}_agents.txt" 2>&1
curl -s "http://127.0.0.1:$PORT/stats" > "$OUT/${LABEL}_stats.json"
cat "$OUT/${LABEL}_sweep.txt" "$OUT/${LABEL}_agents.txt"
