#!/bin/sh
# locals installer: an Apple-silicon Mac, uv (installed if missing), then locals.
#
#   curl -LsSf https://raw.githubusercontent.com/Generated-Computers/Locals/main/install.sh | sh
#   LOCALS_SOURCE=locals sh install.sh     # or a wheel path, instead of the GitHub repository
#
# Afterwards:  locals code ~/my-project   (sets up OpenCode, downloads the model, starts
# the server and opens the agent; the first run takes a minute or two; --model gemma for Gemma 4).
set -eu
ORIG_PATH="$PATH"

say() { printf '%s\n' "$*"; }
fail() { printf 'locals install: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || fail "locals runs on macOS with Apple silicon (MLX); this is $(uname -s)."
[ "$(uname -m)" = "arm64" ] || fail "locals needs an Apple-silicon Mac (M1 or later); this machine is $(uname -m)."
MACOS="$(sw_vers -productVersion 2>/dev/null || echo 0)"
[ "${MACOS%%.*}" -ge 14 ] 2>/dev/null || fail "locals needs macOS 14 (Sonoma) or newer (MLX ships for those); this Mac runs macOS $MACOS."

if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    PATH="$HOME/.local/bin:$PATH"
  else
    say "installing uv (the Python package manager locals is installed with) ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || fail "could not install uv; see https://docs.astral.sh/uv/"
    PATH="$HOME/.local/bin:$PATH"
  fi
fi
command -v uv >/dev/null 2>&1 || fail "uv is not on PATH after installing it; open a new terminal and run this again."

SOURCE="${LOCALS_SOURCE:-git+https://github.com/Generated-Computers/Locals}"  # PyPI ("locals") after the first release
say "installing locals from $SOURCE ..."
uv tool install --force "$SOURCE" || fail "uv tool install $SOURCE failed (locals needs Python 3.11 or newer; uv fetches one)."

BIN="$(uv tool dir --bin 2>/dev/null || printf '%s' "$HOME/.local/bin")"
case ":$ORIG_PATH:" in
  *":$BIN:"*) ;;
  *)
    uv tool update-shell >/dev/null 2>&1 || true
    say ""
    say "locals is installed in $BIN; uv added it to your shell's PATH (in your shell's startup file),"
    say "which this terminal does not search yet:"
    say "open a new terminal, or run:  export PATH=\"$BIN:\$PATH\""
    ;;
esac
say ""
say "$("$BIN/locals" --version 2>/dev/null || echo locals) is ready. Next:"
say "  locals code ~/my-project                 # OpenCode on your project, with Qwen3.5-4B on this Mac"
say "  locals code ~/my-project --model gemma   # the same with Gemma 4 E4B"
say "  locals setup openclaw                    # or OpenClaw (locals setup hermes: Hermes Agent)"
say "  locals serve                             # or only the OpenAI-compatible server, http://127.0.0.1:8321/v1"
