#!/bin/sh
# locals installer: an Apple-silicon Mac, uv (installed if missing), then locals.
#
#   curl -LsSf https://raw.githubusercontent.com/Generated-Computers/Locals/main/install.sh | sh
#   LOCALS_SOURCE=dist/locals-0.2.0-py3-none-any.whl sh install.sh   # a wheel or a folder instead of GitHub's archive
#
# Afterwards:  locals code ~/my-project   (sets up OpenCode, downloads the model, starts
# the server and opens the agent; the first run takes a minute or two; --model gemma for Gemma 4).
set -eu
ORIG_PATH="$PATH"

say() { printf '%s\n' "$*"; }
fail() { printf 'locals install: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || fail "locals runs on macOS with Apple silicon (MLX); this is $(uname -s)."
if [ "$(uname -m)" != "arm64" ]; then
  # a Terminal running under Rosetta reports x86_64 on an M-series Mac: say so instead of "not Apple silicon"
  [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ] && fail "this terminal runs under Rosetta (as an Intel Mac); open a Terminal that runs natively (in Finder: Get Info on Terminal, untick \"Open using Rosetta\") and run this again."
  fail "locals needs an Apple-silicon Mac (M1 or later); this machine is $(uname -m)."
fi
MACOS="$(sw_vers -productVersion 2>/dev/null || echo 0)"
[ "${MACOS%%.*}" -ge 14 ] 2>/dev/null || fail "locals needs macOS 14 (Sonoma) or newer (MLX ships for those); this Mac runs macOS $MACOS."

if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    PATH="$HOME/.local/bin:$PATH"
  else
    say "installing uv (the Python package manager locals is installed with) ..."
    UV_SH="$(mktemp)"
    # downloaded first: piped into sh, a failed download looked like a finished install ("uv is not on PATH")
    curl -LsSf https://astral.sh/uv/install.sh -o "$UV_SH" || fail "could not download uv's installer; check the network"
    sh "$UV_SH" || fail "could not install uv; see https://docs.astral.sh/uv/"
    rm -f "$UV_SH"
    PATH="$HOME/.local/bin:$PATH"
  fi
fi
command -v uv >/dev/null 2>&1 || fail "uv is not on PATH after installing it; open a new terminal and run this again."

# GitHub's archive of the main branch: no git needed (a Mac without the Command Line Tools has none; installing
# from git+https then failed)
SOURCE="${LOCALS_SOURCE:-locals @ https://github.com/Generated-Computers/Locals/archive/refs/heads/main.tar.gz}"
say "installing locals from $SOURCE ..."
BIN="$(uv tool dir --bin 2>/dev/null || printf '%s' "$HOME/.local/bin")"
# with its folder on PATH, uv does not warn that it is not; the PATH is explained once, below. Python 3.12, the one
# locals is tested on (uv took the oldest the package allows, 3.11, for a new user)
PATH="$BIN:$PATH" uv tool install --python 3.12 --force "$SOURCE" \
  || fail "uv tool install $SOURCE failed; its message is above (check the network; with an old uv, run  uv self update  first)."

case ":$ORIG_PATH:" in
  *":$BIN:"*) ;;
  *)
    say ""
    if uv tool update-shell >/dev/null 2>&1; then  # (it said so even when this failed)
      say "locals is installed in $BIN; uv added it to your shell's PATH (in your shell's startup file),"
      say "which this terminal does not search yet:"
    else
      say "locals is installed in $BIN, which is not on your shell's PATH:"
    fi
    say "open a new terminal, or run:  export PATH=\"$BIN:\$PATH\""
    ;;
esac
say ""
say "$("$BIN/locals" --version 2>/dev/null || echo locals) is ready. Next:"
say "  locals code ~/my-project      # a coding assistant (OpenCode) on your project"
say "  locals run \"a task\" --dir ~/my-project --check \"python3 -m pytest -q\"   # a team of agents"
say "  locals launch list            # the other coding tools and apps it works with"
say "  locals start                  # only the server, for your own programs: http://127.0.0.1:8321/v1"
