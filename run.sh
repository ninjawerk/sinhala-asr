#!/usr/bin/env bash
# One command from clone to running demo.
#
# First run: downloads the model weights and language model from the GitHub
# release (~470 MB), installs Python dependencies into .venv/, starts demo.py.
# Later runs: skips everything already present and starts the demo directly.
#
#   ./run.sh              # then open http://localhost:7861
#   ./run.sh --port 8000  # extra arguments are passed through to demo.py
set -euo pipefail
cd "$(dirname "$0")"

RELEASE=https://github.com/ninjawerk/sinhala-asr/releases/latest/download

command -v ffmpeg >/dev/null || {
  echo "ffmpeg is required (macOS: brew install ffmpeg, Debian/Ubuntu: apt install ffmpeg)" >&2
  exit 1; }

# kenlm's PyPI package does not build on Python 3.13+ (it ships C++ generated
# by an old Cython that uses since-removed CPython internals), so pick the
# newest interpreter in the 3.9-3.12 range.
supported() { "$1" -c 'import sys; sys.exit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null; }
PY=
for cand in python3.12 python3.11 python3.10 python3.9 python3; do
  command -v "$cand" >/dev/null && supported "$cand" && { PY=$cand; break; }
done
[ -n "$PY" ] || {
  echo "Python 3.9-3.12 is required (kenlm does not yet build on 3.13+)." >&2
  echo "e.g. macOS: brew install python@3.12, Debian/Ubuntu: apt install python3.12-venv" >&2
  exit 1; }

fetch() {  # fetch <release asset> <destination>
  [ -e "$2" ] && return 0
  echo "downloading $1 ..."
  curl -fL --progress-bar "$RELEASE/$1" -o "$2.part"
  mv "$2.part" "$2"
}

fetch weights.bin weights.bin

mkdir -p lm
if [ ! -e lm/sinhala.arpa ]; then
  fetch sinhala.arpa.gz lm/sinhala.arpa.gz
  echo "decompressing sinhala.arpa.gz ..."
  gunzip lm/sinhala.arpa.gz
fi
if [ ! -e lm/unigrams.txt ]; then
  fetch unigrams.txt.gz lm/unigrams.txt.gz
  gunzip lm/unigrams.txt.gz
fi

# Recreate the venv if it exists but was built with an unsupported Python.
if [ -d .venv ] && ! supported .venv/bin/python; then
  echo "recreating .venv (was built with an unsupported Python version)"
  rm -rf .venv
fi
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install --quiet --requirement requirements.txt

exec .venv/bin/python demo.py "$@"
