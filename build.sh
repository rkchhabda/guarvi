#!/bin/bash
set -e

export CARGO_NET_OFFLINE=false
export CARGO_HOME=/tmp/cargo
export RUSTFLAGS="-C target-dir=/tmp/target"
export PIP_NO_CACHE_DIR=1

mkdir -p /tmp/cargo /tmp/target

python -m pip install --upgrade pip --no-cache-dir
python -m pip install -r requirements-render.txt --no-cache-dir

# Run any additional build commands if needed
if [ -f render_build.sh ]; then
  bash render_build.sh
fi
