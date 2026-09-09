#!/bin/bash
set -e

export CARGO_NET_OFFLINE=false
export CARGO_HOME=/tmp/cargo
export RUSTFLAGS="-C target-dir=/tmp/target"

python -m pip install --upgrade pip
python -m pip install -r requirements-render.txt

# Run any additional build commands if needed
if [ -f render_build.sh ]; then
  bash render_build.sh
fi
