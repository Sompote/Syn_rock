#!/usr/bin/env bash
# Install Blender (default 4.5 LTS) and the Python packages on a Linux GPU box, then run the smoke test.
# Usage (from the repository root): bash blender/setup_linux.sh [series]
set -euo pipefail
SERIES="${1:-4.5}"
DEST="${BLENDER_HOME:-/opt}"
cd "$(dirname "$0")/.."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || echo "nvidia-smi not found"
# xz unpacks the release; headless Blender still links a few X/GL client libraries
if command -v apt-get >/dev/null; then
  apt-get update -qq && apt-get install -y -qq xz-utils curl libxrender1 libxi6 libxkbcommon0 libsm6 libgl1 libegl1 \
    libxxf86vm1 libxfixes3 >/dev/null || true
fi
if ! ls -d "$DEST"/blender-"$SERIES".*-linux-x64 >/dev/null 2>&1; then
  BASE="https://download.blender.org/release/Blender${SERIES}/"
  FILE=$(curl -fsSL "$BASE" | grep -o "blender-${SERIES}\.[0-9]*-linux-x64\.tar\.xz" | sort -uV | tail -1)
  [ -n "$FILE" ] || { echo "no Blender ${SERIES} Linux build at $BASE"; exit 1; }
  echo "downloading $FILE"
  [ -s "/tmp/$FILE" ] || curl -fL --retry 3 "$BASE$FILE" -o "/tmp/$FILE"
  tar -xJf "/tmp/$FILE" -C "$DEST" && rm "/tmp/$FILE"
fi
BL=$(ls -d "$DEST"/blender-"$SERIES".*-linux-x64 | sort -V | tail -1)/blender
python3 -m pip install -q --timeout 60 -r requirements.txt
echo "export BLENDER=$BL" > blender/env.sh
echo "Blender: $BL  (source blender/env.sh)"
"$BL" -b --factory-startup -P blender/smoke_test.py
