#!/usr/bin/env bash
# Full foundation run on one GPU box: 10 x 1,000 synthetic faces, then pretraining and matched fine-tuning.
# Start detached:  nohup setsid bash vast/run_foundation.sh > /workspace/run.log 2>&1 &
set -uo pipefail
cd /workspace/syngit && source blender/env.sh
N_CHUNKS=${N_CHUNKS:-10}; CHUNK=${CHUNK:-1000}; FIRST=${FIRST:-100000}; WORKERS=${WORKERS:-4}
for c in $(seq 0 $((N_CHUNKS - 1))); do
  echo "$(date +%T) chunk $c"
  python3 scripts/blender_batch.py --foundation $CHUNK --first-seed $((FIRST + c * CHUNK)) --out /workspace/syn \
      --workers $WORKERS --samples 32 --prune
done
echo "$(date +%T) GEN_DONE $(ls /workspace/syn/dataset/images | wc -l) images"
[ -d /workspace/rock/dataset ] || tar -xzf /workspace/rock/rock_dataset.tar.gz -C /workspace/rock
python3 train/foundation_train.py --syn /workspace/syn --real /workspace/rock --out /workspace/rock/output_foundation
echo "$(date +%T) ALL_DONE"
