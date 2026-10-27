#!/usr/bin/env bash
set -e
cd /mnt/e/Project/NyxFelineLabs/craftax-world-model
exec /root/miniconda3/envs/craftax-wm/bin/python -u scripts/twm_baseline_diagnostic.py --max-env-steps 1000000 --debug-timing
