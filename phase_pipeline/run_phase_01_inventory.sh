#!/usr/bin/env bash
set -euo pipefail

ROOT="/Volumes/Samsung_1tb/Perez_Cases_combined/COMBINED_DA_Production"
DB="/Volumes/VAULT/_human_/Jorge_Perez_Discovery/perez_db"

python3 phase_pipeline/phase_01_inventory.py \
  --root "$ROOT" \
  --db "$DB"
