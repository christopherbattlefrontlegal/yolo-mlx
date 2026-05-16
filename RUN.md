python3 scripts/run_pipe_case_parallel.py \

  --case-root "/Volumes/Samsung_1tb/Perez_Cases_combined/COMBINED_DA_Production/25AGJ201B" \

  --out-root "$OUT" \

  --hash-workers 32 \

  --workers 8 \

  --batch 16 \

  --sample-fps 15 \

  2>&1 | tee "$LOG"