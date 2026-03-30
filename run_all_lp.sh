#!/bin/bash
set -e

cd /root/autodl-tmp/medical-ijepa/medical-ijepa
LOG_FILE="logs/run_all_lp.log"
mkdir -p logs

echo "========================================" | tee -a "$LOG_FILE"
echo "All LP experiments started at $(date)"   | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

CONFIGS=(
    "configs/eval/nih_lp_ep25.yaml|LP-ep25"
    "configs/eval/nih_lp_ep50.yaml|LP-ep50"
    "configs/eval/nih_linear_probe.yaml|LP-ep100-latest"
    "configs/eval/nih_lp_random.yaml|LP-random"
)

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r cfg name <<< "$entry"
    echo "" | tee -a "$LOG_FILE"
    echo ">>> [$name] START at $(date)" | tee -a "$LOG_FILE"
    echo ">>> Config: $cfg" | tee -a "$LOG_FILE"

    python main_eval_classification.py --fname "$cfg" 2>&1 | tee -a "$LOG_FILE"

    echo ">>> [$name] DONE  at $(date)" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "ALL LP experiments FINISHED at $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

echo ""
echo "===== RESULTS SUMMARY ====="
for f in logs/linear_probe_ep25/lp_ep25_log.csv \
         logs/linear_probe_ep50/lp_ep50_log.csv \
         logs/linear_probe/linear_log.csv \
         logs/linear_probe_random/lp_random_log.csv; do
    if [ -f "$f" ]; then
        echo "--- $f ---"
        head -1 "$f"
        sort -t',' -k3 -rn "$f" | head -1
    fi
done
