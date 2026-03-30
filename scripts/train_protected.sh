#!/bin/bash
############################################################################
# Medical I-JEPA 后台保护训练 (极限压榨版)
# ViT-Large/14 | NIH ChestX-ray14 | 1× RTX PRO 6000 (96GB)
#
# 特性：
#   - nohup + disown 后台保护，终端断开不中断
#   - 自动 checkpoint resume，崩溃自动重启
#   - 连续失败 5 次自动终止，防止无意义重试
#   - SIGTERM/SIGINT 信号捕获，优雅关闭并释放 GPU
#   - GPU 健康监控（每 10 分钟记录温度/功耗/显存）
#   - 训练完成自动通知
#
# 用法：
#   nohup bash scripts/train_protected.sh &          # 首次启动
#   nohup bash scripts/train_protected.sh resume &   # 手动恢复
#   kill $(cat logs/pretrain_vitl14/train.pid)        # 优雅停止
#
# 配置：
#   GPU:    RTX PRO 6000 (96GB)
#   Batch:  640 (单卡极限榨干)
#   Model:  ViT-Large/14 (303M + 11.5M params)
#   VRAM:   ~81 GB / 95 GB (85%)
############################################################################

set -uo pipefail

# ── 基本路径 ─────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${PROJECT_DIR}/configs/pretrain/nih_vitl14.yaml"
LOG_DIR="${PROJECT_DIR}/logs/pretrain_vitl14"
TRAIN_LOG="${LOG_DIR}/train.log"
GPU_LOG="${LOG_DIR}/gpu_monitor.log"
PID_FILE="${LOG_DIR}/train.pid"
FAIL_COUNT_FILE="${LOG_DIR}/.fail_count"
DEVICES="cuda:0"

MAX_FAILURES=5
GPU_MONITOR_INTERVAL=600

mkdir -p "${LOG_DIR}"

# ── 检查是否已有训练在运行 ───────────────────────────────────────────
if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "[ERROR] 训练已在运行 (PID=${OLD_PID})"
        echo "  如需重启，先执行: kill ${OLD_PID}"
        exit 1
    fi
    rm -f "${PID_FILE}"
fi

# ── 环境变量（极致性能）─────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=12
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDNN_V8_API_ENABLED=1

# ── 写入主 PID ───────────────────────────────────────────────────────
echo $$ > "${PID_FILE}"

# ── GPU 健康监控（后台）─────────────────────────────────────────────
gpu_monitor() {
    while true; do
        {
            echo "======== $(date '+%Y-%m-%d %H:%M:%S') ========"
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,fan.speed \
                       --format=csv,noheader -i 0 2>/dev/null
        } >> "${GPU_LOG}"
        sleep "${GPU_MONITOR_INTERVAL}"
    done
}
gpu_monitor &
GPU_MON_PID=$!

# ── 清理函数 ─────────────────────────────────────────────────────────
cleanup() {
    local sig="${1:-EXIT}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 收到 ${sig} 信号，清理中..." | tee -a "${TRAIN_LOG}"
    kill "${GPU_MON_PID}" 2>/dev/null
    if [ -n "${TRAIN_PID:-}" ]; then
        kill -TERM "${TRAIN_PID}" 2>/dev/null
        sleep 3
        kill -9 "${TRAIN_PID}" 2>/dev/null
        pkill -9 -P "${TRAIN_PID}" 2>/dev/null
    fi
    pkill -9 -f "main_pretrain.py" 2>/dev/null
    rm -f "${PID_FILE}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 清理完成" | tee -a "${TRAIN_LOG}"
}
trap 'cleanup SIGTERM' SIGTERM
trap 'cleanup SIGINT'  SIGINT
trap 'cleanup SIGHUP'  SIGHUP
trap 'cleanup EXIT'    EXIT

# ── 设置 checkpoint resume ────────────────────────────────────────────
enable_resume() {
    python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['meta']['load_checkpoint'] = True
with open('${CONFIG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"
}

disable_resume() {
    python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['meta']['load_checkpoint'] = False
with open('${CONFIG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"
}

# ── 初始化失败计数器 ──────────────────────────────────────────────────
if [ "${1:-}" = "resume" ]; then
    echo "0" > "${FAIL_COUNT_FILE}"
    enable_resume
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [RESUME] 手动恢复模式" | tee -a "${TRAIN_LOG}"
else
    echo "0" > "${FAIL_COUNT_FILE}"
    disable_resume
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [START] 全新训练" | tee -a "${TRAIN_LOG}"
fi

# ── 打印配置 ──────────────────────────────────────────────────────────
cat << 'BANNER' | tee -a "${TRAIN_LOG}"
╔══════════════════════════════════════════════════════════════╗
║         Medical I-JEPA Protected Training (极限版)          ║
╠══════════════════════════════════════════════════════════════╣
║  Model:    ViT-Large/14 (303M + 11.5M params, BF16)        ║
║  Dataset:  NIH ChestX-ray14 (~109K images)                 ║
║  GPU:      1× RTX PRO 6000 (96GB) — 极限压榨               ║
║  Batch:    256 (单卡) ≈ VRAM ~60%, 38GB 安全余量             ║
║  LR:       0.001 (与原始 4×4090 配置完全一致)               ║
║  Epochs:   100                                              ║
║  保护:     崩溃自动恢复 | 连续 5 次失败终止                   ║
╚══════════════════════════════════════════════════════════════╝
BANNER

cd "${PROJECT_DIR}"

# ══════════════════════════════════════════════════════════════
#  主训练循环 — 自动重启，最多连续 5 次失败
# ══════════════════════════════════════════════════════════════
while true; do
    FAIL_COUNT=$(cat "${FAIL_COUNT_FILE}" 2>/dev/null || echo 0)

    if [ "${FAIL_COUNT}" -ge "${MAX_FAILURES}" ]; then
        echo "" | tee -a "${TRAIN_LOG}"
        echo "╔══════════════════════════════════════════════════════╗" | tee -a "${TRAIN_LOG}"
        echo "║  [FATAL] 连续失败 ${FAIL_COUNT} 次，已达上限 ${MAX_FAILURES}   ║" | tee -a "${TRAIN_LOG}"
        echo "║  训练终止。请检查日志排查问题后手动重启。            ║" | tee -a "${TRAIN_LOG}"
        echo "║  日志: ${TRAIN_LOG}                                 ║" | tee -a "${TRAIN_LOG}"
        echo "╚══════════════════════════════════════════════════════╝" | tee -a "${TRAIN_LOG}"
        rm -f "${PID_FILE}"
        exit 1
    fi

    if [ "${FAIL_COUNT}" -gt 0 ]; then
        WAIT_SEC=$((FAIL_COUNT * 10))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 第 ${FAIL_COUNT} 次失败后等待 ${WAIT_SEC}s 再重试..." | tee -a "${TRAIN_LOG}"
        sleep "${WAIT_SEC}"
        enable_resume
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 从 checkpoint 自动恢复 (尝试 $((FAIL_COUNT + 1))/${MAX_FAILURES})" | tee -a "${TRAIN_LOG}"
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ── 启动训练进程 ──" | tee -a "${TRAIN_LOG}"

    python main_pretrain.py \
        --fname "${CONFIG}" \
        --devices ${DEVICES} \
        >> "${TRAIN_LOG}" 2>&1 &
    TRAIN_PID=$!

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Python PID: ${TRAIN_PID}" | tee -a "${TRAIN_LOG}"

    wait "${TRAIN_PID}"
    EXIT_CODE=$?

    if [ "${EXIT_CODE}" -eq 0 ]; then
        echo "" | tee -a "${TRAIN_LOG}"
        echo "╔══════════════════════════════════════════════════════╗" | tee -a "${TRAIN_LOG}"
        echo "║  [SUCCESS] 训练正常完成！(exit_code=0)              ║" | tee -a "${TRAIN_LOG}"
        echo "║  $(date '+%Y-%m-%d %H:%M:%S')                      ║" | tee -a "${TRAIN_LOG}"
        echo "╚══════════════════════════════════════════════════════╝" | tee -a "${TRAIN_LOG}"
        echo "0" > "${FAIL_COUNT_FILE}"
        rm -f "${PID_FILE}"
        exit 0
    fi

    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "${FAIL_COUNT}" > "${FAIL_COUNT_FILE}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [CRASH] 训练崩溃 (exit_code=${EXIT_CODE}, 连续失败: ${FAIL_COUNT}/${MAX_FAILURES})" | tee -a "${TRAIN_LOG}"

    pkill -9 -f "main_pretrain.py" 2>/dev/null
    sleep 5

    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 空闲显存: ${FREE_MEM:-unknown} MiB" | tee -a "${TRAIN_LOG}"
done
