#!/bin/bash
############################################################################
# NIH ChestX-ray14（百度 AI Studio 数据集 35660）后台保护下载
#
# - nohup：SSH 断开继续跑
# - PID 文件：避免重复启动；可 kill $(cat pid) 停止
# - 日志：数据目录下 aistudio_download.log、nohup_download.out
# - 断点续传：由 aistudio_dataset_download.py 实现（HTTP Range + 状态 JSON）
#
# 用法：
#   bash scripts/download_nih_aistudio_protected.sh
#   tail -f /root/autodl-tmp/data/nih-chest-xrays-data/aistudio_download.log
#   kill $(cat /root/autodl-tmp/data/nih-chest-xrays-data/aistudio_download.pid)
############################################################################

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${NIH_DOWNLOAD_DIR:-/root/autodl-tmp/data/nih-chest-xrays-data}"
mkdir -p "${OUT_DIR}"

PID_FILE="${OUT_DIR}/aistudio_download.pid"
PY="${PROJECT_DIR}/scripts/aistudio_dataset_download.py"

if [[ ! -f "${PY}" ]]; then
  echo "[ERROR] 未找到 ${PY}"
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}")"
  if kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "[ERROR] 下载已在运行 (PID=${OLD_PID})，日志: ${OUT_DIR}/aistudio_download.log"
    exit 1
  fi
  rm -f "${PID_FILE}"
fi

export PYTHONUNBUFFERED=1
# 令牌优先读环境变量或 ~/.cache/aistudio/.auth/token
if [[ -z "${AISTUDIO_ACCESS_TOKEN:-}" ]] && [[ -f "${HOME}/.cache/aistudio/.auth/token" ]]; then
  export AISTUDIO_ACCESS_TOKEN="$(tr -d '\n' < "${HOME}/.cache/aistudio/.auth/token")"
fi

nohup python3 "${PY}" --out "${OUT_DIR}" --dataset-id 35660 \
  >> "${OUT_DIR}/nohup_download.out" 2>&1 &

echo $! > "${PID_FILE}"
echo "已启动下载 PID=$(cat "${PID_FILE}")"
echo "  输出目录: ${OUT_DIR}"
echo "  主日志:   ${OUT_DIR}/aistudio_download.log"
echo "  nohup:    ${OUT_DIR}/nohup_download.out"
