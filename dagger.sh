#!/usr/bin/env bash
# 自动DAgger：默认从logs/stage2/cvae_d1/best.pt开始，下一轮为D2。
# 运行：bash dagger.sh；选显卡：CUDA_VISIBLE_DEVICES=0 bash dagger.sh
# 预览：bash dagger.sh --dry_run；中断后原命令重跑，会读取状态恢复。
# 默认直到G2 clean全部通过才结束；可用--max_rounds 5限制本次新增训练轮数。
# 可复用已有完整评估：--initial_summary artifacts/stage2/student_evaluation/某目录/summary.json
# 不指定CUDA版本。合格后停止，扰动测试与decoder导出由你另行执行。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -n "${DAGGER_PYTHON:-}" ]]; then
    dagger_python="$DAGGER_PYTHON"
elif command -v python >/dev/null 2>&1; then
    dagger_python="python"
elif [[ -x .venv/bin/python ]]; then
    dagger_python=".venv/bin/python"
else
    echo "请先激活原teacher训练环境，或设置DAGGER_PYTHON为Python可执行文件路径。" >&2
    exit 1
fi
exec "$dagger_python" -u scripts/stage2/auto_dagger.py "$@"
