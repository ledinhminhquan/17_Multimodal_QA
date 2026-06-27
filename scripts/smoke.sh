#!/usr/bin/env bash
# Offline smoke test: no torch / model / network needed (SceneStubVQA + synthetic scenes).
set -euo pipefail
cd "$(dirname "$0")/.."

export MMQA_ARTIFACTS_DIR="${MMQA_ARTIFACTS_DIR:-/tmp/mmqa_smoke}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="src:${PYTHONPATH:-}"

echo "== compile ==";       python -m compileall -q src/mmqa
echo "== data probe ==";    python -m mmqa.cli data || true
echo "== demo agent ==";    python -m mmqa.cli demo-agent --fast
echo "== ask-scene ==";     python -m mmqa.cli ask-scene --question "how many shapes are there?" --scene 0 --fast
echo "== evaluate ==";      python -m mmqa.cli evaluate --fast
echo "== per-type ==";      python -m mmqa.cli per-type
echo "== grade ==";         python -m mmqa.cli grade | python -c "import json,sys;print(json.load(sys.stdin)['summary'])"
echo "OK"
