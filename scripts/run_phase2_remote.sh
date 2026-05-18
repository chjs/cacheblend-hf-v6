#!/usr/bin/env bash
# Bootstrap + run Phase 2 equivalence tests on a vast.ai instance.
# Mirrors scripts/run_phase1_remote.sh — same two-pytest split to keep
# Mistral and Llama-3.1 in separate processes (the parametrized
# fixture's in-process teardown leaves multi-GB of VRAM resident on a
# 24 GB GPU).
#
# Expects HF_TOKEN and (optionally) GITHUB_PAT in env. Use scp to send
# this with the repo, then `bash _vastai_run.sh` (or whatever you named
# it) on the instance.
set -euo pipefail

WORK=/workspace/cacheblend-hf-v6
RESULTS=/workspace/results
mkdir -p "$RESULTS"

echo "=== env ==="
date -u
uname -a
nvidia-smi -L || true
python --version || python3 --version
which pip || which pip3

echo "=== clone repo ==="
if [ ! -d "$WORK/.git" ]; then
  if ! git clone https://github.com/chjs/cacheblend-hf-v6.git "$WORK" 2>&1; then
    echo "(public clone failed; retrying with GITHUB_PAT)"
    git clone "https://x-access-token:${GITHUB_PAT}@github.com/chjs/cacheblend-hf-v6.git" "$WORK"
  fi
else
  ( cd "$WORK" && git fetch && git reset --hard origin/main )
fi
cd "$WORK"
git rev-parse HEAD

echo "=== install ==="
# pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime ships torch already.
# The image's system Python is PEP-668-managed; --break-system-packages
# is the right call inside a disposable container.
PIP="python -m pip install --no-cache-dir --break-system-packages"
$PIP -e .
$PIP "transformers>=5.7" accelerate pytest numpy huggingface_hub

python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.version.cuda)"
python -c "import transformers; print('transformers:', transformers.__version__)"

echo "=== hf auth ==="
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN not set; Mistral-7B-Instruct-v0.2 is gated"
  exit 1
fi
python -c "
import os
from huggingface_hub import login
login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
print('HF login OK')
"

echo "=== Phase 2 tests ==="
export LMC_PHASE2_REAL_MODELS=1

echo "--- pass 1: tiny + Mistral-7B (fp32 / fp16 / bf16) ---"
python -m pytest tests/test_phase2_equivalence.py -v \
    -k "TinyModel or Mistral-7B-Instruct-v0.2" 2>&1 \
    | tee "$RESULTS/phase2_pytest_mistral.log"
RC1=${PIPESTATUS[0]}

echo "--- pass 2: Llama-3.1-8B (fresh process; fp32 / fp16 / bf16) ---"
python -m pytest tests/test_phase2_equivalence.py -v \
    -k "Meta-Llama-3.1-8B-Instruct" 2>&1 \
    | tee "$RESULTS/phase2_pytest_llama.log"
RC2=${PIPESTATUS[0]}

if [ "$RC1" = "0" ] && [ "$RC2" = "0" ]; then
    PYTEST_RC=0
else
    PYTEST_RC=1
fi
echo "pytest rc=$PYTEST_RC (mistral=$RC1 llama=$RC2)"

echo "=== nvidia-smi after run ==="
nvidia-smi || true

echo "=== done ==="
echo "RC=$PYTEST_RC"
exit $PYTEST_RC
