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

# Run each (model, dtype) cell in a *separate* pytest process so the
# previous cell's 14-16 GB of model weights is reclaimed by process
# exit. Phase 1 confirmed that in-process fixture teardown can't be
# trusted to fully release VRAM on a 24 GB GPU; the per-process split
# is the reliable pattern. Extra cost is ~10 s of pytest startup per
# cell, vastly cheaper than another vast.ai run.
declare -i FAIL=0

run_one() {
    local k="$1"
    local out="$2"
    echo "--- $k ---"
    python -m pytest tests/test_phase2_equivalence.py -v -k "$k" 2>&1 | tee "$out"
    local rc=${PIPESTATUS[0]}
    echo "rc=$rc for $k"
    [ "$rc" = "0" ] || FAIL=1
}

# Tiny CPU regression — shares process with Mistral-fp16 (cheap).
run_one "TinyModel or Mistral-7B-Instruct-v0.2 and float16" "$RESULTS/phase2_pytest_mistral_fp16.log"
run_one "Mistral-7B-Instruct-v0.2 and bfloat16"             "$RESULTS/phase2_pytest_mistral_bf16.log"
run_one "Meta-Llama-3.1-8B-Instruct and float16"            "$RESULTS/phase2_pytest_llama_fp16.log"
run_one "Meta-Llama-3.1-8B-Instruct and bfloat16"           "$RESULTS/phase2_pytest_llama_bf16.log"

PYTEST_RC=$FAIL
echo "pytest combined rc=$PYTEST_RC"

echo "=== nvidia-smi after run ==="
nvidia-smi || true

echo "=== done ==="
echo "RC=$PYTEST_RC"
exit $PYTEST_RC
