#!/usr/bin/env bash
# Bootstrap + run Phase 3 tests on a vast.ai instance.
# Mirrors run_phase2_remote.sh's per-cell pytest split: each (model,
# dtype) cell is a separate Python process so the 24 GB GPU memory is
# fully reclaimed at process exit (Phase 1 / Phase 2 established that
# the in-process fixture teardown can't be trusted to fully release
# VRAM under our pytest+parametrize+model-patch combo).
#
# Expects HF_TOKEN (and optionally GITHUB_PAT) in env.
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

echo "=== Phase 3 tests ==="
export LMC_PHASE3_REAL_MODELS=1
declare -i FAIL=0

run_one() {
    local k="$1"
    local out="$2"
    echo "--- $k ---"
    python -m pytest tests/test_phase3_blender.py -v -k "$k" 2>&1 | tee "$out"
    local rc=${PIPESTATUS[0]}
    echo "rc=$rc for $k"
    [ "$rc" = "0" ] || FAIL=1
}

# Cell 1: all tiny CPU tests (cheap; share process with Mistral fp16).
run_one "TinyModel or (Mistral-7B-Instruct-v0.2 and fp16)" "$RESULTS/phase3_pytest_mistral_fp16.log"
run_one "Mistral-7B-Instruct-v0.2 and bf16"                "$RESULTS/phase3_pytest_mistral_bf16.log"
run_one "Meta-Llama-3.1-8B-Instruct and fp16"              "$RESULTS/phase3_pytest_llama_fp16.log"
run_one "Meta-Llama-3.1-8B-Instruct and bf16"              "$RESULTS/phase3_pytest_llama_bf16.log"

PYTEST_RC=$FAIL
echo "pytest combined rc=$PYTEST_RC"

echo "=== nvidia-smi after run ==="
nvidia-smi || true

echo "=== done ==="
echo "RC=$PYTEST_RC"
exit $PYTEST_RC
