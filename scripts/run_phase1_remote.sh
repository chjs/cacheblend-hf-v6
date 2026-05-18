#!/usr/bin/env bash
# Bootstrap + run Phase 1 tests on a vast.ai instance.
# Expects HF_TOKEN and GITHUB_PAT in env. Use scp to send this with the
# repo, then `bash _vastai_run.sh` on the instance.
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
  # Public repo per README — try without auth first.
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
# Just add transformers / accelerate / pytest.
# The image's system Python is PEP-668-managed; --break-system-packages
# is the right call inside a disposable container.
PIP="python -m pip install --no-cache-dir --break-system-packages"
$PIP -e .
$PIP "transformers>=5.7" accelerate pytest numpy huggingface_hub

python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.version.cuda)"
python -c "import transformers; print('transformers:', transformers.__version__)"

echo "=== hf auth ==="
# Use HF_TOKEN from env so we can pull gated Mistral-7B.
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

echo "=== Phase 1 tests ==="
export LMC_PHASE1_REAL_MODELS=1
# Capture both pytest output and a summary for the report.
python -m pytest tests/test_phase1_layerwise.py -v 2>&1 | tee "$RESULTS/phase1_pytest.log"
PYTEST_RC=${PIPESTATUS[0]}
echo "pytest rc=$PYTEST_RC"

echo "=== nvidia-smi after run ==="
nvidia-smi || true

echo "=== done ==="
echo "RC=$PYTEST_RC"
exit $PYTEST_RC
