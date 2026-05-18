#!/usr/bin/env bash
# Bootstrap + run Phase 4 smoke on a vast.ai instance.
# Downloads MuSiQue if not already cached, installs deps, runs unit
# tests, then the smoke comparison on 5 examples by default.
set -euo pipefail

WORK=/workspace/cacheblend-hf-v6
DATA=/workspace/data
RESULTS=/workspace/results
mkdir -p "$DATA" "$RESULTS"

NUM_EXAMPLES="${NUM_EXAMPLES:-5}"
MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
DTYPE="${DTYPE:-bfloat16}"

echo "=== env ==="
date -u
uname -a
nvidia-smi -L || true
python --version || python3 --version

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
$PIP "transformers>=5.7" accelerate pytest numpy huggingface_hub sentence-transformers

python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.version.cuda)"

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

echo "=== fetch MuSiQue ==="
MUSIQUE_ZIP="$DATA/musique_v1.0.zip"
MUSIQUE_JSONL="$DATA/musique_ans_v1.0_train.jsonl"
if [ ! -f "$MUSIQUE_JSONL" ]; then
  # MuSiQue is published under StonyBrookNLP; the v1.0 train file is
  # served via the published Google Drive archive. We fetch a HF mirror
  # to avoid Google Drive's CAPTCHA flow.
  echo "Downloading MuSiQue dataset..."
  python <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
# Mirror: dgslibisey/MuSiQue hosts the same JSONLs.
src = hf_hub_download(repo_id="dgslibisey/MuSiQue",
                      filename="musique_ans_v1.0_train.jsonl",
                      repo_type="dataset")
dst = os.path.join("/workspace/data", "musique_ans_v1.0_train.jsonl")
shutil.copy(src, dst)
print("MuSiQue at", dst)
PY
fi
ls -la "$MUSIQUE_JSONL"
wc -l "$MUSIQUE_JSONL"

echo "=== unit tests ==="
python -m pytest tests/test_phase4_rag_quality.py -v 2>&1 | tee "$RESULTS/phase4_unit_tests.log"

echo "=== integration smoke (num_examples=$NUM_EXAMPLES) ==="
export MUSIQUE_ANS_TRAIN_JSONL="$MUSIQUE_JSONL"
python scripts/run_rag_comparison.py \
  --model "$MODEL" \
  --input-jsonl "$MUSIQUE_JSONL" \
  --num-examples "$NUM_EXAMPLES" \
  --dtype "$DTYPE" \
  --output "$RESULTS/phase4_musique_smoke.md" \
  --write-jsonl-details "$RESULTS/phase4_musique_smoke_details.jsonl" 2>&1 | tee "$RESULTS/phase4_integration.log"

echo "=== smoke markdown ==="
cat "$RESULTS/phase4_musique_smoke.md"

echo "=== done ==="
echo "RC=$?"
