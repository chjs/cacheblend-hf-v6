# cacheblend-hf-v6

A port of **LMCache's CacheBlend** implementation to **HuggingFace
Transformers**, intended to run on a single CUDA GPU with the eager
attention backend.

> **Prime directive.** Follow LMCache. Do not redesign. Do not deviate.
> Do not "improve". Adapt only where an HF API leaves no choice, and
> even then keep the same class names, function names, signatures, and
> field names as LMCache.

## Reference implementation

The source of truth is the `chjs` fork of LMCache, branch
`fix/cacheblend-vllm-v0.17.1-compat`:

- Browse: https://github.com/chjs/LMCache/tree/fix/cacheblend-vllm-v0.17.1-compat
- Clone (on the GPU box):

```bash
git clone --branch fix/cacheblend-vllm-v0.17.1-compat \
  https://github.com/chjs/LMCache.git /workspace/LMCache_reference
```

`docs/LMCACHE_IMPLEMENTATION.md` is a behavior spec extracted from this
branch with file:line citations.

## Target models

- `mistralai/Mistral-7B-Instruct-v0.2` — primary target (standard RoPE,
  no scaling).
- `meta-llama/Meta-Llama-3.1-8B-Instruct` — secondary target. LMCache's
  own `validate_rope_params` would reject this model because of
  Llama-3 RoPE scaling; the HF port sidesteps that by building
  `FusedRope` from HF's own rotary. See README of
  `docs/LMCACHE_IMPLEMENTATION.md` §2.6 for the rationale.

Both share the Llama architecture under HF, so one
`LMCLlamaModel`-equivalent adapter covers them.

## Requirements

- vast.ai (or any single-GPU box) with **≥ 24 GB VRAM** — RTX 4090,
  A5000, A6000, A100, etc.
- CUDA 12.x
- Python ≥ 3.10
- `torch>=2.2`, `transformers>=5.7` (developed against 5.7.0;
  `docs/CODING_CONVENTIONS.md` "Dependencies" notes the back-compat
  branches for older transformers).
- For Phase 4 only: `datasets` (for 2WikiMultihopQA).

**Attention backend:** eager only. No flash-attn, no SDPA, no fused
kernels. Correctness over performance.

## Layout

```
cacheblend-hf-v6/
├── README.md                   ← this file
├── pyproject.toml              ← installable package skeleton
├── .gitignore
├── docs/
│   ├── LMCACHE_IMPLEMENTATION.md   ← behavior spec (source of truth)
│   ├── PROJECT_PLAN.md             ← 4-phase plan
│   ├── CODING_CONVENTIONS.md       ← naming, layout, HF adaptations,
│   │                                 Korean report template
│   ├── VERIFICATION_PROTOCOL.md    ← per-phase pass/fail criteria
│   └── phases/
│       ├── PHASE0_PROMPT.md        ← (this audit; already done)
│       ├── PHASE1_PROMPT.md        ← layerwise forward in HF
│       ├── PHASE2_PROMPT.md        ← prove equivalence to stock forward
│       ├── PHASE3_PROMPT.md        ← port CacheBlend itself
│       └── PHASE4_PROMPT.md        ← RAG quality experiment
├── reports/                    ← per-phase Korean reports written by
│                                 Claude Code (Phase 0 audit already here)
├── lmc/                        ← the ported package (filled in Phase 1+)
├── tests/                      ← pytest suites per phase
├── scripts/                    ← Phase 4 driver
└── results/                    ← Phase 4 result tables
```

The per-file `## Review notes` sections at the bottom of each doc and
phase prompt record the Phase 0 audit corrections, with citations to
LMCache or HF source.

## How to run

Each phase is a fresh Claude Code session driven by its corresponding
prompt in `docs/phases/`.

For each phase in order:

1. Open a fresh Claude Code session in this repo's root.
2. Paste the contents of `docs/phases/PHASE{N}_PROMPT.md` as the
   initial instruction.
3. The phase prompt tells Claude Code to read
   `docs/LMCACHE_IMPLEMENTATION.md`, `docs/CODING_CONVENTIONS.md`, and
   `docs/VERIFICATION_PROTOCOL.md` first.
4. Claude Code implements, runs verification on the GPU box, and writes
   a Korean report to `reports/phaseN.md` following the template in
   `docs/CODING_CONVENTIONS.md` §"Phase reports (Korean)".
5. Review the diff and the report. If verification passes, move to the
   next phase; if it fails, give Claude Code the failure details and
   let it iterate.

Reports are in **Korean**; code, comments, and harness `.md` files stay
in English.

## Status

- Phase 0 (audit): complete — see `reports/phase0_audit.md`.
- Phase 0.5 (repo restructure + GitHub setup): complete — see
  `reports/phase0_5_setup.md`.
- Phases 1–4: pending.
