# Phase 0 — Review and improve the harness

## What this is

The working directory contains a harness of `.md` files that directs
Claude Code to port LMCache's **CacheBlend** implementation to
HuggingFace Transformers across four phases. The harness was written by
another agent based on a quick read of the paper and source. Before any
implementation work begins, the job is to **audit and fix the harness
itself**.

Do not write any implementation code in this phase. Edit only the `.md`
files.

## Working style

- You decide. Do not ask the user clarification questions during this
  review. If something is ambiguous, pick the option that stays closest
  to LMCache's actual behavior and document the decision in the edited
  file.
- Be thorough. The harness will drive 4 phases of work; an error here
  multiplies downstream.
- Be specific. "This section is wrong" is not enough — say what is
  wrong, cite the LMCache file and line that proves it, and rewrite.

## Step 1 — Read everything in the harness

Read every `.md` file currently in the harness in this order:

1. `README.md`
2. `docs/LMCACHE_IMPLEMENTATION.md`
3. `docs/PROJECT_PLAN.md`
4. `docs/CODING_CONVENTIONS.md`
5. `docs/VERIFICATION_PROTOCOL.md`
6. `docs/phases/PHASE1_PROMPT.md`
7. `docs/phases/PHASE2_PROMPT.md`
8. `docs/phases/PHASE3_PROMPT.md`
9. `docs/phases/PHASE4_PROMPT.md`

For each, take notes on:
- Claims about LMCache's behavior that you should verify against the
  source.
- Tolerances, thresholds, or numbers you should sanity-check.
- HF-adaptation guidance — is it actually correct for the current HF
  Transformers API?
- Anything missing, vague, or self-contradictory.

## Step 2 — Read the paper

CacheBlend paper (EuroSys '25):

- arXiv abstract: https://arxiv.org/abs/2405.16444
- arXiv HTML (full text): https://arxiv.org/html/2405.16444v3
- arXiv PDF: https://arxiv.org/pdf/2405.16444

Read sections 1–6 (skim 7+ unless you need details). You are reading the
paper **only to understand context and terminology** — the harness's
prime directive is to follow the LMCache source, not the paper. Where
the paper and the source disagree, the source wins.

## Step 3 — Read the LMCache source

Clone the exact branch the harness targets:

```bash
git clone --branch fix/cacheblend-vllm-v0.17.1-compat \
    https://github.com/chjs/LMCache.git /workspace/LMCache_reference
```

The reference URL is:
https://github.com/chjs/LMCache/tree/fix/cacheblend-vllm-v0.17.1-compat

Read every file listed in `docs/LMCACHE_IMPLEMENTATION.md` §"LMCache
reference checkout". At minimum, fully read:

- `lmcache/v1/compute/blend/blender.py`
- `lmcache/v1/compute/blend/metadata.py`
- `lmcache/v1/compute/blend/utils.py`
- `lmcache/v1/compute/models/base.py`
- `lmcache/v1/compute/models/llama.py`
- `lmcache/v1/compute/models/utils.py`
- `lmcache/v1/compute/attention/abstract.py`
- `lmcache/v1/compute/attention/flash_attn.py`
- `lmcache/v1/compute/attention/metadata.py`
- `lmcache/v1/compute/positional_encoding.py`
- `lmcache/v1/token_database.py`
- `lmcache/v1/cache_engine.py` (the `retrieve_layer` function)
- `lmcache/v1/gpu_connector/gpu_connectors.py` (the
  `VLLMBufferLayerwiseGPUConnector` class)
- `lmcache/v1/config.py` (the blending-related keys)
- `examples/blend_kv_v1/blend.py` (usage)
- `examples/blend_kv_v1/README.md`

While reading, verify each concrete claim the harness makes against
the source.

## Step 4 — Verify HF API specifics

The harness's "HF-required adaptations" section in
`docs/CODING_CONVENTIONS.md` makes assumptions about the HF Transformers
API. Verify these against the **currently installed** `transformers`
version, by reading the actual source of:

- `transformers.models.llama.modeling_llama.LlamaForCausalLM`
- `transformers.models.llama.modeling_llama.LlamaModel`
- `transformers.models.llama.modeling_llama.LlamaDecoderLayer`
- `transformers.models.llama.modeling_llama.LlamaAttention`
- `transformers.models.llama.modeling_llama.LlamaRMSNorm`
- `transformers.models.llama.modeling_llama.LlamaRotaryEmbedding`
- `transformers.models.llama.modeling_llama.apply_rotary_pos_emb`
- Plus the same for `transformers.models.mistral.modeling_mistral.*`.

If any harness instruction in `docs/CODING_CONVENTIONS.md` or any phase
prompt is wrong for the current HF version, fix it.

## Step 5 — Cross-check the four phase prompts

Walk through each phase prompt and check:

- Phase 1: Does the stub blender it describes interact correctly with
  the call sites in `compute_layer`?
- Phase 2: Are the tolerances reasonable for the HF eager backend? Is
  the K, V capture method workable with the modern HF `Cache` object?
- Phase 3: Are all components actually listed? Anything in LMCache that
  Phase 3 depends on but the harness doesn't mention to port?
- Phase 4: Is the "Full KV reuse" baseline definition accurate?

## Step 6 — Apply fixes

Edit the `.md` files in place. For each change:
- Make the edit.
- Add a short rationale at the bottom of the file under a new section
  `## Review notes` (or append to it if it exists), citing the LMCache
  file/line or HF file/line that justifies the change.

## Step 7 — Final report

The Korean audit report lives at `reports/phase0_audit.md`. It follows
the standard template in `docs/CODING_CONVENTIONS.md` §"Phase reports
(Korean)".

> 참고: Phase 0 작업 결과는 `reports/phase0_audit.md` 에 기록되어 있다.
