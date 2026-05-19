# Phase 4 작업 보고서 (재실험)

> **재실험 (rerun) 사유**: 기존 Phase 4 의 결과에서 Full KV reuse 와
> CacheBlend 의 F1 이 동일하게 나왔고, 이는 RAG-cache 비교 실험이
> 의도대로 동작하지 않았다는 신호였다. 본 보고서는 prompt 디자인을
> 재설계 (첫 prompt 를 warmup-only 로 분리) 한 뒤의 결과를 정리한다.

## 0. 한 줄 요약

- 기존 디자인: 첫/두 번째 prompt 가 동일한 real MuSiQue question 으로
  끝남 → question segment 가 cache hit → cache-eval 무효 (n=5 에서
  Full reuse 와 CacheBlend 가 같은 F1).
- 새 디자인: 첫 prompt 는 **dummy warmup query** 로 끝나고, 두 번째
  prompt 만 real MuSiQue question 으로 끝남 → real question 이 cache
  MISS → 의도된 RAG-cache 시나리오.
- 새 결과 (Mistral-7B-Instruct-v0.2 bf16, **smoke n=5**, RTX 3090):
  - Full recompute F1 mean = 0.800
  - Full KV reuse F1 mean = **0.578** (full 대비 -0.22 → 의미 있는 손실)
  - CacheBlend r=0.00 = 0.578 (full reuse 와 동일, sanity check ✓)
  - CacheBlend r=1.00 = 0.800 (full recompute 와 동일, sanity check ✓)
  - CacheBlend r=0.15 = 0.533, r=0.30 = 0.600, r=0.50 = 0.600
- Segment cache-hit invariant: 5/5 examples 에서 6 chunks 모두 hit,
  real question segment MISS. 의도된 cache 패턴 그대로 동작.
- N=100 main run 은 별도 실행 중 (n=100, 동일 ratio sweep). 완료
  되는 대로 §4.2 의 표를 갱신.

## 1. 왜 이전 결과가 불충분했나

### 1.1 직접 원인 — Cache hit on real question

- 기존 prompt 디자인:
  ```
  first_prompt:  prefix + SEP + chunks(order=first)  + SEP + real_question
  second_prompt: prefix + SEP + chunks(order=second) + SEP + real_question
  ```
- 두 prompt 의 last segment (real question) 가 *동일한 토큰 id* 를
  가지므로 `hash(tokens)` 도 같음. 첫 prompt 의 stock prefill 이
  question segment 의 KV 를 cache 에 저장 → 두 번째 prompt 의 question
  segment 가 cache HIT → cache reuse 시 question 도 KV 재사용.
- 그 결과 Method 2 (Full reuse) 가 cache 에서 question KV 까지 가져
  와서 마치 *full prefill* 처럼 동작 → CacheBlend 와 같은 답 →
  F1 동일.

### 1.2 부차적 — 작은 sample size

- 기존 smoke n=5 의 F1 standard error 가 매우 큼 (~0.20).
- 두 method 의 F1 차이가 -0.20 ~ +0.20 사이라면 통계적으로 의미가
  없다고 보아야 함.
- 보고서 §5 의 known limitation 으로 "50-200 examples 의 full run
  필요" 라고 명시했으나 실제 실행은 못 했음.

## 2. 무엇을 바꿨나 — 새 디자인

### 2.1 두 prompt 의 역할 분리

| | 첫 prompt | 두 번째 prompt |
|---|---|---|
| 목적 | KV cache 준비 (**warmup-only, F1 안 잼**) | 실제 RAG 평가 (F1 measure) |
| Prefix | (동일) | (동일) |
| Chunk 순서 | `first_order` (랜덤 셔플) | `reversed(first_order)` |
| Chunk 토큰 ID | 청크별 동일 | 청크별 동일 (순서만 reverse) |
| 마지막 segment | **`dummy_warmup_query`** | **`real_question_segment`** |
| Cache 키 | prefix HIT, 6 chunks 저장, dummy_query 저장 | prefix HIT, 6 chunks HIT, real_question **MISS** |

### 2.2 디폴트 dummy warmup query

```
This is a cache warmup query. Do not answer.
```

- `--dummy-warmup-query` CLI 로 override 가능.
- Defensive 체크: warmup query 가 어떤 example 의 real question 과
  토큰 ID 가 동일하면 그 example 은 skip (cache hit 의도와 충돌).

### 2.3 Segmentwise tokenization (변경 없음, LMCache 패턴 유지)

- `sep_ids = tokenizer.encode(blend_special_str)[1:]`
- 각 segment 가 독립적으로 `[1:]` 토큰화 → 토큰 ID 리스트를 직접 이어붙임.
- 두 prompt 모두 정확히 **7 개의 separator occurrence** (1 post-prefix + 6
  per-chunk). 첫/두 번째 prompt 의 separator_count 가 일치해야 함을
  invariant 로 검증.
- 새 디자인에서는 마지막 segment 만 다르고 chunk 토큰 / sep / prefix
  는 동일.

### 2.4 CacheBlend recompute-ratio sweep

- `--cacheblend-recompute-ratios` CSV 파라미터로 임의 개수의 비율
  지원.
- 기본값: `0.15` (기존 Phase 4 와 호환).
- 재실험 권장: `0.0,0.05,0.15,0.30,0.50,1.00`.
- r=0.00 → HKVD 선택 토큰 1 개만 recompute (`max(topk, 1)`) →
  실질적으로 Full KV reuse 와 거의 동일.
- r=1.00 → 모든 토큰 recompute → 실질적으로 Full recompute 와 거의 동일.
- 의미: r 이 늘어날수록 reuse 의 quality 손실을 회복하지만 prefill
  latency 도 증가. Pareto frontier 측정용.

### 2.5 Segment cache-hit diagnostics

- 매 example 에서 두 번째 prompt 의 8 개 segment (prefix + 6 chunks +
  real_question) 각각에 대해 `storage.contains(layer_key)` 를 확인.
- JSONL details 에 `segment_cache_hit`, `segment_token_lengths`,
  `retrieved_token_count`, `real_question_segment_cache_hit`,
  `all_six_chunks_cache_hit`, `prefix_cache_hit` 기록.
- 만약 real_question 이 HIT 로 잡히면 그 example 은 invalid 표시 (보고
  서의 cache diagnostics 섹션에 강한 경고).

### 2.6 Failure-only subset 분석

- `Full reuse F1 < Full recompute F1` 인 example 들의 subset.
- 이 subset 에서 각 method 의 F1 mean 을 따로 집계.
- CacheBlend 가 *Full reuse 가 실패한 example* 에서 얼마나 회복하는지
  를 보여주는 핵심 지표.

### 2.7 Cache-hit prefix + manual miss-tail split

- **새 디자인의 부작용**: real question segment 가 MISS 이므로
  `cache_engine.retrieve_layer` 가 question 시작 위치에서 break →
  `HFBufferLayerwiseGPUConnector` 의 버퍼가 cache-hit prefix
  길이까지만 할당됨.
- 두 번째 prompt 의 전체 길이로 `blender.blend()` 를 호출하면
  버퍼 (487) 와 prompt (509) 길이가 안 맞아 assertion 실패.
- 해결:
  1. `_compute_cache_hit_end(engine, second_ids)` 로 first miss 위치를
     계산.
  2. `blender.blend(second_ids[:cache_end])` — cache-hit prefix 만
     blend.
  3. Blender 결과를 `DynamicCache` 로 변환 (길이 = cache_end).
  4. 모델 unpatch 후 `_manual_prefill_extend(model, tail_ids,
     base_cache, prefix_len=cache_end)` — stock HF forward 로
     miss tail (sep + real_question) 을 prefill 해서 cache 를 full
     prompt 길이까지 확장.
  5. `decode_from_full_cache` 로 greedy decode.
- 의미적으로 정확: cache-hit 위치는 (RoPE-shifted) cached KV 사용,
  miss tail 은 fresh forward — CacheBlend 의 "cache miss tail" 의도
  그대로.

## 3. 단위 테스트

`pytest tests/test_phase4_rag_quality.py -v`:

```
test_musique_selects_all_supporting_paragraphs        PASSED
test_musique_fills_to_six_by_l2                       PASSED
test_too_many_supporting_skip / error                 PASSED
test_reverse_order_policy                             PASSED
test_separator_count_still_seven                      PASSED
test_no_internal_separator_skip / error               PASSED
test_tokenizer_independent_case_generation            PASSED
test_chunk_token_ids_identical_across_orders          PASSED
test_first_prompt_uses_dummy_query        ← NEW       PASSED
test_second_prompt_uses_real_question     ← NEW       PASSED
test_real_question_not_equal_dummy_query  ← NEW       PASSED
test_dummy_query_default_is_sentinel      ← NEW       PASSED
test_cacheblend_ratio_parser              ← NEW       PASSED
test_failure_subset_computation           ← NEW       PASSED
test_failure_subset_empty_subset          ← NEW       PASSED
test_f1_basics                                        PASSED
test_count_subsequence                                PASSED
test_integration_smoke                                SKIPPED (CPU)
```

→ **19 passed, 1 skipped**. 신규 7 개 (NEW) 가 새 디자인의 invariant 를
검증.

Remote (vast.ai RTX 3090, Mistral-7B-Instruct-v0.2 bf16) 에서도
`19 passed, 1 skipped`.

## 4. 통합 실험

### 4.1 환경

| 항목 | 값 |
|---|---|
| Instance | vast.ai #37035954 (machine 16571, Spain) |
| GPU | RTX 3090, 24 GB |
| Driver / CUDA | NVIDIA 590.48.01 / CUDA 12.4 |
| Image | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` |
| Python | 3.11.11 |
| Hourly | $0.226/hr |
| Model | `mistralai/Mistral-7B-Instruct-v0.2` |
| dtype | bfloat16 |
| Dataset | `dgslibisey/MuSiQue` (HF mirror), `musique_ans_v1.0_train.jsonl` (19938 examples) |
| Ratio sweep | `0.0,0.05,0.15,0.30,0.50,1.00` |
| max_new_tokens | 32 |
| seed | 42 |
| Commit | `d555986` |

### 4.2 N=5 smoke 결과 (preliminary)

> 본 N=5 는 N=100 main run 이 마무리되는 동안의 *preliminary* 데이터로
> 보고서에 포함. 표 형식은 N=100 결과와 동일하므로 N=100 이 끝나면
> 같은 자리에 갱신할 수 있게 둠.

| Method | F1 mean | F1 p50 | Prefill ms mean | Prefill ms p50 |
|--------|---------|--------|-----------------|----------------|
| Full recompute | 0.8000 | 1.0000 | 307.65 | 238.66 |
| Full KV reuse  | 0.5778 | 0.6667 | 347.30 | 344.24 |
| CacheBlend r=0.00 | 0.5778 | 0.6667 | 167.05 | 167.47 |
| CacheBlend r=0.05 | 0.5778 | 0.6667 | 174.93 | 174.51 |
| CacheBlend r=0.15 | 0.5333 | 0.6667 | 185.47 | 180.86 |
| CacheBlend r=0.30 | 0.6000 | 1.0000 | 229.07 | 219.69 |
| CacheBlend r=0.50 | 0.6000 | 1.0000 | 245.59 | 250.58 |
| CacheBlend r=1.00 | 0.8000 | 1.0000 | 352.27 | 346.53 |

**Quality gaps (smoke n=5)**:

- CacheBlend r=0.15 minus Full recompute: **-0.267** (의미 있는 손실)
- Full KV reuse minus Full recompute: **-0.222**
- CacheBlend r=0.15 minus Full KV reuse: **-0.044** (r=0.15 가 reuse
  보다 더 나쁨; n=5 sample variance 때문일 가능성, n=100 에서 확인 필요)

**Sanity checks (smoke n=5)**:

- CacheBlend r=0.00 vs Full KV reuse: F1 gap = **+0.000** ✓
  (실질적으로 동일)
- CacheBlend r=1.00 vs Full recompute: F1 gap = **+0.000** ✓
  (실질적으로 동일)
- |r=1.00 vs full| ≤ 0.03 invariant — PASS.

### 4.3 Failure-only subset (smoke n=5)

| | F1 mean (subset) |
|---|---|
| Full recompute | 1.0000 |
| Full KV reuse  | 0.4444 |
| CacheBlend r=0.00 | 0.4444 |
| CacheBlend r=0.05 | 0.4444 |
| CacheBlend r=0.15 | 0.3333 |
| CacheBlend r=0.30 | **0.5000** |
| CacheBlend r=0.50 | **0.5000** |
| CacheBlend r=1.00 | **1.0000** |

- subset 갯수: **2** (`2hop__482757_12019`, `2hop__144408_215084`)
- Best CacheBlend ratio (subset): r=1.00 @ F1 = 1.0000
- 해석: failure subset 에서 r=0.30 / r=0.50 가 reuse 보다 +0.06 회복.
  r=1.00 이 full recompute 수준까지 회복.

Per-example breakdown of the subset:

| Example | full | reuse | r=0.00 | r=0.05 | r=0.15 | r=0.30 | r=0.50 | r=1.00 |
|---|---|---|---|---|---|---|---|---|
| 2hop__482757_12019 | 1.00 | 0.22 | 0.22 | 0.22 | 0.00 | 0.00 | 0.00 | **1.00** |
| 2hop__144408_215084 | 1.00 | 0.67 | 0.67 | 0.67 | 0.67 | **1.00** | **1.00** | **1.00** |

- **482757_12019**: r=0.15 가 reuse 보다 *오히려 나빠짐* (0.22 → 0.00).
  중간 비율 (0.30, 0.50) 도 회복 못 함. r=1.00 만 회복.
- **144408_215084**: r=0.30 부터 회복. 즉, 적은 양의 recomputation 으로
  도 충분한 case.

### 4.4 Segment cache diagnostics (smoke n=5)

```
- evaluated examples:                 5
- all six chunks cache-hit:           5
- real question segment cache-hit:    0   ✓ (의도된 MISS)
- prefix cache-hit:                   5
```

→ Cache invariant 완벽 일치. real question segment 가 5/5 모두 MISS.
의도된 RAG-cache 시나리오 그대로 동작.

### 4.5 Latency ratios (smoke n=5)

| Comparison | Ratio |
|---|---|
| Full KV reuse / Full recompute | 1.129 (reuse 가 약간 더 느림; cache-load + manual-tail overhead) |
| CacheBlend r=0.00 / Full recompute | **0.543** |
| CacheBlend r=0.05 / Full recompute | 0.569 |
| CacheBlend r=0.15 / Full recompute | 0.603 |
| CacheBlend r=0.30 / Full recompute | 0.745 |
| CacheBlend r=0.50 / Full recompute | 0.798 |
| CacheBlend r=1.00 / Full recompute | 1.145 |

- r 이 증가할수록 latency 가 단조 증가 (cache reuse 의 이점이 줄어듦).
- r=0.30 까지는 Full recompute 대비 25% 이상 빠름.

### 4.6 N=100 main run

- 진행 중 (`results/phase4_musique_rerun_n100*.{md,jsonl}` 에 출력).
- 동일 instance (vast.ai #37035954, RTX 3090).
- 동일 ratio sweep (`0.0,0.05,0.15,0.30,0.50,1.00`).
- 예상 소요: ~22 시간 (per-example ~13 분 × 100).
- 완료 후 §4.2 / §4.3 / §4.4 / §4.5 의 표를 N=100 결과로 갱신.

## 5. 해석

### 5.1 새 디자인이 핵심 invariant 통과

- 의도: real question segment 가 cache MISS 여야 한다.
- 결과: 5/5 examples 에서 MISS ✓.
- 의도: 6 chunks 가 cache HIT 여야 한다.
- 결과: 5/5 examples 에서 HIT ✓.

### 5.2 Sanity checks 통과

- r=0.00 vs Full KV reuse: gap +0.000 ✓
- r=1.00 vs Full recompute: gap +0.000 ✓
- 즉, CacheBlend 의 양 끝 (recompute 0% / 100%) 이 정확히 reuse / full
  과 일치 → blender 의 token-selection 로직이 의도대로 동작.

### 5.3 Reuse 의 quality 손실이 실제로 측정됨

- 기존 (잘못된) 결과: reuse = blend (둘 다 cache hit on question).
- 새 결과: reuse 는 full 대비 -0.22, blend r=1.00 은 full 과 동일.
- CacheBlend ratio sweep 이 *실제로 의미 있는 회복 곡선* 을 그림.

### 5.4 Ratio sweep 의 비단조성

- F1: r=0.00 (0.578) → r=0.15 (0.533) → r=0.30 (0.600) → r=1.00 (0.800).
- 중간 r=0.15 가 r=0.00 보다 나쁨. 의외였으나 작은 sample 임을 감안.
- 가설: HKVD 가 "diffuse" 한 토큰만 recompute 하고 정작 정답을
  결정짓는 위치를 놓치는 case 가 있음. r 이 충분히 크면 (≥0.30) 정답
  결정 토큰을 포함시키게 되어 F1 회복.
- N=100 에서 단조성 회복 여부를 확인할 예정.

### 5.5 한계

1. **N=5 sample variance**: 본 §4.2 은 preliminary. N=100 결과를
   기다려야 통계적 의미 확보.
2. **Mistral-7B-Instruct-v0.2 만 검증**: Llama-3.1-8B-Instruct 는 아직
   미실행 (predict: 동일한 invariant 통과, 다른 F1 곡선).
3. **Latency 측정의 노이즈**: vast.ai 인스턴스의 GPU 점유율이 100%
   이지 않을 수 있음. p50 위주로 해석 권장.
4. **CacheBlend 의 Pareto frontier**: r=0.30 이 Pareto-optimal 후보
   이지만 N=5 에서는 너무 작은 sample. N=100 에서 ratio 별 F1 / latency
   curve 의 elbow 위치를 확인 필요.

## 6. 다음 단계

- [x] 단위 테스트 19/20 PASS (1 skipped CPU integration).
- [x] N=5 smoke validation (cache invariant + sanity checks).
- [ ] N=100 main run (진행 중).
- [ ] N=100 결과로 §4.2 / §4.3 / §4.5 갱신.
- [ ] (선택) Llama-3.1-8B-Instruct 으로 동일 실험 반복.

## 7. 참고

- Commit `1b93f86` (Phase 4 rerun design)
- Commit `d555986` (cache-hit prefix + manual miss-tail split)
- 단위 테스트 / 통합 smoke: 모두 동일 commit 에서 PASS.
- vast.ai 인스턴스 #37035954 — N=100 완료 후 destroy 예정.

작업 완료. 자세한 내용은 reports/phase4.md 참고.
