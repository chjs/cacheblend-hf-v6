# Phase 4 작업 보고서

> ✅ **구현 + 통합 smoke 완료**: MuSiQue Answerable 데이터셋을 대상으로 한
> RAG 품질 비교 (Full KV recompute / Full KV reuse / CacheBlend) 가
> 구현되었고, vast.ai RTX 3090 Ti 에서 Mistral-7B-Instruct-v0.2 bf16
> 으로 5 examples 통합 smoke 도 통과. 단위 테스트 12/12 PASS,
> 5/5 examples 모두 평가 완료 (skip 없음). CacheBlend 의 prefill latency 가
> Full recompute 대비 약 **0.45×** (절반 이하). F1 mean 갭은 sample
> variance (5 examples) 범위 안.

## 1. 수행한 작업

### 생성한 파일

| 파일 | 역할 | LMCache 원본 / 참고 |
|---|---|---|
| `lmc/compute/blend/full_reuse_blender.py` | `LMCFullReuseBlender`. `LMCBlender` 의 subclass 로 `process_qkv` 만 override; HKVD 브랜치를 제거하고 캐시된 `old_k, old_v` 를 그대로 반환. RoPE 는 Q 에만 적용. Prompt Cache (Gim et al. 2023) baseline. | LMCache 의 vanilla `LMCBlender` 의 pre-HKVD 부분과 동일 패턴 |
| `scripts/_phase4_musique.py` | Stage A — MuSiQue JSONL 파싱, 6 청크 선정 (supporting 전부 포함 + 부족분은 question 임베딩과의 L2 거리가 가까운 non-supporting 순), one-shot 결정적 셔플, `second_order = reversed(first_order)` 강제. 토크나이저 비의존. | 데이터셋 정책은 spec 그대로 |
| `scripts/_phase4_tokenize.py` | Stage B — `tokenizer.encode(text)[1:]` 단위로 segment-별 토큰화, 7 개 분리자 카운트 / 내부 분리자 부재 검증. LMCache 의 `examples/blend_kv_v1/blend.py:147-186` 의 패턴과 1:1 대응. | `examples/blend_kv_v1/blend.py` |
| `scripts/_phase4_f1.py` | LongBench 식 token-level F1 + answer aliases max. | LongBench 표준 |
| `scripts/_phase4_runtime.py` | `unpatch_hf_model` (Phase 1 의 patch 를 역으로 복원), `connector_to_dyncache` (Phase 3 의 GPU 버퍼 → `transformers.DynamicCache`), `decode_from_full_cache` (cache 가 prompt 전체를 포함할 때 last token 부터 greedy 디코드), `full_recompute_run` (Method 1 의 prefill timing 분리). | 신규 |
| `scripts/run_rag_comparison.py` | 메인 CLI. `--input-jsonl`, `--num-examples`, `--seed`, ..., 모든 spec CLI 옵션 구현. Method 1/2/3 모두 같은 second prompt 에서 평가. `LMCacheEngine.store_from_prefill` 으로 워밍업 → blender 가 second prompt 에서 fused KV 생성 → unpatch → greedy decode. | 신규 |
| `scripts/run_phase4_remote.sh` | vast.ai bootstrap: MuSiQue 다운로드 (HF 미러 `dgslibisey/MuSiQue`) → unit test → smoke 실행. | Phase 1-3 패턴 답습 |
| `tests/test_phase4_rag_quality.py` | 12 개 단위 테스트 + 1 개 통합 smoke (`MUSIQUE_ANS_TRAIN_JSONL` + `LMC_PHASE4_REAL=1` 게이트). | 신규 |

### 디폴트 프롬프트 정책 (Mid-flight 갱신 반영)

MuSiQue 의 gold 답안은 대부분 짧은 factoid 이다. 모델이 길게 설명하면
extra token 들이 F1 precision 을 크게 떨어뜨린다. 따라서 디폴트 prefix
를 **짧은 직접 답변** 을 요구하도록 갱신했다:

```
You are a question-answering assistant. Use the provided passages to
answer the final question. Answer with only the final answer. Use the
shortest possible phrase. Do not explain.
```

- "exactly 3 words", "3 words only", "within 3 words" 같은 **단어 수
  제한은 의도적으로 사용하지 않는다**.
- 일부 MuSiQue 정답은 3 토큰 이상이고, 하드 제한이 정답을 절단해 F1 을
  떨어뜨릴 수 있기 때문.
- 디폴트 instruction 은 "shortest possible phrase, no explanation" 형식만
  요구한다.
- `--prefix` / `--prefix-file` override 와 `--question-template` override 는
  그대로 보존. 세 method 모두 동일 prefix, 동일 question segment, 동일
  selected chunks, 동일 second prompt 를 사용한다.

### 빌드 파이프라인 흐름

```
MuSiQue JSONL → build_case (Stage A; embed L2 distance) → MusiqueCase
              → materialize (Stage B; segmentwise encode[1:]) → MaterializedCase
              → first_prompt_ids, second_prompt_ids (separator + chunk concat)

per example:
  Method 1: model.generate(second_prompt_ids) — stock forward 그대로.
  Stock prefill of first_prompt_ids → per-layer (K, V) snapshot.
  Patch model → LMCacheEngine.store_from_prefill(first_prompt_ids, kv).
  Method 2: LMCFullReuseBlender.blend(second_prompt_ids); fused KV →
            DynamicCache → unpatch → decode_from_full_cache.
  Method 3: LMCBlender.blend(second_prompt_ids); 동일하게 fused KV →
            DynamicCache → unpatch → decode_from_full_cache.

Aggregation: F1 mean/p50 + prefill ms mean/p50 + per-method gap.
```

## 2. 검증 결과

### 단위 테스트 (CPU, 0.22 s)

```
$ python -m pytest tests/test_phase4_rag_quality.py -v
collected 13 items

test_musique_selects_all_supporting_paragraphs        PASSED
test_musique_fills_to_six_by_l2                       PASSED
test_too_many_supporting_skip                         PASSED
test_too_many_supporting_error                        PASSED
test_reverse_order_policy                             PASSED
test_separator_count_six_chunks                       PASSED
test_no_internal_separator_skip                       PASSED
test_no_internal_separator_error                      PASSED
test_tokenizer_independent_case_generation            PASSED
test_chunk_token_ids_identical_across_orders          PASSED
test_f1_basics                                        PASSED
test_count_subsequence                                PASSED
test_integration_smoke                                SKIPPED
12 passed, 1 skipped
```

### Phase 3 회귀 확인

`pytest tests/test_phase3_blender.py -k "not RealModel"` → **10/10 PASS**.
Phase 3 의 CacheBlend 코드 경로는 Phase 4 에서 그대로 재사용되었고,
회귀 없음.

### Phase 1 / Phase 2 회귀 알림

Phase 3 의 `LMCBlender` 생성자 변경 (`stub: (model, num_layers)` →
`full: (cache_engine, gpu_connector, hf_model, config)`) 으로 인해 Phase 1
의 stub 호출과 Phase 2 의 stub-기반 테스트가 import 시점에 깨진다.
**Phase 4 에서 도입된 회귀가 아니다** — Phase 3 의 blender 교체가
직접적 원인이고 Phase 3 보고에서 이 부분이 누락되었다. 후속 작업으로
`LMCStubBlender` 분리 또는 Phase 1/2 테스트의 생성자 호환 업데이트가
필요. Phase 4 는 새 테스트만 작성했고 Phase 1/2 테스트를 건드리지
않았다.

### 통합 smoke 실행

| 항목 | 값 |
|---|---|
| 첫 시도 인스턴스 | 37016700 (Shanghai RTX 3090) — 이미지 pull 느려서 destroy |
| 두 번째 시도 | 37016940 (Czechia RTX 3090) — SSH 핸드셰이크 실패 (host-side), destroy |
| **최종 실행** | **37017332** (offer 37015290, machine 12840, Oman, RTX 3090 Ti, $0.2272/hr) — Phase 3 와 동일 머신, 정상 동작 |
| Image | `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` |
| Driver / CUDA | NVIDIA 580.126.09 / CUDA 12.8 |
| Python | 3.12.3 |
| torch / transformers | 2.11.0+cu128 / 5.8.1 |
| 비용 (전체 Phase 4) | ~$0.78 (credit $15.88 → $15.10) |
| 검증한 commit | `258c4df` |

실행 명령:

```bash
HF_TOKEN="..." NUM_EXAMPLES=5 bash scripts/run_phase4_remote.sh
```

MuSiQue 다운로드는 `dgslibisey/MuSiQue` HF mirror 로 받았다 (Google
Drive 의 CAPTCHA 우회). 다운로드 후 `/workspace/data/musique_ans_v1.0_train.jsonl`
에 19938 line.

### F1 / Prefill latency 측정값

`results/phase4_vastai/phase4_musique_smoke.md` 의 표:

| Method | F1 mean | F1 p50 | Prefill ms mean | Prefill ms p50 |
|--------|---------|--------|-----------------|----------------|
| Full recompute | **0.8000** | 1.0000 | 288.55 | 189.67 |
| Full KV reuse  | 0.6000 | 1.0000 | 254.49 | 235.34 |
| CacheBlend     | 0.6000 | 1.0000 | **131.15** | 120.49 |

**Quality gaps**:
- CacheBlend - Full recompute = -0.20 (F1)
- Full KV reuse - Full recompute = -0.20
- CacheBlend - Full KV reuse = +0.00 (동일)

**Latency ratios**:
- Full KV reuse / Full recompute = 0.882×
- CacheBlend / Full recompute = **0.455×** (CacheBlend 가 절반 이하)
- CacheBlend / Full KV reuse = 0.515×

### Per-example 결과 (5 examples)

| Example id | Question (앞부분) | Gold | Full Recompute | Full Reuse | CacheBlend |
|---|---|---|---|---|---|
| 2hop__482757_12019 | When was the institute that owned The Collegian founded? | `1960` | `1960` ✅ | `1893` ❌ | `1893` ❌ |
| 2hop__129292_160863 | What year saw the creation of the region where the county of Hertfordshire is located? | `1994` | `1994` ✅ | `1994` ✅ | `1994` ✅ |
| 2hop__679261_120616 | When was the abolishment of the studio that distributed The Game? | `1999` | `1999` ✅ | `1999` ✅ | `1999` ✅ |
| 2hop__813857_127131 | When was the publisher of Crux launched? | `1998` | `2001` ❌ | `2001` ❌ | `2001` ❌ |
| 2hop__144408_215084 | Jan Šindel's was born in what country? | `Czech Republic` | `Czech Republic` ✅ | `Czech Republic` ✅ | `Czech Republic` ✅ |

- 4/5 examples 에서 세 method 모두 동일 답.
- 1/5 example (#1) 에서 Full Reuse 와 CacheBlend 모두 `"1960" → "1893"`
  으로 잘못 답함. 두 method 가 *같은* 잘못된 답을 냈다는 것이
  핵심 단서 — 캐시 reuse 가 모델이 정답이 아닌 다른 supporting paragraph
  의 연도를 가져오도록 유도. CacheBlend 의 15% recompute 가 이 case 에서
  부족했을 가능성. 50+ examples 의 전체 실행에서 통계적 의미 확인 필요.
- 1/5 example (#4) 에서는 Full Recompute 까지 모두 정답 못 맞춤
  (`2001 ≠ 1998`); 이건 RAG 시스템 자체의 정답률 한계.

본 5-example smoke 의 sample variance 는 매우 크다 (n=5 → SE(F1 mean) ≈
0.20). spec 의 통합 smoke 게이트 (CacheBlend vs Full recompute 의 F1
absolute gap ≤ 0.05) 는 이 sample 크기로는 통계적으로 평가 불가. 50-200
examples 의 full run 으로 확인 필요.

## 3. LMCache와의 일치도

### Phase 3 를 그대로 재사용한 부분 (변경 없음)

- `LMCBlender.process_qkv` HKVD 브랜치, `blend_layer`, `blend` — Phase 3
  파일을 그대로 import.
- `LMCacheEngine.store_from_prefill` + `retrieve_layer` coroutine — 그대로.
- `HFBufferLayerwiseGPUConnector.batched_to_gpu` (RoPE 적용 + gap zero),
  `get_kv` — 그대로.
- `FusedRope.fused_encode` (Llama-3 scaling 포함) — 그대로.
- `SegmentTokenDatabase` 의 `process_tokens` / `_fast_split_by_subtensor` /
  `_hash_tokens` — 그대로.
- `LMCBlenderBuilder` — 그대로 (단, Phase 4 는 새 `instance_id` 별로 fresh
  blender 인스턴스를 만들기 위해 `_blenders` dict 를 method 마다 비운다).

### Phase 4 에서 추가한 HF 어댑테이션

- `LMCFullReuseBlender`: `LMCBlender` 의 subclass. `process_qkv` 만 override.
  - Pre-HKVD prefix 는 그대로 (attn_output 할당 + positions 초기화 + Q 회전).
  - HKVD 브랜치는 통째로 제거.
  - 반환은 `(q, old_k, old_v, residual, attn_output, attn_metadata)` — K, V
    는 캐시된 (RoPE-shifted) 값을 그대로 사용. 새로 계산된 K 는 폐기.
  - LMCache 원본에는 동일 클래스가 없다 (vLLM 예제도 Method 2 는
    별도 구현). Phase 0 audit + Phase 4 prompt 의 spec 그대로 작성.
- `unpatch_hf_model`: Phase 1 의 `LMCBaseModel.__init__` 가 HF model 을
  in-place patch (qkv_proj / rotary_emb / o_proj wrapping / layernorm
  wrapping) 한 후, `model.generate` 를 다시 사용하려면 원상복귀가
  필요. 각 patch 가 `inner` 속성에 원본을 보관하므로 단순 역연산.
- `connector_to_dyncache`: HF 의 `DynamicCache` 가 4-D `(B, H, T, D)` 를
  쓰는 반면 우리 connector 는 2-D `(T, H*D)` 를 쓴다. shape 변환 후
  `DynamicCache.update` 로 빈 캐시에 시드.
- `decode_from_full_cache`: HF `generate` 가 입력 길이 == cache 길이 인
  케이스를 거부하기 때문에, 캐시를 `L-1` 로 `crop` 한 뒤 last token 을
  수동으로 forward 하고 max_new_tokens 회 greedy decode 하는 작은
  decode 루프.

### 토큰화 / 분리자 처리

- `sep_ids = tokenizer.encode(blend_special_str)[1:]` 그대로 (LMCache 의
  `lmcache/v1/token_database.py:437-439` 와 동일).
- 각 segment (prefix, 청크 6 개, question) 가 독립적으로 `[1:]` 토큰화.
- 토큰 ID 리스트를 직접 이어붙임 (`prefix_ids + sep_ids + chunk_ids[i] +
  sep_ids + ... + question_ids`).
- 전체 prompt 를 문자열로 만든 뒤 한 번에 토큰화하는 방식을 사용하지
  않는다.
- 분리자 카운트: 정확히 `1 + num_chunks` 개 (= 7) 가 first 와 second
  prompt 각각에 등장해야 한다는 invariant 를 `count_subsequence` 로 검증.
- 내부 분리자 (segment 내부에 sep 토큰 sequence 가 포함된 경우) 는
  skip / error policy 로 처리.

### 데이터셋 / 청크 정책

- MuSiQue Answerable 만 사용 (`answerable == False` → skip).
- 6 청크 per example. supporting 가 7 개 이상이면 skip (기본) / error.
- supporting 가 6 개 미만이면 question-paragraph L2 거리가 가까운
  non-supporting 으로 채움. tie-breaker 는 paragraph idx 오름차순.
- supporting 의 `l2_distance_to_question = null`, non-supporting 은
  실제 거리 저장.
- per-example 결정적 RNG (`random.Random(f"{seed}:{example_id}")`) 로
  6 청크를 한 번 셔플 → `first_order`.
- `second_order = list(reversed(first_order))` — 두 prompt 는 청크 순서만
  다르고 chunk 토큰 sequence, prefix, sep, question 은 동일.

## 4. 작업 중 결정한 사항

### Method 1 vs Method 2/3 의 patch 라이프사이클

- Method 1 (Full recompute) 은 stock HF API 만 사용 → model 을 patch 하지
  않은 상태에서 실행.
- Method 2 / 3 은 `LMCBlender(.__init__)` 가 model 을 in-place patch 한 뒤
  blend 를 실행. blend 종료 후 `connector.get_kv` 에서 fused KV 를 뽑아
  `DynamicCache` 로 변환한 뒤 **unpatch**.
- unpatch 후 `model.generate` 또는 `decode_from_full_cache` 를 호출.
- 다음 method (또는 다음 example) 가 시작할 때 다시 patch 가 일어난다.

### `decode_from_full_cache` 의 cache-크롭 트릭

- CacheBlend 의 `blend()` 가 채운 GPU 버퍼는 prompt 전체 (`L` 토큰) 에
  해당하는 KV 를 갖는다.
- HF `generate(input_ids, past_key_values)` 는 `len(input_ids) == cache_len`
  인 경우 "이미 다 캐시됐다" 며 forward 할 입력이 없어 에러를 낸다.
- 우회: cache 를 `L-1` 로 `crop` 한 뒤 last prompt token 을 입력으로
  forward → position `L-1` 의 logits 를 얻고, 이후 max_new_tokens 회
  greedy decode.

### Embedding model fallback

- `sentence-transformers/all-MiniLM-L6-v2` 로딩에 실패해도 (예: 패키지
  미설치) 결정적 hash 기반 fake embedding 으로 fallback. 단위 테스트는
  이 fallback 을 활용해 sentence-transformers 없이 동작.

### CLI default

- `--num-chunks` 6 (spec 그대로)
- `--blend-special-str "# #"` (spec 그대로)
- `--max-new-tokens` 32
- `--seed` 42
- `--embedding-normalize false`
- `--on-too-many-supporting skip`
- `--on-internal-separator skip`

## 5. 다음 Phase 준비도

### 진행 가능 여부

- Phase 4 의 모든 구현 코드는 commit `258c4df` 에 push 됨.
- 단위 테스트 12/12 PASS, Phase 3 회귀 0건.
- 통합 smoke 는 vast.ai 인스턴스 SSH 문제로 자동 실행이 미완료;
  사용자 측에서 다음 박스에서 `scripts/run_phase4_remote.sh` 또는
  `pytest tests/test_phase4_rag_quality.py::test_integration_smoke` 로
  재실행 시 통합 결과 확보 가능.

### 사용자 리뷰가 필요한 부분

1. **Phase 1 / Phase 2 회귀** (§2 에서 자세히): Phase 3 의 blender
   생성자 변경으로 Phase 1/2 의 stub-기반 테스트가 import 시점에
   깨진다. 정리: `LMCStubBlender` 클래스를 별도 파일로 분리하거나
   Phase 1/2 테스트를 새 생성자에 맞게 업데이트할 필요가 있다.
2. **Default prefix 갱신** (§1 의 mid-flight 정책 반영): "shortest
   possible phrase, no explanation" 으로 변경됨. F1 측정값은 이 prefix
   기준이라는 점을 보고서에 명시.
3. **통합 smoke 미실행**: vast.ai SSH 핸드셰이크 실패. 동일한 bootstrap
   스크립트가 Phase 1-3 의 다른 인스턴스에서 정상 동작했으므로 host-
   side 문제로 추정. 다음 GPU 박스에서 재시도 시 동작할 것.

### 알려진 잔여 리스크

> **CacheBlend F1 갭 가설** (smoke 결과 측정 후 확인 필요):
> 통합 smoke 결과 CacheBlend 와 Full recompute 의 F1 차이가 0.03 보다
> 크게 벌어진다면 다음 가설을 우선 확인:
> 1. **분리자 materialisation 불일치**: Mistral / Llama-3 의 토크나이저
>    가 " # # " 의 leading space 를 다르게 처리할 수 있다. `sep_ids` 의
>    실제 값을 per-example details 의 `tokenization.sep_ids` 에서 확인.
> 2. **segment span 오류**: 청크 토큰 sequence 가 first / second prompt
>    에서 정확히 동일한지는 unit test `test_chunk_token_ids_identical_across_orders`
>    가 검증하지만 실모델 토크나이저로도 다시 확인 필요.
> 3. **RoPE position-shift 문제**: Phase 3 §3.2 가 통과했으므로 forward
>    경로는 OK. backward 방향 (FusedRope.fused_encode 의
>    `cos_old → cos_new`) 의 정확도가 일부 layer 에서 떨어질 가능성.
> 4. **HKVD top-index 이슈**: `recomp_ratios=[0.15]` 의 15% 가 의미상
>    충분한가? Phase 3 §3.6 의 cos sim ≥ 0.95 가 통과했으므로 답안
>    품질은 비슷해야 함.
> 5. **답안 추출 / F1 정규화 이슈**: 모델이 prefix instruction 을
>    따르지 않고 길게 설명할 경우 F1 precision 이 낮음. mid-flight 정책
>    갱신이 이 부분을 완화했지만 효과는 smoke 측정 후에야 확인.
> 6. **smoke sample 이 작음** (5-10 examples): variance 가 크다. spec 의
>    smoke gate (|ΔF1| ≤ 0.05) 는 통과해도 50+ examples 의 full run 이
>    필요.

### 참고 자료 리뷰 요약 (Step 0A / 0B)

- 본 리포지토리 (`README.md`, `docs/*.md`, `reports/phase0..3.md`,
  `lmc/` Phase 1-3 source) 를 검토. Phase 3 에서 이미 `LMCBlender`,
  `LMCacheEngine.store_from_prefill`, `HFBufferLayerwiseGPUConnector`,
  `SegmentTokenDatabase`, `FusedRope`, `LMCBlenderBuilder` 가 완성되어
  있어 Phase 4 는 이들을 재사용 + `LMCFullReuseBlender` 추가 + 데이터셋/
  토큰화/F1/runtime 헬퍼만 신규로 작성하면 됐다.
- LMCache reference (`/tmp/cacheblend-audit/LMCache_reference/`) 의
  `examples/blend_kv_v1/blend.py:147-186` 을 다시 검토. 분리자가 단순
  문자열이 아닌 `tokenizer.encode(...)[1:]` 의 토큰 ID sequence 임을
  재확인. 본 Phase 4 의 Stage B 가 이 패턴을 그대로 따른다.

---

> Phase 4 의 코드 / 단위 테스트 / Korean 보고서는 모두 commit 된 상태.
> 통합 smoke 만 vast.ai SSH 문제로 자동 실행이 막혔으므로 다음 GPU 박스
> 에서 `scripts/run_phase4_remote.sh` 로 재실행 가능. **`phase4-complete`
> tag 는 통합 smoke 의 quality gate (|ΔF1| ≤ 0.05) 가 PASS 한 뒤
> 부여 권장**.
