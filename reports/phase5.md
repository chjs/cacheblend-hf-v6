# Phase 5 작업 보고서

> ✅ **완료**: Phase 3 의 `LMCBlender` 생성자 변경 (`stub: (model, num_layers)`
> → `full: (cache_engine, gpu_connector, hf_model, config)`) 으로 Phase 1 /
> Phase 2 tests 가 import-time `TypeError` 로 깨졌던 것을 Strategy A
> (`LMCStubBlender` 별도 클래스) 로 복구. Local 에서 4 phases 합산
> **30/30 tests PASS**. vast.ai 의 Phase 1 / Phase 2 real-model 회귀 통과
> 도 함께 확인. 또한 사용자의 in-flight 강조 사항인 "첫 번째 prompt =
> KV 워밍업 전용 / 두 번째 prompt = F1 평가 전용" 구분을
> `scripts/run_rag_comparison.py` 의 주석에서 명시화.

## 1. 수행한 작업

### 진단

`pytest tests/test_phase1_layerwise.py` / `pytest tests/test_phase2_equivalence.py`
실행 시 동일한 import-time / setup-time TypeError 발생:

```
E   TypeError: LMCBlender.__init__() got an unexpected keyword argument 'num_layers'
```

원인: Phase 3 가 `lmc/compute/blend/blender.py` 의 `LMCBlender` 를 full
production class 로 교체 (생성자 `(cache_engine, gpu_connector, hf_model,
config)`) 하면서, Phase 1 의 stub 호출 `LMCBlender(model, num_layers=N)`
와 Phase 2 의 동일 호출이 더 이상 유효하지 않게 됨. Phase 3 보고서가
Phase 1/2 회귀를 detect 하지 못한 채 commit 됐다.

### Fix — Strategy A 선택

- **Strategy A (채택)**: 새 클래스 `LMCStubBlender` 를 별도 파일에 두고
  Phase 1/2 tests 는 import 만 바꾼다.
- Strategy B (대안): Phase 1/2 tests 를 full LMCBlender 의 4-인자
  생성자에 맞춰 더미 cache_engine / gpu_connector 를 만들도록 개조.
  - 거절 이유: Phase 1 의 목적이 "compute_layer skeleton 자체" 의 검증
    인데, 풀 의존성을 mock 으로 채우면 검증 범위가 흐려진다. Strategy A
    의 코드 surface 가 약간 더 늘어나지만 분리가 깨끗하다.

### 변경된 / 생성된 파일

| 파일 | 변경 | 설명 |
|---|---|---|
| `lmc/compute/blend/stub_blender.py` | **신규** | `LMCStubBlender` 클래스. Phase 1 stub 그대로 (`hf_model`, `num_layers` 만 받음, `process_qkv` 가 layer 의 `rotary_emb` 로 Q, K 만 회전하고 pass-through). |
| `lmc/compute/blend/__init__.py` | 수정 | `LMCBlender` 와 `LMCStubBlender` 둘 다 re-export. |
| `tests/test_phase1_layerwise.py` | 수정 | `from lmc.compute.blend.blender import LMCBlender` → `from lmc.compute.blend.stub_blender import LMCStubBlender as LMCBlender`. 이후의 사용 코드는 byte-identical. |
| `tests/test_phase2_equivalence.py` | 수정 | 동일한 import 교체. |
| `docs/CODING_CONVENTIONS.md` | 수정 | 파일 layout 섹션 뒤에 "`LMCStubBlender` (test-time only)" 단락 추가. 향후 reader 가 두 클래스의 존재 이유를 즉시 알 수 있게. |
| `scripts/run_rag_comparison.py` | 수정 | (in-flight 사용자 강조 반영) `_per_example` / `_run_cache_method` 의 주석을 보강하여 **first prompt = 워밍업 전용 / second prompt = 평가 전용** 의 역할 분리를 명시. 동작은 그대로. |

총 변경량: 156 insertions, 9 deletions (commit `ca9b4dc`).

## 2. 검증 결과

### Before / After (local CPU)

수정 **이전**:

```
$ pytest tests/test_phase1_layerwise.py -k "TinyModel" -v
...
>       stub = LMCBlender(model, num_layers=len(model.model.layers))
E       TypeError: LMCBlender.__init__() got an unexpected keyword argument 'num_layers'
========== 1 failed, 1 passed, 10 deselected, 4 errors in 4.30s ==========

$ pytest tests/test_phase2_equivalence.py -k "TinyModel" -v
...
>       stub = LMCBlender(model, num_layers=cfg.num_hidden_layers)
E       TypeError: LMCBlender.__init__() got an unexpected keyword argument 'num_layers'
========== 8 deselected, 2 errors in 3.68s ==========
```

수정 **이후**:

```
$ pytest tests/ -q -k "TinyModel or (not RealModel and not integration)"
..............................                                           [100%]
30 passed, 27 deselected in 17.75s
```

전체 4 phases tiny-model 합계 **30/30 PASS**:
- Phase 1: 6/6
- Phase 2: 2/2
- Phase 3: 10/10
- Phase 4: 12/12 (1 통합 smoke 는 게이트로 skip)

### vast.ai real-model 회귀

| 항목 | 값 |
|---|---|
| Instance | 37018941 (offer 31183607, machine 46599, Czechia, RTX A5000 24 GB) |
| Image | `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` |
| 가격 | $0.2898 /hr |
| 검증 commit | `ca9b4dc` |
| 비용 | ~$0.63 (credit $15.10 → $14.47) |

검증한 cells (Phase 1 + Phase 2, 모두 PASS):

| # | Cell | Pytest 결과 | Wall time |
|---|---|---|---|
| 1 | Phase 1: tiny + Mistral-7B-Instruct-v0.2 | **11 passed** | 68.72 s |
| 2 | Phase 1: Meta-Llama-3.1-8B-Instruct (별도 process) | **5 passed** | 74.26 s |
| 3 | Phase 2: tiny + Mistral-7B fp16 | **4 passed** | 56.61 s |
| 4 | Phase 2: Mistral-7B bf16 | **2 passed** | 50.53 s |
| 5 | Phase 2: Llama-3.1-8B fp16 | **2 passed** | 59.87 s |
| 6 | Phase 2: Llama-3.1-8B bf16 | **2 passed** | 54.21 s |

합계 6 cells × 모두 PASS = **26 real-model tests PASS**. 전체 wall time
364.20 s ≈ 6 분 4 초. 원시 로그는 `results/phase5_vastai/phase5_p{1,2}_*.log`
와 `phase5_full_run.log` 에 보관.

### Phase 3 / Phase 4 회귀 확인

본 fix 가 production 코드 (`lmc/compute/blend/blender.py`) 를 건드리지
않았기 때문에 Phase 3 / Phase 4 는 변경 없음:

- `pytest tests/test_phase3_blender.py -k "not RealModel" -q` → **10/10 PASS**
- `pytest tests/test_phase4_rag_quality.py -k "not integration" -q` → **12/12 PASS**

### 사용자의 mid-flight 강조 — 두 prompt 의 역할 분리

사용자가 강조한 점:
- **첫 번째 prompt** = KV 캐시 준비 전용. stock HF prefill 으로 per-chunk
  KV 를 `LMCacheEngine.store_from_prefill` 에 저장. **F1 측정에 사용
  하지 않음.**
- **두 번째 prompt** = F1 평가 전용. 세 method 모두 이 prompt 에 대해
  generation 수행, gold answer 와 비교해 F1 계산.

원래 `scripts/run_rag_comparison.py` 의 동작은 이미 이 구분을 따르고
있었지만 (인자 이름 `first_ids` / `second_ids` 가 그렇게 명명), 주석이
관련 의도를 충분히 명시하지 않았다. Phase 5 commit 에서:

- `_per_example` 의 docstring 에 "Role split (made explicit by user
  request)" 단락 추가.
- Method 1 / 2 / 3 에 `(evaluation)`, warmup 부분에는 `(no F1 measured)`
  라벨을 주석에 명시.
- `_run_cache_method` 의 `engine.store_from_prefill(...)` 직전 주석
  강화 ("first prompt is never generated from").

이 변경은 동작에 영향이 없고 코드 가독성만 향상.

## 3. LMCache와의 일치도

### LMCache 원본과 동일하게 유지

- 본 Phase 는 production 코드 (`LMCBlender`, `LMCacheEngine`,
  `SegmentTokenDatabase`, `FusedRope`, `HFBufferLayerwiseGPUConnector`)
  를 전혀 건드리지 않았다. Phase 3 의 spec-부합 구현이 그대로 보존된다.
- `LMCStubBlender` 는 LMCache 의 `LMCBlender.process_qkv:59-86` (pre-HKVD
  prefix) 를 그대로 따른다. 단 HKVD 브랜치와 cache write-back 은 빠짐.
  Stub 의 존재는 spec 의 일부가 아니라 HF 테스트 인프라의 편의용.

### HF 어댑테이션

- 없음. Stub 은 HF 어댑테이션이 아니라 테스트 인프라의 의존성 해소.

## 4. 작업 중 결정한 사항

1. **Strategy A vs B**: A 선택 (별도 클래스). Phase 1 의 의도 — "compute_
   layer skeleton 만 검증" — 를 보존하기 위함. B 였다면 cache_engine /
   gpu_connector 의 더미를 만들고 Phase 1 tests 가 그 더미 동작에도
   부분적으로 의존하게 되어 검증 범위가 흐려진다.
2. **`LMCBlender as` alias**: Phase 1/2 test 내부의 클래스 사용 사이트를
   최소 변경. `from ... import LMCStubBlender as LMCBlender` 로 import
   레벨에서만 swap. 이후의 `LMCBlender(model, num_layers=N)` 호출은
   byte-identical.
3. **`__init__.py` re-export**: 두 클래스 모두 `lmc.compute.blend.*` 로
   접근 가능하게 했지만, 실제 import 사이트는 명시 경로 (`from
   lmc.compute.blend.stub_blender import LMCStubBlender`) 를 쓴다.
   re-export 는 future-proofing 용.
4. **사용자 강조 사항 반영 시점**: 본 task 도중 user 가 "두 prompt 의
   역할 분리" 를 강조. 같은 commit 에 묶어 처리 — 별도 phase 로 분리할
   가치가 없다. 동작 변경 없음, 주석만 보강.

## 5. 다음 Phase 준비도

### 진행 가능 여부

- ✅ Local 30/30 PASS (4 phases tiny 합산).
- ✅ vast.ai real-model 회귀 26/26 PASS (Phase 1 + Phase 2, Mistral + Llama-3.1, fp16 + bf16; tiny 포함).
- ✅ Phase 3 / Phase 4 의 production 코드 변경 없음.
- 본 commit (`ca9b4dc`) 에 `phase5-complete` tag 부여.
- Phase 6 (full RAG run) 은 clean test suite 위에서 진행 가능.

### 사용자 리뷰가 필요한 부분

1. `LMCStubBlender` 의 존재 자체. LMCache reference 에는 없는 HF 포트
   고유의 테스트 인프라. `docs/CODING_CONVENTIONS.md` 에 짧은 단락
   추가 완료.
2. test import alias (`LMCStubBlender as LMCBlender`) 가 readability
   를 약간 떨어뜨릴 수 있다. 명시적으로 `from ... import LMCStubBlender`
   로 바꾸고 사용 사이트를 모두 갱신하는 것이 cleaner. 추후 정리 가능.

### 알려진 잔여 리스크

- **Phase 1 / Phase 2 의 test 가 stub 의 행동에만 의존**: production
  `LMCBlender` 가 Phase 2 의 등가성 검증에는 직접 검증되지 않는다 (Phase 2
  spec 이 이미 stub 만 검증). production `LMCBlender` 의 정확성은
  Phase 3 §3.1-§3.6 에서 검증되어 있으므로 회귀는 아니다.
- **여전히 Phase 4 통합 smoke 의 F1 갭 (CacheBlend vs Full recompute,
  ~0.20)** 는 sample variance 인지 systemic 오류인지 5-example 로는
  결판 불가. Phase 6 의 full run 에서 50-200 examples 로 확인 필요.
  본 Phase 와는 무관.

---

> Phase 5 의 산출물 (`lmc/compute/blend/stub_blender.py`,
> `lmc/compute/blend/__init__.py`, `tests/test_phase{1,2}_*.py`,
> `docs/CODING_CONVENTIONS.md`, `scripts/run_rag_comparison.py`,
> `results/phase5_vastai/*`) 은 모두 commit 된 상태이며 vast.ai real-
> model 회귀 26/26 PASS. **`phase5-complete` tag 부여 완료**. Phase 6
> 진행 가능.
