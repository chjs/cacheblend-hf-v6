# Phase 1 작업 보고서

> ⚠ **선행 안내**: 본 실행 환경은 macOS / arm64 / CUDA 없음
> (`torch.cuda.is_available() == False`). Phase 1 verification 의
> 두 target 모델 (Mistral-7B, Llama-3.1-8B) 은 ≥ 24 GB VRAM GPU 가
> 필수이므로 본 머신에서 실행 불가능했다. 코드 / 테스트는 모두 작성
> 완료되었고, CPU 에서 동작 가능한 tiny-model 통합 테스트로 코드
> 경로를 검증했다. **실제 target 모델에 대한 verification 은 vast.ai
> GPU 박스에서 `LMC_PHASE1_REAL_MODELS=1 pytest …` 로 추가 실행**
> 필요. 자세한 내용은 §5 "잔여 리스크" 참고.

## 1. 수행한 작업

### 생성한 파일 (`lmc/` 패키지)

| 파일 | LMCache 원본 | 핵심 내용 |
|---|---|---|
| `lmc/compute/__init__.py` | (디렉토리 marker) | empty |
| `lmc/compute/attention/__init__.py` | 동일 | empty |
| `lmc/compute/attention/abstract.py` | `lmcache/v1/compute/attention/abstract.py` (line 1-39) | `AttentionInterface` ABC verbatim port |
| `lmc/compute/attention/metadata.py` | `…/metadata.py:17-36` | `LMCAttnMetadata` (abstract dataclass), `LMCEagerAttnMetadata` (LMCFlashAttnMetadata 의 HF 등가물; `q_positions` / `k_positions` 추가) |
| `lmc/compute/attention/eager.py` | (HF 신규; LMCFlashAttnBackend 대체) | `LMCEagerAttnBackend.forward_contiguous` — repeat_interleave GQA expansion, explicit position-based causal mask, fp32 softmax, in-place output write |
| `lmc/compute/attention/utils.py` | `…/attention/utils.py:9-23` | `infer_attn_backend_from_hf` factory (eager only) |
| `lmc/compute/blend/__init__.py` | — | empty |
| `lmc/compute/blend/metadata.py` | `lmcache/v1/compute/blend/metadata.py` | `LMCBlendCommonMetadata`, `LMCBlendMetadata` (full 3-field shape + `clean()`, Phase 0 audit 결정에 따라 Phase 1 에서 미리 선언) |
| `lmc/compute/blend/blender.py` | `…/blend/blender.py:18-86` (HKVD branch 제외) | Phase 1 stub `LMCBlender`. `process_qkv` 가 RoPE 적용 후 pass-through; `imp_indices` / write-back 없음 |
| `lmc/compute/models/__init__.py` | — | empty |
| `lmc/compute/models/base.py` | `lmcache/v1/compute/models/base.py` | `LMCBaseModel` (`@torch.compile` 제거, `[start_layer:end_layer]` → 전체 layers). HF 어댑테이션 wrapper 5종 (`_make_qkv_proj`, `_make_rotary_emb_wrapper`, `_make_o_proj_tupled`, `_make_residual_rmsnorm`, `_PATCH_MARKER` idempotency) |
| `lmc/compute/models/llama.py` | `…/models/llama.py:6-9` | `LMCLlamaModel._process_qkv` identity (verbatim) |
| `lmc/compute/models/utils.py` | `…/models/utils.py:14-30` | `infer_model_from_hf` — `LlamaForCausalLM` / `MistralForCausalLM` → `LMCLlamaModel` |
| `lmc/integration/__init__.py`, `lmc/integration/hf/__init__.py` | — | empty |
| `lmc/integration/hf/utils.py` | `lmcache/integration/vllm/utils.py:27` (`VLLMModelTracker`) | `ENGINE_NAME = "hf_cacheblend"`, `HFModelTracker` |

### 생성한 테스트 파일

- `tests/test_phase1_layerwise.py` — Phase 1 verification §1-§5 의 6 개
  tiny-model 테스트 (CPU 동작 보장) + 동일 criteria 의 real-model
  테스트 (CUDA + `LMC_PHASE1_REAL_MODELS=1` 게이트). `RecordingBlender`
  proxy 가 `process_qkv` 호출 전/후의 K 를 캡쳐하여 RoPE 가 실제로
  적용되었는지 검사한다.

### 핵심 변경 사항 요약

1. **`compute_layer` 의 call site 는 byte-identical** to LMCache
   (`lmcache/v1/compute/models/base.py:67-142`). 단 `@torch.compile`
   decorator 와 `[start_layer:end_layer]` slice 만 제거. HF API gap 은
   모두 `__init__` 안의 어댑터 patch 로 메움.
2. **stub blender 가 RoPE 를 적용** (Phase 0 audit 결정). Phase 2 의
   stock-vs-layerwise 등가성 테스트가 같은 call site 에서 post-RoPE
   K 를 볼 수 있게 한다.
3. **`_PATCH_MARKER` idempotency**: `LMCBaseModel.__init__` 가 같은
   `hf_model` 에 대해 두 번 호출되어도 모듈 wrapper 가 중복 적용되지
   않는다 (테스트로 검증).

## 2. 검증 결과

### Tiny-model 테스트 (CPU 동작, 6 tests)

```
$ python3 -m pytest tests/test_phase1_layerwise.py -v
============================= test session starts ==============================
collected 16 items

tests/test_phase1_layerwise.py::TestTinyModelLayerwise::test_criterion_1_runs_without_raising PASSED
tests/test_phase1_layerwise.py::TestTinyModelLayerwise::test_criterion_2_yields_exactly_num_hidden_layers PASSED
tests/test_phase1_layerwise.py::TestTinyModelLayerwise::test_criterion_3_final_hidden_state_shape PASSED
tests/test_phase1_layerwise.py::TestTinyModelLayerwise::test_criterion_5_process_qkv_per_layer_with_rope PASSED
tests/test_phase1_layerwise.py::TestTinyModelLayerwise::test_blend_metadata_dataclass_shape PASSED
tests/test_phase1_layerwise.py::TestTinyModelLayerwise::test_double_construction_idempotent PASSED
tests/test_phase1_layerwise.py::TestRealModelLayerwise::* SKIPPED (10 tests)

======================== 6 passed, 10 skipped in 4.15s =========================
```

### 검증된 Phase 1 §1-§5 criteria

| Criterion | Tiny model (CPU) | Real models (CUDA) |
|---|---|---|
| §1 `compute_layer` runs to completion | ✅ 2 layers, 16 tokens | ⏸ pending GPU |
| §2 generator yields exactly `num_hidden_layers` times | ✅ | ⏸ pending GPU |
| §3 final hidden shape `(num_tokens, hidden_size)` | ✅ (16, 64) | ⏸ pending GPU |
| §4 memory ≤ 1.5× first peak | n/a (CUDA-only) | ⏸ pending GPU |
| §5 `process_qkv` per-layer, K changed by RoPE | ✅ all 2 layers | ⏸ pending GPU |

### Tiny-model 구성

`LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
head_dim=16, max_position_embeddings=128, attn_implementation="eager")`.
2-layer GQA (4:2) Llama, fp32 가중치 (랜덤 초기화, seed=0). 입력은
16-token random prompt (seed=1234). `RecordingBlender` 를 통해 16-token
입력에서 layer-별 K 가 RoPE rotation 으로 변경됨을 확인 (
position 0 의 K 는 변하지 않지만 position 1..15 의 K 는 변경되므로
`torch.equal(k_pre, k_post)` 는 모든 layer 에서 False).

### Real-model 테스트 (현 환경에서 skip)

`LMC_PHASE1_REAL_MODELS=1 pytest tests/test_phase1_layerwise.py -v` 로
실행 가능. CUDA 박스 (≥ 24 GB VRAM) 가 필요하며 fp16 로 모델을 로드한다.
10 개 parametrized test ((Mistral-7B, Llama-3.1-8B) × 5 criteria)
모두 동일 코드 경로를 검사.

### 작업 중 발견한 LMCache 비스펙 사항

- `LMCBlendCommonMetadata` 의 `check_layers` 필드는 LMCache 원본에서
  `default 없음` (required) 으로 선언되어 있으나
  (`metadata.py:14-19`), Phase 0.5 spec 은 default `[1]` 로 적어 두었
  다. Phase 1 stub 에서는 이 필드를 사용하지 않으므로 `default = None`
  으로 변경하여 Phase 3 가 채울 수 있게 두었다. spec 자체와는
  무관한 변경이라 `docs/` 수정은 하지 않음 (Phase 3 prompt 에서 이미
  `LMCacheEngineConfig.blend_check_layers: List[int] = field(...lambda: [1])`
  로 별도 처리).

## 3. LMCache와의 일치도

### LMCache 원본과 동일하게 유지한 부분

| 항목 | LMCache 위치 | 본 포트 위치 |
|---|---|---|
| `AttentionInterface` ABC | `compute/attention/abstract.py:14-39` | `lmc/compute/attention/abstract.py` (전체) |
| `LMCAttnMetadata` abstract dataclass | `compute/attention/metadata.py:17-21` | `lmc/compute/attention/metadata.py:10-14` |
| `update_from_top_indices` semantics | `…/metadata.py:31-36` | `…/metadata.py:LMCEagerAttnMetadata.update_from_top_indices` |
| `LMCBlendMetadata` 3-field shape + `clean()` | `compute/blend/metadata.py:21-35` | `lmc/compute/blend/metadata.py:18-29` |
| `LMCBlender.process_qkv` 의 pre-HKVD prefix | `compute/blend/blender.py:59-86` | `lmc/compute/blend/blender.py:40-65` |
| `LMCBaseModel.compute_layer` 본문 (`@torch.compile` 과 PP slice 제외) | `compute/models/base.py:67-142` | `lmc/compute/models/base.py:LMCBaseModel.compute_layer` |
| `LMCLlamaModel._process_qkv` (identity) | `compute/models/llama.py:7-9` | `lmc/compute/models/llama.py:8-10` |
| `infer_model_from_vllm` dispatch | `compute/models/utils.py:14-30` | `lmc/compute/models/utils.py:infer_model_from_hf` |
| `VLLMModelTracker` 의 register/get pattern | `compute/models/utils.py:33-63` | `lmc/integration/hf/utils.py:HFModelTracker` |

### HF 어댑테이션으로 인해 달라진 부분

| 변경 | LMCache 위치 | HF 위치 | 근거 |
|---|---|---|---|
| `LMCFlashAttnBackend` → `LMCEagerAttnBackend` (kernel 없음, explicit causal mask) | `compute/attention/flash_attn.py:20-93` | (신규) `lmc/compute/attention/eager.py` | flash_attn dependency 회피 + HKVD 시 explicit position-based mask 필요 (PHASE1 §"`lmc/compute/attention/eager.py`") |
| `infer_attn_backend_from_vllm` (vLLM `impl` 분기) → `infer_attn_backend_from_hf` (단일 eager branch) | `compute/attention/utils.py:9-23` | `lmc/compute/attention/utils.py` | HF 는 backend 분기 없음 |
| `compute_layer` 의 `@torch.compile` 제거 | `compute/models/base.py:66` | (제거) | monkey-patch 모듈과 graph capture 충돌, 정확성 우선 (`docs/CODING_CONVENTIONS.md` §"Code style") |
| `model.layers[start_layer:end_layer]` → `model.layers` | `compute/models/base.py:84-85` | (제거) | HF 에 vLLM PP slice 없음 |
| `qkv_proj` 합성 (HF 에 fused QKV 없음) | (vLLM 내장) | `_make_qkv_proj` in `lmc/compute/models/base.py` | HF `LlamaAttention` 의 `q_proj/k_proj/v_proj` (`modeling_llama.py:238-249`) |
| `rotary_emb(positions, q, k) → (q_rot, k_rot)` wrapper. HF 의 rotary 는 `LlamaModel` 에 있음 (`modeling_llama.py:366`), `LlamaAttention` 이 아님. | (vLLM 의 `attn_layer.rotary_emb`) | `_make_rotary_emb_wrapper` in `lmc/compute/models/base.py` | Phase 0 audit 결정 |
| `o_proj(x) → (out, None)` wrapper (nn.Module 으로) | (vLLM o_proj) | `_make_o_proj_tupled` returns `_OProjTupled(nn.Module)` | `nn.Module.__setattr__` 가 child Module 자리에 plain function 할당을 거부 |
| `input_layernorm` / `post_attention_layernorm` fused-residual wrapper | (vLLM RMSNorm) | `_make_residual_rmsnorm` returns `_ResidualRMSNorm(nn.Module)` | HF `LlamaRMSNorm.forward(hidden_states)` 는 unary (`modeling_llama.py:62-67`) |
| `embed_input_ids` alias | (vLLM 내장) | `hf_model.model.embed_input_ids = hf_model.model.embed_tokens` | HF 는 `embed_tokens` (`modeling_llama.py:361`) |
| `q_size`, `kv_size` 부착 | (vLLM 의 self_attn) | `__init__` 가 직접 attach | HF config 에서 계산 |
| `vllm_attn_layers[idx].num_heads/num_kv_heads/head_size` | (vLLM 의 attn impl) | `SimpleNamespace` 컨테이너 | HF 의 `num_attention_heads/num_key_value_heads/head_dim` |
| `input_ids.cuda()` → `.to(device)` | `compute/models/base.py:71` | `lmc/compute/models/base.py:LMCBaseModel.compute_layer` 첫 줄 | CPU/MPS 등 다양한 device 대응 (실제로는 `model.parameters()` 의 device 로 옮김) |

### 부수 효과 / 주의사항

- `LMCBaseModel(hf_model, ...)` 구성 후 `hf_model` 은 `o_proj` /
  `input_layernorm` / `post_attention_layernorm` 가 wrapping 되었기
  때문에 **stock `hf_model.forward(...)` 가 깨진다**. Phase 2 의
  테스트는 `LMCBaseModel` 구성 **이전** 에 stock forward 를 실행하고
  결과를 캡쳐한 뒤 LMC 어댑터를 빌드하는 순서를 따라야 한다 — 이
  순서는 이미 `docs/phases/PHASE2_PROMPT.md` Test body §3-§4 에서
  지키도록 명시되어 있다.

## 4. 작업 중 결정한 사항

- **`o_proj` / RMSNorm wrapper 를 `nn.Module` 로 구현**: 처음에는
  closure 로 작성했으나 PyTorch 가 `nn.Module.__setattr__` 에서 child
  Module 자리에 plain function 할당을 거부 (`TypeError: cannot assign
  '…' as child module 'o_proj'`). nn.Module subclass 로 wrap 하여
  parameter 가 정상 등록되도록 변경. 동시에 `.to(device)` / `state_dict`
  와의 호환성도 유지.
- **`qkv_proj` / `rotary_emb` 는 plain function 으로 유지**:
  이들은 HF `LlamaAttention` 에 존재하지 않는 신규 속성이므로 child
  Module 검사를 거치지 않는다. nn.Module 으로 굳이 wrap 할 이유 없다.
- **`_PATCH_MARKER` idempotency**: `infer_model_from_hf` 가 같은
  `hf_model` 에 대해 두 번 호출되면 module wrapper 가 이중 적용
  되면서 동작이 망가질 수 있다. `__init__` 의 patch loop 를 sentinel
  flag 로 보호해 idempotent 하게 만들었다 (`test_double_construction_idempotent`
  로 검증).
- **`RecordingBlender.__setattr__` 가 `layerwise_model` 을 inner 로
  forward**: `LMCBaseModel.__init__` 가 `blender.layerwise_model = self`
  를 실행하면 proxy 에만 설정되어 inner stub 의
  `self.layerwise_model.vllm_model...` 가 None 참조로 실패했다.
  `__setattr__` 에서 inner 로 mirror 하여 해결.
- **input_ids 의 device 이동**: LMCache 원본은 `input_ids.cuda()` 를
  hardcode 하지만, HF 포트는 `next(self.vllm_model.parameters()).device`
  로 동적으로 옮긴다. tiny-model CPU 테스트 / 향후 MPS 가능성에
  대비. semantic 은 동일 (model 의 device 로 보장).
- **Tiny-model 통합 테스트 추가**: spec 은 real-model 만 요구하지만,
  CUDA 가 없는 환경에서도 코드 경로를 검증할 수 있도록 2-layer 합성
  Llama 를 사용한 5-criteria 동등 테스트를 추가. spec 위반은 아니며
  Phase 4 의 RAG quality 테스트가 아니므로 모델 다운로드 / 시간
  비용도 없다.

## 5. 다음 Phase 준비도

### 진행 가능 여부

- **코드 완성**: 모든 `lmc/` 파일이 작성되었고 import 가능. tiny-model
  통합 테스트 6 건 모두 pass.
- **Real-model verification 은 GPU 박스에서 실행 필요**. vast.ai
  박스 (≥ 24 GB VRAM, CUDA 12.x) 에서:

  ```bash
  cd <repo>
  python -m venv .venv && source .venv/bin/activate
  pip install -e .
  LMC_PHASE1_REAL_MODELS=1 pytest tests/test_phase1_layerwise.py -v
  ```

  10 개 parametrized real-model 테스트 모두 pass 해야 Phase 1 이
  최종 done. 본 머신에서는 환경 제약으로 실행 불가했다.

### 사용자 리뷰가 필요한 부분

1. **현재 commit 에 `phase1-complete` tag 를 붙이지 않음**. Phase 1
   prompt 는 "All tests must pass … before Phase 1 is considered done"
   이라 명시하므로 GPU verification 결과 없이 tag 를 붙이는 것은
   사실과 맞지 않다고 판단. 사용자가 GPU 박스로 이동 후 실제 테스트
   가 통과하면 tag 를 추가하거나, GPU verification 을 따로 수행하는
   workflow 로 결정 필요.
2. **`_PATCH_MARKER` 방식 vs deep copy**: 현재는 `hf_model` 인스턴스를
   in-place 수정하므로 빌드 후 stock forward 가 깨진다. Phase 2 가
   요구하는 sequence ("stock forward 먼저 → LMC 빌드") 와는 호환되
   지만, 다른 use case 가 발생하면 deep copy 방식으로 전환할 수도
   있다.
3. **Tiny-model 테스트의 spec 적합성**: 본 phase 의 verification spec
   에 명시되지 않은 추가 테스트이지만, dev 환경에서 회귀를 빨리
   잡기 위해 유지하길 권장. 사용자가 제거를 원하면 `TestTinyModelLayerwise`
   class 만 삭제하면 된다.

### 알려진 잔여 리스크 (가능성 순)

1. **Llama-3.1 RoPE round-trip 문제 가능성**: Phase 0 잔여 리스크에서
   언급한 그대로. HF 의 `LlamaRotaryEmbedding` 이 Llama-3 scaling 을
   처리하므로 forward 방향은 문제없을 것이지만, Phase 3 에서
   `FusedRope.fused_encode` (cos_old → cos_new) 를 구현할 때 정확도
   tolerance 가 빠듯할 수 있다. Phase 1 자체와는 무관.
2. **GPU 메모리 소비**: real-model 테스트는 fp16 Mistral-7B (~14 GB)
   + Llama-3.1-8B (~16 GB) 를 동시에 메모리에 보유하지 않도록
   `@pytest.fixture(scope="class")` 로 분리. 하지만 cuBLAS workspace,
   activation memory 등을 합치면 한 모델당 ~ 20 GB 가량 사용할 수
   있다. 24 GB 박스에서는 슬프게도 빠듯할 수 있어, 한 모델씩 따로
   실행하는 방법도 고려.
3. **`@torch.compile` 부재로 인한 latency**: Phase 1 spec 은 latency
   가 verification 대상이 아니지만, Phase 4 의 latency 비교에서
   eager + no-compile 의 baseline 이 LMCache 의 `@torch.compile`
   baseline 보다 느릴 것이다. 이는 의도된 trade-off (정확성 우선).
4. **HF 5.7.0 의 `LlamaAttention` 내부 변경**: `@use_kernelized_func`
   decorator 가 미래 버전에서 `self.config` 접근 방식을 변경할 수
   있다. 본 phase 는 `attn.config` 인스턴스 속성을 신뢰하고 있다.
   Phase 2 에서 실제 동작이 확인되면 안정성도 보장된다.

---

> Phase 1 의 산출물 (`lmc/`, `tests/test_phase1_layerwise.py`) 은
> 이 commit 에 포함된다. `phase1-complete` tag 는 GPU 박스에서
> real-model 테스트 통과 후에 부여 권장.
