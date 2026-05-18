# Phase 0 작업 보고서

> Phase 0 는 구현 작업 없이 harness `.md` 파일을 감사하고 수정하는
> 단계였다. 따라서 본 보고서의 "검증 결과" 는 실제 LMCache / HF 소스를
> 교차 독해한 결과를 의미한다.

## 1. 수행한 작업

수정한 harness 파일 (모두 영문 그대로 유지, 각 파일 하단에
`## Review notes` 섹션 추가):

### `README.md`
- Llama-3.1 RoPE scaling 주의사항 추가. LMCache 자체
  `validate_rope_params` (`compute/positional_encoding.py:85-109`) 는
  `rope_scaling != None` 을 거부하므로 LMCache 만으로는 Llama-3.1 에
  blending 이 비활성화된다. HF 포트는 HF 의 rotary 를 사용해 Llama-3
  scaling 을 우회한다.
- 본문 끝에 "Review notes" 섹션 안내를 추가하여, 각 파일 하단의 감사
  메모를 참조하도록 유도.

### `docs/LMCACHE_IMPLEMENTATION.md`
- §2.3 `process_qkv` 본문에서 `attn_output = attn_output[:topk_num]`
  의 위치를 LMCache 원본 (`blender.py:103-113`) 대로
  `self.metadata.imp_indices` / `self.metadata.positions` 갱신
  **이후** 로 이동. `assert self.common_metadata.recomp_ratios is not None`
  도 누락되어 있어 추가.
- §2.3 의 invariants 에 "LMCache 는 paper §4.2 의 gradual filtering 을
  구현하지 않는다" 는 문장을 명시. `imp_indices` 는 check layer 에서
  단 1 회 결정되고 이후 layer 에서 재사용된다.
- §2.3 의 끝에 HF 포트에서 `attn_layer.rotary_emb(positions, q, k)`
  가 vLLM 식 API 이며, HF 의 실제 rotary 는 `model.model.rotary_emb`
  에 있다는 주의 추가.
- §2.4 `compute_layer` 의 약식 단계 목록을
  `compute/models/base.py:67-142` 의 실제 코드 블록으로 교체. 누락
  되어 있던 attention 이후의 K, V 2-D 복원 reshape, `input_ids.cuda()`
  호출, `[start_layer:end_layer]` slicing (HF 포트는 full range 사용)
  포함. LMCache 가 `compute_layer` 에 `@torch.compile` 을 적용하고
  있다는 점도 표시 (HF 포트는 의도적으로 제거).
- §2.5 Attention: `LMCAttnMetadata` 가 abstract dataclass 임을
  명시하고, `update_from_top_indices` 가 `cu_seqlens_k` /
  `max_seq_len` 을 건드리지 않는다는 점을 강조.
- §2.6 RoPE: `FusedRope.fused_encode` 가 CUDA kernel 로 in-place 변경
  + return 임을 명시. 또한 `validate_rope_params` 가 Llama-3.1 을
  거부한다는 점, 그리고 HF 포트가 HF 의 rotary 위에 `FusedRope` 를
  구축해 이 제약을 우회한다는 점을 설명.
- §2.7 SegmentTokenDatabase: "advances by sep_len between chunks"
  라는 모호한 표현을 `token_database.py:464-551` 의 `start_idx`
  진행 추적으로 교체 (`idx > 0` 일 때만 `start_idx += sep_len`).
  Hash function default `"builtin"` 명시, `PYTHONHASHSEED` 의 영향
  설명 (단일 프로세스 포트에는 무관).
- §2.8 retrieve_layer + connector: 요약을 `cache_engine.py:902-1055`
  와 `gpu_connectors.py:752-907` 의 코루틴 단계별 추적으로 교체. yield
  횟수 (`num_layers + 2`), ping-pong buffer pattern, RoPE 이후의 gap
  zeroing 이 K 와 V 모두에 적용된다는 점 (line 866), `cached_positions`
  가 layer 0 에서만 갱신된다는 점 포함.
- §2.9 Configuration: env var → field name → default 의 매핑 표를
  `config.py:62-127` 인용과 함께 추가. `LMCACHE_EXTRA_CONFIG` 추가.

### `docs/CODING_CONVENTIONS.md`
- "HF-required adaptations" 의 item 3 (RoPE) 을 재작성: HF 의 rotary
  는 `LlamaModel` 에 있고 attention layer 에 없음 (`modeling_llama.py:366`).
  2-D ↔ 4-D reshape 레시피 명시.
- 새로운 item 9 추가: `past_key_values` 는 `Cache` 객체이며
  `cache.layers[i].keys/.values` 로 접근해야 함 (transformers 5.7.0,
  `cache_utils.py:1173`).
- "Code style" 의 `@torch.compile` 항목 결정: LMCache 는 사용
  (`base.py:66`), HF 포트는 제거. 이유 명시.
- Dependencies 에 transformers 버전 호환성 노트 추가 (5.x / 4.36–4.45 /
  legacy 의 분기).

### `docs/phases/PHASE1_PROMPT.md`
- Stub blender 에 RoPE 를 포함시키기로 결정 (원래 Phase 2 로 미뤄져
  있던 결정을 Phase 1 에서 해결). LMCache `process_qkv` 가
  `blender.py:79-86` 에서 항상 RoPE 를 먼저 적용하므로, Phase 2 의
  stock-vs-layerwise 등가성 테스트가 이를 요구한다.
- Stub blender 가 `LMCBlendMetadata` dataclass 전체를 (positions,
  imp_indices, attn_mask + `clean()`) 미리 선언하도록 변경. Phase 3
  에서 shape 을 바꿀 필요 없음.
- `rotary_emb` wrapper 구성 가이드 (4-D reshape 등) 와
  `infer_attn_backend_from_hf` 의 위치 (LMCache `attention/utils.py:9`
  의 parallel) 를 명시.
- vLLM `start_layer:end_layer` slicing 을 HF 의 full range 로 교체
  하라는 지시.

### `docs/phases/PHASE2_PROMPT.md`
- Phase 1 에서 RoPE 가 해결되었으므로 "if stub does not include RoPE"
  분기 제거.
- `stock_kv[i][0]` indexing 을 feature-detecting `extract_kv(cache, i)`
  helper 로 교체 (transformers 5.x `Cache` 객체 대응).
- `compute_layer` 가 `model.norm` 을 실행하지 않으므로 driver 가
  captured final hidden state 에 `model.norm` 을 적용한 뒤 비교해야
  한다는 점을 명시.
- Recording proxy 패턴으로 변경하여 stub blender 자체는 수정하지
  않게 함 (Phase 3 에서 그대로 교체 가능).
- Mistral `sliding_window` 길이 제약 (≤ 4096 tokens) 표시.

### `docs/phases/PHASE3_PROMPT.md`
- `FusedRope.fused_encode` 의 순수 PyTorch 구현 레시피 추가 (HF
  rotary 의 cos/sin 사용). LMCache 의 CUDA kernel 과 동일한 입출력
  계약 유지.
- 원본 prompt 가 누락한 보조 타입 명시: `LMCacheMetadata`, 최소한의
  `CacheEngineKey` (`split_layers` 포함), `MemoryObj.tensor` shape 과
  `metadata.cached_positions`, `MemoryFormat.KV_2TD`.
- `HFBufferLayerwiseGPUConnector.batched_to_gpu` 의 단계별 동작
  (`num_layers + 2` yields, RoPE-then-gap-zero 순서, `get_kv` aliasing
  요구) 정리.
- `BasicReverseRope` 와 `validate_reverse_correctness` 를 optional
  로 표시.
- HF `ENGINE_NAME` 을 LMCache 의 `"vllm-instance"` 와 구분.

### `docs/phases/PHASE4_PROMPT.md`
- "Full KV reuse" 정의를 구체화: Prompt Cache / Gim et al. 2023 baseline
  과 동일. `LMCFullReuseBlender.process_qkv` 의 완전한 body 제공 (Q
  만 새로 회전, 캐시된 K/V 는 connector 가 이미 회전한 상태를 그대로
  사용, HKVD branch 진입하지 않음).

### `docs/PROJECT_PLAN.md`
- Phase 3 deliverable 목록을 확장하여 `CacheEngineKey.split_layers`,
  `LMCacheMetadata`, `MemoryObj` shape, `MemoryFormat.KV_2TD` 를 포함.
- Phase 1 에서 metadata field shape 을 미리 선언한다는 점 명시.

### `docs/VERIFICATION_PROTOCOL.md`
- Phase 1 §5: stub blender 가 이제 RoPE 를 적용하므로 "returns (q, k, v)
  unchanged" 를 완화.
- Phase 1 §4: "memory does not grow unboundedly" 를 구체적인 1.5×
  smoke threshold 로 변경.
- Phase 2 setup: `Cache` 객체 추출과 `model.norm` 적용 단계 명시.

## 2. 검증 결과

본 phase 는 코드 작성이 없으므로 검증은 **교차 독해** 로 대체했다.

### 읽은 paper

- arXiv 2405.16444 (HTML + abstract) 의 §1–§5. CacheBlend 가 paper
  §4.2 에서 묘사하는 gradual filtering (r₁ > r₂ > … layer-wise narrowing)
  은 LMCache 원본에 구현되어 있지 않다는 점이 가장 중요한 차이.

### 읽은 LMCache 소스 (`chjs/LMCache`, branch
`fix/cacheblend-vllm-v0.17.1-compat`, 로컬: `/tmp/cacheblend-audit/LMCache_reference`)

- `lmcache/v1/compute/blend/blender.py` 전체 (169 lines)
- `lmcache/v1/compute/blend/metadata.py` (35 lines)
- `lmcache/v1/compute/blend/utils.py` (64 lines)
- `lmcache/v1/compute/models/base.py` (143 lines)
- `lmcache/v1/compute/models/llama.py` (10 lines)
- `lmcache/v1/compute/models/utils.py` (64 lines)
- `lmcache/v1/compute/attention/abstract.py`, `metadata.py`,
  `flash_attn.py`, `utils.py`
- `lmcache/v1/compute/positional_encoding.py` (203 lines)
- `lmcache/v1/token_database.py` (552 lines, 주로 `SegmentTokenDatabase`
  와 `_hash_tokens` 에 집중)
- `lmcache/v1/cache_engine.py` 의 `retrieve_layer` (902–1055)
- `lmcache/v1/gpu_connector/gpu_connectors.py` 의
  `VLLMBufferLayerwiseGPUConnector` (615–907)
- `lmcache/v1/config.py` 의 blending 관련 필드
- `lmcache/integration/vllm/utils.py` (`ENGINE_NAME = "vllm-instance"`)
- `examples/blend_kv_v1/blend.py` + `README.md`

### 읽은 HF 소스 (`transformers==5.7.0`)

- `transformers/models/llama/modeling_llama.py` 의 `LlamaRMSNorm`,
  `LlamaRotaryEmbedding`, `apply_rotary_pos_emb`, `LlamaAttention`,
  `LlamaDecoderLayer`, `LlamaModel`, `LlamaForCausalLM`
- `transformers/models/mistral/modeling_mistral.py` (병행 확인)
- `transformers/cache_utils.py` 의 `DynamicCache` 구조 (`.layers[i].keys/.values`)

## 3. LMCache와의 일치도

### 동일하게 유지하기로 결정한 부분 (verbatim port)

- `LMCBlender.process_qkv`, `blend_layer`, `blend` — control flow
  byte-for-byte 일치 목표.
- `LMCBlendMetadata`, `LMCBlendCommonMetadata` — dataclass field 명
  과 default 일치.
- `SegmentTokenDatabase.__init__`, `_fast_split_by_subtensor`,
  `process_tokens` — 동일 알고리즘. `_hash_tokens`,
  `_canonicalize_hash_inputs`, `_make_key_by_hash`, `NONE_HASH = 0` 도
  동일.
- `LMCBaseModel.compute_layer` — step 순서와 reshape pattern 동일.
  단 `@torch.compile` 만 제거, `start_layer:end_layer` 만 full range
  로 교체.
- `LMCFlashAttnMetadata` field 명을 그대로 `LMCEagerAttnMetadata` 에
  유지 (`query_start_loc`, `seq_lens`, `cu_seqlens_k`,
  `max_query_len`, `max_seq_len`). `q_positions` / `k_positions` 만
  추가.
- `VLLMBufferLayerwiseGPUConnector.batched_to_gpu` 의 yield 횟수
  (`num_layers + 2`) 와 `get_kv` aliasing 의미.
- 환경변수 이름 (`LMCACHE_*`) 과 `LMCacheEngineConfig` field 명.

### HF 어댑테이션으로 인해 달라진 부분과 근거

| 변경 | 근거 |
|---|---|
| RoPE wrapper 가 `hf_model.model.rotary_emb` 에서 읽음 (vLLM 은 attention 에 부착) | `transformers/models/llama/modeling_llama.py:366` 에서 `self.rotary_emb = LlamaRotaryEmbedding(...)` 가 `LlamaModel.__init__` 에 있음 |
| `qkv_proj` monkey-patch (HF 에는 fused QKV 없음) | `modeling_llama.py:238-249` 의 `q_proj`/`k_proj`/`v_proj` 분리 |
| `input_layernorm` / `post_attention_layernorm` 을 fused-residual wrapper 로 감쌈 | HF `LlamaRMSNorm` 은 unary (`modeling_llama.py:62-67`) |
| `o_proj` 를 `(out, None)` 반환 wrapper 로 감쌈 | HF `o_proj` 는 plain `nn.Linear` (`modeling_llama.py:247`) |
| `LMCFlashAttnBackend` → `LMCEagerAttnBackend` (+ `q_positions`/`k_positions`) | flash_attn dependency 제거 + HKVD 시 explicit position-based mask 필요 |
| `FusedRope` 가 LMCache CUDA kernel 대신 HF rotary 의 cos/sin 사용 | `lmc_ops.rotary_embedding_k_fused` dependency 회피; 부수효과로 Llama-3 RoPE scaling 지원 |
| `validate_rope_params` 우회 | LMCache 가 `rope_scaling != None` 을 거부 (`positional_encoding.py:99-102`) 하지만 HF 의 rotary 가 이를 처리하므로 Llama-3.1 지원 가능 |
| `past_key_values` 추출 시 `Cache` 객체 분기 | transformers 5.7.0 의 `cache_utils.py:1173` 의 `DynamicCache.layers[i]` |
| `@torch.compile` 제거 | `compute_layer` 내부의 monkey-patched 모듈과 dict-mutating connector path 가 graph capture 와 잘 맞지 않음. 정확성 우선. |
| `compute_layer` 의 `[start_layer:end_layer]` → `model.layers` | HF 에는 vLLM PP slice 가 없음 |

## 4. 작업 중 결정한 사항

- **RoPE in Phase 1 stub blender**: 포함시킴. LMCache `process_qkv`
  는 항상 RoPE 를 먼저 적용하므로 (`blender.py:79-86`), Phase 2 의
  등가성 테스트가 이를 요구한다. Phase 3 의 full blender 와 call site
  가 byte-identical 이 되어야 하므로 stub 에 미리 포함시키는 편이
  자연스럽다.
- **`ENGINE_NAME` 값**: harness 의 `"hf_cacheblend"` 를 유지 (LMCache
  는 `"vllm-instance"`). 두 tracker 가 동일 process 에 공존할 수도
  있으므로 구분.
- **Llama-3.1 RoPE scaling 처리**: LMCache 처럼 거부하는 대신 HF
  rotary 를 활용해 지원하는 방향. 단, README 와
  `docs/LMCACHE_IMPLEMENTATION.md` §2.6 에서 fallback 으로 "Llama-3.1
  drop 후 Mistral 전용 진행" 옵션을 문서화.
- **`@torch.compile` 제거**: LMCache 는 사용하지만 HF 포트는 명시적으로
  제거. `docs/CODING_CONVENTIONS.md` 의 "Code style" 항목에 근거 명시.
- **`BasicReverseRope` / `validate_reverse_correctness`**: optional 로
  표시. 핫패스에 호출되지 않으므로 생략 가능, parity 가 필요하면 포함.

## 5. 다음 Phase 준비도

### 진행 가능 여부

**Phase 1 prompt 를 그대로 따라 진행 가능.** 원래 draft 가 가지고
있던 주요 risk — stub blender 에서 RoPE 누락, `compute_layer` call site
drift, rotary_emb wrapper 의 잘못된 출처, `start_layer:end_layer` slice
— 가 모두 해결되었다.

### 사용자 리뷰가 필요한 부분

- README 의 Llama-3.1 fallback 정책 (HF rotary 우회 vs Mistral-only):
  사용자가 정책 의도를 다르게 가지고 있다면 roll back 필요.
- `ENGINE_NAME = "hf_cacheblend"` 값: 다른 이름 선호 시 roll back.
- `@torch.compile` 제거의 명시성: 향후 latency 측정 시 사용자가
  reverse 하길 원할 수 있음.

### 알려진 잔여 리스크 (가능성 순)

1. **Llama-3.1 RoPE round-trip 정확도**: HF rotary 의 Llama-3 scaling
   에서 `cos_old` → `cos_new` 회전이 Phase 3 RoPE round-trip
   tolerance (fp32 atol 1e-5) 를 만족하는지는 실제 실행 전엔 확신할
   수 없다. fallback: Llama-3.1 을 target 에서 제외하고 Mistral 만으로
   진행 (README 에 명시).
2. **`LMCacheMetadata` 의 최소 필드 집합**: Phase 3 에서 LMCache 의
   `metadata.py` 가 정의하는 추가 필드 (e.g. `use_mla`) 가 어디서
   참조되는지 완전히 추적하지 못했다. Phase 3 실행 중
   `AttributeError` 가 뜨면 그때마다 minimal extension 으로 대응.
3. **`qkv_proj` monkey-patch 의 reshape 방향**: 1-D vs 2-D 출력에 따라
   `dim=-1` vs `dim=0` 조정이 필요할 수 있음. Phase 1 prompt 에 이미
   `dim=-1` 로 명시했으므로 한 줄 수정 수준의 리스크.
4. **HF `output_hidden_states=True` 의 capture 경로**: transformers 5.7.0
   의 `@capture_outputs` decorator 가 `_can_record_outputs` 를 통해
   per-layer hidden state 를 노출하는 메커니즘이 모든 dtype / device
   조합에서 안정적인지 검증 필요. Phase 2 의 첫 실행에서 확인.

---

> Phase 0 의 모든 수정사항은 각 harness `.md` 파일 하단의
> `## Review notes` 섹션에서 line 단위로 인용과 함께 추적할 수 있다.
