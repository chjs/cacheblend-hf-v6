# Phase 0.5 작업 보고서

> Phase 0.5 는 구현 작업 없이 (1) harness 디렉토리 구조 정리, (2) GitHub
> 원격 저장소 연결, (3) 한국어 보고 양식 도입을 수행하는 단계였다.

## 1. 수행한 작업

### 디렉토리 구조 변경 (`mv` + `mkdir`)

이전 (flat layout) → 이후 (project layout):

| 이전 경로 | 이후 경로 |
|---|---|
| `LMCACHE_IMPLEMENTATION.md` | `docs/LMCACHE_IMPLEMENTATION.md` |
| `PROJECT_PLAN.md` | `docs/PROJECT_PLAN.md` |
| `CODING_CONVENTIONS.md` | `docs/CODING_CONVENTIONS.md` |
| `VERIFICATION_PROTOCOL.md` | `docs/VERIFICATION_PROTOCOL.md` |
| `PHASE1_PROMPT.md` | `docs/phases/PHASE1_PROMPT.md` |
| `PHASE2_PROMPT.md` | `docs/phases/PHASE2_PROMPT.md` |
| `PHASE3_PROMPT.md` | `docs/phases/PHASE3_PROMPT.md` |
| `PHASE4_PROMPT.md` | `docs/phases/PHASE4_PROMPT.md` |
| `README.md` | `README.md` (rewritten at root) |

새로 생성한 디렉토리: `docs/`, `docs/phases/`, `reports/`, `lmc/`,
`tests/`, `scripts/`, `results/`.

### 생성한 파일

- `.gitignore` — Python/IDE/모델 weight/HF cache/Phase 4 raw artifacts/
  OS metadata/LMCache_reference 제외.
- `pyproject.toml` — `cacheblend-hf` 패키지 (Python ≥ 3.10,
  `torch>=2.2`, `transformers>=5.7`, `accelerate`, `numpy`; dev extra
  `pytest`; phase4 extra `datasets`). `lmc*` 패턴으로 자동 패키지
  탐색.
- `lmc/__init__.py` — 빈 placeholder (Phase 1 에서 채움).
- `tests/.gitkeep`, `scripts/.gitkeep`, `results/.gitkeep` — 빈 디렉토리
  유지용.
- `docs/phases/PHASE0_PROMPT.md` — 원래 flat layout 에 존재하지
  않았던 파일을 복원. Phase 0 감사 지시서의 핵심 내용을 정리한 후,
  말미에 한국어 안내 ("Phase 0 작업 결과는 `reports/phase0_audit.md`
  에 기록되어 있다") 추가.
- `reports/phase0_audit.md` — Phase 0 감사 결과의 한국어 정식 보고서.
- `reports/phase0_5_setup.md` — 이 파일.

### 수정한 파일

- **`README.md`** — harness 전용에서 project-level README 로 재작성.
  prime directive, target models (Mistral / Llama-3.1 + RoPE scaling
  caveat), 하드웨어/소프트웨어 요구사항, 디렉토리 구조, "How to run"
  워크플로우 포함. 한국어 보고가 기본 산출물임을 명시.
- **`docs/CODING_CONVENTIONS.md`** — "Working style for Claude Code"
  바로 뒤에 신규 섹션 "Phase reports (Korean)" 추가. 보고서 템플릿,
  stdout 한 줄 안내 규칙, Phase 0 / 0.5 도 동일 양식이라는 안내 포함.
  기존 "File layout" 의 `/workspace/cacheblend_hf/` 경로 prefix 를
  실제 repo root 기준으로 정리.
- **`docs/phases/PHASE1_PROMPT.md`** ~ **`PHASE4_PROMPT.md`** — 각
  파일 말미의 "Verification" / "Final report" 블록 뒤에 한국어
  "## 작업 보고" 섹션 추가. 보고서 양식은
  `docs/CODING_CONVENTIONS.md` §"Phase reports (Korean)" 를 참조하도록
  지시. 각 phase 의 보고서가 반드시 포함해야 할 측정값 (e.g. Phase 2
  는 max abs error, Phase 3 는 sub-section 별 tolerance, Phase 4 는
  F1 / latency 표) 도 phase 별로 보완.
- **`docs/phases/PHASE0_PROMPT.md`** — 작업 보고 위치
  (`reports/phase0_audit.md`) 안내 한 줄 추가.
- **모든 `docs/phases/*.md`** — `LMCACHE_IMPLEMENTATION.md`,
  `CODING_CONVENTIONS.md`, `VERIFICATION_PROTOCOL.md`,
  `PROJECT_PLAN.md` 에 대한 평문 cross-reference 를 `docs/` prefix 가
  붙은 repo-root-relative 경로로 일괄 교체 (Python in-place 스크립트
  사용). `docs/` 내부의 sibling reference 는 그대로 bare path 유지.

### Git operations

```
git init -b main
git remote add origin https://github.com/chjs/cacheblend-hf-v6.git
git add .
git commit -m "Phase 0: harness + audit complete ..."  # 1dfb3e5
git tag phase0-complete
git push -u origin main
git push origin phase0-complete
```

원격 저장소가 비어 있었으므로 force-push 나 rebase 는 불필요했다.
Authentication 은 시스템에 설정된 credential helper 가 처리.

## 2. 검증 결과

본 단계는 빌드/테스트가 아니므로 검증은 sanity check 로 대체했다.

### Layout 확인

```
ls -la
├── .git/
├── .gitignore
├── README.md
├── pyproject.toml
├── docs/
│   ├── CODING_CONVENTIONS.md
│   ├── LMCACHE_IMPLEMENTATION.md
│   ├── PROJECT_PLAN.md
│   ├── VERIFICATION_PROTOCOL.md
│   └── phases/
│       ├── PHASE0_PROMPT.md
│       ├── PHASE1_PROMPT.md
│       ├── PHASE2_PROMPT.md
│       ├── PHASE3_PROMPT.md
│       └── PHASE4_PROMPT.md
├── reports/
│   ├── phase0_audit.md
│   └── phase0_5_setup.md
├── lmc/
│   └── __init__.py
├── tests/.gitkeep
├── scripts/.gitkeep
└── results/.gitkeep
```

### Package import

```
$ python3 -c "import lmc; print('lmc package OK; path =', lmc.__file__)"
lmc package OK; path = .../cacheblend-harness/lmc/__init__.py
```

(로컬 dev 머신은 Homebrew Python + PEP 668 정책으로 `pip install -e .`
가 차단된다. 패키지는 `lmc/` 가 cwd 에 존재하기 때문에 그대로
import 된다. vast.ai GPU 박스에는 `venv` 안에서 `pip install -e .` 가
정상 동작할 것.)

### Cross-reference 정합성

`docs/phases/*.md` 의 모든 cross-reference 가 `docs/`-prefixed 경로
로 정리됨. `grep -rn "VERIFICATION_PROTOCOL.md|LMCACHE_IMPLEMENTATION.md|
CODING_CONVENTIONS.md|PROJECT_PLAN.md" docs/phases/ | grep -v "docs/"`
의 결과가 비어 있음을 확인.

### Git 상태

```
$ git log --oneline
1dfb3e5 Phase 0: harness + audit complete

$ git tag
phase0-complete

$ git remote -v
origin  https://github.com/chjs/cacheblend-hf-v6.git (fetch)
origin  https://github.com/chjs/cacheblend-hf-v6.git (push)

$ git ls-remote origin
1dfb3e572b2fe8d7bccd301b13c0b35910927136  HEAD
1dfb3e572b2fe8d7bccd301b13c0b35910927136  refs/heads/main
1dfb3e572b2fe8d7bccd301b13c0b35910927136  refs/tags/phase0-complete
```

원격의 `HEAD`, `refs/heads/main`, `refs/tags/phase0-complete` 모두
동일 commit `1dfb3e5` 를 가리킴.

## 3. LMCache와의 일치도

본 단계는 코드 변경이 없으므로 LMCache 와 직접적인 일치도 비교 대상은
없다. 다만:

- `docs/` 내 spec 파일들의 LMCache file:line 인용은 그대로 보존됨
  (이동만 했고 내용은 수정하지 않음).
- 각 phase prompt 의 "Review notes" 섹션은 Phase 0 작업의 산출물
  이며 line 단위 인용을 그대로 유지.

## 4. 작업 중 결정한 사항

- **Cross-reference 표기 통일**: `docs/phases/*.md` 의 외부 참조는
  repo-root-relative `docs/X.md` 형태로 통일. `docs/` 내부 sibling
  참조는 bare path 유지 (e.g.
  `docs/LMCACHE_IMPLEMENTATION.md` 안에서는 `CODING_CONVENTIONS.md`
  만으로 충분). 이유: phase prompt 는 repo root 에서 paste 되므로
  절대 경로가 명확하지만, docs/ 내부 spec 끼리는 동일 디렉토리 내
  참조가 자연스럽다.
- **`PHASE0_PROMPT.md` 신규 작성**: 원래 flat layout 에 존재하지
  않았던 파일을 명세대로 `docs/phases/PHASE0_PROMPT.md` 로 생성.
  내용은 Phase 0 감사 지시서의 핵심 단계 (read harness → read paper →
  read LMCache → verify HF API → cross-check prompts → apply fixes →
  final report) 를 요약. Phase 0 실행 시 사용자 메시지의 전체
  내용을 그대로 복원하지는 않고, 작업 흐름이 재현 가능한 수준의
  요약으로 정리.
- **`.gitkeep` 사용**: `tests/`, `scripts/`, `results/` 는 Phase 1
  이후에 채워지므로 빈 디렉토리를 git 에 추적시키기 위해 `.gitkeep`
  배치.
- **`.gitignore` 의 `results/*.json` / `results/raw/`**: Phase 4 가
  생성할 large JSON / raw output 은 ignore, 단 markdown 표
  (`results/phase4_*.md`) 는 commit 가능하도록 패턴 선정.
- **`@torch.compile` / Llama-3.1 정책 등 Phase 0 의 결정 사항은
  변경 없이 그대로 유지**.

## 5. 다음 Phase 준비도

### 진행 가능 여부

**Phase 1 실행 준비 완료**:

- `docs/phases/PHASE1_PROMPT.md` 가 한국어 보고 양식 안내까지 포함한
  최종 상태.
- 작업 산출물 위치 (`lmc/`, `tests/`) 가 비어 있고 디렉토리만 존재
  하므로 Claude Code 가 `pip install -e .` 후 곧바로 채워 넣을 수
  있다.
- Phase 0 의 수정사항은 `phase0-complete` 태그로 영구 기록.
- 원격 저장소: https://github.com/chjs/cacheblend-hf-v6.git
  - branch: `main` at `1dfb3e5`
  - tag: `phase0-complete` at `1dfb3e5`

### 사용자 리뷰가 필요한 부분

- 디렉토리 layout 의 세부 (e.g. `docs/phases/` vs `phases/`,
  `reports/` vs `report/`): 사용자가 다른 컨벤션을 선호한다면 roll back.
- 한국어 보고 템플릿의 항목 구성 (5 개 section): 항목 추가/삭제가
  필요하면 `docs/CODING_CONVENTIONS.md` §"Phase reports (Korean)"
  에서 수정.
- `PHASE0_PROMPT.md` 의 내용을 원본 Phase 0 사용자 prompt 그대로
  복원할지, 아니면 현재의 요약본을 유지할지.

### 알려진 잔여 리스크

1. **로컬 dev 머신에서 `pip install -e .` 차단**: PEP 668 정책으로
   Homebrew Python 에 system-wide 설치가 막힌다. vast.ai 의 venv 환경
   에서는 문제없을 것이지만, 실행 전 venv activation 을 확인해야
   한다.
2. **GitHub 원격 저장소 권한**: 현재 push 가 성공했으므로 credential
   helper 가 정상 동작 중이지만, vast.ai 박스에서는 별도로 token /
   ssh key 설정이 필요할 수 있다.
3. **`PHASE0_PROMPT.md` 가 원본 prompt 와 1:1 매칭되지 않음**: 위
   "사용자 리뷰가 필요한 부분" 참고. 사용자가 원본 그대로의 복원을
   원하면 교체 필요.

---

> Phase 0.5 의 모든 변경은 `1dfb3e5` 와 그 이후의 Phase 0.5 commit 에
> 기록된다. Phase 1 은 fresh Claude Code 세션에서
> `docs/phases/PHASE1_PROMPT.md` 를 paste 하여 시작.
