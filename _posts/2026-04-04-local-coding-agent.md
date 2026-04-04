---
layout: post
title: "[Project] Local Coding Agent - Claude Code 아키텍처 기반 로컬 AI 코딩 에이전트"
categories: [Project]
tags: [python, llm-agent, ollama, terminal-ui, agentic-loop, asyncio]
math: false
---

## 프로젝트 개요

Claude Code의 아키텍처 패턴을 참고하여 구축한 터미널 기반 AI 코딩 에이전트다. Ollama + Qwen3:32B를 로컬에서 구동하여 코드 읽기·쓰기·편집·검색·실행을 수행한다. AsyncGenerator 기반 에이전틱 루프, 6개 코어 도구, 3-tier 권한 시스템, 토큰 추적 기반 컨텍스트 관리를 갖추고 있으며, API 비용 없이 프라이버시가 보장되는 로컬 AI 코딩 환경을 제공한다.

Phase 2에서는 2-Stage Permission Classifier(자동 권한 판정), Tier 2 Incremental Compaction(LLM 기반 점진적 컨텍스트 압축), Project Memory Manager(영속적 학습 기록), Auto-Lint/Test Feedback Loop(자동 린트·수정 루프), JSON 기반 Configuration System을 추가하여 에이전트의 자율성과 안정성을 강화했다.

> GitHub: [leeje008/local-coding-agent](https://github.com/leeje008/local-coding-agent)

---

## 핵심 문제와 해결 접근

### 문제 1: 클라우드 AI 코딩 에이전트의 비용과 프라이버시 문제를 어떻게 해결할 것인가

Claude Code, Cursor, GitHub Copilot 등 클라우드 기반 AI 코딩 에이전트는 사용량에 비례하여 API 비용이 발생한다. 또한 코드가 외부 서버로 전송되므로 민감한 프로젝트에서 프라이버시 우려가 있다. 개인 개발 환경에서 비용 부담 없이 지속적으로 사용할 수 있는 코딩 에이전트가 필요하다.

**해결: Ollama + Qwen3:32B 로컬 LLM으로 완전한 로컬 실행**

M4 Pro MacBook(48GB RAM) 환경에서 Ollama로 Qwen3:32B 모델을 구동한다. 모든 추론이 로컬에서 수행되므로 API 비용이 0원이며, 코드가 외부로 전송되지 않는다. 역할별 모델 라우팅(main/classifier/subagent/compactor)을 지원하여, 향후 경량 모델(8B)을 분류기로 활용하는 멀티 모델 구성이 가능하다.

```
[사용자 입력]
     │
     ▼
[LLM Client]  ──→  role-based model routing
     │              ├── main:       qwen3:32b   (코드 생성/분석)
     │              ├── classifier: qwen3:8b    (의도 분류, Phase 2)
     │              ├── subagent:   qwen3:32b   (병렬 탐색, Phase 3)
     │              └── compactor:  qwen3:8b    (컨텍스트 압축, Phase 2)
     │
     ▼
[Ollama Server]  ──→  로컬 GPU 추론 (API 비용 0원)
```

| 비교 항목 | 클라우드 에이전트 | Local Coding Agent |
|-----------|-----------------|-------------------|
| LLM 비용 | 사용량 비례 과금 | 0원 (로컬 GPU) |
| 프라이버시 | 코드 외부 전송 | 완전 로컬 실행 |
| 오프라인 | 불가 | 가능 |
| 모델 선택 | 공급자 제한 | Ollama 지원 모델 자유 선택 |
| 컨텍스트 | 128K+ 토큰 | 32K 토큰 (로컬 제약) |

---

### 문제 2: 로컬 LLM의 제한된 컨텍스트와 불안정한 tool calling을 어떻게 극복할 것인가

로컬 LLM은 클라우드 모델 대비 컨텍스트 윈도우가 작고(기본 2048 토큰), tool calling 성공률이 불안정하다. Qwen3:32B는 thinking 모드에서 tool calling 시 JSON 파싱 실패가 빈번하게 발생한다. 대화가 길어지면 컨텍스트가 초과되어 이전 맥락을 잃는 문제도 있다.

**해결: AsyncGenerator 에이전틱 루프 + Tier 1 컨텍스트 압축 + Qwen3 최적화**

에이전틱 루프를 `AsyncGenerator`로 구현하여, 도구 호출과 권한 확인 시 루프를 일시정지(yield)하고 사용자 응답 후 재개(send)하는 구조를 채택했다. Qwen3의 tool calling 안정성을 위해 `think=False`(사고 과정 비활성화)와 `temperature=0.5`를 적용하여 ~90% 성공률을 확보했다. `num_ctx`를 최소 16384로 설정하여 기본값(2048)의 컨텍스트 한계를 극복한다.

```
[에이전틱 루프 — AsyncGenerator 기반]

User Input ──→ LLM 호출 ──→ 응답 파싱
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
               텍스트 응답   도구 호출    대화 종료
                    │           │           │
                    ▼           ▼           ▼
               yield 출력   권한 확인      return
                    │       (yield/send)
                    │           │
                    │      도구 실행
                    │           │
                    └───────────┘
                         │
                    컨텍스트 체크
                    (토큰 > 임계값?)
                         │
                    ┌────┴────┐
                    ▼         ▼
                 계속      Tier 1 압축
                         (LLM 요약)
                         │
                         ▼
                    압축된 컨텍스트로 계속
```

컨텍스트 관리자가 매 턴마다 토큰 사용량을 추적하고, 임계값 초과 시 Tier 1 마이크로 압축을 실행한다. 압축 과정에서 LLM이 대화 히스토리를 요약하여 토큰 예산 내에서 맥락을 유지한다.

**Phase 2 확장: Tier 2 Incremental Compaction**

Tier 1만으로는 장시간 세션에서 컨텍스트 손실이 불가피하다. Phase 2에서는 Factory.ai의 Anchored Iterative Summarization 패턴을 도입하여 LLM 기반 점진적 요약 시스템을 구축했다. 핵심 원칙은 전체 대화를 재요약하지 않고, 새로 밀려난 메시지만 요약한 뒤 기존 앵커 요약에 병합하는 것이다.

```
[Tier 2 Incremental Compaction — Anchored Iterative Summarization]

토큰 사용률 모니터링 (매 턴)
         │
    ratio >= 0.75?  ──No──→  계속 진행
         │ Yes
         ▼
  새로 밀려난 메시지 추출
  (anchor_idx ~ cutoff)
         │
         ▼
  LLM 요약 (compactor role)
  ├── Goal: 사용자 목적 1문장
  ├── Files Involved: 파일 경로 + 변경 사항
  ├── Key Findings: 핵심 발견/에러
  ├── Completed: 완료된 작업
  └── Next Steps: 다음 단계
         │
         ▼
  기존 앵커 요약과 병합 (중복 제거)
         │
         ▼
  원본 메시지를 요약 메시지로 교체
  (anchor_idx 업데이트)
```

Verbatim 보존 규칙을 적용하여 파일 경로(`src/auth.ts:52`), 함수/클래스명, 에러 메시지, 커밋 SHA 등은 원문 그대로 유지한다. 이를 통해 요약 과정에서의 환각(hallucination)을 방지하고, 압축 후에도 정확한 코드 네비게이션이 가능하다.

| 항목 | Tier 1 (Phase 1) | Tier 2 (Phase 2) |
|------|------------------|------------------|
| 트리거 | 토큰 > 임계값 | 사용률 >= 75% |
| 방식 | 오래된 도구 출력 삭제 | LLM 기반 점진적 요약 |
| 비용 | 무료 (단순 삭제) | LLM 호출 필요 |
| 맥락 보존 | 낮음 (출력만 삭제) | 높음 (구조화된 요약) |
| 대상 | 도구 결과 메시지 | 전체 대화 메시지 |

---

### 문제 3: 코딩 에이전트의 파일 시스템 접근 권한을 어떻게 안전하게 관리할 것인가

AI 코딩 에이전트는 파일 읽기·쓰기·삭제와 셸 명령 실행 권한을 가진다. 무제한 권한을 부여하면 의도치 않은 파일 변경이나 위험한 명령 실행이 발생할 수 있다. Claude Code의 권한 모델을 참고하되, 로컬 환경에 맞는 간결한 구현이 필요하다.

**해결: 3-tier 권한 시스템 (allow / deny / ask)**

도구를 읽기 전용(GlobTool, GrepTool, FileReadTool)과 쓰기(FileWriteTool, FileEditTool, BashTool)로 분류한다. 읽기 전용 도구는 자동 허용하고, 쓰기 도구는 실행 전 사용자 확인을 요청한다. 사용자는 개별 승인(`y`), 거부(`n`), 또는 세션 내 항상 허용(`a`)을 선택할 수 있다.

```
[도구 실행 요청]
       │
       ▼
  Permission Manager
       │
  ┌────┴──────────────────┐
  │  도구 분류 확인         │
  ├────────────────────────┤
  │ READ-ONLY (자동 허용)   │  → GlobTool, GrepTool, FileReadTool
  │ WRITE (확인 필요)       │  → FileWriteTool, FileEditTool, BashTool
  └────┬──────────────────┘
       │ (WRITE 도구인 경우)
       ▼
  사용자 프롬프트
  ├── y (허용)     → 실행
  ├── n (거부)     → 건너뛰기
  └── a (항상허용) → 세션 화이트리스트에 추가 후 실행
```

| 도구 | 분류 | 권한 |
|------|------|------|
| GlobTool | 읽기 전용 | 자동 허용 |
| GrepTool | 읽기 전용 | 자동 허용 |
| FileReadTool | 읽기 전용 | 자동 허용 |
| FileWriteTool | 쓰기 | 확인 필요 |
| FileEditTool | 쓰기 | 확인 필요 |
| BashTool | 쓰기 | 확인 필요 |

**Phase 2 확장: 2-Stage Permission Classifier (Auto Mode)**

Phase 1의 수동 확인 방식은 안전하지만, 반복적인 승인 요청이 작업 흐름을 저해한다. Phase 2에서는 Claude Code의 Auto Mode를 참고하여 2단계 자동 권한 판정 시스템을 구축했다. Stage 1에서 경량 분류를 수행하고, 불확실한 경우에만 Stage 2에서 체인-오브-사고(Chain-of-Thought) 기반 상세 분석을 진행한다.

```
[2-Stage Permission Classifier]

도구 실행 요청
       │
       ▼
  Stage 1: Fast Filter (classifier role, ~100ms)
  ├── 읽기/검색 = always safe
  ├── 프로젝트 파일 쓰기 = safe
  ├── 삭제/force-push/미확인 스크립트 = suspicious
  └── 시스템 디렉토리/외부 서비스 = suspicious
       │
  ┌────┴────────────────┐
  │                     │
  ▼                     ▼
 allow              uncertain / block
  │                     │
  ▼                     ▼
 실행              Stage 2: Reasoning (main role, ~2s)
                   3가지 리스크 평가 (0.0 ~ 1.0):
                   ├── SCOPE ESCALATION: 사용자 요청 범위 초과?
                   ├── UNTRUSTED TARGET: 미확인 시스템 대상?
                   └── DESTRUCTIVE IMPACT: 비가역적 영향?
                        │
                   ┌────┴────┐
                   ▼         ▼
                 allow     block
                   │         │
                   ▼         ▼
                 실행    서킷 브레이커 체크
                        (연속 3회 또는 누적 20회 block 시
                         수동 승인으로 자동 전환)
```

서킷 브레이커는 분류기가 과도하게 차단하는 상황을 방지한다. 연속 3회 block 또는 세션 내 누적 20회 block에 도달하면 자동으로 수동 승인 모드(`ask`)로 전환하여, 분류기 오류로 인한 작업 중단을 방지한다.

---

## 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                    Terminal UI (Rich)                         │
│                                                              │
│  ├── 스트리밍 출력 (토큰 단위 렌더링)                          │
│  ├── 도구 호출 표시 (이름 + 파라미터 + 결과)                   │
│  ├── 마크다운 렌더링                                          │
│  ├── 권한 프롬프트 (y/n/a)                                    │
│  └── 슬래시 커맨드 (/auto /memory /compact /models /config)   │
│                                                              │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────┼───────────────────────────────────────┐
│              Agentic Loop (AsyncGenerator)                    │
│                      │                                       │
│  ┌───────────────────┼──────────────────────────────┐        │
│  │                   │                              │        │
│  ▼                   ▼                              ▼        │
│  LLM Client       Tool Registry              Context Manager│
│  (Ollama)          (6 Core Tools)            (Token Tracking)│
│  │                   │                              │        │
│  ├── chat()          ├── GlobTool                   ├── 토큰 │
│  ├── stream()        ├── GrepTool                   │   카운팅│
│  ├── model routing   ├── FileReadTool               ├── Tier1│
│  │   (role-based)    ├── FileWriteTool              │   압축  │
│  └── keep_alive      ├── FileEditTool               ├── Tier2│
│      preload         └── BashTool                   │   압축  │
│                          │                          └── 예산  │
│                          ▼                             관리   │
│                   Permission Manager                         │
│                   ├── 3-tier (allow / deny / ask)             │
│                   └── 2-Stage Classifier (Phase 2)           │
│                       ├── Stage 1: Fast Filter               │
│                       └── Stage 2: Reasoning + 서킷 브레이커  │
│                                                              │
│  ┌─────────────────── Phase 2 확장 ──────────────────────┐   │
│  │                                                       │   │
│  │  IncrementalCompactor   MemoryManager   AutoTestRunner│   │
│  │  (Tier 2 요약·병합)     (MEMORY.md)     (린트→자동수정)│   │
│  │                                                       │   │
│  │  Config System (config/default_settings.json)         │   │
│  │  └── 역할별 모델·권한·에이전트 파라미터 통합 관리       │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────┼───────────────────────────────────────┐
│              Ollama Server (로컬)                              │
│                      │                                       │
│  역할별 모델 라우팅 (default_settings.json)                    │
│  ├── main:      qwen3:32b (ctx 65536, temp 0.5, 상시 로드)   │
│  ├── classifier: qwen3:32b (ctx 4096,  temp 0.1, 10분 유지)  │
│  ├── subagent:  qwen3:32b (ctx 32768, temp 0.5, 5분 유지)    │
│  └── compactor: qwen3:32b (ctx 16384, temp 0.3, 5분 유지)    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 기술 스택

| 분류 | 기술 | 버전 | 선택 이유 |
|------|------|------|----------|
| **Language** | Python | 3.13+ | 최신 asyncio 개선, 타입 힌트 강화 |
| **Package Manager** | uv | - | pip 대비 10~100배 빠른 의존성 해결 |
| **LLM Runtime** | Ollama | 0.4+ | 로컬 LLM 서빙, REST API, 모델 관리 |
| **LLM Model** | Qwen3:32B | - | 코드 생성 성능, tool calling 지원, 로컬 구동 가능 |
| **Terminal UI** | Rich | 13.0+ | 마크다운 렌더링, 스트리밍 출력, 프로그레스 표시 |
| **Validation** | Pydantic | 2.0+ | 도구 입력 JSON Schema 생성, 타입 검증 |
| **Tokenizer** | qwen-tokenizer | 0.2+ | Qwen3 모델 정확한 토큰 카운팅 |
| **Async** | asyncio + AsyncGenerator | stdlib | 에이전틱 루프 pause/resume, 비동기 도구 실행 |
| **Testing** | pytest + pytest-asyncio | 8.0+ | 비동기 테스트 네이티브 지원 |
| **Linter** | Ruff | 0.11+ | Rust 기반 초고속 린터/포매터 |
| **Build** | Hatchling | - | PEP 517 빌드 백엔드, pyproject.toml 네이티브 |

---

## Phase 2 확장 기능

Phase 2에서는 에이전트의 자율성을 높이고 장시간 세션 안정성을 강화하는 5개 신규 모듈(+1,318줄)을 추가했다.

### Project Memory Manager

프로젝트별 학습 기록을 `.agent/memory/MEMORY.md`에 영속적으로 저장한다. Claude Code의 auto memory 패턴을 참고하되, 코드에서 추론 가능한 정보(파일 구조, git 히스토리 등)는 저장하지 않고 런타임에서만 획득 가능한 비자명(non-obvious) 정보만 기록한다.

```
[Memory Manager 동작 흐름]

에이전트가 새로운 정보 발견
       │
       ▼
  카테고리 분류 (6종)
  ├── build_command:         빌드/린트 명령어
  ├── architecture_decision: 설계 패턴 결정
  ├── error_pattern:         반복되는 에러 패턴
  ├── user_preference:       사용자 컨벤션
  ├── tool_discovery:        유용한 도구 조합
  └── convention:            프로젝트 표준
       │
       ▼
  중복 감지 (기존 MEMORY.md에 동일 내용 존재?)
  ├── 중복 → 건너뛰기
  └── 신규 → 타임스탬프 포맷으로 저장
              [YYYY-MM-DD] **category**: entry
       │
       ▼
  200줄 제한 관리 (초과 시 오래된 항목 정리)
```

### Auto-Lint/Test Feedback Loop

Aider에서 영감을 받은 자동 린트·수정 루프다. 파일 편집 후 즉시 린터를 실행하고, 실패 시 에러 출력을 LLM에 피드백하여 자동 수정을 시도한다. 최대 3회까지 자동 수정을 반복하여 린트 클린 상태를 유지한다.

```
[Auto-Lint Feedback Loop]

FileEditTool / FileWriteTool 실행
       │
       ▼
  자동 린트 실행 (ruff check {file})
       │
  ┌────┴────┐
  ▼         ▼
 pass      fail
  │         │
  ▼         ▼
 완료    에러 출력 캡처 → LLM 피드백
              │
              ▼
         LLM 자동 수정 (FileEdit)
              │
              ▼
         재린트 (최대 3회 반복)
              │
         ┌────┴────┐
         ▼         ▼
        pass     max_attempts 도달
         │         │
         ▼         ▼
        완료    사용자에게 알림
```

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `auto_lint` | `true` | 파일 편집 후 자동 린트 실행 |
| `lint_command` | `ruff check {file}` | 린트 명령어 (`{file}` 플레이스홀더) |
| `auto_test` | `false` | 자동 테스트 실행 (opt-in) |
| `test_command` | `pytest {file} -x` | 테스트 명령어 |
| `max_fix_attempts` | `3` | 자동 수정 최대 시도 횟수 |
| `timeout` | `30` | 명령 실행 타임아웃 (초) |

### Configuration System

JSON 기반 통합 설정 시스템으로, 역할별 모델 파라미터·권한·에이전트 동작·자동 테스트 설정을 하나의 파일(`config/default_settings.json`)에서 관리한다.

```json
{
  "models": {
    "main":       {"name": "qwen3:32b", "num_ctx": 65536, "temperature": 0.5, "keep_alive": "-1"},
    "classifier": {"name": "qwen3:32b", "num_ctx": 4096,  "temperature": 0.1, "keep_alive": "10m"},
    "subagent":   {"name": "qwen3:32b", "num_ctx": 32768, "temperature": 0.5, "keep_alive": "5m"},
    "compactor":  {"name": "qwen3:32b", "num_ctx": 16384, "temperature": 0.3, "keep_alive": "5m"}
  },
  "permissions": {
    "allow": ["GlobTool", "GrepTool", "FileReadTool"],
    "deny": [],
    "ask": ["FileWriteTool", "FileEditTool", "BashTool"]
  },
  "agent": {
    "max_tool_retries": 3,
    "circuit_breaker_threshold": 5,
    "compaction_threshold": 0.6,
    "max_output_chars": 10000,
    "bash_timeout": 30
  }
}
```

역할별 `temperature` 튜닝으로 작업 특성에 맞는 출력을 생성한다. main(0.5)은 창의성과 안정성의 균형, classifier(0.1)는 결정적 판단, compactor(0.3)는 일관된 요약을 지향한다. `keep_alive` 설정으로 자주 사용하는 모델을 메모리에 상주시켜 cold start를 방지한다.

---

## 성과 지표

| 항목 | Phase 1 | Phase 2 |
|------|---------|---------|
| 코어 도구 | 6종 | 6종 (변동 없음) |
| 소스 파일 | 21개 | 26개 (+5 신규 모듈) |
| 테스트 파일 | 8개 | 13개 (+5 신규 테스트) |
| 테스트 수 | 38개 | 65개 (+27) |
| 신규 코드 | - | +1,318줄 |
| LLM 운영 비용 | 0원 | 0원 (로컬 Ollama) |
| Tool Calling 성공률 | ~90% | ~90% (Qwen3 최적화 유지) |
| 권한 관리 | 3-tier 수동 | 2-Stage Classifier + 서킷 브레이커 |
| 컨텍스트 관리 | Tier 1 (삭제) | Tier 1 + Tier 2 (LLM 요약) |
| 메모리 | 없음 | MEMORY.md 영속 기록 |
| 자동 린트 | 없음 | 파일 편집 후 자동 린트·수정 |
| 설정 | 하드코딩 | JSON 기반 통합 설정 |
| 로드맵 | Phase 5 단계 | Phase 2 완료, Phase 3 (Sub-agents) 예정 |
| 대상 환경 | M4 Pro MacBook (48GB RAM) | 동일 |
