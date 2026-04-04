---
layout: post
title: "[Project] Local Coding Agent - Claude Code 아키텍처 기반 로컬 AI 코딩 에이전트"
categories: [Project]
tags: [python, llm-agent, ollama, terminal-ui, agentic-loop, asyncio]
math: false
---

## 프로젝트 개요

Claude Code의 아키텍처 패턴을 참고하여 구축한 터미널 기반 AI 코딩 에이전트다. Ollama + Qwen3:32B를 로컬에서 구동하여 코드 읽기·쓰기·편집·검색·실행을 수행한다. AsyncGenerator 기반 에이전틱 루프, 6개 코어 도구, 3-tier 권한 시스템, 토큰 추적 기반 컨텍스트 관리를 갖추고 있으며, API 비용 없이 프라이버시가 보장되는 로컬 AI 코딩 환경을 제공한다.

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

---

## 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                    Terminal UI (Rich)                         │
│                                                              │
│  ├── 스트리밍 출력 (토큰 단위 렌더링)                          │
│  ├── 도구 호출 표시 (이름 + 파라미터 + 결과)                   │
│  ├── 마크다운 렌더링                                          │
│  └── 권한 프롬프트 (y/n/a)                                    │
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
│  └── model routing   ├── FileReadTool               ├── Tier1│
│                      ├── FileWriteTool              │   압축  │
│                      ├── FileEditTool               └── 예산  │
│                      └── BashTool                      관리   │
│                          │                                   │
│                          ▼                                   │
│                   Permission Manager                         │
│                   (allow / deny / ask)                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────┼───────────────────────────────────────┐
│              Ollama Server (로컬)                              │
│                      │                                       │
│  Qwen3:32B (기본 모델)                                        │
│  └── num_ctx: 16384+ / think: false / temp: 0.5             │
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

## 성과 지표

| 항목 | 수치 |
|------|------|
| 코어 도구 | 6종 (Glob / Grep / FileRead / FileWrite / FileEdit / Bash) |
| 소스 파일 | 21개 (core 7 + tools 9 + permissions 2 + ui 2 + entry 1) |
| 테스트 파일 | 8개 (tools 6 + context 1 + permissions 1) |
| LLM 운영 비용 | 0원 (로컬 Ollama) |
| Tool Calling 성공률 | ~90% (Qwen3 최적화 설정 적용) |
| 로드맵 | Phase 5 (Multi-model → Auto Mode → Sub-agents → MCP → TUI) |
| 대상 환경 | M4 Pro MacBook (48GB RAM) |
