---
layout: post
title: "[Project] 여행 플래너 - AI 기반 여행 일정 자동 생성 및 최적화 플랫폼"
categories: [Project]
tags: [nextjs, react, fastapi, langgraph, llm-agent, typescript, python, travel, vercel, ollama]
math: false
---

## 프로젝트 개요

Next.js 16 + FastAPI + LangGraph 기반의 AI 여행 일정 자동 생성 플랫폼이다. 사용자가 도시·날짜·선호도를 입력하면, 7단계 파이프라인이 지명 정규화 → 경로 최적화(OR-Tools TSP) → POI 탐색 → 3개 LangGraph 에이전트 협업 → 타임테이블/지도 출력까지 전 과정을 자동 처리한다. Vercel(프론트엔드) + Railway(백엔드) + 로컬 MacBook(Ollama LLM) 3-tier 배포 구조로 운영하며, Cloudflare Tunnel을 통해 로컬 GPU를 클라우드에 연결하여 LLM 비용 0원으로 서비스한다.

> GitHub: [leeje008/travel-planner](https://github.com/leeje008/travel-planner)

---

## 핵심 문제와 해결 접근

### 문제 1: Streamlit MVP의 한계를 어떻게 넘을 것인가

초기에는 Streamlit으로 빠르게 MVP를 구현했지만, 프로젝트가 성장하면서 한계가 명확해졌다. 단일 페이지 구조로 화면 분리가 어렵고, 사용자 인터랙션마다 전체 페이지가 리렌더링되며, 모바일 대응이 불가능했다. `session_state` 기반의 전역 상태 관리도 복잡한 플랜 데이터를 다루기에 부족했다.

**해결: Next.js 16 + React 19 + TypeScript 프론트엔드로 전면 마이그레이션**

Streamlit을 완전히 대체하는 Next.js 16 기반 프론트엔드를 구축했다. App Router로 5개 페이지를 분리하고, Zustand 5.0으로 플랜 데이터·채팅 메시지·UI 상태를 독립적으로 관리한다. Tailwind CSS 4 + shadcn/ui로 반응형 디자인 시스템을 구성했다.

```
frontend/
├── app/
│   ├── page.tsx              # 홈 (서버 헬스체크)
│   ├── plan/create/page.tsx  # 여행 플랜 입력 폼
│   ├── plan/result/page.tsx  # 정규화 결과 표시
│   ├── guide/page.tsx        # AI 가이드 채팅
│   └── feedback/page.tsx     # 피드백 수집
├── components/
│   └── ui/                   # shadcn/ui 컴포넌트
├── lib/
│   ├── api.ts                # API 클라이언트 (SSE 지원)
│   ├── store.ts              # Zustand 상태 관리
│   └── types.ts              # TypeScript 타입 정의
```

| 비교 항목 | Streamlit (Before) | Next.js 16 (After) |
|-----------|-------------------|-------------------|
| 라우팅 | 단일 페이지 (multi-page hack) | App Router 5개 페이지 |
| 상태 관리 | session_state (전역 공유) | Zustand 스토어 (도메인 분리) |
| 렌더링 | 전체 리렌더링 | React 19 서버/클라이언트 컴포넌트 분리 |
| 모바일 | 미지원 | Tailwind 반응형 레이아웃 |
| 컴포넌트 | 제한된 위젯 | shadcn/ui + 커스텀 컴포넌트 |
| 배포 | Railway (Python 런타임) | Vercel (Edge 네트워크) |

---

### 문제 2: 프론트엔드와 백엔드 간 실시간 AI 응답을 어떻게 전달할 것인가

AI 가이드 채팅에서 LLM이 긴 응답을 생성하는 동안, 기존 REST 방식으로는 전체 응답이 완성될 때까지 사용자가 빈 화면을 보게 된다. 체감 대기 시간이 10초 이상이 되면 사용자 경험이 크게 저하된다.

**해결: SSE 스트리밍 + LangGraph 대화 메모리**

`POST /api/v1/llm/chat/stream` 엔드포인트에서 Server-Sent Events(SSE)로 토큰 단위 스트리밍을 구현했다. LangGraph 에이전트가 토큰을 생성할 때마다 즉시 클라이언트로 전송하여 첫 토큰 표시까지의 시간을 1초 이내로 단축했다. 대화 컨텍스트는 LangGraph 메모리에 최대 20개 메시지를 유지하여 맥락 있는 대화를 지원한다.

```
[Next.js Client]              [FastAPI Backend]           [LangGraph Agent]
     │                              │                           │
     ├── POST /llm/chat/stream ──→  │                           │
     │   Content-Type:              │                           │
     │   text/event-stream          │                           │
     │                              ├── invoke agent ─────────→ │
     │                              │                           ├── 토큰 생성
     │  ◄── data: {"token":"서울"} ─ │  ◄── yield token ──────── │
     │  ◄── data: {"token":"에서"} ─ │  ◄── yield token ──────── │
     │  ◄── data: {"token":"..."} ── │  ◄── yield token ──────── │
     │  ◄── data: [DONE] ────────── │                           │
     │                              │                           │
     │  대화 메모리 (최대 20 메시지)   │                           │
```

프론트엔드의 `api.ts`에서 `fetch` + `ReadableStream`으로 SSE를 소비하고, Zustand 스토어에 토큰을 누적하여 실시간 렌더링한다.

---

### 문제 3: 클라우드 LLM 비용 없이 로컬 LLM을 프로덕션에 연결할 수 있는가

OpenAI/Anthropic API는 사용량에 비례하여 비용이 발생한다. 포트폴리오 프로젝트에서 지속적인 API 비용을 감당하기 어렵다. MacBook에서 Ollama로 llama3.1:8b를 구동할 수 있지만, Railway/Vercel에서 로컬 네트워크에 접근할 방법이 없다.

**해결: Ollama + Cloudflare Tunnel로 로컬 GPU를 클라우드에 노출**

Cloudflare Tunnel이 로컬 Ollama 서버에 공개 HTTPS URL을 부여한다. FastAPI 백엔드에서 `USE_LOCAL_LLM` 환경 변수로 로컬/클라우드 LLM을 전환할 수 있어, 개발 시에는 로컬 Ollama, 데모 시에는 클라우드 API로 유연하게 운영한다.

```
[Vercel]                [Railway - FastAPI]           [MacBook - Ollama]
  │                           │                              │
  ├── API 호출 ─────────────→ │                              │
  │                           │                              │
  │                           ├── USE_LOCAL_LLM=true         │
  │                           │   Cloudflare Tunnel ───────→ │
  │                           │   (HTTPS 터널 URL)           ├── llama3.1:8b 추론
  │                           │  ◄── SSE 응답 ────────────── │
  │  ◄── SSE 전달 ─────────── │                              │
  │                           │                              │
  │                           ├── USE_LOCAL_LLM=false         │
  │                           │   → OpenAI / Anthropic API   │
  │                           │     (클라우드 폴백)            │
```

Ollama 서버 시작 시 `OLLAMA_ORIGINS="*"` 플래그로 CORS를 허용하고, `OLLAMA_HOST="0.0.0.0"`으로 외부 접속을 개방한다. Cloudflare Tunnel은 별도 인증 없이 안전한 역방향 프록시를 제공한다.

---

### 문제 4: 3개 서비스를 어떻게 통합 개발/배포할 것인가

시스템이 Next.js 프론트엔드, FastAPI 백엔드, Ollama LLM 서버 3개로 분리되면서 개발 환경 구성이 복잡해졌다. 각 서비스를 별도 터미널에서 실행해야 하고, CORS 설정을 와일드카드(`*`)로 열어두면 보안 위험이 있다.

**해결: 통합 dev 스크립트 + 3-tier 배포 아키텍처**

`scripts/dev.sh`가 Ollama → FastAPI → Next.js 순서로 3개 서비스를 자동 기동한다. CORS는 와일드카드 대신 실제 배포 도메인 화이트리스트로 교체했다. `/health` 엔드포인트로 서버 상태를, `/api/v1/llm/models`로 사용 가능한 LLM 모델 목록을 모니터링한다.

| 서비스 | 플랫폼 | 역할 | 포트 |
|--------|--------|------|------|
| Next.js Frontend | Vercel | UI, SSR, 정적 자산 서빙 | 3000 (로컬) |
| FastAPI Backend | Railway | API, DB 접근, LLM 오케스트레이션 | 8000 |
| Ollama LLM | MacBook (로컬) | llama3.1:8b 추론 엔진 | 11434 |
| PostgreSQL 16 | Railway | 데이터 저장 + PGVector 임베딩 | Railway 내부 |
| Redis 7.2 | Railway | 캐싱 | Railway 내부 |

로컬 개발 시 `dev.sh` 하나로 전체 스택이 기동되고, 프로덕션에서는 Vercel + Railway + Cloudflare Tunnel 조합으로 각 서비스가 독립 배포된다.

---

### 문제 5: 여행 경로 최적화와 AI 추천을 어떻게 파이프라인으로 구성할 것인가

여행 일정 생성은 입력 파싱, 지명 정규화, 경로 최적화, POI 탐색, 스토리텔링 등 여러 단계가 순차적으로 실행되어야 한다. 각 단계마다 외부 API 호출과 에러 처리가 필요하고, LLM 에이전트 간 협업이 요구된다.

**해결: 7단계 파이프라인 + 3 LangGraph 에이전트 협업**

전체 프로세스를 7단계로 분리하고, 5단계(AI Package)에서 3개의 전문 에이전트가 협업한다. 각 에이전트는 독립된 역할을 수행하며, LangGraph 상태 머신이 실행 흐름을 제어한다.

```
[1. Input]        사용자 입력 (도시, 날짜, 교통수단, 선호도)
     │
[2. Normalize]    지명 정규화, 좌표 변환, 일정 파싱
     │
[3. Route]        Google OR-Tools TSP/VRP 최적 경로 산출
     │
[4. POI Search]   경로 주변 카페/맛집/관광지 탐색 + 랭킹
     │
[5. AI Package]   LangGraph 3-에이전트 협업
     │             ├── Curator: POI 선별 + 일정 배치
     │             ├── Validator: 시간/거리/영업시간 검증
     │             └── Storyteller: 추천 이유 + 스토리텔링 생성
     │
[6. Output]       타임테이블 + 지도 + 추천 텍스트 조합
     │
[7. Feedback]     사용자 피드백 수집 → RLHF 파이프라인 연동
```

| 에이전트 | 역할 | 입력 | 출력 |
|----------|------|------|------|
| **Curator** | POI 선별 및 일정 배치 | 최적 경로 + POI 목록 | 시간대별 장소 배정 |
| **Validator** | 실현 가능성 검증 | Curator 결과 | 시간/거리/영업시간 검증 결과 |
| **Storyteller** | 추천 이유 생성 | 검증된 일정 | 장소별 추천 이유 + 여행 스토리 |

경로 최적화는 Google OR-Tools의 TSP(외판원 문제) 솔버를 사용하여 방문지 순서를 최적화하고, 교통수단별 이동 시간 매트릭스를 기반으로 현실적인 동선을 산출한다. POI 탐색에는 Google Maps API와 Kakao 로컬 API를 병렬로 호출하여 다양한 소스의 장소 데이터를 수집한다.

---

## 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                     Vercel (프론트엔드)                        │
│                                                              │
│  Next.js 16 + React 19 + TypeScript                         │
│  ├── App Router (5 페이지)                                   │
│  ├── Zustand 5.0 (상태 관리)                                 │
│  ├── Tailwind CSS 4 + shadcn/ui                              │
│  └── SSE 스트리밍 클라이언트                                   │
│                                                              │
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTPS (도메인 화이트리스트 CORS)
┌──────────────────────┼───────────────────────────────────────┐
│                  Railway (백엔드)                              │
│                      │                                       │
│  FastAPI (Python 3.11+)                                      │
│  ├── 7-Stage Pipeline (입력 → 출력)                           │
│  ├── LangGraph 3-Agent (Curator / Validator / Storyteller)   │
│  ├── Google OR-Tools (TSP/VRP 경로 최적화)                    │
│  └── SSE Streaming (/api/v1/llm/chat/stream)                 │
│                      │                                       │
│  ┌───────────────────┼──────────────────┐                    │
│  │                   │                  │                    │
│  ▼                   ▼                  │                    │
│  PostgreSQL 16    Redis 7.2             │                    │
│  + PGVector       (캐싱)                │                    │
│                                         │                    │
└─────────────────────────────────────────┼────────────────────┘
                                          │ Cloudflare Tunnel
┌─────────────────────────────────────────┼────────────────────┐
│                  MacBook (로컬 LLM)      │                    │
│                                         │                    │
│  Ollama                                 │                    │
│  └── llama3.1:8b (로컬 GPU 추론)                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 기술 스택

| 분류 | 기술 | 버전 | 선택 이유 |
|------|------|------|----------|
| **Frontend** | Next.js | 16.2.2 | App Router, SSR, Vercel 네이티브 배포 |
| | React | 19 | 서버/클라이언트 컴포넌트 분리 |
| | TypeScript | 5.x | 타입 안전성, API 응답 타입 정의 |
| | Tailwind CSS | 4 | 유틸리티 퍼스트, 반응형 레이아웃 |
| | shadcn/ui | - | 접근성 준수 컴포넌트 라이브러리 |
| | Zustand | 5.0 | 경량 상태 관리, 보일러플레이트 최소화 |
| | Recharts | 3.8 | React 네이티브 데이터 시각화 |
| **Backend** | Python | 3.11+ | 비동기 네이티브, 타입 힌트 |
| | FastAPI | 0.115+ | async 네이티브, OpenAPI 자동 생성 |
| | Uvicorn | 0.41.0 | ASGI 서버 |
| | Pydantic | 2.x | 요청/응답 유효성 검증 |
| **Database** | PostgreSQL 16 | + PGVector | 벡터 임베딩 + 관계형 데이터 통합 저장 |
| | Redis | 7.2 | API 응답 캐싱, 세션 관리 |
| | SQLAlchemy (async) | 2.0+ | 타입 안전 비동기 ORM |
| | Alembic | 1.18+ | 스키마 마이그레이션 관리 |
| **LLM / AI** | LangChain + LangGraph | 0.3 / 0.2 | 멀티 에이전트 파이프라인, 대화 메모리 |
| | Ollama | llama3.1:8b | 로컬 LLM 추론, API 비용 0원 |
| | OpenAI / Anthropic | - | 클라우드 LLM 폴백 |
| **최적화** | Google OR-Tools | 9.10+ | TSP/VRP 경로 최적화 솔버 |
| **외부 API** | Google Maps API | - | 지오코딩, 장소 검색, 거리 매트릭스 |
| | Kakao 로컬 API | - | 국내 POI 검색, 카테고리 필터링 |
| **Infra** | Vercel | - | Next.js 프론트엔드 Edge 배포 |
| | Railway | - | 백엔드 + PostgreSQL + Redis 호스팅 |
| | Cloudflare Tunnel | - | 로컬 Ollama 서버 외부 노출 |
| | Docker | - | 컨테이너화, 환경 일관성 |
| | uv | - | Python 패키지 매니저 (pip 대체) |

---

## 성과 지표

| 항목 | 수치 |
|------|------|
| AI 파이프라인 단계 | 7단계 (Input → Feedback) |
| LangGraph 에이전트 | 3종 (Curator / Validator / Storyteller) |
| 프론트엔드 페이지 | 5개 (Home / PlanCreate / Result / Guide / Feedback) |
| UI 컴포넌트 | 54+ TypeScript React 컴포넌트 |
| 배포 플랫폼 | 3개 (Vercel + Railway + Local Ollama) |
| LLM 운영 비용 | 0원 (로컬 Ollama) |
| API 엔드포인트 | 7+ (입력/출력/LLM/피드백/헬스체크) |
