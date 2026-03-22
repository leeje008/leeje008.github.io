---
layout: post
title: "[Project] 주식 포트폴리오 에이전트 - 로컬 LLM 기반 투자 분석 도구"
categories: [Project]
tags: [python, streamlit, llm-agent, portfolio-optimization, finance]
math: false
---

## 프로젝트 개요

한국/미국/글로벌 ETF를 통합 관리하는 AI 기반 포트폴리오 분석 도구. **100% 로컬 실행**으로 클라우드 API 비용 없이 LLM 분석, 포트폴리오 최적화, 뉴스 감성 분석, 기술적 분석, 백테스팅, AI 토론까지 제공한다.

> GitHub: [leeje008/stock-agent](https://github.com/leeje008/stock-agent)

---

## 핵심 문제와 해결 접근

### 문제 1: 다중 시장 통합 관리의 어려움

한국 주식(KRX)과 미국 주식(NYSE/NASDAQ) 및 글로벌 ETF를 동시에 보유할 경우, 환율 변환, 서로 다른 거래일, 데이터 소스 분리 등의 문제가 발생한다.

**해결**: 시장 자동 감지 시스템을 구현하여 종목 코드 패턴에 따라 한국 주식은 pykrx, 미국/ETF는 yfinance로 자동 라우팅한다. 모든 자산은 실시간 USD/KRW 환율로 원화 기준 통합 평가되며, 거래일 차이는 forward fill로 처리한다.

### 문제 2: 유료 LLM API 의존성

Claude, GPT 등 클라우드 LLM API는 호출당 비용이 발생하여 개인 투자자가 자유롭게 분석하기 어렵다.

**해결**: Ollama 기반 100% 로컬 LLM으로 전환. 작업 특성에 따른 **듀얼 모델 라우팅** 전략을 적용하여 효율과 품질을 동시에 확보한다:

| 작업 유형 | 모델 | 크기 | 소요 시간 | 선택 이유 |
|---------|------|------|---------|---------|
| 뉴스 감성 분석 (JSON) | llama3.1:8b | 4.9GB | ~2초 | 빠른 구조화 출력 |
| Black-Litterman 뷰 생성 | llama3.1:8b | 4.9GB | ~2초 | 수치 추출 |
| 종합 투자 리포트 | qwen3.5:27b | 17GB | ~15초 | 다중 데이터 종합 |
| 재무제표 분석 | qwen3.5:27b | 17GB | ~15초 | 재무 해석 |
| Bull vs Bear 토론 | qwen3.5:27b | 17GB | ~45초 | 복잡한 논증 |

**설계 원칙**: JSON 분류/추출은 경량 모델(빠르게), 서술적 종합 분석은 대형 모델(정확하게).

### 문제 3: 수학적 최적화와 시장 감성의 괴리

Mean-Variance 최적화는 과거 수익률과 공분산만 사용하므로, 현재 시장 상황이나 뉴스 이벤트를 반영하지 못한다.

**해결**: Black-Litterman 모델에 AI 뉴스 감성 분석 결과를 "뷰(views)"로 주입한다. LLM이 뉴스를 분석하여 종목별 기대수익률 전망과 신뢰도를 생성하면, 이를 Black-Litterman 모델의 omega 행렬에 반영하여 시장 균형 비중을 조정한다.

---

## 시스템 아키텍처

```
Streamlit 대시보드 (7개 탭)
    │
    ├── 포트폴리오 현황 ─── Data Layer ─── yfinance / pykrx / FRED / FX
    ├── 포트폴리오 최적화 ── Optimizer ──── Max Sharpe / Min Vol / Black-Litterman
    ├── 뉴스 & 시장 분석 ── AI Agent ───── News Analyzer (감성 분석)
    ├── 매수 가이드 ──────── AI Agent ───── Market Analyst (종합 리포트)
    ├── 기술적 분석 ──────── Analysis ───── RSI / MACD / Bollinger / MA
    ├── 백테스팅 ──────────── Analysis ───── Walk-Forward 3전략 비교
    └── AI 토론 ──────────── AI Agent ───── Bull vs Bear 3라운드 토론
    │
    └── SQLite (6 테이블) + JSON Cache (6시간 만료)
```

---

## 주요 기능

### 1. 포트폴리오 통합 현황
- KRX/NYSE/NASDAQ/ETF 통합 포트폴리오
- 실시간 시세 조회 + USD/KRW 자동 환산
- 시장별/섹터별 비중 파이차트
- 90일 성과 추적 (일별 스냅샷)

### 2. 포트폴리오 최적화
- **Max Sharpe**: 위험 대비 수익 극대화
- **Min Volatility**: 보수적 투자자용 최소 변동성
- **Black-Litterman**: AI 뉴스 감성을 뷰로 반영한 최적화
- **이산 배분(Discrete Allocation)**: 예산 내 실제 매수 가능 수량 계산
- 효율적 프론티어 시각화

### 3. AI 뉴스 감성 분석
- Google News RSS 기반 실제 뉴스 수집 (API 키 불필요)
- LLM 감성 분석: bullish / neutral / bearish + 신뢰도 점수
- 30일 감성 추이 차트
- 거시경제 대시보드 (미국 기준금리, CPI, 실업률, VIX, 10년물 금리, 달러 인덱스)

### 4. AI 종합 리포트
- 시장 환경 + 포트폴리오 진단 + 최적화 추천 + 매수 가이드 + 리스크 요인
- 마크다운 리포트 다운로드
- 리포트 이력 관리 (SQLite)

### 5. 기술적 분석
- RSI(14), MACD, 볼린저 밴드, 이동평균선 (5/20/60/120일)
- 복합 기술적 시그널 (매수/매도/중립)
- 3패널 인터랙티브 차트 (가격+BB / RSI / MACD)
- AI 재무제표 분석 (Valuation, Profitability, Health, Growth)

### 6. 백테스팅
- 3전략 비교 (Max Sharpe vs Min Vol vs 동일 비중)
- Walk-Forward 방법론 (과적합 방지)
- 성과 지표: 총수익률, 연환산 수익률, 변동성, 샤프 비율, MDD

### 7. AI Bull vs Bear 토론
- 3라운드 구조화 토론
  - Round 1: 강세 애널리스트 (낙관적 시나리오)
  - Round 2: 약세 애널리스트 (비관적 시나리오 + 반론)
  - Round 3: 중재자 종합 (최종 판정 + 신뢰도)
- 에코 챔버 방지를 위한 다관점 분석

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| UI | Streamlit |
| LLM | Ollama (llama3.1:8b + qwen3.5:27b) |
| 최적화 | PyPortfolioOpt (Mean-Variance, Black-Litterman) |
| 시각화 | Plotly |
| 한국 시세 | pykrx |
| 미국 시세 | yfinance |
| 거시경제 | FRED API (fredapi) |
| 뉴스 | feedparser (Google News RSS) |
| DB | SQLite |
| 패키지 관리 | uv |

---

## 데이터베이스 스키마

| 테이블 | 용도 |
|--------|------|
| `portfolio_holdings` | 현재 보유 종목 (종목코드, 시장, 수량, 평균가, 섹터) |
| `transactions` | 거래 이력 (매수/매도, 수량, 가격) |
| `optimization_history` | 최적화 결과 이력 (전략, 비중, 수익률, 변동성, 샤프) |
| `analysis_reports` | AI 분석 리포트 (유형, 내용, 메타데이터) |
| `portfolio_snapshots` | 일별 포트폴리오 스냅샷 (날짜, 총평가액, 보유 내역) |
| `sentiment_history` | 뉴스 감성 분석 이력 (날짜, 감성, 점수, 요약) |

---

## 스마트 캐싱 전략

시세 데이터, 뉴스, 환율 등 외부 API 호출을 최소화하기 위해 6시간 만료 JSON 캐시를 적용한다. `data/cache/` 디렉토리에 파일 기반으로 저장되며, 캐시 키는 종목코드+기간 조합이다.

---

## 증권사 CSV 연동

한국 증권사(신한, KB 등)의 거래내역 CSV를 업로드하면 자동으로 파싱하여 포트폴리오에 반영한다. 증권사별 CSV 포맷 차이를 `broker/csv_parser.py`에서 처리하고, `broker/aggregator.py`로 복수 증권사 데이터를 통합한다.

---

## 비용 정책

| 항목 | 비용 |
|------|------|
| LLM (Ollama) | 무료 (로컬 실행) |
| 시세 (yfinance, pykrx) | 무료 |
| 뉴스 (Google News RSS) | 무료 |
| 경제지표 (FRED) | 무료 |
| DB (SQLite) | 무료 |

---

## 면책 조항

이 도구는 **개인 참고용**이며, AI가 생성한 분석은 투자 자문(financial advice)이 아닙니다. 투자 결정의 책임은 사용자에게 있습니다.
