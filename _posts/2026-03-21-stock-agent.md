---
layout: post
title: "[Project] 주식 포트폴리오 에이전트 - Streamlit + Claude LLM"
categories: [Project]
tags: [python, streamlit, llm-agent, finance, portfolio-optimization]
math: false
---

## 프로젝트 개요

한국/미국 주식과 글로벌 ETF를 통합 관리하고, Mean-Variance 최적화와 LLM 에이전트를 결합하여 투자 의사결정을 지원하는 개인용 포트폴리오 분석 도구이다.

> GitHub: [leeje008/stock-agent](https://github.com/leeje008/stock-agent)

---

## 동기

수학적 포트폴리오 최적화는 과거 데이터 기반의 정량 분석에 강하지만, 뉴스나 거시경제 변화 같은 정성적 요소를 반영하기 어렵다. 반대로 LLM은 뉴스 분석과 종합 판단에 뛰어나지만 수치적 최적화에는 한계가 있다. 이 프로젝트는 **두 접근의 결합**을 목표로 한다.

---

## 아키텍처

```
Streamlit Dashboard (UI)
    │
    ├── Data Layer
    │   ├── yfinance (미국 주식/ETF)
    │   ├── pykrx (한국 주식)
    │   └── FRED API (경제지표)
    │
    ├── Optimization Layer
    │   ├── PyPortfolioOpt (Mean-Variance)
    │   └── Discrete Allocation (정수 매수 수량)
    │
    ├── LLM Agent Layer
    │   ├── News Analyst Agent (감성 분석)
    │   └── Portfolio Manager Agent (종합 추천)
    │
    └── Storage
        └── SQLite (포트폴리오, 거래 기록)
```

---

## 핵심 기능

### 1. 한국/미국 통합 데이터 수집
- yfinance로 미국 주식 및 ETF 시세 수집
- pykrx로 한국 코스피/코스닥 시세 수집
- FRED API로 거시경제 지표 (기준금리, CPI, VIX 등) 수집
- 캐싱 시스템으로 API 호출 최소화

### 2. 포트폴리오 최적화
- **Max Sharpe Ratio**: 위험 대비 수익률 최대화
- **Min Volatility**: 최소 변동성 전략
- **Discrete Allocation**: 예산 제약 하에서 실제 매수 수량 계산

### 3. LLM 에이전트 (Claude API)
- **News Analyst Agent**: 뉴스 기사를 분석하여 시장 감성, 종목별 영향도, 리스크 이벤트 감지
- **Portfolio Manager Agent**: 최적화 결과 + 뉴스 분석 + 거시경제 지표를 종합하여 투자 추천 리포트 생성

### 4. 대시보드
- 포트폴리오 현황 (시장별/섹터별 비중 차트)
- 최적화 결과 시각화
- AI 종합 투자 리포트

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| UI | Streamlit |
| 데이터 수집 | yfinance, pykrx, fredapi |
| 최적화 | PyPortfolioOpt |
| LLM | Anthropic Claude API |
| 시각화 | Plotly |
| DB | SQLite |
| 패키지 관리 | uv |
