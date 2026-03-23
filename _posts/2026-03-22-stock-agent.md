---
layout: post
title: "[Project] 주식 포트폴리오 에이전트 - Streamlit + Claude LLM 기반 투자 분석 및 최적화 도구"
categories: [Project]
tags: [anthropic-claude, finance, llm-agent, portfolio-optimization, python, streamlit]
math: false
---

## 프로젝트 개요

주식 포트폴리오 에이전트 - Streamlit + Claude LLM 기반 투자 분석 및 최적화 도구

> GitHub: [leeje008/stock-agent](https://github.com/leeje008/stock-agent)

---

## 기술 스택

| 구분 | 기술 | 용도 |
|------|------|------|
| **UI** | Streamlit | 대시보드, 사용자 입력, 차트 시각화 |
| **언어** | Python 3.11+ | 전체 백엔드 |
| **LLM** | Ollama (로컬) | 뉴스 분석, 리포트 생성, 토론 — **과금 없음** |
| **LLM SDK** | OpenAI Python SDK | Ollama OpenAI 호환 API 연동 |
| **최적화** | PyPortfolioOpt | Mean-Variance, Black-Litterman, 이산 배분 |
| **차트** | Plotly | 인터랙티브 차트 (서브플롯, 파이차트, 라인차트) |
| **주가 (미국)** | yfinance | NYSE/NASDAQ/ETF 시세, 재무제표 |
| **주가 (한국)** | pykrx | 코스피/코스닥 시세 |
| **뉴스** | feedparser | Google News RSS 파싱 |
| **경제지표** | fredapi | FRED 거시경제 데이터 |
| **DB** | SQLite | 포트폴리오, 리포트, 스냅샷, 감성 이력 |
| **패키지 관리** | uv | 의존성 관리 & 가상환경 |

---
