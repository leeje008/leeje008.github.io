---
layout: post
title: "[Project] 여행 플래너 - AI 기반 여행 일정 자동 생성 도구"
categories: [Project]
tags: [automation, llm-agent, python, travel]
math: false
---

## 프로젝트 개요

여행 플래너 - AI 기반 여행 일정 자동 생성 도구

> GitHub: [leeje008/travel-planner](https://github.com/leeje008/travel-planner)

---

## 주요 기능

- **최적 동선 산출**: Google OR-Tools 기반 TSP 경로 최적화
- **주변 POI 탐색**: 경로 주변 카페/맛집 등 자동 탐색 및 랭킹
- **LLM Agent 패키지**: LangGraph 기반 스토리텔링 + 추천 이유 생성
- **지도/타임테이블**: Folium 지도 + 타임라인 시각화
- **피드백 수집**: 사용자 피드백으로 추천 품질 개선

---

## 기술 스택

| 분류 | 기술 | 버전 |
|------|------|------|
| Language | Python | 3.11+ |
| Framework | FastAPI | 0.133.0 |
| Server | Uvicorn / Gunicorn | 0.41.0 / 25.1.0 |
| Database | PostgreSQL 16 (PGVector) | pgvector 0.4.2 |
| ORM | SQLAlchemy (async) | 2.0.47 |
| DB Driver | asyncpg | 0.31.0 |
| Migration | Alembic | 1.18.4 |
| Cache | Redis | 7.2.0 |
| LLM | LangChain + LangGraph | 1.2.10 / 1.0.9 |
| LLM Provider | OpenAI / Anthropic | langchain-openai 1.1.10 |
| Routing | Google OR-Tools | 9.15.6755 |
| HTTP Client | httpx | 0.28.1 |
| Validation | Pydantic | 2.12.5 |
| Geo | Shapely / Polyline | 2.1.2 / 2.0.4 |
| Frontend (MVP) | Streamlit + Folium | 1.54.0 / 0.20.0 |
| Package Manager | uv | - |
| Container | Docker + docker-compose | - |
