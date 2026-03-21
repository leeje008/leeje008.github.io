---
layout: post
title: "[Project] AI Travel Planner - 경로 최적화 + POI 추천"
categories: [Project]
tags: [python, llm-agent, fastapi, optimization]
math: false
---

## 프로젝트 개요

Google OR-Tools 기반 경로 최적화(TSP)와 LLM 에이전트를 결합하여 최적 여행 동선을 산출하고, 주변 POI(카페/맛집 등)를 자동 추천하는 여행 일정 플래너이다.

> GitHub: [leeje008/travel-planner](https://github.com/leeje008/travel-planner)

---

## 동기

여행 일정을 짤 때 방문지 간 최적 동선과 주변 맛집/카페 탐색은 별도로 이루어지는 경우가 많다. 이 프로젝트는 **경로 최적화와 POI 추천을 하나의 파이프라인**으로 통합하고, LLM이 스토리텔링과 추천 이유를 자연어로 생성하여 사용자 경험을 극대화한다.

---

## 7-Stage Pipeline

```
Stage 1: 사용자 입력 수신 + 파싱/정규화
    │
Stage 2: 최적 경로 생성 (Google OR-Tools TSP)
    │
Stage 3: 주변 POI 탐색 + 랭킹
    │
Stage 4: LLM 패키지 구성 (스토리텔링 + 추천 이유)
    │
Stage 5: 최종 결과 조회 (지도 + 타임테이블)
    │
Stage 6: 사용자 피드백 수신
    │
Stage 7: 추천 품질 개선 반영
```

---

## 핵심 기능

### 1. 최적 동선 산출
- Google OR-Tools 기반 TSP(Traveling Salesman Problem) 경로 최적화
- 방문지 간 이동 시간/거리를 고려한 최적 순서 도출

### 2. 주변 POI 탐색
- 경로 주변 카페/맛집/관광지 자동 탐색
- 리뷰/평점 기반 랭킹 시스템

### 3. LLM Agent 패키지
- LangGraph 기반 멀티 에이전트 파이프라인
- 각 방문지에 대한 스토리텔링 및 추천 이유 생성

### 4. 시각화
- Folium 기반 인터랙티브 지도
- 타임라인 시각화로 일정 한눈에 파악

### 5. 피드백 루프
- 사용자 피드백 수집 및 추천 품질 개선

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| Backend | FastAPI |
| DB | PostgreSQL 16 + PGVector |
| ORM | SQLAlchemy (async) |
| LLM | LangChain + LangGraph |
| 경로 최적화 | Google OR-Tools |
| 캐싱 | Redis |
| Frontend | Streamlit + Folium |
| Infra | Docker |
