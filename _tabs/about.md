---
title: About
icon: fas fa-info-circle
order: 4
description: >

hide_description: true
redirect_from:
  - /download/
---

<div class="about-hero">
  <img src="{{ site.avatar | relative_url }}" class="about-hero__avatar" alt="profile" />
  <div class="about-hero__body">
    <div class="eyebrow">About</div>
    <h1>{{ site.social.name }} <span class="latin">고영헌</span></h1>
    <p class="about-hero__tagline">{{ site.tagline }}</p>
    <div class="about-hero__actions">
      <a class="btn-primary" href="mailto:{{ site.social.email }}">{{ site.social.email }}</a>
      <a class="btn-secondary" href="https://github.com/{{ site.github.username }}">github.com/{{ site.github.username }}</a>
    </div>
  </div>
  <div class="about-stats">
    <div class="stat"><div class="stat__num">{{ site.posts.size }}</div><div class="stat__label">Posts</div></div>
    <div class="stat"><div class="stat__num">{{ site.categories.size }}</div><div class="stat__label">Categories</div></div>
    <div class="stat"><div class="stat__num">{{ site.tags.size }}</div><div class="stat__label">Tags</div></div>
  </div>
</div>

## 경력

- **에이브랩스** (2025.01 ~ 현재) — LLM 에이전트 개발
  - 관리회계 자동화: SQL 쿼리 + Apache Airflow 기반 손익분석 테이블 생성 자동화 (기존 5시간 → 20분으로 단축)
  - MDAI: LangGraph 기반 Multi-Agent 관리회계 분석 시스템 설계·구현. 9개 계열사 대상 자연어 질의 기반 매출·원가·손익 분석 에이전트 개발
  - MDAI 운영 및 유지 보수 — 배포 후 안정화 및 이슈 대응, 기능 개선
  - Power BI 대시보드 운영 및 유지 보수 — 관리회계 및 KPI 대시보드 운영 관리, 데이터/리포트 유지 보수
- **KB라이프생명** (2024.08 ~ 2024.09) — 인턴, 상품전략
  - 위험률 정비 및 감리 규정 체크, 시장조사

## 학력

- **고려대학교 일반대학원** 통계학 석사 (2021.09 ~ 2024.02)
  - 전공: 통계적 데이터과학
  - 학위 논문: *Penalized Neural Network Sufficient Dimension Reduction*
  - 연구 분야: Sufficient Dimension Reduction, Neural Network, Sparse Modeling
- **중앙대학교** 응용통계학 학사 (2014.03 ~ 2021.02)

## 기술 스택

- **언어**: Python, SQL
- **ML/DL**: PyTorch, scikit-learn, NumPy, Pandas
- **LLM & Agent**: LangChain, LangGraph, Anthropic Claude API, Ollama, RAG
- **백엔드**: FastAPI, PostgreSQL, Alembic
- **대시보드**: Streamlit
- **인프라**: Docker, Apache Airflow, Git

## 관심 분야

- LLM Agent & RAG 시스템
- 데이터 파이프라인 자동화
- Sufficient Dimension Reduction
- Sparse Modeling & Variable Selection
- Reinforcement Learning

## 교육 경험

- 고려대학교 조교 — 회귀분석 / 통계소프트웨어 (2021 가을)
- 고려대학교 조교 — 딥러닝을 위한 통계학 (2022 봄)
- 고려대학교 조교 — 통계분석방법론 (2022 가을)

## 연락처

- **GitHub**: [github.com/leeje008](https://github.com/leeje008)
- **이메일**: leeje008@naver.com
