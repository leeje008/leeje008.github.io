---
layout: post
title: "[Project] 보험 상품 QA 에이전트 - LLM 기반 보험 질의응답 시스템"
categories: [Project]
tags: [llm-agent, nlp, python, qa-system, rag]
math: false
---

## 프로젝트 개요

보험 상품 QA 에이전트 - LLM 기반 보험 질의응답 시스템

> GitHub: [leeje008/insurance-qa-agent](https://github.com/leeje008/insurance-qa-agent)

---

## 주요 기능

- **약관 PDF 파싱**: 보험약관 PDF를 관/조/항/호 계층 구조로 자동 파싱
- **하이브리드 검색**: PGVector 시맨틱 검색 + 키워드 검색 + RRF 융합
- **LLM 답변 생성**: Ollama 로컬 LLM 기반 근거 조항 인용 답변
- **답변 검증**: 환각 감지 및 신뢰도 평가 (자동 재생성 루프)
- **용어 사전**: 보험 전문 용어 자동 해설

---

## 기술 스택

| 분류 | 기술 | 버전 |
|------|------|------|
| Language | Python | 3.11+ |
| Framework | FastAPI | 0.134.0 |
| Server | Uvicorn / Gunicorn | 0.41.0 / 25.1.0 |
| Database | PostgreSQL 16 (PGVector) | pgvector 0.4.2 |
| ORM | SQLAlchemy (async) | 2.0.47 |
| DB Driver | asyncpg | 0.31.0 |
| Migration | Alembic | 1.18.4 |
| LLM | LangChain + LangGraph | 1.2.10 / 1.0.10 |
| LLM Provider | Ollama (qwen2.5:14b) | langchain-ollama 1.0.1 |
| Embedding | nomic-embed-text (768 dim) | Ollama |
| PDF 파싱 | PyMuPDF + pdfplumber | 1.27.1 / 0.11.9 |
| HTTP Client | httpx | 0.28.1 |
| Validation | Pydantic | 2.12.5 |
| Frontend (MVP) | Streamlit | 1.54.0 |
| Package Manager | uv | - |
| Container | Docker + docker-compose | - |
