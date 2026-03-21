---
layout: post
title: "[Project] 보험약관 QA 에이전트 - RAG + LangGraph 파이프라인"
categories: [Project]
tags: [python, llm-agent, rag, nlp, fastapi]
math: false
---

## 프로젝트 개요

보험약관 PDF를 자동으로 파싱하고, 하이브리드 검색(시맨틱 + 키워드)과 로컬 LLM을 결합하여 근거 조항을 인용하는 질의응답 시스템이다.

> GitHub: [leeje008/insurance-qa-agent](https://github.com/leeje008/insurance-qa-agent)

---

## 동기

보험약관은 수백 페이지에 달하는 복잡한 법률 문서로, 일반 사용자가 필요한 조항을 찾기 어렵다. 단순 키워드 검색으로는 의미적 유사성을 반영하지 못하고, 일반 LLM은 약관의 정확한 조항 번호를 인용하지 못한다. 이 프로젝트는 **정확한 근거 인용이 가능한 도메인 특화 QA 시스템**을 목표로 한다.

---

## RAG 파이프라인

```
사용자 질문
    │
    ▼
Query Processor (질문 분석/재구성)
    │
    ▼
Hybrid Retriever
    ├── PGVector 시맨틱 검색 (nomic-embed-text)
    ├── 키워드 검색
    └── RRF (Reciprocal Rank Fusion)
    │
    ▼
Answer Generator (Ollama qwen2.5:14b)
    │
    ▼
Answer Validator (환각 감지 + 신뢰도 평가)
    │
    ▼
최종 답변 (근거 조항 인용 포함)
```

---

## 핵심 기능

### 1. 약관 PDF 자동 파싱
- PyMuPDF + pdfplumber 이중 파싱
- **관 → 조 → 항 → 호** 계층 구조 자동 인식
- 상호 참조 관계 추출 (예: "제5조 참조")

### 2. 하이브리드 검색
- PGVector 기반 시맨틱 검색 (nomic-embed-text 768차원)
- TF-IDF 키워드 검색
- RRF(Reciprocal Rank Fusion)로 두 결과 융합

### 3. LangGraph 4-노드 에이전트
- LangGraph 기반 상태 머신으로 답변 생성 파이프라인 구성
- 답변 검증 노드에서 환각 감지 시 자동 재생성

### 4. 보험 용어 사전
- 전문 용어 자동 감지 및 해설 제공

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| Backend | FastAPI, Uvicorn |
| DB | PostgreSQL 16 + PGVector |
| ORM | SQLAlchemy (async) + Alembic |
| LLM | LangChain + LangGraph, Ollama (qwen2.5:14b) |
| Embedding | nomic-embed-text (768dim) |
| PDF 파싱 | PyMuPDF, pdfplumber |
| Frontend | Streamlit |
| Infra | Docker + docker-compose |
| 패키지 관리 | uv |
