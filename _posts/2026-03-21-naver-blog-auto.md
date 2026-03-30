---
layout: post
title: "[Project] 네이버 블로그 자동 글 작성 시스템 - Streamlit + Ollama 기반 콘텐츠 자동화"
categories: [Project]
tags: [automation, llm, ollama, python, streamlit]
math: false
---

## 프로젝트 개요

네이버 블로그 자동 글 작성 시스템 - Streamlit + Ollama 기반 콘텐츠 자동화

> GitHub: [leeje008/naver-blog-auto](https://github.com/leeje008/naver-blog-auto)

---

## 주요 기능

- **블루오션 키워드 추천** — 네이버 자동완성 + LLM 롱테일 키워드 확장 + 경쟁도 분석
- **이미지 기반 초안 생성** — 업로드한 이미지 설명을 반영한 LLM 블로그 글 생성
- **SEO 최적화** — 네이버 D.I.A.+ 알고리즘 기반 6개 항목 점수 분석 + 자동 개선
- **레퍼런스 톤 & 매너 유지** — 기존 블로그 글 크롤링으로 문체 일관성 유지
- **네이버 블로그 업로드** — XML-RPC MetaWeblog API를 통한 원클릭 발행

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| UI | Streamlit (멀티페이지) |
| LLM | Ollama (로컬) |
| 크롤링 | BeautifulSoup + lxml |
| 업로드 | XML-RPC MetaWeblog API |
| 이미지 | Pillow |
| 패키지 관리 | uv |
