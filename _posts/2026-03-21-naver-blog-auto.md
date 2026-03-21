---
layout: post
title: "[Project] 네이버 블로그 자동 작성 도구 - Streamlit + Ollama"
categories: [Project]
tags: [python, streamlit, ollama, llm, automation]
math: false
---

## 프로젝트 개요

블루오션 키워드 발굴부터 이미지 기반 초안 생성, SEO 최적화, 네이버 블로그 자동 업로드까지 원스톱으로 처리하는 1인 사용자용 블로그 자동화 도구이다.

> GitHub: [leeje008/naver-blog-auto](https://github.com/leeje008/naver-blog-auto)

---

## 동기

블로그 운영에서 가장 시간이 많이 드는 작업은 키워드 리서치와 글 작성이다. 특히 네이버 블로그는 D.I.A.+ 알고리즘에 의해 상위 노출이 결정되므로, SEO 최적화가 필수적이다. 이 프로젝트는 **로컬 LLM을 활용하여 API 비용 없이** 키워드 발굴부터 발행까지 자동화한다.

---

## 주요 기능

### 1. 블루오션 키워드 추천
- 네이버 자동완성 크롤링으로 관련 키워드 수집
- LLM 기반 롱테일 키워드 확장
- 경쟁도 분석을 통한 블루오션 키워드 선별

### 2. 이미지 기반 초안 생성
- 사용자가 업로드한 이미지를 분석하여 설명 반영
- LLM이 이미지 컨텍스트를 포함한 블로그 글 자동 생성

### 3. SEO 최적화
- 네이버 D.I.A.+ 알고리즘 기반 6개 항목 점수 분석
- 점수가 낮은 항목을 LLM이 자동으로 개선

### 4. 레퍼런스 톤 & 매너 유지
- 기존 블로그 글을 크롤링하여 문체 분석
- 일관된 톤으로 새 글 생성

### 5. 원클릭 업로드
- XML-RPC MetaWeblog API를 통한 네이버 블로그 자동 발행

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| UI | Streamlit (멀티페이지) |
| LLM | Ollama (로컬) |
| 크롤링 | BeautifulSoup + lxml |
| 업로드 | XML-RPC MetaWeblog API |
| 이미지 처리 | Pillow |
| 패키지 관리 | uv |
