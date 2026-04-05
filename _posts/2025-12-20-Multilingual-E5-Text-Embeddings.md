---
layout: post
title: "[Paper Review] Multilingual E5 Text Embeddings: A Technical Report"
categories: [Paper Review]
tags: [paper-review, nlp, text-embeddings, multilingual]
math: true
---

## Introduction

Wang et al. (2024)의 mE5(Multilingual E5)는 100개 이상의 언어를 지원하는 범용 텍스트 임베딩 모델이다. 정보 검색(IR), 검색 증강 생성(RAG), 의미적 텍스트 유사도(STS) 등 다양한 다운스트림 태스크에서 하나의 임베딩 모델을 공유할 수 있다는 점에서 실용적 가치가 크다. mE5는 **2단계 학습 파이프라인**(대조 학습 사전학습 + 지도 미세조정)을 통해 학습되며, 명령어 기반 변형(instruct variant)은 GPT-4 합성 데이터를 활용하여 태스크 적응 능력을 강화한다.

---

## 모델 아키텍처

mE5는 XLM-RoBERTa를 백본으로 사용하며, 세 가지 크기로 제공된다:

| 모델 | 파라미터 수 | 임베딩 차원 |
|------|-----------|-----------|
| mE5-small | 118M | 384 |
| mE5-base | 278M | 768 |
| mE5-large | 560M | 1024 |

추가로 **mE5-large-instruct** 변형이 존재하며, 이는 입력 앞에 태스크 설명 명령어(instruction)를 접두사로 부여하여 검색, 분류, 클러스터링 등 태스크에 맞는 임베딩을 생성한다.

---

## 2단계 학습 파이프라인

### Stage 1: 대조 학습 사전학습

약 **10억 개의 텍스트 쌍**을 활용한 대규모 대조 학습을 수행한다. 데이터 소스는 웹 크롤링, 병렬 코퍼스, QA 페어 등으로 구성된다. 손실 함수는 InfoNCE를 사용한다:

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(q, d^+) / \tau)}{\exp(\text{sim}(q, d^+) / \tau) + \sum_{j=1}^{N} \exp(\text{sim}(q, d_j^-) / \tau)}
$$

여기서 $\text{sim}(q, d) = \frac{q^T d}{\|q\| \|d\|}$는 코사인 유사도, $\tau$는 온도 파라미터, $d^+$는 양성 문서, $d_j^-$는 음성 문서이다. 배치 내 다른 샘플들을 **인-배치 네거티브(in-batch negatives)**로 활용하여 효율적으로 대비 학습을 수행한다.

### Stage 2: 지도 미세조정

약 **160만 개의 레이블 데이터**로 미세조정한다. NLI, STS, 검색 데이터셋 등에서 수집한 고품질 레이블을 활용하며, 하드 네거티브 마이닝을 통해 학습 난이도를 높인다.

### Instruct 변형: GPT-4 합성 데이터

mE5-large-instruct는 GPT-4를 활용하여 다양한 태스크에 대한 합성 학습 데이터를 생성한다. 각 데이터 포인트에 태스크 설명 명령어가 포함되어 있으며, 이를 통해 모델이 명령어에 따라 임베딩 공간을 조건부로 조정하는 능력을 학습한다.

---

## 평가 벤치마크와 결과

### MTEB (Massive Text Embedding Benchmark)

MTEB는 분류, 클러스터링, 쌍별 분류, 재순위화, 검색, STS, 요약의 7개 태스크 유형을 포괄하는 대규모 벤치마크이다.

| 모델 | MTEB 평균 |
|------|----------|
| Cohere multilingual-v3 | 64.0 |
| mE5-large | 61.5 |
| **mE5-large-instruct** | **64.4** |

### MIRACL (다국어 검색)

MIRACL은 18개 언어에 대한 다국어 정보 검색 벤치마크로, nDCG@10을 주요 지표로 사용한다.

| 모델 | MIRACL nDCG@10 (평균) |
|------|---------------------|
| mE5-base | 58.3 |
| mE5-large | **66.5** |
| mE5-large-instruct | 65.7 |

한국어(ko) 성능의 경우, mE5-base의 nDCG@10은 62.2이며 mE5-large에서 66.5로 향상된다. 이는 모델 크기 증가에 따른 다국어 검색 성능의 개선을 보여준다.

### Bitext Mining

병렬 문장 쌍을 검색하는 bitext mining 태스크에서 mE5-large는 BUCC에서 99.0, Tatoeba에서 83.8의 정확도를 달성하여, 다국어 문장 정렬 능력이 우수함을 입증한다.

---

## 분석과 시사점

mE5의 핵심 기여는 다음과 같다. 첫째, 2단계 학습 파이프라인이 **언어 간 전이(cross-lingual transfer)**에 효과적임을 보인다. 영어 중심 사전학습 데이터에도 불구하고, 한국어 등 비영어 언어에서도 강건한 검색 성능을 달성한다. 둘째, instruct 변형은 명령어를 통해 하나의 모델로 다양한 태스크를 처리할 수 있으므로, 실제 배포 환경에서의 효율성이 높다.

한계로는 mE5-large-instruct가 MIRACL에서 mE5-large보다 오히려 낮은 성능을 보이는 점이 있다. 이는 명령어 기반 학습이 특정 검색 태스크에서는 과적합(overfitting)을 유발할 수 있음을 시사한다.

---

## Reference

- Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., & Wei, F. (2024). Multilingual E5 Text Embeddings: A Technical Report. *arXiv preprint arXiv:2402.05672*.
