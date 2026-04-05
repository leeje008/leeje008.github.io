---
layout: post
title: "[Paper Review] KLUE: Korean Language Understanding Evaluation"
categories: [Paper Review]
tags: [paper-review, nlp, korean-nlp, benchmark]
math: true
---

## Introduction

Park et al. (2021)의 KLUE(Korean Language Understanding Evaluation)는 한국어 자연어 이해(NLU) 모델을 체계적으로 평가하기 위한 벤치마크이다. 영어 중심의 GLUE/SuperGLUE에 대응하는 한국어 벤치마크의 부재는 한국어 NLP 연구의 객관적 비교를 어렵게 만들었다. KLUE는 8개의 NLU 태스크를 포괄하며, 한국어 고유의 언어적 특성(교착어 형태론, 경어법, 띄어쓰기 모호성)을 반영한 데이터 구축 방법론을 제시한다.

---

## 한국어의 언어적 도전 과제

한국어 NLU에서 두 가지 주요 도전 과제가 있다.

**교착어 형태론(Agglutinative Morphology).** 한국어는 어근에 다수의 접사가 결합하여 문법적 기능을 표현한다. 예를 들어, "먹었었겠다"는 하나의 어절이지만 어근+과거+대과거+추측+종결어미로 분석된다. 이로 인해 어휘의 다양성이 매우 높아 서브워드 토크나이저의 설계가 중요하다.

**경어법(Honorifics)과 띄어쓰기.** 한국어의 경어 체계는 동일한 의미를 다양한 표면형으로 표현하게 하며, 띄어쓰기 규칙의 모호성은 토큰화 단계에서 노이즈를 유발한다.

---

## 8개 NLU 태스크

KLUE는 다음 8개 태스크로 구성된다:

| # | 태스크 | 데이터셋 | 평가 지표 |
|---|--------|---------|----------|
| 1 | TC (Topic Classification) | YNAT | Macro F1 |
| 2 | STS (Semantic Textual Similarity) | KLUE-STS | Pearson's $r$ / F1 |
| 3 | NLI (Natural Language Inference) | KLUE-NLI | Accuracy |
| 4 | NER (Named Entity Recognition) | KLUE-NER | Entity-level F1 / Char-level F1 |
| 5 | RE (Relation Extraction) | KLUE-RE | Micro F1 / AUPRC |
| 6 | DP (Dependency Parsing) | KLUE-DP | UAS / LAS |
| 7 | MRC (Machine Reading Comprehension) | KLUE-MRC | EM / ROUGE-W |
| 8 | DST (Dialogue State Tracking) | WoS | JGA / Slot F1 |

### TC: YNAT

연합뉴스 기사 제목을 7개 주제(정치, 경제, 사회, 생활문화, 세계, IT/과학, 스포츠)로 분류하는 태스크이다. 총 50,000개 학습 샘플과 9,107개 평가 샘플로 구성된다.

### STS: 의미적 텍스트 유사도

두 문장 간의 의미적 유사도를 0.0~5.0 스케일로 평가한다. 네이버 스마트스토어 리뷰, 에어비앤비 리뷰 등 다양한 도메인에서 수집한 문장 쌍을 활용한다.

### NLI: 자연 언어 추론

전제(premise)와 가설(hypothesis) 간의 관계를 수반(entailment), 중립(neutral), 모순(contradiction)으로 분류한다.

### MRC: 기계 독해

SQuAD 형식의 추출형 질의응답과 더불어, 답이 존재하지 않는 경우를 판별하는 **unanswerable question** 유형도 포함한다.

### DST: 대화 상태 추적

WoS(Wizard of Seoul) 데이터셋을 활용하며, 5개 도메인(관광, 식당, 숙소, 지하철, 택시)에서 다중 도메인 대화 상태를 추적한다.

---

## 베이스라인 모델

KLUE는 4개의 베이스라인 모델을 제공한다:

| 모델 | 사전학습 데이터 | 특징 |
|------|--------------|------|
| mBERT | 104개 언어 Wikipedia | 다국어 공유 어휘 |
| XLM-R | 100개 언어 CommonCrawl | 대규모 다국어 코퍼스 |
| KLUE-BERT | 한국어 모두의 말뭉치 등 | 한국어 전용 어휘 |
| KLUE-RoBERTa | 한국어 대규모 코퍼스 | 동적 마스킹, NSP 제거 |

KLUE-RoBERTa는 62GB의 한국어 텍스트(모두의 말뭉치, 위키, 뉴스, 댓글 등)로 사전학습되었다. 형태소 분석 기반 토크나이저와 BPE 토크나이저를 비교한 결과, **BPE가 대부분의 태스크에서 형태소 분석 기반보다 우수**한 성능을 보였다.

---

## 주요 실험 결과

KLUE-RoBERTa-large가 대부분의 태스크에서 최고 성능을 달성한다:

| 태스크 | mBERT | XLM-R-large | KLUE-RoBERTa-large |
|--------|-------|-------------|-------------------|
| TC (F1) | 81.5 | 86.2 | **87.0** |
| NLI (Acc) | 73.5 | 83.5 | **85.8** |
| NER (F1) | 72.6 | 82.4 | **86.4** |
| RE (F1) | 58.2 | 66.5 | **71.1** |
| MRC (EM) | 42.3 | 58.1 | **62.3** |
| DST (JGA) | 32.1 | 41.2 | **44.7** |

한국어 전용 모델(KLUE-RoBERTa)이 다국어 모델(mBERT, XLM-R)을 일관되게 상회하며, 이는 한국어 전용 사전학습의 중요성을 보여준다.

---

## 데이터 구축 방법론

KLUE의 데이터 품질 관리는 다음 절차를 따른다: (1) 전문 어노테이터 채용 및 교육, (2) 파일럿 어노테이션으로 가이드라인 반복 개선, (3) 다중 어노테이터 간 일치도(inter-annotator agreement) 측정, (4) 전문가 검수(adjudication) 단계. NER 태스크의 경우 어노테이터 간 일치도(Cohen's $\kappa$)가 0.91로 높은 품질을 달성하였다.

---

## Reference

- Park, S., Moon, J., Kim, S., Cho, W. I., Han, J., Park, J., ... & Cho, K. (2021). KLUE: Korean Language Understanding Evaluation. *Proceedings of the Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*.
