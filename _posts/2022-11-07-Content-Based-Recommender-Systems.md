---
layout: post
title: "[추천시스템] CH04 - Content-Based Recommender Systems"
categories: [Study Note]
tags: [recommender-system, content-based, classification, tf-idf]
math: true
---

## Introduction

Content-Based 추천시스템은 아이템의 **속성(feature)**과 사용자의 과거 선호를 기반��로 추천한다. 협업 필터링이 사용자-아이템 상호작용 패턴만을 활용하는 반면, Content-Based 방식은 아이템 자체의 내용(텍스트, 메타데이터, 키워드 등)을 분석하여 사용자가 좋아할 만한 아이템을 예측한다.

---

## 기본 구성요소

Content-Based 시스템은 세 가지 핵심 컴포넌트로 구성된다:

```
아이템 데이터
     │
     ▼
[Content Analyzer]     원시 데이터 → 구조화된 특징 벡터
     │
     ▼
[Profile Learner]      사용자의 좋아요/싫어요 + 아이템 특징 → 사용자 프로파일
     │
     ▼
[Filtering Component]  새 아이템 특징 vs 사용자 프로파일 → 추천 점수
```

---

## 특징 추출 (Feature Extraction)

### TF-IDF

텍스트 기반 아이템(뉴스, 상품 설명 등)에서 핵심 키워드를 추출하는 표준 방식이다.

**TF(Term Frequency)**: 문서 $d$에서 단어 $t$의 출현 빈도:

$$
\text{tf}(t, d) = \frac{f_{t,d}}{\max_{t' \in d} f_{t',d}}
$$

**IDF(Inverse Document Frequency)**: 전체 문서에서의 희소성:

$$
\text{idf}(t) = \log\left(\frac{N}{n_t}\right)
$$

- $N$: 전체 문서 수, $n_t$: 단어 $t$를 포함하는 문서 수

**TF-IDF 가중치**:

$$
w(t, d) = \text{tf}(t, d) \cdot \text{idf}(t)
$$

많은 문서에 나타나는 일반적인 단어(the, is 등)는 낮은 IDF를, 특정 문서에만 나타나는 키워드는 높은 IDF를 가져 **변별력 있는 특징**이 된다.

### 특징 전처리

- **불용어 제거(Stop-word removal)**: 관사, 전치사 등 의미 없는 단어 제거
- **어근 추출(Stemming)**: running → run, movies → movi 등 어근으로 변환
- **정규화(Normalization)**: 특징 벡터를 단위 벡터로 정규화하여 문서 길이 영향 제거

---

## 특징 선택 (Feature Selection)

차원이 높은 특징 공간에서 예측력이 높은 특징만 선별한다.

### Gini Index

클래스 분포의 **불순도(impurity)**를 측정:

$$
G = 1 - \sum_{i=1}^{c} p_i^2
$$

- $p_i$: 클래스 $i$의 비율
- 값이 0에 가까울수록 순수(특정 클래스 지배), 특징의 분할 능력이 좋음

### Entropy (정보 엔트로피)

$$
H = -\sum_{i=1}^{c} p_i \log_2 p_i
$$

**Information Gain**: 특징 $f$로 분할 전후의 엔트로피 감소량. 정보 이득이 큰 특징을 우선 선택한다.

### Chi-squared ($\chi^2$) 통계량

특징과 클래스 간의 **독립성을 검정**:

$$
\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

- $O_{ij}$: 관측 빈도, $E_{ij}$: 기대 빈도
- 값이 클수록 특징과 클래스 간 연관성이 강함

### 특징 가중치 (Feature Weighting)

이진 선택(포함/제외) 대신 각 특징에 **연속적 가중치**를 부여하는 소프트한 접근도 가능하다. 유사도 계산 시 가중치를 반영하여 중요 특징의 영향력을 높인다.

---

## 사용자 프로파일 학습

사용자가 좋아요/싫어요를 표시한 아이템들의 특징 벡터를 학습 데이터로 사용하여 사용자 프로파일(선호 모델)을 구축한다.

### Nearest Neighbor (k-NN)

새 아이템의 특징 벡터와 사용자가 좋아한 아이템들 간의 **유사도**를 계산:

$$
\text{score}(u, j) = \frac{1}{k} \sum_{i \in N_k(j)} \text{sim}(\mathbf{x}_j, \mathbf{x}_i)
$$

코사인 유사도가 가장 널리 사용된다:

$$
\text{cos}(\mathbf{x}_j, \mathbf{x}_i) = \frac{\mathbf{x}_j^T \mathbf{x}_i}{\|\mathbf{x}_j\| \|\mathbf{x}_i\|}
$$

### Naive Bayes 분류기

베이즈 정리를 활용한 확률적 모델:

$$
P(c \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid c) \cdot P(c)}{P(\mathbf{x})}
$$

**조건부 독립 가정** 하에:

$$
P(\mathbf{x} \mid c) = \prod_{i=1}^{d} P(x_i \mid c)
$$

각 특징의 조건부 확률을 독립적으로 추정하여 계산 효율성을 확보한다. 텍스트 분류에서 특히 효과적이다.

### Rule-Based 분류기

"IF 장르=SF AND 평점>4 THEN 좋아요" 형태의 규칙을 학습한다. **해석 가능성**이 높다는 장점이 있지만, 복잡한 패턴 학습에는 한계가 있다.

### Regression 모델

평점을 연속 값으로 예측하는 회귀 모델:

$$
\hat{r}_{uj} = \mathbf{w}_u^T \mathbf{x}_j + b_u
$$

- $\mathbf{w}_u$: 사용자 $u$의 가중치 벡터 (사용자 프로파일)
- $\mathbf{x}_j$: 아이템 $j$의 특징 벡터

각 사용자별로 개별 회귀 모델을 학습하여 개인화된 예측을 수행한다.

---

## Content-Based의 설명 가능성

Content-Based 시스템은 추천 이유를 **아이템 속성으로 직접 설명**할 수 있다. "당신이 SF 영화를 좋아하기 때문에 이 영화를 추천합니다"와 같은 설명이 가능하여, 사용자 신뢰를 높이는 데 유리하다.

---

## Content-Based vs Collaborative Filtering

| 항목 | Content-Based | Collaborative |
|------|-------------|--------------|
| 필요 데이터 | 아이템 속성 + 개인 이력 | 다수 사용자의 평점 |
| 새 아이템 추천 | **가능** (속성만 있으면) | 불가 (평점 이력 필요) |
| 새 사용자 추천 | 어려움 (이력 부족) | 어려움 (공통 문제) |
| 추천 다양성 | 낮음 (과적합 위험) | **높음** |
| 설명 가능성 | **높음** (속성 기반) | 낮음 (잠재 요인) |
| 도메인 독립성 | 낮음 (속성 설계 필요) | **높음** |

Content-Based의 가장 큰 한계는 **과도한 특화(overspecialization)**이다. 사용자가 이미 좋아하는 유형의 아이템만 추천하여 새로운 발견(serendipity)이 어렵다.

---

## Collaborative Filtering과의 결합

Content-Based의 아이템 특징을 Collaborative Filtering의 입력으로 활용할 수 있다. 사용자 프로파일(특징 가중치 벡터)을 구축한 후, 프로파일 간 유사도를 기반으로 협업 필터링을 수행하면 **두 접근의 장점을 결합**할 수 있다.

---

## Reference

- Aggarwal, C.C. *Recommender Systems: The Textbook*. Springer, 2016. Chapter 4.
