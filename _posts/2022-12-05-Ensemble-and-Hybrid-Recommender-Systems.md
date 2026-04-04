---
layout: post
title: "[추천시스템] CH06 - Ensemble-Based and Hybrid Recommender Systems"
categories: [Study Note]
tags: [recommender-system, hybrid, ensemble, cold-start]
math: true
---

## Introduction

단일 추천 모델은 각각 고유한 약점을 가진다. Collaborative Filtering은 cold-start에 취약하고, Content-Based는 과적합되며, Knowledge-Based는 지식 획득 비용이 높다. **하이브리드 시스템**은 여러 모델을 결합하여 개별 약점을 보완하고 전체 추천 성능을 향상시킨다.

Burke의 분류에 따르면, 하이브리드 추천시스템은 **7가지 설계 아키타입**으로 구분된다.

---

## 앙상블 방법의 분류학적 배경

추천시스템의 앙상블은 기계학습의 앙상블 기법과 동일한 원리를 따른다. 핵심 아이디어는 **다수의 약한 모델(weak learner)을 결합하여 강한 모델을 구성**하는 것이다.

결합의 세 가지 패러다임:

| 패러다임 | 설명 | 예시 |
|----------|------|------|
| **병렬(Parallelized)** | 여러 모델을 독립 실행 후 결과 결합 | Weighted, Mixed |
| **순차(Pipelined)** | 한 모델의 출력이 다음 모델의 입력 | Cascade, Feature Augmentation |
| **선택(Monolithic)** | 상황에 따라 하나의 모델 선택 | Switching |

---

## 7가지 하이브리드 설계

### 1. Weighted Hybrid

여러 추천 모델의 점수를 **가중 평균**으로 결합:

$$
\hat{r}_{uj} = \sum_{i=1}^{T} \alpha_i \cdot f_i(u, j), \quad \sum_{i=1}^{T} \alpha_i = 1
$$

- $$f_i(u, j)$$: $$i$$번째 모델의 예측 점수
- $$\alpha_i$$: 모델 $$i$$의 가중치 (교차 검증으로 최적화)

가장 단순한 하이브리드 방식이며, 모든 모델이 동일 스케일의 점수를 출력해야 한다.

### Bagging (Bootstrap Aggregation)

훈련 데이터에서 **부트스트랩 샘플링**으로 $$T$$개의 서브셋을 생성하고, 각 서브셋으로 동일 알고리즘을 학습한 후 예측을 평균:

$$
\hat{r}_{uj} = \frac{1}{T} \sum_{t=1}^{T} f_t(u, j)
$$

개별 모델의 **분산(variance)을 감소**시켜 안정적인 예측을 달성한다.

### Randomness Injection

부트스트랩 대신 모델 내부에 **의도적 랜덤성을 주입**하여 다양성을 확보한다. 예를 들어, 행렬 분해에서 잠재 차원의 서브셋을 랜덤 선택하거나, 이웃 기반 방법에서 유사도 계산에 노이즈를 추가한다.

---

### 2. Switching Hybrid

상황에 따라 **하나의 모델을 선택**하여 사용:

```
사용자 요청
     │
     ├── 충분한 평점 이력 있음 → Collaborative Filtering
     │
     └── 신규 사용자 (cold-start) → Content-Based 또는 Knowledge-Based
```

#### Cold-Start 대응 전략

| 사용자 유형 | 평점 수 | 선택 모델 |
|-----------|--------|----------|
| 신규 | 0개 | Knowledge-Based (요구사항 질의) |
| 초기 | 1~10개 | Content-Based (소수 이력 활용) |
| 활성 | 10+개 | Collaborative Filtering |

#### Bucket-of-Models

여러 후보 모델을 **교차 검증으로 평가**하고, 각 사용자/상황에서 최고 성능 모델을 선택한다. Switching의 일반화로, 모델 선택 자체를 데이터 기반으로 학습한다.

---

### 3. Cascade Hybrid

한 추천 모델의 출력을 다음 모델이 **순차적으로 정제(refine)**:

$$
\text{Model}_1 \rightarrow \text{후보 집합} \rightarrow \text{Model}_2 \rightarrow \text{최종 랭킹}
$$

1단계에서 광범위한 후보를 추출하고, 2단계에서 정밀한 랭킹을 수행한다.

#### Boosting

Cascade의 특수한 경우로, 이전 모델이 **예측을 틀린 사례에 가중치를 높여** 다음 모델을 학습:

- 각 반복에서 예측 오류가 큰 (사용자, 아이템) 쌍의 가중치 증가
- 최종 예측은 모든 모델의 가중 투표:

$$
\hat{r}_{uj} = \sum_{t=1}^{T} \beta_t \cdot f_t(u, j)
$$

$$\beta_t$$는 모델 $$t$$의 정확도에 비례하여 결정된다. Bagging이 분산을 줄이는 반면, Boosting은 **편향(bias)을 줄이는** 데 효과적이다.

---

### 4. Feature Augmentation Hybrid

한 모델의 **출력을 다른 모델의 입력 특징으로 활용**:

```
[Content-Based] → 사용자 프로파일 특징 벡터
     │
     ▼
[Collaborative Filtering] + 추가 특징 → 최종 추천
```

예를 들어, Content-Based 모델이 생성한 사용자의 장르 선호 벡터를 Collaborative Filtering의 부가 정보로 사용하면, 협업 필터링의 cold-start 문제를 완화할 수 있다.

---

### 5. Meta-Level Hybrid

한 모델이 점수가 아닌 **모델 자체(user profile)**를 생성하고, 이것이 다음 모델의 입력이 된다:

$$
\text{Content-Based} \rightarrow \text{사용자 모델} \rightarrow \text{CF의 유사도 계산에 활용}
$$

Feature Augmentation과의 차이: Feature Augmentation은 **점수**를 전달하고, Meta-Level은 **학습된 모델/표현**을 전달한다.

---

### 6. Feature Combination Hybrid

여러 소스의 특징을 **단일 모델에 직접 결합**:

$$
\hat{r}_{uj} = \mathbf{w}^T [\mathbf{x}_{\text{collab}}; \mathbf{x}_{\text{content}}; \mathbf{x}_{\text{demo}}] + b
$$

협업 필터링의 잠재 요인, 아이템의 콘텐츠 특징, 사용자의 인구통계 특징을 하나의 특징 벡터로 연결(concatenate)하여 회귀 또는 행렬 분해 모델에 입력한다.

#### Side Information이 있는 행렬 분해

표준 행렬 분해를 확장하여 아이템/사용자의 부가 특징을 통합:

$$
\hat{r}_{uj} = \mu + b_u + b_j + \mathbf{u}_u^T \mathbf{v}_j + \mathbf{w}^T \mathbf{x}_{uj}
$$

여기서 $$\mathbf{x}_{uj}$$는 사용자-아이템 쌍의 부가 특징 벡터이다.

---

### 7. Mixed Hybrid

여러 모델의 추천 결과를 **동시에 나란히 제시**:

```
추천 결과 페이지
├── "당신과 비슷한 사용자가 좋아한 영화" (CF)
├── "당신이 좋아한 SF 영화와 유사한 작품" (Content-Based)
└── "이번 주 인기 신작" (Popularity-Based)
```

결합 로직 없이 각 모델이 독립적으로 추천을 제공하며, 사용자가 다양한 관점의 추천을 동시에 볼 수 있다.

---

## 7가지 하이브리드 전략 비교

| 전략 | 결합 방식 | 핵심 장점 | 복잡도 |
|------|----------|----------|--------|
| **Weighted** | 점수 가중 평균 | 구현 단순 | 낮음 |
| **Switching** | 상황별 모델 선택 | Cold-start 대응 | 중간 |
| **Cascade** | 순차 정제 | 단계별 품질 향상 | 중간 |
| **Feature Aug.** | 출력→입력 특징 | 모델 간 정보 전달 | 중간 |
| **Meta-Level** | 모델→입력 모델 | 깊은 모델 통합 | 높음 |
| **Feature Comb.** | 특징 결합 | 다양한 정보 통합 | 중간 |
| **Mixed** | 동시 나열 | 추천 다양성 | 낮음 |

---

## Reference

- Aggarwal, C.C. *Recommender Systems: The Textbook*. Springer, 2016. Chapter 6.
- Burke, R. "Hybrid Recommender Systems: Survey and Experiments." *User Modeling and User-Adapted Interaction*, 12(4), 331-370, 2002.
