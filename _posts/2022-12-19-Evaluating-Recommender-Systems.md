---
layout: post
title: "[추천시스템] CH07 - Evaluating Recommender Systems"
categories: [Study Note]
tags: [recommender-system, evaluation, metrics, statistics]
math: true
---

## Introduction

추천시스템의 성능을 어떻게 평가할 것인가는 모델 설계만큼 중요한 문제다. 잘못된 평가 방식은 실제 사용자 경험과 무관한 모델 최적화로 이어진다. 본 챕터에서는 평가 패러다임, 평가 목표, 구체적 정확도 메트릭, 그리고 평가의 한계를 다룬다.

---

## 평가 패러다임

### User Study (사용자 연구)

실제 사용자를 모집하여 통제된 환경에서 시스템을 평가한다. 만족도, 사용성, 신뢰도 등 **주관적 품질**을 직접 측정할 수 있지만, 비용이 높고 규모 확장이 어렵다.

### Online Evaluation (A/B 테스트)

프로덕션 환경에서 사용자 트래픽을 분할하여 **실시간 행동 데이터**로 평가한다. CTR(클릭률), 체류 시간, 구매 전환율 등을 직접 측정할 수 있어 가장 신뢰도 높은 평가 방식이다. 다만 시스템 변경이 실제 사용자에게 영향을 미치는 위험이 있다.

### Offline Evaluation (오프라인 평가)

과거 수집된 데이터셋을 훈련/테스트로 분할하여 평가한다. **비용이 낮고 반복 실험이 용이**하여 가장 널리 사용되지만, 실제 사용자 행동을 완전히 반영하지 못하는 한계가 있다.

| 평가 방식 | 비용 | 신뢰도 | 반복성 | 주관적 품질 |
|-----------|------|--------|--------|-----------|
| User Study | **높음** | 중간 | 낮음 | **측정 가능** |
| Online (A/B) | 중간 | **높음** | 중간 | 간접 측정 |
| Offline | **낮음** | 낮음 | **높음** | 측정 불가 |

---

## 평가의 일반적 목표

정확도만으로는 추천시스템의 품질을 완전히 평가할 수 없다. 8가지 평가 목표를 종합적으로 고려해야 한다.

### 1. Accuracy (정확도)

예측 평점 또는 랭킹이 실제 사용자 선호와 얼마나 일치하는가. 가장 기본적이고 널리 사용되는 메트릭이다.

### 2. Coverage (커버리지)

시스템이 추천할 수 있는 아이템/사용자의 **비율**. 커버리지가 낮으면 일부 아이템이 영원히 추천되지 않는 문제가 발생한다.

$$
\text{Item Coverage} = \frac{|\text{추천된 아이템 집합}|}{|\text{전체 아이템}|}
$$

### 3. Confidence and Trust (신뢰도)

시스템이 자신의 추천에 대해 얼마나 **확신**하는가. 신뢰 구간을 함께 제공하면 사용자가 추천의 불확실성을 이해할 수 있다.

### 4. Novelty (신규성)

사용자가 **이전에 알지 못했던** 아이템을 추천하는 능력. 이미 알고 있는 아이템의 추천은 정확하더라도 가치가 낮다.

### 5. Serendipity (뜻밖의 발견)

단순히 새로운 것이 아니라, **예상치 못했지만 만족스러운** 아이템을 추천하는 능력. Novelty보다 강한 개념으로, 사용자의 기존 선호 패턴을 벗어나면서도 긍정적 반응을 이끌어내야 한다.

### 6. Diversity (다양성)

추천 목록 내 아이템들 간의 **비유사성**. 유사한 아이템만 나열하면 사용자의 선택 폭이 좁아진다.

$$
\text{Diversity}(L) = \frac{\sum_{i,j \in L, i \neq j} \text{dist}(i, j)}{|L|(|L|-1)/2}
$$

### 7. Robustness (강건성)

가짜 평점 주입(shilling attack) 등의 **공격에 대한 저항력**. 소수의 악의적 사용자가 전체 추천 결과를 왜곡할 수 없어야 한다.

### 8. Scalability (확장성)

사용자/아이템 수가 증가해도 **합리적 시간 내에 추천을 생성**할 수 있는가. 실시간 서비스에서는 예측 지연 시간이 핵심 제약이 된다.

---

## 오프라인 평가 설계

### Netflix Prize 사례

Netflix Prize(2006-2009)는 오프라인 추천시스템 평가의 대표적 벤치마크다:

- **규모**: 480,189명 사용자, 17,770개 영화, 1억+ 평점
- **목표**: 기존 시스템(Cinematch) 대비 RMSE 10% 개선
- **결과**: BellKor 팀이 앙상블 기법으로 목표 달성

### 데이터 분할 전략

#### Hold-Out

전체 데이터를 훈련 세트와 테스트 세트로 **한 번 분할**. 일반적으로 80:20 비율을 사용한다. 구현이 단순하지만, 분할 방식에 따라 결과가 달라질 수 있다.

#### k-Fold Cross-Validation

데이터를 $k$개 폴드로 나누어, 각 폴드를 한 번씩 테스트 세트로 사용:

$$
\text{CV Score} = \frac{1}{k} \sum_{i=1}^{k} \text{Metric}(\text{fold}_i)
$$

Hold-Out보다 **안정적인 성능 추정**을 제공하며, $k=5$ 또는 $k=10$이 일반적이다.

---

## 정확도 메트릭

### 평점 예측 정확도

#### RMSE (Root Mean Squared Error)

$$
\text{RMSE} = \sqrt{\frac{1}{|T|} \sum_{(u,j) \in T} (r_{uj} - \hat{r}_{uj})^2}
$$

큰 오류에 **제곱 페널티**를 부과하여, 극단적 오차를 강하게 벌한다.

#### MAE (Mean Absolute Error)

$$
\text{MAE} = \frac{1}{|T|} \sum_{(u,j) \in T} |r_{uj} - \hat{r}_{uj}|
$$

모든 오류에 **동일한 페널티**를 부과한다. RMSE에 비해 이상치에 덜 민감하다.

#### RMSE vs MAE

| 특성 | RMSE | MAE |
|------|------|-----|
| 큰 오류 민감도 | **높음** (제곱 효과) | 낮음 (선형) |
| 이상치 영향 | 큼 | **작음** |
| Netflix Prize 사용 | **O** | X |
| 해석 용이성 | 중간 | **높음** |

#### Long Tail의 영향

인기 아이템은 평점이 많아 전체 메트릭에 과도한 영향을 미친다. 비인기(long-tail) 아이템의 예측 정확도가 무시될 수 있으므로, 아이템 인기도별 **세그먼트 분석**이 필요하다.

---

### 랭킹 기반 메트릭

실제 추천에서는 정확한 평점보다 **올바른 순위**가 중요하다.

#### Precision@k와 Recall@k

$$
\text{Precision@}k = \frac{|\{\text{relevant items in top-}k\}|}{k}
$$

$$
\text{Recall@}k = \frac{|\{\text{relevant items in top-}k\}|}{|\{\text{all relevant items}\}|}
$$

$$
\text{F1@}k = \frac{2 \cdot \text{Precision@}k \cdot \text{Recall@}k}{\text{Precision@}k + \text{Recall@}k}
$$

Precision은 추천의 **정밀성**을, Recall은 관련 아이템의 **재현율**을 측정한다.

#### NDCG (Normalized Discounted Cumulative Gain)

상위 순위에 더 높은 가중치를 부여하여 **위치 민감한 랭킹 품질**을 측정:

$$
\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}
$$

$$
\text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}
$$

- $rel_i$: $i$번째 위치 아이템의 관련도
- $\text{IDCG@}k$: 이상적 순서에서의 DCG (정규화 상수)

NDCG는 0~1 사이 값을 가지며, 1에 가까울수록 이상적 랭킹에 근접한다.

#### Rank Correlation (순위 상관)

예측 순위와 실제 순위 간의 상관을 측정:

**Spearman 순위 상관계수**:

$$
\rho = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}
$$

$d_i$: $i$번째 아이템의 예측 순위와 실제 순위 간의 차이

#### ROC와 AUC

추천을 이진 분류(관련/비관련)로 처리하여 **True Positive Rate vs False Positive Rate** 곡선을 그린다:

$$
\text{TPR} = \frac{TP}{TP + FN}, \quad \text{FPR} = \frac{FP}{FP + TN}
$$

**AUC(Area Under Curve)**는 ROC 곡선 아래 면적으로, 0.5(랜덤)~1.0(완벽) 범위를 가진다.

---

## 평가의 한계

### Evaluation Gaming

특정 메트릭에 과적합하면 실제 사용자 경험과 괴리가 발생한다. Netflix Prize에서도 RMSE 최적화에 집중하여 **인기 아이템 예측은 정확하지만 개인화된 추천은 부족**한 모델이 상위권을 차지하는 현상이 관찰되었다.

### 오프라인-온라인 괴리

오프라인에서 우수한 모델이 온라인 A/B 테스트에서는 열등한 결과를 보이는 경우가 빈번하다. 오프라인 평가는 **관측된 데이터에 대한 성능만 측정**하며, 추천이 사용자 행동을 변화시키는 피드백 루프(feedback loop)를 반영하지 못한다.

### 종합적 평가의 필요성

단일 메트릭에 의존하지 않고, 정확도·다양성·신규성·커버리지 등 **다차원 메트릭을 종합적으로 고려**해야 실제 서비스 품질을 올바르게 평가할 수 있다.

---

## Reference

- Aggarwal, C.C. *Recommender Systems: The Textbook*. Springer, 2016. Chapter 7.
- Herlocker, J.L., et al. "Evaluating Collaborative Filtering Recommender Systems." *ACM Transactions on Information Systems*, 22(1), 5-53, 2004.
