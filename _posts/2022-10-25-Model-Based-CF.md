---
layout: post
title: "[추천시스템] CH03 - Model-Based Collaborative Filtering"
categories: [Study Note]
tags: [recommender-system, matrix-factorization, collaborative-filtering, statistics]
math: true
---

## Introduction

Neighborhood-Based CF는 직접적인 유사도 계산에 기반하지만, 대규모 데이터에서 계산 비용이 높고 희소성에 취약하다. Model-Based CF는 평점 행렬의 **잠재 구조(latent structure)**를 학습하여 이 문제를 해결한다.

---

## 행렬 분해 (Matrix Factorization)

### 기본 아이디어

평점 행렬 $$R \in \mathbb{R}^{m \times n}$$ ($$m$$: 사용자, $$n$$: 아이템)을 두 개의 저차원 행렬로 분해:

$$
R \approx U V^T
$$

- $$U \in \mathbb{R}^{m \times k}$$: 사용자 잠재 요인 행렬
- $$V \in \mathbb{R}^{n \times k}$$: 아이템 잠재 요인 행렬
- $$k \ll \min(m, n)$$: 잠재 차원 수

사용자 $$u$$의 아이템 $$j$$에 대한 예측: $$\hat{r}_{uj} = u_u^T v_j$$

### 목적 함수

관측된 평점에 대해서만 최적화:

$$
\min_{U, V} \sum_{(u,j) \in \Omega} (r_{uj} - u_u^T v_j)^2 + \lambda(\|U\|_F^2 + \|V\|_F^2)
$$

- $$\Omega$$: 관측된 (사용자, 아이템) 쌍의 집합
- $$\lambda$$: 정규화 파라미터 (과적합 방지)

---

## SVD 기반 접근

### Truncated SVD

완전한 평점 행렬에 대해 SVD를 수행하면:

$$
R = U_m \Sigma V_n^T \approx U_k \Sigma_k V_k^T
$$

상위 $$k$$개 특이값만 사용하여 저랭크 근사를 구성한다.

### 문제점

- 실제 평점 행렬은 **대부분이 결측치**이므로 직접 SVD 적용이 어려움
- 결측값을 0 또는 평균으로 대체하면 편향 발생
- 따라서 **관측값만을 이용한 최적화** 접근이 필요

---

## 최적화 방법

### SGD (Stochastic Gradient Descent)

각 관측 평점 $$(u, j, r_{uj})$$에 대해:

$$
e_{uj} = r_{uj} - u_u^T v_j
$$

$$
u_u \leftarrow u_u + \eta (e_{uj} \cdot v_j - \lambda \cdot u_u)
$$

$$
v_j \leftarrow v_j + \eta (e_{uj} \cdot u_u - \lambda \cdot v_j)
$$

- 장점: 대규모 희소 행렬에 효율적, 구현 간단
- 단점: 수렴이 느릴 수 있음, 학습률 $$\eta$$ 튜닝 필요

### ALS (Alternating Least Squares)

$$U$$와 $$V$$를 번갈아가며 고정하고, 나머지를 최적화:

**Step 1** ($$V$$ 고정, $$U$$ 업데이트): 각 사용자 $$u$$에 대해:

$$
u_u = (V_{I_u}^T V_{I_u} + \lambda I)^{-1} V_{I_u}^T r_u
$$

**Step 2** ($$U$$ 고정, $$V$$ 업데이트): 각 아이템 $$j$$에 대해:

$$
v_j = (U_{I_j}^T U_{I_j} + \lambda I)^{-1} U_{I_j}^T r_j
$$

- 장점: 각 단계가 닫힌 형태의 해를 가짐, **병렬화 용이**
- 단점: 수렴 보장이 SGD보다 강함 (이차 최적화의 블록 좌표 하강)

---

## 편향 모델 (Bias Model)

사용자와 아이템의 고유 편향을 분리하여 모델링:

$$
\hat{r}_{uj} = \mu + b_u + b_j + u_u^T v_j
$$

- $$\mu$$: 전체 평균 평점
- $$b_u$$: 사용자 $$u$$의 편향 (예: 관대한 평가자)
- $$b_j$$: 아이템 $$j$$의 편향 (예: 인기 영화)
- $$u_u^T v_j$$: 잠재 요인에 의한 상호작용

목적 함수:

$$
\min \sum_{(u,j) \in \Omega} (r_{uj} - \mu - b_u - b_j - u_u^T v_j)^2 + \lambda(b_u^2 + b_j^2 + \|u_u\|^2 + \|v_j\|^2)
$$

---

## 암시적 피드백 (Implicit Feedback)

클릭, 구매, 조회 등 명시적 평점이 없는 데이터에서의 행렬 분해.

### Weighted MF

신뢰도 $$c_{uj}$$를 정의:

$$
c_{uj} = 1 + \alpha \cdot f(r_{uj})
$$

여기서 $$f(r_{uj})$$는 상호작용 빈도에 기반한 함수. 상호작용이 없는 경우($$r_{uj} = 0$$)도 **낮은 신뢰도의 부정 피드백**으로 처리한다.

목적 함수:

$$
\min \sum_{u,j} c_{uj}(p_{uj} - u_u^T v_j)^2 + \lambda(\|U\|_F^2 + \|V\|_F^2)
$$

$$p_{uj} \in \{0, 1\}$$: 이진 선호 지표

---

## NMF (Non-negative Matrix Factorization)

$$U \geq 0$$, $$V \geq 0$$ 제약을 부과하면 잠재 요인이 **가법적(additive) 해석**을 가진다. 각 잠재 요인은 "장르", "분위기" 등의 해석 가능한 속성에 대응할 수 있다.

업데이트 규칙 (곱셈적 갱신):

$$
U \leftarrow U \odot \frac{R V}{U V^T V}, \quad V \leftarrow V \odot \frac{R^T U}{V U^T U}
$$

---

## Neighborhood vs Model-Based 비교

| 항목 | Neighborhood | Model-Based (MF) |
|------|-------------|------------------|
| 학습 | 불필요 (lazy) | 오프라인 학습 필요 |
| 예측 속도 | 느림 ($$O(k \cdot n)$$) | **빠름** ($$O(k)$$) |
| 희소성 내성 | 약함 | **강함** |
| 해석 가능성 | 직관적 | 잠재 요인 해석 필요 |
| Cold Start | 취약 | 취약 (공통 문제) |

---

## Reference

- Aggarwal, C.C. *Recommender Systems: The Textbook*. Springer, 2016. Chapter 3.
- Koren, Y., Bell, R. & Volinsky, C. "Matrix Factorization Techniques for Recommender Systems." *Computer*, 42(8), 30-37, 2009.
