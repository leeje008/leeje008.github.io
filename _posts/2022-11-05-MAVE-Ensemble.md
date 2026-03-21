---
layout: post
title: "[Paper Review] Ensemble MAVE - Central Subspace의 소진적 추정"
categories: [Paper Review]
tags: [paper-review, dimension-reduction, nonparametric, statistics]
math: true
---

## Introduction

MAVE(Minimum Average Variance Estimator, Xia et al. 2002)는 $E(Y \mid X)$의 기울기를 비모수적으로 추정하여 central mean subspace를 복원한다. 그러나 MAVE는 조건부 분산 $\text{Var}(Y \mid X)$ 방향 등 central mean subspace 바깥의 방향을 추정하지 못한다.

Yin & Li (2011)는 **특성 함수(characterizing family)** $\mathcal{F}$를 도입하여 MAVE를 반복 적용함으로써 central subspace 전체를 소진적으로 추정하는 Ensemble MAVE를 제안한다.

---

## 이론적 핵심: Characterizing Family

### 정의

함수족 $\mathcal{F}$가 다음을 만족하면 central subspace를 **characterize**한다고 한다:

$$
\text{span}\left(\bigcup_{f \in \mathcal{F}} \mathcal{S}_{E[f(Y)|X]}\right) = \mathcal{S}_{Y|X}
$$

즉, 각 $f \in \mathcal{F}$에 대한 central mean subspace $\mathcal{S}_{E[f(Y)|X]}$를 모두 합치면 central subspace와 동일하다.

### 핵심 예시

$\mathcal{F} = \{e^{\iota t Y} : t \in \mathbb{R}\}$ (특성 함수족)는 central subspace를 characterize한다. 이는 Zhu & Zeng (2006)의 결과에 기반한다.

다른 선택지: Box-Cox 변환, 웨이블릿 기저 등.

---

## Ensemble MAVE 알고리즘

### 절차

1. $\mathcal{F}$에서 확률 측도에 따라 함수 $f_1, \ldots, f_m$을 무작위 추출
2. 각 $f_\ell$에 대해 MAVE (또는 OPG, RMAVE)로 $\hat{\mathcal{S}}_{E[f_\ell(Y)|X]}$를 추정
3. 추정된 부분공간들을 합쳐서 central subspace를 복원:

$$
\hat{\mathcal{S}}_{Y|X} = \text{span}\left(\bigcup_{\ell=1}^{m} \hat{\mathcal{S}}_{E[f_\ell(Y)|X]}\right)
$$

### MAVE의 핵심

$E(Y \mid X = x)$를 국소 선형 근사하고, 커널 가중 최소제곱으로 추정:

$$
\min_{a_i, b_i} \sum_{i=1}^{n} \sum_{j=1}^{n} \left(Y_j - a_i - b_i^T B^T(X_j - X_i)\right)^2 K_h(B^T(X_j - X_i))
$$

$B$와 $(a_i, b_i)$를 반복 최적화하며, $B$의 열공간이 central mean subspace를 추정한다.

---

## 수렴 성질

### 정리 (수렴률)

RMAVE ensemble의 경우, 추정된 투영 행렬 $\hat{P}$는:

$$
\|\hat{P} - P_{\mathcal{S}}\|_F = O_p\left(n^{-2/(2+d)}\right)
$$

이는 RMAVE 자체의 수렴률과 동일하다. 즉, ensemble을 통한 추가적 비용 없이 central subspace 전체를 추정할 수 있다.

### 정리 (차원 결정의 일치성)

교차 검증 기준:

$$
CV(d) = \sum_{k=1}^{K} \sum_{i \in I_k} \left(Y_i - \hat{g}_{-k}(\hat{B}_{-k}^T X_i)\right)^2
$$

$d$의 추정량 $\hat{d} = \arg\min_d CV(d)$는 일치적이다: $P(\hat{d} = d_0) \to 1$.

---

## SIR, SAVE, DR과의 비교

| 방법 | 추정 대상 | 예측변수 조건 | 소진성 |
|------|----------|-------------|--------|
| SIR | $E[X \mid Y]$ | linearity | 비소진적 (대칭 실패) |
| SAVE | $\text{Var}(X \mid Y)$ | linearity + CCV | 소진적 |
| DR | 경험적 방향 | linearity + CCV | 소진적 |
| Ensemble MAVE | $E[f(Y) \mid X]$의 기울기 | **조건 없음** | 소진적 |

Ensemble MAVE의 가장 큰 장점은 예측변수 $X$에 대한 분포 가정(linearity condition 등)이 **불필요**하다는 것이다.

---

## Reference

- Yin, X. & Li, B. "Sufficient Dimension Reduction Based on an Ensemble of Minimum Average Variance Estimators." *Annals of Statistics*, 39(6), 3392-3416, 2011.
