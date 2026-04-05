---
layout: post
title: "[Paper Review] Sufficient Dimension Reduction Based on an Ensemble of MAVE"
categories: [Paper Review]
tags: [paper-review, dimension-reduction, nonparametric, statistics]
math: true
---

## Introduction

MAVE(Minimum Average Variance Estimator, Xia et al. 2002)는 $E(Y \mid X)$의 기울기를 비모수적으로 추정하여 central mean subspace를 복원하는 강력한 방법이다. 그러나 MAVE는 **조건부 평균 부분공간**(central mean subspace) $\mathcal{S}_{E(Y \mid X)}$만을 추정하며, 조건부 분산 $\text{Var}(Y \mid X)$ 방향 등 central subspace $\mathcal{S}_{Y \mid X}$의 나머지 방향을 추정하지 못한다.

Yin & Li (2011)는 **특성화 함수족**(characterizing family) $\mathcal{F}$를 도입하여 MAVE를 반복 적용함으로써 central subspace 전체를 소진적으로 추정하는 Ensemble MAVE를 제안한다. 이 방법의 핵심 장점은 설명변수 $X$에 대한 **분포 가정이 전혀 불필요**하다는 것이다.

## Central Mean Subspace vs Central Subspace

### 정의와 관계

$\mathcal{S}_{E(Y \mid X)}$는 다음을 만족하는 최소 부분공간이다:

$$E(Y \mid X) = E(Y \mid P_{\mathcal{S}} X)$$

여기서 $P_{\mathcal{S}}$는 $\mathcal{S}_{E(Y \mid X)}$ 위로의 직교 사영이다. 항상 $\mathcal{S}_{E(Y \mid X)} \subseteq \mathcal{S}_{Y \mid X}$이 성립하지만, 일반적으로 진부분집합이다.

예를 들어, $Y = (\beta_1^T X)^2 + \beta_2^T X \cdot \varepsilon$인 모형에서 $\mathcal{S}_{Y \mid X} = \text{span}(\beta_1, \beta_2)$이지만, $E(Y \mid X) = (\beta_1^T X)^2$이므로 $\mathcal{S}_{E(Y \mid X)} = \text{span}(\beta_1)$이다. MAVE는 $\beta_2$ 방향을 놓치게 된다.

## Characterizing Family의 이론

### Definition 2.1: Characterizing Family

함수족 $\mathcal{F}$가 다음을 만족하면 central subspace를 **characterize**한다고 한다:

$$\mathcal{S}(\mathcal{F}) := \text{span}\left(\bigcup_{f \in \mathcal{F}} \mathcal{S}_{E[f(Y) \mid X]}\right) = \mathcal{S}_{Y \mid X}$$

즉, 각 $f \in \mathcal{F}$에 대한 central mean subspace $\mathcal{S}_{E[f(Y) \mid X]}$를 모두 합치면 central subspace 전체와 동일하다.

### Lemma 2.1: 포함 관계

임의의 함수족 $\mathcal{F}$에 대해 항상 $\mathcal{S}(\mathcal{F}) \subseteq \mathcal{S}_{Y \mid X}$가 성립한다. 역방향 포함의 조건은 다음과 같다: 만약 모든 $f \in \mathcal{F}$에 대해

$$\text{(1)} \quad Y \perp\!\!\!\perp X \mid B^T X \implies \text{(2)} \quad f(Y) \perp\!\!\!\perp X \mid B^T X$$

가 성립하면, $\mathcal{S}_{Y \mid X} \subseteq \mathcal{S}(\mathcal{F})$이다. 조건 (1)에서 (2)로의 함의는 $f$가 가측(measurable)이면 자동으로 성립하므로, 핵심은 $\mathcal{F}$가 충분히 풍부한지 여부이다.

### Theorem 2.1: 지시함수 기반 Characterization

$\mathcal{F}$가 Borel 집합의 지시함수족 $\mathcal{B} = \{1_A : A \in \mathcal{B}(\mathbb{R})\}$에서 $L_2(F_Y)$-조밀(dense)하면, $\mathcal{F}$는 central subspace를 characterize한다.

**증명의 핵심**: 임의의 Borel 집합 $A$에 대해 $E[1_A(Y) \mid X] = P(Y \in A \mid X)$이다. $Y \perp\!\!\!\perp X \mid B^T X$이면 모든 Borel 집합 $A$에 대해 $P(Y \in A \mid X) = P(Y \in A \mid B^T X)$이므로 $\mathcal{S}_{E[1_A(Y) \mid X]} \subseteq \text{span}(B)$이다. $\mathcal{F}$가 $L_2(F_Y)$에서 조밀하면 임의의 $1_A$를 $\mathcal{F}$의 원소로 근사할 수 있으므로 역방향 포함도 성립한다.

### 특성함수족의 Characterization

$\mathcal{F} = \{e^{itY} : t \in \mathbb{R}\}$ (특성함수족)은 central subspace를 characterize한다. 이는 특성함수의 유일성 정리에 의해 $\{e^{itY}\}$가 $L_2(F_Y)$에서 조밀하기 때문이다. 실수 부분 $\cos(tY)$와 허수 부분 $\sin(tY)$를 분리하여 사용한다.

### 다른 Characterizing Family의 예

- **Box-Cox 변환**: $\{(Y+c)^\lambda : \lambda \in \Lambda, c > 0\}$. $\Lambda$가 무한 집합이면 characterizing family이다.
- **웨이블릿 기저**: $\{\psi_{j,k}(Y) : j, k \in \mathbb{Z}\}$. 정규직교 기저(orthonormal basis)를 형성하므로 $L_2$-조밀성이 보장된다.

## Ensemble MAVE 알고리즘

### MAVE의 핵심

$E(Y \mid X = x)$를 국소 선형 근사하고, 커널 가중 최소제곱으로 추정한다:

$$\min_{a_i, b_i, B} \sum_{i=1}^{n} \sum_{j=1}^{n} \left(Y_j - a_i - b_i^T B^T(X_j - X_i)\right)^2 K_h(B^T(X_j - X_i))$$

여기서 $K_h$는 대역폭 $h$의 커널 함수이다. $B$와 $(a_i, b_i)$를 반복 최적화하며, $B$의 열공간이 central mean subspace $\mathcal{S}_{E(Y \mid X)}$를 추정한다.

### Ensemble 절차

1. Characterizing family $\mathcal{F}$에서 확률 측도 $\nu$에 따라 함수 $f_1, \ldots, f_m$을 무작위 추출
2. 각 $f_\ell$에 대해 변환된 반응변수 $f_\ell(Y_1), \ldots, f_\ell(Y_n)$을 사용하여 RMAVE로 $\hat{\mathcal{S}}_{E[f_\ell(Y) \mid X]}$를 추정. RMAVE(Refined MAVE)는 대역폭을 적응적으로 선택하여 수렴률을 향상시킨 변형이다.
3. 추정된 부분공간들의 합집합으로 central subspace를 복원:

$$\hat{\mathcal{S}}_{Y \mid X} = \text{span}\left(\bigcup_{\ell=1}^{m} \hat{\mathcal{S}}_{E[f_\ell(Y) \mid X]}\right)$$

실제 구현에서는 $m$개의 추정된 기저 행렬 $\hat{B}_1, \ldots, \hat{B}_m$을 열 방향으로 결합한 후 SVD를 수행하여, 유의미한 특이값에 대응하는 좌특이벡터들을 central subspace의 기저로 사용한다.

## 수렴 성질

### 수렴률 (Convergence Rate)

RMAVE ensemble의 경우, 추정된 투영 행렬 $\hat{P}$는:

$$\|\hat{P} - P_{\mathcal{S}}\|_F = O_p\left(n^{-2/(2+d)}\right)$$

여기서 $d = \dim(\mathcal{S}_{Y \mid X})$이다. 이는 RMAVE 자체의 수렴률과 동일하다. 즉, ensemble을 통해 central mean subspace에서 central subspace로 추정 범위를 확장하면서도 추가적인 통계적 비용이 발생하지 않는다.

이 수렴률은 SIR, SAVE, DR의 $O_p(n^{-1/2})$보다 느리지만, 이들 방법이 요구하는 선형성 조건(linearity condition)이 불필요하다는 점에서 trade-off가 있다.

### 차원 결정의 일치성

교차 검증 기준:

$$CV(d) = \sum_{k=1}^{K} \sum_{i \in I_k} \left(Y_i - \hat{g}_{-k}(\hat{B}_{-k}^T X_i)\right)^2$$

여기서 $\hat{g}_{-k}$는 $k$번째 fold를 제외하고 추정한 비모수 회귀 함수, $\hat{B}_{-k}$는 해당 fold를 제외한 MAVE 추정량이다. $d$의 추정량 $\hat{d} = \arg\min_d CV(d)$는 일치적이다:

$$P(\hat{d} = d_0) \to 1 \quad \text{as } n \to \infty$$

## SIR, SAVE, DR과의 비교

| 방법 | 추정 대상 | 예측변수 조건 | 소진성 | 수렴률 |
|------|----------|-------------|--------|--------|
| SIR | $E[X \mid Y]$ | linearity | 비소진적 (대칭 실패) | $O_p(n^{-1/2})$ |
| SAVE | $\text{Var}(X \mid Y)$ | linearity + CCV | 소진적 | $O_p(n^{-1/2})$ |
| DR | 방향적 잔차 | linearity | 소진적 | $O_p(n^{-1/2})$ |
| Ensemble MAVE | $E[f(Y) \mid X]$의 기울기 | **조건 없음** | 소진적 | $O_p(n^{-2/(2+d)})$ |

Ensemble MAVE의 가장 큰 장점은 예측변수 $X$에 대한 분포 가정(linearity condition, constant covariance condition 등)이 **불필요**하다는 것이다. 이는 $X$의 분포가 타원형(elliptical)이 아닌 경우, 예를 들어 이산형 변수가 혼합된 경우에도 적용 가능함을 의미한다.

## 실용적 고려사항

- **$m$의 선택**: 이론적으로 $m \to \infty$이면 $\mathcal{S}(\mathcal{F})$를 복원하지만, 실제로는 $m = 20 \sim 50$ 정도면 충분하다.
- **$\mathcal{F}$의 선택**: 특성함수족이 이론적으로 가장 안정적이며, 실무에서는 $t$를 표준정규분포에서 추출한다.
- **계산 비용**: 각 $f_\ell$에 대해 RMAVE를 수행하므로 $m$배의 비용이 발생하지만, 병렬화가 용이하다.

## Reference

Yin, X. & Li, B. (2011). Sufficient Dimension Reduction Based on an Ensemble of Minimum Average Variance Estimators. *Annals of Statistics*, 39(6), 3392-3416.
