---
layout: post
title: "[Paper Review] Directional Regression for Dimension Reduction"
categories: [Paper Review]
tags: [paper-review, dimension-reduction, statistics]
math: true
---

## Introduction

SIR은 반응 곡면이 대칭이면 실패하고, SAVE는 단조 트렌드 추정에서 효율이 낮다. Li & Wang (2007)은 **경험적 방향(empirical directions)** $\{X_i - X_j\}$를 반응변수에 직접 회귀하는 Directional Regression(DR)을 제안하여, 두 방법의 장점을 자연스럽게 결합한다.

---

## SIR과 SAVE의 커널 행렬

표준화 변수 $Z = \Sigma^{-1/2}(X - \mu)$에 대해:

**SIR**: $M_{SIR} = \text{Cov}[E(Z \mid Y)]$

반응 곡면 $f(\beta^T X)$가 $\beta^T X$에 대해 대칭이면 $E[Z \mid Y]$가 상수가 되어 해당 방향을 감지하지 못한다.

**SAVE**: $M_{SAVE} = E[(I_p - \text{Cov}(Z \mid Y))^2]$

대칭 곡면도 감지하지만, 단조 트렌드에서 $\text{Cov}(Z \mid Y)$의 변동이 미약하여 소표본에서 비효율적이다.

---

## Directional Regression

### DR 커널 행렬

$(Z_1, Y_1)$, $(Z_2, Y_2)$를 $(Z, Y)$의 독립 복사본이라 하고, $\eta_{12} = Z_1^T Z_2$로 정의하면:

$$
M_{DR} = 2E\left[(ZZ^T - E[(1 + \eta_{12}^2)ZZ^T \mid Y])^2\right]
$$

### SIR + SAVE의 자연스러운 결합

$M_{DR}$은 다음과 같이 분해된다:

$$
M_{DR} = 2(M_1 + M_2 + 2M_3)
$$

- $M_1 = E[(I_p - \text{Cov}(Z \mid Y))^2]$ — SAVE 커널
- $M_2 = E[E(Z \mid Y)E(Z \mid Y)^T E(Z \mid Y)E(Z \mid Y)^T]$ — SIR 4차 모멘트
- $M_3 = E[E(Z \mid Y)E(Z \mid Y)^T(I_p - \text{Cov}(Z \mid Y))]$ — 교차 항

DR이 **1차와 2차 역조건부 모멘트 정보를 동시에 활용**함을 보여준다.

### 소진적 추정과 $\sqrt{n}$-일치성

**정리**: Linearity condition과 constant covariance condition 하에서, $M_{DR}$의 열공간은 central space $\mathcal{S}_{Y|X}$를 소진적으로 추정한다.

$$
d(\hat{\mathcal{S}}_{DR}, \mathcal{S}_{Y|X}) = O_p(n^{-1/2})
$$

### 구조적 차원의 순차적 검정

$M_{DR}$의 고유값 $\hat{\lambda}_1 \geq \cdots \geq \hat{\lambda}_p$에 대해:

$$
\Lambda_k = n \sum_{i=k+1}^{p} \hat{\lambda}_i
$$

귀무가설 $H_0: d = k$ 하에서 $\Lambda_k$는 가중 카이제곱 분포를 따른다. $k = 0, 1, 2, \ldots$ 순서로 검정하여 처음 기각하지 못하는 $k$가 추정된 차원이다.

---

## Reference

- Li, B. & Wang, S. "On Directional Regression for Dimension Reduction." *JASA*, 102(479), 997-1008, 2007.
