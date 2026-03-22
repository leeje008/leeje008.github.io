---
layout: post
title: "[Paper Review] Dimension Reduction for Conditional Mean in Regression"
categories: [Paper Review]
tags: [paper-review, dimension-reduction, sufficient-dimension-reduction, central-mean-subspace, statistics]
math: true
---

## Introduction

Cook & Li (2002)는 기존의 central subspace $\mathcal{S}_{Y \mid X}$가 $Y \mid \mathbf{X}$의 **전체 조건부 분포**를 다루는 데 비해, 많은 회귀 문제에서 관심 대상은 **조건부 평균** $E(Y \mid \mathbf{X})$뿐이라는 관찰에서 출발한다. 이를 위해 **Central Mean Subspace (CMS)** $\mathcal{S}_{E(Y \mid X)}$를 도입하고, 기존 SDR 방법(OLS, pHd, SIR, SAVE)이 실제로 어떤 부분공간을 추정하는지를 CMS 관점에서 체계적으로 분석한다.

---

## Central Mean Subspace의 정의

### Definition 1: Mean Dimension-Reduction Subspace

$p \times q$ 행렬 $\boldsymbol{\alpha}$의 열공간 $\mathcal{S}(\boldsymbol{\alpha})$가 다음을 만족하면 **mean dimension-reduction subspace**라 한다:

$$
Y \perp\!\!\!\perp E(Y \mid \mathbf{X}) \mid \boldsymbol{\alpha}^T \mathbf{X}
$$

이는 $E(Y \mid \mathbf{X}) = E(Y \mid \boldsymbol{\alpha}^T \mathbf{X})$와 동치이다. 즉, $\boldsymbol{\alpha}^T \mathbf{X}$만으로 조건부 평균의 모든 정보를 포착한다.

### Proposition 1: 동치 조건

다음 세 조건은 동치이다:
1. $Y \perp\!\!\!\perp E(Y \mid \mathbf{X}) \mid \boldsymbol{\alpha}^T \mathbf{X}$
2. $\text{Cov}(Y, E(Y \mid \mathbf{X})) \mid \boldsymbol{\alpha}^T \mathbf{X}] = 0$
3. $E(Y \mid \mathbf{X})$는 $\boldsymbol{\alpha}^T \mathbf{X}$의 함수

### Definition 2: Central Mean Subspace

$$
\mathcal{S}_{E(Y \mid X)} = \bigcap_m \mathcal{S}_m
$$

모든 mean dimension-reduction subspace의 교집합이 다시 mean dimension-reduction subspace이면, 이를 **CMS**라 정의한다. $\mathbf{X}$의 정의역이 열린 볼록 집합이면 CMS의 존재성과 유일성이 보장된다.

### Central Subspace와의 관계

항상 $\mathcal{S}_{E(Y \mid X)} \subseteq \mathcal{S}_{Y \mid X}$가 성립한다. 등호가 성립하는 경우: 위치 회귀(location regression) $Y \perp\!\!\!\perp \mathbf{X} \mid E(Y \mid \mathbf{X})$에서는 $\mathcal{S}_{E(Y \mid X)} = \mathcal{S}_{Y \mid X}$이다.

Central subspace와 달리 CMS는 $Y$의 일대일 변환에 대해 **불변이 아니다**: $\mathcal{S}_{E(T(Y) \mid X)} \neq \mathcal{S}_{E(Y \mid X)}$ in general. 다만, central subspace는 항상 상한(upper bound): $\mathcal{S}_{E(T(Y) \mid X)} \subseteq \mathcal{S}_{Y \mid X}$.

---

## 기존 SDR 방법의 CMS 관점 재분류

### 표준화 변수

$\mathbf{Z} = \Sigma_{xx}^{-1/2}(\mathbf{X} - E(\mathbf{X}))$를 표준화 변수로 사용하면, $\mathcal{S}_{E(Y \mid X)} = \Sigma_{xx}^{-1/2} \mathcal{S}_{E(Y \mid Z)}$이다.

### Theorem 1: OLS와 지수족 목적함수

$\gamma$를 $\mathcal{S}_{E(Y \mid Z)}$의 기저 행렬이라 하자. $E(\mathbf{Z} \mid \gamma^T \mathbf{Z})$가 $\mathbf{Z}$의 선형함수이고, 목적함수가 자연 지수족:

$$
L(a + \mathbf{b}^T \mathbf{Z}, Y) = -Y(a + \mathbf{b}^T \mathbf{Z}) + \phi(a + \mathbf{b}^T \mathbf{Z})
$$

이면, $(\boldsymbol{\alpha}, \boldsymbol{\beta}) = \arg\min_{a, \mathbf{b}} E[L(a + \mathbf{b}^T \mathbf{Z}, Y)]$의 $\boldsymbol{\beta}$는 $\mathcal{S}_{E(Y \mid Z)}$에 속한다.

OLS는 $\phi(K) = K^2/2$에 대응한다. 따라서 OLS 계수 $\boldsymbol{\beta}_{yz} = E(Y\mathbf{Z})$는 항상 CMS의 벡터이다. 이는 OLS가 central subspace가 아닌 **CMS를 추정**함을 의미한다.

### SIR과 SAVE의 위치

SIR과 SAVE가 추정하는 벡터는 $\mathcal{S}_{Y \mid Z}$에 속하지만, 일반적으로 $\mathcal{S}_{E(Y \mid Z)}$에는 속하지 않는다. $\mathcal{S}_{Y \mid Z}$를 $\eta$로 생성한다 하면:

$$
E(\mathbf{Z} \mid Y) = E[E(\mathbf{Z} \mid \eta^T \mathbf{Z}, Y) \mid Y] = E[E(\mathbf{Z} \mid \eta^T \mathbf{Z}) \mid Y] = P_\eta E(\mathbf{Z} \mid Y)
$$

따라서 $E(\mathbf{Z} \mid Y) \in \mathcal{S}_{Y \mid Z}$이지만, $\eta$를 CMS의 기저 $\gamma$로 교체하면 두 번째 등호가 성립하지 않을 수 있다. 이는 조건부 독립 $Y \perp\!\!\!\perp \mathbf{X} \mid \gamma^T \mathbf{X}$가 아닌 $Y \perp\!\!\!\perp E(Y \mid \mathbf{X}) \mid \gamma^T \mathbf{X}$만 성립하기 때문이다.

### Theorem 2: y-based pHd

$\gamma$를 $\mathcal{S}_{E(Y \mid Z)}$의 기저, $E(\mathbf{Z} \mid \gamma^T \mathbf{Z})$가 선형, $\text{Var}(\mathbf{Z} \mid \boldsymbol{\beta}_{yz}^T \mathbf{Z})$가 $Y$와 비상관이면:

$$
\mathcal{S}(\boldsymbol{\beta}_{yz}, \Sigma_{yzz}) \subseteq \mathcal{S}_{E(Y \mid Z)}
$$

여기서 $\Sigma_{yzz} = E\{(Y - E(Y))\mathbf{Z}\mathbf{Z}^T\}$는 pHd의 3차 모멘트 행렬이다. pHd는 central subspace가 아닌 **CMS를 직접 추정**하는 방법이다.

---

## CMS만을 요구하는 새로운 추정량 (Section 4)

### Population Structure

기존 방법들은 linearity condition **C.1** ($E(\mathbf{Z} \mid \gamma^T \mathbf{Z})$이 선형)과 constant covariance condition **C.2** ($\text{Var}(\mathbf{Z} \mid \gamma^T \mathbf{Z})$가 $Y$와 비상관)를 필요로 한다. Cook & Li는 **C.1만 요구**하는 새로운 추정량 클래스를 제안한다.

### Proposition 2: 잔차 기반 추정

$E(\mathbf{Z} \mid \gamma^T \mathbf{Z})$가 선형이면:

$$
\mathcal{S}_{E(Y \mid Z)} = \mathcal{S}_{E(r \mid Z)} + \mathcal{S}(\boldsymbol{\beta}_{yz})
$$

여기서 $r = Y - E(Y) - \boldsymbol{\beta}_{yz}^T \mathbf{Z}$는 모집단 OLS 잔차이다. 즉, CMS는 OLS 방향 $\boldsymbol{\beta}_{yz}$와 잔차의 CMS $\mathcal{S}_{E(r \mid Z)}$의 합으로 분해된다.

$\mathcal{S}_{E(r \mid Z)}$는 잔차 공분산 행렬 $\Sigma_{rzz} = E(r\mathbf{Z}\mathbf{Z}^T)$의 열공간으로 추정할 수 있으며, 이는 C.2 조건이 불필요하다. 구체적으로:

$$
\Sigma_{rzz} = \Sigma_{yzz} - P_{\boldsymbol{\beta}_{yz}} E(\boldsymbol{\beta}_{yz}^T \mathbf{Z})^3 / \|\boldsymbol{\beta}_{yz}\|^2
$$

---

## 방법론 비교 요약

| 방법 | 추정 대상 | 필요 조건 | $\dim(\mathcal{S}_{E(Y \mid X)}) = 1$일 때 |
|------|---------|---------|-----------|
| OLS ($\boldsymbol{\beta}_{yz}$) | CMS의 벡터 | C.1 | 충분 |
| SIR | $\mathcal{S}_{Y \mid X}$의 벡터 | C.1 | CMS 벡터일 수도, 아닐 수도 |
| SAVE | $\mathcal{S}_{Y \mid X}$의 벡터 | C.1 + C.2 | CMS 벡터일 수도, 아닐 수도 |
| pHd ($\Sigma_{yzz}$) | CMS의 벡터 | C.1 + C.2 | CMS에 직접 기여 |
| **새 추정량** ($\boldsymbol{\beta}_{yz}, \Sigma_{rzz}$) | **CMS** | **C.1만** | 충분 |

---

## 의의

Cook & Li (2002)는 SDR 방법론을 CMS 관점에서 재분류함으로써:

1. OLS와 pHd가 central subspace가 아닌 **CMS를 추정**한다는 사실을 밝힘
2. SIR, SAVE는 central subspace를 추정하므로, 조건부 평균만 관심 있을 때 **과도한 추정**일 수 있음을 지적
3. C.2 조건이 불필요한 새로운 CMS 추정량을 제안하여, pHd의 적용 범위를 확장
4. 이후 Ensemble MAVE (Yin & Li, 2011)에서 CMS와 central subspace의 관계가 characterizing family를 통해 연결됨

---

## Reference

- Cook, R. D. & Li, B. (2002). Dimension Reduction for Conditional Mean in Regression. *The Annals of Statistics*, 30(2), 455-474.
