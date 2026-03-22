---
layout: post
title: "[Paper Review] On Directional Regression for Dimension Reduction"
categories: [Paper Review]
tags: [paper-review, dimension-reduction, sufficient-dimension-reduction, statistics]
math: true
---

## Introduction

Li & Wang (2007)의 "On Directional Regression for Dimension Reduction" (JASA, 102(479))은 충분차원축소(Sufficient Dimension Reduction, SDR) 분야에서 SIR과 SAVE의 장점을 통합하는 새로운 방법론인 Directional Regression(DR)을 제안한다. 기존의 SIR(Sliced Inverse Regression)은 1차 역조건부 모멘트 $E(X \mid Y)$에 기반하여 대칭적 의존 구조를 탐지하지 못하며, SAVE(Sliced Average Variance Estimation)는 2차 모멘트 $\text{Var}(X \mid Y)$를 활용하지만 수렴 속도가 느리다. DR은 이 두 방법의 정보를 동시에 활용하면서도 계산적으로 효율적인 $\sqrt{n}$-일치 추정량을 제공한다.

## Central Subspace와 Dimension Reduction Subspace

$Y \in \mathbb{R}$를 반응변수, $X \in \mathbb{R}^p$를 설명변수, $\mu = E(X)$, $\Sigma = \text{Cov}(X)$라 하자. 차원축소부분공간(Dimension Reduction Subspace, DRS)은 $p \times d$ 행렬 $\eta$의 열공간으로서 다음을 만족하는 부분공간이다:

$$Y \perp\!\!\!\perp X \mid \eta^T X$$

모든 DRS의 교집합이 다시 DRS가 될 때, 이를 **중심부분공간**(central subspace) $\mathcal{S}_{Y \mid X}$라 정의하며, 그 차원 $d = \dim(\mathcal{S}_{Y \mid X})$를 구조적 차원(structural dimension)이라 한다.

표준화된 변수 $Z = \Sigma^{-1/2}(X - \mu)$를 도입하면, $\mathcal{S}_{Y \mid X}$는 $\Sigma^{-1/2}\mathcal{S}_{Y \mid X}$로 변환되며, 이후의 분석은 $Z$ 기반으로 진행된다.

## SIR과 SAVE의 커널 행렬

DR의 동기를 이해하기 위해 기존 방법들의 커널 행렬을 복습한다.

**SIR**: $M_{\text{SIR}} = \text{Cov}[E(Z \mid Y)]$. 반응 곡면 $f(\beta^T X)$가 $\beta^T X$에 대해 대칭이면 $E[Z \mid Y]$가 상수가 되어 해당 방향을 감지하지 못한다.

**SAVE**: $M_{\text{SAVE}} = E[(I_p - \text{Cov}(Z \mid Y))^2]$. 대칭 곡면도 감지하지만, 단조 트렌드에서 $\text{Cov}(Z \mid Y)$의 변동이 미약하여 소표본에서 비효율적이다.

## Directional Regression Matrix

### DR 행렬의 정의

DR의 핵심 아이디어는 독립적인 두 관측치 $(Y, Z)$와 $(\tilde{Y}, \tilde{Z})$의 **방향적 잔차**를 활용하는 것이다. 다음을 정의한다:

$$A(Y, \tilde{Y}) = E\left[(Z - \tilde{Z})(Z - \tilde{Z})^T \mid Y, \tilde{Y}\right]$$

이는 $Y$와 $\tilde{Y}$가 주어졌을 때 $Z - \tilde{Z}$의 조건부 이차 모멘트 행렬이다. 만약 $Z$가 $Y$와 독립이면:

$$E[A(Y, \tilde{Y})] = E[(Z - \tilde{Z})(Z - \tilde{Z})^T] = 2I_p$$

**DR 행렬** $G$는 다음과 같이 정의된다:

$$G = E\left[\left(2I_p - A(Y, \tilde{Y})\right)^2\right]$$

이 행렬은 $Y$와 $X$ 간의 의존성이 없을 때 영행렬이 되며, 의존성의 방향을 담고 있는 고유벡터를 통해 중심부분공간을 추정한다.

### Theorem 1: DR의 일치성

**선형성 조건**(linearity condition)하에서, 임의의 $b \in \mathbb{R}^p$에 대해 $E(b^T Z \mid \eta^T Z)$가 $\eta^T Z$의 선형함수이면:

$$\text{span}(G) \subseteq \mathcal{S}_{Y \mid X}$$

즉, DR 행렬의 열공간은 중심부분공간에 포함된다. 증명의 핵심은 $b \perp \mathcal{S}_{Y \mid X}$인 $b$에 대해 $b^T G b = 0$임을 보이는 것이다. 선형성 조건 하에서 $b^T Z \perp\!\!\!\perp Y \mid \eta^T Z$이므로 $b^T(2I_p - A(Y, \tilde{Y}))b = 0$ a.s.가 성립한다.

### Theorem 2: SIR + SAVE 분해

DR 행렬의 가장 중요한 이론적 결과는 다음의 분해이다:

$$G = 2E\left[E^2(ZZ^T - I_p \mid Y)\right] + 2\left[E\left(E(Z \mid Y)E(Z^T \mid Y)\right)\right]^2 + 2E\left[E(Z^T \mid Y)E(Z \mid Y)\right] \cdot E\left[E(Z \mid Y)E(Z^T \mid Y)\right]$$

이 분해를 구성요소별로 분석하면:

- **첫 번째 항**: $2E[E^2(ZZ^T - I_p \mid Y)]$. 여기서 $E(ZZ^T - I_p \mid Y) = E(ZZ^T \mid Y) - I_p = \text{Var}(Z \mid Y) + E(Z \mid Y)E(Z^T \mid Y) - I_p$이므로, 이 항은 **SAVE 커널**의 정보를 포함한다. 조건부 분산 $\text{Var}(Z \mid Y)$의 변동을 포착하여 대칭적 의존 구조를 탐지할 수 있다.

- **두 번째 항**: $2[E(E(Z \mid Y)E(Z^T \mid Y))]^2$. 이는 $E(Z \mid Y)$의 이차 모멘트 행렬의 제곱으로, **SIR 커널**의 정보에 해당한다. 조건부 기대값의 변동을 포착하여 단조적 의존 구조를 탐지한다.

- **세 번째 항**: $2E[E(Z^T \mid Y)E(Z \mid Y)] \cdot E[E(Z \mid Y)E(Z^T \mid Y)]$. 이는 SIR의 4차 모멘트 정보를 담고 있는 **교차 항**으로, $E(Z \mid Y)$의 norm 정보를 활용한다.

이 분해를 통해 DR이 1차 역조건부 모멘트($E(Z \mid Y)$, SIR)와 2차 역조건부 모멘트($\text{Var}(Z \mid Y)$, SAVE) 정보를 **동시에** 활용함을 엄밀히 확인할 수 있다.

### Theorem 3: 소진성(Exhaustiveness)

DR은 매우 약한 비퇴화(non-degeneracy) 조건 하에서 소진적(exhaustive)이다:

$$\text{span}(G) = \mathcal{S}_{Y \mid X}$$

소진성 조건은 다음과 같다: $\eta^T Z$의 분포가 비퇴화적이면, 즉 특정한 병적인 대칭 구조를 갖지 않으면, DR은 중심부분공간 전체를 복원한다. 구체적으로, $\eta^T Z$의 결합 분포가 유한개의 점에 집중되지 않는 한 소진성이 성립한다. 이는 SIR(대칭 의존 구조 탐지 불가)이나 SAVE(특정 3차 이상 모멘트 구조 탐지 불가)보다 훨씬 완화된 조건이다.

### Theorem 4: DR과 SAVE의 관계

유한 모멘트 조건 하에서:

$$\mathcal{S}_{\text{DR}} = \mathcal{S}_{\text{SAVE}}$$

이는 DR이 SAVE와 동일한 부분공간을 복원하면서도, 추정의 효율성에서 이점을 가짐을 의미한다. DR은 SAVE의 모든 정보를 포함하면서 추가적인 SIR 정보를 활용하여, 유한 표본에서 더 안정적인 추정을 제공한다.

## 추정 절차

### 표본 DR 행렬의 구성

$Y$를 $H$개의 슬라이스 $\{J_1, \ldots, J_H\}$로 나눈다. 각 슬라이스 $h$에 속하는 표본 수를 $n_h$, 비율을 $\hat{p}_h = n_h / n$으로 표기한다. 각 슬라이스에 대해 다음을 계산한다:

$$\hat{M}_h = \frac{1}{n_h} \sum_{i \in J_h} \hat{Z}_i \hat{Z}_i^T, \quad \hat{\mu}_h = \frac{1}{n_h} \sum_{i \in J_h} \hat{Z}_i$$

여기서 $\hat{Z}_i = \hat{\Sigma}^{-1/2}(X_i - \bar{X})$이다. 표본 DR 행렬은:

$$\hat{G} = \frac{2}{H^2} \sum_{h,k=1}^{H} \hat{p}_h \hat{p}_k \left(I_p + \hat{M}_h \hat{M}_k + \hat{\mu}_h \hat{\mu}_k^T \hat{\mu}_k \hat{\mu}_h^T - 2\hat{M}_h - 2\hat{M}_k\right)^2$$

### 고유값 분해와 방향 추정

행렬 $\hat{F} = \hat{G} / (2H^2)$에 대해 고유값 분해를 수행한다:

$$\hat{F} = \sum_{j=1}^{p} \hat{\lambda}_j \hat{v}_j \hat{v}_j^T$$

가장 큰 $d$개의 고유값 $\hat{\lambda}_1 \geq \cdots \geq \hat{\lambda}_d$에 대응하는 고유벡터 $\hat{v}_1, \ldots, \hat{v}_d$가 표준화 공간에서의 중심부분공간 기저를 추정한다. 원래 공간에서의 기저는 $\hat{\Sigma}^{-1/2}\hat{v}_j$로 복원된다.

### 계산 복잡도

DR의 계산 복잡도는 $O(n)$으로, Li, Zha & Chiaromonte (2005)의 Contour Regression이 요구하는 $O(n^2)$에 비해 효율적이다. 이는 DR이 쌍별 비교 대신 슬라이스 기반 통계량을 사용하기 때문이다.

## 구조적 차원 결정: 순차 검정

구조적 차원 $d$를 결정하기 위해 다음의 순차 검정을 사용한다:

$$H_0: \text{rank}(\hat{F}_1) = \ell \quad \text{vs} \quad H_1: \text{rank}(\hat{F}_1) > \ell$$

검정통계량은:

$$T_\ell = n \sum_{j=\ell+1}^{p} \hat{\lambda}_j$$

여기서 $\hat{\lambda}_j$는 $\hat{F}_1$의 $j$번째 고유값이다. $H_0$ 하에서 $T_\ell$은 가중 카이제곱 분포를 따른다:

$$T_\ell \xrightarrow{d} \sum_{j} c_j \chi^2_{1,j}$$

가중치 $c_j$는 잔여 고유값의 점근 분포에서 도출되며, 이를 통해 p-value를 계산한다.

$\ell = 0, 1, 2, \ldots$에 대해 순차적으로 검정하여, 처음으로 $H_0$이 기각되지 않는 $\ell$을 구조적 차원 $\hat{d}$로 결정한다. 이 추정량은 $\sqrt{n}$-일치성을 가진다:

$$d(\hat{\mathcal{S}}_{\text{DR}}, \mathcal{S}_{Y \mid X}) = O_p(n^{-1/2})$$

## 시뮬레이션 결과 요약

논문의 시뮬레이션에서 DR은 다음과 같은 성능을 보인다:
- SIR이 실패하는 대칭 모형(예: $Y = (\beta^T X)^2 + \varepsilon$)에서도 올바른 방향을 추정
- SAVE와 동등한 소진성을 가지면서, 유한 표본에서 더 안정적인 추정
- 구조적 차원 결정에서 순차 검정이 높은 검정력을 보임
- SIR+SAVE를 각각 수행한 후 결합하는 것보다 DR 단독 적용이 더 효율적

## 결론

Directional Regression은 SIR과 SAVE를 통합하는 SDR 방법론으로, 선형성 조건이라는 약한 가정 하에서 $\sqrt{n}$-일치적이고 소진적인 추정을 제공한다. DR 행렬의 SIR+SAVE 분해(Theorem 2)는 이 방법이 1차와 2차 역조건부 모멘트 정보를 모두 활용함을 이론적으로 보여주며, 계산 복잡도 $O(n)$으로 대규모 데이터에도 실용적이다.

## Reference

Li, B. & Wang, S. (2007). On Directional Regression for Dimension Reduction. *Journal of the American Statistical Association*, 102(479), 997-1008.
