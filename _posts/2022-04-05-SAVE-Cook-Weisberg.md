---
layout: post
title: "[Paper Review] Sliced Inverse Regression: Comment (SAVE의 기원)"
categories: [Paper Review]
tags: [paper-review, dimension-reduction, sufficient-dimension-reduction, statistics]
math: true
---

## Introduction

Cook & Weisberg (1991)의 이 논문은 Li (1991)의 SIR 논문에 대한 토론(Comment)이지만, 단순한 코멘트를 넘어 **SAVE(Sliced Average Variance Estimation)**를 최초로 제안한 중요한 논문이다. SIR이 1차 역조건부 모멘트 $E(\mathbf{z} \mid y)$만 활용하여 대칭 의존 구조를 놓치는 한계를 지적하고, 2차 역조건부 모멘트 $\text{Var}(\mathbf{z} \mid y)$를 활용하는 SAVE를 제안한다.

---

## SIR의 Linearity Condition 재해석

### Condition 3.1의 기하학적 의미

Cook & Weisberg는 Li의 linearity condition이 **타원형 대칭(elliptical symmetry)**의 특성화임을 보인다. 표준화된 변수 $\mathbf{z} = (\eta_1, \ldots, \eta_K)^T$와 e.d.r. 방향 $\eta$에 대해:

$$
E(\mathbf{z} \mid y) = E[E(\mathbf{z} \mid \eta^T \mathbf{z}, y) \mid y]
$$

여기서 $E(\mathbf{z} \mid \eta^T \mathbf{z}, y) = P_\eta \mathbf{z} + E(Q_\eta \mathbf{z} \mid \eta^T \mathbf{z})$로 분해할 수 있다. $P_\eta$는 $\eta$의 열공간 위로의 사영, $Q_\eta = I - P_\eta$는 직교 여공간 위로의 사영이다.

$$
E(\mathbf{z} \mid y) = E(P_\eta \mathbf{z} \mid y) + E(Q_\eta \mathbf{z} \mid \eta^T \mathbf{z}) \mid y)
$$

첫째 항은 e.d.r. 부분공간에 속하고, 둘째 항은 직교 여공간에 속한다. $E(\mathbf{z} \mid y)$가 e.d.r. 공간에 포함되려면 둘째 항이 0이어야 하며, 이는 $E(Q_\eta \mathbf{z} \mid \eta^T \mathbf{z}) = 0$과 동치이다. Eaton (1986)은 이 조건이 **구형 대칭(spherical) 분포**를 특성화함을 보였다.

### 실용적 함의

Linearity condition은 $\mathbf{x}$의 분포가 타원형이 아니면 성립하지 않는다. 따라서:
- 지시변수(indicator variable)가 포함된 설계에는 직접 적용이 어려움
- 다항식이나 교호작용항이 함수적으로 관련된 변수를 포함하면 주의 필요
- 그러나 Diaconis & Freedman (1984)의 결과에 의해, 고차원에서 저차원 사영은 근사적으로 정규분포를 따르므로, $p$가 크면 condition 3.1이 근사적으로 성립

---

## SIR의 한계와 SAVE의 동기

### SIR이 실패하는 구체적 예시

$\mathbf{z}_i = (z_1, z_2)^T \sim N_{120}(0, 1)$ i.i.d., 단일 e.d.r. 방향 $\eta^T = (1, 1)$에 대해:

$$
y = (\mu + 2^{1/2} z_1 + 2^{1/2} z_2)^2
$$

이 모형에서 $E(\mathbf{z} \mid y)$를 계산하면:

$$
E(P_\eta \mathbf{z} \mid y) = \begin{pmatrix} z_1 \\ 0 \end{pmatrix}, \quad E[E(Q_\eta \mathbf{z} \mid \eta^T \mathbf{z}) \mid y] = \begin{pmatrix} 0 \\ z_1^2/s \end{pmatrix}
$$

$\mu = 0$이면 $y$는 $\eta^T \mathbf{z}$에 대해 대칭이므로, $y$로 슬라이싱했을 때 슬라이스 평균이 0 근처에 모이고 SIR의 고유값이 비슷한 크기가 되어 방향 식별이 어려워진다. $\mu$가 커지면 대칭이 깨지면서 SIR이 잘 작동한다.

---

## SAVE (Sliced Average Variance Estimation)

### 핵심 아이디어

SIR이 $E(\mathbf{z} \mid y)$의 변동만 보는 것에 대해, SAVE는 **조건부 분산** $\text{Var}(\mathbf{z} \mid y)$의 변동을 본다. $y$에 따라 $\text{Var}(\mathbf{z} \mid y)$가 변하면, 이는 e.d.r. 공간의 정보를 담고 있다.

### SAVE 행렬

$$
M_{\text{SAVE}} = \sum_h (I - \text{Var}(\mathbf{z} \mid y \in I_h))^2
$$

여기서 합은 슬라이스 $I_h$에 대해 취한다. 이 행렬의 이론적 근거:

$$
[I - \text{Var}(\mathbf{z} \mid y)]^2 = P_\eta [I - \text{Var}(\mathbf{z} \mid y)]^2 P_\eta
$$

$\mathbf{x}$가 정규분포일 때, $\text{Var}(\mathbf{z} \mid y)$의 고유값 $w_y$는 중복도 $p - K$를 가지며, 대응하는 고유벡터가 $Q_\eta$의 열공간을 생성한다. 따라서 $I - \text{Var}(\mathbf{z} \mid y)$의 **$K$개를 제외한 나머지 고유벡터**가 e.d.r. 방향을 추정한다.

### SAVE가 대칭 의존 구조를 탐지하는 이유

$y = (\beta^T \mathbf{x})^2 + \varepsilon$인 경우, $E(\mathbf{z} \mid y) \approx 0$이라 SIR이 실패하지만, $\text{Var}(\mathbf{z} \mid y)$는 $y$에 따라 변한다. 구체적으로, $y$가 클 때 $\beta^T \mathbf{x}$의 조건부 분산은 작아지고(큰 제곱값은 좁은 범위의 $\beta^T \mathbf{x}$에서 발생), $y$가 작을 때는 커진다. 이 변동을 SAVE가 포착한다.

### SIR vs SAVE 비교 시뮬레이션

Cook & Weisberg의 Table 1 결과 ($\eta^T = (1, 1)$, $y = (\mu + 2^{1/2}z_1 + 2^{1/2}z_2)^2$):

| $\mu$ | SIR (각도) | SAVE (각도) | pHd (각도) |
|-------|----------|-----------|----------|
| 0 | 87.82° | 0.74° | 8.90° |
| 0.5 | 7.15° | 1.97° | 6.93° |
| 1 | 4.20° | 1.32° | 18.19° |
| 4 | 0.19° | 0.71° | 21.31° |
| 100 | 0.03° | 0.27° | 33.46° |

$\mu = 0$ (완전 대칭)에서 SIR은 거의 직각(87.82°)으로 실패하지만, SAVE는 0.74°로 정확하게 추정한다. $\mu$가 커지면 대칭이 깨져 SIR도 잘 작동한다.

---

## SDR 방법론 체계에서의 위치

Cook & Weisberg의 이 논문은 SDR 방법론의 두 가지 축을 확립한다:

| 정보 원천 | 방법 | 강점 | 약점 |
|---------|------|------|------|
| 1차 모멘트 $E(\mathbf{z} \mid y)$ | **SIR** | 단조 트렌드에 강함 | 대칭 구조 실패 |
| 2차 모멘트 $\text{Var}(\mathbf{z} \mid y)$ | **SAVE** | 대칭 구조 탐지 | 단조 트렌드에서 비효율 |

이후 DR (Li & Wang, 2007)이 이 두 정보를 자연스럽게 결합하고, Ensemble MAVE (Yin & Li, 2011)가 분포 가정 없이 소진적 추정을 달성한다.

---

## Reference

- Cook, R. D. & Weisberg, S. (1991). Sliced Inverse Regression for Dimension Reduction: Comment. *Journal of the American Statistical Association*, 86(414), 328-332.
- Li, K.-C. (1991). Sliced Inverse Regression for Dimension Reduction. *JASA*, 86(414), 316-327.
