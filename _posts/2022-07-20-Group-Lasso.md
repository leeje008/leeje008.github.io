---
layout: post
title: "[Paper Review] Group Lasso - 그룹 변수 선택을 위한 정규화"
categories: [Paper Review]
tags: [paper-review, variable-selection, regularization, statistics]
math: true
---

## Introduction

Lasso(Tibshirani, 1996)는 $L_1$ 페널티를 통해 개별 변수를 선택하는 강력한 방법이지만, 설명변수가 **그룹 구조**를 가질 때는 적절하지 않다. 다요인 ANOVA에서 각 요인은 여러 개의 더미변수로 표현되고, 비모수 가법 모형에서 각 변수의 효과는 기저함수의 그룹으로 표현된다. 이런 상황에서는 개별 변수가 아닌 **요인(그룹) 단위**의 선택이 자연스럽다.

Yuan & Lin (2006)은 Lasso와 LARS를 그룹 설정으로 확장하여 **요인 수준의 변수 선택**과 **정확한 추정**을 동시에 달성하는 Group Lasso를 제안한다.

### Lasso의 한계

Lasso를 그룹 구조가 있는 모형에 직접 적용하면 두 가지 문제가 발생한다:

1. **개별 더미변수 수준에서 선택**이 이루어져, 한 요인 내 일부 수준만 선택되는 비일관적 결과가 나올 수 있다
2. **직교 정규화(orthonormalization) 방식에 의존적**: 같은 요인을 다른 직교 대비로 표현하면 다른 모형이 선택될 수 있다. 이는 요인 선택 문제에서 바람직하지 않다

---

## 문제 설정

$J$개 요인을 가진 선형 모형:

$$
Y = \sum_{j=1}^{J} X_j \beta_j + \varepsilon, \quad \varepsilon \sim N_n(0, \sigma^2 I)
$$

여기서 $X_j \in \mathbb{R}^{n \times p_j}$는 $j$번째 요인의 설계 행렬, $\beta_j \in \mathbb{R}^{p_j}$는 해당 계수 벡터이다. 각 $X_j$는 직교 정규화 $X_j^T X_j = I_{p_j}$를 가정한다.

목표: $\beta_j$를 **벡터 전체 단위로** 0으로 수축시켜 불필요한 요인을 제거.

---

## Group Lasso 정형화

### 목적 함수

양의 정부호 행렬 $K_1, \ldots, K_J$가 주어졌을 때, Group Lasso 추정량은 다음으로 정의된다:

$$
\hat{\beta}^{GL} = \arg\min_{\beta} \frac{1}{2}\left\| Y - \sum_{j=1}^{J} X_j \beta_j \right\|^2 + \lambda \sum_{j=1}^{J} \|\beta_j\|_{K_j}
$$

여기서 $\|\eta\|_K = (\eta^T K \eta)^{1/2}$이다. 실제 구현에서는 $K_j = p_j I_{p_j}$를 사용하므로:

$$
\hat{\beta}^{GL} = \arg\min_{\beta} \frac{1}{2}\left\| Y - \sum_{j=1}^{J} X_j \beta_j \right\|^2 + \lambda \sum_{j=1}^{J} \sqrt{p_j} \|\beta_j\|_2
$$

### 페널티의 기하학적 해석

$\sqrt{p_j}$ 가중치는 그룹 크기에 따른 보정으로, 큰 그룹이 과도하게 불이익 받지 않도록 한다. 페널티의 기하학적 성질을 비교하면:

| 페널티 | 등위선 형태 | 특성 |
|--------|----------|------|
| $L_1$ (Lasso) | 다이아몬드 (꼭짓점) | 개별 좌표 방향으로 희소성 유도 |
| $L_2$ (Ridge) | 원 | 모든 방향 균등 수축, 희소성 없음 |
| Group Lasso | 실린더 + 꼭짓점 | 그룹 내 부드러운 수축 + **그룹 간 희소성** |

$\|\beta_j\|_2$는 그룹 내에서는 Ridge처럼 부드럽게 수축하지만, 그룹 간에는 Lasso처럼 **전체 그룹을 정확히 0**으로 보낸다. 이는 $\|\beta_j\|_2$가 $\beta_j = 0$에서 미분 불가능(non-differentiable)하기 때문이다.

---

## KKT 조건과 반복 알고리즘

### KKT 조건 (Proposition 1)

$K_j = p_j I_{p_j}$일 때, $\beta = (\beta_1', \ldots, \beta_J')'$이 해가 되기 위한 필요충분조건:

$$
\begin{cases}
-X_j^T(Y - X\beta) + \frac{\lambda \beta_j \sqrt{p_j}}{\|\beta_j\|} = 0 & \forall \beta_j \neq 0 \\
\|X_j^T(Y - X\beta)\| \leq \lambda\sqrt{p_j} & \forall \beta_j = 0
\end{cases}
$$

잔차와 그룹 설계 행렬의 상관의 $L_2$ 노름이 임계값 $\lambda\sqrt{p_j}$ 이하이면, 해당 그룹 전체가 제거된다.

### 닫힌 형태의 해 (직교 설계)

$X_j^T X_j = I_{p_j}$일 때, $S_j = X_j^T(Y - X\beta_{-j})$로 정의하면 해는:

$$
\hat{\beta}_j^{GL} = \left(1 - \frac{\lambda\sqrt{p_j}}{\|S_j\|}\right)_+ S_j
$$

이는 **그룹 단위의 soft-thresholding**이다:
- $\|S_j\| \leq \lambda\sqrt{p_j}$이면 그룹이 완전히 제거 ($\hat{\beta}_j = 0$)
- 그렇지 않으면 **방향은 유지**하되 크기만 비례적으로 수축

블록 좌표 하강(block coordinate descent)으로 $j = 1, \ldots, J$를 반복 적용하여 수렴한다.

---

## 해 경로 알고리즘

### Group LARS

LARS(Efron et al., 2004)를 그룹으로 확장한다. 개별 변수의 equicorrelation 조건을 그룹의 equi-norm 조건으로 재정의:

현재 활성 그룹 집합 $\mathcal{A}$에서, 각 그룹의 잔차 상관 노름 $\|X_j^T r\|_{K_j}$가 동일한 조건을 유지하며 해 경로를 추적한다.

**Group Lasso vs Group LARS**: 직교 설계에서 두 방법은 동일한 해를 제공하지만, 일반적인 설계에서는 다를 수 있다. Group Lasso의 해 경로는 일반적으로 구간별 선형(piecewise linear)이 **아닌** 반면, Group LARS의 해 경로는 항상 구간별 선형이다.

### Group Non-negative Garrotte

Breiman (1995)의 non-negative garrotte를 그룹으로 확장:

$$
\hat{c}^{GNG} = \arg\min_{c_j \geq 0} \left\| Y - \sum_{j=1}^{J} c_j X_j \tilde{\beta}_j \right\|^2 + \lambda \sum_{j=1}^{J} c_j
$$

$\tilde{\beta}_j$는 OLS 추정량, $c_j$는 스칼라 수축 계수. $c_j = 0$이면 그룹 전체 제거. 이 문제는 표준 Lasso 문제로 환원되므로 LARS로 효율적으로 풀 수 있다.

---

## 모형 선택 기준: $C_p$ 기준

해 경로에서 최적 $\lambda$를 선택하기 위해 $C_p$ 기준을 도입한다:

$$
C_p = \frac{\|Y - X\hat{\beta}\|^2}{\hat{\sigma}^2} - n + 2 \cdot df
$$

자유도(degrees of freedom) $df$는 Group Lasso에 대해 Yuan & Lin이 유도:

$$
df = \sum_{j=1}^{J} I(\|\hat{\beta}_j\| > 0) + \sum_{j=1}^{J} \frac{\|\hat{\beta}_j\|}{\|\tilde{\beta}_j\|} (p_j - 1)
$$

첫째 항은 활성 그룹의 수, 둘째 항은 각 활성 그룹 내 수축 정도를 반영한다. 그룹이 완전히 수축되지 않으면($\|\hat{\beta}_j\| \approx \|\tilde{\beta}_j\|$), 해당 그룹의 자유도 기여분은 $p_j$에 가까워진다.

---

## 시뮬레이션 결과 요약

Yuan & Lin은 15개 잠재 변수, 각 3개 더미변수(총 45개 변수)의 시뮬레이션에서:
- Group Lasso/LARS/Garrotte 모두 전통적 stepwise backward elimination보다 **예측 정확도와 요인 선택 정확도** 모두 우수
- Group Non-negative Garrotte가 Group Lasso보다 약간 더 나은 요인 선택 수행
- $C_p$ 기준이 교차 검증보다 효율적이면서 유사한 성능

---

## Reference

- Yuan, M. & Lin, Y. "Model Selection and Estimation in Regression with Grouped Variables." *JRSS-B*, 68(1), 49-67, 2006.
