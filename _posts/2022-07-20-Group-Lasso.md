---
layout: post
title: "[Paper Review] Group Lasso - 그룹 변수 선택을 위한 정규화"
categories: [Paper Review]
tags: [paper-review, variable-selection, regularization, statistics]
math: true
---

## Introduction

Lasso(Tibshirani, 1996)는 개별 변수 선택에 탁월하지만, 설명변수가 **그룹 구조**를 가질 때 (예: ANOVA의 요인별 더미변수, 비모수 모형의 기저함수 그룹) 개별 변수가 아닌 **그룹 단위**의 선택이 필요하다. Yuan & Lin (2006)은 이 문제를 해결하는 Group Lasso를 제안한다.

---

## 문제 설정

$J$개 요인을 가진 선형 모형:

$$
Y = \sum_{j=1}^{J} X_j \beta_j + \varepsilon
$$

여기서 $X_j \in \mathbb{R}^{n \times p_j}$는 $j$번째 요인의 설계 행렬, $\beta_j \in \mathbb{R}^{p_j}$는 해당 계수 벡터이다. 각 $X_j$는 직교 정규화(Gram-Schmidt)를 가정한다.

목표: $\beta_j$를 **벡터 전체 단위로** 0으로 수축시켜 불필요한 요인을 제거.

---

## Group Lasso 정형화

### 목적 함수

$$
\hat{\beta}^{GL} = \arg\min_{\beta} \left\| Y - \sum_{j=1}^{J} X_j \beta_j \right\|^2 + \lambda \sum_{j=1}^{J} \sqrt{p_j} \|\beta_j\|_2
$$

핵심 차이:
- **Lasso**: $\lambda \sum_j |\beta_j|$ ($L_1$ norm, 개별 변수)
- **Group Lasso**: $\lambda \sum_j \sqrt{p_j} \|\beta_j\|_2$ ($L_1/L_2$ mixed norm, 그룹 단위)

$\sqrt{p_j}$ 가중치는 그룹 크기에 따른 페널티 보정으로, 큰 그룹이 과도하게 불이익 받지 않도록 한다.

### 왜 $L_1/L_2$ mixed norm인가

$\|\beta_j\|_2$는 그룹 내에서는 Ridge처럼 부드럽게 수축하지만, 그룹 간에는 Lasso처럼 **전체 그룹을 정확히 0**으로 보낸다. 이는 $\|\beta_j\|_2$가 $\beta_j = 0$에서 미분 불가능하기 때문이다.

### KKT 조건

최적해에서 각 그룹 $j$에 대해:

$$
\begin{cases}
\beta_j = 0 & \text{if } \|X_j^T r\|_2 \leq \lambda\sqrt{p_j} \\
X_j^T r = \lambda\sqrt{p_j} \frac{\beta_j}{\|\beta_j\|_2} & \text{if } \beta_j \neq 0
\end{cases}
$$

여기서 $r = Y - X\beta$는 잔차. 잔차와 그룹 설계 행렬의 상관이 임계값 이하이면 그룹 전체가 제거된다.

---

## 해 경로 알고리즘

### Group LARS

LARS(Efron et al., 2004)를 그룹으로 확장한다. 핵심: 현재 활성 그룹 집합 $\mathcal{A}$에서의 equicorrelation 조건을 그룹 단위로 재정의한다.

### Group Non-negative Garrotte

Breiman (1995)의 non-negative garrotte를 그룹으로 확장:

$$
\hat{c}^{GNG} = \arg\min_{c_j \geq 0} \left\| Y - \sum_{j=1}^{J} c_j X_j \tilde{\beta}_j \right\|^2 + \lambda \sum_{j=1}^{J} c_j
$$

여기서 $\tilde{\beta}_j$는 OLS 추정량. 수축 계수 $c_j$가 0이면 그룹 전체가 제거된다.

---

## 이론적 성질

직교 설계($X^TX = I$) 하에서 Group Lasso 해는 닫힌 형태를 가진다:

$$
\hat{\beta}_j^{GL} = \left(1 - \frac{\lambda\sqrt{p_j}}{\|\tilde{\beta}_j\|_2}\right)_+ \tilde{\beta}_j
$$

이는 **그룹 단위의 soft-thresholding**이다. $\|\tilde{\beta}_j\|_2 \leq \lambda\sqrt{p_j}$이면 그룹이 완전히 제거되고, 그렇지 않으면 방향은 유지하되 크기만 수축된다.

---

## Reference

- Yuan, M. & Lin, Y. "Model Selection and Estimation in Regression with Grouped Variables." *JRSS-B*, 68(1), 49-67, 2006.
