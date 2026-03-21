---
layout: post
title: "[Paper Review] Variable Selection via Penalized Neural Network - Drop-Out-One Loss"
categories: [Paper Review]
tags: [paper-review, variable-selection, deep-learning, statistics]
math: true
---

## Introduction

고차원 비모수 회귀에서 변수 선택은 대부분 선형성 또는 가법성(additivity) 가정에 의존한다. Ye & Sun (2018)은 신경망의 보편적 근사 성질을 활용하되, **Drop-Out-One Loss**라는 새로운 통계량으로 각 변수의 유용성을 측정하여 복잡한 비선형 상호작용이 존재하는 상황에서도 변수 선택이 가능한 방법을 제안한다.

---

## 문제 설정

비모수 회귀 모형:

$$
y = f^*(x) + \varepsilon, \quad x \in \mathbb{R}^p, \quad p = O(\exp(n^l)), \; l \in (0, 1)
$$

희소성 가정: $f^*$는 $x$의 부분집합 $\{x_j : j \in S\}$에만 의존하며, $|S| < n$.

목표: 관련 변수 집합 $S$를 식별하고, 좋은 예측 모형을 학습.

---

## Drop-Out-One Loss

### 핵심 아이디어

1. 모든 변수를 사용하여 페널티 부과된 신경망 $\hat{f}$를 학습 (lower bound model)
2. 각 변수 $x_j$에 대해, 해당 변수와 연결된 가중치를 **재학습 없이 제거**
3. 손실 함수의 변화량을 측정:

$$
D_j = \hat{R}(\hat{f}_{-j}) - \hat{R}(\hat{f})
$$

$D_j$가 작으면 $x_j$는 불필요한 변수이므로 제거한다.

### 수학적 정의

첫 번째 은닉층의 가중치를 $W = (w_1, \ldots, w_p)^T$라 하면:

$$
D_j = \frac{1}{n} \sum_{i=1}^{n} \left[\ell(y_i, \hat{f}_{-j}(x_i)) - \ell(y_i, \hat{f}(x_i))\right]
$$

여기서 $\hat{f}_{-j}$는 $w_j$를 0으로 설정한 네트워크이다.

### 변수 제거 기준

임계값 $\tau_n$에 대해:

$$
\hat{S} = \{j : D_j > \tau_n\}
$$

---

## 그룹 변수 선택으로의 확장

변수가 $d$개 그룹 $g_1, \ldots, g_d$로 나뉠 때, 그룹 단위로 Drop-Out-One Loss를 정의:

$$
D_{g_k} = \hat{R}(\hat{f}_{-g_k}) - \hat{R}(\hat{f})
$$

그룹이 겹칠 수 있는(overlapping) 경우도 허용한다.

---

## Oracle 성질

### 정리 (Selection Consistency)

적절한 정규화 조건 하에서, $n \to \infty$일 때:

$$
P(\hat{S} = S) \to 1
$$

즉, 관련 변수를 정확히 선택하고 불필요한 변수를 제거할 확률이 1로 수렴한다.

### 증명의 핵심

1. **관련 변수** ($j \in S$): $f^*$가 $x_j$에 의존하므로, $x_j$ 제거 시 근사 오차가 $\Omega(\delta)$만큼 증가. 신경망의 보편적 근사 성질에 의해 $D_j > \tau_n$ w.h.p.

2. **불필요 변수** ($j \notin S$): $f^*$가 $x_j$에 의존하지 않으므로, 정규화에 의해 $\|w_j\| \approx 0$. 따라서 $D_j \leq \tau_n$ w.h.p.

### Feature Screening과의 관계

기존 feature screening(Fan et al., 2011)은 $E(f_j^2(x_j))$로 변수 유용성을 측정하지만, 이는 **주변 회귀(marginal regression)**에 기반하여 변수 간 상호작용을 놓칠 수 있다. Drop-Out-One Loss는 전체 모형에서 변수를 제거하므로 상호작용 효과도 포착한다.

---

## Reference

- Ye, M. & Sun, Y. "Variable Selection via Penalized Neural Network: a Drop-Out-One Loss Approach." *ICML 2018*.
