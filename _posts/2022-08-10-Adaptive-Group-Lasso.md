---
layout: post
title: "[Paper Review] Adaptive Group Lasso - Oracle 성질을 가진 그룹 변수 선택"
categories: [Paper Review]
tags: [paper-review, variable-selection, regularization, statistics]
math: true
---

## Introduction

Group Lasso(Yuan & Lin, 2006)는 그룹 단위 변수 선택이 가능하지만, 모든 그룹에 **동일한 페널티 강도** $$\lambda$$를 적용하기 때문에 추정 비효율성과 선택 비일치성 문제가 있다. 이는 Lasso가 가진 근본적 한계와 동일하다: 강한 신호의 계수도 약한 신호와 같은 정도로 수축된다.

Wang & Leng (2008)은 Zou (2006)의 adaptive lasso 아이디어를 그룹 설정으로 확장하여, 각 그룹에 **데이터 적응적(data-adaptive) 가중치**를 부여하는 Adaptive Group Lasso(agLasso)를 제안한다.

---

## Group Lasso의 한계

$$p$$개 그룹, $$j$$번째 그룹이 $$d_j$$개 변수를 포함하는 모형:

$$
y_i = \sum_{j=1}^{p} x_{ij}^T \beta_j + e_i, \quad i = 1, \ldots, n
$$

Group Lasso:

$$
\hat{\beta}^{GL} = \arg\min_{\beta} \frac{1}{2}\|Y - X\beta\|^2 + n\lambda \sum_{j=1}^{p} \|\beta_j\|
$$

**문제 1 — 추정 비효율성**: 참인 모형의 비영(nonzero) 그룹에도 수축 편향이 존재하여, oracle 추정량(참인 모형을 알고 추정)과 비교 시 점근 효율이 떨어진다.

**문제 2 — 선택 비일치성**: 특정 조건 하에서 $$n \to \infty$$일 때에도 참인 모형을 정확히 복원하지 못할 수 있다. Fan & Li (2001)가 지적한 것처럼, 모든 계수에 동일한 수축량을 적용하면 큰 계수의 편향과 작은 계수의 불충분한 수축이 동시에 발생한다.

---

## Adaptive Group Lasso

### 정형화

$$
Q(\beta) = \frac{1}{2}\|Y - X\beta\|^2 + n \sum_{j=1}^{p} \lambda_j \|\beta_j\|
$$

여기서 적응적 가중치 $$\lambda_j$$는 초기 일치 추정량 $$\tilde{\beta}$$ (예: OLS)를 사용하여:

$$
\lambda_j = \lambda \|\tilde{\beta}_j\|^{-\gamma}, \quad \gamma > 0
$$

**핵심 직관**:
- $$\|\tilde{\beta}_j\|$$가 큰 그룹(강한 신호) → 작은 $$\lambda_j$$ → **약하게 수축** → 편향 최소화
- $$\|\tilde{\beta}_j\|$$가 작은 그룹(약한 신호/잡음) → 큰 $$\lambda_j$$ → **강하게 수축** → 정확한 제거

이렇게 **그룹마다 다른 수축량**을 적용함으로써, Group Lasso의 "one-size-fits-all" 한계를 극복한다.

### Group Lasso와의 관계

agLasso의 정형화에서 $$\lambda_j$$를 원래 튜닝 파라미터 $$\lambda$$와 가중치의 곱으로 분해하면:

$$
Q(\beta) = \frac{1}{2}\|Y - X\beta\|^2 + n\lambda \sum_{j=1}^{p} \underbrace{\|\tilde{\beta}_j\|^{-\gamma}}_{\text{adaptive weight}} \|\beta_j\|
$$

$$\gamma = 0$$이면 모든 가중치가 1로 동일해져 Group Lasso로 환원된다.

---

## Oracle 성질

**Oracle 추정량**이란 참인 활성 그룹 집합 $$\{1, \ldots, p_0\}$$을 미리 알고, 해당 그룹만으로 추정한 OLS 추정량이다. agLasso가 oracle 성질을 가진다는 것은 다음 두 조건을 동시에 만족한다는 의미이다.

### 정리 1 (Estimation Consistency)

$$\sqrt{n} a_n \to_p 0$$이면:

$$
\hat{\beta} - \beta = O_p(n^{-1/2})
$$

여기서 $$a_n = \max\{\lambda_j : j \leq p_0\}$$. 활성 그룹에 대한 수축량이 충분히 작으면, 추정량이 $$\sqrt{n}$$-일치적이다.

### 정리 2 (Selection Consistency)

$$\sqrt{n} a_n \to_p 0$$이고 $$\sqrt{n} b_n \to_p \infty$$이면:

$$
P(\hat{\beta}_b = 0) \to 1
$$

여기서 $$b_n = \min\{\lambda_j : j > p_0\}$$, $$\hat{\beta}_b$$는 비활성 그룹의 추정량. 즉, 참이 아닌 그룹을 정확히 0으로 추정할 확률이 1로 수렴한다.

### 정리 3 (Oracle Property)

위 두 조건이 동시에 만족되면:

$$
\sqrt{n}(\hat{\beta}_a - \beta_a) \xrightarrow{d} \mathcal{N}(0, \sigma^2 \Sigma_a^{-1})
$$

여기서 $$\Sigma_a = E[x_{i,a} x_{i,a}^T]$$는 활성 그룹만의 정보 행렬. 이는 **참인 모형을 미리 알고 추정한 oracle 추정량과 동일한 점근 분포**이다.

### $$\gamma$$와 $$\lambda$$의 조건

$$\lambda_j = \lambda \|\tilde{\beta}_j\|^{-\gamma}$$에서 $$\lambda = n^\alpha$$로 설정하면:

$$
\begin{cases}
a_n \leq \lambda \cdot O(n^{\gamma/2}) = O(n^{\alpha + \gamma/2}) & \to 0 \text{ (필요조건: } \alpha + \gamma/2 < 0) \\
b_n \geq \lambda \cdot O(n^{\gamma/2}) = O(n^{\alpha + \gamma/2}) & \to \infty \text{ (필요조건: } \alpha > 0)
\end{cases}
$$

따라서 $$\alpha \in (-(1+\gamma)/2, -1/2)$$를 만족하는 $$\alpha$$가 존재하려면 $$\gamma > 0$$이면 충분하다. 실제로는 $$\gamma = 1$$이 가장 많이 사용된다.

---

## 튜닝 파라미터 선택

$$p$$차원 튜닝 파라미터 $$(\lambda_1, \ldots, \lambda_p)$$를 교차 검증으로 선택하는 것은 비현실적이다. 대신 $$\lambda_j = \lambda \|\tilde{\beta}_j\|^{-\gamma}$$로 단일 파라미터 $$\lambda$$로 축소한 후, 다음 기준들로 $$\lambda$$를 선택한다:

$$
\begin{aligned}
C_p &= \frac{\|Y - X\hat{\beta}\|^2}{\hat{\sigma}^2} - n + 2 \cdot df \\
\text{BIC} &= \log\left(\frac{\|Y - X\hat{\beta}\|^2}{n}\right) + \frac{\log n}{n} \cdot df
\end{aligned}
$$

여기서 $$df = \sum_{j=1}^{p} I(\|\hat{\beta}_j\| > 0) + \sum_{j:\hat{\beta}_j \neq 0} \frac{\|\hat{\beta}_j\|}{\|\tilde{\beta}_j\|} (d_j - 1)$$. BIC이 모형 선택 일치성에서 $$C_p$$보다 우수한 경향을 보인다.

---

## 시뮬레이션 결과

Wang & Leng은 Yuan & Lin (2006)의 시뮬레이션 설정을 차용하여 agLasso, aLasso, gLasso 세 방법을 비교:

- **모형 선택 정확도**: agLasso가 모든 표본 크기와 노이즈 수준에서 가장 높은 correct model 비율
- **모형 크기**: agLasso가 가장 작은 평균 모형 크기 (불필요한 요인을 더 정확히 제거)
- **예측 정확도**: 세 방법 간 outsample MSE 차이는 크지 않으나, agLasso가 약간 우수
- **BIC 선택 기준**: $$C_p$$, GCV, AIC보다 모형 크기 과대추정이 적어 가장 효과적

---

## Reference

- Wang, H. & Leng, C. "A Note on Adaptive Group Lasso." *Computational Statistics and Data Analysis*, 52, 5277-5286, 2008.
- Zou, H. "The Adaptive Lasso and Its Oracle Properties." *JASA*, 101(476), 1418-1429, 2006.
- Yuan, M. & Lin, Y. "Model Selection and Estimation in Regression with Grouped Variables." *JRSS-B*, 68(1), 49-67, 2006.
