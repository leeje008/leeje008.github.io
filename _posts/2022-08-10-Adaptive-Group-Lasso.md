---
layout: post
title: "[Paper Review] Adaptive Group Lasso - Oracle 성질을 가진 그룹 변수 선택"
categories: [Paper Review]
tags: [paper-review, variable-selection, regularization, statistics]
math: true
---

## Introduction

Group Lasso(Yuan & Lin, 2006)는 그룹 단위 변수 선택이 가능하지만, 모든 그룹에 **동일한 수축량**을 적용하기 때문에 추정 비효율성과 선택 비일치성 문제가 있다. Wang & Leng (2008)은 adaptive lasso(Zou, 2006)의 아이디어를 그룹 설정으로 확장하여 **Oracle 성질**을 보장하는 Adaptive Group Lasso를 제안한다.

---

## Group Lasso의 한계

$p$개 그룹, $j$번째 그룹이 $d_j$개 변수를 포함하는 모형:

$$
y_i = \sum_{j=1}^{p} x_{ij}^T \beta_j + e_i
$$

Group Lasso: $\hat{\beta}^{GL} = \arg\min \|Y - X\beta\|^2 + \lambda_n \sum_{j=1}^{p} \|\beta_j\|_2$

**문제 1 - 추정 비효율성**: 참인 모형을 알고 있는 oracle 추정량과 비교했을 때, Group Lasso 추정량은 추가적인 편향(bias)을 가진다.

**문제 2 - 선택 비일치성**: $n \to \infty$일 때 참인 모형을 정확히 복원하지 못할 수 있다. 이는 모든 그룹에 동일한 $\lambda_n$을 적용하기 때문이다.

---

## Adaptive Group Lasso

### 정형화

$$
\hat{\beta}^{aGL} = \arg\min_{\beta} \|Y - X\beta\|^2 + \lambda_n \sum_{j=1}^{p} \hat{w}_j \|\beta_j\|_2
$$

여기서 적응적 가중치 $\hat{w}_j$는 초기 추정량 $\tilde{\beta}_j$ (예: OLS)를 사용하여:

$$
\hat{w}_j = \|\tilde{\beta}_j\|_2^{-\gamma}, \quad \gamma > 0
$$

**직관**: $\|\tilde{\beta}_j\|_2$가 작은(약한 신호) 그룹은 큰 가중치를 받아 강하게 수축되고, 큰(강한 신호) 그룹은 작은 가중치를 받아 약하게 수축된다.

---

## Oracle 성질

다음 조건 하에서 Adaptive Group Lasso는 **Oracle 성질**을 만족한다:

### 정리 (Selection Consistency)

$\lambda_n / \sqrt{n} \to 0$이고 $\lambda_n n^{(\gamma-1)/2} \to \infty$이면:

$$
P\left(\hat{\beta}_j^{aGL} = 0 \text{ for all } j > p_0\right) \to 1
$$

즉, 참이 아닌 그룹을 정확히 제거할 확률이 1로 수렴한다.

### 정리 (Estimation Efficiency)

참인 그룹 $\{1, \ldots, p_0\}$에 대해:

$$
\sqrt{n}(\hat{\beta}_{\mathcal{A}}^{aGL} - \beta_{\mathcal{A}}^*) \xrightarrow{d} \mathcal{N}(0, \sigma^2 C_{\mathcal{A}}^{-1})
$$

여기서 $C_{\mathcal{A}} = E[x_{i,\mathcal{A}} x_{i,\mathcal{A}}^T]$는 참인 그룹만의 정보 행렬. 이는 **참인 모형을 미리 알고 추정한 oracle 추정량과 동일한 점근 분포**이다.

### Oracle 성질의 의미

1. **변수 선택 일치성**: 표본이 커지면 참인 모형을 정확히 복원
2. **추정 효율성**: oracle 추정량과 동일한 점근 효율

이 두 성질을 동시에 만족하는 것이 Group Lasso와의 결정적 차이이다.

---

## $\gamma$와 $\lambda_n$의 선택

- $\gamma > 1$이면 조건 $\lambda_n n^{(\gamma-1)/2} \to \infty$가 더 쉽게 충족된다
- 실제로는 $\gamma = 1$이 가장 많이 사용됨
- $\lambda_n$은 BIC 기준으로 선택:

$$
\text{BIC}(\lambda) = \log\left(\frac{\text{RSS}(\lambda)}{n}\right) + \frac{\log n}{n} \cdot df(\lambda)
$$

여기서 $df(\lambda) = \sum_{j:\hat{\beta}_j \neq 0} d_j$ (활성 그룹의 총 변수 수)

---

## Reference

- Wang, H. & Leng, C. "A Note on Adaptive Group Lasso." *Computational Statistics and Data Analysis*, 52, 5277-5286, 2008.
