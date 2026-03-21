---
layout: post
title: "[Paper Review] Sparse PCA - Elastic Net을 활용한 희소 주성분 분석"
categories: [Paper Review]
tags: [paper-review, dimension-reduction, regularization, statistics]
math: true
---

## Introduction

PCA는 강력한 차원 축소 도구이지만, 각 주성분이 **모든 원래 변수의 선형 결합**이므로 해석이 어렵다. Zou, Hastie & Tibshirani (2006)는 PCA를 회귀 문제로 재정형화한 뒤 Elastic Net 페널티를 부과하여, **희소한 로딩(sparse loadings)**을 가진 주성분을 추정하는 Sparse PCA(SPCA)를 제안한다.

---

## PCA의 회귀적 재정형화

### SVD 관점의 PCA

데이터 $X \in \mathbb{R}^{n \times p}$ (열 중심화)의 SVD를 $X = UDV^T$라 하면:
- 주성분(PC): $Z = UD$
- 로딩: $V$의 열

### PCA를 회귀로 보기

**정리**: 첫 $k$개 로딩 $V_k$는 다음 최적화의 해이다:

$$
\hat{V}_k = \arg\min_{A, B} \sum_{i=1}^{n} \|x_i - AB^T x_i\|^2 + \lambda \sum_{j=1}^{k} \|\beta_j\|^2 \quad \text{s.t. } A^TA = I_k
$$

$\lambda \to 0$이면 $\hat{B} \to V_k$. 핵심 통찰: PCA가 **자기 자신에 대한 회귀 문제**임을 보여준다.

---

## Sparse PCA

### 정형화

회귀 문제에 Elastic Net 페널티를 추가한다:

$$
(\hat{A}, \hat{B}) = \arg\min_{A, B} \sum_{i=1}^{n} \|x_i - AB^T x_i\|^2 + \lambda \sum_{j=1}^{k} \|\beta_j\|^2 + \sum_{j=1}^{k} \lambda_{1,j} \|\beta_j\|_1
$$

제약 조건: $A^TA = I_k$

- **Ridge 항** ($\|\beta_j\|^2$): PCA와의 연결을 유지
- **Lasso 항** ($\|\beta_j\|_1$): 로딩의 희소성(sparsity) 유도

### 교대 최적화

**Step 1** ($A$ 고정, $B$ 업데이트): 각 $\beta_j$에 대해 독립적인 Elastic Net 문제:

$$
\hat{\beta}_j = \arg\min_{\beta} \|X\alpha_j - X\beta\|^2 + \lambda\|\beta\|^2 + \lambda_{1,j}\|\beta\|_1
$$

**Step 2** ($B$ 고정, $A$ 업데이트): Procrustes 문제로 귀결:

$$
\hat{A} = UV^T, \quad \text{where } X^TX B = UDV^T \text{ (SVD)}
$$

수렴할 때까지 반복한다.

---

## 수정된 주성분의 설명 분산

기존 PCA의 분산 분해 ($\sum \lambda_i^2$)가 희소 로딩에서는 성립하지 않는다. Zou et al.은 수정된 분산 공식을 제시한다:

$\hat{V}_j$가 $j$번째 희소 로딩이면, 조정된 분산:

$$
\text{Adj.Var}(\hat{V}_j) = \hat{V}_j^T (X^TX / n) \hat{V}_j
$$

총 설명 분산은 QR 분해를 통해 직교화한 후 계산한다.

---

## 유전자 발현 데이터에의 적용

$p \gg n$인 유전자 발현 어레이에서 SPCA의 효율적 변형:

$X^TX$의 고유값 분해를 활용하여 $n \times n$ 문제로 축소하고, 각 주성분에 대한 Elastic Net을 $n$차원에서 수행한다. 이는 $p$가 수만 개인 경우에도 실용적이다.

---

## Reference

- Zou, H., Hastie, T. & Tibshirani, R. "Sparse Principal Component Analysis." *JCGS*, 15(2), 265-286, 2006.
