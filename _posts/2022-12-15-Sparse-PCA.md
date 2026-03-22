---
layout: post
title: "[Paper Review] Sparse Principal Component Analysis"
categories: [Paper Review]
tags: [paper-review, dimension-reduction, pca, statistics]
math: true
---

## Introduction

주성분 분석(PCA)은 고차원 데이터의 분산을 최대화하는 방향을 찾는 가장 기본적인 차원 축소 방법이다. 그러나 PCA의 로딩(loading) 벡터는 **모든 원래 변수의 비영(non-zero) 선형 결합**이므로 해석이 어렵다. 특히 유전자 발현 데이터처럼 \(p \gg n\)인 고차원 설정에서, 수천 개 변수가 모두 관여하는 주성분은 실질적 해석이 불가능하다.

Zou, Hastie & Tibshirani (2006)는 PCA를 회귀 문제로 재정형화한 핵심적 통찰을 바탕으로, Elastic Net 페널티를 부과하여 **희소한 로딩(sparse loadings)**을 가진 주성분을 추정하는 Sparse PCA(SPCA)를 제안한다.

## PCA의 SVD 관점

### 기본 정형화

열 중심화된 데이터 행렬 \(X \in \mathbb{R}^{n \times p}\)의 특이값 분해(SVD)를 다음과 같이 쓴다:

$$X = UDV^T$$

여기서 \(U \in \mathbb{R}^{n \times r}\)는 좌특이벡터 행렬 (\(U^T U = I_r\)), \(D = \text{diag}(d_1, \ldots, d_r)\)은 특이값 행렬 (\(d_1 \geq \cdots \geq d_r > 0\)), \(V \in \mathbb{R}^{p \times r}\)는 우특이벡터 행렬 (\(V^T V = I_r\))이다. \(r = \min(n-1, p)\)는 \(X\)의 랭크이다.

PCA의 구성 요소:
- **주성분 점수(PC scores)**: \(Z = UD\), 즉 \(Z_i = Xv_i\) (\(i\)번째 열)
- **로딩(loadings)**: \(V\)의 열 \(v_1, \ldots, v_r\)
- **분산 설명**: \(i\)번째 PC가 설명하는 분산은 \(d_i^2 / (n-1)\)

## PCA의 회귀적 재정형화

### Theorem 1: PCA 로딩 = Ridge 회귀 계수

\(i\)번째 PC 점수 \(Z_i = XV_i\)에 대해, Ridge 회귀:

$$\hat{\beta}_{\text{ridge}} = \arg\min_\beta \|Z_i - X\beta\|^2 + \lambda\|\beta\|^2$$

의 해는 다음과 같다:

$$\hat{\beta}_{\text{ridge}} = (X^T X + \lambda I)^{-1} X^T Z_i$$

\(\lambda \to 0\)일 때, 정규화된 추정량:

$$\hat{v} = \frac{\hat{\beta}_{\text{ridge}}}{\|\hat{\beta}_{\text{ridge}}\|} = V_i$$

즉, **PCA 로딩은 PC 점수를 반응변수, 원래 변수를 설명변수로 한 Ridge 회귀의 계수 방향과 동일**하다. 이 정리가 SPCA의 이론적 기반이다.

### Naive Elastic Net 접근

Theorem 1에서 Ridge 페널티에 L1 페널티를 추가하면 희소한 로딩을 얻을 수 있다:

$$\hat{\beta} = \arg\min_\beta \|Z_i - X\beta\|^2 + \lambda\|\beta\|^2 + \lambda_1\|\beta\|_1$$

이것이 **Naive Elastic Net** 추정량이다. \(\lambda_1 > 0\)이면 일부 계수가 정확히 0이 되어 희소한 로딩을 얻는다.

그러나 이 접근에는 문제가 있다: \(Z_i\)가 미지수(unknown)이다. PCA를 먼저 수행한 뒤 희소화하는 2단계 절차는 비효율적이며, 주성분 간의 직교성도 보장되지 않는다.

## Self-contained SPCA 기준

### Theorem 2-3: 결합 최적화

Zou et al.은 PCA와 희소화를 **동시에** 수행하는 다음의 결합 최적화 문제를 제안한다:

$$(\hat{A}, \hat{B}) = \arg\min_{A, B} \sum_{i=1}^{n} \|x_i - AB^T x_i\|^2 + \lambda \sum_{j=1}^{k} \|\beta_j\|^2 + \sum_{j=1}^{k} \lambda_{1,j} \|\beta_j\|_1$$

$$\text{subject to } A^T A = I_k$$

여기서 \(A = (\alpha_1, \ldots, \alpha_k) \in \mathbb{R}^{p \times k}\), \(B = (\beta_1, \ldots, \beta_k) \in \mathbb{R}^{p \times k}\)이다.

**Theorem 2**: \(\lambda_{1,j} = 0\) (모든 \(j\))이면, \(\hat{B}\)의 열은 기존 PCA 로딩 \(V_1, \ldots, V_k\)에 비례한다.

**Theorem 3**: 일반적인 \(\lambda_{1,j} > 0\)에서, \(\hat{\beta}_j\)는 수정된(modified) 주성분의 희소 로딩이다. 구체적으로, \(\hat{\beta}_j / \|\hat{\beta}_j\|\)가 \(j\)번째 SPCA 로딩이 된다.

### 목적함수의 해석

$$\sum_{i=1}^{n} \|x_i - AB^T x_i\|^2$$

이 항은 \(A\)의 열공간으로 투영했을 때의 **재구성 오차**(reconstruction error)이다. \(B\)가 이 재구성에서 각 방향의 가중치를 결정한다. Elastic Net 페널티 \(\lambda\|\beta_j\|^2 + \lambda_{1,j}\|\beta_j\|_1\)는 \(B\)에만 부과되어 로딩의 희소성을 유도한다.

## 교대 최적화 알고리즘

### Step 1: \(A\) 고정, \(B\) 업데이트

\(A\)가 고정되면, 각 \(\beta_j\)에 대해 독립적인 Elastic Net 문제가 된다:

$$\hat{\beta}_j = \arg\min_\beta \|X\alpha_j - X\beta\|^2 + \lambda\|\beta\|^2 + \lambda_{1,j}\|\beta\|_1$$

이는 표준적인 Elastic Net 문제로, LARS-EN 알고리즘이나 좌표 하강법(coordinate descent)으로 효율적으로 풀 수 있다.

### Step 2: \(B\) 고정, \(A\) 업데이트

\(B\)가 고정되면, 문제는 **Procrustes 회전** 문제로 귀결된다:

$$\hat{A} = \arg\min_{A: A^T A = I_k} \|X - XBA^T\|_F^2$$

이의 해는:

$$\hat{A} = UV^T, \quad \text{where } X^T X B = UDV^T \text{ (SVD)}$$

이는 \(X^T X B\)의 SVD에서 좌특이벡터와 우특이벡터를 곱한 것으로, 닫힌 형태(closed-form)의 해를 가진다.

### 알고리즘 수렴

두 단계를 교대로 반복하면, 목적함수 값이 단조 감소하므로 수렴이 보장된다. 실제로 10~20회 반복이면 충분히 수렴한다.

### 초기화

\(A\)를 기존 PCA의 처음 \(k\)개 로딩 \(V_1, \ldots, V_k\)로 초기화한다.

## 수정된 주성분의 설명 분산

### 문제점

기존 PCA에서 총 분산은 \(\sum_{i=1}^{r} d_i^2\)이고, \(k\)개 PC가 설명하는 분산 비율은 \(\sum_{i=1}^{k} d_i^2 / \sum_{i=1}^{r} d_i^2\)이다. 그러나 SPCA의 수정된 로딩 \(\hat{V}_j = \hat{\beta}_j / \|\hat{\beta}_j\|\)는 일반적으로 직교하지 않으므로, 이 공식이 성립하지 않는다.

### Adjusted Total Variance

Zou et al.은 다음의 **조정된 분산** 공식을 제안한다:

1. 수정된 PC를 \(\hat{Z}_j = X\hat{V}_j\)로 정의
2. \(\hat{Z}_1, \ldots, \hat{Z}_k\)에 QR 분해를 적용하여 직교화: \(\hat{Z} = QR\)
3. 조정된 분산: \(\text{Adj.Var} = \sum_{j=1}^{k} \|Q_j\|^2 = \text{tr}(R^T R)\)

이 공식은 수정된 PC들 간의 상관을 제거한 후의 분산을 측정하므로, 해석이 올바르다.

## \(p \gg n\)인 경우의 효율적 계산

유전자 발현 데이터 등 \(p \gg n\)인 경우, \(X^T X \in \mathbb{R}^{p \times p}\)를 직접 계산하는 것은 비효율적이다. 대신:

1. \(X^T X\)의 고유값 분해 대신 \(XX^T \in \mathbb{R}^{n \times n}\)의 고유값 분해를 사용
2. \(XX^T = UD^2 U^T\)에서 \(V = X^T U D^{-1}\)로 변환
3. Elastic Net 단계에서도 \(X\alpha_j\)를 미리 계산하여 \(n\)차원 문제로 축소

이를 통해 \(p\)가 수만 개인 경우에도 SPCA를 실용적으로 적용할 수 있다.

## SCoTLASS와의 비교

Jolliffe, Trendafilov & Uddin (2003)의 SCoTLASS는 PCA에 직접 L1 제약을 부과한다:

$$\max_v v^T X^T X v \quad \text{s.t. } v^T v = 1, \|v\|_1 \leq t$$

이 문제는 **비볼록**(non-convex)이며, \(p\)가 클 때 계산이 매우 비싸다. SPCA는 회귀적 재정형화를 통해 볼록 최적화(각 \(\beta_j\)에 대한 Elastic Net)로 변환하여 계산 효율성을 크게 향상시켰다.

## 유전자 발현 데이터 적용

논문에서는 유전자 발현 데이터에 SPCA를 적용하여, 각 주성분이 소수의 유전자 그룹(gene set)으로 구성됨을 보인다. 기존 PCA에서는 수천 개의 유전자가 모두 비영 로딩을 가지지만, SPCA에서는 10~50개의 유전자만 선택되어 생물학적으로 해석 가능한 주성분을 제공한다.

## 결론

Sparse PCA는 PCA를 Ridge 회귀로 재정형화한 핵심 통찰(Theorem 1)을 바탕으로, Elastic Net 페널티를 통해 희소한 로딩을 추정한다. 결합 최적화 기준(Theorem 2-3)은 PCA와 변수 선택을 동시에 수행하며, Procrustes + Elastic Net의 교대 최적화로 효율적으로 풀 수 있다. \(p \gg n\)인 고차원 설정에서도 \(n \times n\) 행렬로의 축소를 통해 실용적 적용이 가능하다.

## Reference

Zou, H., Hastie, T. & Tibshirani, R. (2006). Sparse Principal Component Analysis. *Journal of Computational and Graphical Statistics*, 15(2), 265-286.
