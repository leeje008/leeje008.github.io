---
layout: post
title: "[Dimension Reduction] PCA 이론 및 수렴 분석"
categories: [Paper Review]
tags: [paper-review, dimension-reduction, statistics]
math: true
---

## 1. Review Some Math

### Symmetric Matrix & Orthogonal Matrix

대칭행렬: $$A = A^T$$

직교행렬: $$AA^T = A^TA = I$$, 따라서 $$A^{-1} = A^T$$

다음 명제들은 동치:
- $$A$$가 가역행렬
- $$Ax = b$$가 유일한 해를 가짐
- $$A$$의 열들이 선형독립
- $$\det(A) \neq 0$$

### Eigenvector와 Eigenvalue

$$Ax = \lambda x$$

### Eigen-decomposition

$$A \in \mathbb{R}^{n \times n}$$이 $$n$$개의 선형독립인 고유벡터 $$\{v_1, v_2, \ldots, v_n\}$$을 가질 때:

$$A = V \text{Diag}(\Lambda) V^T = V \text{Diag}(\Lambda) V^{-1}$$

대칭행렬의 경우:

$$A = \sum_{i=1}^{n} \lambda_i v_i v_i^T$$

투영 연산자: $$v_i v_i^T x = P_{v_i}(x)$$

---

## 2. PCA (Principal Component Analysis)

### 정의

임의의 랜덤벡터 $$X \in \mathbb{R}^p$$에 대해 분산:

$$\text{Var}(X) = E[(X - \mu_X)(X - \mu_X)^T] \in \mathbb{R}^{p \times p}$$

단위벡터 $$v$$ ($$\|v\| = 1$$)로의 투영 $$v^TX$$의 변동성:

$$\text{Var}(v^T X) = v^T \text{Var}(X) v \geq 0$$

### 목표

최대 변동성을 갖는 방향 찾기:

$$\beta_1 = \underset{\beta}{\text{argmax}} \; \text{Var}(\beta^T X) \quad \text{s.t.} \; \|\beta\| = 1$$

순차적으로 $$\beta_2, \ldots, \beta_p$$를 찾는다 (라그랑주 승수법 사용, 이전 주성분에 직교하는 조건 추가).

### 용어

- **Loadings of PCA**: $$\{\beta_1, \ldots, \beta_q\}$$
- **First PC (score)**: $$\beta_1^T X$$
- **Second PC (score)**: $$\beta_2^T X$$

---

## 3. Convergence Analysis of PCA

### Vector Norm 성질

- $$\langle x, y \rangle = \langle y, x \rangle$$ (대칭성)
- $$\langle x, x \rangle \geq 0$$, $$x = 0 \iff \langle x, x \rangle = 0$$
- $$\|x\| = \sqrt{\langle x, x \rangle}$$
- $$|\langle x, y \rangle| \leq \|x\| \cdot \|y\|$$ (코시-슈바르츠 부등식)
- $$\|x + y\| \leq \|x\| + \|y\|$$ (삼각부등식)

### Matrix Norm

$$\|A\| := \sup\{\|Ax\|_2 : x \in \mathbb{R}^n, \|x\|_2 = 1\}$$

### 주요 정리

**Theorem 1.** $$\Sigma = \text{Var}(X)$$가 양의 반정치(p.s.d) 행렬이면 직교벡터와 양의 고유값으로 분해 가능하다.

**Theorem 2.** $$v_1$$, $$\hat{v}_{1,n}$$이 각각 모집단, 표본 고유벡터이고 $$d_n = \langle v_1, \hat{v}_{1,n} \rangle$$일 때:

$$\langle \hat{v}_{1,n}, \Sigma \hat{v}_{1,n} \rangle \leq \lambda_1 d_n^2 + \lambda_2 (1 - d_n^2)$$

**Theorem 3.** $$S_n$$이 표본공분산 행렬일 때:

$$|\langle \hat{v}_{1,n}, \Sigma \hat{v}_{1,n} \rangle - \langle \hat{v}_{1,n}, S_n \hat{v}_{1,n} \rangle| \leq \|\Sigma - S_n\|$$

**Theorem 4.** $$n \rightarrow \infty$$일 때:

$$|\langle \hat{v}_{1,n}, \Sigma \hat{v}_{1,n} \rangle - \langle v_1, \Sigma v_1 \rangle| \rightarrow 0$$

**Lemma 1.** $$d_n \rightarrow 1$$ (표본 고유벡터가 모집단 고유벡터로 수렴)

**Theorem 5.** 투영 행렬 $$Q_i$$, $$Q_{i,n}$$에 대해:

$$Q_i \Sigma Q_i = \sum_{j=i+1}^{p} \lambda_j v_j v_j^T$$

$$\|Q_{i,n} S_n Q_{i,n} - Q_i \Sigma Q_i\| \rightarrow 0$$

**Theorem 6.** $$d_{i,n} = \|\langle v_i, \hat{v}_{i,n} \rangle\|$$에 대해 모든 $$i \in \{1, 2, \ldots, p\}$$에서:

$$d_{i,n} \rightarrow 1$$

이는 **표본 고유벡터가 모집단 고유공간으로 수렴**함을 의미한다.

---

## Reference

- Jolliffe, I.T. "Principal Component Analysis." *Springer Series in Statistics*, 2002.
