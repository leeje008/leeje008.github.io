---
layout: post
title: "[Paper Review] Variable Selection via Penalized Neural Network: a Drop-Out-One Loss Approach"
categories: [Paper Review]
tags: [paper-review, variable-selection, deep-learning, statistics]
math: true
---

## Introduction

고차원 비모수 회귀에서 변수 선택은 대부분 선형성 또는 가법성(additivity) 가정에 의존한다. Ye & Sun (2018)은 신경망의 보편적 근사 성질을 활용하되, **Drop-Out-One Loss**라는 새로운 통계량으로 각 변수(또는 변수 그룹)의 유용성을 측정하여, 복잡한 비선형 상호작용이 존재하는 상황에서도 변수 선택이 가능한 GEPNN(Greedy Elimination Penalized Neural Network)을 제안한다.

## 문제 설정

비모수 회귀 모형:

$$y = f^*(x) + \varepsilon, \quad x \in \mathbb{R}^p, \quad p = O(\exp(n^l)), \; l \in (0, 1)$$

희소성 가정: $$f^*$$는 $$x$$의 부분집합 $$\{x_j : j \in S\}$$에만 의존하며, $$|S| < n$$. 목표는 관련 변수 집합 $$S$$를 정확히 식별하고, 좋은 예측 모형을 동시에 학습하는 것이다.

## 네트워크 구조와 페널티

### PNN (Penalized Neural Network)

단일 은닉층 신경망을 사용한다:

$$f_\eta(x) = \beta^T \psi(w^T x + t) + b$$

여기서 $$\psi$$는 $$\tanh$$ 활성화 함수, $$w \in \mathbb{R}^{p \times H}$$는 입력-은닉층 가중치, $$\beta \in \mathbb{R}^H$$는 은닉-출력층 가중치, $$t \in \mathbb{R}^H$$는 바이어스이다. 전체 파라미터는 $$\eta = (w, \beta, t, b)$$이다.

### 페널티 구조

PNN의 정규화된 목적함수:

$$\hat{\eta} = \arg\min_\eta \frac{1}{n}\sum_{i=1}^{n} \ell(y_i, f_\eta(x_i)) + \text{pen}(\eta)$$

페널티는 두 부분으로 구성된다:

$$\text{pen}(\eta) = \lambda_0 \|\theta\|_2^2 + \lambda_1 \sum_{j=1}^{p} \Omega_\alpha(w_{j,*})$$

여기서 $$\theta = (\beta, t, b)$$는 첫 번째 층을 제외한 파라미터, $$w_{j,*} \in \mathbb{R}^H$$는 $$j$$번째 입력 변수에 연결된 가중치 벡터이다.

**Sparse Group Lasso 페널티**:

$$\Omega_\alpha(w) = (1-\alpha)\|w\|_1 + \alpha\|w\|_2$$

이는 SPINN(Feng & Simon, 2019)과 동일한 페널티 형태로, 그룹 수준($$\|w\|_2$$)과 개별 수준($$\|w\|_1$$)의 희소성을 동시에 유도한다.

## Drop-Out-One Loss

### 핵심 아이디어

PNN 학습 후 각 변수 그룹 $$g_j$$의 유용성을 **재학습 없이** 평가한다. 변수 그룹 $$g_j$$에 연결된 가중치를 0으로 설정한 네트워크 $$\hat{\eta}^{-g_j}$$를 구성하고, 검증 데이터 $$\tilde{D}_n = \{(\tilde{x}_i, \tilde{y}_i)\}_{i=1}^{\tilde{n}}$$에서의 손실 변화를 측정한다:

$$\Delta_{\tilde{n}} L(\hat{\eta}^{-g_j}, \hat{\eta}) = \frac{1}{\tilde{n}} \sum_{i=1}^{\tilde{n}} \left[\ell(\tilde{y}_i, f_{\hat{\eta}^{-g_j}}(\tilde{x}_i)) - \ell(\tilde{y}_i, f_{\hat{\eta}}(\tilde{x}_i))\right]$$

$$\Delta_{\tilde{n}} L$$이 크면 $$g_j$$가 예측에 중요한 변수이고, 작으면 불필요한 변수이다.

### 재학습 불필요의 이점

기존의 변수 선택 방법(예: backward elimination)은 변수를 제거할 때마다 모형을 재학습해야 하므로 $$O(p)$$번의 재학습이 필요하다. Drop-Out-One Loss는 가중치를 0으로 설정하기만 하면 되므로 재학습 비용이 없다.

## GEPNN 알고리즘

### Algorithm 1: PNN 학습 (GIST + Block-wise Descent)

1. 활성 변수 집합 $$\gamma$$를 초기화 ($$\gamma = \{1, \ldots, p\}$$ 또는 이전 반복의 선택 결과)
2. GIST 알고리즘으로 최적화:
   - 매끄러운 부분(손실 + Ridge)에 대한 경사 하강
   - $$j \in \gamma$$인 변수에 대해서만 $$w_{j,*}$$를 업데이트 (block-wise descent)
   - Sparse Group Lasso의 proximal operator 적용: soft-thresholding + soft-scaling
3. 수렴 시 $$\hat{\eta}$$ 반환

Block-wise descent는 이미 제거된 변수($$j \notin \gamma$$)의 가중치를 0으로 고정하여 계산 효율을 높인다.

### Algorithm 2: Greedy Elimination

1. 전체 변수로 PNN 학습 (Algorithm 1)
2. 각 그룹 $$g_j$$에 대해 Drop-Out-One Loss $$\Delta_{\tilde{n}} L(\hat{\eta}^{-g_j}, \hat{\eta})$$ 계산
3. 임계값 $$\delta$$-th 백분위수를 계산:

$$\tau = \text{Percentile}_\delta\left(\{\Delta_{\tilde{n}} L(\hat{\eta}^{-g_j}, \hat{\eta})\}_{j=1}^{p}\right)$$

4. $$\Delta_{\tilde{n}} L < \tau$$인 그룹을 제거
5. 남은 변수로 PNN 재학습 (Algorithm 1)
6. 더 이상 제거할 변수가 없을 때까지 반복

$$\delta$$는 보통 25 또는 50으로 설정한다. 매 반복에서 하위 $$\delta\%$$의 변수를 제거하는 탐욕적(greedy) 전략이다.

## 이론적 성질: Oracle Property

### Assumption 4 (Identifiability)

$$\text{supp}(\eta_0) = S$$ for all $$\eta_0 \in \text{Eq}_0$$. 즉, 참 함수 $$f^*$$를 표현하는 모든 최적 파라미터 $$\eta_0$$에서, 관련 변수와 연결된 가중치가 비영(non-zero)이다. 이는 관련 변수의 효과가 불필요한 변수로 대체될 수 없음을 의미하며, 선형 모형의 **restricted isometry condition**에 대응하는 조건이다.

### Theorem 1: 첫 번째 반복의 Oracle Property

적절한 정규화 조건 하에서, GEPNN의 첫 번째 반복에서:

$$P(S^{(1)} = S) \to 1 \quad \text{as } n \to \infty$$

여기서 $$S^{(1)}$$은 첫 번째 반복 후 남은 변수 집합이다. 즉, 첫 번째 반복만으로도 관련 변수를 정확히 선택할 확률이 1로 수렴한다.

### Theorem 2: 개별 변수 선택의 Oracle Property

$$g_j = \{j\}$$ (개별 변수)인 경우, Assumption 4 하에서:

$$P(\hat{S} = S) \to 1$$

증명의 핵심은 두 단계로 나뉜다:

1. **관련 변수** ($$j \in S$$): Assumption 4에 의해 $$\|w_{j,*}\|_2 > 0$$이므로, $$f_{\hat{\eta}^{-g_j}}$$와 $$f_{\hat{\eta}}$$의 차이가 $$\Omega(\delta)$$만큼 존재. 따라서 $$\Delta_{\tilde{n}} L > \tau$$ w.h.p.

2. **불필요 변수** ($$j \notin S$$): Sparse Group Lasso 정규화에 의해 $$\|w_{j,*}\|_2 \to 0$$이므로, $$f_{\hat{\eta}^{-g_j}} \approx f_{\hat{\eta}}$$. 따라서 $$\Delta_{\tilde{n}} L \leq \tau$$ w.h.p.

### Corollary 3-4: 그룹 변수 선택

Assumption 4를 그룹 수준으로 약화한 **Assumption 4***:

$$\text{supp}_G(\eta_0) = S_G \quad \text{for all } \eta_0 \in \text{Eq}_0$$

여기서 $$S_G$$는 관련 그룹의 집합이다. 이 약화된 조건 하에서도 동일한 oracle property가 그룹 변수 선택에 대해 성립한다.

## 겹치는 그룹 지원: LR-OGL

변수 그룹이 겹치는(overlapping) 경우, Overlapping Group Lasso (OGL)를 latent variable representation으로 변환하여 처리한다:

$$\Omega_{\text{OGL}}(w) = \min_{v: \sum_g v_g = w} \sum_{g} \|v_g\|_2$$

이를 통해 겹치는 그룹 구조에서도 GEPNN을 적용할 수 있다.

## 시뮬레이션 결과

### 평가 지표

- **FSR** (False Selection Rate): 선택된 변수 중 불필요한 변수의 비율
- **NSR** (Negative Selection Rate): 관련 변수 중 선택되지 않은 변수의 비율

### 비교 방법

SPINN, $$\ell_1$$-NN, GAM, Random Forest, BART, Knockoffs, SIS-SCAD 등과 비교하여, GEPNN이 특히 비선형 상호작용이 강한 설정에서 FSR과 NSR 모두 우수한 성능을 보였다.

## 결론

GEPNN은 Drop-Out-One Loss라는 계산 효율적인 변수 중요도 측정법과 탐욕적 제거 전략을 결합하여, 재학습 없이도 신경망 기반 변수 선택을 수행한다. Oracle property는 Assumption 4라는 식별 가능성 조건 하에서 보장되며, 이 조건은 선형 모형의 restricted isometry condition에 비해 상당히 자연스러운 조건이다.

## Reference

Ye, M. & Sun, Y. (2018). Variable Selection via Penalized Neural Network: a Drop-Out-One Loss Approach. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80, 5601-5610.
