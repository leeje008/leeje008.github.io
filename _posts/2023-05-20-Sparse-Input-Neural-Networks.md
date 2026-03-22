---
layout: post
title: "[Paper Review] Sparse-Input Neural Networks for High-dimensional Nonparametric Regression and Classification"
categories: [Paper Review]
tags: [paper-review, deep-learning, variable-selection, high-dimensional, statistics]
math: true
---

## Introduction

고차원(\(p \gg n\)) 비모수 회귀에서 신경망은 일반적으로 학습 표본이 부족하여 사용이 어렵다. Feng & Simon (2019)은 **첫 번째 층에 Sparse Group Lasso 페널티**를 부과하여 소수의 입력 변수만 선택하는 Sparse-Input Neural Network(SPINN)을 제안하고, 초과 위험(excess risk)이 전체 변수 수 \(p\)의 **로그에만 비례**하여 증가함을 증명한다. 이는 신경망에 대한 최초의 고차원 변수 선택 이론을 제공하는 연구이다.

## 네트워크 구조

### SPINN의 정형화

\(L\)층 신경망을 다음과 같이 정의한다. 은닉층 \(a = 1, \ldots, L\)에 대해:

$$z_a(x) = \psi(\theta_a z_{a-1}(x) + t_a)$$

여기서 \(\psi\)는 활성화 함수(예: ReLU, sigmoid), \(\theta_a\)는 \(a\)번째 층의 가중치 행렬, \(t_a\)는 바이어스이다. \(z_0(x) = x\)로 놓으면, 최종 출력은:

$$f_\eta(x) = \theta_{L+1} z_L(x) + t_{L+1}$$

여기서 \(\eta = \{\theta_1, t_1, \ldots, \theta_{L+1}, t_{L+1}\}\)는 전체 파라미터 집합이다.

### 핵심: 첫 번째 층의 구조

첫 번째 층의 가중치 행렬 \(\theta_1 \in \mathbb{R}^{h_1 \times p}\)에서, \(j\)번째 입력 변수에 연결된 가중치 벡터를 \(\theta_{1, \cdot, j} \in \mathbb{R}^{h_1}\)로 표기한다. 만약 \(\theta_{1, \cdot, j} = 0\)이면, 입력 변수 \(x_j\)는 네트워크의 출력에 **전혀 영향을 미치지 않는다**. 이것이 변수 선택의 기반이 된다.

## 손실 함수와 페널티

### 정규화된 목적함수

SPINN의 목적함수는 세 가지 구성요소로 이루어진다:

$$\hat{\eta} = \arg\min_{\eta} \frac{1}{n}\sum_{i=1}^{n} \ell(y_i, f_\eta(x_i)) + \lambda_0 \sum_{a=1}^{L+1} \|\theta_a\|_2^2 + \lambda \sum_{j=1}^{p} \Omega_\alpha(\theta_{1, \cdot, j})$$

각 항의 역할:
- **경험적 손실** \(\frac{1}{n}\sum \ell(y_i, f_\eta(x_i))\): 데이터 적합도. 회귀에서는 제곱 손실, 분류에서는 교차 엔트로피 등.
- **Ridge 페널티** \(\lambda_0 \sum_a \|\theta_a\|_2^2\): 모든 층의 가중치에 대한 전역적 정규화. 과적합 방지.
- **Sparse Group Lasso 페널티** \(\lambda \sum_j \Omega_\alpha(\theta_{1, \cdot, j})\): **첫 번째 층에만** 적용. 변수 선택 유도.

### Sparse Group Lasso 페널티

$$\Omega_\alpha(w) = (1 - \alpha)\|w\|_1 + \alpha\|w\|_2$$

여기서 \(\alpha \in [0, 1]\)는 그룹 수준과 개별 수준의 희소성을 조절하는 혼합 파라미터이다:

- **Group Lasso 부분** (\(\alpha\|w\|_2\)): \(\theta_{1, \cdot, j}\) 전체를 그룹으로 묶어 0으로 수축. 그룹 전체가 0이면 입력 변수 \(x_j\)가 선택에서 제외된다. 이는 변수 선택을 유도한다.
- **Lasso 부분** (\((1-\alpha)\|w\|_1\)): 그룹 내 개별 가중치의 추가 희소성. 선택된 변수 내에서도 불필요한 연결을 제거한다.

\(\alpha = 1\)이면 순수 Group Lasso, \(\alpha = 0\)이면 순수 Lasso가 된다.

## GIST 최적화 알고리즘

Sparse Group Lasso 페널티는 비미분 가능(non-differentiable)하므로, **일반화된 경사 하강법**(Generalized Iterative Shrinkage-Thresholding, GIST)을 사용한다.

### 알고리즘 구조

각 반복에서 다음 세 단계를 수행한다:

**Step 1**: 매끄러운 부분(경험적 손실 + Ridge)에 대한 경사 하강:

$$\tilde{\eta}^{(t)} = \eta^{(t)} - s \nabla_\eta \left[\frac{1}{n}\sum_{i=1}^{n} \ell(y_i, f_{\eta^{(t)}}(x_i)) + \lambda_0 \sum_a \|\theta_a^{(t)}\|_2^2\right]$$

여기서 \(s\)는 스텝 크기이다.

**Step 2**: Lasso 부분에 대한 soft-thresholding (원소별):

$$\hat{w}_k = \text{sign}(\tilde{w}_k) \max(|\tilde{w}_k| - s\lambda(1-\alpha), 0)$$

**Step 3**: Group Lasso 부분에 대한 soft-scaling (그룹별):

$$\theta_{1, \cdot, j}^{(t+1)} = \hat{w}_j \cdot \max\left(1 - \frac{s\lambda\alpha}{\|\hat{w}_j\|_2}, 0\right)$$

이 연산은 그룹 벡터 \(\hat{w}_j\)의 norm이 임계값 \(s\lambda\alpha\) 이하이면 전체 그룹을 0으로 수축시킨다.

### 단조 라인 탐색

수렴을 보장하기 위해 **단조 라인 탐색**(monotone line search)을 적용한다. 스텝 크기 \(s\)를 다음 조건을 만족할 때까지 축소한다:

$$F(\eta^{(t+1)}) \leq Q_s(\eta^{(t+1)}, \eta^{(t)})$$

여기서 \(F\)는 전체 목적함수, \(Q_s\)는 2차 근사 상한이다.

## 이론적 보장

### Oracle Inequality

참 함수 \(f^*\)가 \(s = |S|\)개 변수만 사용하는 희소 신경망으로 근사 가능하면, SPINN 추정량 \(\hat{f}\)에 대해:

$$E[\ell(\hat{f})] - \inf_{f \in \mathcal{F}} E[\ell(f)] = O_p\left(n^{-1} s^{5/2} \log p\right)$$

여기서 \(\mathcal{F}\)는 신경망 함수 공간, \(S\)는 활성 변수 집합이다.

**핵심**: 초과 위험이 \(p\)에 대해 **\(\log p\)로만 증가**한다. 이는 \(p\)가 수만에서 수십만이어도 SPINN이 적용 가능함을 의미한다. \(s\)에 대한 의존이 \(s^{5/2}\)인 것은 비선형 상호작용 모델링의 대가이며, Lasso의 \(s\)나 SpAM의 \(s\)보다 크지만 복잡한 함수 근사의 이점으로 상쇄된다.

### 불필요 변수 가중치의 수렴

불필요한 변수 \(j \notin S\)에 연결된 가중치:

$$\|\theta_{1, \cdot, j}\|_2 \xrightarrow{p} 0$$

즉, 불필요한 입력 변수의 가중치가 확률적으로 0에 수렴한다. 이는 신경망에서 **변수 선택의 일치성**을 보장하는 최초의 이론적 결과이다.

### Sieve 추정 관점

SPINN은 **sieve estimation** 관점에서도 해석할 수 있다. 표본 크기 \(n\)이 증가함에 따라 은닉 유닛 수와 층 수를 함께 증가시키면, 신경망 함수 공간이 점점 확대되어 참 함수를 근사한다. 구체적으로, 은닉 유닛 수 \(h = O(n^{d/(2+d)})\)로 설정하면 비모수적 최적 수렴률에 근접한다.

## 기존 방법과의 이론적 비교

| 방법 | 초과 위험 수렴률 | 상호작용 모델링 | 고차원 적용 |
|------|-----------------|---------------|-----------|
| Lasso | \(O(s \log p / n)\) | 불가 (선형) | 가능 |
| SpAM | \(O(s n^{-2/3})\) | 불가 (가법) | 가능 |
| SPINN | \(O(s^{5/2} \log p / n)\) | **가능** | 가능 |
| Random Forest | 이론적 보장 부족 | 가능 | 취약 |

SPINN은 상호작용을 모델링하면서도 고차원에서 \(\log p\) 스케일링을 달성하는 유일한 방법이다. Lasso와 SpAM은 각각 선형성과 가법성이라는 강한 구조적 가정을 필요로 하는 반면, SPINN은 이러한 가정 없이도 이론적 보장을 제공한다.

## 결론

SPINN은 첫 번째 층에 Sparse Group Lasso를 적용하여 고차원 비모수 회귀에서 변수 선택과 복잡한 함수 추정을 동시에 수행한다. Oracle inequality에 의한 \(O_p(n^{-1} s^{5/2} \log p)\) 수렴률은 고차원에서도 실용적 적용이 가능함을 보장하며, GIST 알고리즘은 비미분 가능 페널티의 효율적 최적화를 가능하게 한다.

## Reference

Feng, J. & Simon, N. (2019). Sparse-Input Neural Networks for High-dimensional Nonparametric Regression and Classification. *arXiv:1711.07592*.
