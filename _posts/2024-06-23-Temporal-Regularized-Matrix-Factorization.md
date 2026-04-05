---
layout: post
title: "[Paper Review] Temporal Regularized Matrix Factorization (TRMF)"
categories: [Paper Review]
tags: [paper-review, time-series, matrix-factorization]
math: true
---

## Introduction

시계열 데이터 분석에서 두 가지 주요 도전 과제가 있다:

1. 수천 개의 센서에서 매시간 수년간 수집되어 **대규모**이다
2. 센서 오작동, 폐색 또는 인적 오류로 인한 **누락값**을 포함한다

AR(AutoRegressive)이나 DLM(Dynamic Linear Model) 같은 전통적 접근은 저차원 시계열에는 적합하지만, 대규모 시계열 데이터나 누락값이 있는 경우 처리가 어렵다. 이 논문은 **행렬 인수분해(Matrix Factorization)**를 사용하여 이러한 문제를 해결하는 TRMF를 제안한다.

---

## Motivation

행렬 $Y \in \mathbb{R}^{n \times T}$를 정의하며, $Y_{i,t}$는 $i$번째 시계열의 $t$번째 시점을 나타낸다.

행렬 인수분해 프레임워크에서 $Y_{i,t}$는 $f_i^T x_t$의 내적으로 추정된다.

### 목적 함수

$$
\min_{F, X} \sum_{(i,t) \in \Omega} (Y_{i,t} - f_i^T x_t)^2 + \lambda_f R_F(F) + \lambda_x R_x(X)
$$

여기서 $\Omega$는 관찰된 항목의 집합이다.

Frobenius norm을 사용하면 데이터가 i.i.d.라고 가정하므로 **시간 의존성 정보를 포착할 수 없다**.

### 그래프 기반 정규화

시간 의존성을 적용하기 위해 지연 집합 $\mathcal{L}$과 가중치 벡터 $w$를 정의한다:

$$
G(X | G, \eta) = \frac{1}{2} \sum_{l \in \mathcal{L}} \sum_{t: t > l} w_l (x_t - x_{t-l})^2 + \frac{\eta}{2} \sum_t \|x_t\|^2
$$

이 접근 방식의 단점:
1. 두 시점 간의 **음의 상관 의존성**을 포착할 수 없다
2. 시간 지연 $\mathcal{L}$을 추론해야 하므로 단순한 지연 집합만 사용되어 실패로 이어진다

---

## TRMF

위의 제한을 해결하기 위해 다음 모델을 고려한다:

$$
x_t = M_\Theta(\{x_{t-l}: l \in \mathcal{L}\}) + \varepsilon_t
$$

여기서 $\varepsilon_t$는 가우시안 노이즈 벡터이고, $M_\Theta$는 $\Theta$, $\mathcal{L}$에 의해 매개변수화된 시계열 모델이다.

전통적인 통계 접근을 취하면, $T_M(X | \Theta)$는 $M_\Theta$가 주어졌을 때의 음의 로그 우도 함수이다:

$$
T_M(X | \Theta) = -\log P(x_1, \ldots, x_T | \Theta)
$$

### TRMF 목적 함수

$$
\min_{F, X, \Theta} \sum_{(i,t) \in \Omega} (Y_{i,t} - f_i^T x_t)^2 + \lambda_f R_F(F) + \lambda_x T_M(X | \Theta) + \lambda_\theta R_\theta(\Theta)
$$

위의 손실 함수는 **교대 최소화 프로세스**로 최적화할 수 있다.

### TRMF의 응용

1. **시계열 예측**: 잠재 임베딩 $\{x_t: 1, \ldots, T\}$에 대해 $M_\Theta$를 찾으면 $y_t = Fx_t$를 사용하여 시계열 값을 예측
2. **누락값 대체**: $f_i^T x_t$를 사용하여 누락값을 대체

---

## TRMF with AR

AR 모델을 적용하면 시간 정규화기는 다음과 같다:

$$
T_{AR}(X | \mathcal{L}, W, \eta) = \frac{1}{2} \sum_{t=m}^{T} \left\|x_t - \sum_{l \in \mathcal{L}} W^{(l)} x_{t-l}\right\|^2 + \frac{\eta}{2} \sum_t \|x_t\|^2
$$

여기서 $W = \{W^{(l)} \in \mathbb{R}^{k \times k}: l \in \mathcal{L}\}$이다.

$W^{(l)}$의 모든 원소를 학습하면 $|\mathcal{L}| k^2$개의 변수가 필요하여 과적합 위험이 있다. 이를 방지하기 위해 **$W^{(l)}$의 대각선 요소만 사용**한다:

$$
T_{AR}(\bar{x} | \mathcal{L}, \bar{w}, \eta) = \frac{1}{2} \sum_{t=m}^{T} \left\|x_t - \sum_{l \in \mathcal{L}} w_l x_{t-l}\right\|^2 + \frac{\eta}{2} \sum_t \|\bar{x}\|^2
$$

$W^{(l)}$이 대각행렬이더라도, TRMF는 $\{f_i\}$를 통해 시계열 간의 상관관계를 포착할 수 있다. 또한 다른 방법과 달리 $\mathcal{L}$의 선택이 더 유연하다.

---

## Reference

- Yu, H.-F., Rao, N., & Dhillon, I. S. "Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction." *NeurIPS 2016*.
