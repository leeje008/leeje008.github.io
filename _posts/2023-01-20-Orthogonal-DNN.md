---
layout: post
title: "[Paper Review] Orthogonal Deep Neural Networks - 직교 정규화와 일반화 오차"
categories: [Paper Review]
tags: [paper-review, deep-learning, regularization, generalization]
math: true
---

## Introduction

심층 신경망(DNN)은 과잉 매개변수화(over-parameterized)되어 있지만 실용적으로 잘 작동한다. Jia et al. (2019)은 이 현상을 **가중치 행렬의 특이값 스펙트럼** 관점에서 분석하고, 각 가중치 행렬이 직교(orthogonal)에 가까울수록 일반화 오차가 최소화됨을 증명한다.

---

## Local Isometry와 일반화 오차

### DNN의 Local Isometry 성질

$f_W: \mathbb{R}^d \to \mathbb{R}^c$를 $L$개 층의 DNN, $W = \{W^{(l)}\}_{l=1}^{L}$을 가중치 행렬이라 하자.

**정리 (Local Isometry)**: 실제적으로 의미 있는 데이터 분포(instance-wise variation space)에서, DNN은 입력 공간의 국소적 선형 분할(linear partition) 위에서 **국소적 등거리 사상(locally isometric)**이다.

각 분할 영역의 반지름은 가중치 행렬의 spectral norm $\|W^{(l)}\|_2$에 의해 제어된다.

### 일반화 오차 경계

**정리**: 확률 $1 - \delta$ 이상으로:

$$
R(f_W) - \hat{R}(f_W) \leq O\left(\frac{M}{\sqrt{n}} \cdot \prod_{l=1}^{L} \|W^{(l)}\|_2 \cdot \sum_{l=1}^{L} \frac{\sqrt{r_l}}{\|W^{(l)}\|_2}\right)
$$

여기서 $r_l$은 $W^{(l)}$의 특이값 비율(range)에 의존하는 항이다.

이 경계는 특이값 스펙트럼에 대해 **scale-sensitive** (스펙트럼의 절대 크기)이고 **range-sensitive** (최대-최소 특이값 비율)이다.

### 최적 조건

**따름정리**: 일반화 오차 경계는 각 $W^{(l)}$의 특이값이 **모두 동일**할 때 최소화된다. 이 조건을 만족하는 가장 자연스러운 선택이 **직교 가중치 행렬** ($W^TW = I$ 또는 $WW^T = I$)이다.

---

## OrthDNN 알고리즘

### Strict OrthDNN

각 층의 가중치를 Stiefel manifold $\{W : W^TW = I\}$ 위에서 최적화한다. Cayley 변환 기반의 사영 알고리즘을 사용:

$$
W^{(t+1)} = \left(I + \frac{\tau}{2}A\right)^{-1}\left(I - \frac{\tau}{2}A\right) W^{(t)}
$$

여기서 $A = GW^T - WG^T$, $G$는 유클리드 기울기.

### Singular Value Bounding (SVB)

Strict OrthDNN의 계산 비용을 줄이는 근사 방법:

매 $k$ 에폭마다 각 가중치 행렬의 SVD $W = U\Sigma V^T$를 수행하고, 특이값을 구간 $[\sigma_{min}, \sigma_{max}]$로 클리핑:

$$
\sigma_i' = \text{clip}(\sigma_i, \sigma_{min}, \sigma_{max}), \quad W' = U\Sigma' V^T
$$

이는 직교 조건의 **완화(relaxation)**로, 특이값의 range를 제한하여 일반화 오차 경계를 간접적으로 최적화한다.

### Bounded Batch Normalization (BBN)

기존 Batch Normalization은 특이값 스펙트럼을 왜곡할 수 있다. BBN은 BN의 스케일 파라미터 $\gamma$를 $[\gamma_{min}, \gamma_{max}]$로 바운딩하여 OrthDNN과 호환되게 한다.

---

## Reference

- Jia, K., Li, S., Wen, Y., Liu, T. & Tao, D. "Orthogonal Deep Neural Networks." *IEEE TPAMI*, 2019.
