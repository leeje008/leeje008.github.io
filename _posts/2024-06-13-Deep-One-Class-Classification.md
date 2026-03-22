---
layout: post
title: "[Paper Review] Deep One Class Classification"
categories: [Paper Review]
tags: [paper-review, anomaly-detection, deep-learning]
math: true
---

## Introduction

이상 탐지(Anomaly Detection)는 데이터 내의 이상치를 식별하는 문제로, 지도 학습, 반지도 학습 또는 비지도 학습으로 접근할 수 있다. One-Class SVM이나 커널 밀도 추정과 같은 전통적인 이상 탐지 방법은 대규모 데이터셋을 처리할 때 어려움을 겪는다.

이 논문은 Deep Neural Network를 활용하여 기존 커널 기반 방법의 한계를 극복하는 Deep SVDD를 제안한다.

### 주요 표기법

- $$R$$: 초구(hypersphere)의 반지름
- $$c$$: 초구의 중심
- $$\mathcal{X}$$: 데이터 공간
- $$\varphi(\cdot)$$: 특성 매핑 또는 기저 함수
- $$\kappa(\cdot, \cdot)$$: 커널 함수
- $$\xi$$: 슬랙 변수
- $$\mathcal{F}_k$$: 커널로 색인화된 힐베르트 공간

---

## One-Class SVM

$$\mathcal{X} \subseteq \mathbb{R}^d$$, $$\kappa(\cdot, \cdot): \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$$를 PSD(양의 반정치) 행렬이라 하면, 목적 함수는:

$$
\min_{w, \rho, \xi} \frac{1}{2} \|w\|^2_{\mathcal{F}_k} - \rho + \frac{1}{\nu n} \sum_i \xi_i
$$

제약 조건:

$$
\langle w, \varphi_k(x_i) \rangle_{\mathcal{F}_k} \geq \rho - \xi_i, \quad \xi_i \geq 0
$$

이는 $$\mathcal{F}_k$$에서 **최대 마진 초평면**을 찾는 문제이다.

---

## SVDD (Support Vector Data Description)

목적 함수:

$$
\min_{R, c, \xi} R^2 + \frac{1}{\nu n} \sum_i \xi_i
$$

제약 조건:

$$
\|\varphi_k(x_i) - c\|^2_{\mathcal{F}_k} \leq R^2 + \xi_i, \quad \xi_i \geq 0
$$

이는 $$\mathcal{F}_k$$에서 **초구(hypersphere)**를 찾아 이상치를 탐지한다.

One-Class SVM과 SVDD 모두 데이터를 힐베르트 공간으로 매핑하지만, One-Class SVM은 초평면을 찾고 SVDD는 초구를 찾는다는 차이가 있다.

### 커널 기반 모델의 장단점

**장점:**
- 이차 프로그래밍으로 해결 가능
- 가우시안 커널 사용 시 일관된 밀도 추정 가능
- 극도로 고차원 데이터에 적용 가능 ($$n \ll p$$)

**단점:**
- 높은 계산 비용 (일반적으로 $$O(n^3)$$)
- 대규모 데이터셋에 적용 불가능

---

## Deep Approaches to Anomaly Detection

### Soft-boundary Deep SVDD

$$\varphi(\cdot; W)$$를 $$L$$개 계층의 특성 맵이라 하면 ($$W = \{W_1, W_2, \ldots, W_L\}$$), 목적 함수:

$$
\min_{R, W} R^2 + \frac{1}{\nu n} \sum_i \max\left(0, \|\varphi(x_i; W) - c\|^2 - R^2\right) + \frac{\lambda}{2} \sum_l \|W_l\|^2_F
$$

- 첫 번째 항: 초구 부피 최소화 (커널 SVDD와 유사)
- $$\nu$$ 조정으로 일부 데이터(이상치)를 경계 밖에 허용
- 세 번째 항: 프로베니우스 노름 정규화

### One-Class Deep SVDD

$$
\min_W \frac{1}{n} \sum_i \|\varphi(x_i; W) - c\|^2 + \frac{\lambda}{2} \sum_l \|W_l\|^2_F
$$

Soft-boundary와 달리 모든 데이터 포인트의 중심까지 **평균 거리를 최소화**하여 초구를 수축시킨다.

### 결정 함수

테스트 데이터 $$x \in \mathcal{X}$$와 학습된 파라미터 $$W^*$$, $$R^*$$에 대해:

$$
f(x) = \text{sign}(s(x) - R^*)
$$

여기서:

$$
s(x) = \|\varphi(x; W^*) - c\|^2
$$

### 최적화

- SGD 또는 Adam 같은 optimizer 사용
- 좌표 하강 알고리즘(coordinate descent) 활용
  1. $$R$$을 고정하고 $$W$$ 학습
  2. $$n$$ 에포크 후 $$W$$를 고정하고 $$R$$ 업데이트

---

## Reference

- Ruff, L., et al. "Deep One-Class Classification." *ICML 2018*.
