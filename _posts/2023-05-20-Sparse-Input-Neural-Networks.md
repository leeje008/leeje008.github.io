---
layout: post
title: "[Paper Review] Sparse-Input Neural Networks for High-dimensional Nonparametric Regression"
categories: [Paper Review]
tags: [paper-review, deep-learning, variable-selection, high-dimensional, statistics]
math: true
---

## Introduction

고차원($p \gg n$) 비모수 회귀에서 신경망은 일반적으로 학습 표본이 부족하여 사용되지 않는다. Feng & Simon (2019)은 **첫 번째 층에 Sparse Group Lasso 페널티**를 부과하여 소수의 입력 변수만 선택하는 Sparse-Input Neural Network(SPINN)을 제안하고, 초과 위험(excess risk)이 $p$의 **로그에만 비례**하여 증가함을 증명한다.

---

## 문제 설정과 동기

### 기존 방법의 한계

| 방법 | 장점 | 한계 |
|------|------|------|
| Lasso | $p \gg n$에서 작동 | 선형 관계만 모델링 |
| SpAM (Sparse Additive Models) | 비모수적, 변수 선택 | 가법 구조만, 상호작용 불가 |
| Random Forest | 상호작용 포착 | 이론적 보장 부족, 고차원 취약 |
| 일반 Neural Network | 복잡한 상호작용 근사 | $p \gg n$에서 과적합 |

핵심 관찰: 신경망은 상대적으로 **적은 파라미터**로 다변량 상호작용 함수를 근사할 수 있다 (Barron, 1993). 다항식이나 스플라인은 상호작용 차수에 대해 지수적으로 항이 증가하지만, 신경망은 은닉 유닛 수에 선형적으로 증가한다.

---

## Sparse-Input Neural Networks (SPINN)

### 정형화

첫 번째 층의 가중치를 $W^{(1)} \in \mathbb{R}^{h \times p}$라 하자. $j$번째 입력 변수에 연결된 가중치 벡터를 $W^{(1)}_{\cdot j} \in \mathbb{R}^h$로 표기하면:

$$
\hat{\theta} = \arg\min_{\theta} \frac{1}{n}\sum_{i=1}^{n} \ell(y_i, f_\theta(x_i)) + \lambda_1 \sum_{j=1}^{p} \|W^{(1)}_{\cdot j}\|_2 + \lambda_2 \|W^{(1)}\|_1
$$

- **Group Lasso 항** ($\|W^{(1)}_{\cdot j}\|_2$): 같은 입력 노드에 연결된 가중치를 **그룹으로 묶어** 0으로 수축. 그룹 전체가 0이면 해당 입력 변수가 선택에서 제외된다.

- **$L_1$ 항** ($\|W^{(1)}\|_1$): 그룹 내 개별 가중치의 추가 희소성.

이 두 항의 결합이 **Sparse Group Lasso** (Simon et al., 2013) 페널티이다.

---

## 이론적 보장

### 정리 (Oracle Inequality)

참 함수 $f^*$가 $s$개 변수만 사용하는 희소 신경망으로 근사 가능하면, SPINN 추정량 $\hat{f}$에 대해:

$$
E[\ell(\hat{f})] - \inf_{f \in \mathcal{F}} E[\ell(f)] = O_p\left(n^{-1} s^{5/2} \log p\right)
$$

여기서 $\mathcal{F}$는 신경망 함수 공간이다.

**핵심**: 초과 위험이 $p$에 대해 **$\log p$로만 증가**한다. 이는 $p$가 수만~수십만이어도 SPINN이 적용 가능함을 의미한다.

### 정리 (불필요 변수 가중치의 수렴)

불필요한 변수 $j \notin S$에 연결된 가중치:

$$
\|W^{(1)}_{\cdot j}\|_2 \xrightarrow{p} 0
$$

즉, 불필요한 입력 변수의 가중치가 확률적으로 0에 수렴한다. 이는 신경망에 대한 **최초의 불필요 파라미터 수축률(shrinkage rate) 이론적 결과**이다.

---

## 최적화 알고리즘

Sparse Group Lasso 페널티는 비미분 가능하므로 **일반화된 경사 하강법(proximal gradient descent)**을 사용:

$$
\theta^{(t+1)} = \text{prox}_{\eta\Omega}\left(\theta^{(t)} - \eta \nabla \hat{R}(\theta^{(t)})\right)
$$

Proximal operator는 Group Lasso 부분에 대해 그룹별 soft-thresholding을, $L_1$ 부분에 대해 원소별 soft-thresholding을 순차 적용한다.

---

## 기존 방법과의 이론적 비교

| 방법 | 수렴률 | 상호작용 | 고차원 |
|------|--------|---------|--------|
| Lasso | $O(s \log p / n)$ | 불가 (선형) | 가능 |
| SpAM | $O(s n^{-2/3})$ | 불가 (가법) | 가능 |
| SPINN | $O(s^{5/2} \log p / n)$ | **가능** | 가능 |

SPINN은 상호작용을 모델링하면서도 고차원에서 $\log p$ 스케일링을 달성하는 유일한 방법이다. 대신 $s$에 대한 의존이 $s^{5/2}$로 Lasso보다 크지만, 이는 비선형/상호작용 모델링의 대가이다.

---

## Reference

- Feng, J. & Simon, N. "Sparse-Input Neural Networks for High-dimensional Nonparametric Regression and Classification." *arXiv:1711.07592*, 2019.
