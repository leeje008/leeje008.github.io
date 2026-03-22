---
layout: post
title: "[Paper Review] Orthogonal Deep Neural Networks"
categories: [Paper Review]
tags: [paper-review, deep-learning, regularization, generalization]
math: true
---

## Introduction

심층 신경망(DNN)은 과잉 매개변수화(over-parameterized)되어 있음에도 실용적으로 잘 일반화된다. Jia et al. (2019)은 이 현상을 **가중치 행렬의 특이값 스펙트럼** 관점에서 분석하고, 각 가중치 행렬이 직교(orthogonal)에 가까울수록 일반화 오차가 최소화됨을 증명한다. 이를 바탕으로 Stiefel manifold 위에서 최적화하는 Strict OrthDNN과, 계산 효율적인 근사법인 SVB(Singular Value Bounding)를 제안한다.

## Classification-Representation-Learning 문제

### 문제 정형화

DNN을 분류기 $$f$$와 특징 추출기 $$T$$로 분리한다. 전체 위험(risk)은:

$$R(f, T) = E_{(x,y) \sim P}\left[L(f(Tx), y)\right]$$

여기서 $$T: \mathbb{R}^d \to \mathbb{R}^k$$는 DNN의 은닉층이 수행하는 표현 학습(representation learning), $$f: \mathbb{R}^k \to \{1, \ldots, C\}$$는 마지막 층의 분류기이다.

학습 데이터 $$S_m = \{(x_i, y_i)\}_{i=1}^m$$에 대한 경험적 위험은:

$$R_m(f, T) = \frac{1}{m} \sum_{i=1}^{m} L(f(Tx_i), y_i)$$

**일반화 오차**(Generalization Error)는:

$$GE(f_{S_m}) = |R(f_{S_m}) - R_m(f_{S_m})|$$

## Local Isometry와 일반화 오차 경계

### $$\delta$$-Isometry의 정의

사상 $$T$$가 **$$\delta$$-등거리**(isometric)라 함은, 입력 공간의 거리 $$\rho_P$$와 출력 공간의 거리 $$\rho_Q$$ 사이에:

$$|\rho_Q(Tx, Tx') - \rho_P(x, x')| \leq \delta \quad \forall x, x' \in \mathcal{X}$$

$$\delta$$가 작을수록 $$T$$는 입력 공간의 기하학적 구조를 잘 보존한다.

### Theorem 2.2: Covering Number 기반 일반화 오차 경계

$$\mathcal{X}$$의 $$\gamma/2$$-covering number를 $$N_{\gamma/2}(\mathcal{X}, \rho)$$라 하면, 확률 $$1 - \eta$$ 이상으로:

$$GE(f_{S_m}) \leq \frac{2}{\gamma} + \sqrt{\frac{2 \ln N_{\gamma/2}(\mathcal{X}, \rho) + 2\ln(2/\eta)}{m}}$$

이 경계는 covering number가 작을수록, 즉 데이터 공간이 효과적으로 압축될수록 일반화가 잘 됨을 의미한다.

## DNN의 등거리 성질 분석

### Lemma 3.1: 선형 DNN의 $$\delta$$-Isometry

가중치 행렬 $$W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$$를 가진 $$L$$층 선형 DNN에서 전체 사상 $$T = W^{(L)} \cdots W^{(1)}$$에 대해:

$$\delta = \max\left(\left|\prod_{l=1}^{L} \sigma_{\max}(W^{(l)}) - 1\right|, \left|1 - \prod_{l=1}^{L} \sigma_{\min}(W^{(l)})\right|\right)$$

여기서 $$\sigma_{\max}, \sigma_{\min}$$은 각각 최대, 최소 특이값이다. $$\delta$$는 특이값 스펙트럼에 의해 완전히 결정되며, 모든 특이값이 1일 때 $$\delta = 0$$이 된다.

### Lemma 3.2: 비선형 DNN의 국소 선형 분할

ReLU 등의 구간선형(piecewise linear) 활성화 함수를 사용하는 DNN은 입력 공간을 **유한개의 선형 영역**(linear regions)으로 분할한다. 각 영역 $$\mathcal{R}_i$$에서 DNN은 아핀 사상 $$T_i(x) = A_i x + b_i$$로 작동하며, $$A_i$$는 해당 영역에서의 야코비안이다.

따라서 비선형 DNN은 **국소적으로(locally)** 선형 DNN과 동일한 분석이 적용된다.

### Lemma 3.3: Covering Ball의 지름

$$\gamma/2$$-covering ball의 지름은 가중치 행렬의 최대 특이값 곱에 반비례한다:

$$\gamma \propto \frac{1}{\prod_{l=1}^{L} \sigma_{\max}(W^{(l)})}$$

직관적으로, 특이값이 크면 입력 공간이 크게 확대(stretch)되어 더 많은 covering ball이 필요하고, 일반화 오차가 증가한다.

### Theorem 3.2: 주요 일반화 오차 경계

확률 $$1 - \eta$$ 이상으로:

$$GE(f_{S_m}) \leq O\left(\frac{1}{\sqrt{m}} \cdot \prod_{l=1}^{L} \sigma_{\max}(W^{(l)}) \cdot \sum_{l=1}^{L} \frac{\sqrt{h_l}}{\sigma_{\min}(W^{(l)})}\right)$$

여기서 $$h_l$$은 $$l$$번째 층의 은닉 유닛 수이다.

이 경계는 두 가지 측면에서 특이값에 민감하다:
- **Scale-sensitive**: $$\prod_l \sigma_{\max}(W^{(l)})$$에 비례. 특이값의 절대 크기가 클수록 경계가 커진다.
- **Range-sensitive**: $$\sigma_{\max}(W^{(l)}) / \sigma_{\min}(W^{(l)})$$ (조건수)에 의존. 특이값의 범위가 넓을수록 경계가 커진다.

### Lemma 3.5: 최적 조건 — 직교 가중치

일반화 오차 경계를 최소화하는 최적 조건은 각 층의 **모든 특이값이 동일**한 것이다:

$$\sigma_1(W^{(l)}) = \sigma_2(W^{(l)}) = \cdots = \sigma_{\min(n_l, n_{l-1})}(W^{(l)})$$

이 조건을 만족하는 가장 자연스러운 선택이 **직교 가중치 행렬**이다:
- 정방행렬인 경우: $$W^T W = I$$ (직교 행렬, 모든 특이값 = 1)
- 비정방행렬인 경우: $$W^T W = I$$ 또는 $$WW^T = I$$ (Stiefel manifold 위의 행렬)

## OrthDNN 알고리즘

### Strict OrthDNN

각 층의 가중치를 **Stiefel manifold** $$\mathcal{V}_{n_l, n_{l-1}} = \{W \in \mathbb{R}^{n_l \times n_{l-1}} : W^T W = I_{n_{l-1}}\}$$ 위에서 최적화한다. Cayley 변환 기반의 사영 알고리즘:

$$W^{(t+1)} = \left(I + \frac{\tau}{2}A\right)^{-1}\left(I - \frac{\tau}{2}A\right) W^{(t)}$$

여기서 $$A = GW^T - WG^T$$는 반대칭 행렬이고, $$G = \nabla_W \mathcal{L}$$은 유클리드 기울기, $$\tau$$는 학습률이다. 이 업데이트는 Stiefel manifold 위의 retraction으로, $$W^{(t+1)}$$이 직교 조건을 정확히 만족함을 보장한다.

**단점**: 매 단계마다 행렬 역산 $$(I + \frac{\tau}{2}A)^{-1}$$이 필요하여 계산 비용이 높다.

### Approximate OrthDNN via SVB (Singular Value Bounding)

Strict OrthDNN의 계산 비용을 줄이는 근사 방법이다. 매 $$k$$ 에폭마다 각 가중치 행렬에 대해:

**SVB 알고리즘**:
1. SVD 수행: $$W = U\Sigma V^T$$
2. 특이값을 구간 $$[1-\varepsilon, 1+\varepsilon]$$로 클리핑:

$$\sigma_i' = \text{clip}(\sigma_i, 1-\varepsilon, 1+\varepsilon) = \min(\max(\sigma_i, 1-\varepsilon), 1+\varepsilon)$$

3. 재구성: $$W' = U\Sigma' V^T$$ (여기서 $$\Sigma' = \text{diag}(\sigma_1', \ldots, \sigma_r')$$)

$$\varepsilon = 0$$이면 모든 특이값이 1로 고정되어 Strict OrthDNN과 동일하다. $$\varepsilon > 0$$은 직교 조건의 **완화(relaxation)**로, 특이값의 range를 $$[1-\varepsilon, 1+\varepsilon]$$로 제한하여 일반화 오차 경계를 간접적으로 최적화한다.

### Degenerate / Bounded Batch Normalization (DBN / BBN)

기존 Batch Normalization (BN)은 아핀 변환 $$\gamma \hat{z} + \beta$$를 포함하는데, 스케일 파라미터 $$\gamma$$가 자유롭게 변하면 SVB의 효과를 상쇄할 수 있다.

- **DBN (Degenerate BN)**: $$\gamma = 1$$, $$\beta = 0$$으로 고정. 즉 정규화만 수행.
- **BBN (Bounded BN)**: $$\gamma$$를 $$[\gamma_{\min}, \gamma_{\max}]$$로 바운딩. BN의 표현력을 유지하면서 특이값 제어와 호환.

실험에서 BBN이 DBN보다 더 좋은 성능을 보이며, 기존 BN과 비교해도 일반화 성능이 향상됨을 확인하였다.

## 실험 결과 요약

- **CIFAR-10/100**: SVB + BBN 적용 시 VGG, ResNet 등에서 기존 대비 0.5~1.5% 테스트 정확도 향상
- **ImageNet**: ResNet-18에서 SVB 적용 시 top-1 에러 0.4% 감소
- 특이값 스펙트럼 분석: SVB 적용 후 특이값이 1 근처에 집중되며, 학습 과정에서 특이값의 발산이 방지됨
- Strict OrthDNN은 성능은 유사하지만 계산 비용이 2~3배 높아 실용성이 떨어짐

## 결론

OrthDNN은 DNN의 일반화 오차를 가중치 행렬의 특이값 스펙트럼으로 분석하고, 직교 가중치가 최적임을 이론적으로 증명한다. 실용적 구현인 SVB는 주기적 특이값 클리핑으로 직교 조건을 근사하며, BBN과 결합하여 기존 아키텍처에 쉽게 적용할 수 있다.

## Reference

Jia, K., Li, S., Wen, Y., Liu, T. & Tao, D. (2019). Orthogonal Deep Neural Networks. *arXiv:1905.05929*.
