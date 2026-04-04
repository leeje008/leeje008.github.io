---
layout: post
title: "[Paper Review] KTO: Model Alignment as Prospect Theoretic Optimization"
categories: [Paper Review]
tags: [paper-review, preference-learning, llm-alignment, prospect-theory]
math: true
---

## Introduction

Ethayarajh et al. (2024)의 KTO(Kahneman-Tversky Optimization)는 대규모 언어 모델(LLM) 정렬(alignment)에서 기존 DPO가 요구하는 **쌍별 선호 데이터**(chosen vs. rejected 쌍)의 필요성을 제거한 방법이다. 현실에서 선호 데이터는 대부분 "좋다/나쁘다"라는 이진(pointwise) 피드백으로 존재하며, 쌍을 구성하는 것은 비용이 크다. KTO는 행동경제학의 **전망 이론(Prospect Theory)**을 도입하여, 인간이 손실에 대해 이득보다 더 강하게 반응하는 성질을 손실 함수에 반영한다.

---

## Prospect Theory와 가치 함수

Kahneman & Tversky (1979)의 전망 이론에 따르면, 인간은 절대적 효용이 아니라 **기준점(reference point) 대비 상대적 변화**에 반응하며, 손실 영역에서 이득 영역보다 더 가파른 기울기를 가진다. KTO는 이를 다음의 가치 함수로 모델링한다:

$$
v(z;\, \lambda,\, \alpha,\, z_0) = \begin{cases} (z - z_0)^{\alpha} & \text{if } z \geq z_0 \\ -\lambda \cdot (z_0 - z)^{\alpha} & \text{if } z < z_0 \end{cases}
$$

여기서 $$z_0$$는 기준점, $$\alpha \in (0, 1]$$는 민감도 파라미터, $$\lambda > 1$$는 **손실 회피 계수(loss aversion coefficient)**이다. $$\lambda > 1$$이므로 동일한 크기의 변화에 대해 손실이 이득보다 더 큰 심리적 영향을 미친다.

---

## Human-Aware Losses (HALOs)

저자들은 인간의 인지 편향을 반영하는 손실 함수 계열을 **HALO(Human-Aware Loss)**로 정의한다.

**정의 (HALO).** 손실 함수 $$f$$가 다음을 만족하면 human-aware loss이다: 각 입력-출력 쌍 $$(x, y)$$에 대해 $$a_{x,y} \in \{-1, +1\}$$가 존재하여, $$f$$가 Kahneman-Tversky 가치 함수의 구조를 따르는 특정 조건들을 충족한다.

### Theorem 3.5

DPO와 PPO-Clip 모두 HALO의 특수한 경우임을 보인다. DPO는 $$\alpha = 1$$이고 손실 회피가 대칭인 경우에 해당하며, PPO-Clip은 클리핑 영역 내에서 HALO 구조를 만족한다. 이는 기존 정렬 방법들이 암묵적으로 전망 이론의 구조를 내포하고 있음을 시사한다.

---

## KTO 손실 함수

### 암묵적 보상

정책 $$\pi_\theta$$와 참조 정책 $$\pi_{\text{ref}}$$ 간의 암묵적 보상을 다음과 같이 정의한다:

$$
r_\theta(x, y) = \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}
$$

### 기준점 $$z_0$$

기준점은 참조 정책 대비 현재 정책의 평균적 KL 발산으로 설정한다:

$$
z_0 = \text{KL}\!\left(\pi_\theta(y' \mid x) \;\|\; \pi_{\text{ref}}(y' \mid x)\right)
$$

실제 구현에서는 배치 내 샘플들로 $$z_0$$를 추정한다.

### KTO 손실

$$
\mathcal{L}_{\text{KTO}}(\pi_\theta, \pi_{\text{ref}}) = \mathbb{E}_{x, y}\!\left[\lambda_y \cdot v(x, y)\right]
$$

구체적으로, desirable 출력($$y \sim y_{\text{desirable}}$$)과 undesirable 출력($$y \sim y_{\text{undesirable}}$$)에 대해:

$$
v(x, y) = \begin{cases} \sigma\!\left(\beta\left(r_\theta(x, y) - z_0\right)\right) & \text{if } y \text{ is desirable} \\ -\sigma\!\left(\beta\left(z_0 - r_\theta(x, y)\right)\right) & \text{if } y \text{ is undesirable} \end{cases}
$$

여기서 $$\sigma$$는 시그모이드 함수이고 $$\beta$$는 KL 제약의 강도를 조절한다.

---

## 하이퍼파라미터와 손실 회피

KTO의 주요 하이퍼파라미터는 $$\beta$$, $$\lambda_D$$(desirable 가중치), $$\lambda_U$$(undesirable 가중치)이다. 저자들은 **손실 회피 비율(loss aversion ratio)**을 다음과 같이 설정할 것을 권장한다:

$$
\frac{\lambda_D \cdot n_D}{\lambda_U \cdot n_U} \in \left[1,\, \frac{3}{2}\right]
$$

여기서 $$n_D$$, $$n_U$$는 각각 desirable, undesirable 샘플의 수이다. 이 비율이 1보다 크면 모델이 나쁜 출력을 피하는 데 상대적으로 더 집중하며, 이는 전망 이론의 손실 회피와 일치한다.

---

## 실험 결과

| 모델 | 방법 | GPT-4 Win Rate (%) |
|------|------|---------------------|
| Pythia 1.4B | DPO | 27.6 |
| Pythia 1.4B | KTO | **31.2** |
| Llama-7B | DPO | 42.1 |
| Llama-7B | KTO | **44.5** |
| Llama-13B | DPO | 47.8 |
| Llama-13B | KTO | **49.3** |

KTO는 쌍별 선호 데이터 없이도 DPO와 동등하거나 우수한 성능을 달성한다. 특히 데이터 효율성 면에서 KTO는 desirable/undesirable 비율이 불균형한 경우에도 강건한 성능을 보인다. $$n_D : n_U = 9:1$$인 극단적 불균형 상황에서도 $$\lambda_D, \lambda_U$$ 조정만으로 안정적 학습이 가능하다.

---

## 의의와 한계

KTO의 핵심 기여는 다음과 같다: (1) 쌍별 선호 데이터가 불필요하므로 데이터 수집 비용이 크게 절감되며, (2) 전망 이론이라는 이론적 기반을 통해 정렬 손실 함수 설계에 대한 통합적 프레임워크(HALO)를 제시한다. 다만, 가치 함수의 파라미터($$\alpha$$, $$\lambda$$) 선택이 경험적이며, 매우 큰 모델(100B+ 규모)에서의 검증은 추가로 필요하다.

---

## Reference

- Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., & Kiela, D. (2024). KTO: Model Alignment as Prospect Theoretic Optimization. *Proceedings of the 41st International Conference on Machine Learning (ICML)*.
- Kahneman, D., & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk. *Econometrica*, 47(2), 263-292.
