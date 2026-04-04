---
layout: post
title: "[Paper Review] Proximal Policy Optimization Algorithms (PPO)"
categories: [Paper Review]
tags: [paper-review, reinforcement-learning, policy-gradient, optimization]
math: true
---

## Introduction

강화학습에서 정책 경사(policy gradient) 방법은 높은 표현력과 범용성을 갖추고 있으나, 샘플 효율성이 낮고 학습이 불안정하다는 한계가 있다. Trust Region Policy Optimization(TRPO)은 정책 업데이트의 크기를 제한하여 안정성을 확보하였지만, 구현이 복잡하고 2차 최적화가 필요하다는 단점이 있다. 이 논문은 TRPO의 안정성을 유지하면서도 1차 최적화만으로 구현 가능한 **Proximal Policy Optimization(PPO)**을 제안한다.

---

## Policy Gradient 배경

정책 경사 방법의 핵심 추정식은 다음과 같다:

$$
\hat{g} = \hat{\mathbb{E}}_t \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \hat{A}_t \right]
$$

여기서 $$\hat{A}_t$$는 시점 $$t$$에서의 이점 함수(advantage function) 추정치이다. 이를 기반으로 하는 서로게이트 목적 함수는 다음과 같다:

$$
L^{CPI}(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} \hat{A}_t \right] = \hat{\mathbb{E}}_t \left[ r_t(\theta) \hat{A}_t \right]
$$

여기서 $$r_t(\theta) = \pi_\theta(a_t | s_t) / \pi_{\theta_{\text{old}}}(a_t | s_t)$$는 확률 비율(probability ratio)이다.

---

## TRPO의 제약 조건

TRPO는 위 서로게이트 목적 함수를 KL 발산 제약 하에서 최대화한다:

$$
\max_\theta \; \hat{\mathbb{E}}_t \left[ r_t(\theta) \hat{A}_t \right] \quad \text{s.t.} \quad \hat{\mathbb{E}}_t \left[ \text{KL}[\pi_{\theta_{\text{old}}}(\cdot | s_t) \| \pi_\theta(\cdot | s_t)] \right] \leq \delta
$$

이 방법은 단조 개선(monotonic improvement)을 보장하지만, 컨쥬게이트 경사법 등 복잡한 2차 최적화가 필요하다.

---

## Clipped Surrogate Objective

PPO의 핵심 기여는 클리핑된 서로게이트 목적 함수이다:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

하이퍼파라미터 $$\epsilon$$은 일반적으로 0.1 또는 0.2로 설정된다. 이점이 양수일 때 $$r_t(\theta)$$가 $$1 + \epsilon$$을 초과하면 클리핑되어 과도한 정책 변화를 방지한다. 이점이 음수일 때는 $$1 - \epsilon$$ 미만으로 클리핑된다. 이로써 TRPO와 유사한 신뢰 영역 효과를 1차 최적화만으로 달성한다.

---

## Adaptive KL Penalty

대안적 방법으로, KL 발산에 대한 적응형 페널티 계수를 사용할 수도 있다:

$$
L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t \left[ r_t(\theta) \hat{A}_t - \beta \cdot \text{KL}[\pi_{\theta_{\text{old}}}(\cdot | s_t) \| \pi_\theta(\cdot | s_t)] \right]
$$

각 업데이트 후 실제 KL 발산 $$d$$를 측정하여 $$d < d_{\text{targ}} / 1.5$$이면 $$\beta \leftarrow \beta / 2$$, $$d > d_{\text{targ}} \times 1.5$$이면 $$\beta \leftarrow 2\beta$$로 조정한다. 실험 결과 클리핑 방식이 더 우수한 성능을 보였다.

---

## Actor-Critic 알고리즘과 GAE

PPO는 가치 함수와 정책을 공유 네트워크로 학습하며, 결합된 목적 함수는 다음과 같다:

$$
L^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t \left[ L^{CLIP}_t(\theta) - c_1 L^{VF}_t(\theta) + c_2 S[\pi_\theta](s_t) \right]
$$

여기서 $$L^{VF}_t = (V_\theta(s_t) - V_t^{\text{targ}})^2$$은 가치 함수 손실, $$S[\pi_\theta]$$는 엔트로피 보너스이다. $$c_1 = 0.5$$, $$c_2 = 0.01$$이 일반적으로 사용된다.

이점 함수는 Generalized Advantage Estimation(GAE)으로 추정한다:

$$
\hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \cdots
$$

여기서 $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$는 TD 잔차이다. $$\lambda = 0.95$$, $$\gamma = 0.99$$가 표준 설정이다.

---

## 실험 결과

| 환경 | PPO (Clip) | TRPO | A2C | CEM |
|------|-----------|------|-----|-----|
| MuJoCo 연속 제어 | **최고** | 준수 | 보통 | 낮음 |
| Atari 게임 | **최고** | - | 준수 | - |

PPO는 7개 MuJoCo 환경 중 6개에서 최고 성능을 달성하였으며, Atari 49개 게임에서도 ACER 대비 우수한 성능을 보였다. 특히 구현의 단순성과 하이퍼파라미터 민감도 측면에서 TRPO 대비 큰 장점을 가진다.

---

## Conclusion

PPO는 클리핑된 서로게이트 목적 함수를 통해 TRPO의 안정적 학습을 1차 최적화로 달성하는 실용적 알고리즘이다. 미니배치 SGD와 다중 에폭 업데이트가 가능하여 샘플 효율성이 높고, 연속 및 이산 행동 공간 모두에서 우수한 성능을 보인다. OpenAI의 기본 강화학습 알고리즘으로 채택되어 이후 RLHF 등 다양한 응용에 핵심적으로 활용되고 있다.

---

## Reference

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. "Proximal Policy Optimization Algorithms." *arXiv preprint arXiv:1707.06347, 2017*.
