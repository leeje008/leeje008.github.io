---
layout: post
title: "[Paper Review] A Contextual-Bandit Approach to Personalized News Article Recommendation"
categories: [Paper Review]
tags: [paper-review, reinforcement-learning, contextual-bandits, exploration-exploitation]
math: true
---

## Introduction

웹 서비스에서 개인화 추천은 사용자의 문맥(context) 정보를 활용하여 최적의 콘텐츠를 선택하는 문제로 정의할 수 있다. 기존의 추천 시스템은 대부분 탐색(exploration)과 활용(exploitation)의 균형을 체계적으로 다루지 못한다. 이 논문은 **K-armed contextual bandit** 프레임워크를 도입하여 개인화 뉴스 추천 문제를 정식화하고, **LinUCB** 알고리즘을 제안한다. Yahoo! Front Page에서의 대규모 실험을 통해 기존 방법 대비 CTR 12.5% 향상을 달성하였다.

---

## Contextual Bandit 정의

각 시점 $$t = 1, 2, \ldots, T$$에서 에이전트는 다음을 수행한다:

1. 문맥 $$x_{t,a}$$를 포함한 $$K$$개 팔(arm)의 집합 $$\mathcal{A}_t$$를 관찰
2. 이전 경험에 기반하여 팔 $$a_t \in \mathcal{A}_t$$를 선택
3. 선택한 팔에 대한 보상 $$r_{t, a_t}$$를 관찰

목표는 총 $$T$$ 시점 동안의 누적 보상을 최대화하는 것이며, 리그렛은 다음과 같이 정의된다:

$$
R_A(T) = \mathbb{E}\left[\sum_{t=1}^{T} r_{t, a_t^*}\right] - \mathbb{E}\left[\sum_{t=1}^{T} r_{t, a_t}\right]
$$

여기서 $$a_t^*$$는 시점 $$t$$에서의 최적 팔이다.

---

## 기존 접근법의 한계

### $$\epsilon$$-Greedy

확률 $$\epsilon$$으로 무작위 팔을 선택하고, $$1 - \epsilon$$으로 현재 최적 팔을 선택한다. 단순하지만 **문맥 정보를 활용한 불확실성 추정이 불가능**하여 비효율적 탐색을 수행한다.

### UCB (Upper Confidence Bound)

각 팔의 보상 상한을 추정하여 불확실한 팔에 대한 탐색을 유도하지만, 전통적 UCB는 문맥 정보를 고려하지 않는다.

---

## LinUCB: Disjoint 모델

각 팔 $$a$$에 대해 보상의 기대값이 문맥 벡터와 선형 관계에 있다고 가정한다:

$$
\mathbb{E}[r_{t,a} | x_{t,a}] = x_{t,a}^T \theta_a^*
$$

릿지 회귀(ridge regression)를 통해 $$\theta_a$$를 추정한다:

$$
\hat{\theta}_a = (D_a^T D_a + I_d)^{-1} D_a^T c_a = A_a^{-1} b_a
$$

여기서 $$D_a$$는 문맥 벡터의 설계 행렬, $$c_a$$는 보상 벡터이다. $$A_a = D_a^T D_a + I_d$$, $$b_a = D_a^T c_a$$로 정의한다.

신뢰 상한 기반의 팔 선택 규칙은 다음과 같다:

$$
a_t = \arg\max_{a \in \mathcal{A}_t} \left( x_{t,a}^T \hat{\theta}_a + \alpha \sqrt{x_{t,a}^T A_a^{-1} x_{t,a}} \right)
$$

두 번째 항 $$\alpha \sqrt{x_{t,a}^T A_a^{-1} x_{t,a}}$$은 추정의 불확실성을 나타내며, 데이터가 축적될수록 줄어든다.

---

## LinUCB: Hybrid 모델

모든 팔이 공유하는 공통 파라미터 $$\beta^*$$를 추가하면:

$$
\mathbb{E}[r_{t,a} | x_{t,a}] = z_{t,a}^T \beta^* + x_{t,a}^T \theta_a^*
$$

여기서 $$z_{t,a}$$는 사용자-팔 상호작용의 공유 특성 벡터이다. 이 모델은 팔 간 정보 공유를 가능하게 하여, 새로운 팔(cold-start)에 대한 탐색 효율성을 높인다.

---

## 오프라인 평가 방법론

실시간 A/B 테스트 없이 배너 로그 데이터로 정책을 평가하는 방법을 제안한다. 기록된 데이터 $$(x_t, a_t, r_t)$$에서 무작위 정책으로 수집된 로그만 사용하여 새로운 정책의 기대 보상을 불편 추정할 수 있다:

$$
\hat{V}_\pi = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} r_t \cdot \mathbf{1}[\pi(x_t) = a_t]
$$

이 방법은 로그 데이터가 균일 무작위 정책으로 수집된 경우 편향이 없다.

---

## 실험 결과

Yahoo! Today Module의 실제 트래픽 데이터(4,500만 이벤트)에서 실험하였다:

| 알고리즘 | 상대 CTR 향상 |
|---------|-------------|
| Random | 기준선 |
| $$\epsilon$$-Greedy (Disjoint) | +7.1% |
| LinUCB (Disjoint) | +10.3% |
| LinUCB (Hybrid) | **+12.5%** |

LinUCB Hybrid 모델이 가장 우수한 성능을 보였으며, 탐색 파라미터 $$\alpha$$의 적절한 설정이 중요함을 확인하였다. 문맥 정보를 활용한 체계적 탐색이 단순 $$\epsilon$$-Greedy 대비 큰 성능 차이를 만들어낸다.

---

## Conclusion

이 논문은 개인화 추천을 contextual bandit 문제로 정식화하고, 선형 보상 가정 하에서 효율적인 탐색-활용 균형을 달성하는 LinUCB를 제안하였다. 릿지 회귀 기반의 신뢰 상한 계산이 핵심이며, 오프라인 평가 방법론을 통해 실용적 배포 파이프라인을 함께 제시하였다.

---

## Reference

- Li, L., Chu, W., Langford, J., & Schapire, R. E. "A Contextual-Bandit Approach to Personalized News Article Recommendation." *WWW 2010*.
