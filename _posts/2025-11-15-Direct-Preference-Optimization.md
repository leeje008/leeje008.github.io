---
layout: post
title: "[Paper Review] Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
categories: [Paper Review]
tags: [paper-review, reinforcement-learning, preference-learning, llm-alignment]
math: true
---

## Introduction

대규모 언어 모델(LLM)의 정렬(alignment)을 위한 표준 파이프라인은 **RLHF(Reinforcement Learning from Human Feedback)**로, SFT(Supervised Fine-Tuning) 후 보상 모델을 학습하고 PPO 등의 강화학습 알고리즘으로 정책을 최적화한다. 그러나 이 파이프라인은 보상 모델 학습, 정책 최적화, 하이퍼파라미터 튜닝 등 복잡한 다단계 과정을 필요로 한다. 이 논문은 보상 모델 없이 **인간 선호 데이터로부터 직접 정책을 최적화**하는 **Direct Preference Optimization(DPO)**을 제안한다.

---

## RLHF 파이프라인

### Bradley-Terry 선호 모델

인간 선호를 모델링하기 위해 Bradley-Terry 모델을 사용한다. 프롬프트 $x$에 대해 응답 $y_1$이 $y_2$보다 선호될 확률은:

$$
p^*(y_1 \succ y_2 | x) = \frac{\exp(r^*(x, y_1))}{\exp(r^*(x, y_1)) + \exp(r^*(x, y_2))} = \sigma(r^*(x, y_1) - r^*(x, y_2))
$$

여기서 $r^*(x, y)$는 잠재적 보상 함수이고, $\sigma$는 시그모이드 함수이다.

### 보상 모델 학습

선호 데이터셋 $\mathcal{D} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^N$에서 보상 모델 $r_\phi$를 학습한다:

$$
\mathcal{L}_R(r_\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]
$$

### RL 최적화

학습된 보상 모델로 정책을 최적화하되, 참조 정책 $\pi_{\text{ref}}$로부터 과도한 이탈을 방지한다:

$$
\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} \left[ r_\phi(x, y) \right] - \beta \, \text{KL}[\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x)]
$$

---

## DPO 유도

### 최적 정책의 닫힌 형태

위 KL 제약 강화학습 문제의 최적 정책은 다음과 같이 닫힌 형태로 표현된다:

$$
\pi_r(y | x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$

여기서 $Z(x) = \sum_y \pi_{\text{ref}}(y | x) \exp(\frac{1}{\beta} r(x, y))$는 분배 함수이다.

### 보상의 재매개변수화

위 식을 보상에 대해 정리하면:

$$
r(x, y) = \beta \log \frac{\pi_r(y | x)}{\pi_{\text{ref}}(y | x)} + \beta \log Z(x)
$$

이를 Bradley-Terry 모델에 대입하면 $Z(x)$가 상쇄된다.

---

## DPO 손실 함수

최종 DPO 목적 함수는 다음과 같다:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right]
$$

이 손실 함수는 선호 응답 $y_w$의 암시적 보상을 비선호 응답 $y_l$보다 높이도록 정책을 직접 최적화한다. 보상 모델을 별도로 학습할 필요가 없으며, 표준 교차 엔트로피 손실과 유사한 형태로 구현이 매우 단순하다.

---

## 이론적 분석

### 보상 동치류 (Theorem 1)

동일한 선호 분포를 유도하는 보상 함수의 집합은 동치류를 형성한다. 두 보상 함수 $r(x, y)$와 $r'(x, y)$가 동일한 최적 정책을 유도하기 위한 필요충분조건은:

$$
r(x, y) = r'(x, y) + f(x)
$$

즉, 프롬프트에만 의존하는 함수 $f(x)$의 차이까지 보상은 유일하게 결정된다. DPO는 이 동치류 내에서 올바른 정책을 학습한다.

### 경사 분석

DPO 손실의 경사를 분석하면, 가중치 $w(x, y_w, y_l)$는 암시적 보상 모델이 잘못 순위를 매긴 샘플에 더 높은 가중치를 부여한다:

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \, \mathbb{E} \left[ \underbrace{\sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w))}_{\text{가중치: 잘못된 순위에 높은 값}} \left( \nabla_\theta \log \pi_\theta(y_w | x) - \nabla_\theta \log \pi_\theta(y_l | x) \right) \right]
$$

---

## 실험 결과

세 가지 과제에서 DPO와 기존 방법을 비교하였다:

| 과제 | DPO | PPO (RLHF) | SFT | Best of N |
|------|-----|-----------|-----|-----------|
| 감성 제어 (IMDb) | **최고** | 준수 | 기준선 | 준수 |
| 텍스트 요약 (TL;DR) | **최고** | 동등 | 기준선 | 준수 |
| 대화 (Anthropic HH) | **최고** | 준수 | 기준선 | 준수 |

DPO는 모든 과제에서 PPO 기반 RLHF와 동등하거나 우수한 성능을 달성하였으며, 학습 안정성과 계산 효율성 측면에서 큰 이점을 보였다.

---

## Conclusion

DPO는 보상 모델 학습과 RL 최적화를 단일 손실로 대체하여 RLHF의 복잡성을 근본적으로 해소한다. 변수 치환을 통한 보상-정책 재매개변수화가 핵심이며, LLM 정렬 연구의 패러다임을 전환한 중요한 연구이다.

---

## Reference

- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." *NeurIPS 2023*.
