---
layout: post
title: "[Paper Review] Statistical Power Analysis for the Behavioral Sciences"
categories: [Paper Review]
tags: [paper-review, statistics, power-analysis, effect-size]
math: true
---

## Introduction

Cohen (1988)의 *Statistical Power Analysis for the Behavioral Sciences*는 통계적 검정력 분석(power analysis)의 표준 교과서로, 행동과학 연구에서의 표본 크기 결정과 효과 크기(effect size) 해석에 대한 체계적 프레임워크를 제시한다. 이 저서의 핵심 기여는 통계적 검정의 네 가지 구성요소(유의수준, 검정력, 효과 크기, 표본 크기) 간의 정량적 관계를 명확히 하고, 효과 크기에 대한 관행적 기준(conventional benchmarks)을 제안한 점이다.

---

## 통계적 검정의 네 가지 구성요소

통계적 가설 검정은 다음 네 가지 양(quantity)으로 완전히 결정된다:

| 구성요소 | 기호 | 설명 |
|---------|------|------|
| 유의수준 | $$\alpha$$ | 제1종 오류 확률 (보통 0.05) |
| 검정력 | $$1 - \beta$$ | 참인 대립가설을 기각하는 확률 |
| 표본 크기 | $$n$$ | 각 그룹의 관측값 수 |
| 효과 크기 | $$d, r, f, w$$ 등 | 모집단에서의 효과의 크기 |

핵심 원리는 다음과 같다: **네 가지 양 중 세 가지를 알면 나머지 하나가 결정된다.** 검정력 분석의 가장 일반적인 활용은 원하는 $$\alpha$$, 검정력, 효과 크기를 지정한 후 필요한 표본 크기 $$n$$을 역산하는 것이다.

$$
\text{Power} = P(\text{reject } H_0 \mid H_1 \text{ is true}) = 1 - \beta
$$

---

## 효과 크기 측도

### 평균 차이: Cohen's $$d$$

두 집단의 평균 차이를 표준화한 측도이다:

$$
d = \frac{\mu_1 - \mu_2}{\sigma_{\text{pooled}}}
$$

여기서 $$\sigma_{\text{pooled}}$$는 합동 표준편차(pooled standard deviation)이다. Cohen은 다음의 관행적 기준을 제안한다:

| 크기 | $$d$$ 값 | 해석 |
|------|---------|------|
| 소(small) | 0.2 | 육안으로 잘 보이지 않는 차이 |
| 중(medium) | 0.5 | 관찰 가능한 차이 |
| 대(large) | 0.8 | 명확히 큰 차이 |

예를 들어, $$d = 0.5$$이면 두 집단의 분포가 약 67% 겹치며, $$d = 0.8$$이면 약 53% 겹친다.

### 상관계수: $$r$$

두 변수 간 선형 관계의 강도를 나타내는 Pearson 상관계수 $$r$$ 자체가 효과 크기 측도이다. 기준: $$r = 0.1$$(소), $$r = 0.3$$(중), $$r = 0.5$$(대).

### ANOVA: Cohen's $$f$$

$$k$$개 그룹 평균 간 변동을 모집단 표준편차로 표준화한다:

$$
f = \frac{\sigma_m}{\sigma}
$$

여기서 $$\sigma_m = \sqrt{\frac{1}{k}\sum_{j=1}^{k}(\mu_j - \mu)^2}$$는 집단 평균들의 표준편차, $$\sigma$$는 집단 내 표준편차이다. 기준: $$f = 0.10$$(소), $$f = 0.25$$(중), $$f = 0.40$$(대).

### 카이제곱 검정: Cohen's $$w$$

관측 비율과 기대 비율 간의 차이를 측정한다:

$$
w = \sqrt{\sum_{i=1}^{m} \frac{(P_{1i} - P_{0i})^2}{P_{0i}}}
$$

여기서 $$P_{0i}$$는 귀무가설 하의 기대 비율, $$P_{1i}$$는 대립가설 하의 비율이다. 기준: $$w = 0.1$$(소), $$w = 0.3$$(중), $$w = 0.5$$(대).

---

## 검정력 분석의 실제 활용

### 사전 검정력 분석 (A Priori Power Analysis)

연구 설계 단계에서 필요한 표본 크기를 결정한다. 예를 들어, 독립 이표본 $$t$$-검정에서 $$\alpha = 0.05$$, 검정력 $$= 0.80$$, $$d = 0.5$$일 때:

$$
n \approx \frac{2(z_{\alpha/2} + z_\beta)^2}{d^2} = \frac{2(1.96 + 0.84)^2}{0.25} \approx 63 \text{ (per group)}
$$

따라서 각 그룹에 약 63명, 총 126명이 필요하다.

### 사후 검정력 분석 (Post Hoc Power Analysis)

이미 수행된 연구에서 관측된 효과 크기와 표본 크기에 기반하여 검정력을 역산한다. Cohen은 행동과학 분야의 전형적 연구들이 **검정력 0.50 미만**인 경우가 많음을 지적하며, 이는 제2종 오류의 위험이 과소평가되고 있음을 경고한다.

---

## 검정력과 표본 크기의 관계

검정력은 표본 크기에 따라 단조증가하며, 그 관계는 비중심 분포(noncentral distribution)를 통해 정확히 기술된다. 비중심 $$t$$-분포에서 비중심 모수(noncentrality parameter)는 다음과 같다:

$$
\delta = d \cdot \sqrt{\frac{n}{2}}
$$

따라서 동일한 효과 크기 $$d$$에서 표본 크기가 4배가 되면 비중심 모수가 2배가 되어 검정력이 크게 증가한다.

---

## 의의와 영향

Cohen의 이 저서는 연구 설계에서 표본 크기 정당화(sample size justification)를 표준 관행으로 정착시켰다. 현대의 학술 저널과 연구비 심사에서 사전 검정력 분석은 필수적 요건으로 자리잡았다. 다만, 관행적 기준($$d = 0.2, 0.5, 0.8$$)의 기계적 적용은 연구 맥락을 무시할 수 있으므로, Cohen 자신도 **가능하면 선행 연구나 이론에 기반한 효과 크기 추정치를 사용할 것**을 강조하였다.

---

## Reference

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
