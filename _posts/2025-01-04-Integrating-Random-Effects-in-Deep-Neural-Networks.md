---
layout: post
title: "[Paper Review] Integrating Random Effects in Deep Neural Networks"
categories: [Paper Review]
tags: [paper-review, deep-learning, statistics]
math: true
---

## Introduction

계층적 데이터셋(hierarchical datasets)에서는 데이터 포인트가 그룹으로 묶인다 (예: 학교별 학생, 병원별 환자). 이러한 데이터에서 **그룹별 변동성(group-specific variations)**을 모델링하는 것이 중요하다.

| 접근 방식 | 장점 | 한계 |
|-----------|------|------|
| 통계적 모델 | 랜덤 효과 처리 가능 | 비선형 관계 포착 불가 |
| 딥러닝 | 비선형성 모델링 우수 | 계층 구조 반영 불가 |

이 논문은 **랜덤 효과(Random Effects)를 DNN에 직접 통합**하여 두 가지 장점을 동시에 확보하는 방법을 제안한다.

---

## Proposed Approach

### Model Definition

그룹 $i$에 속하는 데이터 포인트에 대해:

$$
y = f(x; \theta) + u_i + \varepsilon
$$

- $f(x; \theta)$: 뉴럴 네트워크 출력 (파라미터 $\theta$)
- $u_i$: 그룹별 랜덤 효과
- $\varepsilon$: 잔차 노이즈

### Learning Objective

손실 함수는 네트워크 오류와 랜덤 효과 정규화를 함께 포함한다:

$$
\mathcal{L}(\theta, u) = \sum_{i} \sum_{j \in G_i} \left(y_j - f(x_j; \theta) - u_i\right)^2 + \lambda \sum_i u_i^2
$$

여기서 $\lambda$는 정규화 강도를 제어하는 파라미터이다.

### Group-Level Aggregation

랜덤 효과는 그룹 내 샘플 수를 기반으로 업데이트된다.

---

## Training Procedure (EM Algorithm)

**E-Step**: 현재 모델 파라미터를 사용하여 각 그룹의 랜덤 효과를 추정한다 (조건부 기댓값).

$$
\hat{u}_i = \frac{1}{|G_i| + \lambda} \sum_{j \in G_i} (y_j - f(x_j; \theta))
$$

**M-Step**: 랜덤 효과를 고정하고 네트워크 파라미터를 손실 최소화로 업데이트한다.

$$
\theta^{(t+1)} = \arg\min_\theta \sum_{i} \sum_{j \in G_i} \left(y_j - f(x_j; \theta) - \hat{u}_i\right)^2
$$

**수렴할 때까지 반복**한다.

---

## Results and Applications

### Performance

- 전통적 통계 모델보다 비선형성 포착에서 우수
- 표준 DNN보다 계층적 데이터 처리에서 우수
- 그룹 효과를 명시적으로 분리하여 해석 가능성 향상

### Applications

| 분야 | 적용 예시 |
|------|----------|
| 헬스케어 | 병원별 환자 결과 모델링 |
| 교육 | 학교별 학생 성과 분석 |
| 소셜 네트워크 | 그룹별 사용자 행동 포착 |

---

## Conclusion

이 논문은 계층적 통계 모델링과 딥러닝 사이의 간극을 메운다. 랜덤 효과를 DNN에 통합함으로써 비선형성과 그룹별 변동성을 동시에 모델링할 수 있으며, EM 알고리즘을 통해 효율적으로 학습할 수 있다.

---

## Reference

- Simchoni, G. & Rosset, S. "Integrating Random Effects in Deep Neural Networks." *JMLR 2023*.
