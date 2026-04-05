---
layout: post
title: "[Paper Review] Matrix Factorization Techniques for Recommender Systems"
categories: [Paper Review]
tags: [paper-review, recommender-system, matrix-factorization, collaborative-filtering]
math: true
---

## Introduction

추천 시스템은 크게 **콘텐츠 기반 필터링(content-based filtering)**과 **협업 필터링(collaborative filtering)**으로 나뉜다. 콘텐츠 기반 방법은 아이템의 속성 프로필을 생성하여 사용자 선호와 매칭하지만, 명시적 프로필 수집이 어렵다는 한계가 있다. 협업 필터링은 사용자-아이템 상호작용 데이터만으로 추천이 가능하며, 이 논문은 협업 필터링의 **잠재 요인 모델(latent factor model)** 중 행렬 분해(Matrix Factorization) 기법을 체계적으로 정리한다.

---

## 이웃 기반 vs 잠재 요인 모델

협업 필터링은 두 가지 주요 접근법이 있다:

| 구분 | 이웃 기반(Neighborhood) | 잠재 요인(Latent Factor) |
|------|----------------------|----------------------|
| 원리 | 유사 사용자/아이템의 평점 활용 | 저차원 잠재 공간으로 매핑 |
| 장점 | 해석 용이, 지역적 관계 포착 | 전역적 구조 포착, 확장성 |
| 단점 | 희소성에 취약 | 해석이 상대적으로 어려움 |

행렬 분해는 잠재 요인 모델의 대표적 방법으로, Netflix Prize에서 핵심적인 역할을 수행하였다.

---

## 기본 행렬 분해 모델

사용자 $u$를 벡터 $p_u \in \mathbb{R}^f$, 아이템 $i$를 벡터 $q_i \in \mathbb{R}^f$로 매핑하여 평점을 내적으로 예측한다:

$$
\hat{r}_{ui} = q_i^T p_u
$$

잠재 차원 $f$는 일반적으로 20~200 사이로 설정된다. 학습은 관찰된 평점에 대한 정규화된 제곱 오차를 최소화한다:

$$
\min_{q^*, p^*} \sum_{(u,i) \in \kappa} \left( r_{ui} - q_i^T p_u \right)^2 + \lambda \left( \|q_i\|^2 + \|p_u\|^2 \right)
$$

여기서 $\kappa$는 관찰된 $(u, i)$ 쌍의 집합이고, $\lambda$는 정규화 파라미터이다.

---

## 학습 알고리즘

### SGD (Stochastic Gradient Descent)

각 학습 샘플에 대해 예측 오차 $e_{ui} = r_{ui} - q_i^T p_u$를 계산하고 경사 방향으로 파라미터를 갱신한다:

$$
q_i \leftarrow q_i + \gamma (e_{ui} \cdot p_u - \lambda \cdot q_i)
$$

$$
p_u \leftarrow p_u + \gamma (e_{ui} \cdot q_i - \lambda \cdot p_u)
$$

구현이 단순하고 수렴이 빠르다는 장점이 있다.

### ALS (Alternating Least Squares)

$q_i$와 $p_u$ 중 하나를 고정하면 목적 함수가 이차 형태가 되어 닫힌 형태(closed-form)로 풀 수 있다. 두 변수를 교대로 최적화하며, 암시적 피드백이 많거나 병렬화가 필요한 경우에 적합하다.

---

## 편향 추가

사용자와 아이템의 고유 편향을 반영하면 예측 정확도가 크게 향상된다:

$$
\hat{r}_{ui} = \mu + b_i + b_u + q_i^T p_u
$$

여기서 $\mu$는 전체 평균 평점, $b_i$는 아이템 편향, $b_u$는 사용자 편향이다. 예를 들어, 전체 평균이 3.7이고 어떤 영화가 평균보다 0.5점 높으며, 특정 사용자가 평균보다 0.3점 엄격하다면 기저 추정치는 $3.7 + 0.5 - 0.3 = 3.9$가 된다.

---

## 암시적 피드백과 추가 입력

명시적 평점 외에 클릭, 구매 등의 암시적 피드백을 통합할 수 있다. 사용자 $u$가 암시적 선호를 보인 아이템 집합을 $N(u)$라 하면:

$$
\hat{r}_{ui} = \mu + b_i + b_u + q_i^T \left( p_u + |N(u)|^{-1/2} \sum_{j \in N(u)} y_j \right)
$$

이를 통해 평점이 적은 사용자에 대해서도 안정적인 잠재 벡터 추정이 가능하다.

---

## 시간적 역학

사용자 선호와 아이템 인기도는 시간에 따라 변화한다. 시간 의존적 모델은 다음과 같다:

$$
\hat{r}_{ui}(t) = \mu + b_i(t) + b_u(t) + q_i^T p_u(t)
$$

아이템 편향 $b_i(t)$는 시간 구간별 이동(bin-based drift)으로 모델링하고, 사용자 편향 $b_u(t)$와 잠재 벡터 $p_u(t)$는 기저값에 시간 편차를 더하는 방식으로 구현한다. Netflix Prize 우승 솔루션에서 시간적 역학의 반영이 가장 큰 성능 향상 요인이었다.

---

## Conclusion

행렬 분해 기법은 협업 필터링의 핵심 방법론으로, 편향 모델링, 암시적 피드백 통합, 시간적 역학 반영을 통해 지속적으로 정확도를 개선할 수 있다. SGD와 ALS를 통한 효율적 학습이 가능하며, Netflix Prize에서 그 효과가 실증적으로 검증되었다. 이후 딥러닝 기반 추천 시스템의 기초가 되는 중요한 연구이다.

---

## Reference

- Koren, Y., Bell, R., & Volinsky, C. "Matrix Factorization Techniques for Recommender Systems." *IEEE Computer, 2009*.
