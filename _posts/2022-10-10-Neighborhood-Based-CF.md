---
layout: post
title: "[추천시스템] CH02 - Neighborhood-Based Collaborative Filtering"
categories: [Study Note]
tags: [recommender-system, collaborative-filtering, statistics]
math: true
---

## Introduction

Neighborhood-Based Collaborative Filtering은 추천 시스템의 가장 기본적인 접근 방식으로, 사용자-아이템 평점 행렬 \(R\)에서 유사한 사용자 또는 유사한 아이템을 찾아 빈 평점을 예측한다.

### 주요 표기법

- \(R\): 평점 행렬 (rating matrix)
- \(r_{uj}\): 사용자 \(u\)의 아이템 \(j\)에 대한 평점
- \(I_u\): 사용자 \(u\)가 평점을 매긴 아이템 집합
- \(P_u(j)\): 사용자 \(u\)에 대한 top-\(k\) 유사 사용자 집합
- \(Q_i(j)\): 아이템 \(i\)에 대한 top-\(k\) 유사 아이템 집합

---

## 유사도 측도 (Similarity Measures)

### 피어슨 상관계수

사용자 \(u\)의 평균 평점:

$$
\mu_u = \frac{\sum_{k \in I_u} r_{uk}}{|I_u|}
$$

사용자 \(u\)와 \(v\) 간의 피어슨 상관:

$$
\text{Pearson}(u, v) = \frac{\sum_{k \in I_u \cap I_v} (r_{uk} - \mu_u)(r_{vk} - \mu_v)}{\sqrt{\sum_{k} (r_{uk} - \mu_u)^2} \sqrt{\sum_{k} (r_{vk} - \mu_v)^2}}
$$

평균 중심화(mean-centering)를 통해 **사용자 간 평점 스케일 차이를 보정**한다는 점에서 코사인 유사도보다 선호된다.

### 코사인 유사도

$$
\text{Cosine}(u, v) = \frac{\sum_{k \in I_u \cap I_v} r_{uk} \cdot r_{vk}}{\sqrt{\sum_{k} r_{uk}^2} \sqrt{\sum_{k} r_{vk}^2}}
$$

평균 보정이 없어 사용자 간 평점 기준이 다를 때 편향이 발생할 수 있다.

### 할인된 유사도 (Discounted Similarity)

공통 평점 아이템이 적을 때 유사도의 신뢰성이 낮아지는 문제를 해결:

$$
\text{DiscountedSim}(u, v) = \text{Sim}(u, v) \cdot \frac{\min(|I_u \cap I_v|, \beta)}{\beta}
$$

\(|I_u \cap I_v| < \beta\)이면 유사도에 페널티를 부과한다.

---

## 평점 예측

### User-Based 예측

사용자 \(u\)의 아이템 \(j\)에 대한 평점 예측:

$$
\hat{r}_{uj} = \mu_u + \frac{\sum_{v \in P_u(j)} \text{Sim}(u, v) \cdot (r_{vj} - \mu_v)}{\sum_{v \in P_u(j)} |\text{Sim}(u, v)|}
$$

유사 사용자들의 **평균 중심화된 평점**을 유사도로 가중 평균하여 예측한다.

### Z-Score 정규화

사용자마다 평점의 분산도 다를 수 있으므로:

$$
z_{uj} = \frac{r_{uj} - \mu_u}{\sigma_u}
$$

예측: \(\hat{r}_{uj} = \mu_u + \sigma_u \cdot \frac{\sum_{v} \text{Sim}(u,v) \cdot z_{vj}}{\sum_{v} |\text{Sim}(u,v)|}\)

---

## Long-Tail 분포 처리

인기 아이템은 많은 사용자가 평가하여 유사도 계산에 과도한 영향을 미친다. 역문서 빈도(IDF) 방식의 가중치:

$$
w_j = \log\left(\frac{m}{m_j}\right)
$$

- \(m\): 전체 사용자 수
- \(m_j\): 아이템 \(j\)를 평가한 사용자 수

이 가중치를 피어슨 상관계수에 반영하면, 니치 아이템의 공통 선호가 더 강한 유사도 신호로 작동한다.

---

## User-Based vs Item-Based

| 구분 | User-Based | Item-Based |
|------|-----------|-----------|
| 유사도 기준 | 사용자 간 | 아이템 간 |
| 계산 복잡도 | \(O(m^2 n)\) | \(O(n^2 m)\) |
| 일반적 성능 | 보통 | **더 우수** |
| 이유 | 사용자 취향은 변하지만 아이템 특성은 안정적 | |

---

## 차원 축소 접근

평점 행렬의 희소성(sparsity) 문제를 완화하기 위한 방법:

- **클러스터링**: 사용자/아이템을 군집화하여 유사도 계산 범위를 축소. 맨해튼 거리가 선호됨.
- **PCA**: 평균 대체(mean imputation) 후 공분산 행렬의 고유값 분해로 저차원 표현 획득.
- **행렬 분해**: \(R \approx Q_d \Sigma P_d^T\). 희소 행렬에서 더 효과적.

---

## 그래프 기반 방법

사용자와 아이템을 노드, 상호작용을 엣지로 모델링한다.

### Katz 지수

경로 길이에 따른 감쇄 가중치 \(\beta\)를 적용하여 노드 간 유사도를 측정:

$$
K = (I - \beta A)^{-1} - I
$$

짧은 경로(작은 \(t\))에 더 큰 가중치를 부여하여, 직접적 연결이 강한 유사도 신호를 가지도록 한다.

---

## Reference

- Aggarwal, C.C. *Recommender Systems: The Textbook*. Springer, 2016. Chapter 2.
