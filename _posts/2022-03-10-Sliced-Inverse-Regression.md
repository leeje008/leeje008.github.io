---
layout: post
title: "[Paper Review] Sliced Inverse Regression for Dimension Reduction"
categories: [Paper Review]
tags: [paper-review, dimension-reduction, sufficient-dimension-reduction, statistics]
math: true
---

## Introduction

Li (1991)의 Sliced Inverse Regression(SIR)은 충분차원축소(Sufficient Dimension Reduction) 분야의 시초가 되는 논문이다. 고차원 설명변수 \(\mathbf{x} \in \mathbb{R}^p\)와 반응변수 \(y\) 간의 관계를 탐색할 때, 비모수적 방법은 차원의 저주(curse of dimensionality)로 인해 급격히 성능이 저하된다. SIR은 **역회귀(inverse regression)** 아이디어를 활용하여, 순방향 회귀(\(y\)를 \(\mathbf{x}\)에 회귀)의 고차원 문제를 역방향 회귀(\(\mathbf{x}\)를 \(y\)에 회귀)의 저차원 문제로 전환한다.

### 모형

$$
y = f(\beta_1^T \mathbf{x}, \beta_2^T \mathbf{x}, \ldots, \beta_K^T \mathbf{x}, \varepsilon)
$$

여기서 \(\beta_k\)는 미지의 \(p\)-벡터, \(f\)는 **완전히 미지의 함수**, \(\varepsilon\)는 \(\mathbf{x}\)와 독립인 오차이다. 이 모형의 핵심: 링크 함수 \(f\)의 형태에 대한 가정이 전혀 없다. \(\beta_k\)들이 생성하는 선형 부분공간 \(B = \text{span}(\beta_1, \ldots, \beta_K)\)를 **유효 차원축소(e.d.r.) 공간**이라 하며, \(K \ll p\)일 때 \(p\)차원 문제를 \(K\)차원으로 축소할 수 있다.

---

## 역회귀 곡선과 핵심 정리

### 역회귀 곡선

\(y\)가 변함에 따라 조건부 기대값 \(E(\mathbf{x} \mid y)\)가 \(\mathbb{R}^p\) 공간에서 그리는 궤적을 **역회귀 곡선(inverse regression curve)**이라 한다. 순방향 회귀 \(E(y \mid \mathbf{x})\)는 \(p\)차원 곡면이지만, 역회귀 곡선은 1차원 매개변수 \(y\)에 의해 인덱싱되므로 추정이 훨씬 용이하다.

### Linearity Condition (Condition 3.1)

임의의 \(b \in \mathbb{R}^p\)에 대해:

$$
E(b^T \mathbf{x} \mid \beta_1^T \mathbf{x}, \ldots, \beta_K^T \mathbf{x}) = c_0 + c_1 \beta_1^T \mathbf{x} + \cdots + c_K \beta_K^T \mathbf{x}
$$

즉, \(\mathbf{x}\)의 임의의 선형결합의 e.d.r. 변수들에 대한 조건부 기대값이 선형이어야 한다. 이 조건은 \(\mathbf{x}\)가 타원형 대칭 분포(elliptically symmetric distribution)를 따르면 자동으로 성립한다. 정규분포가 대표적 예시이다.

### Theorem 3.1: SIR의 핵심 정리

Condition 3.1 하에서, 중심화된 역회귀 곡선 \(E(\mathbf{x} \mid y) - E(\mathbf{x})\)는 e.d.r. 방향 \(\beta_k\)들이 생성하는 **\(K\)-차원 아핀 부분공간에 포함**된다.

표준화된 변수 \(\mathbf{z} = \Sigma_{\mathbf{xx}}^{-1/2}(\mathbf{x} - E(\mathbf{x}))\)를 사용하면:

$$
E(\mathbf{z} \mid y) \in \text{span}(\eta_1, \ldots, \eta_K)
$$

여기서 \(\eta_k = \beta_k \Sigma_{\mathbf{xx}}^{1/2}\)는 표준화된 e.d.r. 방향이다.

### Corollary 3.1

\(\text{Cov}[E(\mathbf{z} \mid y)]\)의 고유벡터 중 가장 큰 \(K\)개의 고유값에 대응하는 것들이 표준화된 e.d.r. 방향 \(\eta_1, \ldots, \eta_K\)를 추정한다.

---

## SIR 알고리즘

1. \(\mathbf{x}\)를 표본 평균과 공분산으로 표준화: \(\hat{\mathbf{z}}_i = \hat{\Sigma}_{\mathbf{xx}}^{-1/2}(\mathbf{x}_i - \bar{\mathbf{x}})\)
2. \(y\)의 범위를 \(H\)개 슬라이스 \(I_1, \ldots, I_H\)로 분할, 각 슬라이스 비율 \(\hat{p}_h = n_h / n\)
3. 각 슬라이스 내 \(\hat{\mathbf{z}}_i\)의 표본 평균 계산: \(\hat{\mathbf{m}}_h = (1/n\hat{p}_h) \sum_{y_i \in I_h} \hat{\mathbf{z}}_i\)
4. 가중 공분산 행렬 구성: \(\hat{V} = \sum_{h=1}^{H} \hat{p}_h \hat{\mathbf{m}}_h \hat{\mathbf{m}}_h^T\)
5. \(\hat{V}\)의 고유값 분해. 상위 \(K\)개 고유벡터 \(\hat{\eta}_k\) 추출
6. 원래 스케일로 복원: \(\hat{\beta}_k = \hat{\eta}_k \hat{\Sigma}_{\mathbf{xx}}^{-1/2}\)

### 구현의 간결성

SIR의 핵심적 장점은 **비모수적 평활(smoothing)이 불필요**하다는 것이다. 슬라이스 내 평균만 계산하면 되므로, 커널 회귀, 최근접 이웃 등의 대역폭 선택 문제가 없다. 또한 슬라이스 수 \(H\)의 선택에 대해 결과가 강건(robust)하다.

---

## 점근 이론

### \(\sqrt{n}\)-일치성

\(\hat{p}_h \to p_h = P(y \in I_h)\)이 \(n^{-1/2}\) 속도로 수렴하므로, \(\hat{V} \to V\)도 \(\sqrt{n}\) 속도로 수렴한다. 따라서 \(\hat{V}\)의 고유벡터 \(\hat{\eta}_k\)도 \(\sqrt{n}\)-일치적으로 \(\eta_k\)를 추정한다.

### Theorem 5.1: 차원 결정을 위한 검정

\(\mathbf{x}\)가 정규분포를 따를 때, \(K\)개 성분 후 나머지 \(p - K\)개 고유값의 평균:

$$
n(p - K) \bar{\lambda}_{(p-K)} \sim \chi^2_{(p-K)(H-K-1)}
$$

여기서 \(\bar{\lambda}_{(p-K)} = \frac{1}{p-K} \sum_{k=K+1}^{p} \hat{\lambda}_k\). 이 검정을 통해 유의미한 성분의 수 \(K\)를 결정할 수 있다.

### 추정 효율성의 근사 (식 5.1)

\(\hat{B}\)를 \(K\)개의 추정된 e.d.r. 방향이 생성하는 부분공간이라 하면, 제곱 다중 상관계수의 기대값:

$$
E[R^2(\hat{B})] \approx 1 - \frac{p - K}{n}\left(-1 + \frac{1}{K}\sum_{k=1}^{K}\frac{1}{\lambda_k}\right)
$$

이 근사식은 \(\lambda_k\)가 크면(강한 신호) \(R^2\)가 1에 가까움을 보여준다.

---

## SIR의 한계: 대칭 의존 구조

SIR의 가장 중요한 한계는 **대칭적 의존 구조를 탐지하지 못한다**는 것이다.

예: \(y = (\beta^T \mathbf{x})^2 + \varepsilon\)일 때, \(E(\mathbf{x} \mid y)\)는 \(\beta\) 방향으로 대칭이므로 역회귀 곡선이 \(\beta\) 방향의 변동을 보이지 않는다. 즉, \(E(\mathbf{z} \mid y) \approx 0\)이 되어 SIR이 \(\beta\)를 감지하지 못한다.

이 한계를 극복하기 위해 이후 SAVE (Cook & Weisberg, 1991), DR (Li & Wang, 2007) 등이 제안되었다.

---

## 역사적 의의

SIR은 다음과 같은 점에서 SDR 분야의 기초를 놓았다:

1. **역회귀 패러다임**: 고차원 순방향 회귀 → 저차원 역방향 회귀로의 전환
2. **모형 자유(model-free)**: 링크 함수 \(f\)에 대한 가정 없이 차원축소
3. **계산 효율성**: 비모수 평활 없이 표본 평균과 주성분 분석만으로 구현
4. **이론적 프레임워크**: linearity condition, e.d.r. 공간 등 이후 SDR 연구의 표준 용어 확립

---

## Reference

- Li, K.-C. (1991). Sliced Inverse Regression for Dimension Reduction. *Journal of the American Statistical Association*, 86(414), 316-327.
