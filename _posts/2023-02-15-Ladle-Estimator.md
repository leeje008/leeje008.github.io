---
layout: post
title: "[Paper Review] Ladle Estimator - 고유값과 고유벡터 변동성을 결합한 차원 결정"
categories: [Paper Review]
tags: [paper-review, dimension-reduction, order-determination, bootstrap, statistics]
math: true
---

## Introduction

PCA, 정준상관분석(CCA), 독립성분분석(ICA), 충분차원축소(SDR) 등 다양한 통계적 방법론에서 **차원 결정(order determination)** 문제가 공통적으로 등장한다. 이는 랜덤 행렬 $M$의 랭크 $d$를 추정하는 문제로 귀결된다.

기존의 차원 결정 방법은 두 가지 접근으로 나뉜다:

1. **고유값 기반**: scree plot의 elbow, 정보 기준(AIC, BIC), 순차 검정 — 고유값 $\hat{\lambda}_k$가 $k = d$ 이후 급격히 감소하는 패턴을 이용
2. **고유벡터 기반**: 부트스트랩 고유벡터의 변동성(Ye & Weiss, 2003) — $k > d$일 때 고유벡터의 부트스트랩 변동성이 급증하는 패턴을 이용

Luo & Li (2016)는 이 두 정보를 **동시에 결합**하는 Ladle Estimator를 제안한다. 핵심 관찰: 고유값이 가까울 때 고유벡터 변동성은 크고, 고유값이 떨어져 있을 때 변동성은 작다. 이 **상보적(complementary) 패턴**을 결합하면 단독 사용보다 정확한 차원 결정이 가능하다.

---

## 문제 설정

### 일반적 프레임워크

$(X, Y)$의 분포 $F$에 대해 행렬 값 통계적 범함수(statistical functional) $M : \mathfrak{F} \to \mathbb{R}^{p \times p}$를 정의한다. 표본 추정량은 $\hat{M} = M(F_n)$이고, 부트스트랩 추정량은 $M^* = M(F_n^*)$이다.

### 적용 예시

| 설정 | 행렬 $M$ | 랭크 $d$의 의미 |
|------|---------|-------------|
| PCA | $\text{Var}(Z) - \sigma^2 I_p$ | 주성분의 수 |
| CCA | $\Sigma_{XX}^{-1/2} \Sigma_{XY} \Sigma_{YY}^{-1} \Sigma_{YX} \Sigma_{XX}^{-1/2}$ | 정준상관 벡터의 수 |
| ICA | $[E\{(Z^T Z) ZZ^T\} - (p+2)I_p]^2$ | 초과 첨도를 가진 성분의 수 |
| SDR (SIR) | $\text{Var}[E(Z \mid Y)]$ | 구조적 차원 |
| SDR (DR) | $E[(2I_p - A(Y, \tilde{Y}))^2]$ | 구조적 차원 |

---

## 고유값-고유벡터 패턴의 상보성

$M$의 고유값을 $\lambda_1 \geq \cdots \geq \lambda_d > 0 = \lambda_{d+1} = \cdots = \lambda_p$로 놓고, 대응하는 고유벡터를 $v_1, \ldots, v_p$라 하자.

### 고유값 패턴 (Scree Plot)

$\hat{\lambda}_k$는 $k \leq d$일 때 큰 값, $k > d$일 때 0에 가까운 값을 가진다. Scree plot은 $\hat{\lambda}_k$가 급격히 감소하는 elbow를 찾는다.

### 고유벡터 변동성 패턴

부트스트랩 고유벡터 $\hat{v}_{1}^*, \ldots, \hat{v}_{k}^*$의 변동성을 측정하기 위해, $\hat{B}_k = (\hat{v}_1, \ldots, \hat{v}_k)$와 $B_{k,j}^* = (v_{1,j}^*, \ldots, v_{k,j}^*)$ 간의 부분공간 불일치를 정의한다:

$$
f_n^0(k) = \begin{cases} 0, & k = 0 \\ n^{-1} \sum_{j=1}^{n_b} \{1 - |\det(\hat{B}_k^T B_{k,j}^*)|\}, & k = 1, \ldots, p-1 \end{cases}
$$

여기서 $1 - |\det(\hat{B}_k^T B_{k,j}^*)|$는 두 $k$-차원 부분공간의 불일치도를 0(완전 일치)과 1(완전 직교) 사이로 측정한다.

**두 가지 시나리오**:

**(i)** $\lambda_k > \lambda_{k+1}$: $v_k$와 $v_{k+1}$은 서로 다른 고유공간에 속하므로, 모든 부트스트랩 표본에서 동일한 $k$-차원 부분공간을 추정 → $f_n(k)$가 **작다**

**(ii)** $\lambda_k = \lambda_{k+1}$: $v_k$와 $v_{k+1}$이 같은 고유공간에 속하여, 부트스트랩마다 해당 고유공간 내에서 임의의 방향이 선택됨 → $f_n(k)$가 **크다**

특히, $k = d$에서 $\lambda_d > 0 = \lambda_{d+1}$이므로 $f_n(d)$는 작고, $k = d+1$에서 $\lambda_{d+1} = \lambda_{d+2} = 0$이므로 $f_n(d+1)$은 크다. 따라서 $f_n(k)$는 $k = d$에서 **점프**가 발생한다.

### 상보성의 핵심

- $k \leq d$: 고유값 $\phi_n(k)$는 크고, 고유벡터 변동성 $f_n(k)$는 작다
- $k > d$: 고유값 $\phi_n(k)$는 작고, 고유벡터 변동성 $f_n(k)$는 크다
- $k = d$ 부근: **두 함수 모두** 유용한 정보를 제공하며, 결합하면 더 sharp한 최솟값을 형성

---

## Ladle Estimator

### 정규화

고유값과 고유벡터 변동성을 동일한 스케일로 결합하기 위해 정규화한다:

$$
f_n(k) = \frac{f_n^0(k)}{1 + \sum_{i=0}^{q} f_n^0(i)}, \quad \phi_n(k) = \frac{\hat{\lambda}_{k+1}}{1 + \sum_{i=0}^{q} \hat{\lambda}_{i+1}}
$$

여기서 $q = \lfloor p / \log(p) \rfloor$은 탐색 범위의 상한이다. 분모의 1은 $d = p - 1$일 때의 안정성을 위해 도입된다.

### 목적 함수와 정의

Ladle 목적 함수는 두 정규화된 함수의 합이다:

$$
g_n(k) = f_n(k) + \phi_n(k)
$$

**Definition 1** (Ladle Estimator):

$$
\hat{d} = \arg\min_{k \in \mathcal{D}(g_n)} g_n(k)
$$

여기서 $\mathcal{D}(g_n) = \{0, 1, \ldots, q\}$이다.

### 직관적 해석

- $k < d$: $\phi_n(k)$가 크므로 (아직 유의미한 고유값이 남아있으므로) $g_n(k)$가 크다
- $k > d$: $f_n(k)$가 크므로 (고유벡터 변동성이 증가하므로) $g_n(k)$가 크다
- $k = d$: $\phi_n(d)$와 $f_n(d)$ **모두 작으므로** $g_n(d)$가 최소

이 곡선의 형태가 국자(ladle) 모양을 닮아 Ladle Estimator라 명명되었다.

---

## 이론적 성질

### 가정

**Assumption 1**: $\hat{M} = M + E_n H(X, Y) + o_p(n^{-1/2})$. 통계적 범함수 $M$이 Fréchet 미분가능하면 자동으로 성립한다.

**Assumption 2** (Self-similarity): 부트스트랩 추정량 $M^*$이 다음을 만족:

$$
n^{1/2}\{\text{vech}(M^*) - \text{vech}(\hat{M})\} \xrightarrow{d} N(0, \text{Var}_F[\text{vech}\{H(X,Y)\}])
$$

이는 $n^{1/2}(\hat{M} - M)$의 점근 분포를 부트스트랩이 올바르게 모방함을 의미한다. 복원 추출(sampling with replacement)이 이 가정의 핵심이다.

**Assumption 3**: $Z_n = O_p(c_n)$이면 $E(c_n^{-1} Z_n) = O(1)$. 비음 확률변수에 대한 기술적 조건.

### Theorem 1: 고유값-고유벡터 패턴의 엄밀한 특성화

$c_n = [\log\{\log(n)\}]^{-2}$으로 놓으면, Assumptions 1-3 하에서 거의 모든 수열 $S$에 대해:

$$
f_n(k) = \begin{cases} O_p(n^{-1}), & \lambda_k > \lambda_{k+1} \\ O_p^+(c_n), & \lambda_k = \lambda_{k+1} \end{cases}
$$

**해석**: 연속적인 고유값이 서로 다르면 ($\lambda_k > \lambda_{k+1}$) 고유벡터 변동성은 $O_p(n^{-1})$로 무시할 수 있지만, 같으면 ($\lambda_k = \lambda_{k+1}$) $O_p^+(c_n)$으로 무한히 느리게 소멸한다. $c_n$의 수렴 속도가 극히 느리므로 ($\{\log(\log 10^4)\}^{-2} \approx 0.2$), 실제로는 $O_p^+(1)$로 취급할 수 있다.

특히, $\lambda_d > 0 = \lambda_{d+1}$이므로 $f_n$은 **항상** $k = d$에서 점프가 발생하며, 이는 비영(nonzero) 고유값의 중복도와 무관하다.

### Theorem 2: Ladle Estimator의 일치성

Assumptions 1-3 하에서, 임의의 양반정치 행렬 $M \in \mathbb{R}^{p \times p}$, $\text{rank}(M) = d \in \{0, \ldots, p-1\}$에 대해:

$$
\Pr\left\{\lim_{n \to \infty} \Pr(\hat{d} = d \mid S) = 1\right\} = 1
$$

즉, Ladle Estimator는 거의 확실히(almost surely) 일치적이다. 이 결과는 고유값의 중복도에 대한 가정 없이 성립한다.

**특수한 경우**: 모든 비영 고유값이 동일한 경우 ($\lambda_1 = \cdots = \lambda_d > 0$), Theorem 1에 의해 $f_n$ 자체의 최솟값도 $k = d$에서 달성되므로 $f_n$만으로도 일치적이다. 그러나 $\phi_n$을 결합하면 $k < d$에서 $g_n$의 하강 추세가 증폭되어 유한 표본 성능이 향상된다.

---

## 기존 방법과의 비교

### 시뮬레이션 설정

Luo & Li는 6가지 설정(PCA, CCA, ICA, SDR-SIR, SDR-DR, 비선형 SDR)에서 $p = 10$과 $p = 40$으로 실험:

| 방법 | 유형 | 튜닝 파라미터 | 적용 범위 |
|------|------|-----------|---------|
| 순차 검정 (ST) | 고유값 기반 | 유의수준 | PCA 불가 |
| 정보 기준 (IC) | 고유값 기반 | 없음 | 설정별 다름 |
| Ye-Weiss (YW) | 고유벡터 기반 | 임계값 $\delta$ | 범용 |
| **Ladle** | **결합** | **없음** | **범용** |

### 주요 결과 ($p = 10$)

| 설정 | ST | IC | YW1 | YW2 | YW3 | **Ladle** |
|------|----|----|-----|-----|-----|-----------|
| PCA ($d=3$) | — | 85 | 73 | 23 | — | **99** |
| CCA ($d=2$) | 92 | 97 | 89 | 36 | — | **96** |
| ICA ($d=2$) | — | 75 | 90 | 63 | — | **90** |
| SDR-SIR ($d=2$) | (78,75) | 0 | 48 | 95 | 86 | **91** |
| SDR-DR ($d=2$) | (79,96) | 4 | 87 | 89 | 27 | **98** |
| NSDR ($d=1$) | — | — | 79 | 99 | 95 | **99** |

Ladle Estimator는 **모든 설정에서 안정적으로 높은 정확도**를 보이며, 특정 설정에서 실패하는 다른 방법들과 달리 범용적이다.

### Ladle의 장점

1. **튜닝 파라미터 불필요**: 순차 검정의 유의수준, Ye-Weiss의 임계값 $\delta$ 등이 필요 없다
2. **범용성**: PCA, CCA, ICA, SDR 등 다양한 설정에 동일하게 적용 가능
3. **상보적 정보 활용**: 고유값만 또는 고유벡터만 사용하는 방법보다 안정적
4. **이론적 보장**: 약한 가정 하에서 일치성이 증명됨

---

## 실용적 고려사항

- **부트스트랩 표본 수**: 원래 표본 크기 $n$과 동일하게 설정, 복원 추출 사용
- **탐색 범위**: $k \in \{0, 1, \ldots, \lfloor p / \log(p) \rfloor\}$. $p = 10$이면 $q = 4$, $p = 40$이면 $q = 10$
- **Ladle plot**: $(k, g_n(k))$의 산점도를 scree plot의 대안으로 사용. Elbow가 아닌 **최솟값**을 찾으므로 더 객관적
- **계산 비용**: $n_b$번의 부트스트랩 각각에서 $\hat{M}^*$의 고유값 분해가 필요. $p$가 크면 비용이 증가하지만, 병렬화 가능

---

## Reference

- Luo, W. & Li, B. (2016). Combining Eigenvalues and Variation of Eigenvectors for Order Determination. *Biometrika*, 103(4), 875-887.
- Ye, Z. & Weiss, R. E. (2003). Using the Bootstrap to Select One of a New Class of Dimension Reduction Methods. *JASA*, 98(464), 968-979.
