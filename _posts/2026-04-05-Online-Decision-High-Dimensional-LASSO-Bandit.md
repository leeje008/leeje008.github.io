---
layout: post
title: "[Paper Review] Online Decision Making with High-Dimensional Covariates — LASSO Bandit"
categories: [Paper Review]
tags: [paper-review, contextual-bandits, lasso, high-dimensional-statistics, sparse-regression]
math: true
---

## Introduction

Contextual bandit 문제에서 LinUCB (Li et al., 2010)와 그 이론적 완성형인 Abbasi-Yadkori et al. (2011)은 $\tilde{\mathcal{O}}(d\sqrt{T})$의 regret bound를 제공한다. 여기서 $d$는 context 차원이다. 이 bound는 $d$에 **선형** 의존이므로, 실제 응용에서 흔히 마주치는 **고차원 context** ($d \gg T$) 상황에서는 사실상 무용지물이 된다. 예컨대 의료 의사결정에서 환자의 feature는 수백~수천 차원에 이를 수 있는데, 이때 LinUCB는 초기 수많은 라운드를 단순 탐색에 소비하게 된다.

Bastani & Bayati (2020)의 "Online Decision Making with High-Dimensional Covariates" (*Operations Research* 68(1))는 이 문제에 대한 첫 번째 체계적 답을 제공한다. 통계학에서 고차원 회귀의 표준 도구인 **LASSO estimator** (Tibshirani, 1996)를 contextual bandit 프레임워크에 결합하여, regret이 $d$의 다항식이 아닌 $\text{polylog}(d)$로 스케일하는 알고리즘을 제안하였다.

이 리뷰는 논문의 핵심 아이디어인 **forced-sampling 기반 이중 추정 구조**와 그 뒤에 숨은 통계적 원리를 중심으로 정리한다. 본 리뷰의 전편에 해당하는 [LinUCB 논문 리뷰]({% post_url 2025-10-10-Contextual-Bandits-Linear-Payoff %})를 먼저 읽으면 맥락이 더 분명해진다.

---

## 1. 문제 정의와 Sparsity 가정

### 1.1 기본 설정

매 시점 $t = 1, 2, \ldots, T$에서 에이전트는 context $x_t \in \mathbb{R}^d$를 관찰하고, $K$개의 arm 중 하나 $a_t \in \{1, \ldots, K\}$를 선택한다. 각 arm $a$에 대해 보상의 조건부 기대값이 선형이라 가정한다:

$$
\mathbb{E}[r_t \mid x_t, a_t = a] = x_t^\top \beta_a^*, \qquad \beta_a^* \in \mathbb{R}^d
$$

여기까지는 LinUCB와 동일하다. 결정적 차이는 다음 가정에서 나온다.

### 1.2 Sparsity 가정

각 arm의 진짜 파라미터가 **sparse** 하다고 가정한다:

$$
\|\beta_a^*\|_0 \le s_0 \ll d, \qquad \forall a \in \{1, \ldots, K\}
$$

즉 $d$차원 context 중 실제로 보상에 영향을 주는 feature는 최대 $s_0$개뿐이다. 이 가정은 통계학에서 고차원 회귀를 다룰 때의 표준 가정이며, 실제 응용에서 대체로 타당하다 — 의료 의사결정에서 환자의 수백 개 혈액 지표 중 특정 약의 효과를 결정하는 것은 소수의 표지자이다.

### 1.3 왜 이 가정이 regret을 개선시키는가

LinUCB의 regret bound $\tilde{\mathcal{O}}(d\sqrt{T})$에서 $d$가 나오는 본질적 이유는 **confidence ellipsoid의 반지름이 $d$차원 모든 방향으로 퍼진다**는 점이다. 만약 참 파라미터가 대부분의 방향에서 0이라는 사실을 미리 안다면, 그 방향들은 탐색할 필요가 없고 알고리즘은 $s_0$ 차원 부분공간에 집중할 수 있다. 통계적으로 이것은 **ambient dimension $d$에서 effective dimension $s_0$로** 문제를 축소하는 것과 같다.

---

## 2. LASSO Estimator — 통계학적 배경

### 2.1 기본 정의

Offline 환경에서 $n$개의 $(x_i, y_i)$ 관측이 주어졌을 때, LASSO estimator (Tibshirani, 1996)는 다음과 같다:

$$
\hat{\beta}^{\text{LASSO}}(\lambda) = \arg\min_{\beta \in \mathbb{R}^d} \left\{ \frac{1}{n} \sum_{i=1}^n (y_i - x_i^\top \beta)^2 + \lambda \|\beta\|_1 \right\}
$$

$\ell_1$ 페널티는 해의 희소성을 유도한다 — 많은 좌표가 정확히 0이 되고, 자동적으로 변수 선택이 이루어진다.

### 2.2 Compatibility Condition

LASSO의 regret(추정 오차) 분석을 위해 **compatibility condition**이 필요하다. 설계 행렬 $X$의 Gram 행렬 $\hat{\Sigma} = X^\top X / n$에 대해, 어떤 상수 $\phi > 0$과 sparsity $s$가 존재하여 다음이 성립한다:

$$
\|\beta_S\|_1^2 \le \frac{s}{\phi^2} \cdot \beta^\top \hat{\Sigma} \beta, \qquad \forall \beta \in \mathcal{C}(S, 3)
$$

여기서 $S$는 support, $\mathcal{C}(S, 3) = \{\beta : \|\beta_{S^c}\|_1 \le 3\|\beta_S\|_1\}$는 cone이다. 직관적으로 이 조건은 "**LASSO 해의 후보 방향에서 설계 행렬이 비퇴행(non-degenerate)**"임을 보장한다.

이 조건 하에서 고전적 결과 (Bühlmann & van de Geer, 2011)는 다음을 말한다. 적절한 $\lambda$ 선택으로:

$$
\|\hat{\beta}^{\text{LASSO}} - \beta^*\|_1 \le C \cdot \frac{s_0 \sqrt{\log d}}{\sqrt{n}}
$$

즉 LASSO의 추정 오차는 **$d$가 아닌 $\log d$에만 의존**하며, 이것이 고차원 통계의 핵심 결과이다.

### 2.3 왜 Bandit에서 그대로 쓸 수 없는가

Bandit 환경에서는 두 가지 문제가 추가된다.

1. **Adaptive data collection**. $x_t$와 $a_t$가 과거에 의존하므로 i.i.d. 가정이 깨진다. Compatibility condition을 표본에서 직접 검증할 수 없다.
2. **Arm별 편향된 샘플**. 정책이 특정 arm을 집중적으로 선택하면 그 arm의 데이터 분포가 전체 context 분포와 달라진다. 이것이 LASSO estimator에 **bias**를 유도한다.

Bastani–Bayati의 핵심 기여는 이 두 문제를 **forced sampling**으로 우회하는 것이다.

---

## 3. LASSO Bandit 알고리즘

### 3.1 이중 추정자(Dual-Estimator) 구조

각 arm $a$에 대해 두 종류의 LASSO estimator를 동시에 유지한다.

- **Forced-sample estimator** $\hat{\beta}_a^{\text{FS}}$: 강제 샘플링으로 수집한 데이터만 사용
- **All-sample estimator** $\hat{\beta}_a^{\text{AS}}$: 지금까지 arm $a$가 선택된 **모든** 데이터 사용

두 estimator의 역할은 다음과 같이 분리된다.

| Estimator | 데이터 출처 | 통계적 성질 | 용도 |
|---|---|---|---|
| $\hat{\beta}_a^{\text{FS}}$ | Forced samples (i.i.d.) | 편향 없음, 일관된 수렴 | Arm 후보 집합 pruning |
| $\hat{\beta}_a^{\text{AS}}$ | All samples (adaptive) | 많은 데이터, 낮은 분산 | 최종 arm 선택 |

### 3.2 Forced Sampling Schedule

미리 정해진 **forced sample set** $\mathcal{T}_a \subseteq \{1, 2, \ldots\}$이 각 arm $a$에 대해 주어진다. 규칙은 간단하다.

$$
\mathcal{T}_a = \{(2^n - 1) \cdot Kq + j : n = 0, 1, 2, \ldots; \; j = (a-1)q + 1, \ldots, aq\}
$$

여기서 $q$는 forced-sampling 주기 파라미터이다. 풀어쓰면: 지수적으로 간격이 넓어지는 블록들에서 각 arm을 $q$번씩 의무적으로 선택한다. 시점 $t \in \mathcal{T}_a$이면 context와 무관하게 arm $a$를 선택한다.

전체 forced samples의 개수는 $T$ 시점까지 $\mathcal{O}(\log T)$에 불과하며, 이것이 forced-sampling의 오버헤드를 최소로 유지하는 핵심이다.

### 3.3 알고리즘 Pseudocode

시점 $t$에서:

1. **Forced sampling**: $t \in \mathcal{T}_a$인 arm이 있으면 그 arm 선택, reward 관측 후 $\hat{\beta}_a^{\text{FS}}$, $\hat{\beta}_a^{\text{AS}}$ 업데이트 → 다음 시점으로
2. **그 외**: $K^{\text{FS}} = \{a : x_t^\top \hat{\beta}_a^{\text{FS}} \ge \max_{a'} x_t^\top \hat{\beta}_{a'}^{\text{FS}} - h/2\}$ 구성 (후보 arm 집합, $h$는 localization parameter)
3. $a_t = \arg\max_{a \in K^{\text{FS}}} x_t^\top \hat{\beta}_a^{\text{AS}}$ 선택
4. Reward 관측 후 $\hat{\beta}_{a_t}^{\text{AS}}$ 업데이트

### 3.4 왜 이 구조가 작동하는가

핵심 통찰은 **역할 분리**이다.

- $\hat{\beta}_a^{\text{FS}}$는 독립 샘플에서 추정되므로 고전적 LASSO 결과가 그대로 적용된다. 즉 "진짜 suboptimal arm을 걸러내는 데" 사용해도 통계적으로 안전하다.
- $\hat{\beta}_a^{\text{AS}}$는 adaptive collection으로 bias가 있을 수 있지만, 이미 forced-sample로 "최적 근방 arm만" 남긴 뒤에 사용하므로 그 bias가 제한된다. 대신 훨씬 많은 데이터를 활용하여 분산을 낮춘다.

이것은 통계학의 **two-stage procedure** 아이디어와 본질이 같다 — 먼저 안전한 추정으로 모델 선택을 하고, 그 다음 효율적인 추정으로 파라미터를 확정한다.

---

## 4. 이론적 결과

### 4.1 새 Tail Inequality

논문의 진정한 기술적 기여는 **adaptive data collection 하에서 LASSO estimator의 수렴 보장**이다. 고전적 LASSO 분석은 i.i.d. 데이터를 전제하는데, bandit 환경은 그렇지 않다. 저자들은 다음 형태의 새 tail inequality를 증명한다.

각 arm의 context 분포에 대한 compatibility condition이 성립하고 forced sampling이 충분하다면, 어떤 상수 $C_1, C_2$에 대해:

$$
\mathbb{P}\left( \|\hat{\beta}_a^{\text{FS}} - \beta_a^*\|_1 > C_1 \cdot \frac{s_0 \sqrt{\log(dT)}}{\sqrt{n_a^{\text{FS}}}} \right) \le C_2 \cdot d^{-c}
$$

즉 adaptive data가 아닌 **i.i.d.인 forced sample 부분**에 대해 고전적 LASSO concentration을 그대로 적용할 수 있다.

### 4.2 Regret Bound

위 tail inequality와 localization 분석을 결합하면 누적 regret bound가 나온다.

$$
\boxed{\; \mathbb{E}[R_T] \le \mathcal{O}\!\left( s_0^2 \cdot (\log T + \log d)^2 \right) \;}
$$

**이것이 논문의 메인 결과이다.** 해석:

- **$d$ 의존성이 polylogarithmic**. LinUCB의 $\tilde{\mathcal{O}}(d\sqrt{T})$와 비교하면 고차원에서 극적인 개선.
- **$T$ 의존성도 polylogarithmic**. LinUCB의 $\sqrt{T}$보다도 좋다. 단, 상수에 $s_0^2$가 들어있어 sparsity가 크면 불리해진다.
- Contextual bandit 문헌에서 **처음으로 $d$에 polylog 스케일**하는 bound를 제시했다.

### 4.3 Compatibility Condition의 확률적 보장

정리의 전제로 나오는 compatibility condition은 데이터에 의존하는 조건이므로, 이것이 "높은 확률로" 성립함을 따로 보여야 한다. 저자들은 context 분포가 분산이 충분히 잘 행동(well-behaved)하는 조건 하에서 forced sample이 쌓일수록 이 조건이 높은 확률로 만족됨을 증명한다. 이는 분포적 가정과 forced sampling의 두 축이 결합되어야 알고리즘이 원리적으로 작동함을 의미한다.

---

## 5. 실험 — Warfarin Dosing

논문은 의료 용량 결정 문제에 알고리즘을 적용한다. **Warfarin**(혈전 방지제)은 환자 간 최적 용량이 크게 다르며 너무 적으면 혈전, 너무 많으면 출혈 위험이 있다. 저자들은 International Warfarin Pharmacogenetics Consortium 데이터 (5,700명)를 사용한다.

- **Context**: 인구통계 + 임상 + 유전 feature (수십~수백 차원)
- **Arm**: 3개의 용량 카테고리 (저/중/고)
- **Reward**: 올바른 용량을 선택했는지의 binary 지표

결과:

| 방법 | 올바른 용량 비율 |
|---|---|
| 고정 용량 (clinical baseline) | 약 61% |
| Logistic regression (오프라인) | 약 66% |
| OFUL / LinUCB | 약 65% |
| **LASSO Bandit** | **약 69%** |

**의사의 평균 처방 정확도조차 상회**하며, LinUCB 대비 일관된 우위를 보인다. 고차원 의료 feature 환경에서 sparsity 가정이 얼마나 강력한지를 보여주는 좋은 예시이다.

---

## 6. LinUCB와의 비교

| 항목 | LinUCB | LASSO Bandit |
|---|---|---|
| 파라미터 가정 | 제한 없음 ($\|\beta_a^*\|_2 \le S$) | Sparse ($\|\beta_a^*\|_0 \le s_0$) |
| 추정자 | Ridge regression (closed-form) | LASSO (볼록 최적화 반복) |
| Regret bound | $\tilde{\mathcal{O}}(d\sqrt{T})$ | $\mathcal{O}(s_0^2 (\log T + \log d)^2)$ |
| 고차원 적합성 | $d$ 작을 때 우수 | $d \gg T$ 상황에서 결정적 |
| 탐색 방식 | UCB (optimism) | Forced sampling + greedy |
| 구현 복잡도 | 매우 낮음 | 중간 (LASSO 주기적 재계산) |
| 튜닝 파라미터 | 탐색 상수 $\alpha$ 하나 | $\lambda$, $q$, $h$ — 세 개 |

정리하면: **중·저차원에서는 LinUCB**, **고차원에서는 LASSO Bandit**. 임계점은 대략 $d > \sqrt{T}$ 근처이지만 실제 응용에서는 경험적으로 결정해야 한다.

---

## 7. 한계와 후속 연구

### 7.1 튜닝 복잡성
Forced sampling 주기 $q$, localization $h$, LASSO 정규화 $\lambda$ — 세 파라미터의 상호작용이 민감하다. 특히 $h$가 너무 작으면 최적 arm을 놓치고, 너무 크면 suboptimal arm까지 all-sample 단계에 남는다.

### 7.2 Arm 수에 대한 의존성
Regret bound에는 명시적으로 $K$가 들어 있지는 않지만, forced sampling 비율이 $Kq / T$이므로 실질적으로 $K$가 크면 순수 탐색 라운드가 많아진다.

### 7.3 후속 연구 — Doubly-Robust LASSO Bandit
Kim & Paik (NeurIPS 2019)은 누락 데이터 처리의 doubly-robust 기법을 LASSO bandit에 결합하여 forced sampling을 제거하고 튜닝 파라미터를 줄였다. Regret bound도 $\log d$로 더 개선된다. 자세한 내용은 [다음 리뷰]({% post_url 2026-04-05-Doubly-Robust-LASSO-Bandit %})를 참고하라.

### 7.4 기타 확장
- **Thresholded Lasso Bandit** (Ariu et al., 2022): Thresholded LASSO로 sparsity 보장을 강화
- **High-dimensional Sparse Linear Bandits** (Hao et al., NeurIPS 2020): Minimax regret 하한 연구
- **Sparse linear bandit + experimental design**: 탐색을 experimental design 관점으로 재해석

---

## 8. Conclusion

Bastani & Bayati (2020)는 LinUCB의 $d$ 선형 의존성을 뚫는 첫 번째 체계적 시도이다. 핵심은 세 가지이다.

1. **Sparsity 가정**: $\|\beta_a^*\|_0 \le s_0$ 로 ambient dimension에서 effective dimension으로 문제를 축소
2. **Forced-sampling + Dual estimator**: 독립 샘플의 이론적 엄밀성과 전체 샘플의 효율성을 결합
3. **새 tail inequality**: Adaptive data collection 하에서 LASSO estimator의 수렴을 최초로 보장

결과적으로 regret이 $d$의 polylog에만 의존하는, 고차원 bandit 문헌의 ***첫 번째 breakthrough*** 가 만들어졌다. 이후의 sparse bandit 연구는 모두 이 논문을 출발점으로 삼는다.

> 한 줄 요약: **"고차원 bandit = 기존 LinUCB에 통계학의 sparse regression 도구를 forced-sampling이라는 안전 장치와 함께 이식한 것."**

---

## References

- Bastani, H., & Bayati, M. "Online Decision Making with High-Dimensional Covariates." *Operations Research*, 68(1): 276–294, 2020.
- Tibshirani, R. "Regression Shrinkage and Selection via the Lasso." *JRSS-B*, 58(1): 267–288, 1996.
- Bühlmann, P., & van de Geer, S. *Statistics for High-Dimensional Data: Methods, Theory and Applications*. Springer, 2011.
- Abbasi-Yadkori, Y., Pál, D., & Szepesvári, C. "Improved Algorithms for Linear Stochastic Bandits." *NeurIPS 2011*.
- Li, L., Chu, W., Langford, J., & Schapire, R. E. "A Contextual-Bandit Approach to Personalized News Article Recommendation." *WWW 2010*.
- Kim, G.-S., & Paik, M. C. "Doubly-Robust Lasso Bandit." *NeurIPS 2019*.
- Hao, B., Lattimore, T., & Wang, M. "High-Dimensional Sparse Linear Bandits." *NeurIPS 2020*.
