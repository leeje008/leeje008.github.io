---
layout: post
title: "[Paper Review] Doubly-Robust Lasso Bandit"
categories: [Paper Review]
tags: [paper-review, contextual-bandits, lasso, doubly-robust, missing-data, sparse-regression]
math: true
---

## Introduction

Bastani & Bayati (2020)의 [LASSO Bandit]({% post_url 2026-04-05-Online-Decision-High-Dimensional-LASSO-Bandit %})은 contextual bandit의 regret을 context 차원 $d$의 polylog에만 의존하도록 개선한 첫 번째 알고리즘이다. 그러나 그 알고리즘은 세 가지 불편한 점을 가진다.

1. **Forced sampling 블록**이 별도로 존재해야 하고, 그 스케줄 파라미터 $q$가 민감하다.
2. **두 개의 LASSO estimator** ($\hat{\beta}^{\text{FS}}$, $\hat{\beta}^{\text{AS}}$)를 arm별로 유지해야 한다.
3. Localization parameter $h$ 튜닝이 까다롭다.

Kim & Paik (NeurIPS 2019)의 "Doubly-Robust Lasso Bandit"은 통계학의 **누락 데이터(missing data) 처리 문헌**에서 유래한 **doubly-robust estimator** 기법을 LASSO bandit에 결합하여, 위 세 가지 문제를 한 번에 해결한다. 결과적으로:

- **단일 LASSO estimator**만 필요
- **Forced sampling 블록이 없음**
- Regret bound가 $\log d$로 스케일하며 **arm 수 $K$에 의존하지 않음**

이 리뷰는 doubly-robust 아이디어가 왜 bandit 문제에 자연스럽게 맞는지, 그리고 그것이 어떻게 LASSO와 결합되는지를 중심으로 정리한다.

---

## 1. 출발점 — Bandit을 Missing Data로 재해석

### 1.1 관측의 비대칭성

시점 $t$에서 context $x_t$가 주어지고 arm $a_t$를 선택하면, 에이전트는 **선택한 arm의 reward만** 관측한다. 선택하지 않은 $K-1$개 arm의 reward는 영원히 알 수 없다. 이 구조는 통계학적으로 다음과 깊이 닮았다.

> **각 시점에 $K$개의 "잠재 결과"(potential outcome)가 존재하는데, 그중 하나만 관측되고 나머지는 missing이다.**

이것은 Rubin (1974)의 potential outcome framework 및 인과추론 문헌의 중심 아이디어이다. Bandit은 결국 **시간에 걸쳐 누적되는 missing-not-at-random 문제**로 볼 수 있다.

### 1.2 Missing Data 문헌의 두 가지 기법

이 관점에서 두 고전적 추정자가 후보로 떠오른다.

**(a) Inverse Propensity Weighting (IPW) — Horvitz–Thompson 1952.** 선택 확률(propensity) $\pi_t(a) := \mathbb{P}(a_t = a \mid x_t)$ 의 역수로 관측된 reward를 가중:

$$
\tilde{r}_t^{\text{IPW}}(a) = \frac{\mathbf{1}\{a_t = a\}}{\pi_t(a)} \cdot r_t
$$

이것은 **unbiased**이지만 $\pi_t(a)$가 0에 가까울 때 분산이 폭발한다.

**(b) Regression Imputation.** 모델 $\hat{\mu}_a(x_t)$로 선택하지 않은 arm의 reward를 채워 넣는다. 낮은 분산이지만 $\hat{\mu}_a$의 편향이 그대로 전가된다.

두 접근은 trade-off 관계이고, 한쪽이 잘못되면 추정이 무너진다.

### 1.3 Doubly-Robust Estimator

Robins, Rotnitzky & Zhao (1994)의 doubly-robust estimator는 두 접근을 결합하여 **둘 중 하나만 옳아도 consistent**한 추정량을 만든다:

$$
\tilde{r}_t^{\text{DR}}(a) = \hat{\mu}_a(x_t) + \frac{\mathbf{1}\{a_t = a\}}{\pi_t(a)} \cdot (r_t - \hat{\mu}_a(x_t))
$$

**두 이름의 유래**:
- Propensity $\pi_t$가 옳다면: 두 번째 항의 기대값이 $r_t - \hat{\mu}_a(x_t)$가 되어 $\hat{\mu}_a$의 오차를 자동 보정
- Regression $\hat{\mu}_a$가 옳다면: 두 번째 항의 기대값이 0이 되어 첫 번째 항만 남음

즉 둘 중 하나만 맞아도 실제 기대값 $\mathbb{E}[r_t \mid x_t, a]$에 수렴한다. "Doubly-robust"라는 이름은 여기에서 나온다.

Kim & Paik의 핵심 관찰: **이 기법을 LASSO bandit의 매 step 추정에 그대로 이식하면, forced sampling 없이도 모든 arm에 대한 데이터를 활용할 수 있다.**

---

## 2. Doubly-Robust LASSO Bandit 알고리즘

### 2.1 Pseudo-Reward 구성

매 시점 $t$에서 에이전트는 모든 arm $a$에 대해 **pseudo-reward**를 구성한다:

$$
\tilde{r}_t(a) = x_t^\top \hat{\beta}_a^{(t-1)} + \frac{\mathbf{1}\{a_t = a\}}{\hat{\pi}_t(a)} \cdot (r_t - x_t^\top \hat{\beta}_a^{(t-1)})
$$

여기서 $\hat{\beta}_a^{(t-1)}$는 이전 시점까지의 LASSO 추정값이다. 핵심: **선택하지 않은 arm에 대해서도 pseudo-reward를 "채워 넣을" 수 있다**. 선택한 arm은 그 값이 실제 reward에 잘 맞도록 보정되고, 선택하지 않은 arm은 regression 예측값 $x_t^\top \hat{\beta}_a^{(t-1)}$이 그대로 사용된다.

### 2.2 Single LASSO Update

매 시점 각 arm $a$에 대해 **단일 LASSO**를 푼다:

$$
\hat{\beta}_a^{(t)} = \arg\min_{\beta} \left\{ \frac{1}{t} \sum_{s=1}^{t} (\tilde{r}_s(a) - x_s^\top \beta)^2 + \lambda_t \|\beta\|_1 \right\}
$$

**주목할 점**: 합이 $s=1$부터 $t$까지 — 즉 **모든 시점의 데이터가 모든 arm의 LASSO 업데이트에 기여**한다. Bastani–Bayati에서는 arm $a$ 데이터만 사용해서 $\hat{\beta}_a$를 추정했지만, 여기서는 doubly-robust 보정을 거친 pseudo-reward를 통해 arm 간 정보가 통합된다. 이것이 regret bound에서 $K$ 의존성이 사라지는 수학적 근원이다.

### 2.3 Arm 선택

새 context $x_{t+1}$이 오면:

$$
a_{t+1} = \arg\max_a x_{t+1}^\top \hat{\beta}_a^{(t)}
$$

**UCB bonus도 없고, forced sampling도 없다.** 순수 greedy이다. 탐색의 필요성은 doubly-robust 보정이 가져오는 pseudo-reward의 자연적 노이즈가 대신한다.

### 2.4 Propensity 추정

알고리즘이 작동하려면 $\hat{\pi}_t(a)$가 필요하다. Kim & Paik은 두 가지 옵션을 제시한다.

- **Exploration phase**: 초기 몇 라운드 uniform random으로 돌려 $\pi_t(a) = 1/K$로 고정
- **Action history 기반**: 이전 선택 이력에서 평균을 추정

실무에서는 약간의 $\epsilon$-greedy mixing (예: $\epsilon = 0.05$)을 통해 propensity가 0으로 떨어지는 것을 방지한다.

---

## 3. 이론적 결과

### 3.1 가정

- 각 arm의 파라미터가 sparse: $\|\beta_a^*\|_0 \le s_0$
- Context 분포의 compatibility condition 성립
- Propensity가 아래로 유계: $\hat{\pi}_t(a) \ge \pi_{\min} > 0$
- Reward noise가 sub-Gaussian

### 3.2 주 정리

위 가정 하에서, 누적 regret은 다음과 같이 bound된다.

$$
\boxed{\; \mathbb{E}[R_T] \le \mathcal{O}\!\left( s_0 \log(dT) \cdot \sqrt{T} \right) \;}
$$

Bastani–Bayati의 $\mathcal{O}(s_0^2 (\log T + \log d)^2)$와 비교하면:

- **$d$ 의존성**: 양쪽 모두 logarithmic. 같은 order.
- **$T$ 의존성**: Bastani–Bayati $(\log T)^2$ vs Kim–Paik $\sqrt{T}$. $T$가 매우 클 때는 Bastani–Bayati가 우위.
- **$s_0$ 의존성**: Bastani–Bayati $s_0^2$ vs Kim–Paik $s_0$. **$s_0$가 클 때 Kim–Paik이 유리**.
- **$K$ 의존성**: **Kim–Paik은 $K$가 등장하지 않는다.** Bastani–Bayati는 forced sampling 비율 때문에 실질적으로 $K$ 영향을 받는다.

$T$ 측면에서 Bastani–Bayati가 better order를 보이는 것은 그들이 더 강한 분포 가정을 사용했기 때문이며, Kim–Paik의 가정이 더 완화되어 있다. 실무에서는 두 방법이 유사한 regret을 보이는 경우가 많다.

### 3.3 증명의 핵심 아이디어

Doubly-robust pseudo-reward의 **이중 편향 상쇄(bias cancellation)** 구조가 핵심이다. Pseudo-reward를 decompose하면:

$$
\tilde{r}_t(a) - x_t^\top \beta_a^* = \underbrace{(x_t^\top \hat{\beta}_a^{(t-1)} - x_t^\top \beta_a^*)}_{\text{regression error}} + \underbrace{\frac{\mathbf{1}\{a_t = a\}}{\hat{\pi}_t(a)}(r_t - x_t^\top \hat{\beta}_a^{(t-1)})}_{\text{IPW correction}}
$$

첫째 항과 둘째 항이 서로를 상쇄하는 방향으로 작동한다. 저자들은 이 상쇄 구조와 LASSO의 restricted eigenvalue condition을 결합하여 adaptive 데이터 환경에서도 LASSO의 oracle inequality가 거의 그대로 성립함을 보인다.

**모든 arm의 데이터가 모든 arm의 LASSO 업데이트에 기여**하기 때문에 유효 표본 크기가 arm 수와 무관하게 전체 $T$에 수렴한다. 이것이 bound에서 $K$ 의존성을 지워버리는 이유이다.

---

## 4. 실험 결과

논문은 synthetic 데이터와 Warfarin dosing 데이터 양쪽에서 실험한다.

### 4.1 Synthetic — Correlated Contexts

Arm 간 context가 상관된 시나리오에서, 기존 LASSO Bandit은 arm별로 데이터를 분리해서 추정하므로 상관 구조를 활용하지 못한다. DR-LASSO는 모든 데이터가 모든 arm의 추정에 기여하므로 상관 구조를 자연스럽게 활용한다.

| 알고리즘 | 누적 regret (1000 rounds, $d = 100$, $s_0 = 5$) |
|---|---|
| LinUCB | ~580 |
| OFUL | ~520 |
| LASSO Bandit (Bastani–Bayati) | ~310 |
| **DR-LASSO Bandit** | **~240** |

### 4.2 Warfarin Dosing

실제 의료 데이터에서도 DR-LASSO가 Bastani–Bayati보다 약간 우수하며, 특히 **hyperparameter 튜닝에 훨씬 덜 민감**한 장점이 관찰된다. Forced sampling 주기 $q$가 없고, localization $h$가 없으며, 실질적으로 조정할 것이 LASSO 정규화 $\lambda$ 하나뿐이기 때문이다.

---

## 5. Bastani–Bayati vs Kim–Paik — 정밀 비교

| 항목 | Bastani–Bayati (2020) | Kim–Paik (2019) |
|---|---|---|
| 핵심 기법 | Forced sampling + Dual estimator | Doubly-robust pseudo-reward |
| Estimator 수 | Arm당 2개 ($\hat{\beta}^{\text{FS}}$, $\hat{\beta}^{\text{AS}}$) | Arm당 1개 |
| Forced sampling | **필요** | **불필요** |
| 탐색 메커니즘 | Forced blocks | Propensity의 자연적 노이즈 |
| Regret bound | $\mathcal{O}(s_0^2 (\log T + \log d)^2)$ | $\mathcal{O}(s_0 \log(dT) \sqrt{T})$ |
| $K$ 의존성 | 실질적으로 있음 | **없음** |
| 튜닝 파라미터 | $\lambda, q, h$ | $\lambda$ (+ propensity 하한) |
| 구현 복잡도 | 중간 | **낮음** |
| 강점 시나리오 | $T$가 매우 크고 arm 수 적음 | **Arm 간 상관, $K$ 큼, 튜닝 민감** |

실무 결정 기준:

- Arm 수가 적고 데이터가 많을 때: **Bastani–Bayati** (polylog $T$ 이득)
- Arm 수가 크거나 arm 간 상관이 있을 때: **Kim–Paik** (K-free, 구현 단순)
- 튜닝 비용이 중요할 때: **Kim–Paik**

---

## 6. 한계와 후속 연구

### 6.1 Propensity 하한 가정
$\hat{\pi}_t(a) \ge \pi_{\min}$ 가정은 순수 greedy 알고리즘과 양립하지 않는다. 실제로 $\epsilon$-greedy 섞기가 필요하며, 이것이 약간의 비최적성을 도입한다.

### 6.2 Propensity 추정 오차
Propensity를 추정해서 사용하는 경우, 그 추정 오차가 pseudo-reward에 전파된다. 이론 분석에서는 propensity가 정확하다고 가정하지만 실무에서는 완벽하지 않다.

### 6.3 비선형 보상
선형 가정이 깨지면 doubly-robust 상쇄 구조도 무너진다. Kernel·neural 확장은 이후 연구에서 다뤄진다.

### 6.4 후속 연구
- **Thresholded Lasso Bandit** (Ariu et al., 2022)
- **Sparse Generalized Linear Bandits** — 선형 가정 완화
- **Offline policy evaluation via doubly-robust**: 이 논문의 기법은 bandit offline evaluation에도 직접 이식된다 (Dudík et al., 2011).

---

## 7. Conclusion

Kim & Paik (2019)은 Bastani–Bayati의 LASSO Bandit을 **세 방향으로 개선**하였다.

1. **Forced sampling 제거**: Pseudo-reward 구성으로 모든 시점을 모든 arm 업데이트에 활용
2. **단일 LASSO**: Dual estimator 구조의 단순화
3. **$K$-free regret bound**: Arm 수에 의존하지 않는 regret

이 모든 개선의 뿌리는 **통계학의 누락 데이터 이론에서 온 doubly-robust estimator**이다. Bandit 문제를 potential outcome framework로 재해석하는 관점의 전환이 핵심이었고, 그것이 알고리즘의 단순화와 이론적 개선을 동시에 가능하게 했다.

> 한 줄 요약: **"Bandit을 missing data 문제로 보면 doubly-robust가 자연스럽게 나오고, 그것이 LASSO bandit의 forced sampling을 대체한다."**

Sparse bandit 시리즈를 이어 읽으려면 [LinUCB 이론 Deep Dive]({% post_url 2025-10-10-Contextual-Bandits-Linear-Payoff %})와 [LASSO Bandit 원논문 리뷰]({% post_url 2026-04-05-Online-Decision-High-Dimensional-LASSO-Bandit %})를 참고하라.

---

## References

- Kim, G.-S., & Paik, M. C. "Doubly-Robust Lasso Bandit." *Advances in Neural Information Processing Systems (NeurIPS)* 32, 2019.
- Bastani, H., & Bayati, M. "Online Decision Making with High-Dimensional Covariates." *Operations Research*, 68(1): 276–294, 2020.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. "Estimation of Regression Coefficients When Some Regressors Are Not Always Observed." *JASA*, 89(427): 846–866, 1994.
- Horvitz, D. G., & Thompson, D. J. "A Generalization of Sampling Without Replacement from a Finite Universe." *JASA*, 47(260): 663–685, 1952.
- Dudík, M., Langford, J., & Li, L. "Doubly Robust Policy Evaluation and Learning." *ICML 2011*.
- Rubin, D. B. "Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies." *Journal of Educational Psychology*, 66(5): 688–701, 1974.
- Tibshirani, R. "Regression Shrinkage and Selection via the Lasso." *JRSS-B*, 58(1): 267–288, 1996.
