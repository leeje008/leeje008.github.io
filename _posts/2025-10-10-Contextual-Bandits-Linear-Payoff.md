---
layout: post
title: "[Paper Review] A Contextual-Bandit Approach to Personalized News Article Recommendation"
categories: [Paper Review]
tags: [paper-review, reinforcement-learning, contextual-bandits, exploration-exploitation]
math: true
---

## Introduction

웹 서비스에서 개인화 추천은 사용자의 문맥(context) 정보를 활용하여 최적의 콘텐츠를 선택하는 문제로 정의할 수 있다. 기존의 추천 시스템은 대부분 **탐색(exploration)과 활용(exploitation)의 균형**을 체계적으로 다루지 못한다. 이 논문은 **K-armed contextual bandit** 프레임워크를 도입하여 개인화 뉴스 추천 문제를 정식화하고, **LinUCB** 알고리즘을 제안한다. Yahoo! Front Page의 대규모 실험에서 기존 방법 대비 CTR 12.5% 향상을 달성하였다.

이 글은 논문의 핵심 알고리즘인 LinUCB를 **통계학의 회귀 분석 지식만을 전제로** 단계적으로 유도하는 데 목적이 있다. 강화학습 사전 지식이 많지 않더라도, 익숙한 ridge regression에서 출발하여 confidence ellipsoid, Optimism in the Face of Uncertainty, regret bound까지 자연스럽게 확장할 수 있도록 구성하였다.

---

## 1. 문제 정의 — "온라인 선택 문제로서의 회귀"

### 1.1 상호작용 루프

매 시점 $t = 1, 2, \ldots, T$에서 다음이 반복된다.

1. 환경이 context $x_t \in \mathbb{R}^d$를 제시 (사용자 프로필, 시간, 디바이스 등)
2. 에이전트가 action $a_t \in \mathcal{A} = \{1, \ldots, K\}$를 선택 (예: 어떤 뉴스 기사를 보여줄 것인가)
3. 환경이 reward $r_t \in \mathbb{R}$를 반환 (예: 클릭 여부, 평점)
4. 에이전트는 $(x_t, a_t, r_t)$를 관찰하고 다음 라운드로 이동

목표는 누적 보상을 최대화하는 것이며, 등가적으로 **누적 리그렛(cumulative regret)** 을 최소화하는 것이다:

$$
R_T = \mathbb{E}\left[\sum_{t=1}^{T} r_{t, a_t^*}\right] - \mathbb{E}\left[\sum_{t=1}^{T} r_{t, a_t}\right]
$$

여기서 $a_t^*$는 매 시점의 최적 action이다.

### 1.2 핵심 가정 — Linear Realizability

LinUCB는 보상의 조건부 기대값이 context에 대해 **선형**이라 가정한다:

$$
\mathbb{E}[r_t \mid x_t, a_t = a] = x_t^\top \theta_a^*, \qquad \theta_a^* \in \mathbb{R}^d
$$

즉 **각 action마다 독립된 선형 회귀 모델**이 존재한다고 보는 것이다. 추가로 noise는 conditionally $R$-sub-Gaussian이라 가정한다:

$$
\mathbb{E}[\exp(\lambda(r_t - x_t^\top \theta_{a_t}^*)) \mid \mathcal{F}_{t-1}] \le \exp(\lambda^2 R^2 / 2), \quad \forall \lambda \in \mathbb{R}
$$

이 가정은 뒤에서 유도할 self-normalized concentration inequality의 전제 조건이다. Gaussian noise는 물론, 유계(bounded) noise도 자동으로 이 클래스에 속한다.

### 1.3 지도학습과 무엇이 다른가

통계학 관점에서 보면 이것은 사실상 **$K$개의 병렬 선형 회귀 문제**이다. 그러나 결정적 차이가 하나 있다.

> **관측 분포 자체가 에이전트의 정책에 의존한다.**

전형적인 지도학습에서는 $(X, Y)$가 외부 분포에서 i.i.d.로 주어진다. 반면 Bandit에서는 **어떤 $(x_t, a)$ 쌍을 관측할지 에이전트가 직접 선택한다.** 이 순간부터 "고정된 분포"라는 전제가 깨지고, 고전적 OLS/ridge의 이론적 성질이 그대로 성립하지 않는다. 이것이 Bandit 이론에서 regret 분석과 **adaptive concentration inequality**가 반드시 필요한 근본 이유이다.

---

## 2. 왜 단순 방법으로는 충분하지 않은가

### 2.1 ε-Greedy의 문제

확률 $\epsilon$으로 무작위 action, $1-\epsilon$로 현재 추정 최적 action을 선택하는 방법이다. 구현은 단순하지만 치명적 결함이 있다.

- **문맥 정보를 탐색에 활용하지 못한다.** 어떤 방향이 불확실한지 모르는 채 무차별적으로 랜덤 선택한다.
- 이미 확실하게 알고 있는 영역에서도 $\epsilon$만큼 계속 낭비하게 된다.

### 2.2 전통 UCB의 문제

각 action별 보상의 상한을 추정하여 "불확실한 action"을 우선 탐색하는 UCB1 (Auer et al., 2002)은 ε-greedy보다 훨씬 효율적이지만, **context를 전혀 고려하지 않는다.** 동일한 action이라도 사용자/상황에 따라 최적성이 달라지는 개인화 추천 문제에는 부적합하다.

LinUCB는 이 두 접근의 장점을 결합한다 — **선형 회귀로 context를 활용**하면서, **UCB 원리로 불확실성 기반 탐색**을 수행한다.

---

## 3. Ridge Regression으로부터의 유도

### 3.1 Offline Ridge

Action $a$가 선택된 라운드만 따로 모으면 다음과 같다.

$$
X_a \in \mathbb{R}^{n_a \times d}, \qquad y_a \in \mathbb{R}^{n_a}
$$

여기서 $n_a$는 지금까지 action $a$가 선택된 횟수, $X_a$는 해당 시점들의 context를 쌓은 설계 행렬, $y_a$는 관측된 보상 벡터이다. Ridge estimator는 통계 시간에 배우는 그것 그대로이다:

$$
\hat{\theta}_a = \arg\min_\theta \left\{ \|y_a - X_a \theta\|_2^2 + \lambda \|\theta\|_2^2 \right\} = (X_a^\top X_a + \lambda I_d)^{-1} X_a^\top y_a
$$

표기의 편의를 위해 다음과 같이 정의한다.

$$
V_a := X_a^\top X_a + \lambda I_d \in \mathbb{R}^{d \times d}, \qquad b_a := X_a^\top y_a \in \mathbb{R}^d
$$

$$
\hat{\theta}_a = V_a^{-1} b_a
$$

### 3.2 Online (Recursive) Ridge

Bandit은 데이터가 한 점씩 순차적으로 들어오므로, 매번 처음부터 다시 ridge를 풀 필요 없이 **재귀적 업데이트**가 가능하다. 새 관측 $(x_t, r_t)$가 들어오면:

$$
\boxed{\; V_a \leftarrow V_a + x_t x_t^\top, \qquad b_a \leftarrow b_a + r_t x_t \;}
$$

이 **두 줄이 LinUCB 학습의 전부**이다. 즉 LinUCB의 학습 과정은 recursive ridge regression에 지나지 않는다. 실무에서는 Sherman–Morrison 항등식으로 $V_a^{-1}$까지 $\mathcal{O}(d^2)$에 직접 업데이트할 수 있다:

$$
V_a^{-1} \leftarrow V_a^{-1} - \frac{V_a^{-1} x_t x_t^\top V_a^{-1}}{1 + x_t^\top V_a^{-1} x_t}
$$

> **이 섹션에서 얻어야 할 것**: LinUCB의 학습 부분은 새로운 알고리즘이 아니다. 통계에서 익숙한 ridge를 온라인화한 것뿐이다. 진짜 새로운 부분은 다음 섹션부터 등장한다.

---

## 4. Confidence Ellipsoid — UCB가 나오는 곳

### 4.1 Self-Normalized Concentration Inequality

Ridge regression에서 $\hat{\theta}_a$가 참값 $\theta_a^*$로부터 얼마나 떨어질 수 있는지는 고전적 결과로 알려져 있지만, 그것은 i.i.d. 가정을 전제한다. Bandit의 adaptive 데이터 수집 환경에서는 Abbasi-Yadkori, Pál, Szepesvári (NeurIPS 2011)의 **self-normalized martingale inequality**가 필요하다. 적어도 확률 $1-\delta$로 다음이 성립한다:

$$
\|\hat{\theta}_a - \theta_a^*\|_{V_a} \le R\sqrt{d \log\!\left(\frac{1 + n_a/\lambda}{\delta}\right)} + \sqrt{\lambda}\,S =: \beta_t(\delta)
$$

여기서 $\|z\|_V := \sqrt{z^\top V z}$는 $V$-가중 노름이고, $\|\theta_a^*\|_2 \le S$는 참 파라미터의 prior norm bound이다.

**기하적 해석.** 진짜 파라미터 $\theta_a^*$는 중심 $\hat{\theta}_a$, $V_a$-metric 하에서 반경 $\beta_t$인 타원체 안에 놓인다. $V_a$의 eigenvalue가 큰 방향(관측이 많이 쌓인 방향)은 타원체가 좁고, 작은 방향(아직 덜 본 방향)은 넓다. 즉 **"어느 방향을 얼마나 확실히 알고 있는가"가 타원체 모양에 자동으로 새겨진다.**

### 4.2 예측값의 신뢰 구간 → UCB

임의의 context $x$에 대해 예측값 $x^\top \hat{\theta}_a$가 참값 $x^\top \theta_a^*$로부터 얼마나 벗어나는지를 보고 싶다. Cauchy–Schwarz를 $V_a$-노름 하에 적용하면:

$$
|x^\top \hat{\theta}_a - x^\top \theta_a^*| \le \|x\|_{V_a^{-1}} \cdot \|\hat{\theta}_a - \theta_a^*\|_{V_a} \le \beta_t \cdot \|x\|_{V_a^{-1}}
$$

따라서 보상의 **Upper Confidence Bound** 는 다음과 같이 자연스럽게 정의된다.

$$
\boxed{\; \text{UCB}_t(a) = \underbrace{x_t^\top \hat{\theta}_a}_{\text{exploit}} + \underbrace{\alpha \cdot \sqrt{x_t^\top V_a^{-1} x_t}}_{\text{explore}} \;}
$$

이론적으로 $\alpha = \beta_t$로 두면 $1-\delta$ 커버리지가 보장되지만, 실무에서는 상수 하이퍼파라미터로 튜닝하는 경우가 많다 (흔히 $\alpha \in [0.5, 2]$).

### 4.3 통계학적 해석 — Leverage와의 연결

$x^\top V_a^{-1} x$라는 양은 단순한 계산식이 아니다. 이것은 **ridge 예측값의 (스케일된) 분산**에 해당한다. 실제로 ridge estimator의 예측 분산은 다음과 같이 쓰인다:

$$
\text{Var}(x^\top \hat{\theta}_a) = \sigma^2 \, x^\top V_a^{-1} X_a^\top X_a V_a^{-1} x
$$

데이터가 충분히 쌓여 $X_a^\top X_a \approx V_a$이면 $\text{Var}(x^\top \hat{\theta}_a) \approx \sigma^2 \, x^\top V_a^{-1} x$가 된다. 따라서 $\|x\|_{V_a^{-1}}$는 **선형회귀의 hat matrix에 나오는 leverage와 유사한 역할**을 한다. 즉:

> **"지금까지 본 데이터로부터 $x$ 방향의 정보가 얼마나 부족한가"를 측정하는 양.**

부족할수록 예측의 불확실성이 크고, 따라서 탐색 보너스가 커진다. 이것이 LinUCB의 탐색 항이 통계적으로 원리적인 이유이다.

---

## 5. Optimism in the Face of Uncertainty (OFU)

### 5.1 행동 규칙

LinUCB의 선택 규칙은 한 줄이다.

$$
a_t = \arg\max_{a \in \mathcal{A}} \text{UCB}_t(a)
$$

원칙은 **"불확실할 때는 가장 낙관적인 추정값을 믿고 선택하라 (Optimism in the Face of Uncertainty)"**. 이 원리가 왜 원리적으로 작동하는지 잠깐 살펴보자.

### 5.2 Regret Bound 유도 스케치

Confidence ellipsoid가 $\theta_a^*$를 포함하는 "좋은 사건" 하에서, 최적 action $a^*$의 UCB는 반드시 그 진짜 보상 이상이다:

$$
\text{UCB}_t(a^*) \ge x_t^\top \theta_{a^*}^*
$$

LinUCB는 $\text{UCB}_t(a_t) \ge \text{UCB}_t(a^*)$인 $a_t$를 선택하므로, 즉시적 regret은 다음과 같이 bound된다:

$$
\text{regret}_t = x_t^\top \theta_{a^*}^* - x_t^\top \theta_{a_t}^* \le \text{UCB}_t(a_t) - x_t^\top \theta_{a_t}^* \le 2\beta_t \|x_t\|_{V_{a_t}^{-1}}
$$

**핵심은 우변에 등장한 $\|x_t\|_{V_{a_t}^{-1}}$의 누적합이다.** 이것을 다루기 위해 Abbasi-Yadkori et al. (2011, Lemma 11)의 **elliptical potential lemma**가 사용된다. $\|x_t\|_2 \le L$이라 할 때:

$$
\sum_{t=1}^{T} \|x_t\|_{V_{a_t}^{-1}}^2 \le 2d \log\!\left(1 + \frac{TL^2}{\lambda d}\right)
$$

Cauchy–Schwarz를 적용하면 누적 regret은 다음과 같이 결론난다.

$$
\boxed{\; R_T = \sum_{t=1}^{T} \text{regret}_t \le \tilde{\mathcal{O}}(d\sqrt{T}) \;}
$$

**해석.** Regret은 context 차원 $d$에 선형, 시간 $T$에 루트로 증가한다. 즉 시간이 지날수록 평균적 실수는 $R_T / T \to 0$으로 줄어든다는 의미이다. 이것이 LinUCB의 간판 결과이다.

---

## 6. Elliptical Potential Lemma의 직관

앞 섹션에서 마치 마술처럼 등장한 bound의 본질을 짚고 넘어가자. 이 lemma는 LinUCB의 작동 원리를 가장 직관적으로 보여준다.

$V_a$는 매 step마다 rank-1 업데이트 $x_t x_t^\top$씩 커지므로 $\det(V_a)$는 단조 증가한다. 행렬식의 rank-1 업데이트 공식에 의해:

$$
\det(V_a + x x^\top) = \det(V_a) \cdot (1 + x^\top V_a^{-1} x)
$$

양변에 $\log$를 취하고 전체 시간에 대해 telescoping하면:

$$
\log\det(V_T) - \log\det(V_0) = \sum_{t=1}^{T} \log(1 + \|x_t\|_{V_{a_t}^{-1}}^2)
$$

그리고 $u \le 1$에 대해 $\log(1 + u) \ge u/2$이므로,

$$
\sum_{t=1}^{T} \|x_t\|_{V_{a_t}^{-1}}^2 \lesssim \log\det(V_T) \lesssim d \log T
$$

이 결과가 말하는 바는 한 줄로 요약된다.

> **큰 탐색 보너스 $\|x_t\|_{V_{a_t}^{-1}}^2$는 자주 발생할 수 없다.**

왜냐하면 그런 step은 $V_a$의 determinant를 크게 증가시키고, 그 결과 다음 step부터는 해당 방향의 불확실성이 즉시 줄어들기 때문이다. 이것을 **self-limiting 성질**이라 부른다. 탐색 보너스가 스스로를 소진시키는 구조 — 이것이 UCB 계열 알고리즘이 왜 "무작정 탐색"이 아니라 **원리적으로 효율적**인지에 대한 본질이다.

---

## 7. 탐색–활용 Trade-off의 기하학적 본질

여기까지의 내용을 한 그림으로 요약하면, LinUCB는 매 시점 다음 두 양을 동시에 관리하는 알고리즘이다.

| 항 | 의미 | 역할 |
|---|---|---|
| $x^\top \hat{\theta}_a$ | 현재 추정값 | **Exploit** — 지금까지의 최선 |
| $\|x\|_{V_a^{-1}}$ | 추정값의 불확실성 | **Explore** — 아직 덜 본 방향에 가중 |

핵심은 $V_a^{-1}$이 **"어떤 context 방향을 얼마나 봤는가"를 데이터로부터 자동으로 기록**한다는 점이다. 새로운 유형의 context가 들어오면 해당 방향에서 $\|x\|_{V_a^{-1}}$가 커져 자연스럽게 탐색을 유도하고, 이미 충분히 관측된 방향이라면 작아져 exploit으로 수렴한다.

이것이 $\epsilon$-greedy와 결정적으로 다른 점이다. $\epsilon$-greedy는 "어디가 불확실한지" 전혀 알지 못한 채 고정 비율로 무차별 탐색한다. 반면 LinUCB는 **정보 기하학적으로 필요한 방향에만 탐색**한다. 같은 예산으로 훨씬 많은 정보를 얻는 셈이다.

---

## 8. Hybrid 모델 — 논문의 또 다른 기여

지금까지 설명한 것은 각 action이 독립된 파라미터 $\theta_a^*$를 가지는 **disjoint 모델**이다. Li et al. (2010)은 여기에 한 걸음 더 나아가, action 간 **공통 파라미터 $\beta^*$** 를 공유하는 hybrid 모델을 제안하였다.

$$
\mathbb{E}[r_{t,a} \mid x_{t,a}, z_{t,a}] = z_{t,a}^\top \beta^* + x_{t,a}^\top \theta_a^*
$$

여기서 $z_{t,a}$는 사용자와 action의 상호작용 특성을 나타내는 **공유 feature**이다 (예: 사용자 성별 × 기사 카테고리 교차 feature). 이 구조의 장점은 두 가지이다.

- **정보 공유.** 모든 action 데이터가 $\beta^*$ 추정에 기여하므로, 개별 action의 관측 부족을 보완한다.
- **Cold-start 개선.** 한 번도 본 적 없는 새로운 action이라도 $z^\top \beta^*$ 부분을 통해 즉시 합리적 예측이 가능하다.

Ridge regression의 closed-form 업데이트는 disjoint 모델보다 약간 복잡해지지만 (block matrix inversion), 본질은 같다 — **공유 부분과 개별 부분을 함께 푸는 확장된 ridge regression**이다.

---

## 9. Bayesian 쌍둥이 — Thompson Sampling

LinUCB와 수학적 뿌리가 **완전히 동일**한 알고리즘이 하나 있다. Bayesian 관점을 채택하면 자연스럽게 등장하는 Thompson Sampling이다.

**Prior:** $\theta_a \sim \mathcal{N}(0, \lambda^{-1} I)$
**Likelihood:** $r_t \mid x_t, \theta_{a_t} \sim \mathcal{N}(x_t^\top \theta_{a_t}, \sigma^2)$

Conjugate Gaussian 관계에 의해 posterior는 다음과 같이 닫힌 형태로 나온다.

$$
\theta_a \mid \mathcal{D}_t \sim \mathcal{N}(\hat{\theta}_a, \sigma^2 V_a^{-1})
$$

**중심도 $\hat{\theta}_a$, 공분산도 $\sigma^2 V_a^{-1}$** — LinUCB에 나온 양들과 정확히 같다. 단지 둘의 사용 방식이 다를 뿐이다.

- **LinUCB**: posterior의 **낙관적 상한**을 계산해서 argmax
- **Thompson Sampling**: posterior에서 **한 번 sampling**해서 그 sample에 대해 greedy

알고리즘 절차로 쓰면:

1. 각 $a$에 대해 $\tilde{\theta}_a \sim \mathcal{N}(\hat{\theta}_a, \sigma^2 V_a^{-1})$ 샘플
2. $a_t = \arg\max_a x_t^\top \tilde{\theta}_a$

탐색은 posterior의 분산에서 자동으로 주입된다. Agrawal & Goyal (ICML 2013)의 결과에 따르면 regret은 $\tilde{\mathcal{O}}(d\sqrt{T})$로 LinUCB와 같은 order이며, 상수만 다르다.

### LinUCB vs Thompson Sampling

| 관점 | LinUCB | Thompson Sampling |
|---|---|---|
| **성격** | Deterministic (같은 입력 → 같은 출력) | Stochastic (매번 다른 샘플) |
| **튜닝** | 탐색 상수 $\alpha$ | Noise 분산 $\sigma^2$ |
| **이론** | Frequentist confidence set | Bayesian posterior |
| **실무** | 디버깅/재현 쉬움 | $K$ 클 때 성능 우위 보고 많음 |
| **공유 자원** | $V_a, b_a, \hat{\theta}_a$ | 동일 — 구현 비용 거의 동일 |

---

## 10. 오프라인 평가 방법론

논문의 또 다른 기여는 A/B 테스트 없이 **로그 데이터만으로 새 정책을 평가**하는 방법론이다. 무작위 정책으로 수집된 기록 $(x_t, a_t, r_t)$가 있을 때, 새 정책 $\pi$의 기대 보상은 다음과 같이 불편 추정된다.

$$
\hat{V}_\pi = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} r_t \cdot \mathbf{1}[\pi(x_t) = a_t]
$$

즉 "새 정책이 당시에 실제로 선택한 action과 같았던 시점들"만 골라서 해당 보상의 평균을 취한다. 무작위 로깅 정책 하에서는 이것이 진짜 기대 보상의 unbiased estimator가 된다.

이 방법론은 이후 **Inverse Propensity Scoring (IPS)**, **Doubly Robust estimator** 등으로 발전하여 off-policy evaluation 분야의 기초가 되었다. 로그가 완전 무작위가 아닐 때는 기록 시점의 action 선택 확률(propensity)로 가중치를 보정해야 한다.

---

## 11. 실험 결과

Yahoo! Today Module의 실제 트래픽(4,500만 이벤트)에서 실험한 결과는 다음과 같다.

| 알고리즘 | 상대 CTR 향상 |
|---------|-------------|
| Random | 기준선 |
| $\epsilon$-Greedy (Disjoint) | +7.1% |
| LinUCB (Disjoint) | +10.3% |
| **LinUCB (Hybrid)** | **+12.5%** |

Hybrid 모델이 가장 우수하며, 탐색 파라미터 $\alpha$의 적절한 설정이 결정적임이 확인되었다. 문맥 정보를 활용한 **체계적 탐색**이 $\epsilon$-greedy의 무차별 탐색 대비 상당한 성능 차이를 만든다.

---

## 12. 이론적 한계와 확장

LinUCB는 우아한 이론과 실용성을 겸비한 알고리즘이지만, 그 핵심 가정들을 벗어나는 상황에서는 확장이 필요하다.

### 12.1 Adaptive Data Collection으로 인한 Bias
$x_t, a_t$가 과거에 의존하므로 i.i.d. 기반 고전 OLS 이론이 그대로 성립하지 않는다. 반드시 self-normalized martingale 기반 confidence set을 사용해야 하며, 일반 ridge CI 공식으로 대체하면 under-coverage가 발생한다.

### 12.2 선형 가정 위반
실제 보상이 context에 대해 비선형일 경우 linear realizability가 무너진다. 대응책:
- **Kernel LinUCB**: RKHS로 확장하여 비선형 관계를 암묵적으로 표현
- **Neural LinUCB** (Zhou, Li, Gu, ICML 2020): 심층 신경망의 최종층을 선형 bandit으로 간주

### 12.3 Non-stationarity
사용자 취향이나 콘텐츠 분포가 시간에 따라 변하면 stationary 가정이 깨진다. 대응책:
- **Discounted LinUCB**: $V_a \leftarrow \gamma V_a + x x^\top$ with $\gamma < 1$ — 과거 관측을 지수적으로 잊음
- **Sliding Window LinUCB**: 최근 $W$개 관측만 유지

### 12.4 Reward Scale
LinUCB의 regret bound는 $|r_t| \le 1$ 가정에 의존한다. 실무에서는 보상을 $[0, 1]$로 정규화하는 것이 전제이다. 큰 스케일 보상을 그대로 넣으면 confidence ellipsoid의 $R$ 값이 커져 탐색이 과도하게 부풀려진다.

### 12.5 고차원 Context와 Sparse 확장
Regret bound가 $d$에 **선형**이라는 점은 $d \gg T$인 고차원 응용에서 LinUCB를 사실상 무용지물로 만든다. 통계학의 sparse regression 도구(LASSO)를 bandit에 결합하면 $d$ 의존성을 polylog로 축소할 수 있다:
- **LASSO Bandit** (Bastani & Bayati, 2020): Forced sampling + dual LASSO estimator로 regret $\mathcal{O}(s_0^2 (\log T + \log d)^2)$ 달성 → [논문 리뷰]({% post_url 2026-04-05-Online-Decision-High-Dimensional-LASSO-Bandit %})
- **Doubly-Robust LASSO Bandit** (Kim & Paik, NeurIPS 2019): Missing data 처리의 doubly-robust 기법으로 forced sampling을 제거하고 $K$-free bound 달성 → [논문 리뷰]({% post_url 2026-04-05-Doubly-Robust-LASSO-Bandit %})

---

## 13. Conclusion

LinUCB는 한 줄로 정리하면 다음과 같다.

> **각 action마다 ridge regression을 온라인으로 돌리면서, 예측값의 confidence ellipsoid 반지름을 탐색 보너스로 더해 선택하는 알고리즘.**

핵심 구성 요소 세 가지:

1. **학습**: $V_a \leftarrow V_a + xx^\top$, $\; b_a \leftarrow b_a + rx$, $\; \hat{\theta}_a = V_a^{-1} b_a$ — 재귀적 ridge regression
2. **불확실성 정량화**: $\|x\|_{V_a^{-1}}$ — ridge 예측 분산과 leverage의 일반화
3. **행동 규칙**: $a_t = \arg\max_a \{x_t^\top \hat{\theta}_a + \alpha \|x_t\|_{V_a^{-1}}\}$ — Optimism in the Face of Uncertainty

이로부터 $\tilde{\mathcal{O}}(d\sqrt{T})$의 regret bound가 elliptical potential lemma를 통해 유도된다. 이 결과의 본질은 **"큰 탐색 보너스는 자기-한정적으로 감소한다"** 는 기하학적 성질에 있으며, 이것이 단순한 랜덤 탐색과 원리적으로 구분되는 지점이다.

나아가 Bayesian 쌍둥이인 Thompson Sampling은 정확히 동일한 $V_a, b_a$를 사용하면서 posterior sampling이라는 다른 경로로 탐색을 달성한다. 두 방법 모두 통계학의 선형 모델이 강화학습의 온라인 의사결정 문제로 확장될 때 어떻게 자연스럽게 연결되는지를 보여주는 좋은 예시이다.

---

## References

- Li, L., Chu, W., Langford, J., & Schapire, R. E. "A Contextual-Bandit Approach to Personalized News Article Recommendation." *WWW 2010*.
- Abbasi-Yadkori, Y., Pál, D., & Szepesvári, C. "Improved Algorithms for Linear Stochastic Bandits." *NeurIPS 2011*.
- Agrawal, S., & Goyal, N. "Thompson Sampling for Contextual Bandits with Linear Payoffs." *ICML 2013*.
- Auer, P., Cesa-Bianchi, N., & Fischer, P. "Finite-time Analysis of the Multiarmed Bandit Problem." *Machine Learning*, 2002.
- Zhou, D., Li, L., & Gu, Q. "Neural Contextual Bandits with UCB-based Exploration." *ICML 2020*.
- Lattimore, T., & Szepesvári, C. *Bandit Algorithms*. Cambridge University Press, 2020.
