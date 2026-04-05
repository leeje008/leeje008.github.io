---
layout: post
title: "[추천시스템] CH01 - An Introduction to Recommender Systems"
categories: [Study Note]
tags: [recommender-system, overview]
math: true
---

## Introduction

추천시스템은 사용자의 과거 행동, 선호, 아이템 속성 등을 활용하여 사용자가 관심을 가질 만한 아이템을 자동으로 제안하는 시스템이다. 정보 과부하(information overload) 시대에 사용자가 방대한 선택지에서 의미 있는 아이템을 발견할 수 있도록 돕는 것이 핵심 목표다.

---

## 평점 행렬 (Ratings Matrix)

추천시스템의 기본 입력은 $m$명의 사용자와 $n$개의 아이템으로 구성된 평점 행렬이다:

$$
R \in \mathbb{R}^{m \times n}
$$

- $r_{uj}$: 사용자 $u$가 아이템 $j$에 부여한 평점
- 대부분의 항목이 **결측(missing)**이며, 관측된 평점은 전체의 극히 일부

추천시스템의 목표는 이 불완전한 행렬의 빈 항목을 예측하여 높은 예측 평점을 가진 아이템을 추천하는 것이다.

### 평점 유형

| 유형 | 설명 | 예시 |
|------|------|------|
| **Interval-based** | 연속적 구간 값 | 1~5점 별점 |
| **Ordinal** | 순서가 있는 범주 | "싫어요 < 보통 < 좋아요" |
| **Binary** | 이진 선호 | 좋아요/싫어요, 찬성/반대 |
| **Unary** | 상호작용 유무만 관측 | 클릭, 구매, 조회 (암시적 피드백) |

Unary 데이터에서는 관측되지 않은 항목이 "싫어함"인지 "아직 모름"인지 구분할 수 없다는 점에서 다른 유형과 본질적으로 다르다.

---

## 추천시스템의 기본 모델

### Collaborative Filtering (협업 필터링)

**사용자-아이템 간 상호작용 패턴**만을 활용한다. 아이템의 속성 정보 없이, 유사한 취향의 사용자가 좋아한 아이템을 추천하는 방식이다.

- **Neighborhood-Based**: 유사 사용자/아이템을 직접 찾아 예측 (CH02)
- **Model-Based**: 잠재 요인 모델(행렬 분해 등)을 학습하여 예측 (CH03)

협업 필터링은 **결측값 분석(missing value analysis)**의 일반화로 볼 수 있다. 또한 분류(classification)와 회귀(regression)의 일반화이기도 하다. 하나의 열을 타겟으로 보면 나머지 열이 특징 변수가 되어, 기존 지도학습 프레임워크와 동일한 구조를 가진다.

### Content-Based Filtering (콘텐츠 기반 필터링)

아이템의 **속성(attribute)**과 사용자의 과거 선호를 기반으로 추천한다. 사용자가 좋아한 아이템과 속성이 유사한 아이템을 제안한다. 텍스트 기반 아이템에서는 TF-IDF, 키워드 매칭 등이 활용된다. (CH04)

### Knowledge-Based Systems (지식 기반 시스템)

사용자의 **명시적 요구사항**과 도메인 지식을 활용하여 아이템을 추천한다. 평점 이력이 없어도 작동하므로 cold-start 상황에 강하다. Constraint-Based와 Case-Based로 나뉜다. (CH05)

### Demographic Filtering (인구통계 기반 필터링)

사용자의 나이, 성별, 직업 등 **인구통계학적 속성**을 기반으로 유사 그룹의 선호를 추천에 ��용한다. 단독으로 사용되기보다 다른 모델의 보조 정보로 활용되는 경우가 많다.

### Hybrid and Ensemble Systems

위의 여러 모델을 **결합**하여 개별 모델의 약점을 보완한다. Weighted, Switching, Cascade, Feature Augmentation 등 다양한 결합 전략이 있다. (CH06)

---

## 도메인별 챌린지

### Context-Based 추천

시간, 장소, 동반자, 기기 등 **상황(context)** 정보를 반영한다. 동일한 사용자라도 상황에 따라 선호가 달라��� 수 있다.

### Time-Sensitive 추천

사용자 선호는 **시간에 따라 변화(drift)**한다. 최신 상호작용에 더 큰 가중치를 부여하는 시간 감쇠(temporal decay) 모델이 필요하다.

### Location-Based 추천

사용자의 **현재 위치**를 기반으로 주변 레스토랑, 관광지 등을 추천한다. 지리적 근접성이 핵심 신호가 된다.

### Social 추천

소셜 네트워크의 **친구 관계와 신뢰(trust)**를 추천에 반영한다. 친구가 좋아한 아이템, 소셜 태깅 정보 등을 활용하며, 신뢰 기반 추천은 공격에 더 강건하��.

---

## 고급 주제

### Cold-Start 문제

새로운 사용자(user cold-start) 또는 새로운 아이템(item cold-start)에 대해 충분한 상호작용 데이터가 없어 추천이 어려운 문제다. Content-Based나 Knowledge-Based 접근이 대안이 되며, 하이브리드 전략이 효과적이다.

### 공격 저항성 (Attack Resistance)

악의적 사용자가 가짜 평점을 삽입하여 특정 아이템의 추천 순위를 조작하려는 **셸링 공격(shilling attack)**에 대한 방어가 필요하다. 이상 탐지와 강건한 모델 설계가 핵심이다.

### 그룹 추천 (Group Recommendation)

여러 사용자로 구성된 그룹에게 공동으로 만족할 수 있는 아이템을 추천하는 문제다. 개인 선호의 집계 전략(평균, 최소 불만족 등)이 핵심이다.

### 프라이버시 (Privacy)

추천을 위해 수집하는 사용자 데이터는 민감한 개인정보를 포함할 수 있다. 차분 프라이버시(differential privacy), 연합 학습(federated learning) 등의 기법으로 프라이버시를 보호하면서 추천 품질을 유지하는 연구가 진���되고 있다.

---

## 추천시스템 모델 비교

| 모델 | 입력 데이터 | 장점 | 약점 |
|------|-----------|------|------|
| **Collaborative** | 평점/상호작용 | 도메인 독립적 | Cold-start, 희소성 |
| **Content-Based** | 아이템 속성 | 새 아이템 추천 가능 | 과도한 특화(overspecialization) |
| **Knowledge-Based** | 사용자 요구사항 + 도메인 지식 | Cold-start 면역 | 지식 획득 비용 |
| **Demographic** | 인구통계 속성 | 데이터 요구 최소 | 개인화 수준 낮음 |
| **Hybrid** | 복합 | 개별 모델 약점 보완 | 설계 복잡성 |

---

## Reference

- Aggarwal, C.C. *Recommender Systems: The Textbook*. Springer, 2016. Chapter 1.
