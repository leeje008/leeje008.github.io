---
layout: post
title: "[파이썬 아키텍처] CH01 - Domain Modeling"
categories: [Study Note]
tags: [python, architecture, ddd, domain-model, value-object, entity]
math: false
---

## Introduction

`Architecture Patterns with Python` (Harry Percival & Bob Gregory, O'Reilly 2020, 일명 *Cosmic Python*)은 DDD(Domain-Driven Design), TDD, Event-Driven Architecture를 파이썬 생태계에 어떻게 녹여낼 것인지를 다룬 책이다. 1장은 책 전체에서 가장 중요한 출발점인 **도메인 모델(Domain Model)** 을 다룬다.

핵심 명제는 다음과 같다.

> 코드 중에서 **비즈니스에 가장 가깝고, 가장 자주 변경되며, 가장 큰 가치를 만들어내는 부분**이 도메인 모델이다. 따라서 도메인 모델은 인프라(DB, 웹 프레임워크, 외부 API)로부터 격리되어 **순수하고, 테스트하기 쉽고, 비즈니스 언어로 읽혀야 한다**.

본 챕터에서 사용하는 예제는 가구 회사 MADE.com을 모티브로 한 **재고 할당(allocation)** 도메인이다. 여러 `Batch`(입고 배치)가 있고, 들어오는 `OrderLine`(주문 라인)을 적절한 배치에 할당해야 한다.

---

## 1. Domain Model이란 무엇인가

도메인은 "여러분이 해결하려는 문제"이며, 모델은 "유용한 속성을 포착한 그 문제의 지도"이다. 도메인 모델은 비즈니스 전문가가 머릿속에 가지고 있는 **비즈니스 프로세스나 현상에 대한 정신적 지도(mental map)** 를 코드로 옮긴 것이다.

### Ubiquitous Language

DDD의 핵심 원칙 중 하나는 **유비쿼터스 언어(ubiquitous language)** 이다. 비즈니스 전문가가 사용하는 용어를 그대로 클래스명, 메서드명, 변수명에 사용한다. 예제 도메인의 어휘:

- **Product**: 식별자 SKU(Stock Keeping Unit)로 구분되는 상품
- **Customer**: 주문(`Order`)을 넣는 주체. 주문은 여러 `OrderLine`(SKU + 수량)을 갖는다
- **Batch**: 구매 부서가 발주한 입고 단위. 고유 reference, SKU, 수량, ETA(도착 예정일)를 가진다
- **Allocate**: `OrderLine`을 특정 `Batch`에 할당. 할당된 만큼 가용 수량(`available_quantity`)이 줄어든다

비즈니스 규칙:
1. 같은 `OrderLine`을 두 `Batch`에 할당할 수 없다 (idempotent).
2. SKU가 다르면 할당할 수 없다.
3. 가용 수량이 부족하면 할당할 수 없다.
4. 현재 재고가 있는 batch(`eta=None`)를 미래 도착 batch보다 우선 할당한다.

---

## 2. Unit Testing Domain Models

도메인 모델은 외부 의존성이 없으므로 **테스트가 가장 단순한 영역**이다. 테스트 이름과 변수명이 비즈니스 어휘를 그대로 반영해야, 비개발자도 테스트만 보고 비즈니스 규칙을 검증할 수 있다.

```python
def test_allocating_to_a_batch_reduces_the_available_quantity():
    batch = Batch("batch-001", "SMALL-TABLE", qty=20, eta=date.today())
    line = OrderLine("order-ref", "SMALL-TABLE", 2)
    batch.allocate(line)
    assert batch.available_quantity == 18
```

이 테스트가 통과하도록 다음과 같은 모델을 작성한다.

```python
@dataclass(frozen=True)
class OrderLine:
    orderid: str
    sku: str
    qty: int

class Batch:
    def __init__(self, ref: str, sku: str, qty: int, eta: Optional[date]):
        self.reference = ref
        self.sku = sku
        self.eta = eta
        self._purchased_quantity = qty
        self._allocations: Set[OrderLine] = set()

    def allocate(self, line: OrderLine):
        if self.can_allocate(line):
            self._allocations.add(line)

    def deallocate(self, line: OrderLine):
        if line in self._allocations:
            self._allocations.remove(line)

    @property
    def allocated_quantity(self) -> int:
        return sum(line.qty for line in self._allocations)

    @property
    def available_quantity(self) -> int:
        return self._purchased_quantity - self.allocated_quantity

    def can_allocate(self, line: OrderLine) -> bool:
        return self.sku == line.sku and self.available_quantity >= line.qty
```

핵심 설계 결정:
- `_allocations`를 **set**으로 두어 idempotency를 자연스럽게 보장
- `available_quantity`를 **계산된 property**로 만들어 모순 상태 자체를 표현 불가능하게 함
- `OrderLine`은 `frozen=True` dataclass로 **불변(immutable)** 하게 만듦

---

## 3. Value Object와 Entity의 구분

DDD의 가장 기초적이지만 중요한 구분이다.

| 구분 | Value Object | Entity |
|------|--------------|--------|
| 정체성 | 데이터 자체로 정의 | 시간에 걸친 영속적 정체성 |
| 변경 | 불변(immutable) | 가변(mutable) |
| 동등성 | 모든 속성이 같으면 동등 | 식별자(reference)가 같으면 동등 |
| 예시 | `Money('gbp', 10)`, `OrderLine` | `Batch`, `Person` |

### Value Object 구현

`@dataclass(frozen=True)` 또는 `NamedTuple`을 사용하면 값 동등성과 불변성을 거저 얻는다.

```python
@dataclass(frozen=True)
class Money:
    currency: str
    value: int

assert Money('gbp', 10) == Money('gbp', 10)  # True — 값 동등성
```

Value Object 위에 **수학 연산**을 정의하는 것도 자연스럽다.

```python
fiver = Money('gbp', 5)
tenner = Money('gbp', 10)
assert fiver + fiver == tenner
```

### Entity 구현

`Batch`는 동일한 reference를 가지면 다른 속성이 바뀌어도 같은 batch이다. 따라서 `__eq__`와 `__hash__`를 reference 기반으로 명시적으로 구현한다.

```python
class Batch:
    ...
    def __eq__(self, other):
        if not isinstance(other, Batch):
            return False
        return other.reference == self.reference

    def __hash__(self):
        return hash(self.reference)
```

> **주의**: `__hash__`를 수정할 때는 `__eq__`도 함께 수정해야 한다. Entity의 hash는 정체성 속성(예: `reference`)을 기반으로 하고, 그 속성은 가능하면 read-only로 만든다.

---

## 4. Domain Service: 모든 것이 객체일 필요는 없다

> Sometimes, it just isn't a thing. — Eric Evans

`OrderLine`을 **여러 Batch 중 적절한 하나**에 할당하는 동작은 어떤 단일 entity에도 자연스럽게 속하지 않는다. 이런 경우 Evans는 **Domain Service**로 분리하라고 한다.

파이썬은 멀티패러다임 언어이므로, "동사"는 굳이 클래스로 감싸지 말고 **그냥 함수로** 만든다. `FooManager`, `BarBuilder`보다는 `manage_foo()`, `build_bar()`가 더 표현력이 높다.

```python
def allocate(line: OrderLine, batches: List[Batch]) -> str:
    try:
        batch = next(b for b in sorted(batches) if b.can_allocate(line))
    except StopIteration:
        raise OutOfStock(f'Out of stock for sku {line.sku}')
    batch.allocate(line)
    return batch.reference
```

> Domain Service ≠ Service Layer Service. Domain service는 **비즈니스 개념**을, service-layer service는 **유스케이스(use case)** 를 표현한다 (4장에서 다룸).

---

## 5. Magic Method로 도메인 의미 표현하기

`sorted(batches)`가 "현재 재고 우선, 그 다음은 ETA가 빠른 순"으로 정렬되도록 `__gt__`를 정의한다. 비교 연산자에 도메인 규칙(우선순위)을 자연스럽게 녹인다.

```python
class Batch:
    def __gt__(self, other):
        if self.eta is None:
            return False         # 현재 재고는 항상 우선
        if other.eta is None:
            return True
        return self.eta > other.eta
```

이렇게 하면 호출 측 코드(`sorted(batches)`)는 비즈니스 규칙을 모르고도 올바르게 동작한다.

---

## 6. Exception도 도메인 개념이다

비즈니스 전문가와의 대화에서 등장하는 "재고 부족(out of stock)"이라는 **개념** 자체를 코드에 그대로 옮긴다.

```python
class OutOfStock(Exception):
    pass
```

예외 이름은 entity, value object, service와 마찬가지로 **유비쿼터스 언어**의 일부이다. `RuntimeError("no batch")` 같은 익명 예외가 아니라, 도메인 어휘인 `OutOfStock`을 던져야 한다.

---

## 요약 및 다음 장 연결

**1장 핵심 정리**
- 도메인 모델은 비즈니스에 가장 가까운 코드. 인프라로부터 분리해 순수하게 유지한다
- **Value Object**는 불변·값 동등성, **Entity**는 가변·정체성 동등성
- 모든 것이 객체일 필요는 없다 — 동사는 함수(domain service)로
- `__eq__`, `__hash__`, `__gt__` 같은 magic method로 도메인 의미를 idiomatic Python에 녹인다
- 도메인 예외도 유비쿼터스 언어의 일부

**다음 장 예고**
이상적인 도메인 모델을 만들었지만, 실제 시스템은 데이터를 어딘가에 영속(persist)해야 한다. 2장에서는 **Repository 패턴**을 통해 도메인 모델을 데이터베이스로부터 분리하는 방법을 다룬다. 핵심 도구는 **의존성 역전 원칙(DIP)** 이다.
