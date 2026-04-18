---
layout: post
title: "[파이썬 아키텍처] CH07 - Aggregates and Consistency Boundaries"
categories: [Study Note]
tags: [python, architecture, ddd, aggregate, concurrency, optimistic-locking, bounded-context]
math: false
---

## Introduction

Part I의 마지막 장이자 Part II로의 다리. 지금까지의 패턴(Repository, Service Layer, UoW)은 **누가 데이터를 변경하는가**를 깔끔히 정리했지만, 여전히 답하지 않은 질문이 있다.

> 동시에 같은 stock에 대해 두 개의 할당 요청이 들어오면 어떻게 되는가? **불변식(invariant)** 을 어떻게 보장하는가?

답은 DDD의 **Aggregate** — 일관성 경계(consistency boundary)를 가진 객체 집합을 단일 단위로 다루는 패턴이다. 본 장은 이 개념을 도입하고, **Optimistic Concurrency** 로 데이터 무결성을 강제하는 방법까지 보여준다.

---

## 1. Invariants, Constraints, Consistency

> **Constraint**: 모델이 도달할 수 있는 상태를 제한하는 규칙
> **Invariant**: 연산이 끝났을 때 항상 참이어야 하는 조건

본 도메인의 두 가지 비즈니스 규칙:

1. **An order line can be allocated to only one batch at a time** — 한 OrderLine은 동시에 두 batch에 할당될 수 없음
2. **We can't allocate to a batch if available quantity is less than line qty** — 가용 수량 ≥ 0이 항상 유지되어야 함

단일 사용자/단일 스레드라면 쉽다. 그러나 **동시성**이 들어오면 어려워진다. 두 요청이 동시에 같은 batch를 읽어서 둘 다 할당을 시도하면 oversell이 발생한다.

가장 단순한 해법: **DB 테이블 락**. 그러나 시간당 수만 건의 주문을 처리한다면 batches 테이블 전체에 락을 걸 수 없다 — 데드락 또는 성능 붕괴.

---

## 2. Aggregate란

> An **AGGREGATE** is a cluster of associated objects that we treat as a unit for the purpose of data changes. — Eric Evans, *Domain-Driven Design*

Aggregate는 다른 도메인 객체를 포함하는 도메인 객체이며, **그 안의 객체를 수정하는 유일한 방법은 aggregate 전체를 로드해 그 위의 메서드를 호출하는 것**이다.

핵심 직관:
- DEADLY-SPOON과 FLIMSY-DESK는 **함께 일관성을 유지할 필요가 없다** → 동시 할당 가능
- 같은 SKU에 대한 batch들은 **함께 일관성을 유지해야 한다** → 같은 aggregate

> Aggregate는 도메인 모델의 **"public" 클래스**이며, 나머지 entity와 value object는 **"private"** 이다. (파이썬의 `_leading_underscore` 관례의 한 단계 위 추상화.)

---

## 3. Aggregate 선택하기

후보:
- `Shipment` — 한 배송에 여러 batch
- `Warehouse` — 한 창고에 여러 batch
- `GlobalSkuStock` — 한 SKU의 모든 batch

처음 둘은 **너무 큰 경계**(같은 창고의 다른 SKU 할당까지 직렬화). 세 번째가 적절한 granularity. 이름이 길어서 `Product`로 명명. 1장에서 첫 번째로 떠올린 도메인 어휘이기도 하다.

```python
class Product:
    def __init__(self, sku: str, batches: List[Batch]):
        self.sku = sku
        self.batches = batches

    def allocate(self, line: OrderLine) -> str:
        try:
            batch = next(
                b for b in sorted(self.batches) if b.can_allocate(line)
            )
            batch.allocate(line)
            return batch.reference
        except StopIteration:
            raise OutOfStock(f'Out of stock for sku {line.sku}')
```

`allocate()` domain service가 `Product.allocate()` 메서드로 흡수된다.

> **Bounded Context**: allocation 서비스의 `Product`는 `(sku, batches)`만 가진다. 이커머스의 `Product`는 `(sku, description, price, image_url, dimensions, ...)`. 같은 단어가 컨텍스트마다 다른 모델을 가진다는 것이 DDD의 핵심 통찰. Microservice 경계와도 자연스럽게 맞물린다.

---

## 4. One Aggregate = One Repository

새로운 규칙: **Repository는 aggregate에 대해서만 만든다.**

```python
class AbstractUnitOfWork(abc.ABC):
    products: repository.AbstractProductRepository

class AbstractProductRepository(abc.ABC):
    @abc.abstractmethod
    def add(self, product): ...

    @abc.abstractmethod
    def get(self, sku) -> model.Product: ...
```

Service layer도 `BatchRepository` 대신 `ProductRepository`를 쓴다.

```python
def allocate(orderid: str, sku: str, qty: int,
             uow: AbstractUnitOfWork) -> str:
    line = OrderLine(orderid, sku, qty)
    with uow:
        product = uow.products.get(sku=line.sku)
        if product is None:
            raise InvalidSku(f'Invalid sku {line.sku}')
        batchref = product.allocate(line)
        uow.commit()
    return batchref
```

이제 외부에서 `Batch`를 직접 만지는 코드가 없다. 모든 변경은 `Product`를 통해 일어난다.

---

## 5. Performance에 대해

> "한 line만 필요한데 모든 batch를 로드하는 게 비효율 아닌가?"

저자의 답:
1. **단일 read + 단일 update**가 ad-hoc 쿼리들보다 보통 더 빠르다
2. 데이터 구조가 작다 (수십 batch는 ms 단위)
3. 활성 batch는 SKU당 ~20개로 통제됨
4. 만약 정말 수천 개라면 SQLAlchemy lazy loading

> **올바른 단일 aggregate는 없다.** 성능 문제가 생기면 경계를 다시 그려도 된다.

---

## 6. Optimistic Concurrency with Version Numbers

전체 batches 테이블에 락을 거는 대신, **Product 행의 version_number** 하나에 락을 좁힌다.

```
Tx1: read product (version=3)        Tx2: read product (version=3)
Tx1: allocate, version → 4            Tx2: allocate, version → 4
Tx1: COMMIT (success)                  Tx2: COMMIT (FAIL — concurrent update)
```

DB 무결성 규칙으로 **version_number 업데이트 충돌 시 한 트랜잭션만 성공**시킨다.

> **Version 숫자가 중요한 게 아니다.** 중요한 건 *Product 행이 매 변경마다 수정된다는 사실*이다. 랜덤 UUID로 대체해도 된다.

### Optimistic vs Pessimistic

| | Optimistic | Pessimistic |
|---|---|---|
| 가정 | 충돌은 드물다 | 충돌이 자주 일어난다 |
| 메커니즘 | version 충돌 감지 → 재시도 | 사전에 락 (`SELECT FOR UPDATE`) |
| 실패 처리 | 클라이언트가 retry | DB가 알아서 대기 |
| 성능 | 충돌이 적을 때 빠름 | 락 경합 시 느림 |

### Version Number를 어디에 둘 것인가

세 옵션:
1. **도메인에 두기**: `Product.allocate()`가 `version_number += 1`
2. **Service layer**: commit 직전에 증가
3. **Infra (UoW/Repository)**: "마법으로" 처리

저자는 1번 선호. 가장 명시적이고 단순.

```python
class Product:
    def __init__(self, sku, batches, version_number=0):
        self.sku = sku
        self.batches = batches
        self.version_number = version_number

    def allocate(self, line: OrderLine) -> str:
        try:
            batch = next(b for b in sorted(self.batches) if b.can_allocate(line))
            batch.allocate(line)
            self.version_number += 1
            return batch.reference
        except StopIteration:
            raise OutOfStock(f'Out of stock for sku {line.sku}')
```

---

## 7. 동시성 동작 테스트

`time.sleep()`으로 느린 트랜잭션 시뮬레이션 후 두 스레드를 동시에 띄운다.

```python
def try_to_allocate(orderid, sku, exceptions):
    line = model.OrderLine(orderid, sku, 10)
    try:
        with unit_of_work.SqlAlchemyUnitOfWork() as uow:
            product = uow.products.get(sku=sku)
            product.allocate(line)
            time.sleep(0.2)
            uow.commit()
    except Exception as e:
        exceptions.append(e)

def test_concurrent_updates_to_version_are_not_allowed(...):
    ...
    thread1 = threading.Thread(target=try_to_allocate_order1)
    thread2 = threading.Thread(target=try_to_allocate_order2)
    thread1.start(); thread2.start()
    thread1.join(); thread2.join()

    assert version == 2  # 한 번만 증가
    assert 'could not serialize access due to concurrent update' in str(exception)
```

### 강제 옵션 1: REPEATABLE READ

```python
DEFAULT_SESSION_FACTORY = sessionmaker(bind=create_engine(
    config.get_postgres_uri(),
    isolation_level="REPEATABLE READ",
))
```

### 강제 옵션 2: SELECT FOR UPDATE (Pessimistic)

```python
def get(self, sku):
    return self.session.query(model.Product) \
        .filter_by(sku=sku) \
        .with_for_update() \
        .first()
```

순서가 `read1, read2, write1, write2(fail)` → `read1, write1, read2, write2(succeed)` 로 바뀐다.

---

## 8. Trade-off

| Pros | Cons |
|------|------|
| Aggregate가 도메인 모델의 public 진입점 명시 | 새 개발자에겐 또 하나의 개념 (entity/VO에 더해 세 번째 타입) |
| 명시적 일관성 경계 → ORM 성능 문제 회피 | "한 번에 하나의 aggregate만 수정" 규칙은 큰 사고 전환 |
| 상태 변경 책임을 단일 객체에 집중 → 추론 쉬움 | aggregate 간 **eventual consistency** 처리 복잡 |

---

## 🎯 Part I 회고

7장으로 Part I이 마무리된다. 우리가 도달한 모습:

```
[Flask API] → [Service Layer] → [UoW] → [Repository] → [Product Aggregate] → [Batch, OrderLine]
                                            ↓
                                      [SQLAlchemy / DB]
```

**달성한 것**:
- **순수 도메인 모델**: 비즈니스 어휘로 쓰인 living documentation
- **인프라 분리**: DB/API 핸들러를 외부 어댑터로
- **DIP + Repository + UoW**: high gear / low gear TDD 가능, 건강한 테스트 피라미드
- **Aggregate**: 일관성 경계 명시, Optimistic Concurrency

**여전히 남은 비용 인정**:
- 모든 패턴은 비용을 동반한다 — 단순 CRUD 앱이라면 Django로 충분. 도메인 복잡도가 패턴 비용을 정당화해야 한다

---

## 다음 장 예고 (Part II 시작)

> *"The big idea is messaging." — Alan Kay*

지금까지는 **단일 aggregate, 단일 트랜잭션**에 갇혀 있었다. Part II는 다음 질문을 다룬다.

> **여러 aggregate 또는 여러 마이크로서비스에 걸치는 프로세스를 어떻게 모델링할 것인가?**

도구:
- **Domain Events** — 일관성 경계를 넘는 워크플로 트리거
- **Message Bus** — 모든 entrypoint에서 use case를 호출하는 통일된 방법
- **CQRS** — 읽기/쓰기 분리로 event-driven 트레이드오프 완화
- **Dependency Injection** — Part II 끝에서 모든 느슨한 끝을 정리

**8장**부터 본격적으로 시작한다 — "재고 부족 시 buying team에게 알림" 같은 평범한 요구사항이 어떻게 Big Ball of Mud로 미끄러지는지를 보고, **Domain Events** 패턴으로 부수효과를 분리하는 방법을 다룬다.
