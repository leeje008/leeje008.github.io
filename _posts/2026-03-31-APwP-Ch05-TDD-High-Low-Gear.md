---
layout: post
title: "[파이썬 아키텍처] CH05 - TDD in High Gear and Low Gear"
categories: [Study Note]
tags: [python, architecture, ddd, tdd, testing, test-pyramid]
math: false
---

## Introduction

4장에서 service layer를 도입하면서, 같은 비즈니스 동작을 **여러 추상화 수준**에서 테스트할 수 있게 되었다. 5장은 이 새로운 자유로 무엇을 할지 — **테스트를 어느 계층에 작성할 것인가** — 에 대한 가이드라인을 다룬다.

핵심 메타포는 자전거 기어 변속이다.

> 새 프로젝트를 시작하거나 까다로운 문제와 씨름할 때는 **저단 기어(low gear, 도메인 모델 직접 테스트)** 로 천천히 정확하게.
> 평범한 기능 추가/버그 수정에서는 **고단 기어(high gear, service layer 테스트)** 로 빠르게 멀리.

---

## 1. 현재 테스트 피라미드

서비스 레이어를 추가한 시점의 테스트 분포:

```
$ grep -c test_ test_*.py
tests/unit/test_allocate.py:4
tests/unit/test_batches.py:8
tests/unit/test_services.py:3
tests/integration/test_orm.py:6
tests/integration/test_repository.py:2
tests/e2e/test_api.py:2
```

→ unit 15개, integration 8개, E2E 2개. 이미 건강한 피라미드 모양.

---

## 2. 도메인 테스트를 service layer로 옮길 것인가?

**같은 시나리오**를 두 수준으로 표현해 비교한다.

### Domain-layer 테스트
```python
def test_prefers_current_stock_batches_to_shipments():
    in_stock_batch = Batch("in-stock-batch", "RETRO-CLOCK", 100, eta=None)
    shipment_batch = Batch("shipment-batch", "RETRO-CLOCK", 100, eta=tomorrow)
    line = OrderLine("oref", "RETRO-CLOCK", 10)
    allocate(line, [in_stock_batch, shipment_batch])
    assert in_stock_batch.available_quantity == 90
```

### Service-layer 테스트
```python
def test_prefers_warehouse_batches_to_shipments():
    in_stock_batch = Batch("in-stock-batch", "RETRO-CLOCK", 100, eta=None)
    shipment_batch = Batch("shipment-batch", "RETRO-CLOCK", 100, eta=tomorrow)
    repo = FakeRepository([in_stock_batch, shipment_batch])
    services.allocate(line, repo, FakeSession())
    assert in_stock_batch.available_quantity == 90
```

> 테스트 한 줄 한 줄은 시스템을 특정 모양으로 고정하는 **접착제(glue)** 와 같다. 저수준 테스트가 많을수록 변경이 어려워진다.

도메인 테스트가 너무 많으면, 도메인 모델을 리팩토링할 때 수십~수백 개 테스트를 함께 수정해야 한다. Service layer를 통한 테스트는 도메인의 **public 동작**만 검증하므로, 내부 구조 변경에 둔감하다.

---

## 3. Coupling vs Design Feedback Trade-off

테스트 수준에 따른 trade-off:

| 수준 | Coupling | Design Feedback | 적합한 시점 |
|------|----------|-----------------|-------------|
| **HTTP/E2E** | 낮음 | 거의 없음 | 큰 변경(스키마 등) 후 안전망 |
| **Service Layer** | 중간 | 중간 | 일상적 기능 추가/수정 |
| **Domain Model** | 높음 | 매우 높음 | 새 프로젝트, 복잡한 도메인 문제 |

XP의 "코드의 목소리를 들어라(listen to the code)" 원칙은 가까이서 코드를 만질 때만 발동된다. HTTP API 테스트는 너무 추상적이라 객체 설계에 대한 피드백을 못 준다. 반대로 도메인 테스트는 살아있는 문서(living documentation) 역할도 겸한다 — 비즈니스 어휘로 쓰여졌으니.

---

## 4. High Gear / Low Gear

- **High Gear (service layer)**: 평범한 기능. coupling 낮고 coverage 높음. `add_stock`, `cancel_order` 같은 정직한 유스케이스
- **Low Gear (domain model)**: 새 프로젝트 출발 시점, 또는 까다로운 도메인 규칙 작업. 피드백이 즉각적

자전거가 정지 상태에서는 저단 기어가 필요하지만, 일단 출발하면 고단 기어로 효율을 올린다. 가파른 언덕(복잡한 새 도메인)을 만나면 다시 저단으로 내려간다.

---

## 5. Service Layer를 도메인으로부터 완전히 분리하기

지금의 service-layer 테스트는 여전히 도메인 객체에 의존한다 — `OrderLine`을 직접 인스턴스화한다.

```python
def allocate(line: OrderLine, repo: AbstractRepository, session) -> str:
```

이를 **primitive 타입**만 받도록 바꾼다.

```python
def allocate(orderid: str, sku: str, qty: int,
             repo: AbstractRepository, session) -> str:
```

테스트도 이에 맞춰:

```python
def test_returns_allocation():
    batch = model.Batch("batch1", "COMPLICATED-LAMP", 100, eta=None)
    repo = FakeRepository([batch])
    result = services.allocate("o1", "COMPLICATED-LAMP", 10, repo, FakeSession())
    assert result == "batch1"
```

하지만 여전히 fixture에서 `Batch` 객체를 만든다. 두 단계 더 나아갈 수 있다.

### Mitigation 1: Fixture 함수에 도메인 의존성 격리

```python
class FakeRepository(set):
    @staticmethod
    def for_batch(ref, sku, qty, eta=None):
        return FakeRepository([model.Batch(ref, sku, qty, eta)])

def test_returns_allocation():
    repo = FakeRepository.for_batch("batch1", "COMPLICATED-LAMP", 100, eta=None)
    result = services.allocate("o1", "COMPLICATED-LAMP", 10, repo, FakeSession())
    assert result == "batch1"
```

도메인 의존성이 한 곳에 모임.

### Mitigation 2: 누락된 service를 추가

테스트에서 도메인 객체를 만들어야 한다는 것은 **service layer가 미완성**이라는 신호일 수 있다. `add_batch` service를 추가하면 테스트가 service만으로 닫힌다.

```python
def add_batch(ref: str, sku: str, qty: int, eta: Optional[date],
              repo: AbstractRepository, session):
    repo.add(model.Batch(ref, sku, qty, eta))
    session.commit()
```

```python
def test_allocate_returns_allocation():
    repo, session = FakeRepository([]), FakeSession()
    services.add_batch("batch1", "COMPLICATED-LAMP", 100, None, repo, session)
    result = services.allocate("o1", "COMPLICATED-LAMP", 10, repo, session)
    assert result == "batch1"
```

> **새 service를 단지 테스트 의존성을 빼기 위해 만들어야 하는가?** 보통은 No. 다만 `add_batch`는 어차피 언젠가 필요하므로 정당화된다.

이제 service-layer 테스트는 service에만 의존 — 도메인 모델을 자유롭게 리팩토링할 수 있다.

---

## 6. E2E 테스트로의 파급 효과

`add_stock` raw SQL fixture도 새 `add_batch` API endpoint로 대체할 수 있다.

```python
@app.route("/add_batch", methods=['POST'])
def add_batch():
    session = get_session()
    repo = repository.SqlAlchemyRepository(session)
    eta = request.json['eta']
    if eta is not None:
        eta = datetime.fromisoformat(eta).date()
    services.add_batch(
        request.json['ref'], request.json['sku'],
        request.json['qty'], eta, repo, session
    )
    return 'OK', 201
```

E2E 테스트가 SQL이 아닌 API 호출로 fixture를 만든다 → 데이터베이스 의존성 제거.

```python
def post_to_add_batch(ref, sku, qty, eta):
    requests.post(f'{config.get_api_url()}/add_batch',
                  json={'ref': ref, 'sku': sku, 'qty': qty, 'eta': eta})
```

---

## 7. 테스트 종류별 Rules of Thumb

저자들의 권고:

| 테스트 종류 | 권장 개수 | 목적 |
|------------|----------|------|
| **E2E** | 기능당 **1개 (happy path)** | 모든 부품이 올바르게 연결됨을 증명 |
| **Service-layer (unit)** | **대부분** | 비즈니스 로직과 edge case의 핵심 커버리지 |
| **Domain model (unit)** | **소수, 핵심만** | 가장 빠른 피드백. 나중에 service로 흡수되면 삭제 가능 |
| **Unhappy path** | E2E 1개 + service unit 다수 | 에러 처리도 기능이다 |

추가 헬퍼 원칙:
- **Service layer는 primitive로** 표현 → 도메인 변화에 둔감
- 모든 setup이 **service를 통해** 가능하도록 service를 충분히 만들어 둔다 — repository나 DB를 직접 hack하지 말 것

---

## 요약 및 다음 장 연결

**5장 핵심 정리**
- 테스트는 **접착제**다 — 너무 많으면 변경이 어려워진다
- **High gear (service)**: 일상 작업의 기본. **Low gear (domain)**: 까다로운 신영역
- Service layer는 **primitive**로 표현해 도메인 결합을 끊는다
- 빠진 service(`add_batch`)를 채우면 테스트가 service만으로 자급한다
- E2E는 happy path 1개로 충분, unhappy 1개 추가

**다음 장 예고**
Service layer는 여전히 **`session` 객체에 직접 결합**되어 있다. 6장 **Unit of Work 패턴**이 이 마지막 결합을 끊는다. UoW는 "원자적 연산"의 추상화이며, repository와 service layer를 매끄럽게 묶는 **마지막 퍼즐 조각**이다.
