---
layout: post
title: "[파이썬 아키텍처] CH06 - Unit of Work Pattern"
categories: [Study Note]
tags: [python, architecture, ddd, unit-of-work, sqlalchemy, context-manager, atomicity]
math: false
---

## Introduction

Repository(2장)는 **영속성에 대한 추상화**, Service Layer(4장)는 **유스케이스에 대한 추상화**였다. 6장의 **Unit of Work (UoW)** 는 마지막 한 조각 — **원자적 연산(atomic operation)에 대한 추상화** 이다.

문제 의식: 4장의 service layer는 여전히 SQLAlchemy `session` 객체에 직접 결합돼 있다. UoW를 도입하면 service layer는 더 이상 DB 세션을 알 필요가 없어지고, **모든 영속성 책임이 단일 진입점**으로 모인다.

읽는 법: "you-wow"라고 발음한다.

---

## 1. UoW가 제공하는 세 가지

> 만약 Repository가 *영속 저장소*에 대한 추상화라면, UoW는 *원자적 연산*에 대한 추상화이다.

UoW가 client에게 주는 것:
1. **안정적 DB 스냅샷** — 연산 도중 객체가 변하지 않음
2. **모든 변경의 일괄 영속화** — 중간 실패 시 일관성 깨지지 않음
3. **Persistence에 대한 단순 API + Repository 접근 진입점**

목표 모습:

```python
def allocate(orderid: str, sku: str, qty: int,
             uow: unit_of_work.AbstractUnitOfWork) -> str:
    line = OrderLine(orderid, sku, qty)
    with uow:
        batches = uow.batches.list()
        ...
        batchref = model.allocate(line, batches)
        uow.commit()
```

`with uow:` — **context manager**가 트랜잭션 경계를 시각적으로 드러낸다. 파이썬다운(idiomatic) 표현.

---

## 2. Abstract UoW

```python
class AbstractUnitOfWork(abc.ABC):
    batches: repository.AbstractRepository

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.rollback()

    @abc.abstractmethod
    def commit(self):
        raise NotImplementedError

    @abc.abstractmethod
    def rollback(self):
        raise NotImplementedError
```

핵심:
- `.batches` 속성으로 repository 노출
- `__exit__`에서 **기본 동작은 rollback** — `commit()`이 이미 호출되었다면 rollback은 no-op
- `with` 블록을 빠져나가면 무조건 정리됨

---

## 3. Real Implementation — SQLAlchemy

```python
DEFAULT_SESSION_FACTORY = sessionmaker(
    bind=create_engine(config.get_postgres_uri()))

class SqlAlchemyUnitOfWork(AbstractUnitOfWork):
    def __init__(self, session_factory=DEFAULT_SESSION_FACTORY):
        self.session_factory = session_factory

    def __enter__(self):
        self.session = self.session_factory()
        self.batches = repository.SqlAlchemyRepository(self.session)
        return super().__enter__()

    def __exit__(self, *args):
        super().__exit__(*args)
        self.session.close()

    def commit(self):
        self.session.commit()

    def rollback(self):
        self.session.rollback()
```

`__enter__`마다 새 세션을 시작하고 그 세션을 사용하는 repository를 인스턴스화한다. 통합 테스트에서는 SQLite로, 프로덕션에서는 Postgres로 swap 가능.

---

## 4. Fake UoW — 테스트가 더 깔끔해진다

```python
class FakeUnitOfWork(unit_of_work.AbstractUnitOfWork):
    def __init__(self):
        self.batches = FakeRepository([])
        self.committed = False

    def commit(self):
        self.committed = True

    def rollback(self):
        pass

def test_add_batch():
    uow = FakeUnitOfWork()
    services.add_batch("b1", "CRUNCHY-ARMCHAIR", 100, None, uow)
    assert uow.batches.get("b1") is not None
    assert uow.committed
```

이전에 분리되어 있던 `FakeRepository` + `FakeSession`이 단일 `FakeUnitOfWork`로 통합. 호출자 입장에서 인자가 하나로 줄어든다.

### "Don't Mock What You Don't Own"

저자들이 강조하는 격언. SQLAlchemy `Session`을 직접 mock해도 같은 속도 이점을 얻지만:
- `Session`은 **풍부한 API**(임의 쿼리 가능)를 노출 → 데이터 접근 코드가 코드베이스에 흩뿌려질 위험
- UoW는 우리가 직접 만든 **얇은 인터페이스** → 책임이 명확

> 우리가 소유하지 않은 것을 mock하지 말고, 그 위에 단순한 추상화를 만들어 그것을 mock하라.

---

## 5. Service Layer 단순화

```python
def add_batch(ref: str, sku: str, qty: int, eta: Optional[date],
              uow: unit_of_work.AbstractUnitOfWork):
    with uow:
        uow.batches.add(model.Batch(ref, sku, qty, eta))
        uow.commit()

def allocate(orderid: str, sku: str, qty: int,
             uow: unit_of_work.AbstractUnitOfWork) -> str:
    line = OrderLine(orderid, sku, qty)
    with uow:
        batches = uow.batches.list()
        if not is_valid_sku(line.sku, batches):
            raise InvalidSku(f'Invalid sku {line.sku}')
        batchref = model.allocate(line, batches)
        uow.commit()
    return batchref
```

> service layer의 외부 의존성이 **단 하나**(`AbstractUnitOfWork`)로 줄었다.

---

## 6. Commit/Rollback 동작의 명시적 테스트

```python
def test_rolls_back_uncommitted_work_by_default(session_factory):
    uow = unit_of_work.SqlAlchemyUnitOfWork(session_factory)
    with uow:
        insert_batch(uow.session, 'batch1', 'MEDIUM-PLINTH', 100, None)
    new_session = session_factory()
    rows = list(new_session.execute('SELECT * FROM "batches"'))
    assert rows == []  # commit 안 했으니 비어 있음

def test_rolls_back_on_error(session_factory):
    class MyException(Exception): pass
    uow = unit_of_work.SqlAlchemyUnitOfWork(session_factory)
    with pytest.raises(MyException):
        with uow:
            insert_batch(uow.session, 'batch1', 'LARGE-FORK', 100, None)
            raise MyException()
    new_session = session_factory()
    rows = list(new_session.execute('SELECT * FROM "batches"'))
    assert rows == []  # 예외 시 자동 rollback
```

---

## 7. Explicit vs Implicit Commit

대안으로 "에러 없으면 자동 commit, 있으면 자동 rollback" UoW를 만들 수도 있다.

```python
def __exit__(self, exn_type, exn_value, traceback):
    if exn_type is None:
        self.commit()
    else:
        self.rollback()
```

저자들은 **명시적 commit**을 선호한다. 이유:

> **Safe by default.** 기본 동작은 "아무것도 변경하지 않기". 시스템을 변경하는 경로가 단 하나(완전 성공 + 명시적 commit)뿐이라 추론이 쉽다.

---

## 8. UoW로 다중 연산을 원자 단위로 묶기

### Example 1: Reallocate (해제 후 재할당)

```python
def reallocate(line: OrderLine, uow: AbstractUnitOfWork) -> str:
    with uow:
        batch = uow.batches.get(sku=line.sku)
        if batch is None:
            raise InvalidSku(f'Invalid sku {line.sku}')
        batch.deallocate(line)
        allocate(line)
        uow.commit()
```

`deallocate()`가 실패하면 `allocate()`도 호출되지 않는다. `allocate()`가 실패하면 `deallocate()`도 commit되지 않는다.

### Example 2: 컨테이너 사고로 수량 변경

```python
def change_batch_quantity(batchref: str, new_qty: int,
                          uow: AbstractUnitOfWork):
    with uow:
        batch = uow.batches.get(reference=batchref)
        batch.change_purchased_quantity(new_qty)
        while batch.available_quantity < 0:
            line = batch.deallocate_one()
        uow.commit()
```

여러 line을 deallocate해야 할 수 있지만 — 어떤 단계에서 실패해도 **일부만 commit되는 일은 없다**.

---

## 9. 통합 테스트 정리

이제 DB를 가리키는 통합 테스트가 세 종류이다.

```
tests/
└── integration/
    ├── test_orm.py          ← SQLAlchemy 학습용. 삭제 가능
    ├── test_repository.py
    └── test_uow.py
```

> **5장 교훈의 재확인**: 더 나은 추상화를 만들면 그 위에서 테스트할 수 있고, 하위 디테일을 자유롭게 변경할 수 있다.

---

## 10. Trade-off

| Pros | Cons |
|------|------|
| 원자적 연산에 대한 깔끔한 추상화. context manager로 시각적 grouping | ORM이 이미 적당한 추상화를 가짐 (SQLAlchemy `Session` 자체가 UoW 패턴) |
| 트랜잭션 시작/종료 시점 명시 → safe by default | 롤백, 멀티스레딩, 중첩 트랜잭션 신중히 고려 필요 |
| Repository들의 자연스러운 거주지 | 단순 앱이라면 Django/Flask-SQLAlchemy로 충분 |
| 후속 장(8장 message bus)에서 events와 통합 시 결정적 역할 | |

> SQLAlchemy 공식 문서도 같은 권고를 한다 — *"세션과 트랜잭션의 라이프사이클은 비즈니스 로직 외부에 두라."*

---

## 요약 및 다음 장 연결

**6장 핵심 정리**
- UoW = **원자적 연산에 대한 추상화** + Repository들의 진입점
- Context manager(`with uow:`)로 트랜잭션 경계를 시각화
- **Safe by default**: 명시적 commit 없으면 rollback
- "Don't mock what you don't own" — 외부 라이브러리 대신 우리 추상화를 mock
- Service layer가 마침내 **DB 세션을 모르는 상태**가 됨

**다음 장 예고**
지금까지의 패턴은 모두 **단일 batch 단위**로 동작한다. 7장 **Aggregates and Consistency Boundaries**는 다음 질문에 답한다.
- 동시성(concurrency) 하에서 **불변식(invariant)** 을 어떻게 보장하는가?
- Repository는 **어떤 단위**에 대해 만들어야 하는가?
- DDD의 **Aggregate** 개념과 **Optimistic Concurrency**가 등장한다.

이 챕터는 Part I의 마무리이자 Part II(Event-Driven Architecture)로의 다리이다.
