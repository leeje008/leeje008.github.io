---
layout: post
title: "[파이썬 아키텍처] CH00 - Prerequisites (책을 읽기 전에)"
categories: [Study Note]
tags: [python, architecture, prerequisites, type-hints, dataclass, abc, context-manager, sqlalchemy, flask, pytest]
math: false
---

## Introduction

*Architecture Patterns with Python* (Harry Percival & Bob Gregory, O'Reilly — 일명 **Cosmic Python**)은 **DDD · TDD · Event-Driven Architecture**를 파이썬으로 풀어낸다. 저자들도 preface에서 명시하듯, 독자가 DDD나 아키텍처 패턴을 몰라도 좋다. 다만 **"복잡한 파이썬 애플리케이션을 다뤄본 경험"** 은 전제한다 — 책에 흩어진 파이썬 중·고급 관용구가 눈에 익어야 패턴 본질에 집중할 수 있다.

이 글(CH00)은 본편(Ch1~Ch7)에 들어가기 전 **책이 실제로 사용하는 파이썬 기능만** 골라 정리한 선행 학습 노트다. PDF 본문을 스캔해 import 문 / 데코레이터 / dunder / 타입 힌트 / with 문 / class 정의를 뽑아, 빈도 높은 것부터 배치했다. 각 항목에는 **"책 어느 챕터에서 쓰이는지"** 와 **"2026년 현재 권장 방식과의 차이"** 도 함께 표시한다.

---

## 0. 책이 실제로 쓰는 라이브러리 한눈에

PDF 본문 import 스캔 결과:

| 라이브러리 | 용도 | 첫 등장 |
|---|---|---|
| `dataclasses` | Value Object 표현 (`@dataclass(frozen=True)`) | Ch1 |
| `typing` | `List`, `Optional`, `NewType`, `NamedTuple` 등 타입 힌트 | Ch1 |
| `abc` | 추상 인터페이스 (`ABC`, `@abstractmethod`) | Ch2 |
| `sqlalchemy` | ORM — `create_engine`, `sessionmaker`, `mapper` | Ch2 |
| `flask` | 웹 엔트리포인트 (`Flask`, `jsonify`, `request`) | Ch4 |
| `pytest` | 테스트 (`@pytest.fixture`, `pytest.raises`, `mark`) | 전체 |
| `contextlib` | `@contextmanager` 대체안 | Ch6 |
| `hashlib`, `os`, `shutil`, `pathlib` | Ch3 토이 예제(디렉터리 동기화) | Ch3 |
| `collections` / `dataclasses` | `namedtuple` vs `@dataclass` 비교 | Ch1 |
| `tenacity` | 재시도 패턴 (`Retrying`) | Ch8+ |

> Docker / Redis / Postgres 도 후반부에 등장하지만, **Part I(Ch1~Ch7)** 만 다루는 이 노트에서는 **앞쪽 6개**가 핵심이다.

---

## 1. Type Hints

### 1-1. 왜 쓰는가
파이썬은 동적 타입 언어지만, 본 책은 거의 모든 함수 시그니처에 타입 힌트를 단다. 타입은 **런타임에 영향을 주지 않으며**, 다음 세 가지 이득만을 위해 존재한다.

1. IDE 자동완성 / 오류 표시
2. mypy 같은 정적 타입 체커가 버그를 사전 차단
3. **문서 효과** — 시그니처만 보고 의도 파악 가능

### 1-2. 책이 쓰는 형태

```python
from typing import List, Optional
from datetime import date

def allocate(orderid: str, sku: str, qty: int,
             eta: Optional[date]) -> str:
    ...

def __init__(self, sku: str, batches: List[Batch]):
    ...
```

### 1-3. 구 문법 vs 신 문법
책은 2020년 기준(Python 3.7~3.8)이라 `typing.List`, `typing.Optional`을 쓴다. **2026년 권장 방식**:

| 버전 | 구 | 신 |
|---|---|---|
| 3.9+ (PEP 585) | `List[Batch]` | `list[Batch]` |
| 3.10+ (PEP 604) | `Optional[date]` / `Union[int, str]` | `date \| None` / `int \| str` |

```python
# 책 스타일 (3.8)
def f(xs: List[int]) -> Optional[str]: ...

# 2026년 스타일 (3.10+)
def f(xs: list[int]) -> str | None: ...
```

둘 다 동작하지만, 새 코드라면 후자를 권장한다.

### 1-4. `TYPE_CHECKING`
순환 import 피하려고 "타입만 필요한 import"를 런타임에서 제외할 때 쓴다.

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .model import OrderLine  # 런타임 로드 안 됨
```

> **책에서**: Ch1 전반(함수 시그니처), Ch2 AbstractRepository, Ch4 service layer 시그니처에서 지속 사용.

---

## 2. Dataclass — Value Object의 관용구

### 2-1. 기본
`@dataclass`는 `__init__`, `__repr__`, `__eq__`를 자동 생성한다.

```python
from dataclasses import dataclass

@dataclass
class Batch:
    reference: str
    sku: str
    qty: int
```

### 2-2. `frozen=True` — 책의 핵심
책은 **Value Object**(정체성이 아닌 값으로 동등한 객체 — 예: `Money`, `OrderLine`)에 `frozen=True`를 쓴다.

```python
@dataclass(frozen=True)
class OrderLine:
    orderid: str
    sku: str
    qty: int
```

`frozen=True`가 주는 것:
- 필드 할당 시 `FrozenInstanceError` → **불변 보장**
- `__hash__` 자동 생성 → `set`, `dict key`로 사용 가능
- 동등성은 **모든 필드값**으로 판정

**왜 hash와 frozen이 함께 가는가?** Python 규칙상 `eq=True, frozen=False`면 `__hash__`가 `None`이 된다 — 가변 객체를 `set`에 넣으면 hash가 바뀌어 자료구조가 깨질 수 있기 때문. `frozen=True`여야 안전하게 hashable.

### 2-3. Entity vs Value Object

| | Value Object | Entity |
|---|---|---|
| 정체성 | 값으로 동등 (`Money(100, "USD") == Money(100, "USD")`) | ID로 동등 |
| 가변성 | 불변 | 가변 |
| dataclass | `@dataclass(frozen=True)` | `@dataclass(eq=False)` + 수동 `__eq__` |
| 책 예시 | `OrderLine` | `Batch`, `Product` |

> **책에서**: Ch1에서 두 종류를 대비해서 소개. `OrderLine`만 frozen, `Batch`는 가변.

---

## 3. Magic Method (Dunder)

책에 실제로 등장하는 dunder 목록 (빈도순):

| dunder | 역할 | 책의 용도 |
|---|---|---|
| `__init__` | 생성자 | 거의 모든 클래스 |
| `__eq__` | 동등성 | Entity는 ID로 비교 |
| `__hash__` | 해시 | `set`/`dict` 저장 시 |
| `__gt__` | `>` 연산자 | `sorted(batches)`로 ETA 빠른 순 정렬 |
| `__repr__` | 디버깅 문자열 | 테스트 실패 시 가독성 |
| `__enter__` / `__exit__` | with 문 | UoW (Ch6) |
| `__call__` | 인스턴스 호출 | 메시지 핸들러 (Part II) |

### Entity의 `__eq__` + `__hash__`

```python
class Batch:
    def __eq__(self, other):
        if not isinstance(other, Batch):
            return False
        return other.reference == self.reference

    def __hash__(self):
        return hash(self.reference)
```

**핵심 규칙**: `__eq__`를 정의하면 `__hash__`도 같이 정의해야 한다. 그렇지 않으면 set에 넣었을 때 "동등하지만 다른 버킷" 같은 버그가 발생.

### 정렬 — `__gt__` 하나만 있어도 된다

```python
class Batch:
    def __gt__(self, other):
        if self.eta is None:
            return False
        if other.eta is None:
            return True
        return self.eta > other.eta

sorted([b1, b2, b3])  # __gt__만으로 동작
```

`functools.total_ordering` 데코레이터를 쓰면 `__eq__` + `__lt__`만으로 나머지 비교 연산자가 자동 생성된다.

> **책에서**: Ch1 Batch 클래스에 이 패턴이 그대로 등장. "재고가 있는 batch를 먼저, ETA가 빠른 batch를 먼저" 할당하는 비즈니스 규칙을 `sorted()` 한 줄로 구현.

---

## 4. Context Manager & `with` 문

### 4-1. 프로토콜
`with` 블록은 **리소스 획득/해제를 보장**한다. 객체가 `__enter__`/`__exit__`를 가지면 context manager가 된다.

```python
class SqlAlchemyUnitOfWork:
    def __enter__(self):
        self.session = self.session_factory()
        return self  # as 절에 바인딩

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            self.rollback()
        self.session.close()

with SqlAlchemyUnitOfWork() as uow:
    uow.products.get(sku="...")
    uow.commit()
# 여기서 __exit__ 자동 호출 (예외 유무와 무관)
```

`__exit__`의 세 인자:
- 정상 종료 시: 세 인자 모두 `None`
- 예외 발생 시: `(type, value, traceback)` 전달
- `__exit__`가 `True`를 반환하면 **예외가 억제**됨

### 4-2. `@contextmanager` — 더 간결한 방법

`contextlib.contextmanager` 데코레이터로 **제너레이터 기반** context manager를 만들 수 있다. yield 앞이 `__enter__`, 뒤가 `__exit__` 역할.

```python
from contextlib import contextmanager

@contextmanager
def tmp_session():
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

with tmp_session() as s:
    s.query(...)
```

책은 **클래스 기반**을 선호 — UoW가 state(`.products`, `.events`)를 필드로 가져야 해서 클래스가 자연스럽다. 저자도 "간단하면 `@contextmanager` 써도 된다"고 주석으로 언급.

> **책에서**: Ch6 Unit of Work 패턴의 심장. `with uow:` 한 줄로 트랜잭션 경계가 **시각적으로** 드러난다. 이게 왜 "Pythonic"한지가 Ch6의 주제.

---

## 5. Abstract Base Class (ABC)

### 5-1. 기본
파이썬의 **명시적 인터페이스** 선언 방법.

```python
import abc

class AbstractRepository(abc.ABC):
    @abc.abstractmethod
    def add(self, batch): ...

    @abc.abstractmethod
    def get(self, reference): ...
```

- `abc.ABC` 상속 + `@abc.abstractmethod` 데코레이터
- abstract 메서드를 구현 안 하면 **인스턴스화 시 `TypeError`**
- 단순 `pass` 또는 `...`(Ellipsis)로 본문 작성

### 5-2. ABC vs Protocol — 2026년의 선택지

Python 3.8+ (PEP 544)부터 `typing.Protocol`로 **구조적 서브타이핑**이 가능해졌다. 상속 없이 "메서드 시그니처만 맞으면 호환".

```python
from typing import Protocol

class Repository(Protocol):
    def add(self, batch) -> None: ...
    def get(self, reference) -> Batch: ...

class SqlAlchemyRepository:  # Protocol을 상속하지 않아도 됨
    def add(self, batch): ...
    def get(self, reference): ...
```

| | `abc.ABC` | `typing.Protocol` |
|---|---|---|
| 서브타이핑 | 명목적 (상속 필요) | 구조적 (덕 타이핑) |
| 미구현 시 | 인스턴스화 불가 | mypy 에러만, 런타임은 통과 |
| isinstance | 바로 가능 | `@runtime_checkable` 필요 |
| 외부 클래스 | 수정해야 상속 가능 | 수정 없이 호환 판정 |

**책의 선택**: ABC. 이유는 **DIP(의존성 역전)를 시각적으로 드러내기 위해** — `class SqlAlchemyRepository(AbstractRepository)`라고 쓰면 "이 클래스가 어떤 계약을 지키는지" 한눈에 보인다. 2026년에 새로 쓴다면 Protocol을 선호할 여지도 있다.

> **책에서**: Ch2 `AbstractRepository`, Ch6 `AbstractUnitOfWork`, Ch7 `AbstractProductRepository`, Part II `AbstractNotifications`. 모두 같은 패턴.

---

## 6. Iterator · Generator · `next()` · `StopIteration`

Ch7 `Product.allocate()`의 핵심 관용구:

```python
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

분해:
- `(b for b in sorted(self.batches) if b.can_allocate(line))` — **제너레이터 표현식** (lazy)
- `next(gen)` — 첫 항목만 꺼냄
- 매치 없으면 `StopIteration` 발생 → 도메인 예외로 변환

**장점**: 전체 리스트를 순회하지 않고 **첫 매치에서 멈춤**. C# `FirstOrDefault`의 파이썬 버전.

`next(gen, default)` 형태로 기본값 지정도 가능 — 이 경우 `StopIteration`이 발생하지 않는다.

---

## 7. 도메인 예외

```python
class OutOfStock(Exception):
    pass

class InvalidSku(Exception):
    pass
```

- `Exception` 상속 + `pass`로 충분
- 메시지는 `raise OutOfStock(f'Out of stock for sku {sku}')`로 전달
- `pytest.raises(OutOfStock, match='SMALL-FORK')` 로 메시지까지 검증 가능

> **책에서**: Ch1(`OutOfStock`), Ch4(`InvalidSku`) — 비즈니스 규칙 위반을 **도메인 어휘**로 표현.

---

## 8. SQLAlchemy 핵심

### 8-1. Engine · Session · sessionmaker

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("postgresql://...")
Session = sessionmaker(bind=engine)
session = Session()

session.add(obj)
session.commit()
session.close()
```

- **Engine**: DB 연결 풀 (앱당 1개)
- **Session**: 한 트랜잭션의 작업 단위 — 사실 SQLAlchemy 자체가 UoW 패턴. 책이 Ch6에서 UoW 래퍼를 또 만드는 이유를 Ch6가 해명한다
- **sessionmaker**: Session 팩토리

### 8-2. Declarative vs Classical(Imperative) Mapping

**Declarative** (일반적이며, 책도 처음엔 이것으로 시작):
```python
from sqlalchemy.orm import declarative_base
Base = declarative_base()

class Batch(Base):
    __tablename__ = "batches"
    id = Column(Integer, primary_key=True)
    reference = Column(String(255))
```

→ 도메인 클래스가 `Base`를 상속 = **ORM이 모델을 오염시킴**.

**Classical / Imperative** (책이 Ch2에서 선택하는 방식):
```python
from sqlalchemy import Table, Column, Integer, String
from sqlalchemy.orm import registry

mapper_registry = registry()

batches = Table(
    "batches", mapper_registry.metadata,
    Column("id", Integer, primary_key=True),
    Column("reference", String(255)),
)

def start_mappers():
    mapper_registry.map_imperatively(model.Batch, batches)
```

→ 도메인 `Batch`는 **순수 Python class**. ORM 매핑은 분리된 파일(`orm.py`)에서 선언. **DIP 달성** — 도메인은 ORM을 모른다.

### 8-3. 2026년 변화 — SQLAlchemy 2.x

책(2020)은 SQLAlchemy 1.3 기반. 2023년 출시된 2.x에서:

| 책 (1.x) | 2.x (2026 권장) |
|---|---|
| `declarative_base()` 함수 | `DeclarativeBase` 클래스 상속 |
| `mapper()` 함수 | `registry.map_imperatively()` |
| `Column(...)` 직접 | `Mapped[int]`, `mapped_column()` 권장 (PEP 484 통합) |

책의 Classical Mapping 개념은 그대로 유효 — API만 `registry.map_imperatively()`로 바뀌었다. 예제 코드를 2.x로 돌리려면 이 부분만 수정하면 된다.

> **책에서**: Ch2에서 Declarative → Classical로 리팩토링하는 과정이 **DIP의 구체적 시연**이다.

---

## 9. Flask 핵심

책이 쓰는 Flask API는 `Flask`, `jsonify`, `request` 세 가지뿐.

```python
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/allocate", methods=["POST"])
def allocate_endpoint():
    data = request.get_json()  # 2026 권장 (request.json도 여전히 동작)
    try:
        batchref = services.allocate(
            data["orderid"], data["sku"], data["qty"]
        )
    except (model.OutOfStock, services.InvalidSku) as e:
        return jsonify({"message": str(e)}), 400
    return jsonify({"batchref": batchref}), 201
```

### 9-1. 2026년 권장 변화

| 책 | 2026 권장 |
|---|---|
| `request.json` | `request.get_json()` (옵션 제공: `force=True`, `silent=True`) |
| `return jsonify({...}), 201` | dict 직접 반환도 자동 JSON화 (Flask 1.1+) |
| 단일 `app` | **Blueprint**로 기능별 분리 권장 (규모 커지면) |

### 9-2. 왜 Flask인가
책의 Flask 핸들러는 **"얇게"** 유지된다 — 파싱·응답 코드만 남기고 비즈니스 로직은 service layer로 위임. 이게 Ch4의 주제. Django/FastAPI 대신 Flask를 쓰는 것도 **프레임워크 의존성을 최소화**해 패턴을 드러내기 위함.

> **책에서**: Ch4 첫 등장. 이후 챕터에선 거의 건드리지 않음 — entrypoint는 "얇아야" 한다는 게 핵심.

---

## 10. pytest 핵심

### 10-1. 기본 테스트
파일 `test_*.py`, 함수 `test_*`만 지키면 자동 발견.

```python
def test_allocating_reduces_available_quantity():
    batch = Batch("b1", "SKU", 20, eta=None)
    line = OrderLine("o1", "SKU", 2)
    batch.allocate(line)
    assert batch.available_quantity == 18
```

### 10-2. Fixture

```python
import pytest

@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    yield engine  # yield 뒤는 teardown
    engine.dispose()

def test_something(in_memory_db):  # fixture가 인자로 주입
    ...
```

**conftest.py**: `tests/` 폴더에 두면 자동 공유됨. 책은 이걸 적극 활용.

### 10-3. 예외 검증

```python
def test_raises_out_of_stock():
    with pytest.raises(OutOfStock, match="SMALL-FORK"):
        allocate(...)
```

`match` 인자는 **정규식**으로 예외 메시지까지 검증.

### 10-4. 마커

```python
@pytest.mark.usefixtures("restart_api")
def test_api(...):
    ...

@pytest.mark.parametrize("qty,expected", [(10, 90), (20, 80)])
def test_allocate(qty, expected):
    ...
```

`@pytest.mark.usefixtures('restart_api')` 는 책에서 E2E 테스트가 자주 씀 — Flask API를 띄우고 끄는 fixture를 적용.

### 10-5. 책의 테스트 디렉터리 구조

```
tests/
├── conftest.py           # 공용 fixture
├── unit/                 # 빠른 도메인·서비스 테스트 (DB 미사용)
│   ├── test_batches.py
│   └── test_services.py
├── integration/          # DB/ORM/Repository (SQLite or Postgres)
│   ├── test_orm.py
│   ├── test_repository.py
│   └── test_uow.py
└── e2e/                  # Flask 띄운 HTTP 테스트 (happy + unhappy 각 1개)
    └── test_api.py
```

이 구조 자체가 **테스트 피라미드**를 구현한다 — Ch5의 주제.

---

## 11. Test Double — Mock vs Fake

책의 강한 입장: **"Don't mock what you don't own"**.

| 종류 | 구현 | 검증 대상 | 책 | 학파 |
|---|---|---|---|---|
| **Mock** | `unittest.mock.MagicMock`, 호출 기록 | "어떻게 호출됐는가" | **거의 안 씀** | London-school |
| **Fake** | 실제 동작하는 단순 구현체 | "결과 상태가 맞는가" | **주력** | Classicist |
| **Stub** | 정해진 값만 리턴 | 간접 입력 제공 | 드물게 | - |

```python
# 책의 FakeRepository (Ch2)
class FakeRepository(set):
    def add(self, batch):
        super().add(batch)
    def get(self, reference):
        return next(b for b in self if b.reference == reference)
    def list(self):
        return list(self)
```

`set`을 상속해서 공짜로 저장·순회를 얻는다. **프로덕션 `SqlAlchemyRepository`와 같은 인터페이스를 구현**하므로, 테스트만 쓰는 코드가 아니다 — "같은 추상화를 구현한 두 가지 구현"이다.

> **참고**: Martin Fowler의 *Mocks Aren't Stubs* 가 이 구분의 고전. 책은 **classicist**(상태 기반 테스트) 쪽.

---

## 12. 동시성 기초 (Ch7 배경)

### 12-1. GIL
CPython의 **Global Interpreter Lock** — 한 번에 한 스레드만 Python 바이트코드 실행.
- **CPU-bound**: 멀티스레드로 가속 안 됨
- **I/O-bound** (네트워크, DB, 파일): 스레드가 blocking 구간에서 GIL을 놓으므로 멀티스레드가 유효

DB 트랜잭션은 I/O-bound → 스레드로 concurrency 시뮬레이션 가능. 책은 이걸 이용해 race condition을 재현한다.

### 12-2. threading 기본

```python
import threading

def worker(arg):
    ...

t1 = threading.Thread(target=worker, args=("a",))
t2 = threading.Thread(target=worker, args=("b",))
t1.start(); t2.start()
t1.join(); t2.join()  # 둘 다 끝날 때까지 대기
```

### 12-3. Race Condition & Lost Update

두 스레드가 같은 데이터를 읽고 각자 수정한 뒤 쓰면, **한 쪽 변경이 사라진다**.

```
T1: read(version=3)      T2: read(version=3)
T1: write(version=4)     T2: write(version=4)  ← T1 변경 덮어씀
```

DB 수준에서 막는 방법 두 가지 (Ch7 주제):

| | Optimistic | Pessimistic |
|---|---|---|
| 방식 | version 번호로 충돌 감지 → 한쪽 실패 | `SELECT FOR UPDATE`로 사전 락 |
| 성능 | 충돌이 드물 때 빠름 | 락 경합 시 느림 |
| 책 Ch7 | **기본 선택** | 대안 소개 |

### 12-4. PostgreSQL Transaction Isolation Level

| Level | 막는 문제 |
|---|---|
| READ COMMITTED (기본) | dirty read |
| REPEATABLE READ | non-repeatable read, phantom read |
| SERIALIZABLE | 모든 이상 현상 |

책은 `REPEATABLE READ`로 올려 **concurrent update를 감지**한다.

---

## 13. 함수형 맛보기 (Ch3 배경)

Ch3의 **Functional Core, Imperative Shell (FCIS)** 에 필요한 개념:

- **Pure Function**: 같은 입력 → 같은 출력, 부작용 없음
- **Immutability**: 입력을 수정하지 않고 새 객체 반환
- **Higher-order Function**: 함수를 인자/반환값으로 — `sorted(key=...)`, `map`, `filter`

```python
# impure - I/O와 섞임
def sync(src, dst):
    for f in os.listdir(src):  # 파일시스템 의존
        shutil.copy(...)

# pure core - dict in, list out
def determine_actions(src_hashes: dict, dst_hashes: dict) -> list:
    actions = []
    for sha, name in src_hashes.items():
        if sha not in dst_hashes:
            actions.append(("copy", name))
    return actions
```

순수 함수는 **테스트가 자명**하다 — fixture, mock 없이 `assert f(x) == y`.

> **책에서**: Ch3 디렉터리 동기화 토이 예제. 실제 프로젝트(Part I~II)에서도 "도메인 서비스는 순수하게, I/O는 service layer에"라는 원칙이 이어진다.

---

## 14. 디렉터리 구조 & 패키지

책의 최종 구조 (Ch4 이후):

```
src/allocation/
├── domain/              # 순수 도메인 모델 (Ch1)
│   ├── __init__.py
│   └── model.py
├── service_layer/       # 유스케이스 (Ch4)
│   ├── __init__.py
│   └── services.py
├── adapters/            # driven adapters: DB, Redis (Ch2)
│   ├── __init__.py
│   ├── orm.py
│   └── repository.py
└── entrypoints/         # driving adapters: Flask, CLI (Ch4)
    ├── __init__.py
    └── flask_app.py
```

### 14-1. `__init__.py` & 패키지
- `__init__.py`가 있으면 해당 폴더는 **패키지**
- 상대 import: `from ..domain import model` (한 단계 위)
- `pip install -e .` (`setup.py` 또는 `pyproject.toml` 기반) → `from allocation.domain import model` 절대 import 가능

### 14-2. Ports & Adapters 어휘
- **adapters** = secondary / driven (DB, 메시지 브로커 등 — 앱이 호출하는 쪽)
- **entrypoints** = primary / driving (Flask, CLI — 앱을 호출하는 쪽)
- **port**(추상 인터페이스)는 어댑터 파일 안에 둔다

이 어휘는 **Hexagonal Architecture**에서 왔다 — 책이 "Ports & Adapters"라는 별명을 선호하는 이유는 "헥사곤"보다 의미가 명확해서.

---

## 15. 읽는 순서 권장

1. **Ch0** ← 지금 이 글
2. **Ch1 Domain Modeling** — dataclass · dunder · 도메인 예외
3. **Ch2 Repository** — DIP · ABC · Classical Mapping
4. **Ch3 Coupling & Abstractions** — FCIS · DI · mock 대신 fake
5. **Ch4 Flask & Service Layer** — 외부 세계 연결
6. **Ch5 TDD High/Low Gear** — 테스트 전략
7. **Ch6 Unit of Work** — context manager로 트랜잭션 추상화
8. **Ch7 Aggregates** — 동시성, 일관성 경계

---

## 16. 공식 자료 & 참고

### 원서 / 번역서 / 공식 사이트
- 원서: [Architecture Patterns with Python (O'Reilly)](https://www.oreilly.com/library/view/architecture-patterns-with/9781492052197/)
- 무료 온라인판: [cosmicpython.com](https://www.cosmicpython.com/)
- 저자 예제 코드: [github.com/cosmicpython/code](https://github.com/cosmicpython/code)
- 한국어판: **『파이썬으로 살펴보는 아키텍처 패턴』**(오현석 역, 한빛미디어)

### Python 공식 문서
- [PEP 484 — Type Hints](https://peps.python.org/pep-0484/)
- [PEP 544 — Protocols (Structural Subtyping)](https://peps.python.org/pep-0544/)
- [PEP 585 — Type Hinting Generics In Standard Collections](https://peps.python.org/pep-0585/)
- [PEP 604 — Union Types with `|`](https://peps.python.org/pep-0604/)
- [typing — Support for type hints](https://docs.python.org/3/library/typing.html)
- [dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [abc — Abstract Base Classes](https://docs.python.org/3/library/abc.html)
- [contextlib](https://docs.python.org/3/library/contextlib.html)

### 외부 라이브러리
- [SQLAlchemy 2.0 Declarative Mapping Styles](https://docs.sqlalchemy.org/en/20/orm/declarative_styles.html)
- [SQLAlchemy 2.0 — `registry.map_imperatively()`](https://docs.sqlalchemy.org/en/20/orm/mapping_api.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [pytest Documentation](https://docs.pytest.org/)

### 이론적 배경
- Martin Fowler — [Mocks Aren't Stubs](https://martinfowler.com/articles/mocksArentStubs.html)
- Eric Evans — *Domain-Driven Design* (2003, 일명 Blue Book)
- Alistair Cockburn — [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)

---

## 다음 장

**Ch1 — Domain Modeling**으로 본편 시작. 이 Ch0의 1~5절(type hints / dataclass / dunder / ABC는 Ch2부터)이 그대로 쓰인다. 재고 할당 문제를 **Batch**(Entity) + **OrderLine**(Value Object) + **allocate()**(Domain Service)로 모델링하며, 유비쿼터스 언어의 중요성을 경험한다.
