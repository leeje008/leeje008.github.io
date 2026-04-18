---
layout: post
title: "[파이썬 아키텍처] CH02 - Repository Pattern"
categories: [Study Note]
tags: [python, architecture, ddd, repository-pattern, sqlalchemy, dependency-inversion]
math: false
---

## Introduction

1장에서 만든 도메인 모델은 **순수한 파이썬 객체**였다. 외부 의존성이 없으니 테스트하기 쉽다. 하지만 실제 시스템은 어딘가에 데이터를 저장해야 한다. 데이터베이스가 등장하는 순간 도메인 모델이 인프라에 오염되는 위험이 시작된다.

본 장의 목표는 다음 한 문장으로 요약된다.

> **도메인 모델은 영속성(persistence)에 대해 무지(ignorant)해야 한다.** 그리고 그 격리를 달성하는 도구가 **Repository 패턴**이다.

---

## 1. 의존성 역전 원칙(DIP)을 데이터 접근에 적용

전형적인 layered architecture는 UI → 비즈니스 로직 → DB 순서로 단방향 의존성이 흐른다. 하지만 이 구조는 비즈니스 로직이 DB의 형식에 끌려다니게 만든다.

대신 **양파 아키텍처(onion architecture)** 또는 **포트와 어댑터(ports and adapters)** 가 제안하는 방향은 다음과 같다.

> 도메인 모델을 안쪽에 두고, 모든 의존성이 안쪽으로 흐르게 한다.

| 용어 | 정의 | 본 장의 예시 |
|------|------|--------------|
| **Port** | 애플리케이션과 외부의 경계 인터페이스 | `AbstractRepository` |
| **Adapter** | 그 인터페이스의 구체적 구현 | `SqlAlchemyRepository`, `FakeRepository` |

저자들은 "양파 아키텍처, 헥사고날 아키텍처, 클린 아키텍처는 본질적으로 같다 — 모두 DIP의 구체적 적용"이라고 못박는다.

---

## 2. ORM "정상 사용법"의 문제

전형적인 SQLAlchemy declarative 스타일은 다음과 같다.

```python
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class OrderLine(Base):
    id = Column(Integer, primary_key=True)
    sku = Column(String(250))
    qty = Column(Integer)
```

문제는 명백하다. **도메인 모델이 ORM에 의존**하게 된다. SQLAlchemy를 모르고는 `OrderLine`을 이해할 수 없다. Django ORM도 동일한 문제를 가진다(오히려 더 심각).

### 의존성 뒤집기: ORM이 모델에 의존하게 만들기

SQLAlchemy의 **Classical Mapping**을 사용하면 스키마와 도메인 모델을 분리할 수 있다.

```python
from sqlalchemy.orm import mapper
from sqlalchemy import Table, Column, Integer, String, MetaData
import model

metadata = MetaData()

order_lines = Table(
    'order_lines', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('sku', String(255)),
    Column('qty', Integer, nullable=False),
    Column('orderid', String(255)),
)

def start_mappers():
    lines_mapper = mapper(model.OrderLine, order_lines)
```

핵심:
- `model.py`는 SQLAlchemy를 import하지 않는다 — 순수 도메인
- `orm.py`가 `model`을 import하고, `start_mappers()`를 호출해야 비로소 객체-테이블 매핑이 활성화됨
- `start_mappers()`를 부르지 않으면 도메인 모델은 DB의 존재 자체를 모른다 → **순수한 단위 테스트 가능**

---

## 3. Repository 패턴 — 영속성 위의 단순한 추상화

Repository는 **영속 저장소를 마치 메모리 컬렉션처럼** 보이게 하는 추상화이다. 가장 단순한 형태는 두 메서드만 가진다.

```python
import abc

class AbstractRepository(abc.ABC):
    @abc.abstractmethod
    def add(self, batch: model.Batch):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, reference) -> model.Batch:
        raise NotImplementedError
```

> **TIP**: `list`나 `delete`, `update`는 왜 없을까? 이상적으로는 도메인 객체를 한 번에 하나씩 수정하고, `delete`는 보통 soft-delete(예: `batch.cancel()`)로 처리하며, `update`는 6장의 **Unit of Work** 패턴이 담당한다.

### SQLAlchemy 기반 구현

```python
class SqlAlchemyRepository(AbstractRepository):
    def __init__(self, session):
        self.session = session

    def add(self, batch):
        self.session.add(batch)

    def get(self, reference):
        return self.session.query(model.Batch) \
            .filter_by(reference=reference).one()

    def list(self):
        return self.session.query(model.Batch).all()
```

설계 결정 한 가지: **`commit()`은 repository 안에 두지 않는다**. 호출자(서비스 레이어)의 책임이다. 이는 6장 UoW에서 자연스러워진다.

---

## 4. Fake Repository — 테스트가 거저 쉬워진다

Repository 패턴의 가장 큰 실용적 이득은 **테스트용 fake 구현이 trivial**해진다는 점이다.

```python
class FakeRepository(AbstractRepository):
    def __init__(self, batches):
        self._batches = set(batches)

    def add(self, batch):
        self._batches.add(batch)

    def get(self, reference):
        return next(b for b in self._batches if b.reference == reference)

    def list(self):
        return list(self._batches)
```

```python
fake_repo = FakeRepository([batch1, batch2, batch3])
```

> **설계 피드백 도구**: fake를 만들기 어렵다면, 추상화가 너무 복잡하다는 신호다.

---

## 5. ABC vs Duck Typing vs Protocol

본 책은 교육적 목적으로 `abc.ABC`를 사용하지만, 실무에서는 다음 세 옵션이 모두 유효하다.

| 방식 | 장점 | 단점 |
|------|------|------|
| `abc.ABC` | 명시적, IDE 지원 | 보일러플레이트 많음, 무시되기 쉬움 |
| Duck typing | Pythonic, 가벼움 | 인터페이스가 암묵적 |
| **PEP 544 Protocol** | 타입 체킹 + 상속 없음 | 비교적 최근 도입 |

저자들의 실무 권고: ABC가 유지되지 않고 무시되기 시작하면 차라리 삭제하라. Pythonista에게 repository란 "`add(thing)`과 `get(id)`를 가진 객체"일 뿐이다.

---

## 6. Trade-off

> 경제학자는 모든 것의 가격을 알지만 가치는 모르고, 프로그래머는 모든 것의 이득을 알지만 trade-off는 모른다. — Rich Hickey

| Pros | Cons |
|------|------|
| 영속성과 도메인 모델 사이에 단순한 인터페이스 | ORM이 이미 일정 수준의 decoupling을 제공함 |
| Fake repository로 단위 테스트가 매우 쉬워짐 | 손으로 ORM 매핑 유지하는 추가 코드 필요 |
| 영속성을 나중에 생각해도 됨 — 도메인 우선 설계 | 추상화 계층 증가 → 유지보수 비용, "WTF factor" |
| DB 스키마 단순. 객체→테이블 매핑을 완전 제어 | |

> **언제 Repository를 쓰면 안 되는가?** 앱이 단순 CRUD 래퍼라면 도메인 모델도 repository도 필요 없다. 도메인 복잡도가 높을수록 인프라로부터의 독립이 더 큰 가치를 발휘한다.

---

## 요약 및 다음 장 연결

**2장 핵심 정리**
- DIP를 ORM에 적용: **ORM이 모델에 의존**하게 만든다 (역방향)
- Repository는 영속성 위의 단순한 추상화. `add`, `get`만으로 시작
- SQLAlchemy의 Classical Mapping으로 도메인 ↔ 스키마 분리
- Fake repository는 한 페이지짜리 코드 — 테스트가 거저 쉬워짐
- ABC, duck typing, Protocol 모두 유효한 선택. 무엇을 쓰든 핵심은 DIP

**다음 장 예고**
3장은 본격적인 패턴이 아닌 **막간(interlude)** 이다. "좋은 추상화란 무엇인가?"를 디렉터리 동기화 toy example을 통해 탐구한다. 핵심 메시지: 결합도(coupling)를 줄이는 추상화의 선택이 테스트 가능성과 직결된다는 것.
