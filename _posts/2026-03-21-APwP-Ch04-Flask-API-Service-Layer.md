---
layout: post
title: "[파이썬 아키텍처] CH04 - Our First Use Case: Flask API and Service Layer"
categories: [Study Note]
tags: [python, architecture, ddd, service-layer, flask, use-case, dependency-inversion]
math: false
---

## Introduction

지금까지 우리는 **순수한 도메인 모델**(1장)과 그 모델을 영속성으로부터 격리하는 **Repository**(2장), 그리고 좋은 추상화를 식별하는 사고법(3장)을 다루었다. 4장에서는 마침내 도메인을 **외부 세계(HTTP)** 에 연결한다.

도구는 **Flask + Service Layer**. 본 장의 핵심 질문은 다음 한 줄로 압축된다.

> 컨트롤러(Flask)와 도메인 모델 사이에는 **무엇이 들어가야 하는가**?

답은 **유스케이스(use case)를 표현하는 service layer**이다. 이 계층은 orchestration logic을 담아 web layer를 얇게, domain layer를 순수하게 유지한다.

---

## 1. 첫 번째 E2E 테스트

용어 논쟁(E2E vs functional vs integration)을 피하기 위해 저자들은 단순히 **"fast tests"** 와 **"slow tests"** 로 나누는 것도 충분하다고 한다.

```python
@pytest.mark.usefixtures('restart_api')
def test_api_returns_allocation(add_stock):
    sku = random_sku()
    earlybatch = random_batchref(1)
    laterbatch = random_batchref(2)
    add_stock([
        (laterbatch, sku, 100, '2011-01-02'),
        (earlybatch, sku, 100, '2011-01-01'),
    ])
    data = {'orderid': random_orderid(), 'sku': sku, 'qty': 3}
    r = requests.post(f'{config.get_api_url()}/allocate', json=data)
    assert r.status_code == 201
    assert r.json()['batchref'] == earlybatch
```

`random_*()` 헬퍼는 UUID로 데이터 충돌을 방지한다. `add_stock`은 raw SQL로 fixture를 만든다.

---

## 2. 단순 구현 — 그리고 그것의 한계

가장 직관적인 Flask 구현:

```python
@app.route("/allocate", methods=['POST'])
def allocate_endpoint():
    session = get_session()
    batches = repository.SqlAlchemyRepository(session).list()
    line = model.OrderLine(
        request.json['orderid'],
        request.json['sku'],
        request.json['qty'],
    )
    batchref = model.allocate(line, batches)
    return jsonify({'batchref': batchref}), 201
```

문제:
- **`commit()` 누락** → 할당이 DB에 저장되지 않음
- **에러 처리 부재** — `OutOfStock`, `InvalidSku`(존재하지 않는 SKU)는 어디서 잡을까?
- 핸들러에 검증/orchestration 로직을 욱여넣기 시작하면 금방 ugly해진다

```python
def is_valid_sku(sku, batches):
    return sku in {b.sku for b in batches}

@app.route("/allocate", methods=['POST'])
def allocate_endpoint():
    session = get_session()
    batches = repository.SqlAlchemyRepository(session).list()
    line = model.OrderLine(...)
    if not is_valid_sku(line.sku, batches):
        return jsonify({'message': f'Invalid sku {line.sku}'}), 400
    try:
        batchref = model.allocate(line, batches)
    except model.OutOfStock as e:
        return jsonify({'message': str(e)}), 400
    session.commit()
    return jsonify({'batchref': batchref}), 201
```

E2E 테스트가 빠르게 늘어난다 → 곧 **inverted test pyramid (ice-cream cone)** 가 된다.

---

## 3. Service Layer 도입

Flask 핸들러가 하는 일을 분해해 보면:

| 역할 | 위치 |
|------|------|
| HTTP 파싱, JSON 변환, 상태 코드 | **Web layer (Flask)** |
| Repository에서 객체 가져오기, validation, commit | **Service layer (= Use Case)** |
| 비즈니스 규칙 (어느 batch에 할당할지) | **Domain layer** |

이 중간층을 **Service Layer** (또는 **orchestration layer**, **use-case layer**) 라고 부른다.

### Service-layer 함수의 전형적 구조

```python
class InvalidSku(Exception):
    pass

def is_valid_sku(sku, batches):
    return sku in {b.sku for b in batches}

def allocate(line: OrderLine, repo: AbstractRepository, session) -> str:
    batches = repo.list()
    if not is_valid_sku(line.sku, batches):
        raise InvalidSku(f'Invalid sku {line.sku}')
    batchref = model.allocate(line, batches)
    session.commit()
    return batchref
```

전형적 4단계:
1. **Repository에서 객체 fetch**
2. **현재 상태에 대한 검증/사전 조건 확인**
3. **Domain service 호출**
4. **변경 사항 저장(commit)**

> **DIP가 작동한다**: `repo: AbstractRepository`로 타입 힌트를 명시. 테스트는 `FakeRepository`를, 프로덕션은 `SqlAlchemyRepository`를 주입한다.

---

## 4. Service Layer 단위 테스트

Repository 추상화 + Fake 덕분에 service layer를 **빠른 단위 테스트**로 검증할 수 있다.

```python
class FakeSession:
    committed = False
    def commit(self):
        self.committed = True

def test_returns_allocation():
    line = model.OrderLine("o1", "COMPLICATED-LAMP", 10)
    batch = model.Batch("b1", "COMPLICATED-LAMP", 100, eta=None)
    repo = FakeRepository([batch])
    result = services.allocate(line, repo, FakeSession())
    assert result == "b1"

def test_error_for_invalid_sku():
    line = model.OrderLine("o1", "NONEXISTENTSKU", 10)
    batch = model.Batch("b1", "AREALSKU", 100, eta=None)
    repo = FakeRepository([batch])
    with pytest.raises(services.InvalidSku):
        services.allocate(line, repo, FakeSession())

def test_commits():
    ...
    services.allocate(line, repo, session)
    assert session.committed is True
```

`FakeSession`은 임시방편. 6장(Unit of Work)에서 우아하게 제거된다.

---

## 5. 정리된 Flask 핸들러

Service layer 도입 후 Flask는 **순수하게 web 책임만** 진다.

```python
@app.route("/allocate", methods=['POST'])
def allocate_endpoint():
    session = get_session()
    repo = repository.SqlAlchemyRepository(session)
    line = model.OrderLine(
        request.json['orderid'],
        request.json['sku'],
        request.json['qty'],
    )
    try:
        batchref = services.allocate(line, repo, session)
    except (model.OutOfStock, services.InvalidSku) as e:
        return jsonify({'message': str(e)}), 400
    return jsonify({'batchref': batchref}), 201
```

E2E 테스트는 **happy path 1개 + unhappy path 1개**만 남기고, 나머지는 service-layer 단위 테스트로 옮긴다.

---

## 6. "Service"가 두 개? — Domain Service vs Application Service

| 종류 | 다른 이름 | 역할 |
|------|----------|------|
| **Domain Service** | (도메인 개념) | 도메인 모델의 일부지만 entity/value object에 속하지 않는 로직. 예: `model.allocate()`, 세금 계산 |
| **Application Service** (= Service Layer) | use-case service, orchestration layer | 외부 요청을 받아 워크플로를 조율. DB → 도메인 → 저장 |

저자들도 "이름 우리가 지은 거 아니다, 미안하다"고 솔직히 인정한다.

---

## 7. 폴더 구조

이 시점부터 디렉터리 구조가 구체적 의미를 가진다.

```
.
├── config.py
├── domain/                ← 순수 도메인 모델 (1장)
│   ├── __init__.py
│   └── model.py
├── service_layer/         ← 유스케이스 (4장)
│   ├── __init__.py
│   └── services.py
├── adapters/              ← Driven adapters (2장)
│   ├── __init__.py
│   ├── orm.py
│   └── repository.py
├── entrypoints/           ← Driving adapters (4장)
│   ├── __init__.py
│   └── flask_app.py
└── tests/
    ├── unit/              ← 빠른 도메인/서비스 테스트
    ├── integration/       ← ORM, repository
    └── e2e/               ← happy + unhappy 1개씩
```

- **adapters** = secondary/driven adapters (DB, Redis 등)
- **entrypoints** = primary/driving adapters (Flask, CLI 등)
- **port**(추상 인터페이스)는 어댑터와 같은 파일에 둔다

---

## 8. Trade-off

| Pros | Cons |
|------|------|
| 모든 유스케이스를 한 곳에 캡처 | 또 하나의 추상화 계층 |
| Web 책임과 도메인 책임의 명확한 분리 | 너무 많은 로직을 service layer에 넣으면 **Anemic Domain** anti-pattern |
| Repository + Fake와 결합해 빠른 워크플로 테스트 | 단순 web 앱이라면 컨트롤러로 충분 ("fat models, thin controllers") |
| 도메인을 API 뒤에 두어 자유롭게 리팩토링 가능 | |

---

## 요약 및 다음 장 연결

**4장 핵심 정리**
- Service layer = **유스케이스**의 표현. 외부 요청을 받아 도메인을 orchestrate
- 전형적 4단계: fetch → validate → domain call → commit
- DIP를 service layer에 적용 → `AbstractRepository` 의존 → `FakeRepository`로 빠른 테스트
- Flask 핸들러는 web 책임만, E2E 테스트는 happy/unhappy 1개씩으로 충분
- 폴더 구조에 ports & adapters 어휘를 반영한다 (`adapters/`, `entrypoints/`)

**남은 어색함**
- Service layer가 여전히 **`OrderLine` 객체에 결합** → 5장에서 primitive로 풀기
- Service layer가 **session에 직접 결합** → 6장 Unit of Work로 해결

**다음 장 예고**
5장은 **TDD in High Gear and Low Gear** — service layer 도입으로 가능해진 새로운 테스트 전략을 다룬다. 도메인 테스트를 service layer로 올려야 할까? 어떤 종류의 테스트를 어떤 빈도로 작성할까?
