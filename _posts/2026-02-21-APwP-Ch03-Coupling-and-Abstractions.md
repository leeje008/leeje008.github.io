---
layout: post
title: "[파이썬 아키텍처] CH03 - A Brief Interlude: On Coupling and Abstractions"
categories: [Study Note]
tags: [python, architecture, abstraction, coupling, dependency-injection, testing]
math: false
---

## Introduction

3장은 본격적인 패턴 챕터가 아닌 **막간(interlude)** 이다. 2장에서 Repository를 "영속성 위의 추상화"라고 불렀는데, **그래서 좋은 추상화란 무엇인가?** 라는 메타 질문에 답한다.

다루는 도구는 toy 예제 — **두 디렉터리를 동기화하는 프로그램**이다. 이 작은 예제로 다음 세 가지를 시연한다.

1. coupling을 줄이는 추상화의 선택
2. **Functional Core, Imperative Shell (FCIS)** 패턴
3. **Dependency Injection으로 edge-to-edge testing** 하기

저자들은 이 챕터에서 한 가지 강한 의견을 표명한다 — **mock을 쓰지 마라**.

---

## 1. Coupling이 무엇이 문제인가

> 컴포넌트 A를 B를 깨뜨리지 않고 변경할 수 없을 때, 우리는 둘이 **결합(coupled)** 되었다고 한다.

- **국소적 coupling은 좋다**: cohesion 높은 코드 — 시계의 톱니바퀴처럼 맞물려 동작
- **전역적 coupling은 재앙이다**: 변경 비용이 초선형(superlinear)으로 폭발 → Ball of Mud

해법은 단순하다. **추상화로 디테일을 숨기고, 의존하는 종류(arrows)의 개수를 줄인다.**

---

## 2. 토이 예제 — 디렉터리 동기화

요구사항:
1. source에 있고 dest에 없는 파일 → **복사**
2. 같은 내용인데 이름이 다르면 → dest 파일을 **rename**
3. dest에만 있는 파일 → **삭제**

내용 동등성 확인은 SHA-1 해시로 한다. 첫 구현은 다음과 같이 모든 일을 한 함수에서 처리한다.

```python
def sync(source, dest):
    source_hashes = {}
    for folder, _, files in os.walk(source):
        for fn in files:
            source_hashes[hash_file(Path(folder) / fn)] = fn

    seen = set()
    for folder, _, files in os.walk(dest):
        for fn in files:
            dest_path = Path(folder) / fn
            dest_hash = hash_file(dest_path)
            seen.add(dest_hash)
            if dest_hash not in source_hashes:
                dest_path.remove()
            elif dest_hash in source_hashes and fn != source_hashes[dest_hash]:
                shutil.move(dest_path, Path(folder) / source_hashes[dest_hash])

    for src_hash, fn in source_hashes.items():
        if src_hash not in seen:
            shutil.copy(Path(source) / fn, Path(dest) / fn)
```

**문제**: 비즈니스 로직("두 디렉터리의 차이 계산")이 I/O(`pathlib`, `shutil`, `hashlib`)와 단단히 결합. 테스트마다 `tempfile.mkdtemp()`로 실제 디렉터리 만들고 정리해야 한다 — 느리고 장황.

---

## 3. 책임 분리 → 추상화 선택

코드 안에 사실은 **세 가지 책임**이 섞여 있다.

| 책임 | 추상화 후보 |
|------|-------------|
| 1. 파일시스템 조사하여 hash 사전 만들기 | `dict[hash, path]` |
| 2. 파일이 새것/이름변경/잉여인지 판단 | 순수 함수 |
| 3. 복사/이동/삭제 수행 | "action 명령" 리스트 |

핵심 트릭: **"무엇을 할지(what)"와 "어떻게 할지(how)"를 분리한다.**

```python
("COPY", "sourcepath", "destpath"),
("MOVE", "old", "new"),
("DELETE", "path"),
```

이렇게 하면 테스트는 다음과 같이 단순해진다.

```python
def test_when_a_file_exists_in_the_source_but_not_the_destination():
    src_hashes = {'hash1': 'fn1'}
    dst_hashes = {}
    actions = determine_actions(src_hashes, dst_hashes,
                                 Path('/src'), Path('/dst'))
    assert list(actions) == [('copy', Path('/src/fn1'), Path('/dst/fn1'))]
```

> "**실제 파일시스템이 주어졌을 때 무엇이 일어났는가**"를 검증하던 테스트가, "**파일시스템의 추상화가 주어졌을 때 어떤 액션 추상화가 산출되는가**"를 검증하는 테스트로 바뀐다.

---

## 4. Functional Core, Imperative Shell (FCIS)

Gary Bernhardt가 명명한 패턴. 시스템을 두 층으로 나눈다.

- **Functional Core**: 외부 상태에 의존하지 않는 순수 로직
- **Imperative Shell**: 입력 수집 → core 호출 → 결과 적용 (I/O와 부작용 담당)

```python
def sync(source, dest):
    # imperative shell step 1: gather inputs
    source_hashes = read_paths_and_hashes(source)
    dest_hashes = read_paths_and_hashes(dest)
    # step 2: call functional core
    actions = determine_actions(source_hashes, dest_hashes, source, dest)
    # imperative shell step 3: apply outputs
    for action, *paths in actions:
        if action == 'copy':
            shutil.copyfile(*paths)
        elif action == 'move':
            shutil.move(*paths)
        elif action == 'delete':
            os.remove(paths[0])
```

`determine_actions()`는 입력도 출력도 단순한 자료구조 — 테스트가 자명해진다.

---

## 5. Edge-to-Edge Testing with DI

Functional core만 단위 테스트하면 통합 부분의 회귀를 잡지 못한다. 대안은 **`sync()` 자체에 의존성을 인자로 주입**해 fake로 테스트하는 것이다.

```python
def sync(reader, filesystem, source_root, dest_root):
    source_hashes = reader(source_root)
    dest_hashes = reader(dest_root)
    for sha, filename in source_hashes.items():
        if sha not in dest_hashes:
            filesystem.copy(dest_root / filename, source_root / filename)
        elif dest_hashes[sha] != filename:
            filesystem.move(dest_root / dest_hashes[sha], dest_root / filename)
    for sha, filename in dest_hashes.items():
        if sha not in source_hashes:
            filesystem.delete(dest_root / filename)
```

```python
class FakeFileSystem(list):
    def copy(self, src, dest): self.append(('COPY', src, dest))
    def move(self, src, dest): self.append(('MOVE', src, dest))
    def delete(self, dest):    self.append(('DELETE', dest))

def test_rename():
    source = {"sha1": "renamed-file"}
    dest = {"sha1": "original-file"}
    filesystem = FakeFileSystem()
    reader = {"/source": source, "/dest": dest}
    sync(reader.pop, filesystem, "/source", "/dest")
    assert filesystem == [("MOVE", "/dest/original-file", "/dest/renamed-file")]
```

> **장점**: 프로덕션 코드와 동일한 함수를 테스트한다.
> **단점**: stateful 컴포넌트를 명시적으로 인자로 노출시켜야 한다 (DHH가 "test-induced design damage"라 비판한 그것).

저자들의 입장: 이 trade-off는 받아들일 만하다. 테스트 가능성을 위한 설계는 곧 **확장성을 위한 설계**이기 때문이다.

---

## 6. 왜 mock.patch를 쓰지 않는가

저자들은 mock 사용을 **code smell**로 본다. 이유 세 가지:

1. **설계 개선이 없다**: `mock.patch`로 단위 테스트는 가능해지지만, `--dry-run` 플래그를 추가하거나 FTP 서버로 동기화하려면 결국 추상화가 필요하다.
2. **구현 디테일에 결합**: mock 테스트는 "어떻게 호출했는가"를 검증한다 → 리팩토링에 취약.
3. **이야기를 가린다**: setup 코드가 너무 많아져 "이 코드가 무엇을 하는가"를 테스트가 설명하지 못함.

> 우리는 TDD를 **테스트 실천이기 이전에 설계 실천**으로 본다. 테스트는 우리 설계 결정의 기록이다.

### Mocks vs Fakes

| 종류 | 정의 | 학파 |
|------|------|------|
| **Mock** | 어떻게 호출되었는지 검증 (`assert_called_once_with`) | London-school TDD |
| **Fake** | 실제로 동작하는 단순 구현체 (in-memory repo 등) | Classic-style TDD |

본 책은 **classicist** 입장 — 호출 행동이 아니라 **상태(state)** 를 기준으로 테스트한다. (Martin Fowler의 *Mocks Aren't Stubs* 참조)

---

## 7. 좋은 추상화를 찾는 휴리스틱

저자들이 제시하는 자문 질문:

- 이 지저분한 시스템의 상태를 표현할 **익숙한 파이썬 자료구조**가 있는가?
- 그 상태를 반환하는 **단일 함수**를 상상할 수 있는가?
- 시스템을 어디에서 **자를(seam) 수 있는가** — 어디에 추상화를 끼워 넣을 것인가?
- 책임을 **다른 컴포넌트로 나누는 합리적인 방법**은 무엇인가?
- 어떤 **암묵적 개념**을 명시적으로 끄집어낼 수 있는가?
- **의존성**과 **핵심 비즈니스 로직**의 경계는 어디인가?

---

## 요약 및 다음 장 연결

**3장 핵심 정리**
- coupling은 국소적으론 좋고 전역적으론 나쁘다 — 추상화로 줄인다
- "**무엇**"과 "**어떻게**"를 분리하라 (action 리스트로 표현)
- **Functional Core, Imperative Shell**: 순수 로직과 I/O를 격리
- **Edge-to-edge testing**: DI로 fake를 주입해 시스템 전체를 빠르게 테스트
- **Mock 대신 Fake**를 선호 — 설계 개선과 결합도 감소를 동시에
- TDD = 설계 실천 우선, 테스트 실천은 그 다음

**다음 장 예고**
4장에서는 다시 allocation 프로젝트로 돌아온다. **Flask API + Service Layer**를 도입해, 도메인 모델을 외부 세계(HTTP)와 연결한다. orchestration 로직과 비즈니스 로직의 차이가 명확해진다.
