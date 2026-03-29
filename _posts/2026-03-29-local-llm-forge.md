---
layout: post
title: "[Project] Local LLM Forge - Apple Silicon 로컬 LLM 최적화 자동화 도구"
categories: [Project]
tags: [python, mlx, apple-silicon, llm, quantization, optimization]
math: false
---

## 프로젝트 개요

Apple Silicon Mac에서 대형 언어 모델을 로컬 실행하기 위한 최적화 자동화 도구. 하드웨어 분석부터 양자화 전략 선택, 모델 변환, 추론 최적화, 배포까지 전 과정을 **단일 CLI 파이프라인**으로 처리한다. MLX 프레임워크 기반으로 Apple Silicon의 Unified Memory를 최대한 활용한다.

> GitHub: [leeje008/local-llm-forge](https://github.com/leeje008/local-llm-forge)

---

## 핵심 문제와 해결 접근

### 문제 1: LLM 로컬 실행 가능 여부 판단이 어렵다

"이 모델을 내 Mac에서 돌릴 수 있는가?"라는 질문에 답하려면 모델 파라미터 수, 양자화 수준별 메모리 요구량, KV 캐시 오버헤드, 시스템 여유 메모리 등을 종합적으로 계산해야 한다. 예를 들어 72B 모델은 FP16 기준 153GB가 필요하지만, 3-bit 양자화를 적용하면 ~31GB로 줄어든다. 이 계산을 수동으로 하는 것은 비현실적이다.

**해결: 5단계 의사결정 트리 + 하드웨어 자동 프로파일링**

```
[1] 하드웨어 프로파일링
    ├── CPU 코어 수, GPU 코어 수, 메모리 대역폭 감지
    └── Unified Memory 총량 + 시스템 사용량 → 가용 메모리 계산
    ↓
[2] 모델 분석
    ├── 파라미터 수 추출 (config.json / safetensors 메타데이터)
    └── 아키텍처 파싱 (hidden_size, num_layers, num_heads 등)
    ↓
[3] 양자화별 메모리 예측
    ├── FP16 / int8 / int4 / HQQ 3-bit / HQQ 2-bit 각각 계산
    └── KV 캐시 오버헤드 포함 (context length 기반)
    ↓
[4] 실행 가능성 라우팅
    ├── ✅ DIRECT: FP16으로도 여유 있음
    ├── ⚡ QUANTIZE: 양자화 적용 시 실행 가능
    └── ❌ REJECT: 어떤 양자화로도 불가
    ↓
[5] 최적 전략 추천
    └── 품질-메모리 트레이드오프 기반 양자화 수준 + 배치 설정 제안
```

하드웨어 정보는 시스템 API를 통해 자동 수집하므로 사용자가 스펙을 직접 입력할 필요가 없다. M4 Pro 48GB 기준으로 Qwen2.5-72B를 3-bit로 양자화하면 약 31GB에 실행 가능하다는 판단을 자동으로 내린다.

### 문제 2: 양자화 전략 선택의 복잡성

양자화 방식마다 품질 손실, 메모리 절감률, 추론 속도가 다르다. MLX 네이티브(int4/int8)는 안정적이지만 압축률이 제한적이고, HQQ(2-3bit)는 극단적 압축이 가능하지만 품질 저하 위험이 있다. 모델 크기와 하드웨어 여유 메모리에 따라 최적 전략이 달라진다.

**해결: 메모리 버짓 기반 자동 양자화 선택**

| 양자화 방식 | 압축률 | 품질 영향 | 적용 조건 |
|-----------|--------|----------|----------|
| FP16 (무변환) | 1x | 없음 | 메모리 여유 충분 |
| int8 (MLX) | 2x | 최소 | FP16 불가, int8은 가능 |
| int4 (MLX) | 4x | 경미 | 대부분의 로컬 실행 시나리오 |
| HQQ 3-bit | ~5.3x | 중간 | int4로도 메모리 초과 시 |
| HQQ 2-bit | ~8x | 상당 | 극단적 메모리 제약 (최후 수단) |

시스템은 가용 메모리에서 KV 캐시와 시스템 오버헤드를 뺀 **순수 모델 버짓**을 계산한 뒤, 품질 손실이 가장 적은 양자화 수준을 자동 선택한다. 사용자가 원하면 수동으로 양자화 수준을 지정할 수도 있다.

### 문제 3: Apple Silicon 추론 성능을 더 끌어올릴 수 있는가

기본 MLX 추론만으로는 토큰 생성 속도가 제한적이다. 특히 긴 프롬프트나 대형 모델에서는 지연이 체감된다.

**해결: 4가지 추론 최적화 기법 적용**

| 최적화 기법 | 효과 | 원리 |
|-----------|------|------|
| **Speculative Decoding** | 1.2x 속도 향상 | 소형 드래프트 모델로 후보 토큰 배치 생성 → 대형 모델이 검증 |
| **KV Cache 양자화** | 메모리 50% 절감 | Attention의 Key/Value 캐시를 저정밀도로 저장 |
| **프롬프트 캐싱** | 반복 프롬프트 즉시 처리 | 동일 프롬프트 프리픽스의 KV 상태를 캐시하여 재사용 |
| **Metal 커스텀 커널** | GPU 활용 극대화 | Apple GPU에 최적화된 연산 커널 직접 작성 |

Qwen2.5-7B 4-bit 기준 M4 Pro에서 기본 55.7 tok/s → Speculative Decoding 적용 시 67.6 tok/s로 약 21% 향상을 달성했다.

---

## 시스템 아키텍처

```
forge CLI
    │
    ├── forge analyze ─── Analyzer ──── 모델 메타데이터 + 파라미터 추출
    ├── forge route ───── Router ────── 하드웨어 프로파일 → 실행 가능성 판정
    ├── forge optimize ── Optimizer ─── 양자화 전략 선택 + 모델 변환
    ├── forge run ──────── Engine ────── MLX 추론 (+ Speculative Decoding)
    ├── forge deploy ──── Pipeline ──── 서버 배포 + Ollama 설정 생성
    └── forge bench ───── Optimizer ─── 성능 벤치마크 (tok/s, 메모리, 지연)
    │
    └── src/forge/
         ├── analyzer/   (하드웨어 + 메모리 분석)
         ├── optimizer/  (양자화 전략 + 프로파일링)
         ├── pipeline/   (모델 변환 + 배포)
         ├── engine/     (MLX 추론 + 최적화)
         ├── router/     (실행 가능성 라우팅)
         └── metal/      (GPU 커스텀 커널)
```

---

## CLI 명령어

| 명령어 | 기능 | 설명 |
|-------|------|------|
| `forge analyze` | 모델 분석 | 파라미터 수, 아키텍처, 양자화별 메모리 예측 |
| `forge route` | 실행 가능성 판정 | 하드웨어 대비 모델 적합성 라우팅 |
| `forge optimize` | 최적화 + 변환 | 모델 다운로드 → 양자화 → 로컬 저장 |
| `forge run` | 텍스트 생성 | MLX 기반 추론 (스트리밍 출력) |
| `forge deploy` | 서버 배포 | API 서버 기동 + Ollama 호환 설정 생성 |
| `forge bench` | 벤치마크 | tok/s, 메모리 사용량, 첫 토큰 지연 측정 |

---

## 성능 벤치마크 (M4 Pro 48GB)

| 모델 | 양자화 | 메모리 | 기본 속도 | + Speculative | 비고 |
|------|--------|--------|----------|--------------|------|
| Qwen2.5-7B | int4 | ~4.5GB | 55.7 tok/s | 67.6 tok/s | +21% |

KV Cache 양자화를 병행하면 긴 컨텍스트(4K+ 토큰)에서 메모리 사용량이 추가로 50% 절감된다.

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| 언어 | Python (95.7%) + Metal Shading Language (4.3%) |
| 추론 프레임워크 | MLX (Apple Silicon 네이티브) |
| 양자화 | MLX int4/int8 + HQQ 2-3bit |
| CLI | Click / Typer |
| 하드웨어 감지 | macOS system_profiler + psutil |
| 모델 허브 | Hugging Face Hub |
| 배포 | Ollama 호환 설정 자동 생성 |
| 패키지 관리 | uv |

---

## 요구 사항

- Apple Silicon Mac (M1 이상)
- Python 3.12+
- Unified Memory 16GB 이상 (48GB 권장)
- macOS 14.0+
