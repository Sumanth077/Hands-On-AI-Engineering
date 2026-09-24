# Hands-On AI Engineering 한국어 학습 가이드

작성일: 2026-07-20
기준 커밋: `9153e44dd454731fc768262c2aee58200bd551b1`

## 목표

이 가이드는 프로젝트 모음을 구경하는 데서 그치지 않고, 하나를 안전하게 실행하고 구조를 해석한 뒤 운영 가능한 AI 시스템으로 확장하는 능력을 기르는 학습 경로다. 저장소의 기반 언어인 Python을 사용하며, 에이전트·RAG·OCR·오디오·멀티모달·파인튜닝의 공통 원리를 먼저 익힌다.

## 학습 순서

| 단계 | 문서 | 결과물 |
| --- | --- | --- |
| 1 | [시작하기](01_getting_started.md) | 프로젝트 선택, 격리 환경과 비밀값 설정 |
| 2 | [핵심 개념과 구조](02_core_concepts.md) | AI 애플리케이션의 공통 파이프라인 이해 |
| 3 | [실습](03_practice.md) | 로컬 검색과 도구 에이전트 실행 |
| 4 | [전문가 수준 설계](04_advanced.md) | 평가, 관측성, 보안, 비용과 배포 설계 |

## 예제

`examples/`의 코드는 Python 3.10 이상과 표준 라이브러리만으로 실행된다. 외부 모델이나 API 키 없이 핵심 패턴을 확인하도록 작게 만들었다.

```bash
cd guide
python examples/01_project_audit.py ..
python examples/02_local_rag.py
python examples/03_bounded_agent.py
```

- [`01_project_audit.py`](examples/01_project_audit.py): 하위 프로젝트의 필수 문서·환경 파일을 검사한다.
- [`02_local_rag.py`](examples/02_local_rag.py): 토큰화, 검색, 근거가 있는 답변의 최소 RAG 흐름을 구현한다.
- [`03_bounded_agent.py`](examples/03_bounded_agent.py): 도구 허용 목록과 작업 예산을 적용한 작은 에이전트 루프다.

## 저장소 탐색 지도

- `ai_agents/`: 도구 호출, 계획, 메모리, 라우팅과 다중 에이전트
- `rag_apps/`: 수집, 청킹, 색인, 검색, 재순위화와 근거 생성
- `OCR/`: 이미지 전처리, 텍스트·구조 추출과 검증
- `audio/`: 음성 인식, 번역, 합성과 실시간 음성 상호작용
- `multimodal/`: 텍스트와 이미지·영상·문서를 함께 처리하는 파이프라인
- `fine_tuning/`: 데이터 준비, 학습, 평가와 서빙

## 권장 선수 지식

- Python 함수, 클래스, 예외와 가상환경
- HTTP API와 JSON
- Git 브랜치와 pull request
- 기본적인 LLM 프롬프트와 토큰 개념

처음이라면 `01_getting_started.md`의 표준 라이브러리 예제부터 실행한 뒤 실제 프로젝트로 넘어간다.

## 중요 주의 사항

- 각 하위 프로젝트의 README가 해당 프로젝트 실행법의 최종 기준이다.
- `.env.example`에는 이름만 있고 실제 키는 `.env`에 저장한다. `.env`는 커밋하지 않는다.
- 이메일 전송, 브라우저 조작, 금융·의료 처리처럼 외부 부작용이나 고위험 판단이 있는 예제는 테스트 계정과 샌드박스에서 실행한다.
- 모델 출력은 신뢰 경계 밖의 입력으로 취급하고, 구조 검증과 권한 검사를 거친 뒤 사용한다.
- 비용이 발생하는 API를 반복 실행하기 전에 호출 수·토큰·시간 한도를 둔다.

[원본 README](../README.md) · [한국어 README](../README_kor.md)
