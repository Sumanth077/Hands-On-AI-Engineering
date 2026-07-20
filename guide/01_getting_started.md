# 01. 시작하기

## 1. 프로젝트 고르기

처음에는 입력과 출력이 명확하고 외부 부작용이 적은 프로젝트를 고른다.

| 관심사 | 추천 출발점 | 먼저 확인할 것 |
| --- | --- | --- |
| 문서 질의 | `rag_apps/` | 지원 파일, 임베딩 모델, 벡터 DB |
| 데이터 질의 | `ai_agents/langchain_data_agent` | 샘플 DB와 읽기 전용 SQL 제한 |
| 이미지 추출 | `OCR/` | 입력 형식, 로컬/원격 모델 여부 |
| 에이전트 | `ai_agents/` | 호출 가능한 도구와 외부 부작용 |
| 오디오·영상 | `audio/`, `multimodal/` | 파일 크기, 코덱과 API 비용 |
| 모델 학습 | `fine_tuning/` | GPU 메모리, 데이터 라이선스와 평가셋 |

README에 요구 사항, 설치, 환경 변수와 실행 명령이 충분히 적혀 있는 프로젝트를 우선한다.

## 2. 독립 환경 만들기

각 프로젝트는 의존성 버전이 다를 수 있으므로 저장소 전체에 하나의 환경을 공유하지 않는다.

```bash
cd <category>/<project_name>
python -m venv .venv
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

```bash
# macOS/Linux
source .venv/bin/activate
python -m pip install --upgrade pip
```

`requirements.txt`가 있으면 다음을 사용한다.

```bash
python -m pip install -r requirements.txt
```

`pyproject.toml`과 `uv.lock`이 있으면 프로젝트 README가 안내하는 `uv sync`를 우선한다. lock 파일은 재현 가능한 버전을 제공하므로 임의로 갱신하지 않는다.

## 3. 환경 변수 설정

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

```bash
# macOS/Linux
cp .env.example .env
```

`.env.example`의 변수마다 다음을 확인한다.

- 어떤 공급자에서 발급하는가?
- 무료 한도와 과금 단위는 무엇인가?
- 읽기 전용 또는 최소 권한 키를 만들 수 있는가?
- 로그와 오류 메시지에 키가 출력되지 않는가?

실제 키, 고객 데이터와 개인 정보는 저장소에 커밋하지 않는다.

## 4. 실행 전 정적 검사

[`examples/01_project_audit.py`](examples/01_project_audit.py)로 프로젝트 구성의 기본 조건을 확인할 수 있다.

```bash
cd guide
python examples/01_project_audit.py ../rag_apps/hybrid_rag_system
```

검사가 성공해도 애플리케이션의 정확성이나 보안을 보장하지는 않는다. README와 코드를 읽어 네트워크 호출, 파일 쓰기, 데이터 삭제와 메시지 전송 지점을 확인한다.

## 5. 작은 입력으로 첫 실행

첫 실행에서는 다음 원칙을 적용한다.

1. 공개되거나 합성한 작은 입력을 사용한다.
2. 한 번의 모델 호출 또는 최소 작업 수로 제한한다.
3. 결과뿐 아니라 검색 문서, 도구 인자, 오류와 지연 시간을 기록한다.
4. 예상 결과를 미리 적고 실제 결과와 비교한다.
5. 실패한 입력도 보존해 회귀 테스트로 만든다.

## 6. 흔한 문제

- **모듈을 찾을 수 없음**: 가상환경 활성화와 `python -m pip --version`의 경로를 확인한다.
- **API 인증 실패**: `.env` 변수명, 키 권한과 공급자 endpoint를 확인한다.
- **모델 이름 오류**: 모델 이름은 자주 바뀌므로 공급자 문서와 프로젝트 README를 확인한다.
- **네이티브 패키지 설치 실패**: Python 버전과 운영체제용 wheel 지원 여부를 확인한다.
- **출력이 비어 있음**: 수집·청킹·검색·생성 단계를 나눠 어느 단계에서 데이터가 사라졌는지 확인한다.

[다음: 핵심 개념과 구조](02_core_concepts.md)
