<p align="center">
  <a href="https://aiengineering.beehiiv.com/">
    <img src="assets/theaiengineering_logo.jpeg" alt="Hands-On AI Engineering 배너" width="150">
  </a>
</p>
<div align="center">

# 🚀 Hands-On AI Engineering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[English](README.md) · [한국어 학습 가이드](guide/README.md)

</div>

언어 모델, 멀티모달 모델, OCR 시스템, RAG 파이프라인과 AI 에이전트를 아우르는 실전·프로덕션 지향 AI 프로젝트 모음입니다. 각 프로젝트는 실제 AI 애플리케이션을 학습하고 실험하며 구축할 수 있도록 설계되었습니다.

## 📋 목차

- [🎯 이 저장소를 사용하는 이유](#-이-저장소를-사용하는-이유)
- [🗂️ 프로젝트 카테고리](#️-프로젝트-카테고리)
- [🚀 시작하기](#-시작하기)
- [🤝 기여하기](#-기여하기)
- [📜 라이선스](#-라이선스)

---

## 🎯 이 저장소를 사용하는 이유

- **실습 중심 학습**: 각 프로젝트에 완성된 코드, 설정 방법과 문서가 포함됩니다.
- **프로덕션 지향**: 실제 환경에 맞게 확장할 수 있는 모범 사례를 지향합니다.
- **다양한 사용 사례**: RAG부터 다중 에이전트 워크플로와 전문 애플리케이션까지 다룹니다.
- **다양한 모델 공급자**: OpenAI, Anthropic, Google과 오픈 소스 모델을 사용합니다.
- **활발한 커뮤니티**: 프로젝트가 정기적으로 추가·갱신됩니다.

---

## 🗂️ 프로젝트 카테고리

### 🤖 AI 에이전트

다양한 자동화 작업을 수행하는 지능형 에이전트입니다.

- [**Multi-Agent Financial Analyst**](./ai_agents/multi_agent_financial_analyst) — 전문 에이전트 팀이 종합 금융 분석을 수행합니다.
- [**FinAgent**](./ai_agents/finagent) — 주식시장 분석과 인사이트를 제공하는 금융 보조 에이전트입니다.
- [**Daily AI News Digest**](./ai_agents/daily-news-digest) — Karpathy가 선별한 92개 기술 블로그에서 최근 뉴스를 평가해 매일 Telegram으로 전달합니다.
- [**Agentic Form Filler**](./ai_agents/agentic-form-filler) — 문서 레이아웃 분석과 다중 턴 정보 수집으로 양식을 채웁니다.
- [**AI Travel Planning Agent**](./ai_agents/ai_travel_planning_agent) — 항공편, 호텔과 일자별 일정을 포함한 여행 계획을 생성합니다.
- [**Competitive Intelligence Agent**](./ai_agents/competitive_intelligence_agent) — 자사 관점에서 경쟁사를 분석해 영업 배틀카드를 만듭니다.
- [**Multi-Agent Research Assistant (AG2)**](./ai_agents/multi_agent_research_assistant_ag2) — 세 전문 에이전트가 협업해 구조화된 연구 보고서를 작성합니다.
- [**Self-Reflective Agentic RAG**](./ai_agents/agentic_rag_system) — 검색 문맥을 평가하고 필요하면 질문을 다시 작성한 뒤 검증된 근거로 답합니다.
- [**Agentic SQL Search**](./ai_agents/agentic_sql_search) — 자연어 질문을 SQL로 변환·실행하고 결과를 설명합니다.
- [**Stock Portfolio Analyst**](./ai_agents/stock_portfolio_analyst) — 손익, 집중 위험과 리밸런싱을 분석합니다.
- [**Eagle Eye**](./ai_agents/eagle_eye) — GitHub PR diff를 검토하고 사용자 승인 후 피드백을 게시합니다.
- [**CartMate — AI Customer Support Agent**](./ai_agents/ai_customer_support_agent) — 고객을 기억하며 이전 대화부터 이어가는 전자상거래 지원 에이전트입니다.
- [**Multi-Agent Coding Assistant**](./ai_agents/multi_agent_coding_assistant) — 계획자, 코더와 리뷰어가 협업하는 코딩 파이프라인입니다.
- [**Startup Analyst**](./ai_agents/startup_analyst) — 기업 사이트를 조사해 시장, 재무, 팀과 위험을 다루는 투자 보고서를 만듭니다.
- [**Research Team**](./ai_agents/research_team) — 웹과 내부 문서를 조사한 뒤 팀 리더가 결과를 종합합니다.
- [**GitHub Intelligence Agent**](./ai_agents/github_intelligence_agent) — 저장소, 기여자, 이슈와 코드베이스에 관한 질문을 조사합니다.
- [**Smolagents Code Agent**](./ai_agents/smolagents_code_agent) — 각 단계에서 Python 코드를 작성·실행하며 웹 정보를 활용합니다.
- [**Agent Discovery Agent**](./ai_agents/agent_discovery_agent) — 여러 에이전트 생태계와 프로토콜의 도구를 검색·비교합니다.
- [**Cal Scheduling Agent**](./ai_agents/cal_scheduling_agent) — 자연어로 Cal.com 예약을 생성·변경·취소하고 가용 시간을 확인합니다.
- [**Hacker News Newsletter Agent**](./ai_agents/hacker_news_newsletter_agent) — 최신 기사 내용을 수집해 HTML 뉴스레터를 만들고 이메일로 보냅니다.
- [**Hotel Finder Agent**](./ai_agents/hotel_finder_agent) — 위치, 날짜, 가격, 등급과 편의시설을 기준으로 호텔을 검색합니다.
- [**Marketing Strategy Agent**](./ai_agents/marketing_strategy_agent) — 시장 분석, 전략과 크리에이티브를 순차 생성하는 다중 에이전트입니다.
- [**Brand Monitor**](./ai_agents/brand_monitor_agent) — 웹과 소셜 플랫폼의 브랜드 언급을 수집해 채널별 인텔리전스 브리프를 만듭니다.
- [**AI Debate Agent**](./ai_agents/ai_debate_agent) — 두 LLM이 상반된 입장을 논하고 심판 모델이 평가합니다.
- [**Browser Automation Agent**](./ai_agents/browser_automation_agent) — 자연어 지시에 따라 브라우저를 자율 조작합니다.
- [**Documentation QnA Agent**](./ai_agents/documentation_qna_agent) — 문서 URL의 내용을 가져와 질문에 답합니다.
- [**Job Posting Agent**](./ai_agents/job_posting_agent) — 회사와 직무에 맞춘 채용 공고를 생성합니다.
- [**LangChain Data Agent**](./ai_agents/langchain_data_agent) — 자연어로 Chinook SQLite 데이터베이스를 질의합니다.
- [**Travel Planner Agent**](./ai_agents/travel_planner_agent) — 날씨, 예산, 준비물과 일정을 포함한 여행 계획을 만듭니다.
- [**Personal Finance Agent**](./ai_agents/personal_finance_agent) — 은행 거래 CSV를 분류하고 소비 내역 질문에 답합니다.
- [**Offline Medical Agent**](./ai_agents/offline_medical_agent) — 원격 진료 환경을 위한 완전 오프라인 임상 프로토콜 RAG입니다.
- [**Customer Query Routing and Resolution Agent**](./ai_agents/customer_query_routing_agent) — 고객 문의를 부서로 라우팅하고 로컬 영속 메모리를 이용해 근거 있는 답을 생성합니다.
- [**Email Auto Responder**](./ai_agents/email_auto_responder) — 읽지 않은 Gmail을 분류하고 전문적인 회신 초안을 만듭니다.
- [**LLM Agri Bot**](./ai_agents/llm_agri_bot) — 작물 건강, 날씨, 해충과 재배 시기를 안내하는 농업 보조 에이전트입니다.

### 📸 OCR

이미지와 문서에서 구조와 의미를 추출하는 프로젝트입니다.

- [**AI Receipt and Expense Tracker**](./OCR/receipt_expense_tracker) — 영수증 사진을 구조화하고 로컬 SQLite 지출 장부에 기록합니다.
- [**Image-to-Structured-Data Extractor**](./OCR/image_to_structured_data) — 이미지를 검증된 구조화 JSON으로 변환합니다.
- [**LaTeX Formula OCR**](./OCR/latex_formula_ocr) — 이미지와 PDF의 수식을 LaTeX로 추출합니다.
- [**Medical Prescription Digitizer**](./OCR/medical_prescription_digitizer) — 처방전을 구조화하고 RxNorm으로 약품명을 검증합니다.

### 🎧 오디오

오디오 이해와 분석 프로젝트입니다.

- [**Music Explorer**](./audio/music_explorer) — 음원이나 YouTube 영상에 대해 전사, 감정, 악기와 시간대별 분석을 질의합니다.
- [**Multilingual Audio Translator**](./audio/multilingual_audio_translator) — 음성을 전사·번역하고 합성 음성으로 재생합니다.
- [**Customer Support Voice Agent**](./audio/customer_support_voice_agent) — 통화에 응답하고 webhook으로 실시간 문맥을 주입하는 고객 지원 음성 에이전트입니다.

### 🎬 멀티모달

비전, 영상과 언어 모델을 결합한 프로젝트입니다.

- [**GLM-OCR Pro**](./multimodal/glm_ocr_pro) — Ollama 기반 GLM-OCR로 이미지와 PDF를 구조화된 Markdown으로 변환합니다.
- [**Video Understanding Agent**](./multimodal/video_understanding_agent) — YouTube 영상을 장, 핵심 요점과 실행 항목으로 요약합니다.
- [**Multimodal Weather App**](./multimodal/multimodal_weather_app) — 지도 이미지에서 도시를 찾고 실시간 날씨를 조회합니다.
- [**Multimodal RAG**](./multimodal/multimodal_rag) — 텍스트, URL, PDF, 이미지, 오디오와 영상을 하나의 검색 인덱스에 수집합니다.
- [**Image Question Answering**](./multimodal/image_question_answering) — PDF 페이지를 이미지로 렌더링해 차트와 표에 관한 시각 질문에 답합니다.
- [**Medical Document Parser**](./multimodal/medical_document_parser) — 의료 PDF와 이미지에서 구조화된 임상 프로필을 추출합니다.

### 📚 RAG 애플리케이션

외부 지식을 검색해 답변의 근거를 보강하는 시스템입니다.

- [**Agentic RAG with O3-Mini & DuckDuckGo**](./rag_apps/agentic_rag_with_o3_mini_and_duckduckgo) — 실시간 웹 검색을 사용하는 에이전트형 RAG입니다.
- [**Agentic RAG with Qwen & FireCrawl**](./rag_apps/agentic_rag_with_qwen_and_firecrawl) — 웹 스크래핑과 검색을 결합한 RAG입니다.
- [**Vision RAG**](./rag_apps/vision_rag) — 시각 콘텐츠를 처리하고 질의하는 멀티모달 RAG입니다.
- [**Clinical RAG with ADE**](./rag_apps/clinical_rag_with_ade) — 시각 중심 문서 파싱과 근거 기반 추론을 결합한 임상 RAG입니다.
- [**YouTube Transcript RAG**](./rag_apps/youtube_transcript_rag) — 영상 전사를 검색하고 타임스탬프가 있는 답변을 제공합니다.
- [**GraphRAG Knowledge System**](./rag_apps/graphrag_knowledge_system) — 문서에서 지식 그래프를 구축하고 개체·주제 질의를 지원합니다.
- [**Hybrid RAG System**](./rag_apps/hybrid_rag_system) — 지식 그래프와 벡터 저장소의 검색 문맥을 융합합니다.
- [**HyDE RAG**](./rag_apps/hyde_rag) — 가상 답변 임베딩을 생성·평균해 관련 청크를 검색합니다.
- [**Rock Music RAG**](./rag_apps/rock_music_rag) — Wikipedia 기반 록 음악 지식 베이스와 BM25 검색을 제공합니다.
- [**RAG Agent with Database Routing**](./rag_apps/rag_agent_with_database_routing) — 여러 전문 Qdrant DB로 질의를 라우팅하고 필요하면 웹 검색으로 대체합니다.
- [**Reasoning RAG**](./rag_apps/reasoning_rag) — 웹 자료에 대한 인용 답변과 단계별 실행 추적을 제공합니다.

### 🎛️ 파인튜닝

특정 작업에 맞춰 모델을 학습·미세조정하는 프로젝트입니다.

- [**Text-to-SQL Inventory Specialist**](./fine_tuning/text_to_sql_inventory) — 미세조정된 소형 모델로 자연어 재고 질문을 SQL로 변환합니다.

---

## 🚀 시작하기

이 저장소의 프로젝트는 서로 독립적입니다. 먼저 [한국어 학습 가이드](guide/README.md)에서 프로젝트 선택법과 안전한 실행 절차를 확인하세요.

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering
cd <category>/<project>

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

프로젝트에 `pyproject.toml`이 있으면 해당 README의 `uv` 또는 패키지 설치 명령을 우선합니다. `.env.example`을 `.env`로 복사하되 실제 API 키는 절대 커밋하지 마세요.

---

## 🤝 기여하기

기여를 환영합니다. 자세한 내용은 [CONTRIBUTING.md](CONTRIBUTING.md)를 확인하세요.

1. 먼저 프로젝트 또는 개선 내용을 설명하는 issue를 만듭니다.
2. 적절한 카테고리를 선택하고 `snake_case` 폴더명을 사용합니다.
3. 하나의 pull request에는 하나의 프로젝트만 포함합니다.
4. 프로젝트에 포괄적인 `README.md`, `requirements.txt` 또는 `pyproject.toml`, `.env.example`을 포함합니다.
5. API 키와 비밀값이 커밋되지 않았는지 확인하고 실행을 검증합니다.

새 프로젝트 문서는 [.github/README_TEMPLATE.md](.github/README_TEMPLATE.md)를 따릅니다.

---

## 📜 라이선스

이 저장소는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](./LICENSE)를 확인하세요.

---

## 🙏 감사의 말

Hands-On AI Engineering 프로젝트에 기여한 모든 분께 감사드립니다.

<div align="center">

**[AI Engineering Community](https://aiengineering.beehiiv.com/)가 ❤️를 담아 만들었습니다.**

후원 또는 협업 문의: [sumanth@devable.ai](mailto:sumanth@devable.ai)

[⬆ 맨 위로](#-hands-on-ai-engineering)

</div>
