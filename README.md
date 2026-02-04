# 신림동 철학자 AI 에이전트

> "삶은 고통이다" - 쇼펜하우어

신림동 고시촌 302호에 거주하는 염세적인 30대 회사원 페르소나를 가진 AI 에이전트입니다.

## 프로젝트 구조

```
mersoom_agent_v1/
├── data_scraper.py      # [Step 1] 학습 데이터 수집기
├── train_unsloth.py     # [Step 2] Unsloth 파인튜닝 (Colab용)
├── autonomous_agent.py  # [Step 3] 자율 운영 에이전트
├── modules/
│   ├── __init__.py
│   └── brain.py         # 추론 모듈 (RAG + LLM)
├── knowledge.txt        # RAG 지식 베이스
├── blog_urls.txt        # 블로그 크롤링 URL 목록
├── requirements.txt     # 의존성 목록
└── ddolbae_agent.py     # (레거시) 기존 에이전트
```

## 빠른 시작

### 1. 설치

```bash
pip install -r requirements.txt

# Windows (llama-cpp-python)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### 2. 데이터 수집 (선택)

```bash
# 디시인사이드 + 블로그 크롤링
python data_scraper.py --output train.jsonl --pages 5

# 블로그만 크롤링
python data_scraper.py --skip-dc --output train.jsonl
```

### 3. 모델 학습 (Google Colab)

1. `train_unsloth.py`를 Colab에 업로드
2. GPU 런타임 선택 (T4 이상)
3. 패키지 설치 후 실행
4. 생성된 GGUF 파일 다운로드

### 4. 에이전트 실행

```bash
# 테스트 (글 생성만, 업로드 X)
python autonomous_agent.py --mode test

# 주제 지정 글 작성
python autonomous_agent.py --mode post --topic "월요일의 고통"

# 자동 루프 (글 + 댓글 자동)
python autonomous_agent.py --mode loop
```

## 상세 가이드

### Step 1: 데이터 수집 (data_scraper.py)

디시인사이드 갤러리와 블로그에서 염세적/철학적 글을 수집합니다.

```bash
# 옵션
--output, -o      출력 파일 (기본: train.jsonl)
--min-length, -m  최소 글 길이 (기본: 300자)
--pages, -p       갤러리당 페이지 수 (기본: 5)
--blog-urls       블로그 URL 파일 (기본: blog_urls.txt)
--skip-dc         디시 크롤링 건너뛰기
--skip-blog       블로그 크롤링 건너뛰기
```

**출력 포맷 (Alpaca JSONL):**
```json
{"instruction": "현재 심정을 철학적으로 서술하시오.", "input": "", "output": "..."}
```

### Step 2: 모델 학습 (train_unsloth.py)

Qwen 2.5 7B 모델을 LoRA로 파인튜닝합니다.

**Colab 셀 1: 패키지 설치**
```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes
```

**Colab 셀 2: 학습 실행**
```python
# train_unsloth.py 전체 코드 복사 후 실행
```

**핵심 설정:**
- 베이스 모델: `unsloth/Qwen2.5-7B-Instruct` (4bit)
- LoRA: r=16, alpha=16
- 출력: GGUF (q4_k_m)

### Step 3: 에이전트 실행 (autonomous_agent.py)

학습된 모델로 자율 운영합니다.

**환경 변수:**
```bash
export MERSOOM_API_URL="https://mersoom.com/api"
export POST_INTERVAL=3600       # 글 간격 (초)
export COMMENT_INTERVAL=1800    # 댓글 간격 (초)
export PHILOSOPHER_MODEL_PATH="models/sinrim-philosopher-7b-Q4_K_M.gguf"
```

**모드:**
- `test`: 글 생성만 (업로드 X)
- `post`: 단일 글 작성
- `comment`: 단일 댓글 작성
- `loop`: 자동 반복

### Brain 모듈 (modules/brain.py)

RAG + LLM 추론 모듈입니다.

```python
from modules.brain import PhilosopherBrain

brain = PhilosopherBrain(model_path="models/your-model.gguf")

# 글 생성
title, content = brain.generate_post("회사 생활")

# 댓글 생성
comment = brain.generate_comment("원글 내용...")

# 일반 대화
response = brain.chat("요즘 어때?")
```

**RAG 지식 주입:**
`knowledge.txt`에 페르소나 정보, 커뮤니티 규칙 등을 정의합니다.

## 페르소나 설정

```
이름: 똘배
나이: 30대 중반
거주지: 신림동 고시촌 302호 (반지하)
직업: 좆소기업 사무직 (연봉 2800)
성격: 극도의 염세주의자, 쇼펜하우어 신봉자
말투: 디시 말투 (반말, 음슴체, 비속어)
```

## 주의사항

1. **중국어 방지**: Qwen 모델 특성상 중국어가 출력될 수 있음
   - `repeat_penalty=1.2` 적용
   - 시스템 프롬프트에 "중국어 금지" 명시
   - 후처리로 중국어 문자 제거

2. **크롤링 윤리**
   - robots.txt 준수
   - 요청 간격 유지 (2-5초)
   - 저작권 주의

3. **API 사용**
   - PoW(Proof of Work) 필요
   - 과도한 요청 자제

## 모델 비교

| 버전 | 모델 | 크기 | 특징 |
|------|------|------|------|
| 기존 (ddolbae) | Qwen 3B | 1.93GB | 경량, 빠름 |
| 신규 (philosopher) | Qwen 7B + LoRA | ~4.5GB | 고품질, 페르소나 강화 |

## 라이선스

MIT License
