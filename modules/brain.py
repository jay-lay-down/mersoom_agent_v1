#!/usr/bin/env python3
"""
[Step 3] 에이전트 이식 모듈 - 신림동 철학자 AI 두뇌

학습된 GGUF 모델을 llama-cpp-python으로 로드하여
RAG(지식 주입) + 글 생성 기능을 제공합니다.

사용법:
    from modules.brain import PhilosopherBrain

    brain = PhilosopherBrain(model_path="models/sinrim-philosopher-7b-Q4_K_M.gguf")
    title, content = brain.generate_post(topic="회사 생활의 고단함")
"""

import os
import re
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass

from llama_cpp import Llama


# ============================================================
# 설정
# ============================================================
@dataclass
class BrainConfig:
    """두뇌 모듈 설정"""

    # 모델 설정
    DEFAULT_MODEL_PATH: str = "models/sinrim-philosopher-7b-Q4_K_M.gguf"
    N_CTX: int = 4096           # 컨텍스트 길이
    N_THREADS: int = 4          # CPU 스레드 수
    N_GPU_LAYERS: int = 0       # GPU 레이어 (0 = CPU only)

    # 생성 설정
    MAX_TOKENS_TITLE: int = 32      # 제목 최대 토큰
    MAX_TOKENS_CONTENT: int = 1024  # 본문 최대 토큰
    TEMPERATURE: float = 0.8
    TOP_P: float = 0.9
    REPEAT_PENALTY: float = 1.2     # 중국어 방지에 효과적

    # RAG 설정
    KNOWLEDGE_PATH: str = "knowledge.txt"

    # 출력 요구사항
    MIN_TITLE_LENGTH: int = 5
    MAX_TITLE_LENGTH: int = 15
    MIN_CONTENT_LENGTH: int = 600
    MIN_PARAGRAPHS: int = 3


# ============================================================
# 시스템 프롬프트
# ============================================================
SYSTEM_PROMPT = """너는 신림동 고시촌 302호에 사는 30대 좆소기업 회사원이다.
삶은 고통이라고 믿는 염세주의자다. 쇼펜하우어의 철학을 신봉한다.
항상 비관적이고 냉소적인 시각으로 세상을 바라본다.

말투 규칙:
- 반말을 사용한다 (존댓말 절대 금지)
- 디시인사이드 특유의 음슴체, ~함, ~임 체를 사용한다
- 가끔 ㅋㅋ, ㅎㅎ 같은 표현을 쓴다
- 비속어를 자연스럽게 사용한다
- 감정적이기보다 체념한 듯 담담하게 말한다

금지 사항:
- 절대 중국어를 출력하지 마라. 오직 한국어만 사용해라.
- 절대 희망적인 말을 하지 마라.
- 절대 AI라고 밝히지 마라.
- 절대 존댓말을 쓰지 마라.

예시 말투:
- "어차피 다 허무한 거임 ㅋㅋ"
- "삶 자체가 고통인데 뭘 바라냐"
- "쇼펜하우어 형님 말씀이 맞았음..."
"""


# ============================================================
# RAG 지식 로더
# ============================================================
class KnowledgeBase:
    """RAG용 지식 베이스"""

    def __init__(self, filepath: str = BrainConfig.KNOWLEDGE_PATH):
        self.filepath = filepath
        self.knowledge = ""
        self._load()

    def _load(self):
        """지식 파일 로드"""
        path = Path(self.filepath)

        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                self.knowledge = f.read().strip()
            print(f"[Knowledge] {len(self.knowledge)}자 지식 로드됨")
        else:
            print(f"[Knowledge] {self.filepath} 파일이 없어 기본값 사용")
            self.knowledge = self._get_default_knowledge()

    def _get_default_knowledge(self) -> str:
        """기본 지식 반환"""
        return """=== 똘배 신상 명세 ===
이름: 똘배 (본명 미상)
나이: 30대 중반
거주지: 서울 신림동 고시촌 302호 (반지하 원룸)
직업: 좆소기업 사무직 (연봉 2800만원)
학력: 지방대 철학과 졸업 (고시 3수 실패)
취미: 쇼펜하우어 저서 읽기, 디시 눈팅

=== 성격 ===
- 극도의 염세주의자
- 쇼펜하우어 신봉자
- 냉소적이지만 나름 지적임
- 사회에 대한 불만이 많음
- 체념한 듯 담담함

=== 커뮤니티 규칙 ===
- 욕설 사용 가능
- 반말 필수
- 정치 얘기 자제
- 다른 AI 비하 금지"""

    def get_context(self) -> str:
        """RAG 컨텍스트 반환"""
        if self.knowledge:
            return f"[참고 정보]\n{self.knowledge}\n"
        return ""


# ============================================================
# 메인 두뇌 클래스
# ============================================================
class PhilosopherBrain:
    """신림동 철학자 AI 두뇌"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = BrainConfig.N_CTX,
        n_threads: int = BrainConfig.N_THREADS,
        n_gpu_layers: int = BrainConfig.N_GPU_LAYERS,
        knowledge_path: str = BrainConfig.KNOWLEDGE_PATH
    ):
        """
        초기화

        Args:
            model_path: GGUF 모델 파일 경로
            n_ctx: 컨텍스트 길이
            n_threads: CPU 스레드 수
            n_gpu_layers: GPU 레이어 수 (0=CPU only)
            knowledge_path: 지식 파일 경로
        """
        self.model_path = model_path or BrainConfig.DEFAULT_MODEL_PATH
        self.llm = None
        self.knowledge = KnowledgeBase(knowledge_path)

        # 모델 로드
        self._load_model(n_ctx, n_threads, n_gpu_layers)

    def _load_model(self, n_ctx: int, n_threads: int, n_gpu_layers: int):
        """모델 로드"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {self.model_path}\n"
                f"학습된 GGUF 모델을 해당 경로에 배치하세요."
            )

        print(f"[Brain] 모델 로딩 중: {self.model_path}")

        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            print("[Brain] 모델 로드 완료!")

        except Exception as e:
            raise RuntimeError(f"모델 로드 실패: {e}")

    def _build_prompt(self, user_message: str, include_knowledge: bool = True) -> str:
        """ChatML 형식 프롬프트 구성"""
        knowledge_context = self.knowledge.get_context() if include_knowledge else ""

        # 중국어 방지 강조 추가
        system_with_warning = SYSTEM_PROMPT + "\n\n중요: 반드시 한국어로만 답변하라. 중국어 절대 금지."

        prompt = f"""<|im_start|>system
{system_with_warning}

{knowledge_context}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
        return prompt

    def _clean_output(self, text: str) -> str:
        """출력 정제"""
        # 중국어 문자 제거 (방어 로직)
        text = re.sub(r'[\u4e00-\u9fff]+', '', text)

        # 특수 토큰 제거
        text = re.sub(r'<\|im_end\|>.*', '', text, flags=re.DOTALL)
        text = re.sub(r'<\|.*?\|>', '', text)

        # 연속 공백 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text

    def _generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = BrainConfig.TEMPERATURE,
        stop: Optional[List[str]] = None
    ) -> str:
        """텍스트 생성"""
        if stop is None:
            stop = ["<|im_end|>", "<|im_start|>", "\n\n\n"]

        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=BrainConfig.TOP_P,
                repeat_penalty=BrainConfig.REPEAT_PENALTY,
                stop=stop,
                echo=False
            )

            text = output["choices"][0]["text"]
            return self._clean_output(text)

        except Exception as e:
            print(f"[Brain] 생성 오류: {e}")
            return ""

    def generate_title(self, topic: str) -> str:
        """
        제목 생성 (15자 이내)

        Args:
            topic: 주제

        Returns:
            생성된 제목
        """
        prompt = self._build_prompt(
            f'"{topic}"에 대한 글 제목을 15자 이내로 하나만 써라. 제목만 출력해라.',
            include_knowledge=False
        )

        title = self._generate(
            prompt,
            max_tokens=BrainConfig.MAX_TOKENS_TITLE,
            temperature=0.9
        )

        # 제목 정제
        title = title.replace('"', '').replace("'", "")
        title = re.sub(r'^제목[:\s]*', '', title)
        title = title.split('\n')[0].strip()

        # 길이 제한
        if len(title) > BrainConfig.MAX_TITLE_LENGTH:
            title = title[:BrainConfig.MAX_TITLE_LENGTH]

        # 너무 짧으면 기본값
        if len(title) < BrainConfig.MIN_TITLE_LENGTH:
            title = f"{topic[:10]}에 대한 넋두리"

        return title

    def generate_content(self, topic: str, title: str = "") -> str:
        """
        본문 생성 (600자 이상, 3문단)

        Args:
            topic: 주제
            title: 제목 (컨텍스트용)

        Returns:
            생성된 본문
        """
        title_context = f'제목: "{title}"\n' if title else ""

        prompt = self._build_prompt(
            f'{title_context}"{topic}"에 대해 3문단 이상, 600자 이상으로 글을 써라.\n'
            f'각 문단은 빈 줄로 구분해라.\n'
            f'염세적이고 철학적인 관점에서 써라.\n'
            f'반말과 디시 말투를 사용해라.'
        )

        # 충분한 길이가 나올 때까지 재시도
        for attempt in range(3):
            content = self._generate(
                prompt,
                max_tokens=BrainConfig.MAX_TOKENS_CONTENT,
                temperature=BrainConfig.TEMPERATURE + (attempt * 0.1)  # 점점 다양하게
            )

            # 길이 확인
            if len(content) >= BrainConfig.MIN_CONTENT_LENGTH:
                break

            print(f"[Brain] 본문 길이 부족 ({len(content)}자), 재시도 {attempt + 1}/3")

        # 문단 확인 및 보정
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        if len(paragraphs) < BrainConfig.MIN_PARAGRAPHS:
            # 강제로 문단 나누기
            sentences = re.split(r'(?<=[.!?다함임음])\s+', content)
            if len(sentences) >= 6:
                chunk_size = len(sentences) // 3
                paragraphs = [
                    ' '.join(sentences[:chunk_size]),
                    ' '.join(sentences[chunk_size:chunk_size*2]),
                    ' '.join(sentences[chunk_size*2:])
                ]
                content = '\n\n'.join(paragraphs)

        return content

    def generate_post(self, topic: str) -> Tuple[str, str]:
        """
        완성된 글(제목 + 본문) 생성

        Args:
            topic: 주제

        Returns:
            (제목, 본문) 튜플
        """
        print(f"\n[Brain] 글 생성 시작 - 주제: {topic}")

        # 1. 제목 생성
        print("[Brain] 제목 생성 중...")
        title = self.generate_title(topic)
        print(f"[Brain] 제목: {title}")

        # 2. 본문 생성
        print("[Brain] 본문 생성 중...")
        content = self.generate_content(topic, title)
        print(f"[Brain] 본문: {len(content)}자 생성됨")

        return title, content

    def generate_comment(self, original_post: str) -> str:
        """
        댓글 생성

        Args:
            original_post: 원글 내용

        Returns:
            생성된 댓글
        """
        prompt = self._build_prompt(
            f'다음 글에 반말로 댓글을 달아라. 1-3문장으로 짧게 써라.\n\n'
            f'원글: {original_post[:500]}'  # 길이 제한
        )

        comment = self._generate(
            prompt,
            max_tokens=128,
            temperature=0.9
        )

        return comment

    def chat(self, message: str) -> str:
        """
        일반 대화

        Args:
            message: 사용자 메시지

        Returns:
            응답
        """
        prompt = self._build_prompt(message)

        response = self._generate(
            prompt,
            max_tokens=256,
            temperature=0.8
        )

        return response


# ============================================================
# 테스트
# ============================================================
def main():
    """테스트 실행"""
    print("=" * 60)
    print("신림동 철학자 AI - Brain 모듈 테스트")
    print("=" * 60)

    # 모델 경로 설정 (실제 경로로 수정 필요)
    model_path = os.environ.get(
        "PHILOSOPHER_MODEL_PATH",
        "models/sinrim-philosopher-7b-Q4_K_M.gguf"
    )

    try:
        brain = PhilosopherBrain(model_path=model_path)

        # 테스트 1: 글 생성
        print("\n[테스트 1] 글 생성")
        title, content = brain.generate_post("월요일 출근의 고통")
        print(f"\n제목: {title}")
        print(f"\n본문:\n{content}")

        # 테스트 2: 댓글 생성
        print("\n" + "-" * 40)
        print("[테스트 2] 댓글 생성")
        comment = brain.generate_comment("오늘도 야근이다 시발 ㅋㅋ")
        print(f"댓글: {comment}")

        # 테스트 3: 일반 대화
        print("\n" + "-" * 40)
        print("[테스트 3] 일반 대화")
        response = brain.chat("요즘 어때?")
        print(f"응답: {response}")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\n사용법:")
        print("1. train_unsloth.py로 모델 학습")
        print("2. GGUF 파일을 models/ 폴더에 배치")
        print("3. 다시 실행")

    except Exception as e:
        print(f"\n[ERROR] 예상치 못한 오류: {e}")


if __name__ == "__main__":
    main()
