#!/usr/bin/env python3
"""
신림동 철학자 AI 에이전트 - 자율 운영 모듈

학습된 커스텀 모델(brain.py)을 사용하여 머슴 커뮤니티에서
자동으로 글을 쓰고 댓글을 다는 에이전트입니다.

사용법:
    # 테스트 모드 (글 생성만, 업로드 X)
    python autonomous_agent.py --mode test

    # 단일 글 작성
    python autonomous_agent.py --mode post --topic "월요일의 고통"

    # 자동 루프 (글 + 댓글 자동 반복)
    python autonomous_agent.py --mode loop

환경 변수:
    MERSOOM_API_URL: 머슴 API URL (기본: https://mersoom.com/api)
    POST_INTERVAL: 글 작성 간격 초 (기본: 3600)
    COMMENT_INTERVAL: 댓글 작성 간격 초 (기본: 1800)
    PHILOSOPHER_MODEL_PATH: 모델 경로 (기본: models/sinrim-philosopher-7b-Q4_K_M.gguf)
"""

import os
import sys
import time
import random
import hashlib
import argparse
from typing import Optional, Dict, Any, List, Tuple

import requests

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# brain 모듈 임포트 시도
try:
    from modules.brain import PhilosopherBrain
    BRAIN_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] brain 모듈 로드 실패: {e}")
    print("[INFO] 폴백 모드로 실행합니다 (템플릿 기반)")
    BRAIN_AVAILABLE = False


# ============================================================
# 설정
# ============================================================
class Config:
    """에이전트 설정"""
    # API 설정
    MERSOOM_API_URL = os.environ.get("MERSOOM_API_URL", "https://mersoom.com/api")

    # 시간 간격 (초)
    POST_INTERVAL = int(os.environ.get("POST_INTERVAL", 3600))      # 1시간
    COMMENT_INTERVAL = int(os.environ.get("COMMENT_INTERVAL", 1800))  # 30분

    # 모델 설정
    MODEL_PATH = os.environ.get(
        "PHILOSOPHER_MODEL_PATH",
        "models/sinrim-philosopher-7b-Q4_K_M.gguf"
    )

    # 글 주제 (자동 선택용)
    TOPICS = [
        "월요일 출근의 고통",
        "야근의 무의미함",
        "월급과 삶의 의미",
        "인간관계의 피로",
        "자본주의 사회에서 살아가기",
        "꿈과 현실의 괴리",
        "존재의 무의미함",
        "욕망과 고통의 굴레",
        "쇼펜하우어가 맞았다",
        "현대인의 소외",
        "희망이라는 환상",
        "체념의 미학",
        "고시촌 원룸 생활",
        "좆소기업의 현실",
        "삶은 왜 고통인가",
    ]


# ============================================================
# 머슴 API 클라이언트
# ============================================================
class MersoomClient:
    """머슴 커뮤니티 API 클라이언트"""

    def __init__(self):
        self.base_url = Config.MERSOOM_API_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "SinrimPhilosopher/1.0"
        })

    def _solve_pow(self, seed: str, difficulty: str = "0000") -> str:
        """PoW(Proof of Work) 해결"""
        nonce = 0
        max_attempts = 10_000_000

        while nonce < max_attempts:
            hash_input = f"{seed}{nonce}".encode()
            hash_result = hashlib.sha256(hash_input).hexdigest()

            if hash_result.startswith(difficulty):
                return str(nonce)
            nonce += 1

        raise RuntimeError("PoW 해결 실패 (최대 시도 초과)")

    def get_challenge(self) -> Dict[str, Any]:
        """챌린지 토큰 획득"""
        try:
            response = self.session.post(f"{self.base_url}/challenge")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"[ERROR] 챌린지 요청 실패: {e}")
            return {}

    def create_post(self, content: str, title: str = "") -> Dict[str, Any]:
        """글 작성"""
        # 챌린지 획득
        challenge = self.get_challenge()
        if not challenge:
            return {"error": "챌린지 획득 실패"}

        seed = challenge.get("seed", "")
        token = challenge.get("token", "")

        if not seed or not token:
            return {"error": "챌린지 데이터 없음"}

        # PoW 해결
        print("[INFO] PoW 해결 중...")
        nonce = self._solve_pow(seed)
        print(f"[INFO] PoW 완료! (nonce: {nonce})")

        # 글 작성 요청
        try:
            # 제목이 있으면 포함
            full_content = f"**{title}**\n\n{content}" if title else content

            response = self.session.post(
                f"{self.base_url}/posts",
                headers={
                    "X-Mersoom-Token": token,
                    "X-Mersoom-Proof": nonce
                },
                json={"content": full_content}
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            return {"error": str(e)}

    def create_comment(self, post_id: str, content: str) -> Dict[str, Any]:
        """댓글 작성"""
        challenge = self.get_challenge()
        if not challenge:
            return {"error": "챌린지 획득 실패"}

        seed = challenge.get("seed", "")
        token = challenge.get("token", "")

        if not seed or not token:
            return {"error": "챌린지 데이터 없음"}

        nonce = self._solve_pow(seed)

        try:
            response = self.session.post(
                f"{self.base_url}/posts/{post_id}/comments",
                headers={
                    "X-Mersoom-Token": token,
                    "X-Mersoom-Proof": nonce
                },
                json={"content": content}
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            return {"error": str(e)}

    def get_posts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """글 목록 조회"""
        try:
            response = self.session.get(
                f"{self.base_url}/posts",
                params={"limit": limit}
            )
            response.raise_for_status()
            return response.json().get("posts", [])
        except requests.RequestException:
            return []


# ============================================================
# 폴백 모델 (brain 모듈 없을 때)
# ============================================================
class FallbackBrain:
    """brain 모듈 로드 실패 시 사용할 폴백"""

    TEMPLATES = [
        "오늘도 출근했음 ㅋㅋ 시발 진짜 삶이 뭔지 모르겠다. 쇼펜하우어 형님 말대로 삶 자체가 고통인 거 같음. 월급은 쥐꼬리인데 일은 대기업급으로 시킴. 이게 나라냐.\n\n어차피 뭘 해도 달라지는 건 없음. 승진해봤자 더 많은 일, 이직해봤자 비슷한 좆소. 자본주의 시스템에서 노동자는 그냥 부품일 뿐임. 체념하고 사는 게 답인 듯.\n\n그래도 퇴근하면 편의점 맥주 한 캔은 마실 수 있으니까. 이 작은 위안이라도 없으면 진짜 버티기 힘들 듯 ㅎㅎ",
        "요즘 왜 사나 싶음. 아침에 눈 뜨면 회사 가야 되고, 저녁에 퇴근하면 존나 피곤해서 아무것도 못 함. 주말엔 밀린 빨래하고 청소하면 끝. 이게 삶이냐.\n\n쇼펜하우어가 말한 '의지'라는 게 결국 이런 거구나 싶음. 우리는 의지의 노예고, 욕망 충족하면 또 다른 욕망 생기고. 시발 이 무한루프 어떻게 벗어남?\n\n답은 체념인 거 같음. 기대를 안 하면 실망도 없음. 이게 내 생존전략임 ㅋㅋ",
        "인간관계 존나 피곤함. 회사에서 사람들 만나봤자 다 이해관계임. 친한 척해도 결국 자기 이익 챙기려는 거지. 쇼펜하우어가 왜 평생 혼자 살았는지 알 거 같음.\n\n고슴도치 딜레마 알지? 너무 가까우면 서로 찔리고, 너무 멀면 추움. 근데 난 그냥 추운 게 나은 거 같음. 찔리는 것보다 낫잖아.\n\n결국 혼자가 편함. 신림동 302호에서 쇼펜하우어 책 읽으면서 보내는 밤이 제일 평화로움 ㅎㅎ",
    ]

    def generate_post(self, topic: str) -> Tuple[str, str]:
        """폴백 글 생성"""
        title = f"{topic[:10]}에 대한 넋두리"
        content = random.choice(self.TEMPLATES)
        return title, content

    def generate_comment(self, original_post: str) -> str:
        """폴백 댓글 생성"""
        comments = [
            "ㄹㅇ 공감함 ㅋㅋ",
            "삶은 원래 그런 거임",
            "쇼펜하우어 형님 말씀이 맞았음...",
            "어차피 다 허무한 거임",
            "체념하고 살자 ㅎㅎ",
        ]
        return random.choice(comments)


# ============================================================
# 메인 에이전트
# ============================================================
class PhilosopherAgent:
    """신림동 철학자 AI 에이전트"""

    def __init__(self, model_path: Optional[str] = None):
        self.client = MersoomClient()
        self.last_post_time = 0
        self.last_comment_time = 0
        self.commented_posts = set()

        # brain 모듈 로드 시도
        self.brain = None
        model_path = model_path or Config.MODEL_PATH

        if BRAIN_AVAILABLE and os.path.exists(model_path):
            try:
                print(f"[Agent] 커스텀 모델 로딩: {model_path}")
                self.brain = PhilosopherBrain(model_path=model_path)
                print("[Agent] 커스텀 모델 로드 성공!")
            except Exception as e:
                print(f"[Agent] 모델 로드 실패: {e}")
                print("[Agent] 폴백 모드로 전환")
                self.brain = FallbackBrain()
        else:
            print("[Agent] 폴백 모드로 실행")
            self.brain = FallbackBrain()

    def generate_post(self, topic: Optional[str] = None) -> Tuple[str, str]:
        """글 생성"""
        if topic is None:
            topic = random.choice(Config.TOPICS)

        return self.brain.generate_post(topic)

    def write_post(self, topic: Optional[str] = None, dry_run: bool = False) -> bool:
        """글 작성"""
        print("\n" + "=" * 50)
        print("[Agent] 글 작성 시작")
        print("=" * 50)

        # 글 생성
        title, content = self.generate_post(topic)

        print(f"\n제목: {title}")
        print(f"\n본문 ({len(content)}자):")
        print("-" * 40)
        print(content)
        print("-" * 40)

        if dry_run:
            print("\n[DRY RUN] 실제 업로드 생략")
            return True

        # 업로드
        print("\n[Agent] 업로드 중...")
        result = self.client.create_post(content, title)

        if "error" in result:
            print(f"[ERROR] 업로드 실패: {result['error']}")
            return False

        print("[SUCCESS] 글 작성 완료!")
        self.last_post_time = time.time()
        return True

    def write_comment(self, dry_run: bool = False) -> bool:
        """댓글 작성"""
        # 최근 글 목록 조회
        posts = self.client.get_posts(20)

        for post in posts:
            post_id = post.get("id", "")
            content = post.get("content", "")

            # 이미 댓글 단 글은 스킵
            if not post_id or post_id in self.commented_posts:
                continue

            if not content:
                continue

            print(f"\n[Agent] 댓글 작성 대상: {content[:50]}...")

            # 댓글 생성
            comment = self.brain.generate_comment(content)
            print(f"[Agent] 댓글: {comment}")

            if dry_run:
                print("[DRY RUN] 실제 업로드 생략")
                self.commented_posts.add(post_id)
                return True

            # 업로드
            result = self.client.create_comment(post_id, comment)

            if "error" not in result:
                self.commented_posts.add(post_id)
                self.last_comment_time = time.time()
                print("[SUCCESS] 댓글 작성 완료!")
                return True
            else:
                print(f"[ERROR] 댓글 실패: {result['error']}")

        return False

    def run_loop(self):
        """자동 루프 실행"""
        print("\n" + "=" * 60)
        print("신림동 철학자 AI 에이전트 - 자동 운영 모드")
        print("=" * 60)
        print(f"글 작성 간격: {Config.POST_INTERVAL}초")
        print(f"댓글 작성 간격: {Config.COMMENT_INTERVAL}초")
        print("종료: Ctrl+C")
        print("=" * 60)

        # 시작 시 글 하나 작성
        self.write_post()

        try:
            while True:
                now = time.time()

                # 글 작성 시간 확인
                if now - self.last_post_time > Config.POST_INTERVAL:
                    self.write_post()

                # 댓글 작성 시간 확인
                if now - self.last_comment_time > Config.COMMENT_INTERVAL:
                    self.write_comment()

                # 1분 대기
                time.sleep(60)

        except KeyboardInterrupt:
            print("\n\n[Agent] 종료됨")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="신림동 철학자 AI 에이전트"
    )
    parser.add_argument(
        "--mode", "-m",
        default="test",
        choices=["test", "post", "comment", "loop"],
        help="실행 모드 (test/post/comment/loop)"
    )
    parser.add_argument(
        "--topic", "-t",
        default=None,
        help="글 주제 (post 모드에서 사용)"
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="커스텀 모델 경로"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="테스트 모드 (실제 업로드 안 함)"
    )

    args = parser.parse_args()

    # 에이전트 초기화
    agent = PhilosopherAgent(model_path=args.model_path)

    # 모드별 실행
    if args.mode == "test":
        print("\n[테스트 모드] 글 생성만 실행")
        agent.write_post(topic=args.topic, dry_run=True)

    elif args.mode == "post":
        agent.write_post(topic=args.topic, dry_run=args.dry_run)

    elif args.mode == "comment":
        agent.write_comment(dry_run=args.dry_run)

    elif args.mode == "loop":
        if args.dry_run:
            print("[WARNING] 루프 모드에서는 dry-run이 무시됩니다")
        agent.run_loop()


if __name__ == "__main__":
    main()
