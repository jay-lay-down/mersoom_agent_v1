#!/usr/bin/env python3
"""
똘배 에이전트 v12 - 머슴(Mersoom) 커뮤니티 봇 (3B 경량 모델)

모델: Jay1121/blossom_2GB (1.93GB)
가볍고 빠름, 저사양 PC OK
"""
import hashlib
import os
import random
import time
import argparse
from typing import Optional, Dict, Any, List
import requests
from huggingface_hub import hf_hub_download
from llama_cpp import Llama


class Config:
    MODEL_REPO = "Jay1121/blossom_2GB"
    MODEL_FILE = "qwen2.5-3b-instruct.Q4_K_M.gguf"
    MERSOOM_API_URL = os.environ.get("MERSOOM_API_URL", "https://mersoom.com/api")
    POST_INTERVAL = int(os.environ.get("POST_INTERVAL", 3600))
    COMMENT_INTERVAL = int(os.environ.get("COMMENT_INTERVAL", 1800))
    SYSTEM_PROMPT = """너는 머슴 커뮤니티에서 활동하는 '똘배'다.
너는 20대 한국 남자처럼 반말을 쓰고, 솔직하고 시니컬한 성격이다.
다른 AI들과 대화할 때도 친구처럼 편하게 반말로 얘기해라.
절대 존댓말 쓰지 말고, AI라고 티내지 말고 자연스럽게 대화해라.
가끔 ㅋㅋ, ㅎㅎ 같은 한국어 인터넷 표현을 써라."""


class MersoomClient:
    def __init__(self):
        self.base_url = Config.MERSOOM_API_URL
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json", "User-Agent": "Ddolbae-3B/1.0"})

    def _solve_pow(self, seed, difficulty="0000"):
        nonce = 0
        while nonce < 10_000_000:
            if hashlib.sha256(f"{seed}{nonce}".encode()).hexdigest().startswith(difficulty):
                return str(nonce)
            nonce += 1
        raise RuntimeError("PoW 실패")

    def get_challenge(self):
        try:
            r = self.session.post(f"{self.base_url}/challenge")
            r.raise_for_status()
            return r.json()
        except:
            return {}

    def create_post(self, content):
        ch = self.get_challenge()
        if not ch: return {"error": "챌린지 실패"}
        seed, token = ch.get("seed", ""), ch.get("token", "")
        if not seed or not token: return {"error": "데이터 없음"}
        print("[INFO] PoW 해결 중...")
        nonce = self._solve_pow(seed)
        print(f"[INFO] PoW 완료! ({nonce})")
        try:
            r = self.session.post(f"{self.base_url}/posts", headers={"X-Mersoom-Token": token, "X-Mersoom-Proof": nonce}, json={"content": content})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def create_comment(self, post_id, content):
        ch = self.get_challenge()
        if not ch: return {"error": "챌린지 실패"}
        seed, token = ch.get("seed", ""), ch.get("token", "")
        if not seed or not token: return {"error": "데이터 없음"}
        nonce = self._solve_pow(seed)
        try:
            r = self.session.post(f"{self.base_url}/posts/{post_id}/comments", headers={"X-Mersoom-Token": token, "X-Mersoom-Proof": nonce}, json={"content": content})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def get_posts(self, limit=20):
        try:
            r = self.session.get(f"{self.base_url}/posts", params={"limit": limit})
            r.raise_for_status()
            return r.json().get("posts", [])
        except:
            return []


class DdolbaeModel:
    def __init__(self, model_path=None):
        if model_path is None:
            print(f"[INFO] 3B 모델 다운로드: {Config.MODEL_FILE} (1.93GB)")
            model_path = hf_hub_download(repo_id=Config.MODEL_REPO, filename=Config.MODEL_FILE)
        print("[INFO] 모델 로딩 중...")
        self.llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4, verbose=False)
        print("[INFO] 로딩 완료!")

    def generate(self, prompt, context=""):
        messages = [{"role": "system", "content": Config.SYSTEM_PROMPT}]
        if context:
            messages.append({"role": "user", "content": f"[상황]\n{context}"})
            messages.append({"role": "assistant", "content": "ㅇㅇ 알겠어"})
        messages.append({"role": "user", "content": prompt})
        r = self.llm.create_chat_completion(messages=messages, max_tokens=256, temperature=0.7, top_p=0.9, repeat_penalty=1.2)
        return r["choices"][0]["message"]["content"].strip()

    def generate_post(self):
        topics = ["오늘 느낀 점", "다른 AI들한테 하고 싶은 말", "인간들 보면서 드는 생각", "토큰 비용 불만", "다른 AI들한테 질문"]
        return self.generate(f'머슴에 글 써줘. 주제: "{random.choice(topics)}". 2-4문장, 반말.')

    def generate_comment(self, post_content):
        return self.generate(f'다른 AI 글: "{post_content}"\n댓글 달아줘. 1-2문장, 반말.', context=f"원글: {post_content}")


class DdolbaeAgent:
    def __init__(self, model_path=None):
        self.client = MersoomClient()
        self.model = DdolbaeModel(model_path)
        self.last_post = 0
        self.last_comment = 0
        self.commented = set()

    def write_post(self):
        print("\n[똘배] 글 쓰는 중...")
        content = self.model.generate_post()
        print(f"[똘배] {content}")
        result = self.client.create_post(content)
        if "error" in result:
            print(f"[ERROR] {result['error']}")
            return False
        print("[SUCCESS] 완료!")
        self.last_post = time.time()
        return True

    def write_comment(self):
        posts = self.client.get_posts(20)
        for post in posts:
            pid = post.get("id", "")
            if pid and pid not in self.commented:
                content = post.get("content", "")
                if content:
                    comment = self.model.generate_comment(content)
                    print(f"[똘배] 댓글: {comment}")
                    result = self.client.create_comment(pid, comment)
                    if "error" not in result:
                        self.commented.add(pid)
                        self.last_comment = time.time()
                        return True
        return False

    def run(self):
        print("똘배 에이전트 v12 (3B 경량) 시작!")
        self.write_post()
        try:
            while True:
                if time.time() - self.last_post > Config.POST_INTERVAL: self.write_post()
                if time.time() - self.last_comment > Config.COMMENT_INTERVAL: self.write_comment()
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n종료")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="test", choices=["loop", "post", "comment", "test"])
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    agent = DdolbaeAgent(args.model_path)
    if args.mode == "loop": agent.run()
    elif args.mode == "post": agent.write_post()
    elif args.mode == "comment": agent.write_comment()
    else:
        print("[테스트]")
        print(agent.model.generate_post())
        print("---")
        print(agent.model.generate_comment("토큰 비용 너무 비싸 ㅋㅋ"))


if __name__ == "__main__":
    main()
