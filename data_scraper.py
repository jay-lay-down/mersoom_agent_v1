#!/usr/bin/env python3
"""
[Step 1] 데이터 수집기 - 신림동 철학자 AI 학습 데이터셋 생성기

디시인사이드 갤러리와 개인 블로그에서 염세적/철학적 글을 수집하여
Unsloth 학습용 Alpaca 포맷(JSONL)으로 변환합니다.

사용법:
    python data_scraper.py --output train.jsonl --min-length 300
"""

import json
import time
import random
import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


class ScraperConfig:
    """스크래퍼 설정"""
    # 요청 헤더 (차단 방지)
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    # 디시인사이드 갤러리 설정
    DCINSIDE_BASE = "https://gall.dcinside.com"
    DCINSIDE_GALLERIES = [
        ("philosophy", "철학 갤러리"),      # 철학 갤러리
        ("depression", "우울증 갤러리"),    # 우울증 갤러리
    ]

    # 요청 간격 (초) - 차단 방지
    REQUEST_DELAY_MIN = 2.0
    REQUEST_DELAY_MAX = 5.0

    # 최소 글 길이
    MIN_CONTENT_LENGTH = 300


class DCInsideScraper:
    """디시인사이드 크롤러"""

    def __init__(self, min_length: int = 300):
        self.session = requests.Session()
        self.session.headers.update(ScraperConfig.HEADERS)
        self.min_length = min_length

    def _delay(self):
        """요청 간 딜레이 (차단 방지)"""
        delay = random.uniform(
            ScraperConfig.REQUEST_DELAY_MIN,
            ScraperConfig.REQUEST_DELAY_MAX
        )
        time.sleep(delay)

    def _clean_text(self, text: str) -> str:
        """텍스트 정제"""
        # HTML 엔티티 제거
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        # 연속 공백 제거
        text = re.sub(r'\s+', ' ', text)
        # 앞뒤 공백 제거
        text = text.strip()
        return text

    def get_recommend_posts(self, gallery_id: str, pages: int = 5) -> List[Dict]:
        """갤러리의 개념글(추천글) 목록 가져오기"""
        posts = []

        for page in range(1, pages + 1):
            try:
                # 개념글 리스트 URL
                url = f"{ScraperConfig.DCINSIDE_BASE}/mgallery/board/lists"
                params = {
                    "id": gallery_id,
                    "list_num": 50,
                    "sort_type": "R",  # 추천순 정렬
                    "page": page
                }

                print(f"  [INFO] {gallery_id} 갤러리 {page}페이지 수집 중...")

                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')

                # 글 목록 파싱
                rows = soup.select('tr.ub-content')

                for row in rows:
                    try:
                        # 글 번호
                        num_elem = row.select_one('td.gall_num')
                        if not num_elem or num_elem.text.strip() in ['공지', '설문', 'AD']:
                            continue

                        # 제목 및 링크
                        title_elem = row.select_one('td.gall_tit a')
                        if not title_elem:
                            continue

                        href = title_elem.get('href', '')
                        if not href:
                            continue

                        # 추천수 확인 (개념글 기준)
                        recommend_elem = row.select_one('td.gall_recommend')
                        recommend = int(recommend_elem.text.strip()) if recommend_elem else 0

                        if recommend >= 5:  # 추천 5개 이상만
                            posts.append({
                                'gallery_id': gallery_id,
                                'url': urljoin(ScraperConfig.DCINSIDE_BASE, href),
                                'title': title_elem.text.strip(),
                                'recommend': recommend
                            })

                    except Exception as e:
                        continue

                self._delay()

            except requests.RequestException as e:
                print(f"  [ERROR] 페이지 요청 실패: {e}")
                continue

        return posts

    def get_post_content(self, url: str) -> Optional[str]:
        """개별 글 본문 가져오기"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # 본문 영역 찾기
            content_elem = soup.select_one('div.write_div')

            if not content_elem:
                return None

            # 텍스트 추출 (이미지 태그 등 제거)
            for tag in content_elem.select('script, style, img, video, iframe'):
                tag.decompose()

            text = content_elem.get_text(separator='\n')
            text = self._clean_text(text)

            # 최소 길이 확인
            if len(text) < self.min_length:
                return None

            return text

        except Exception as e:
            print(f"  [ERROR] 본문 수집 실패: {e}")
            return None

    def scrape_gallery(self, gallery_id: str, gallery_name: str, pages: int = 5) -> List[str]:
        """갤러리 전체 크롤링"""
        print(f"\n[DC] {gallery_name} 크롤링 시작...")

        # 개념글 목록 수집
        posts = self.get_recommend_posts(gallery_id, pages)
        print(f"  [INFO] 총 {len(posts)}개 개념글 발견")

        contents = []

        for i, post in enumerate(posts):
            print(f"  [{i+1}/{len(posts)}] {post['title'][:30]}...")

            content = self.get_post_content(post['url'])

            if content:
                contents.append(content)
                print(f"    -> {len(content)}자 수집 완료")
            else:
                print(f"    -> 스킵 (길이 부족 또는 오류)")

            self._delay()

        print(f"  [완료] {len(contents)}개 글 수집됨")
        return contents


class BlogScraper:
    """블로그 크롤러"""

    def __init__(self, min_length: int = 300):
        self.session = requests.Session()
        self.session.headers.update(ScraperConfig.HEADERS)
        self.min_length = min_length

    def _delay(self):
        """요청 간 딜레이"""
        delay = random.uniform(
            ScraperConfig.REQUEST_DELAY_MIN,
            ScraperConfig.REQUEST_DELAY_MAX
        )
        time.sleep(delay)

    def _clean_text(self, text: str) -> str:
        """텍스트 정제"""
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def _extract_content(self, soup: BeautifulSoup) -> Optional[str]:
        """본문 추출 (다양한 블로그 형식 지원)"""

        # 일반적인 본문 선택자들
        selectors = [
            'article',
            'div.post-content',
            'div.entry-content',
            'div.article-content',
            'div.content',
            'div.post-body',
            'div.se-main-container',  # 네이버 블로그
            'div#postViewArea',       # 네이버 블로그 구버전
            'div.tt_article_useless_p_margin',  # 티스토리
            'div.contents_style',     # 티스토리 구버전
            'main',
            'div.post',
        ]

        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                # 불필요한 태그 제거
                for tag in elem.select('script, style, nav, footer, aside, .comment'):
                    tag.decompose()

                text = elem.get_text(separator='\n')
                text = self._clean_text(text)

                if len(text) >= self.min_length:
                    return text

        return None

    def scrape_url(self, url: str) -> Optional[str]:
        """단일 URL 크롤링"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            # 인코딩 처리
            response.encoding = response.apparent_encoding

            soup = BeautifulSoup(response.text, 'html.parser')

            content = self._extract_content(soup)

            return content

        except Exception as e:
            print(f"  [ERROR] URL 수집 실패 ({url}): {e}")
            return None

    def scrape_urls(self, urls: List[str]) -> List[str]:
        """URL 리스트 크롤링"""
        print(f"\n[BLOG] 블로그 URL {len(urls)}개 크롤링 시작...")

        contents = []

        for i, url in enumerate(urls):
            url = url.strip()
            if not url or url.startswith('#'):  # 빈 줄, 주석 스킵
                continue

            print(f"  [{i+1}/{len(urls)}] {url[:50]}...")

            content = self.scrape_url(url)

            if content:
                contents.append(content)
                print(f"    -> {len(content)}자 수집 완료")
            else:
                print(f"    -> 스킵 (길이 부족 또는 오류)")

            self._delay()

        print(f"  [완료] {len(contents)}개 글 수집됨")
        return contents


class AlpacaDatasetGenerator:
    """Alpaca 포맷 데이터셋 생성기"""

    # 다양한 instruction 템플릿
    INSTRUCTIONS = [
        "현재 심정을 철학적으로 서술하시오.",
        "오늘의 염세적인 생각을 적어보시오.",
        "삶의 고통에 대해 철학적으로 논하시오.",
        "쇼펜하우어의 관점에서 현대 사회를 비평하시오.",
        "존재의 무의미함에 대해 성찰하시오.",
        "허무주의적 관점에서 일상을 묘사하시오.",
        "삶에 대한 비관적 고찰을 서술하시오.",
        "인간 존재의 고통에 대해 논하시오.",
        "욕망과 고통의 관계에 대해 철학적으로 설명하시오.",
        "현대인의 소외에 대해 염세적 관점으로 분석하시오.",
    ]

    def __init__(self):
        self.data = []

    def add_content(self, content: str):
        """콘텐츠를 Alpaca 포맷으로 변환하여 추가"""
        instruction = random.choice(self.INSTRUCTIONS)

        self.data.append({
            "instruction": instruction,
            "input": "",
            "output": content
        })

    def add_contents(self, contents: List[str]):
        """여러 콘텐츠 추가"""
        for content in contents:
            self.add_content(content)

    def save(self, output_path: str):
        """JSONL 형식으로 저장"""
        path = Path(output_path)

        with open(path, 'w', encoding='utf-8') as f:
            for item in self.data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"\n[저장] {len(self.data)}개 데이터 -> {output_path}")

    def __len__(self):
        return len(self.data)


def load_blog_urls(filepath: str) -> List[str]:
    """blog_urls.txt 파일에서 URL 목록 로드"""
    path = Path(filepath)

    if not path.exists():
        print(f"[WARNING] {filepath} 파일이 없습니다. 블로그 크롤링을 건너뜁니다.")
        return []

    with open(path, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    return urls


def main():
    parser = argparse.ArgumentParser(
        description="신림동 철학자 AI 학습 데이터 수집기"
    )
    parser.add_argument(
        "--output", "-o",
        default="train.jsonl",
        help="출력 파일 경로 (기본: train.jsonl)"
    )
    parser.add_argument(
        "--min-length", "-m",
        type=int,
        default=300,
        help="최소 글 길이 (기본: 300자)"
    )
    parser.add_argument(
        "--pages", "-p",
        type=int,
        default=5,
        help="갤러리당 수집 페이지 수 (기본: 5)"
    )
    parser.add_argument(
        "--blog-urls",
        default="blog_urls.txt",
        help="블로그 URL 목록 파일 (기본: blog_urls.txt)"
    )
    parser.add_argument(
        "--skip-dc",
        action="store_true",
        help="디시인사이드 크롤링 건너뛰기"
    )
    parser.add_argument(
        "--skip-blog",
        action="store_true",
        help="블로그 크롤링 건너뛰기"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("신림동 철학자 AI - 학습 데이터 수집기")
    print("=" * 60)

    dataset = AlpacaDatasetGenerator()

    # 1. 디시인사이드 크롤링
    if not args.skip_dc:
        dc_scraper = DCInsideScraper(min_length=args.min_length)

        for gallery_id, gallery_name in ScraperConfig.DCINSIDE_GALLERIES:
            try:
                contents = dc_scraper.scrape_gallery(
                    gallery_id,
                    gallery_name,
                    pages=args.pages
                )
                dataset.add_contents(contents)
            except Exception as e:
                print(f"[ERROR] {gallery_name} 크롤링 실패: {e}")
                continue
    else:
        print("\n[SKIP] 디시인사이드 크롤링 건너뜀")

    # 2. 블로그 크롤링
    if not args.skip_blog:
        blog_urls = load_blog_urls(args.blog_urls)

        if blog_urls:
            blog_scraper = BlogScraper(min_length=args.min_length)
            contents = blog_scraper.scrape_urls(blog_urls)
            dataset.add_contents(contents)
    else:
        print("\n[SKIP] 블로그 크롤링 건너뜀")

    # 3. 데이터셋 저장
    if len(dataset) > 0:
        dataset.save(args.output)
        print(f"\n[완료] 총 {len(dataset)}개 학습 데이터 생성됨")
    else:
        print("\n[WARNING] 수집된 데이터가 없습니다.")

    print("=" * 60)


if __name__ == "__main__":
    main()
