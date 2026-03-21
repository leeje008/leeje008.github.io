"""
GitHub 레포지토리 → 깃블로그 프로젝트 포스트 자동 동기화 스크립트

사용법:
  python scripts/sync_projects.py                    # 전체 스캔 (새 레포 감지 + 기존 업데이트)
  python scripts/sync_projects.py --repo stock-agent  # 특정 레포만 업데이트

필요 환경변수:
  GITHUB_TOKEN: GitHub Personal Access Token
  ANTHROPIC_API_KEY: Claude API Key (선택, 없으면 템플릿 기반 생성)
"""

import os
import sys
import json
import re
import base64
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

GITHUB_USER = "leeje008"
POSTS_DIR = Path(__file__).parent.parent / "_posts"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# 블로그 포스트에서 제외할 레포
EXCLUDE_REPOS = {"leeje008.github.io", "job-tracker"}

# 레포 → 포스트 파일 매핑 (기존 수동 작성 포스트)
REPO_POST_MAP = {
    "stock-agent": "stock-agent",
    "insurance-qa-agent": "insurance-qa-agent",
    "naver-blog-auto": "naver-blog-auto",
    "travel-planner": "travel-planner",
    "NN_SDR": "nn-sdr",
}


def github_api(endpoint: str) -> dict | list:
    url = f"https://api.github.com{endpoint}"
    req = Request(url)
    if GITHUB_TOKEN:
        req.add_header("Authorization", f"token {GITHUB_TOKEN}")
    req.add_header("Accept", "application/vnd.github.v3+json")
    with urlopen(req) as resp:
        return json.loads(resp.read().decode())


def get_public_repos() -> list[dict]:
    repos = github_api(f"/users/{GITHUB_USER}/repos?per_page=100&type=owner")
    return [
        r for r in repos
        if not r["private"]
        and not r["fork"]
        and r["name"] not in EXCLUDE_REPOS
    ]


def get_readme(repo_name: str) -> str:
    try:
        data = github_api(f"/repos/{GITHUB_USER}/{repo_name}/readme")
        content = base64.b64decode(data["content"]).decode("utf-8")
        return content
    except HTTPError:
        return ""


def get_existing_project_posts() -> dict[str, Path]:
    """기존 프로젝트 포스트를 {slug: path} 형태로 반환"""
    posts = {}
    for f in POSTS_DIR.glob("*.md"):
        content = f.read_text(encoding="utf-8")
        if "categories: [Project]" in content:
            slug = f.stem.split("-", 3)[-1] if len(f.stem.split("-")) >= 4 else f.stem
            posts[slug] = f
    return posts


def generate_post_with_claude(repo: dict, readme: str) -> str:
    """Claude API로 블로그 포스트 생성"""
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    topics = ", ".join(repo.get("topics", []))
    prompt = f"""아래 GitHub 레포지토리 정보를 기반으로 Jekyll 블로그 포스트를 작성해주세요.

레포지토리: {repo['name']}
설명: {repo.get('description', 'N/A')}
언어: {repo.get('language', 'N/A')}
Topics: {topics}

README 내용:
{readme[:8000]}

아래 형식으로 작성하세요. frontmatter 포함, 마크다운 형식.
- categories는 반드시 [Project]
- tags는 레포의 topics 기반으로 설정
- 프로젝트 개요, 핵심 기능, 기술 스택을 포함
- GitHub 링크 포함
- 한국어로 작성
- frontmatter의 title은 "[Project] 프로젝트명 - 한줄 설명" 형식

---
layout: post
title: "[Project] ..."
categories: [Project]
tags: [...]
math: false
---

(본문)
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def generate_post_template(repo: dict, readme: str) -> str:
    """Claude API 없이 템플릿 기반 포스트 생성"""
    name = repo["name"]
    desc = repo.get("description", f"{name} 프로젝트")
    lang = repo.get("language", "Python")
    topics = repo.get("topics", [])
    tags_str = ", ".join(topics) if topics else lang.lower()

    # README에서 주요 섹션 추출
    sections = {"features": "", "tech_stack": "", "structure": ""}

    if readme:
        # 주요 기능 섹션 추출
        feature_match = re.search(
            r"##\s*주요\s*기능(.*?)(?=##|\Z)", readme, re.DOTALL
        )
        if feature_match:
            sections["features"] = feature_match.group(1).strip()[:2000]

        # 기술 스택 섹션 추출
        tech_match = re.search(
            r"##\s*기술\s*스택(.*?)(?=##|\Z)", readme, re.DOTALL
        )
        if tech_match:
            sections["tech_stack"] = tech_match.group(1).strip()[:1500]

    post = f"""---
layout: post
title: "[Project] {desc}"
categories: [Project]
tags: [{tags_str}]
math: false
---

## 프로젝트 개요

{desc}

> GitHub: [leeje008/{name}](https://github.com/leeje008/{name})

---
"""

    if sections["features"]:
        post += f"""
## 주요 기능

{sections['features']}

---
"""

    if sections["tech_stack"]:
        post += f"""
## 기술 스택

{sections['tech_stack']}
"""

    return post


def find_existing_post(repo_name: str, existing_posts: dict[str, Path]) -> Path | None:
    """레포에 해당하는 기존 포스트 파일 찾기"""
    # 직접 매핑 확인
    slug = REPO_POST_MAP.get(repo_name, repo_name.lower())
    for key, path in existing_posts.items():
        if slug in key.lower() or repo_name.lower() in key.lower():
            return path
    return None


def sync_repo(repo: dict, existing_posts: dict[str, Path], force: bool = False) -> str:
    """단일 레포를 블로그에 동기화. 반환값: 'created', 'updated', 'skipped'"""
    name = repo["name"]
    readme = get_readme(name)

    existing_post = find_existing_post(name, existing_posts)

    if existing_post and not force:
        # 기존 포스트가 있으면 README 기반으로 갱신
        if not readme:
            return "skipped"

        if ANTHROPIC_API_KEY:
            new_content = generate_post_with_claude(repo, readme)
        else:
            new_content = generate_post_template(repo, readme)

        # frontmatter가 포함되지 않은 경우 기존 것 유지
        if not new_content.startswith("---"):
            return "skipped"

        existing_post.write_text(new_content, encoding="utf-8")
        return "updated"

    elif not existing_post:
        # 새 포스트 생성
        today = datetime.now().strftime("%Y-%m-%d")
        slug = name.lower().replace("_", "-")
        filename = f"{today}-{slug}.md"
        filepath = POSTS_DIR / filename

        if ANTHROPIC_API_KEY:
            content = generate_post_with_claude(repo, readme)
        else:
            content = generate_post_template(repo, readme)

        if content.startswith("---"):
            filepath.write_text(content, encoding="utf-8")
            return "created"

    return "skipped"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GitHub → Blog 동기화")
    parser.add_argument("--repo", help="특정 레포만 업데이트")
    parser.add_argument("--force", action="store_true", help="기존 포스트 강제 갱신")
    parser.add_argument("--dry-run", action="store_true", help="실제 파일 변경 없이 확인만")
    args = parser.parse_args()

    print(f"[sync] Posts directory: {POSTS_DIR}")
    print(f"[sync] Claude API: {'available' if ANTHROPIC_API_KEY else 'not set (template mode)'}")

    existing_posts = get_existing_project_posts()
    print(f"[sync] Existing project posts: {len(existing_posts)}")

    repos = get_public_repos()
    print(f"[sync] Public repos (excl. blog/job-tracker): {len(repos)}")

    if args.repo:
        repos = [r for r in repos if r["name"] == args.repo]
        if not repos:
            print(f"[error] Repo '{args.repo}' not found")
            sys.exit(1)

    results = {"created": [], "updated": [], "skipped": []}

    for repo in repos:
        if args.dry_run:
            existing = find_existing_post(repo["name"], existing_posts)
            status = "would update" if existing else "would create"
            print(f"  [dry-run] {repo['name']}: {status}")
            continue

        status = sync_repo(repo, existing_posts, force=args.force)
        results[status].append(repo["name"])
        print(f"  [{status}] {repo['name']}")

    if not args.dry_run:
        print(f"\n[summary] Created: {len(results['created'])}, "
              f"Updated: {len(results['updated'])}, "
              f"Skipped: {len(results['skipped'])}")

    # GitHub Actions 출력
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            has_changes = bool(results["created"] or results["updated"])
            f.write(f"has_changes={str(has_changes).lower()}\n")
            f.write(f"created={','.join(results['created'])}\n")
            f.write(f"updated={','.join(results['updated'])}\n")


if __name__ == "__main__":
    main()
