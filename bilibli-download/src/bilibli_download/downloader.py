from __future__ import annotations

import argparse
import os
import subprocess
from typing import List

from playwright.sync_api import sync_playwright

DEFAULT_URL = "https://space.bilibili.com/1497330079/lists/725831?type=season"
URL_ENV_VAR = "BILIBILI_URL"
DEFAULT_BBDOWN_COMMAND = "BBDown.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Bilibili season video links and download them with BBDown."
    )
    parser.add_argument(
        "url",
        nargs="?",
        help=f"Bilibili season/list URL. Defaults to ${URL_ENV_VAR} or the built-in URL.",
    )
    parser.add_argument(
        "--url",
        dest="url_option",
        help="Bilibili season/list URL. Overrides the positional URL and environment variable.",
    )
    return parser.parse_args()


def resolve_url(url: str | None) -> str:
    return url or os.getenv(URL_ENV_VAR) or DEFAULT_URL


def collect_video_links(url: str = DEFAULT_URL, wait_ms: int = 5000, headless: bool = False) -> List[str]:
    """Collect unique Bilibili video links from a season list page."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()

        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(wait_ms)

        links = page.eval_on_selector_all(
            'a[href*="/video/BV"]',
            "els => els.map(e => e.href)",
        )

        browser.close()

    # Keep insertion order while removing duplicates.
    return list(dict.fromkeys(links))


def download_videos(links: List[str], bbdown_command: str = DEFAULT_BBDOWN_COMMAND) -> None:
    """Download all given links using BBDown."""
    for link in links:
        print(f"开始下载: {link}")
        subprocess.run([bbdown_command, link], check=False)


def main() -> None:
    args = parse_args()
    url = resolve_url(args.url_option or args.url)
    bbdown_command = os.getenv("BBDOWN_COMMAND", DEFAULT_BBDOWN_COMMAND)
    links = collect_video_links(url)
    download_videos(links, bbdown_command=bbdown_command)


if __name__ == "__main__":
    main()
