from __future__ import annotations

import os
import subprocess
from typing import List

from playwright.sync_api import sync_playwright

URL = "https://space.bilibili.com/1497330079/lists/1240875?type=season"
DEFAULT_BBDOWN_COMMAND = "BBDown.exe"


def collect_video_links(url: str = URL, wait_ms: int = 5000, headless: bool = False) -> List[str]:
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
    bbdown_command = os.getenv("BBDOWN_COMMAND", DEFAULT_BBDOWN_COMMAND)
    links = collect_video_links()
    download_videos(links, bbdown_command=bbdown_command)


if __name__ == "__main__":
    main()
