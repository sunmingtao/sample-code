# bilibli-download

`bilibli-download` is a small Python project that:

1. opens a Bilibili season page,
2. collects all unique video links (`/video/BV...`),
3. downloads each video using `BBDown`.

## Project structure

```text
bilibli-download/
├── pyproject.toml
├── README.md
└── src/
    └── bilibli_download/
        ├── __init__.py
        ├── __main__.py
        └── downloader.py
```

## Requirements

- Python 3.10+
- `BBDown` installed and available in your PATH (or set `BBDOWN_COMMAND`)

> On Windows, default command is `BBDown.exe`.

## Setup in a virtual environment

From inside the `bilibli-download` folder:

### 1) Create and activate venv

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux (bash/zsh):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install project dependencies

```bash
pip install --upgrade pip
pip install -e .
```

### 3) Install Playwright browser

```bash
playwright install chromium
```

## Run

You can run either command:

```bash
bilibli-download
```

or:

```bash
python -m bilibli_download
```

## Optional configuration

- `BILIBILI_URL`: override the Bilibili season/list URL.
- `BBDOWN_COMMAND`: override executable used for downloading.

Examples:

```bash
bilibli-download "https://space.bilibili.com/example/lists/example?type=season"
bilibli-download --url "https://space.bilibili.com/example/lists/example?type=season"
BILIBILI_URL="https://space.bilibili.com/example/lists/example?type=season" bilibli-download
BBDOWN_COMMAND=BBDown bilibli-download
```

## Notes

- By default, Chromium is launched with `headless=False`, so a browser window appears.
