from __future__ import annotations

import argparse
import subprocess
import sys


def _run_bootstrap() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip_review", "--auto"], check=True)
    subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Discord bot.")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Install/update dependencies and Playwright browsers before starting.",
    )
    args = parser.parse_args()

    if args.bootstrap:
        _run_bootstrap()
    from bot import main as bot_main

    bot_main()


if __name__ == "__main__":
    main()
