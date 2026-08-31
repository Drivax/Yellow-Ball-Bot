"""
run_weekly_update.py
---------------------
Orchestrates the full weekly data refresh: re-downloads raw ATP match data,
then rebuilds the cleaned dataset and engineered feature matrix.

Intended to be triggered on a schedule (see .github/workflows/weekly-scrape.yml)
but can also be run manually.

Usage:
    python scraping/run_weekly_update.py
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

STEPS = [
    ("Downloading raw ATP match data", BASE_DIR / "scraping" / "scraper_atp.py"),
    ("Cleaning and merging match data", BASE_DIR / "preprocessing" / "cleaner.py"),
    ("Building engineered feature matrix", BASE_DIR / "preprocessing" / "feature_engineering.py"),
]


def main() -> int:
    for description, script in STEPS:
        logger.info("=== %s (%s) ===", description, script.name)
        result = subprocess.run([sys.executable, str(script)], cwd=BASE_DIR)
        if result.returncode != 0:
            logger.error("Step failed: %s (exit code %d)", script.name, result.returncode)
            return result.returncode
    logger.info("Weekly update completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
