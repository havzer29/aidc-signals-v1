"""Utility helpers for the AIDC signals pipeline."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

LOG_FORMAT = "%(message)s"


def setup_json_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Configure a JSON structured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    handler: logging.Handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_json(logger: logging.Logger, **payload: Any) -> None:
    """Emit a JSON log entry."""
    logger.info(json.dumps(payload, ensure_ascii=False))


@dataclass
class RateLimiter:
    """Simple rate limiter supporting minimum delay between calls."""

    min_delay: float
    last_call: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delta = now - self.last_call
        if delta < self.min_delay:
            time.sleep(self.min_delay - delta)
        self.last_call = time.monotonic()


@dataclass
class Backoff:
    base_delay: float = 0.8
    max_delay: float = 10.0
    factor: float = 2.0

    def compute(self, attempt: int) -> float:
        jitter = random.uniform(0.8, 1.2)
        delay = min(self.max_delay, self.base_delay * (self.factor ** attempt))
        return delay * jitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
OUT_DIR = PROJECT_ROOT / "out"
CACHE_DIR = PROJECT_ROOT / "cache"


def ensure_directories() -> None:
    for directory in (OUT_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()


def load_companies() -> pd.DataFrame:
    companies_path = CONFIG_DIR / "companies.csv"
    df = pd.read_csv(companies_path)
    df["aliases"] = df["aliases"].fillna("")
    return df


def load_keywords() -> List[str]:
    path = CONFIG_DIR / "keywords.txt"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fp:
        return [line.strip() for line in fp if line.strip()]


def load_rss_feeds() -> List[str]:
    path = CONFIG_DIR / "rss_feeds.txt"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fp:
        return [line.strip() for line in fp if line.strip() and not line.startswith("#")]


def load_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def chunked(iterable: Iterable[Any], size: int) -> Iterable[List[Any]]:
    chunk: List[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def read_domain_tiers() -> Dict[str, Dict[str, Any]]:
    path = CONFIG_DIR / "domain_tiers.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def alias_matches(text: str, aliases: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(alias.lower() in lowered for alias in aliases if alias)


__all__ = [
    "setup_json_logger",
    "log_json",
    "RateLimiter",
    "Backoff",
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "OUT_DIR",
    "CACHE_DIR",
    "load_companies",
    "load_keywords",
    "load_rss_feeds",
    "load_json",
    "save_jsonl",
    "env_int",
    "env_float",
    "chunked",
    "read_domain_tiers",
    "alias_matches",
]
