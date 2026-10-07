from __future__ import annotations

import logging
import sys
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date as date_type
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote_plus

sys.path.append(str(Path(__file__).resolve().parent.parent))

import requests 
import yfinance as yf 

from db.models import upsert_news_sentiment 

logger = logging.getLogger("streamquant.sentiment")

FINBERT_MODEL_NAME = "ProsusAI/finbert"
MAX_HEADLINES = 30
BATCH_SIZE = 16
MAX_TOKENS = 128 
HTTP_TIMEOUT = 10

_model_lock = threading.Lock()
_model_bundle = None  

def _extract_title(item: dict) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    title = item.get("title")
    if not title and isinstance(item.get("content"), dict):
        title = item["content"].get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return None


def _fetch_yfinance_headlines(symbol: str) -> List[str]:
    try:
        items = yf.Ticker(symbol).news or []
    except Exception as exc:
        logger.warning("yfinance news failed for %s: %s", symbol, exc)
        return []
    return [t for t in (_extract_title(i) for i in items) if t]


def _fetch_google_news_headlines(symbol: str) -> List[str]:
    """Free, key-less fallback: Google News RSS search for the ticker."""
    url = (
        "https://news.google.com/rss/search?q="
        f"{quote_plus(symbol + ' stock')}&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        resp = requests.get(
            url, timeout=HTTP_TIMEOUT, headers={"User-Agent": "Mozilla/5.0 StreamQuant"}
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as exc:
        logger.warning("Google News RSS fallback failed for %s: %s", symbol, exc)
        return []

    titles = []
    for item in root.iter("item"):
        title = item.findtext("title")
        if title and title.strip():
            titles.append(title.strip())
    return titles


def fetch_headlines(symbol: str, limit: int = MAX_HEADLINES) -> List[str]:
    """Return up to `limit` unique, recent headlines for `symbol`.

    Uses yfinance first; falls back to Google News RSS only if yfinance is empty.
    """
    symbol = symbol.upper().strip()
    headlines = _fetch_yfinance_headlines(symbol)
    if not headlines:
        logger.info("No yfinance news for %s; using Google News RSS fallback.", symbol)
        headlines = _fetch_google_news_headlines(symbol)

    seen, unique = set(), []
    for h in headlines:
        key = h.lower()
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique[:limit]

def _load_finbert():
    """Lazy, thread-safe singleton so the model loads once per process."""
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle
    with _model_lock:
        if _model_bundle is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info("Loading %s ...", FINBERT_MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            label_index = {v.lower(): int(k) for k, v in model.config.id2label.items()}
            missing = {"positive", "negative", "neutral"} - set(label_index)
            if missing:
                raise RuntimeError(f"Unexpected FinBERT labels, missing: {missing}")
            _model_bundle = (tokenizer, model, device, label_index)
    return _model_bundle


@dataclass
class HeadlineSentiment:
    headline: str
    positive: float
    negative: float
    neutral: float

    @property
    def score(self) -> float:
        """Composite in [-1, +1]: P(positive) - P(negative)."""
        return self.positive - self.negative


@dataclass
class SentimentResult:
    symbol: str
    score: float = 0.0 
    positive: float = 0.0 
    negative: float = 0.0
    neutral: float = 0.0
    headline_count: int = 0
    details: List[HeadlineSentiment] = field(default_factory=list)


def analyze_headlines(headlines: List[str]) -> List[HeadlineSentiment]:
    """Run FinBERT over each headline and return per-headline probabilities."""
    if not headlines:
        return []

    import torch

    tokenizer, model, device, idx = _load_finbert()
    results: List[HeadlineSentiment] = []

    for start in range(0, len(headlines), BATCH_SIZE):
        batch = headlines[start : start + BATCH_SIZE]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().tolist()

        for text, p in zip(batch, probs):
            results.append(
                HeadlineSentiment(
                    headline=text,
                    positive=p[idx["positive"]],
                    negative=p[idx["negative"]],
                    neutral=p[idx["neutral"]],
                )
            )
    return results


def score_headlines(symbol: str, headlines: List[str]) -> SentimentResult:
    """Aggregate per-headline FinBERT output into one composite score for the symbol."""
    details = analyze_headlines(headlines)
    if not details:
        return SentimentResult(symbol=symbol.upper())

    n = len(details)
    return SentimentResult(
        symbol=symbol.upper(),
        score=sum(d.score for d in details) / n,
        positive=sum(d.positive for d in details) / n,
        negative=sum(d.negative for d in details) / n,
        neutral=sum(d.neutral for d in details) / n,
        headline_count=n,
        details=details,
    )

def analyze_and_store(
    symbol: str, run_date: Optional[date_type] = None
) -> Optional[SentimentResult]:

    symbol = symbol.upper().strip()
    run_date = run_date or date_type.today()

    headlines = fetch_headlines(symbol)
    if not headlines:
        logger.warning("No headlines found for %s; skipping store.", symbol)
        return None

    result = score_headlines(symbol, headlines)
    upsert_news_sentiment(
        ticker=symbol,
        sentiment_date=run_date,
        sentiment_score=result.score,
        positive_prob=result.positive,
        negative_prob=result.negative,
        neutral_prob=result.neutral,
        headline_count=result.headline_count,
    )
    logger.info(
        "%s %s: score=%+.3f from %d headlines", symbol, run_date, result.score, result.headline_count
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from db.database import engine
    from db.models import Base

    Base.metadata.create_all(bind=engine)

    for sym in sys.argv[1:] or ["AAPL"]:
        res = analyze_and_store(sym)
        if res:
            print(f"{res.symbol}: {res.score:+.3f} ({res.headline_count} headlines)")