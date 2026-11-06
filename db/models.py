from datetime import date as date_type
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, declarative_base

Base = declarative_base()

class StockPrice(Base):
    __tablename__ = "stock_prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)

    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)

    sma_ratio = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    rsi = Column(Float, nullable=True)
    daily_return = Column(Float, nullable=True)
    volume_z_score = Column(Float, nullable=True)

    rolling_sharpe = Column(Float)
    rolling_max_drawdown = Column(Float)
    atr_20 = Column(Float)
    
    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_ticker_daily_bar"),
    )


class StockNewsSentiment(Base):
    """Daily aggregated FinBERT news sentiment, one row per (ticker, date)."""

    __tablename__ = "stock_news_sentiment"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)

    # Composite score in [-1.0, +1.0]: mean over headlines of P(positive) - P(negative)
    sentiment_score = Column(Float, nullable=False)

    # Mean class probabilities across the day's headlines
    positive_prob = Column(Float, nullable=False)
    negative_prob = Column(Float, nullable=False)
    neutral_prob = Column(Float, nullable=False)

    headline_count = Column(Integer, nullable=False, default=0)

    # Provenance tracking: 'finbert' vs 'lexicon_fallback'
    method = Column(String(30), nullable=False, server_default="finbert", default="finbert")

    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_news_sentiment_ticker_date"),
    )


class OrderAudit(Base):
    """Compliance & Risk Audit Trail: links every executed order with its source model signal and features."""
    __tablename__ = "order_audit_trail"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_order_id = Column(String(64), unique=True, nullable=False, index=True)
    broker_order_id = Column(String(64), nullable=True, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    qty = Column(Float, nullable=False)
    executed_price = Column(Float, nullable=True)
    signal = Column(String(30), nullable=False)
    confidence_pct = Column(Float, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    sentiment_method = Column(String(30), nullable=True)
    status = Column(String(30), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


def upsert_news_sentiment(
    ticker: str,
    sentiment_date: date_type,
    sentiment_score: float,
    positive_prob: float,
    negative_prob: float,
    neutral_prob: float,
    headline_count: int,
    method: str = "finbert",
    session: Optional[Session] = None,
) -> None:
    """Insert or update the news sentiment row for (ticker, sentiment_date).

    Uses PostgreSQL ``INSERT ... ON CONFLICT (ticker, date) DO UPDATE`` so
    re-running for the same ticker and day overwrites the row instead of
    duplicating it.
    """
    values = {
        "ticker": ticker.upper(),
        "date": sentiment_date,
        "sentiment_score": float(sentiment_score),
        "positive_prob": float(positive_prob),
        "negative_prob": float(negative_prob),
        "neutral_prob": float(neutral_prob),
        "headline_count": int(headline_count),
        "method": str(method),
    }

    stmt = pg_insert(StockNewsSentiment).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker", "date"],
        set_={
            "sentiment_score": stmt.excluded.sentiment_score,
            "positive_prob": stmt.excluded.positive_prob,
            "negative_prob": stmt.excluded.negative_prob,
            "neutral_prob": stmt.excluded.neutral_prob,
            "headline_count": stmt.excluded.headline_count,
            "method": stmt.excluded.method,
            "updated_at": func.now(),
        },
    )

    owns_session = session is None
    if owns_session:
        # Imported lazily so importing db.models never touches the DB engine.
        from db.database import SessionLocal

        session = SessionLocal()

    try:
        session.execute(stmt)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        if owns_session:
            session.close()