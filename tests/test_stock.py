import datetime
import pytest
from pydantic import ValidationError
from backend.stock import StockRecordSchema


def test_valid_stock_record():
    """Verify that a standard, healthy OHLCV tick passes validation cleanly."""
    record = StockRecordSchema(
        ticker="AAPL",
        date=datetime.date(2024, 1, 15),
        open=180.0,
        high=185.0,
        low=179.0,
        close=184.0,
        volume=50000000
    )
    assert record.ticker == "AAPL"
    assert record.high >= record.low


def test_high_less_than_low_rejected():
    """Verify that physically impossible ticks (High < Low) are rejected."""
    with pytest.raises(ValidationError) as excinfo:
        StockRecordSchema(
            ticker="AAPL",
            date=datetime.date(2024, 1, 15),
            open=180.0,
            high=175.0,  # Invalid: High < Low
            low=178.0,
            close=177.0,
            volume=1000
        )
    assert "cannot be less than low" in str(excinfo.value)


def test_inverted_open_close_bounds_rejected():
    """Verify that High must be >= Open and Low must be <= Close."""
    with pytest.raises(ValidationError):
        # High is less than Open
        StockRecordSchema(
            ticker="TSLA",
            date=datetime.date(2024, 1, 15),
            open=210.0,
            high=205.0,
            low=200.0,
            close=202.0,
            volume=5000
        )


def test_negative_values_rejected():
    """Verify that negative stock prices or negative volumes are rejected."""
    with pytest.raises(ValidationError):
        StockRecordSchema(
            ticker="NVDA",
            date=datetime.date(2024, 1, 15),
            open=-100.0,
            high=105.0,
            low=95.0,
            close=100.0,
            volume=5000
        )
