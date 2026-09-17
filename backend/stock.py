from __future__ import annotations

import datetime

from pydantic import BaseModel, Field, model_validator


class StockRecordSchema(BaseModel):

    ticker: str = Field(..., min_length=1, description="Exchange ticker symbol, e.g. 'AAPL'.")
    date: datetime.date
    open: float = Field(..., gt=0, description="Opening price. Must be > 0.")
    high: float = Field(..., gt=0, description="Highest traded price. Must be > 0.")
    low: float = Field(..., gt=0, description="Lowest traded price. Must be > 0.")
    close: float = Field(..., gt=0, description="Closing price. Must be > 0.")
    volume: int = Field(..., ge=0, description="Shares traded. Must be >= 0.")

    @model_validator(mode="after")
    def validate_price_relationships(self) -> "StockRecordSchema":
        errors: list[str] = []

        if self.high < self.low:
            errors.append(f"high ({self.high}) cannot be less than low ({self.low}).")
        if self.high < self.open:
            errors.append(f"high ({self.high}) cannot be less than open ({self.open}).")
        if self.high < self.close:
            errors.append(f"high ({self.high}) cannot be less than close ({self.close}).")
        if self.low > self.open:
            errors.append(f"low ({self.low}) cannot be greater than open ({self.open}).")
        if self.low > self.close:
            errors.append(f"low ({self.low}) cannot be greater than close ({self.close}).")

        if errors:
            raise ValueError(
                f"Corrupt OHLC tick for {self.ticker} on {self.date}: " + "; ".join(errors)
            )

        return self