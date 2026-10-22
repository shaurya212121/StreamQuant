import numpy as np
import pandas as pd


def test_max_drawdown_calculation():
    """Verify that peak-to-trough equity drop is computed correctly."""
    # Equity curve starts at 100k, drops to 80k (20% drawdown), then recovers to 110k
    equity = pd.Series([100000.0, 90000.0, 80000.0, 95000.0, 110000.0])
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_drawdown = float(drawdown.min())

    assert np.isclose(max_drawdown, -0.20, atol=1e-4)


def test_sortino_ratio_downside_only():
    """Verify that downside deviation only penalizes negative returns below target."""
    returns = pd.Series([0.02, 0.03, -0.01, 0.04, -0.02, 0.01])
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()

    assert downside_std > 0
    assert len(downside_returns) == 2  # Only the negative returns
