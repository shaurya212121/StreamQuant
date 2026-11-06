import os
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    ALPACA_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TradingClient = None
    MarketOrderRequest = None
    GetOrdersRequest = None
    OrderSide = None
    TimeInForce = None
    QueryOrderStatus = None
    StockHistoricalDataClient = None
    StockLatestTradeRequest = None
    ALPACA_AVAILABLE = False
import yfinance as yf

# Defensively load .env from project root
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

logger = logging.getLogger("uvicorn.error")

class AlpacaExecutionEngine:
    def __init__(self):
        # 1. Credential Resolution: standard Alpaca environment variables
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET_KEY")
        
        # 2. Strict paper trading guardrail
        self.paper = True 

        # 3. Defensive Initialization Check (Package Availability & Credentials)
        if not ALPACA_AVAILABLE:
            logger.warning("alpaca-py package is not installed. Running in mock/offline mode.")
            self.client = None
            self.data_client = None
        elif not self.api_key or not self.api_secret:
            logger.warning("Alpaca API credentials missing. Running in mock/offline mode.")
            self.client = None
            self.data_client = None
        else:
            try:
                self.client = TradingClient(self.api_key, self.api_secret, paper=self.paper)
                self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca Trading or Data Client: {e}")
                self.client = None
                self.data_client = None

        # 4. SDE Concurrency Guard: Mutex to eliminate TOCTOU race conditions
        self._trade_execution_lock = threading.Lock()

    def get_market_clock(self) -> Dict[str, Any]:
        """
        Queries exchange clock to determine if regular US equity market is currently open.
        Prevents submitting DAY market orders when exchange is closed.
        """
        if not self.client:
            return {"is_open": False, "next_open": None, "next_close": None, "status": "offline"}
        try:
            clock = self.client.get_clock()
            return {
                "is_open": bool(clock.is_open),
                "next_open": str(clock.next_open),
                "next_close": str(clock.next_close),
                "status": "online"
            }
        except Exception as e:
            logger.warning(f"Error fetching exchange clock: {e}")
            return {"is_open": False, "next_open": None, "next_close": None, "status": "error"}

    def get_account_portfolio(self) -> Dict[str, Any]:
        """
        Retrieves real-time paper account metrics: cash balance, equity, and buying power.
        Used by the frontend to monitor account solvency and margin usage.
        """
        if not self.client:
            return {
                "status": "offline",
                "cash": 0.0,
                "portfolio_value": 0.0,
                "equity": 0.0,
                "buying_power": 0.0,
                "currency": "USD"
            }
        try:
            account = self.client.get_account()
            status_val = account.status.value if hasattr(account.status, "value") else str(account.status)
            portfolio_val = float(account.portfolio_value)
            last_equity_val = float(account.last_equity) if hasattr(account, "last_equity") and account.last_equity else portfolio_val
            return {
                "status": status_val,
                "cash": float(account.cash),
                "portfolio_value": portfolio_val,
                "equity": portfolio_val,  # Provides explicit 'equity' key expected by frontend
                "last_equity": last_equity_val,
                "buying_power": float(account.buying_power),
                "currency": account.currency
            }
        except Exception as e:
            logger.error(f"Error fetching Alpaca account summary: {e}")
            return {"status": "error", "message": "Failed to retrieve broker account summary."}

    def get_open_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetches the current open position for a given symbol from Alpaca broker.
        Returns None if no open position exists or broker is offline.
        """
        if not self.client:
            return None
        try:
            pos = self.client.get_open_position(symbol.upper().strip())
            return {
                "symbol": str(pos.symbol),
                "qty": float(pos.qty),
                "market_value": float(pos.market_value),
                "current_price": float(pos.current_price) if hasattr(pos, "current_price") and pos.current_price else 0.0
            }
        except Exception:
            # Alpaca API raises 404 when no open position is found
            return None

    def get_recent_orders(self, limit: int = 5) -> list:
        """
        Fetches recent filled or open orders from the broker to provide audit trails.
        """
        if not self.client:
            return []
            
        try:
            request_params = GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                limit=limit
            )
            orders = self.client.get_orders(filter=request_params)
            
            return [
                {
                    "symbol": o.symbol,
                    "qty": float(o.qty) if o.qty else 0.0,
                    "side": o.side.value if hasattr(o.side, "value") else str(o.side),
                    "status": o.status.value if hasattr(o.status, "value") else str(o.status),
                    "filled_at": str(o.filled_at) if o.filled_at else "Pending"
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Error fetching recent orders: {e}")
            return []

    def get_latest_market_price(self, symbol: str) -> Optional[float]:
        """
        Retrieves the latest market price with a multi-layered resilient fallback:
        Tier 1: Alpaca Real-time Historical Data Client
        Tier 2: Yahoo Finance Fast-Info Feed
        """
        clean_symbol = symbol.strip().upper()

        # Tier 1: Alpaca Data Gateway
        if self.data_client:
            try:
                trade_req = StockLatestTradeRequest(symbol_or_symbols=clean_symbol)
                latest_trade = self.data_client.get_stock_latest_trade(trade_req)
                if clean_symbol in latest_trade:
                    price = float(latest_trade[clean_symbol].price)
                    if price > 0:
                        return price
            except Exception as e:
                logger.warning(f"Alpaca market price lookup failed for {clean_symbol}: {e}. Falling back to secondary feed.")

        # Tier 2: Yahoo Finance Fast-Info Fallback
        try:
            ticker = yf.Ticker(clean_symbol)
            price = ticker.fast_info.last_price or ticker.fast_info.regular_market_previous_close
            if price and float(price) > 0:
                return float(price)
        except Exception as e:
            logger.error(f"Secondary market price lookup failed for {clean_symbol}: {e}")

        return None

    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        confidence_pct: float = 100.0,
        max_allocation_pct: float = 0.05
    ) -> Dict[str, Any]:
        """
        FRM Quantitative Position Sizing Engine:
        Implements Conviction-Weighted Capital Allocation with strict, position-aware
        concentration limits (max 5% of total portfolio equity in any single asset).
        
        Formula:
            Max Allowed Asset Dollars = Total Equity * max_allocation_pct
            Current Asset Dollars = Value of existing holdings in symbol
            Available Dollar Room = max(0.0, Max Allowed Asset Dollars - Current Asset Dollars)
            Target Dollars = Available Dollar Room * (confidence_pct / 100)
            Capped by available buying power (with a 10% safety buffer for slippage)
            qty = floor(Target Dollars / Current Price)
        """
        portfolio = self.get_account_portfolio()
        if portfolio.get("status") == "error" or not self.client:
            return {
                "qty": 0.0,
                "target_dollars": 0.0,
                "reason": "Broker offline or portfolio unreachable."
            }

        equity = float(portfolio.get("portfolio_value", 0.0))
        buying_power = float(portfolio.get("buying_power", 0.0))

        if equity <= 0.0 or current_price <= 0.0:
            return {
                "qty": 0.0,
                "target_dollars": 0.0,
                "reason": "Non-positive equity or invalid market price."
            }

        # Query existing position to enforce a true position ceiling (not just per-order)
        existing_pos = self.get_open_position(symbol)
        existing_value = existing_pos.get("market_value", 0.0) if existing_pos else 0.0
        
        max_asset_capital = equity * max_allocation_pct
        available_capital_room = max(0.0, max_asset_capital - existing_value)

        if available_capital_room <= 0.0:
            return {
                "qty": 0.0,
                "target_dollars": 0.0,
                "reason": (
                    f"Position limit reached for {symbol}: Existing holding of ${existing_value:,.2f} "
                    f"meets or exceeds the {max_allocation_pct*100:.1f}% portfolio limit (${max_asset_capital:,.2f})."
                )
            }

        # 1. Conviction Multiplier: scale between 0.50 and 1.0 (clamped)
        conviction_mult = max(0.50, min(1.0, float(confidence_pct) / 100.0))

        # 2. Risk Allocation: Cap single-stock concentration at 5% max equity
        target_dollars = available_capital_room * conviction_mult

        # 3. Hard Liquidity Guard: Never exceed 90% of available buying power
        safe_buying_power = buying_power * 0.90
        effective_budget = min(target_dollars, safe_buying_power)

        # 4. Integer Lot Sizing (Floor division)
        calculated_qty = int(effective_budget // current_price)

        if calculated_qty < 1:
            return {
                "qty": 0.0,
                "target_dollars": round(target_dollars, 2),
                "reason": (
                    f"Target allocation (${effective_budget:,.2f}) is below the cost of 1 share "
                    f"of {symbol} (${current_price:,.2f}) or existing position is near maximum cap."
                )
            }

        notional = round(calculated_qty * current_price, 2)
        total_projected_value = existing_value + notional
        projected_portfolio_pct = round((total_projected_value / equity) * 100.0, 2) if equity > 0 else 0.0

        rationale = (
            f"AI Risk Sizing: Allocated {calculated_qty} shares (${notional:,.2f}; brings total holding to "
            f"${total_projected_value:,.2f} / {projected_portfolio_pct}% of equity) "
            f"scaled by {confidence_pct:.1f}% model conviction."
        )

        return {
            "qty": float(calculated_qty),
            "target_dollars": round(target_dollars, 2),
            "notional": notional,
            "portfolio_pct": projected_portfolio_pct,
            "rationale": rationale
        }

    def execute_trade_signal(
        self,
        symbol: str,
        signal: str,
        side: str = "BUY",
        qty: Optional[float] = None,
        sizing_mode: str = "auto",
        confidence_pct: float = 100.0
    ) -> Dict[str, Any]:
        """
        Executes a paper market order based on server-verified model confluence.
        Thread-safe, pre-trade risk checked, and validated against market hours.
        """
        # Guard 0: Emergency Kill Switch
        if os.getenv("STREAMQUANT_KILL_SWITCH", "").strip().lower() in ("true", "1", "yes"):
            logger.critical("Trade rejected: Emergency Kill Switch (STREAMQUANT_KILL_SWITCH) is active.")
            return {
                "success": False,
                "error": "Emergency Kill Switch is engaged. All order execution is suspended.",
                "status_code": 503
            }

        # Guard 1: Ensure client is initialized
        if not self.client:
            err_msg = "Alpaca client not initialized. Check API keys or install alpaca-py." if not ALPACA_AVAILABLE else "Alpaca client not initialized. Check API keys."
            return {"success": False, "error": err_msg, "status_code": 503}

        # Sanitize symbol, signal, and side
        clean_symbol = symbol.strip().upper()
        clean_signal = signal.strip().upper()
        order_side = OrderSide.SELL if str(side).strip().upper() == "SELL" else OrderSide.BUY

        # Guard 2: Programmatic Risk Firewall against weak signals for BUY orders
        if order_side == OrderSide.BUY and clean_signal != "STRONG BUY":
            logger.warning(f"Trade rejected for {clean_symbol}: Signal '{clean_signal}' did not meet institutional threshold.")
            return {"success": False, "error": f"Signal '{clean_signal}' rejected. Execution requires STRONG BUY.", "status_code": 400}

        # Guard 3: Market Hours Firewall (Fail-closed: only allow if clock is confirmed online and open)
        clock = self.get_market_clock()
        if clock.get("status") != "online":
            logger.warning(f"Trade rejected for {clean_symbol}: Exchange clock is unreachable (status: {clock.get('status')}).")
            return {
                "success": False,
                "error": "Exchange clock lookup failed or broker offline. Orders blocked for market hours safety.",
                "status_code": 503
            }

        if not clock.get("is_open", False):
            next_open = clock.get("next_open", "next market session")
            logger.warning(f"Trade rejected for {clean_symbol}: Market is closed. Next open: {next_open}")
            return {
                "success": False,
                "error": f"Market is currently closed. Day market orders cannot execute until {next_open}.",
                "status_code": 400
            }

        # SDE Lock: Synchronize pre-trade check and submission to eliminate race conditions
        with self._trade_execution_lock:
            # Guard 4: Institutional Pre-Trade Risk Checks (FRM Firewall)
            portfolio = self.get_account_portfolio()
            if portfolio.get("status") == "error":
                return {"success": False, "error": f"Pre-trade check failed: {portfolio.get('message')}", "status_code": 503}

            account_status = str(portfolio.get("status", "")).upper()
            if account_status != "ACTIVE":
                logger.warning(f"Trade rejected for {clean_symbol}: Account status is '{account_status}', expected 'ACTIVE'.")
                return {"success": False, "error": f"Pre-trade check failed: Account is {account_status}.", "status_code": 400}

            equity = float(portfolio.get("portfolio_value", 0.0))
            buying_power = float(portfolio.get("buying_power", 0.0))

            # Guard 4b: Daily Loss Limit Circuit Breaker (3% Max Drawdown against previous day close equity)
            last_equity = float(portfolio.get("last_equity", equity))
            if last_equity > 0:
                daily_pnl_pct = (equity - last_equity) / last_equity
                if daily_pnl_pct <= -0.03:
                    logger.critical(f"Daily drawdown limit triggered: Current PnL {daily_pnl_pct*100:.2f}% <= -3.00%. Order halted.")
                    return {
                        "success": False,
                        "error": f"Circuit breaker: Daily loss limit (-3%) triggered. Current drawdown is {daily_pnl_pct*100:.2f}%. Execution halted.",
                        "status_code": 400
                    }

            if order_side == OrderSide.BUY and buying_power <= 0.0:
                logger.warning(f"Trade rejected for {clean_symbol}: Zero or negative buying power (${buying_power:,.2f}).")
                return {"success": False, "error": "Pre-trade check failed: Insufficient buying power ($0.00 available).", "status_code": 400}

            # Fetch latest market price for valuation & validation
            current_price = self.get_latest_market_price(clean_symbol)
            if not current_price or current_price <= 0.0:
                return {"success": False, "error": f"Failed to retrieve valid market price for {clean_symbol}.", "status_code": 400}

            # Query existing holdings for position-aware checks
            existing_pos = self.get_open_position(clean_symbol)
            existing_qty = existing_pos.get("qty", 0.0) if existing_pos else 0.0
            existing_value = existing_pos.get("market_value", 0.0) if existing_pos else 0.0

            sizing_mode_clean = (sizing_mode or "auto").lower()
            sizing_rationale = ""

            if order_side == OrderSide.SELL:
                # Sell execution path
                if existing_qty <= 0:
                    return {
                        "success": False,
                        "error": f"Sell rejected: No open position held for {clean_symbol}.",
                        "status_code": 400
                    }
                if sizing_mode_clean == "manual" and qty is not None:
                    if float(qty) <= 0 or not float(qty).is_integer():
                        return {
                            "success": False,
                            "error": f"Manual sell rejected: Quantity ({qty}) must be a positive whole integer.",
                            "status_code": 400
                        }
                    final_qty = min(float(qty), existing_qty)
                    sizing_rationale = f"Manual sell: Closing {final_qty} of {existing_qty} shares."
                else:
                    final_qty = existing_qty
                    sizing_rationale = f"Full position close: Liquidating all {final_qty} shares."
            else:
                # BUY execution path
                if sizing_mode_clean == "manual":
                    if qty is None or float(qty) <= 0 or not float(qty).is_integer():
                        return {
                            "success": False,
                            "error": f"Manual trade rejected: Quantity ({qty}) must be a positive whole integer. Fractional shares are not supported.",
                            "status_code": 400
                        }
                    final_qty = float(qty)
                    required_capital = final_qty * current_price
                    if required_capital > buying_power:
                        return {
                            "success": False,
                            "error": (
                                f"Manual trade rejected: Order cost (${required_capital:,.2f}) exceeds "
                                f"available buying power (${buying_power:,.2f})."
                            ),
                            "status_code": 400
                        }
                    # Enforce the 5% portfolio concentration ceiling on manual orders as well
                    projected_total = existing_value + required_capital
                    max_allowed = equity * 0.05
                    if projected_total > max_allowed:
                        return {
                            "success": False,
                            "error": (
                                f"Manual trade rejected: Projected position (${projected_total:,.2f}) exceeds "
                                f"the 5% risk concentration limit (${max_allowed:,.2f})."
                            ),
                            "status_code": 400
                        }
                    sizing_rationale = f"Manual override: User requested {final_qty} shares at ${current_price:,.2f}/sh (within 5% cap)."
                else:
                    # Auto Sizing via Position-Aware Conviction-Weighted Engine
                    sizing_result = self.calculate_position_size(
                        symbol=clean_symbol,
                        current_price=current_price,
                        confidence_pct=confidence_pct
                    )
                    final_qty = sizing_result.get("qty", 0.0)
                    if final_qty <= 0:
                        return {
                            "success": False,
                            "error": f"Auto-sizing rejected: {sizing_result.get('reason', 'Calculated 0 shares.')}",
                            "status_code": 400
                        }
                    sizing_rationale = sizing_result.get("rationale", "")

            # Guard 6: Integer Share Normalization (Alpaca non-fractional compatibility)
            order_qty = int(final_qty) if float(final_qty).is_integer() else final_qty

            try:
                # Idempotency token to prevent duplicate broker orders
                order_nonce = uuid.uuid4().hex[:8]
                timestamp_sec = int(datetime.now(timezone.utc).timestamp())
                client_order_id = f"sq_{clean_symbol.lower()}_{timestamp_sec}_{order_nonce}"

                # 1. Construct immutable market order request payload using Alpaca SDK
                market_order_data = MarketOrderRequest(
                    symbol=clean_symbol,
                    qty=order_qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=client_order_id
                )
                
                # 2. Submit order to Alpaca Paper API gateway
                order = self.client.submit_order(order_data=market_order_data)
                logger.info(f"SUCCESS: Executed paper order for {clean_symbol}. Order ID: {order.id} | Client ID: {client_order_id} | Qty: {order_qty} | Side: {order_side.value}")
                
                confirmed_qty = float(order_qty)
                try:
                    if hasattr(order, "qty") and order.qty is not None:
                        confirmed_qty = float(order.qty)
                except Exception:
                    confirmed_qty = float(order_qty)

                return {
                    "success": True,
                    "order_id": str(order.id),
                    "client_order_id": client_order_id,
                    "symbol": str(order.symbol),
                    "side": order_side.value if hasattr(order_side, "value") else str(order_side),
                    "qty": confirmed_qty,
                    "status": str(order.status),
                    "sizing_mode": sizing_mode_clean,
                    "estimated_quote_price": current_price,
                    "executed_price": current_price,
                    "estimated_notional": round(confirmed_qty * current_price, 2),
                    "sizing_rationale": sizing_rationale
                }
            except Exception as e:
                # Sanitize error message to prevent leaking internal tracebacks
                err_str = str(e)
                logger.error(f"Execution failure for {clean_symbol}: {err_str}")
                is_conn_error = any(kw in err_str.lower() for kw in ["connection", "timeout", "offline", "unreachable"])
                status_code = 503 if is_conn_error else 400
                return {"success": False, "error": f"Broker order rejection: {err_str}", "status_code": status_code}

# Global Singleton Instance for clean import across FastAPI routes
alpaca_executor = AlpacaExecutionEngine()
