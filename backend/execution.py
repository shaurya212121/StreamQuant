import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

load_dotenv()
logger = logging.getLogger("uvicorn.error")

class AlpacaExecutionEngine:
    def __init__(self):
        # 1. Credential Resolution with Fallback Chains
        self.api_key = os.getenv("APES_API_KEY") or os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("APES_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
        
        # 2. FRM Guardrail: Strictly enforce paper trading mode
        self.paper = True 

        # 3. Defensive Initialization Check
        if not self.api_key or not self.api_secret:
            logger.warning("Alpaca API credentials missing. Running in mock/offline mode.")
            self.client = None
        else:
            try:
                self.client = TradingClient(self.api_key, self.api_secret, paper=self.paper)
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca Trading Client: {e}")
                self.client = None

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
                "buying_power": 0.0,
                "currency": "USD"
            }
     try:
            account = self.client.get_account()
            return {
                "status": account.status,
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "currency": account.currency
            }
     except Exception as e: #if any network or API error occurs 
            logger.error(f"Error fetching Alpaca account summary: {e}")
            return {"status": "error", "message": str(e)}

    def get_recent_orders(self, limit: int = 5) -> list:
        """
        Fetches recent filled or open orders from the broker to provide audit trails.
        """
        # Guard 1: If client is offline, return an empty list gracefully
        if not self.client:
            return []
            
        try:
            request_params = GetOrdersRequest(
                status=QueryOrderStatus.ALL,  # Fetch all statuses (filled, cancelled, open)
                limit=limit                   # Restrict how many records to pull back (default 5)
            )
            
            # 2. Query the Alpaca API
            orders = self.client.get_orders(filter=request_params)
            
            return [
                {
                    "symbol": o.symbol,
                    "qty": float(o.qty) if o.qty else 0.0,
                    "side": o.side.value,
                    "status": o.status.value,
                    "filled_at": str(o.filled_at) if o.filled_at else "Pending"
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Error fetching recent orders: {e}")
            return []

    def execute_trade_signal(self, symbol: str, qty: float, signal: str) -> Dict[str, Any]:
        """
        Executes a paper market order based on model signal confluence.
        FRM Firewall: Rejects any trade unless signal is strictly 'STRONG BUY' and client is active.
        """
        # Guard 1: Ensure client is initialized (credentials present)
        if not self.client:
            return {"success": False, "error": "Alpaca client not initialized. Check API keys."}

        # Guard 2: Programmatic Risk Firewall against weak signals
        if signal.upper() != "STRONG BUY":
            logger.warning(f"Trade rejected for {symbol}: Signal '{signal}' did not meet institutional threshold.")
            return {"success": False, "error": f"Signal '{signal}' rejected. Execution requires STRONG BUY."}

        try:
            # 1. Construct immutable market order request payload using Alpaca SDK
            market_order_data = MarketOrderRequest(
                symbol=symbol.upper(),
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            
            # 2. Submit order to Alpaca Paper API gateway
            order = self.client.submit_order(order_data=market_order_data)
            logger.info(f"SUCCESS: Executed paper order for {symbol}. Order ID: {order.id}")
            
            return {
                "success": True,
                "order_id": str(order.id),
                "symbol": order.symbol,
                "qty": str(order.qty),
                "status": str(order.status)
            }
        except Exception as e:
            # Catch exchange rejection errors (e.g. insufficient buying power, market closed)
            logger.error(f"Execution failure for {symbol}: {e}")
            return {"success": False, "error": str(e)}

# Global Singleton Instance for clean import across FastAPI routes
alpaca_executor = AlpacaExecutionEngine()
