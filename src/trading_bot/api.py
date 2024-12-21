import base64
import datetime
import json
import uuid
import logging
import requests
import time
from typing import Any, Dict, Optional, List
from nacl.signing import SigningKey
from requests import Session
import hashlib

from .utils.logger import logger
from .utils.rate_limiter import RateLimit
from .config import (
    API_KEY,
    PUBLIC_KEY,
    PRIVATE_KEY,
    INVESTMENT_AMOUNT,
    TARGET_PROFIT,
    SYMBOL,
    BASE_URL
)

# Constants for API rate limiting and timeouts
DEFAULT_RATE_LIMIT = 30  # requests per minute
REQUEST_TIMEOUT = 10  # seconds
RETRY_DELAY = 60  # seconds between retries
ORDER_CHECK_INTERVAL = 6  # seconds

class CryptoAPITrading:
    """Cryptocurrency trading API interface."""
    
    def __init__(self, 
                 api_key: str = API_KEY,
                 api_secret: str = PRIVATE_KEY,
                 base_url: str = BASE_URL,
                 rate_limit: int = DEFAULT_RATE_LIMIT):
        """Initialize API trading interface.
        
        Args:
            api_key: API key
            api_secret: API secret
            base_url: Base API URL
            rate_limit: Maximum requests per minute
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip('/')  # Remove trailing slash
        self.session = Session()
        self.rate_limiter = RateLimit(rate_limit)
        
        # Initialize signing key
        try:
            # Ensure private key is exactly 32 bytes using SHA256
            key_bytes = hashlib.sha256(self.api_secret.encode()).digest()[:32]
            self.private_key = SigningKey(key_bytes)
            logger.debug("Private key initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize private key: {e}")
            raise

    @staticmethod
    def _get_current_timestamp() -> int:
        return int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp())

    @staticmethod
    def get_query_params(key: str, *args: Optional[str]) -> str:
        if not args:
            return ""

        params = []
        for arg in args:
            params.append(f"{key}={arg}")

        return "?" + "&".join(params)

    def make_api_request(self, method: str, path: str, body: str = "") -> Any:
        """Make API request with proper error handling."""
        timestamp = self._get_current_timestamp()
        headers = self.get_authorization_header(method, path, body, timestamp)
        url = f"{self.base_url}{path}"

        try:
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            elif method == "POST":
                if body:
                    response = requests.post(url, headers=headers, json=json.loads(body), timeout=REQUEST_TIMEOUT)
                else:
                    response = requests.post(url, headers=headers, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 401:
                logger.error("Authentication failed. Please check your API credentials.")
                logger.error(f"Response: {response.text}")
                return None
            elif response.status_code == 404:
                logger.error(f"Endpoint not found: {url}")
                logger.error(f"Response: {response.text}")
                return None
            
            response.raise_for_status()
            
            return response.json() if response.content else None

        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            if hasattr(e, 'response'):
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
                logger.error(f"Request URL: {url}")
                logger.error(f"Request headers: {headers}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {str(e)}")
            return None

    def get_authorization_header(
            self, method: str, path: str, body: str, timestamp: int
    ) -> Dict[str, str]:
        body = body if body else ""
        message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body}"
        logger.debug(f"Message to sign: {message_to_sign}")
        signed = self.private_key.sign(message_to_sign.encode("utf-8"))
        
        headers = {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
            "x-timestamp": str(timestamp),
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        logger.debug(f"Generated headers: {headers}")
        return headers

    def get_account(self) -> Any:
        """Get account information."""
        path = "/v1/crypto/trading/accounts/"  # Updated endpoint path
        return self.make_api_request("GET", path)

    # The symbols argument must be formatted in trading pairs, e.g "BTC-USD", "ETH-USD". If no symbols are provided,
    # all supported symbols will be returned
    def get_trading_pairs(self, *symbols: Optional[str]) -> Any:
        """Get trading pairs information."""
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/v1/crypto/trading/trading_pairs/{query_params}"  # Updated endpoint path
        return self.make_api_request("GET", path)

    # The asset_codes argument must be formatted as the short form name for a crypto, e.g "BTC", "ETH". If no asset
    # codes are provided, all crypto holdings will be returned
    def get_holdings(self, *asset_codes: Optional[str]) -> Any:
        """Get holdings information."""
        query_params = self.get_query_params("asset_code", *asset_codes)
        path = f"/v1/crypto/trading/holdings/{query_params}"  # Updated endpoint path
        return self.make_api_request("GET", path)

    # The symbols argument must be formatted in trading pairs, e.g "BTC-USD", "ETH-USD". If no symbols are provided,
    # the best bid and ask for all supported symbols will be returned
    def get_best_bid_ask(self, *symbols: Optional[str]) -> Any:
        """Get best bid/ask prices."""
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/v1/crypto/marketdata/best_bid_ask/{query_params}"  # Updated endpoint path
        return self.make_api_request("GET", path)

    # The symbol argument must be formatted in a trading pair, e.g "BTC-USD", "ETH-USD"
    # The side argument must be "bid", "ask", or "both".
    # Multiple quantities can be specified in the quantity argument, e.g. "0.1,1,1.999".
    def get_estimated_price(self, symbol: str, side: str, quantity: str) -> Any:
        path = f"/api/v1/crypto/marketdata/estimated_price/?symbol={symbol}&side={side}&quantity={quantity}"
        return self.make_api_request("GET", path)

    def place_order(
            self,
            client_order_id: str,
            side: str,
            order_type: str,
            symbol: str,
            order_config: Dict[str, str],
    ) -> Any:
        body = {
            "client_order_id": client_order_id,
            "side": side,
            "type": order_type,
            "symbol": symbol,
            f"{order_type}_order_config": order_config,
        }
        path = "/api/v1/crypto/trading/orders/"
        return self.make_api_request("POST", path, json.dumps(body))

    def cancel_order(self, order_id: str) -> Any:
        path = f"/api/v1/crypto/trading/orders/{order_id}/cancel/"
        return self.make_api_request("POST", path, "")

    def get_order(self, order_id: str) -> Any:
        path = f"/api/v1/crypto/trading/orders/{order_id}/"
        return self.make_api_request("GET", path)

    def get_orders(self) -> Any:
        path = "/api/v1/crypto/trading/orders/"
        return self.make_api_request("GET", path)
    
    def get_current_btc_price(self) -> Any:
        bid_ask = self.get_best_bid_ask("BTC-USD")
        current_btc_price = bid_ask["results"][0]["price"]
        return round(float(current_btc_price), 2)
    
    def get_open_order_count(self) -> Any:
        orders = self.get_orders()
        count = 0
        for order in orders["results"]:
            if order["state"] == "open":
                count += 1
        return count
    
    def cancel_all_open_orders(self) -> Any:
        orders = self.get_orders()
        for order in orders["results"]:
            if order["state"] == "open":
                self.cancel_order(order["id"])
        return
    
    def get_available_btc(self) -> Any:
        holdings = self.get_holdings("BTC")
        if holdings["results"]:
            return float(holdings["results"][0]["quantity_available_for_trading"])
        return 0.0
    
    def wait_for_open_orders_to_execute(self) -> Any:
        print("Waiting for open orders to execute" + time.strftime("%H:%M:%S"))
        while self.get_open_order_count() > 0:
            time.sleep(ORDER_CHECK_INTERVAL)
        print("All open orders executed" + time.strftime("%H:%M:%S"))
        return
    
    def get_best_ask(self) -> Any:
        bid_ask = self.get_best_bid_ask("BTC-USD")
        return round(float(bid_ask["results"][0]["ask_inclusive_of_buy_spread"]), 2)

def main():
    """Run the trading bot."""
    try:
        # Check for required credentials
        if not all([API_KEY, PRIVATE_KEY, PUBLIC_KEY]):
            logger.error("API credentials not found in environment variables")
            return
            
        # Initialize API client with proper error handling
        try:
            api = CryptoAPITrading(
                api_key=API_KEY,
                api_secret=PRIVATE_KEY,
                base_url=BASE_URL
            )
            logger.info("API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize API client: {e}")
            return
        
        # Test API connection
        logger.info("Testing API connection...")
        account = api.get_account()
        if not account:
            logger.error("Failed to connect to API. Please check your credentials and network connection.")
            return
            
        logger.info("Account connected successfully")
        logger.info("Checking BTC holdings...")
        
        holdings = api.get_holdings("BTC")
        if holdings and holdings.get("results"):
            btc_holdings = holdings["results"][0]
            btc_in_usd = float(btc_holdings["quantity_available_for_trading"]) * api.get_current_btc_price()
            logger.info(f"Current BTC holdings: {btc_in_usd:.2f} USD")
        else:
            logger.info("No BTC holdings found")
            
        logger.info(f"Open orders: {api.get_open_order_count()}")
        
        # Start trading loop with improved error handling
        while True:
            try:
                # Wait for any open orders to execute
                logger.info("Checking open orders...")
                while api.get_open_order_count() > 0:     
                    time.sleep(ORDER_CHECK_INTERVAL)
                
                # Get current market prices with validation
                current_btc_price = api.get_best_ask()
                if current_btc_price is None:
                    logger.error("Failed to get current BTC price")
                    time.sleep(RETRY_DELAY)
                    continue
                    
                sell_price = round(current_btc_price * (1 + TARGET_PROFIT), 2)
                logger.info(f"Current BTC price: ${current_btc_price}")
                logger.info(f"Target sell price: ${sell_price}")
                
                available_btc = api.get_available_btc()
                logger.info(f"Available BTC: {available_btc}")
                
                # Place orders based on position with validation
                if available_btc == 0.0:
                    quantity = round(INVESTMENT_AMOUNT / current_btc_price, 8)
                    buy_order_id = str(uuid.uuid4())
                    buy_order = api.place_order(
                        buy_order_id,
                        "buy",
                        "market",
                        SYMBOL,
                        {"asset_quantity": str(quantity)}
                    )
                    if buy_order:
                        logger.info(f"Placed buy order: {buy_order}")
                    else:
                        logger.error("Failed to place buy order")
                        time.sleep(RETRY_DELAY)
                        continue
                else:
                    sell_order_id = str(uuid.uuid4())
                    sell_order = api.place_order(
                        sell_order_id,
                        "sell",
                        "limit",
                        SYMBOL,
                        {
                            "limit_price": str(sell_price),
                            "quote_amount": str(round(available_btc * current_btc_price, 2))
                        }
                    )
                    if sell_order:
                        logger.info(f"Placed sell order: {sell_order}")
                    else:
                        logger.error("Failed to place sell order")
                        time.sleep(RETRY_DELAY)
                        continue
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(RETRY_DELAY)
                
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return

if __name__ == "__main__":
    main()