import base64
import datetime
import json
import uuid
import logging
import requests
from typing import Any, Dict, Optional, List
from nacl.signing import SigningKey
import time

from utils.logger import logger
from rate_limiter import RateLimit
from config import API_KEY, PUBLIC_KEY, PRIVATE_KEY, INVESTMENT_AMOUNT, TARGET_PROFIT, SYMBOL, BASE_URL

class CryptoAPITrading:
    def __init__(self):
        self.api_key = API_KEY
        private_key_seed = base64.b64decode(PRIVATE_KEY)
        self.private_key = SigningKey(private_key_seed)
        self.base_url = BASE_URL

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
        timestamp = self._get_current_timestamp()
        headers = self.get_authorization_header(method, path, body, timestamp)
        url = self.base_url + path

        try:
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method == "POST":
                if body:
                    response = requests.post(url, headers=headers, json=json.loads(body), timeout=10)
                else:
                    response = requests.post(url, headers=headers, timeout=10)
            
            if response.status_code == 401:
                logger.error(f"Authentication failed. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
            
            response.raise_for_status()
            
            return response.json() if response.content else None

        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            if hasattr(e, 'response'):
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
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
        path = "/api/v1/crypto/trading/accounts/"
        return self.make_api_request("GET", path)

    # The symbols argument must be formatted in trading pairs, e.g "BTC-USD", "ETH-USD". If no symbols are provided,
    # all supported symbols will be returned
    def get_trading_pairs(self, *symbols: Optional[str]) -> Any:
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/api/v1/crypto/trading/trading_pairs/{query_params}"
        return self.make_api_request("GET", path)

    # The asset_codes argument must be formatted as the short form name for a crypto, e.g "BTC", "ETH". If no asset
    # codes are provided, all crypto holdings will be returned
    def get_holdings(self, *asset_codes: Optional[str]) -> Any:
        query_params = self.get_query_params("asset_code", *asset_codes)
        path = f"/api/v1/crypto/trading/holdings/{query_params}"
        return self.make_api_request("GET", path)

    # The symbols argument must be formatted in trading pairs, e.g "BTC-USD", "ETH-USD". If no symbols are provided,
    # the best bid and ask for all supported symbols will be returned
    def get_best_bid_ask(self, *symbols: Optional[str]) -> Any:
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/api/v1/crypto/marketdata/best_bid_ask/{query_params}"
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
            time.sleep(6)
        print("All open orders executed" + time.strftime("%H:%M:%S"))
        return
    
    def get_best_ask(self) -> Any:
        bid_ask = self.get_best_bid_ask("BTC-USD")
        return round(float(bid_ask["results"][0]["ask_inclusive_of_buy_spread"]), 2)

def main():
    if not API_KEY or not PRIVATE_KEY:
        logger.error("API credentials not found in environment variables")
        return
        
    api = CryptoAPITrading()
    print(api.get_account())
    print("--------------------------------")
    holdings = api.get_holdings("BTC")
    print(holdings)
    
    # Add check for empty results
    if holdings["results"]:
        btc_in_usd = float(holdings["results"][0]["quantity_available_for_trading"]) * api.get_current_btc_price()
        print(f"BTC in USD: {btc_in_usd}")
    else:
        print("No BTC holdings found")
    print("--------------------------------")
    print ("open order count: " + str(api.get_open_order_count()))

    #print ("best bid ask: " + str(api.get_best_bid_ask("BTC-USD")))
    #return
    while True:
        print("Waiting for open orders to execute")
        while api.get_open_order_count() > 0:     
            time.sleep(6)

        current_btc_price =  api.get_best_ask()
        sell_price = round(current_btc_price * 1.0005, 2)
        print(f"Current BTC buy price: {current_btc_price}")
        print(f"Sell price: ~{sell_price}")
        print(f"Available BTC: {api.get_available_btc()}")

        if api.get_available_btc() == 0.0:
            buy_order_id = str(uuid.uuid4())
            buy_order = api.place_order(buy_order_id, "buy", "market", "BTC-USD", {"asset_quantity": round(200000 / current_btc_price, 8)})
            print(buy_order)

        print("Waiting for open orders to execute")
        while api.get_open_order_count() > 0: 
            time.sleep(6)

        sell_order_id = str(uuid.uuid4())
        sell_order = api.place_order(sell_order_id, "sell", "limit", "BTC-USD", {"limit_price": sell_price, "quote_amount": "200500"})
        print(sell_order)     

if __name__ == "__main__":
    main()