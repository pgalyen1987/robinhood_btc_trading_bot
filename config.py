import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

API_KEY = os.getenv('ROBINHOOD_API_KEY')
PRIVATE_KEY = os.getenv('ROBINHOOD_PRIVATE_KEY')
PUBLIC_KEY = os.getenv('ROBINHOOD_PUBLIC_KEY')
# Trading parameters
INVESTMENT_AMOUNT = 10.0
TARGET_PROFIT = 1.0
SYMBOL = "BTC-USD"

# API endpoints
BASE_URL = "https://trading.robinhood.com"