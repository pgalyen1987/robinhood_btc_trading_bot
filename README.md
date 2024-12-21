# Robinhood Bitcoin Trading Bot

A sophisticated cryptocurrency trading bot that uses machine learning to optimize trading strategies on Robinhood's crypto platform. The bot downloads historical BTC data from Yahoo Finance, performs technical analysis, and executes trades through the Robinhood API.

## Features

- Historical data analysis using Yahoo Finance
- Technical indicators (RSI, MACD)
- Machine learning-based strategy optimization
- Backtesting with visualization
- Real-time trading dashboard
- Performance analytics
- Automated trading execution

## Requirements

- Python 3.8+
- TA-Lib
- Robinhood API credentials

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/robinhood_btc_trading_bot.git
cd robinhood_btc_trading_bot
```

2. Install TA-Lib system dependencies:
```bash
# For Debian/Ubuntu
sudo apt-get install ta-lib

# For macOS
brew install ta-lib

# For Windows
# Download and install from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your Robinhood API credentials:
```bash
ROBINHOOD_API_KEY=your_api_key
ROBINHOOD_PRIVATE_KEY=your_private_key
ROBINHOOD_PUBLIC_KEY=your_public_key
```

## Usage

The bot can be run in different modes:

1. Backtesting:
```bash
./start.py --mode backtest --days 365 --interval 1h
```

2. Strategy Optimization:
```bash
./start.py --mode optimize --days 365 --interval 1h
```

3. Live Trading:
```bash
./start.py --mode live
```

4. Dashboard:
```bash
./start.py --mode dashboard
```

### Command Line Arguments

- `--mode`: Operating mode (backtest, optimize, live, dashboard)
- `--days`: Days of historical data to use (default: 730)
- `--interval`: Data interval (1m, 5m, 15m, 30m, 1h, 1d)
- `--symbol`: Trading symbol (default: BTC-USD)
- `--debug`: Enable debug logging

## Project Structure

```
robinhood_btc_trading_bot/
├── src/
│   └── trading_bot/
│       ├── api/            # Robinhood API integration
│       ├── backtesting/    # Backtesting engine
│       ├── data/           # Data management
│       ├── ml/             # Machine learning models
│       ├── optimization/   # Strategy optimization
│       ├── trading/        # Trading strategies
│       ├── utils/          # Utilities
│       └── visualization/  # Data visualization
├── data/                   # Historical data
├── logs/                   # Application logs
├── models/                 # Trained ML models
├── plots/                  # Generated plots
├── start.py               # Main entry point
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Configuration

The bot's behavior can be configured through environment variables in the `.env` file:

```bash
# API credentials
ROBINHOOD_API_KEY=your_api_key
ROBINHOOD_PRIVATE_KEY=your_private_key
ROBINHOOD_PUBLIC_KEY=your_public_key

# Trading parameters
INVESTMENT_AMOUNT=10000
TARGET_PROFIT=0.005
COMMISSION_RATE=0.001

# Optimization settings
OPTIMIZATION_DAYS=30
OPTIMIZATION_INTERVAL=24

# Logging
LOG_LEVEL=INFO
```

## Disclaimer

This bot is for educational purposes only. Cryptocurrency trading carries significant risks. Always do your own research and never trade with money you can't afford to lose.