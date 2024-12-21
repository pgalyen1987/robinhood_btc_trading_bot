// API configuration
export const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
export const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';

// Chart configuration
export const CHART_CONFIG = {
  candlestick: {
    upColor: '#26a69a',
    downColor: '#ef5350',
    borderUpColor: '#26a69a',
    borderDownColor: '#ef5350',
    wickUpColor: '#26a69a',
    wickDownColor: '#ef5350',
  },
  volume: {
    upColor: 'rgba(38, 166, 154, 0.3)',
    downColor: 'rgba(239, 83, 80, 0.3)',
  },
};

// Trading configuration
export const TRADING_CONFIG = {
  updateInterval: 5000, // 5 seconds
  maxDataPoints: 1000, // Maximum number of candlesticks to display
  defaultTimeframe: '1h',
  defaultSymbol: 'BTC-USD',
};

// UI configuration
export const UI_CONFIG = {
  theme: {
    primary: '#2196f3',
    secondary: '#f50057',
    success: '#4caf50',
    error: '#f44336',
    warning: '#ff9800',
    info: '#2196f3',
  },
  toast: {
    duration: 5000,
    position: 'top-right',
  },
}; 