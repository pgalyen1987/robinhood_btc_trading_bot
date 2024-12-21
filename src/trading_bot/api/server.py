"""FastAPI server for trading bot backend."""

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

from ..data.data_manager import DataManager
from ..optimization.optimizer import MLOptimizer
from ..trading.strategy import TradingStrategy
from ..utils.logger import logger

app = FastAPI(title="Trading Bot API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
data_manager = DataManager()
active_connections: List[WebSocket] = []

@app.get("/")
async def root():
    """Root endpoint."""
    return {"status": "ok", "service": "Trading Bot API"}

@app.get("/data/historical")
async def get_historical_data(
    symbol: str = "BTC-USD",
    interval: str = "1h",
    days: int = 30
) -> Dict[str, Any]:
    """Get historical market data."""
    try:
        data = data_manager.get_data(
            days_back=days,
            interval=interval,
            use_cache=True
        )
        return {
            "status": "success",
            "data": data.to_dict(orient="records"),
            "metadata": {
                "symbol": symbol,
                "interval": interval,
                "start_date": data.index[0].isoformat(),
                "end_date": data.index[-1].isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/strategy/optimize")
async def optimize_strategy(
    symbol: str = "BTC-USD",
    days: int = 30,
    config_path: str = "config.json"
) -> Dict[str, Any]:
    """Run strategy optimization."""
    try:
        # Load configuration
        with open(config_path) as f:
            config = json.load(f)
            
        # Get data
        data = data_manager.get_data(days_back=days)
        
        # Initialize optimizer
        optimizer = MLOptimizer(
            data=data,
            strategy_class=TradingStrategy
        )
        
        # Run optimization
        result = optimizer.optimize(
            param_ranges=config["optimization"]["parameter_ranges"]
        )
        
        return {
            "status": "success",
            "optimization_result": {
                "parameters": result.parameters,
                "metrics": result.metrics,
                "score": result.score,
                "feature_importance": result.feature_importance,
                "timestamp": result.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Process incoming messages if needed
            await broadcast_update({"type": "acknowledgment", "data": data})
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_connections.remove(websocket)

async def broadcast_update(message: Dict[str, Any]):
    """Broadcast update to all connected clients."""
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server."""
    import uvicorn
    logger.info(f"Starting API server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server() 