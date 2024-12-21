class DataDownloader:
    def __init__(self, 
                 symbol: str,
                 api_key: str = None,
                 timeframe: str = "1d",
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None):
        """Initialize data downloader.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USD')
            api_key: API key for data provider
            timeframe: Data timeframe (e.g., '1d', '1h')
            start_date: Start date for historical data
            end_date: End date for historical data
        """
        self.symbol = symbol
        self.api_key = api_key
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        # ... rest of initialization code ... 