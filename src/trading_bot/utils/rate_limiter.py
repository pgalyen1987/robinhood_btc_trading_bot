"""Rate limiter implementation for API requests."""

import time
from typing import Optional
from datetime import datetime, timedelta
from collections import deque

class RateLimit:
    """Rate limiter for API requests."""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in time window
            time_window: Time window in seconds (default: 60)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        
    def wait(self) -> None:
        """Wait if necessary to respect rate limit."""
        now = datetime.now()
        
        # Remove old requests outside the time window
        while self.requests and (now - self.requests[0]).total_seconds() > self.time_window:
            self.requests.popleft()
            
        # If at rate limit, wait until oldest request expires
        if len(self.requests) >= self.max_requests:
            wait_time = self.time_window - (now - self.requests[0]).total_seconds()
            if wait_time > 0:
                time.sleep(wait_time)
                
        # Add current request
        self.requests.append(now)
        
    def remaining(self) -> int:
        """Get remaining requests in current time window."""
        now = datetime.now()
        
        # Remove old requests
        while self.requests and (now - self.requests[0]).total_seconds() > self.time_window:
            self.requests.popleft()
            
        return self.max_requests - len(self.requests) 