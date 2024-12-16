import datetime
from collections import deque
from time import sleep
import numpy as np

class RateLimit:
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window  # in seconds
        self.calls = deque()

    def __call__(self):
        now = datetime.datetime.now(tz=datetime.timezone.utc).timestamp()
        
        # Remove calls older than our time window
        while self.calls and self.calls[0] <= now - self.time_window:
            self.calls.popleft()
        
        # If we haven't hit our limit, add the call
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return
        
        # We've hit our limit, wait until we can make another call
        sleep_time = self.calls[0] - (now - self.time_window)
        if sleep_time > 0:
            sleep(sleep_time)
        self.calls.popleft()
        self.calls.append(now) 