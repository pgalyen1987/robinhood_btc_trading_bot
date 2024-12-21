#!/bin/bash

echo "Starting Trading Bot System..."
echo
echo "Loading environment variables..."

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    source .env
fi

echo "Starting application..."
python3 ./start.py "$@"  # Changed from main.py to start.py