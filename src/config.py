import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
API_KEY = os.getenv('COINBASE_API_KEY')
API_SECRET = os.getenv('COINBASE_API_SECRET')

# Validate API credentials are loaded
if not all([API_KEY, API_SECRET]):
    raise ValueError("Missing API credentials. Please check your .env file.")

# Trading parameters
TRADING_PAIR_LIMIT = 20  # Number of trading pairs to monitor
BUY_THRESHOLD = 2.0     # Minimum price increase percentage to trigger buy
STOP_LOSS = -1.0       # Percentage loss at which to sell
TAKE_PROFIT = 3.0      # Percentage gain at which to sell
CHECK_INTERVAL = 60    # Time in seconds between market checks 