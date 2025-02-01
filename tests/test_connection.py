import os
import sys
import logging

# Import these after setting up sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import MomentumTradingBot, API_KEY, API_SECRET

# Set up logging to both file and console
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def test_api_connection():
    """Test the API connection to Coinbase."""
    try:
        # Initialize the bot
        logger.info("Initializing bot...")
        bot = MomentumTradingBot(API_KEY, API_SECRET)
        
        # Test public API first
        logger.info("Testing public API...")
        # ticker = bot.exchange.fetch_ticker('BTC-USD')
        # logger.info(f"BTC-USD price: ${ticker['last']:.2f}")
        
        # Test private API
        logger.info("Testing private API (account balance)...")
        balance = bot.exchange.fetch_balance()
        
        logger.info("✅ Connection successful!")
        
        usd_balance = balance.get("USD", {}).get("free", 0)
        logger.info(f"Available USD balance: ${usd_balance:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Connection failed: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error details: {str(e)}")
        raise


if __name__ == "__main__":
    test_api_connection() 