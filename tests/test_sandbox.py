# -*- coding: utf-8 -*-

import os
import sys
import ccxt
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_sandbox_connection():
    """Test connection to Coinbase Pro sandbox."""
    try:
        # Initialize exchange with sandbox credentials
        exchange = ccxt.coinbase({
            'apiKey': os.getenv('COINBASE_API_KEY'),
            'secret': os.getenv('COINBASE_API_SECRET'),
            'verbose': True,  # For debugging
        })
        
        # Test connection by fetching balance
        logger.info("Fetching balance from sandbox...")
        balance = exchange.fetch_balance()
        
        # Print results
        logger.info("Connection successful!")
        logger.info(f"Balance: {balance}")
        
        return balance
        
    except Exception as e:
        logger.error(f"Error connecting to sandbox: {str(e)}")
        raise

if __name__ == "__main__":
    test_sandbox_connection() 