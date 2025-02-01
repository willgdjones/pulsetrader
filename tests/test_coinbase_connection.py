import os
import sys
import logging
import uuid
from json import dumps
from dotenv import load_dotenv
from coinbase.rest import RESTClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_coinbase_connection():
    """Test connection to Coinbase Advanced Trade API."""
    try:
        # Initialize client
        client = RESTClient(
            api_key=os.getenv('COINBASE_API_KEY'),
            api_secret=os.getenv('COINBASE_API_SECRET')
        )
        
        # Test getting accounts
        logger.info("Fetching accounts...")
        accounts = client.get_accounts()
        logger.info("Accounts retrieved successfully:")
        logger.info(dumps(accounts.to_dict(), indent=2))
        
        # Test getting BTC-USD product
        logger.info("\nFetching BTC-USD product info...")
        product = client.get_product("BTC-USD")
        logger.info(dumps(product.to_dict(), indent=2))
        
        # Test getting current price
        logger.info("\nFetching BTC-USD market price...")
        market_trades = client.get_market_trades("BTC-USD", limit=1)
        if market_trades.trades:
            latest_price = market_trades.trades[0].price
            logger.info(f"Current BTC-USD price: ${float(latest_price):,.2f}")
        
        # Optional: Test placing a small order (commented out for safety)
        """
        logger.info("\nPlacing test order...")
        order = client.create_order(
            client_order_id=str(uuid.uuid4()),
            product_id="BTC-USD",
            side="BUY",
            order_configuration={
                "market_market_ioc": {
                    "quote_size": "10"  # $10 USD worth of BTC
                }
            }
        )
        logger.info("Test order placed successfully:")
        logger.info(dumps(order.to_dict(), indent=2))
        """
        
    except Exception as e:
        logger.error(f"Error in Coinbase connection test: {str(e)}")
        raise

if __name__ == "__main__":
    test_coinbase_connection() 