"""Trading bot implementation using Coinbase Advanced Trade API."""

import logging
import uuid
from datetime import datetime, timedelta
from coinbase.rest import RESTClient

class MomentumTradingBot:
    """Bot that implements momentum-based trading strategy."""

    def __init__(self, api_key: str, api_secret: str, trading_pair_limit: int = 20):
        """Initialize the trading bot with API credentials and settings."""
        self.client = RESTClient(
            api_key=api_key,
            api_secret=api_secret
        )
        self.trading_pair_limit = trading_pair_limit
        self.positions = {}
        
    def get_top_moving_coins(self):
        """Get top moving coins based on recent price changes."""
        try:
            # Get all available products
            products = self.client.get_products()
            
            # Filter USD pairs and calculate momentum
            momentum_data = []
            for product in products.products:
                if product.product_id.endswith("-USD"):
                    # Get recent candles
                    candles = self.client.get_product_candles(
                        product.product_id,
                        granularity="ONE_MINUTE",
                        start=datetime.now() - timedelta(minutes=5),
                        end=datetime.now()
                    )
                    
                    if candles.candles and len(candles.candles) >= 2:
                        start_price = float(candles.candles[-2].close)
                        end_price = float(candles.candles[-1].close)
                        price_change = ((end_price - start_price) / start_price) * 100
                        
                        momentum_data.append({
                            'symbol': product.product_id,
                            'price_change': price_change,
                            'current_price': end_price
                        })
            
            # Sort by price change
            momentum_data.sort(key=lambda x: x['price_change'], reverse=True)
            return momentum_data[:self.trading_pair_limit]
            
        except Exception as e:
            logging.error(f"Error in get_top_moving_coins: {str(e)}")
            return []

    def should_buy(self, momentum_data, threshold=2.0):
        return momentum_data['price_change'] > threshold

    def should_sell(self, symbol, current_price, buy_price, stop_loss=-1.0, take_profit=3.0):
        price_change = ((current_price - buy_price) / buy_price) * 100
        return price_change <= stop_loss or price_change >= take_profit

    def execute_trade(self, symbol, side, amount):
        try:
            order = self.client.create_order(
                client_order_id=str(uuid.uuid4()),
                product_id=symbol,
                side=side,
                order_configuration={
                    "market_market_ioc": {
                        "quote_size": str(amount)
                    }
                }
            )
            logging.info(f"Executed {side} order for {symbol}: {order}")
            return order
        except Exception as e:
            logging.error(f"Error executing trade: {str(e)}")
            return None

    def run(self):
        while True:
            try:
                # Get top moving coins
                momentum_data = self.get_top_moving_coins()
                
                # Check for selling positions
                for symbol, position in list(self.positions.items()):
                    current_price = self.client.get_product_ticker(symbol)['price']
                    if self.should_sell(symbol, current_price, position['buy_price']):
                        # Execute sell order
                        self.execute_trade(symbol, 'sell', position['amount'])
                        del self.positions[symbol]
                
                # Check for new buying opportunities
                for coin in momentum_data:
                    if (coin['symbol'] not in self.positions and 
                        self.should_buy(coin)):
                        # Calculate position size (example: 10% of available USD)
                        balance = self.client.get_account_balance(coin['symbol'])
                        usd_balance = float(balance['amount'])
                        position_size = usd_balance * 0.1
                        
                        # Execute buy order
                        order = self.execute_trade(
                            coin['symbol'], 
                            'buy', 
                            position_size
                        )
                        
                        if order:
                            self.positions[coin['symbol']] = {
                                'buy_price': coin['current_price'],
                                'amount': position_size
                            }
                
                # Wait for next iteration
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                time.sleep(60) 