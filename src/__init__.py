"""Trading bot package."""
from .trading_bot import MomentumTradingBot
from .config import API_KEY, API_SECRET, TRADING_PAIR_LIMIT

__all__ = ['MomentumTradingBot', 'API_KEY', 'API_SECRET', 'TRADING_PAIR_LIMIT'] 