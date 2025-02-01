"""Main entry point for the trading bot."""

from src import MomentumTradingBot, API_KEY, API_SECRET, TRADING_PAIR_LIMIT


def main():
    """Initialize and run the trading bot."""
    bot = MomentumTradingBot(
        API_KEY,
        API_SECRET,
        TRADING_PAIR_LIMIT,
    )
    bot.run()


if __name__ == "__main__":
    main() 