import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define market indicators to track
MARKET_INDICATORS = {
    # Major Stock Indices
    'indices': [
        '^GSPC',    # S&P 500
        '^DJI',     # Dow Jones
        '^IXIC',    # NASDAQ
        '^VIX',     # Volatility Index
    ],
    # Currency pairs
    'forex': [
        'EURUSD=X',  # EUR/USD
        'JPY=X',     # USD/JPY
        'GBPUSD=X',  # GBP/USD
    ],
    # Commodities
    'commodities': [
        'GC=F',     # Gold
        'SI=F',     # Silver
        'CL=F',     # Crude Oil
    ],
    # Bond yields
    'bonds': [
        '^TNX',     # 10-year Treasury Yield
        '^IRX',     # 13-week Treasury Bill Rate
    ]
}

def get_market_data(symbol: str, days: int = 2) -> Optional[Dict]:
    """
    Fetch market data for a given symbol.
    
    Args:
        symbol: The ticker symbol to fetch
        days: Number of days of historical data
        
    Returns:
        Dictionary containing market metrics or None if fetch fails
    """
    try:
        ticker = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get historical data
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            return None
            
        # Calculate daily change
        latest_close = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else latest_close
        daily_change = ((latest_close - prev_close) / prev_close) * 100
        
        return {
            'symbol': symbol,
            'current_price': latest_close,
            'daily_change_pct': daily_change,
            'volume': hist['Volume'].iloc[-1] if 'Volume' in hist else None,
            'high': hist['High'].iloc[-1],
            'low': hist['Low'].iloc[-1]
        }
    except Exception as e:
        logger.warning(f"Failed to fetch data for {symbol}: {str(e)}")
        return None

def analyze_market_conditions() -> Dict:
    """
    Analyze current market conditions across different sectors.
    
    Returns:
        Dictionary containing market analysis and risk indicators
    """
    market_data = {}
    risk_signals = []
    
    # Fetch data for all indicators
    for category, symbols in MARKET_INDICATORS.items():
        market_data[category] = []
        for symbol in symbols:
            data = get_market_data(symbol)
            if data:
                market_data[category].append(data)
                
                # Check for significant moves
                if abs(data['daily_change_pct']) > 2:
                    risk_signals.append(
                        f"Large move in {symbol}: {data['daily_change_pct']:.2f}%"
                    )
    
    # Analyze VIX (Volatility Index)
    vix_data = next(
        (data for data in market_data['indices'] if data['symbol'] == '^VIX'),
        None
    )
    if vix_data:
        vix_level = vix_data['current_price']
        if vix_level > 30:
            risk_signals.append(f"High market fear (VIX: {vix_level:.2f})")
        elif vix_level < 15:
            risk_signals.append(f"Low market fear (VIX: {vix_level:.2f})")
    
    return {
        'timestamp': datetime.now().isoformat(),
        'market_data': market_data,
        'risk_signals': risk_signals,
        'market_sentiment': calculate_market_sentiment(market_data)
    }

def calculate_market_sentiment(market_data: Dict) -> str:
    """
    Calculate overall market sentiment based on various indicators.
    
    Args:
        market_data: Dictionary containing market data by category
        
    Returns:
        String indicating market sentiment: Bullish, Bearish, or Neutral
    """
    # Count positive and negative moves
    positive_moves = 0
    negative_moves = 0
    total_indicators = 0
    
    for category in market_data.values():
        for indicator in category:
            if indicator['daily_change_pct'] > 0.5:
                positive_moves += 1
            elif indicator['daily_change_pct'] < -0.5:
                negative_moves += 1
            total_indicators += 1
    
    if total_indicators == 0:
        return "Neutral"
    
    positive_ratio = positive_moves / total_indicators
    negative_ratio = negative_moves / total_indicators
    
    if positive_ratio > 0.6:
        return "Bullish"
    elif negative_ratio > 0.6:
        return "Bearish"
    return "Neutral"

def print_market_summary(analysis: Dict):
    """Print a formatted summary of market conditions."""
    logger.info("\n=== MARKET OVERVIEW ===")
    logger.info(f"Analysis Time: {analysis['timestamp']}")
    logger.info(f"Overall Sentiment: {analysis['market_sentiment']}")
    
    logger.info("\n=== MARKET INDICES ===")
    for index in analysis['market_data']['indices']:
        logger.info(
            f"{index['symbol']:<6} | "
            f"${index['current_price']:<8,.2f} | "
            f"{index['daily_change_pct']:+.2f}%"
        )
    
    logger.info("\n=== FOREX MARKETS ===")
    for pair in analysis['market_data']['forex']:
        logger.info(
            f"{pair['symbol']:<8} | "
            f"{pair['current_price']:<8,.4f} | "
            f"{pair['daily_change_pct']:+.2f}%"
        )
    
    logger.info("\n=== COMMODITIES ===")
    for commodity in analysis['market_data']['commodities']:
        logger.info(
            f"{commodity['symbol']:<4} | "
            f"${commodity['current_price']:<8,.2f} | "
            f"{commodity['daily_change_pct']:+.2f}%"
        )
    
    logger.info("\n=== BOND YIELDS ===")
    for bond in analysis['market_data']['bonds']:
        logger.info(
            f"{bond['symbol']:<4} | "
            f"{bond['current_price']:.2f}% | "
            f"{bond['daily_change_pct']:+.2f}%"
        )
    
    if analysis['risk_signals']:
        logger.info("\n=== RISK SIGNALS ===")
        for signal in analysis['risk_signals']:
            logger.info(f"• {signal}")

if __name__ == "__main__":
    analysis = analyze_market_conditions()
    print_market_summary(analysis) 