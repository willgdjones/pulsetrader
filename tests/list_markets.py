import os
import logging
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from coinbase.rest import RESTClient
from tqdm import tqdm
from openai import OpenAI
import yfinance as yf
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Define major cryptocurrencies to focus on
MAJOR_CRYPTOS = {
    'BTC', 'ETH', 'SOL', 'AVAX', 'MATIC', 'LINK', 'DOT', 'ADA', 
    'DOGE', 'XRP', 'UNI', 'AAVE', 'LTC', 'BCH', 'ATOM'
}

# Stablecoins to exclude
STABLECOINS = {'USDT', 'USDC', 'GUSD', 'DAI', 'BUSD', 'PAX', 'UST'}

def analyze_market_with_llm(market_data):
    """
    Use GPT-4 to analyze market data and provide investment recommendations.
    """
    # Prepare market data for the prompt
    candles = market_data['candles']
    if not candles:
        return "HOLD", ["Insufficient data for analysis"], []

    # Format candle data
    candle_data = []
    for candle in candles:
        candle_data.append({
            'date': parse_timestamp(candle.start).strftime('%Y-%m-%d'),
            'open': float(candle.open),
            'high': float(candle.high),
            'low': float(candle.low),
            'close': float(candle.close),
            'volume': float(candle.volume)
        })

    # Calculate some basic metrics to help the LLM
    current_price = market_data['price']
    latest_close = float(candles[-1].close)
    price_change_24h = ((current_price - latest_close) / latest_close) * 100
    
    # Format the historical data
    history_json = json.dumps([{
        'date': c['date'],
        'close': c['close'],
        'volume': c['volume']
    } for c in candle_data], indent=2)

    # Create a structured prompt with proper escaping
    prompt = f"""Analyze this cryptocurrency market data and return a JSON response.

Asset: {market_data['id']}
Current Price: ${market_data['price']:,.2f}
24h Change: {price_change_24h:+.2f}%
24h Volume: ${market_data['volume']:,.2f}
Status: {market_data['status']}

Recent price history:
{history_json}

Respond with ONLY a JSON object in this exact format (no additional text):
{{
    "recommendation": "BUY" or "SELL" or "HOLD",
    "metrics": [
        "Key observation about price trends",
        "Key observation about volume",
        "Key observation about volatility"
    ],
    "rationale": [
        "Primary reason for recommendation",
        "Secondary factors considered",
        "Risk assessment"
    ]
}}"""

    try:
        # Get analysis from GPT-4
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a cryptocurrency analyst. "
                        "Respond only with a valid JSON object. "
                        "No additional text or explanation."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        # Get the response content
        content = response.choices[0].message.content.strip()
        
        # Try to find JSON in the response if there's any extra text
        try:
            # First try direct JSON parsing
            analysis = json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to find JSON-like structure
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(0))
            else:
                raise ValueError("No valid JSON found in response")

        # Validate the response structure
        required_keys = ['recommendation', 'metrics', 'rationale']
        if not all(k in analysis for k in required_keys):
            raise ValueError("Missing required keys in response")

        return (
            analysis["recommendation"],
            analysis["metrics"],
            analysis["rationale"]
        )

    except Exception as e:
        logger.error(f"Error getting LLM analysis: {str(e)}")
        
        # Provide a basic analysis based on price change
        if price_change_24h > 5:
            rec = "SELL"  # Potential profit taking
            metrics = [
                f"Strong upward movement: +{price_change_24h:.2f}%",
                f"Current price: ${current_price:,.2f}",
                f"24h volume: ${market_data['volume']:,.2f}"
            ]
            rationale = [
                "Price has moved up significantly in 24h",
                "Consider taking profits",
                "Watch for potential reversal"
            ]
        elif price_change_24h < -5:
            rec = "BUY"  # Potential dip buying
            metrics = [
                f"Strong downward movement: {price_change_24h:.2f}%",
                f"Current price: ${current_price:,.2f}",
                f"24h volume: ${market_data['volume']:,.2f}"
            ]
            rationale = [
                "Price has dropped significantly in 24h",
                "Potential buying opportunity",
                "Monitor for further downside"
            ]
        else:
            rec = "HOLD"
            metrics = [
                f"Stable price movement: {price_change_24h:.2f}%",
                f"Current price: ${current_price:,.2f}",
                f"24h volume: ${market_data['volume']:,.2f}"
            ]
            rationale = [
                "No significant price movement in 24h",
                "Market showing stability",
                "Continue monitoring for breakout"
            ]
        
        return rec, metrics, rationale

def parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object."""
    try:
        # Try parsing as ISO format
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        try:
            # Try parsing as Unix timestamp
            return datetime.fromtimestamp(float(timestamp_str))
        except (ValueError, TypeError):
            logger.warning(f"Could not parse timestamp: {timestamp_str}")
            return datetime.utcnow()

def get_candle_data(client, product_id, days=7):
    """Get candle data for the specified product."""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    # Convert to Unix timestamps (seconds since epoch)
    start_timestamp = int(start_time.timestamp())
    end_timestamp = int(end_time.timestamp())
    
    try:
        candles = client.get_candles(
            product_id=product_id,
            start=start_timestamp,
            end=end_timestamp,
            granularity='ONE_DAY'
        )
        return candles.candles if candles else []
    except Exception as e:
        logger.warning(f"Could not fetch candles for {product_id}: {str(e)}")
        return []

def format_candle_row(date, open_price, high, low, close, volume):
    """Format a single candle row with color coding."""
    change = ((close - open_price) / open_price) * 100
    direction = "🟢" if close >= open_price else "🔴"
    return (
        f"{date.strftime('%Y-%m-%d')} | "
        f"O: ${open_price:<10,.2f} | "
        f"H: ${high:<10,.2f} | "
        f"L: ${low:<10,.2f} | "
        f"C: ${close:<10,.2f} | "
        f"Vol: ${volume:<12,.0f} | "
        f"{direction} {change:+.2f}%"
    )

def get_yfinance_crypto_data(symbol, days=7):
    """
    Get additional crypto market data from Yahoo Finance.
    
    Args:
        symbol (str): Trading symbol (e.g., 'BTC-USD')
        days (int): Number of days of historical data to fetch
    """
    try:
        # Convert Coinbase symbol format to Yahoo Finance format
        base_currency = symbol.split('-')[0]
        yf_symbol = f"{base_currency}-USD"  # Use hyphenated format for crypto
        
        logger.info(f"Fetching yfinance data for {yf_symbol}")
        ticker = yf.Ticker(yf_symbol)
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        hist = ticker.history(start=start_date, end=end_date, interval='1d')
        
        if hist.empty:
            logger.warning(f"No historical data found for {yf_symbol}")
            return None
            
        # Get additional info
        info = ticker.info
        
        # Handle cases where info might be incomplete
        market_data = {
            'market_cap': None,
            'volume_24h': None,
            'circulating_supply': None,
            'fifty_day_avg': None,
            'two_hundred_day_avg': None,
            'historical_volatility': None
        }
        
        # Safely extract data from info dictionary
        if info:
            market_data.update({
                'market_cap': info.get('marketCap'),
                'volume_24h': info.get('volume24Hr', info.get('volume')),  # Try both keys
                'circulating_supply': info.get('circulatingSupply'),
                'fifty_day_avg': info.get('fiftyDayAverage'),
                'two_hundred_day_avg': info.get('twoHundredDayAverage')
            })
        
        # Calculate volatility if we have price data
        if not hist.empty and len(hist['Close']) > 1:
            market_data['historical_volatility'] = calculate_historical_volatility(hist['Close'])
        
        return market_data
        
    except Exception as e:
        logger.warning(f"Could not fetch yfinance data for {symbol}: {str(e)}")
        return None

def calculate_historical_volatility(prices, window=30):
    """Calculate historical volatility from price data."""
    try:
        if len(prices) < 2:
            return None
        
        # Calculate daily returns
        returns = prices.pct_change().dropna()
        
        # Calculate rolling standard deviation
        volatility = returns.rolling(window=min(window, len(returns))).std()
        
        # Annualize volatility (multiply by sqrt of trading days in a year)
        annualized_vol = volatility.iloc[-1] * (252 ** 0.5) * 100
        
        return round(annualized_vol, 2)
    except Exception:
        return None

def get_market_info(client, limit=10):
    """Get top USD markets by volume and their candle data."""
    try:
        # Get all products
        logger.info("Fetching available markets...")
        products = client.get_products()
        
        # Filter for USD markets and major cryptos only
        filtered_products = [
            p for p in products.products 
            if (p.quote_currency_id == 'USD' and 
                p.base_currency_id in MAJOR_CRYPTOS and
                p.base_currency_id not in STABLECOINS)
        ]
        
        logger.info(f"Found {len(filtered_products)} major markets")
        
        # Get market data for filtered products
        market_data = []
        logger.info("Fetching market data...")
        for product in tqdm(filtered_products, desc="Processing markets"):
            try:
                # Get latest trades
                trades = client.get_market_trades(
                    product_id=product.product_id,
                    limit=100
                )
                
                if trades and trades.trades:
                    volume = sum(
                        float(trade.size) * float(trade.price) 
                        for trade in trades.trades
                    )
                    latest_price = float(trades.trades[0].price)
                    
                    # Get candle data
                    candles = get_candle_data(client, product.product_id)
                    
                    # Get additional data from yfinance
                    yf_data = get_yfinance_crypto_data(product.product_id)
                    
                    market_data.append({
                        'id': product.product_id,
                        'base': product.base_currency_id,
                        'quote': product.quote_currency_id,
                        'price': latest_price,
                        'volume': volume,
                        'min_size': product.base_min_size,
                        'status': product.status,
                        'candles': candles,
                        'yf_data': yf_data
                    })
                    
            except Exception as e:
                logger.warning(
                    f"Could not fetch data for {product.product_id}: {str(e)}"
                )
        
        # Sort by volume and get top markets
        market_data.sort(key=lambda x: x['volume'], reverse=True)
        top_markets = market_data[:limit]
        
        # Display a summary of recommendations first
        logger.info("\n=== TRADING RECOMMENDATIONS SUMMARY ===")
        logger.info("=" * 50)
        
        # Collect all analyses first
        market_analyses = []
        for market in top_markets:
            if market['candles']:
                recommendation, metrics, rationale = analyze_market_with_llm(market)
                market_analyses.append({
                    'id': market['id'],
                    'recommendation': recommendation,
                    'price': market['price'],
                    'volume': market['volume'],
                    'metrics': metrics,
                    'rationale': rationale
                })

        # Display recommendations grouped by type
        for rec_type in ['BUY', 'SELL', 'HOLD']:
            matches = [m for m in market_analyses if m['recommendation'] == rec_type]
            if matches:
                logger.info(f"\n{rec_type} RECOMMENDATIONS:")
                logger.info("-" * 40)
                for m in matches:
                    logger.info(
                        f"{m['id']:<12} | "
                        f"Price: ${m['price']:<10,.2f} | "
                        f"Vol: ${m['volume']:,.0f}"
                    )
                    logger.info(f"Key Reason: {m['rationale'][0]}")
                    logger.info("-" * 40)

        # Then display detailed analysis for each market
        logger.info("\n\n=== DETAILED MARKET ANALYSES ===")
        for market in top_markets:
            logger.info(f"\n{market['id']} Market Summary:")
            logger.info("=" * 100)
            
            # Display market data
            logger.info(f"Current Price: ${market['price']:,.2f}")
            logger.info(f"24h Volume: ${market['volume']:,.0f}")
            logger.info(f"Min Size: {market['min_size']}")
            logger.info(f"Status: {market['status']}")
            
            # Display yfinance data if available
            if market['yf_data']:
                yf_data = market['yf_data']
                logger.info("\nAdditional Market Metrics (Yahoo Finance):")
                logger.info("-" * 50)
                if yf_data['market_cap']:
                    logger.info(f"Market Cap: ${yf_data['market_cap']:,.0f}")
                if yf_data['circulating_supply']:
                    logger.info(f"Circulating Supply: {yf_data['circulating_supply']:,.0f}")
                if yf_data['fifty_day_avg']:
                    logger.info(f"50-Day Average: ${yf_data['fifty_day_avg']:,.2f}")
                if yf_data['two_hundred_day_avg']:
                    logger.info(f"200-Day Average: ${yf_data['two_hundred_day_avg']:,.2f}")
                if yf_data['historical_volatility']:
                    logger.info(f"30-Day Annualized Volatility: {yf_data['historical_volatility']}%")

            if market['candles']:
                logger.info("\nDaily Candles (Past Week):")
                logger.info("-" * 100)
                for candle in market['candles']:
                    logger.info(format_candle_row(
                        parse_timestamp(candle.start),
                        float(candle.open),
                        float(candle.high),
                        float(candle.low),
                        float(candle.close),
                        float(candle.volume)
                    ))
                
                # Find the matching analysis
                analysis = next(
                    (m for m in market_analyses if m['id'] == market['id']), 
                    None
                )
                if analysis:
                    logger.info("\nAI Market Analysis:")
                    logger.info("-" * 100)
                    logger.info(
                        f"RECOMMENDATION: {analysis['recommendation']} "
                        f"{'🟢' if analysis['recommendation'] == 'BUY' else '🔴' if analysis['recommendation'] == 'SELL' else '⚪'}"
                    )
                    logger.info("\nKey Metrics and Observations:")
                    for metric in analysis['metrics']:
                        logger.info(f"• {metric}")
                    logger.info("\nRationale:")
                    for reason in analysis['rationale']:
                        logger.info(f"• {reason}")
            else:
                logger.info("\nNo candle data available")
            
            logger.info("=" * 100)
        
        # Show overall summary
        if top_markets:
            total_volume = sum(m['volume'] for m in top_markets)
            avg_price = sum(m['price'] for m in top_markets)/len(top_markets)
            
            logger.info("\nOverall Summary:")
            logger.info("-" * 30)
            logger.info(f"Total Volume: ${total_volume:,.0f}")
            logger.info(f"Average Price: ${avg_price:,.2f}")
        
        return top_markets
        
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        raise

if __name__ == "__main__":
    client = RESTClient(
        api_key=os.getenv('COINBASE_API_KEY'),
        api_secret=os.getenv('COINBASE_API_SECRET')
    )
    get_market_info(client) 