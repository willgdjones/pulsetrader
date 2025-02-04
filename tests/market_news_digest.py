import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import json
import requests
from bs4 import BeautifulSoup
import time

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

# Define timeframes for analysis
TIMEFRAMES = {
    'daily': '1d',
    'weekly': '1wk',
    'monthly': '1mo',
    'quarterly': '3mo'
}

# Define market sectors and their key tickers
MARKET_SECTORS = {
    'Technology': [
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSM', 'AVGO', 'ORCL'
    ],
    'Finance': [
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'C', 'SPGI'
    ],
    'Healthcare': [
        'JNJ', 'UNH', 'LLY', 'PFE', 'ABT', 'TMO', 'DHR', 'BMY'
    ],
    'Crypto': [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'MATIC-USD'
    ],
    'Energy': [
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'VLO', 'MPC', 'KMI'
    ],
    'Consumer': [
        'AMZN', 'WMT', 'PG', 'KO', 'PEP', 'COST', 'NKE', 'MCD'
    ]
}

def calculate_period_change(hist_data: pd.DataFrame) -> Tuple[float, float]:
    """Calculate price change and volatility for a period."""
    if hist_data.empty or len(hist_data) < 2:
        return 0.0, 0.0
        
    start_price = hist_data['Close'].iloc[0]
    end_price = hist_data['Close'].iloc[-1]
    price_change = ((end_price - start_price) / start_price) * 100
    
    # Calculate volatility
    returns = hist_data['Close'].pct_change().dropna()
    volatility = returns.std() * (252 ** 0.5) * 100  # Annualized
    
    return price_change, volatility

def fetch_news_data(ticker: str) -> List[Dict]:
    """
    Fetch news from multiple sources for a given ticker.
    """
    news_items = []
    
    try:
        # 1. Yahoo Finance News
        stock = yf.Ticker(ticker)
        if hasattr(stock, 'news'):
            yf_news = stock.news[:10] if stock.news else []  # Get more news items
            for news in yf_news:
                news_items.append({
                    'source': 'Yahoo Finance',
                    'title': news.get('title', ''),
                    'link': news.get('link', ''),
                    'publisher': news.get('publisher', ''),
                    'published': datetime.fromtimestamp(
                        news.get('providerPublishTime', 0)
                    ).strftime('%Y-%m-%d %H:%M:%S'),
                    'summary': news.get('summary', '')
                })
        
        # 2. Fetch from Seeking Alpha (if not crypto)
        if not ticker.endswith('-USD'):
            sa_url = f"https://seekingalpha.com/api/v3/symbols/{ticker}/news"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(sa_url, headers=headers)
            if response.status_code == 200:
                sa_data = response.json()
                if 'data' in sa_data:
                    for item in sa_data['data'][:5]:
                        news_items.append({
                            'source': 'Seeking Alpha',
                            'title': item.get('attributes', {}).get('title', ''),
                            'link': f"https://seekingalpha.com{item.get('links', {}).get('self', '')}",
                            'published': item.get('attributes', {}).get('publishOn', ''),
                            'summary': item.get('attributes', {}).get('summary', '')
                        })
        
        # 3. For crypto, add CoinDesk news
        if ticker.endswith('-USD'):
            coin = ticker.split('-')[0].lower()
            cd_url = f"https://www.coindesk.com/search?s={coin}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(cd_url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = soup.find_all('article', limit=5)
                for article in articles:
                    title = article.find('h6')
                    link = article.find('a')
                    if title and link:
                        news_items.append({
                            'source': 'CoinDesk',
                            'title': title.text.strip(),
                            'link': f"https://www.coindesk.com{link.get('href', '')}",
                            'published': datetime.now().strftime('%Y-%m-%d'),
                            'summary': ''
                        })
        
        # Add sentiment analysis for news titles
        if news_items:
            news_items = analyze_news_sentiment(news_items)
        
        return news_items
        
    except Exception as e:
        logger.warning(f"Error fetching news for {ticker}: {str(e)}")
        return []

def analyze_news_sentiment(news_items: List[Dict]) -> List[Dict]:
    """
    Analyze sentiment of news articles using OpenAI.
    """
    try:
        # Prepare news titles for analysis
        titles = [item['title'] for item in news_items if item['title']]
        
        if not titles:
            return news_items
            
        prompt = f"""Analyze the sentiment of these news headlines:

Headlines:
{json.dumps(titles, indent=2)}

For each headline, provide a JSON object with these fields:
- sentiment: "positive", "negative", or "neutral"
- implication: a brief summary of market implications

Respond with a JSON array containing one object per headline.
Example format:
[
    {{"sentiment": "positive", "implication": "Indicates strong growth potential"}},
    {{"sentiment": "negative", "implication": "Suggests market challenges"}}
]"""

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial news analyst. Analyze sentiment "
                        "objectively and respond with a valid JSON array only. "
                        "Each array item should be a JSON object with 'sentiment' "
                        "and 'implication' fields."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        try:
            # Parse the response content as JSON
            content = response.choices[0].message.content.strip()
            # Handle potential extra text around JSON
            content = content[content.find('['):content.rfind(']')+1]
            sentiments = json.loads(content)
            
            # Update each news item with its sentiment analysis
            for item, sentiment in zip(news_items, sentiments):
                if isinstance(sentiment, dict):
                    item['sentiment'] = sentiment.get('sentiment', 'neutral')
                    item['implication'] = sentiment.get('implication', '')
                else:
                    item['sentiment'] = 'neutral'
                    item['implication'] = ''
            
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Could not parse sentiment analysis response: {str(e)}")
            # Set default values for items without sentiment
            for item in news_items:
                item['sentiment'] = 'neutral'
                item['implication'] = ''
        
        return news_items
        
    except Exception as e:
        logger.warning(f"Error analyzing news sentiment: {str(e)}")
        # Return items with default neutral sentiment
        for item in news_items:
            item['sentiment'] = 'neutral'
            item['implication'] = ''
        return news_items

def fetch_ticker_data(ticker: str) -> Optional[Dict]:
    """Fetch recent data and news for a single ticker."""
    try:
        stock = yf.Ticker(ticker)
        timeframe_data = {}
        
        # Fetch data for each timeframe
        for period_name, period in TIMEFRAMES.items():
            hist = stock.history(period=period)
            if not hist.empty:
                change, vol = calculate_period_change(hist)
                timeframe_data[period_name] = {
                    'price_change': change,
                    'volatility': vol,
                    'volume': hist['Volume'].sum() if 'Volume' in hist else None,
                    'high': hist['High'].max(),
                    'low': hist['Low'].min(),
                    'close': hist['Close'].iloc[-1]
                }
        
        if not timeframe_data:
            logger.warning(f"No historical data found for {ticker}")
            return None
        
        # Fetch comprehensive news data
        news_data = fetch_news_data(ticker)
        
        # Initialize data dictionary
        data = {
            'symbol': ticker,
            'current_price': timeframe_data['daily']['close'],
            'timeframes': timeframe_data,
            'news': news_data,
            'calendar': None
        }
        
        # Get calendar events for non-crypto assets
        if not ticker.endswith('-USD'):
            try:
                data['calendar'] = (
                    stock.calendar if hasattr(stock, 'calendar') else None
                )
            except Exception as e:
                logger.debug(f"Could not fetch calendar for {ticker}: {str(e)}")
        
        return data
        
    except Exception as e:
        logger.warning(f"Error fetching data for {ticker}: {str(e)}")
        return None

def generate_sector_summary(sector_name: str, sector_data: List[Dict]) -> str:
    """Generate an AI summary for a market sector using OpenAI."""
    try:
        # Prepare the data for GPT analysis
        market_summary = []
        news_summary = []
        
        # Group news by sentiment
        positive_news = []
        negative_news = []
        neutral_news = []
        
        for stock in sector_data:
            if not stock:
                continue
                
            # Format price differently for crypto vs stocks
            is_crypto = stock['symbol'].endswith('-USD')
            price_format = ',.0f' if is_crypto else ',.2f'
            
            # Create summary for each timeframe
            timeframe_summary = []
            for period, data in stock['timeframes'].items():
                timeframe_summary.append(
                    f"{period}: {data['price_change']:+.2f}% "
                    f"(vol: {data['volatility']:.1f}%)"
                )
            
            market_summary.append({
                'symbol': stock['symbol'],
                'price': f"${stock['current_price']:{price_format}}",
                'timeframes': timeframe_summary
            })
            
            # Process news with sentiment
            if stock.get('news'):
                for news in stock['news']:
                    news_item = {
                        'symbol': stock['symbol'],
                        'title': news.get('title', ''),
                        'source': news.get('source', 'Unknown'),
                        'published': news.get('published', ''),
                        'sentiment': news.get('sentiment', 'neutral'),
                        'implication': news.get('implication', ''),
                        'link': news.get('link', '')
                    }
                    
                    if news.get('sentiment') == 'positive':
                        positive_news.append(news_item)
                    elif news.get('sentiment') == 'negative':
                        negative_news.append(news_item)
                    else:
                        neutral_news.append(news_item)

        # Sort news by recency and limit to most recent
        for news_list in [positive_news, negative_news, neutral_news]:
            news_list.sort(
                key=lambda x: x['published'],
                reverse=True
            )
        
        # Prepare news summary
        news_summary = {
            'positive': positive_news[:3],  # Most recent 3 positive news
            'negative': negative_news[:3],  # Most recent 3 negative news
            'neutral': neutral_news[:3]     # Most recent 3 neutral news
        }

        prompt = f"""Analyze this {sector_name} sector data and create a 
comprehensive market summary.

Market Data:
{json.dumps(market_summary, indent=2)}

Recent News Analysis:
Positive News:
{json.dumps(news_summary['positive'], indent=2)}

Negative News:
{json.dumps(news_summary['negative'], indent=2)}

Neutral News:
{json.dumps(news_summary['neutral'], indent=2)}

Create a detailed analysis of the {sector_name} sector covering:

1. Current Market State:
   - Notable price movements across different timeframes
   - Volatility trends and significant changes
   - Overall sector sentiment

2. News Impact Analysis:
   - Key positive developments and their market implications
   - Significant challenges or concerns
   - Important neutral developments or industry changes

3. Trend Analysis:
   - Emerging patterns across different timeframes
   - Correlation between news events and price movements
   - Sector-wide trends and their implications

Format the response as a news article with:
- A clear, informative headline
- An executive summary of current developments
- Detailed analysis paragraphs covering price movements and news impact
- A conclusion paragraph about longer-term implications
- Include relevant news citations with their sources
- Maintain a neutral, fact-based tone throughout"""

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial news analyst creating "
                        "comprehensive market summaries. Focus on facts, data, "
                        "and multi-timeframe analysis. Cite specific news "
                        "sources when discussing developments. Avoid speculation "
                        "and maintain objectivity."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000  # Increased for more detailed analysis
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error generating {sector_name} summary: {str(e)}")
        return f"Error generating {sector_name} sector summary"

def generate_market_digest() -> str:
    """Generate a comprehensive market digest with AI-powered analysis."""
    logger.info("Generating market digest...")
    
    # Fetch data for all sectors in parallel
    sector_data = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        for sector, tickers in MARKET_SECTORS.items():
            futures = [
                executor.submit(fetch_ticker_data, ticker) 
                for ticker in tickers
            ]
            sector_data[sector] = [
                f.result() for f in futures 
                if f.result() is not None
            ]
    
    # Generate AI summaries for each sector
    sector_summaries = {}
    for sector, data in sector_data.items():
        if data:
            logger.info(f"Generating summary for {sector} sector...")
            sector_summaries[sector] = generate_sector_summary(sector, data)
    
    # Compile the full digest
    digest = ["# Daily Market Digest\n"]
    digest.append(
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
    )
    
    # Add sector summaries
    for sector, summary in sector_summaries.items():
        digest.append(f"\n## {sector} Sector\n")
        digest.append(summary)
        digest.append("\n" + "-" * 80 + "\n")
    
    return "\n".join(digest)

def save_digest(digest: str, output_dir: str = "reports") -> str:
    """Save the digest to a markdown file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{output_dir}/market_digest_{timestamp}.md"
    
    with open(filename, 'w') as f:
        f.write(digest)
    
    logger.info(f"Digest saved to {filename}")
    return filename

if __name__ == "__main__":
    try:
        digest = generate_market_digest()
        filename = save_digest(digest)
        print(digest)
    except Exception as e:
        logger.error(f"Error generating market digest: {str(e)}") 