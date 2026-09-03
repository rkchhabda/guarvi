#!/usr/bin/env python3
"""Fetch and process news and sentiment data.

This script provides a framework for integrating news sentiment data
from various sources (free and paid).

Sources:
- NewsAPI (free tier: 100 requests/day)
- GNews (free tier: 100 requests/day)
- RSS feeds from financial news sites
- Twitter/X API (paid)
- Reddit API (free with limits)
- Paid terminals: Bloomberg, Reuters, Refinitiv
- Indian sources: Moneycontrol, Economic Times, Business Standard, Mint

Features computed:
- Daily sentiment score (-1 to 1)
- Sentiment momentum (change over N days)
- News volume (article count)
- Positive/negative article ratio
- Entity-specific sentiment (per symbol)
- Sector-level sentiment
- Event detection (earnings, M&A, regulatory, etc.)
"""
from __future__ import annotations

import sys
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("news_sentiment")


# ---------------------------------------------------------------------------
# News API Clients
# ---------------------------------------------------------------------------

class NewsAPIClient:
    """Client for NewsAPI.org"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
    
    def fetch_everything(
        self,
        query: str,
        from_date: str,
        to_date: str,
        language: str = 'en',
        sort_by: str = 'publishedAt',
        page_size: int = 100,
    ) -> List[Dict]:
        """Fetch articles from NewsAPI."""
        import requests
        
        url = f"{self.base_url}/everything"
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'language': language,
            'sortBy': sort_by,
            'pageSize': page_size,
            'apiKey': self.api_key,
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json().get('articles', [])
            else:
                logger.error(f"NewsAPI error: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"NewsAPI request failed: {e}")
        
        return []


class GNewsClient:
    """Client for GNews.io"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://gnews.io/api/v4"
    
    def fetch_news(
        self,
        query: str,
        from_date: str,
        to_date: str,
        language: str = 'en',
        max_results: int = 100,
    ) -> List[Dict]:
        """Fetch articles from GNews."""
        import requests
        
        url = f"{self.base_url}/search"
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'lang': language,
            'max': max_results,
            'apikey': self.api_key,
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json().get('articles', [])
            else:
                logger.error(f"GNews error: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"GNews request failed: {e}")
        
        return []


class RSSFeedClient:
    """Client for RSS feeds from financial news sites."""
    
    INDIAN_FEEDS = {
        'moneycontrol': 'https://www.moneycontrol.com/rss/latestnews.xml',
        'economic_times': 'https://economictimes.indiatimes.com/rssfeedsdefault.cms',
        'business_standard': 'https://www.business-standard.com/rss/latest.rss',
        'livemint': 'https://www.livemint.com/rss/markets',
        'financial_express': 'https://www.financialexpress.com/feed/',
        'zeebiz': 'https://www.zeebiz.com/rss/feed.xml',
    }
    
    def __init__(self):
        pass
    
    def fetch_feed(self, feed_url: str, max_items: int = 50) -> List[Dict]:
        """Fetch and parse RSS feed."""
        import feedparser
        
        try:
            feed = feedparser.parse(feed_url)
            articles = []
            for entry in feed.entries[:max_items]:
                articles.append({
                    'title': entry.get('title', ''),
                    'description': entry.get('summary', entry.get('description', '')),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', entry.get('updated', '')),
                    'source': feed.feed.get('title', ''),
                })
            return articles
        except Exception as e:
            logger.error(f"RSS feed parse failed for {feed_url}: {e}")
            return []
    
    def fetch_all_indian_feeds(self, max_items_per_feed: int = 50) -> List[Dict]:
        """Fetch all Indian financial news RSS feeds."""
        all_articles = []
        for name, url in self.INDIAN_FEEDS.items():
            logger.info(f"Fetching RSS feed: {name}")
            articles = self.fetch_feed(url, max_items_per_feed)
            for a in articles:
                a['source_name'] = name
            all_articles.extend(articles)
        return all_articles


# ---------------------------------------------------------------------------
# Sentiment Analysis
# ---------------------------------------------------------------------------

# Financial sentiment lexicon (simplified)
POSITIVE_WORDS = {
    'surge', 'rally', 'gain', 'rise', 'jump', 'climb', 'soar', 'rocket',
    'bullish', 'optimistic', 'positive', 'strong', 'growth', 'profit',
    'beat', 'exceed', 'outperform', 'upgrade', 'buy', 'recommend',
    'record', 'high', 'breakthrough', 'milestone', 'expansion', 'acquisition',
    'merger', 'partnership', 'deal', 'contract', 'order', 'dividend',
    'bonus', 'split', 'buyback', 'investment', 'funding', 'ipo',
}

NEGATIVE_WORDS = {
    'fall', 'drop', 'decline', 'slide', 'plunge', 'crash', 'tumble', 'sink',
    'bearish', 'pessimistic', 'negative', 'weak', 'loss', 'miss', 'disappoint',
    'underperform', 'downgrade', 'sell', 'reduce', 'cut', 'layoff', 'job cut',
    'bankruptcy', 'default', 'fraud', 'scam', 'investigation', 'probe',
    'lawsuit', 'litigation', 'penalty', 'fine', 'warning', 'alert',
    'risk', 'concern', 'worry', 'fear', 'uncertainty', 'volatility',
}

# Indian market specific terms
INDIAN_POSITIVE = {
    'nifty', 'sensex', 'record high', 'all-time high', 'lifetime high',
    'fii buying', 'dii buying', 'inflow', 'investment', 'gdp growth',
    'reform', 'policy support', 'rate cut', 'liquidity', 'stimulus',
}

INDIAN_NEGATIVE = {
    'fii selling', 'dii selling', 'outflow', 'withdrawal', 'rate hike',
    'inflation', 'deficit', 'downgrade', 'credit negative', 'stress',
    'npa', 'bad loan', 'default', 'haircut', 'resolution', 'insolvency',
}


def simple_sentiment_score(text: str) -> float:
    """
    Compute simple lexicon-based sentiment score.
    
    Returns score between -1 (negative) and 1 (positive).
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    
    pos_count = len(words & POSITIVE_WORDS) + len(words & INDIAN_POSITIVE)
    neg_count = len(words & NEGATIVE_WORDS) + len(words & INDIAN_NEGATIVE)
    
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    
    return (pos_count - neg_count) / total


def vader_sentiment_score(text: str) -> float:
    """
    Compute VADER sentiment score (requires nltk).
    
    Returns compound score between -1 and 1.
    """
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
        return scores['compound']
    except ImportError:
        logger.warning("nltk not installed, falling back to simple sentiment")
        return simple_sentiment_score(text)
    except Exception as e:
        logger.error(f"VADER sentiment failed: {e}")
        return simple_sentiment_score(text)


def finbert_sentiment_score(text: str) -> float:
    """
    Compute FinBERT sentiment score (requires transformers).
    
    Returns score between -1 and 1.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        # Load FinBERT (financial BERT)
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # FinBERT labels: 0=negative, 1=neutral, 2=positive
        score = probs[0][2].item() - probs[0][0].item()  # positive - negative
        return score
    except ImportError:
        logger.warning("transformers not installed, falling back to VADER")
        return vader_sentiment_score(text)
    except Exception as e:
        logger.error(f"FinBERT sentiment failed: {e}")
        return vader_sentiment_score(text)


# ---------------------------------------------------------------------------
# Symbol Extraction
# ---------------------------------------------------------------------------

# NSE symbol patterns
NSE_SYMBOLS = {
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR',
    'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'LT', 'AXISBANK',
    'BAJFINANCE', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'TITAN',
    'ULTRACEMCO', 'NESTLEIND', 'POWERGRID', 'NTPC', 'ONGC',
    'COALINDIA', 'TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'ADANIENT',
    'ADANIPORTS', 'BAJAJFINSV', 'BAJAJ-AUTO', 'HEROMOTOCO', 'EICHERMOT',
    'DRREDDY', 'CIPLA', 'DIVISLAB', 'TORNTPHARM', 'BIOCON',
    'WIPRO', 'TECHM', 'HCLTECH', 'MPHASIS', 'LTIM',
    'DMART', 'TRENT', 'INDHOTEL', 'JUBLFOOD', 'ZOMATO',
    'PAYTM', 'NYKAA', 'POLICYBZR', 'DELHIVERY', 'CARERATING',
}


def extract_symbols(text: str) -> List[str]:
    """Extract stock symbols from text."""
    if not text:
        return []
    
    text_upper = text.upper()
    found = []
    
    for symbol in NSE_SYMBOLS:
        # Match whole word boundaries
        pattern = r'\b' + re.escape(symbol) + r'\b'
        if re.search(pattern, text_upper):
            found.append(symbol)
    
    return found


def extract_sector(text: str) -> Optional[str]:
    """Extract sector mention from text."""
    sector_keywords = {
        'bank': 'Banks', 'banking': 'Banks', 'financial': 'Financials',
        'it': 'IT', 'technology': 'IT', 'software': 'IT',
        'auto': 'Auto', 'automobile': 'Auto', 'vehicle': 'Auto',
        'pharma': 'Healthcare', 'pharmaceutical': 'Healthcare', 'healthcare': 'Healthcare',
        'fmcg': 'FMCG', 'consumer': 'FMCG',
        'energy': 'Energy', 'oil': 'Energy', 'gas': 'Energy',
        'metal': 'Metals', 'steel': 'Metals', 'aluminium': 'Metals',
        'cement': 'Cement', 'construction': 'CapitalGoods',
        'realty': 'Realty', 'real estate': 'Realty',
        'telecom': 'Telecom', 'telecommunication': 'Telecom',
    }
    
    text_lower = text.lower()
    for keyword, sector in sector_keywords.items():
        if keyword in text_lower:
            return sector
    
    return None


# ---------------------------------------------------------------------------
# News Processing Pipeline
# ---------------------------------------------------------------------------

def process_news_articles(
    articles: List[Dict],
    sentiment_method: str = 'vader',  # 'simple', 'vader', 'finbert'
) -> pd.DataFrame:
    """
    Process raw news articles into structured DataFrame with sentiment.
    
    Parameters
    ----------
    articles : List[Dict]
        List of article dicts with keys: title, description, published, source, link
    sentiment_method : str
        Sentiment analysis method
    
    Returns
    -------
    DataFrame with columns: date, title, description, source, sentiment,
                            symbols, sector, url
    """
    if not articles:
        return pd.DataFrame()
    
    processed = []
    
    for article in articles:
        title = article.get('title', '') or ''
        description = article.get('description', '') or article.get('summary', '') or ''
        content = f"{title}. {description}"
        
        # Parse date
        published = article.get('published', article.get('publishedAt', article.get('date', '')))
        try:
            date = pd.to_datetime(published).date()
        except Exception:
            date = pd.Timestamp.now().date()
        
        # Sentiment
        if sentiment_method == 'simple':
            sentiment = simple_sentiment_score(content)
        elif sentiment_method == 'vader':
            sentiment = vader_sentiment_score(content)
        elif sentiment_method == 'finbert':
            sentiment = finbert_sentiment_score(content)
        else:
            sentiment = simple_sentiment_score(content)
        
        # Extract symbols and sector
        symbols = extract_symbols(content)
        sector = extract_sector(content)
        
        processed.append({
            'date': date,
            'title': title,
            'description': description,
            'source': article.get('source', article.get('source_name', '')),
            'url': article.get('link', article.get('url', '')),
            'sentiment': sentiment,
            'symbols': symbols,
            'sector': sector,
        })
    
    df = pd.DataFrame(processed)
    
    if df.empty:
        return df
    
    # Explode symbols (one row per symbol per article)
    df = df.explode('symbols').rename(columns={'symbols': 'symbol'})
    df = df.dropna(subset=['symbol'])
    
    return df


def aggregate_daily_sentiment(
    news_df: pd.DataFrame,
    symbols: List[str] = None,
) -> pd.DataFrame:
    """
    Aggregate article-level sentiment to daily symbol-level features.
    
    Features per symbol per date:
    - sentiment_mean: Mean sentiment score
    - sentiment_median: Median sentiment score
    - sentiment_std: Std of sentiment scores
    - article_count: Number of articles
    - positive_ratio: Ratio of positive articles (sentiment > 0.1)
    - negative_ratio: Ratio of negative articles (sentiment < -0.1)
    - sentiment_momentum_5d: 5-day change in mean sentiment
    - news_volume_zscore: Z-score of article count
    """
    if news_df.empty:
        return pd.DataFrame()
    
    df = news_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    if symbols:
        df = df[df['symbol'].isin(symbols)]
    
    # Daily aggregation per symbol
    daily = df.groupby(['symbol', 'date']).agg(
        sentiment_mean=('sentiment', 'mean'),
        sentiment_median=('sentiment', 'median'),
        sentiment_std=('sentiment', 'std'),
        article_count=('sentiment', 'count'),
        positive_count=('sentiment', lambda x: (x > 0.1).sum()),
        negative_count=('sentiment', lambda x: (x < -0.1).sum()),
    ).reset_index()
    
    daily['positive_ratio'] = daily['positive_count'] / daily['article_count'].replace(0, np.nan)
    daily['negative_ratio'] = daily['negative_count'] / daily['article_count'].replace(0, np.nan)
    daily['neutral_ratio'] = 1 - daily['positive_ratio'] - daily['negative_ratio']
    
    # Rolling features per symbol
    daily = daily.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    for symbol, group in daily.groupby('symbol'):
        idx = group.index
        grp = group.sort_values('date')
        
        # Sentiment moving averages
        daily.loc[idx, 'sentiment_5d_ma'] = grp['sentiment_mean'].rolling(5, min_periods=3).mean().values
        daily.loc[idx, 'sentiment_20d_ma'] = grp['sentiment_mean'].rolling(20, min_periods=10).mean().values
        
        # Sentiment momentum
        daily.loc[idx, 'sentiment_momentum_5d'] = grp['sentiment_mean'].diff(5).values
        daily.loc[idx, 'sentiment_momentum_20d'] = grp['sentiment_mean'].diff(20).values
        
        # Z-scores
        for w in [20, 60]:
            ma = grp['sentiment_mean'].rolling(w, min_periods=w//2).mean()
            std = grp['sentiment_mean'].rolling(w, min_periods=w//2).std(ddof=0)
            daily.loc[idx, f'sentiment_zscore_{w}d'] = ((grp['sentiment_mean'] - ma) / std.replace(0, np.nan)).values
        
        # News volume features
        daily.loc[idx, 'article_count_5d_ma'] = grp['article_count'].rolling(5, min_periods=3).mean().values
        daily.loc[idx, 'article_count_20d_ma'] = grp['article_count'].rolling(20, min_periods=10).mean().values
        
        vol_ma = grp['article_count'].rolling(20, min_periods=10).mean()
        vol_std = grp['article_count'].rolling(20, min_periods=10).std(ddof=0)
        daily.loc[idx, 'news_volume_zscore_20d'] = ((grp['article_count'] - vol_ma) / vol_std.replace(0, np.nan)).values
        
        # Positive/negative momentum
        daily.loc[idx, 'positive_ratio_5d_ma'] = grp['positive_ratio'].rolling(5, min_periods=3).mean().values
        daily.loc[idx, 'negative_ratio_5d_ma'] = grp['negative_ratio'].rolling(5, min_periods=3).mean().values
    
    return daily


def aggregate_sector_sentiment(
    news_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate sentiment to sector level.
    
    Features per sector per date:
    - sector_sentiment_mean: Mean sentiment across sector symbols
    - sector_article_count: Total articles mentioning sector
    - sector_positive_ratio: Positive article ratio
    - sector_news_volume_zscore: Z-score of sector news volume
    """
    if news_df.empty:
        return pd.DataFrame()
    
    df = news_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter articles with sector info
    df = df.dropna(subset=['sector'])
    
    if df.empty:
        return pd.DataFrame()
    
    sector_daily = df.groupby(['sector', 'date']).agg(
        sector_sentiment_mean=('sentiment', 'mean'),
        sector_sentiment_median=('sentiment', 'median'),
        sector_article_count=('sentiment', 'count'),
        sector_positive_ratio=('sentiment', lambda x: (x > 0.1).mean()),
        sector_negative_ratio=('sentiment', lambda x: (x < -0.1).mean()),
        unique_symbols=('symbol', 'nunique'),
    ).reset_index()
    
    # Rolling features per sector
    sector_daily = sector_daily.sort_values(['sector', 'date']).reset_index(drop=True)
    
    for sector, group in sector_daily.groupby('sector'):
        idx = group.index
        grp = group.sort_values('date')
        
        sector_daily.loc[idx, 'sector_sentiment_5d_ma'] = grp['sector_sentiment_mean'].rolling(5, min_periods=3).mean().values
        sector_daily.loc[idx, 'sector_sentiment_20d_ma'] = grp['sector_sentiment_mean'].rolling(20, min_periods=10).mean().values
        
        vol_ma = grp['sector_article_count'].rolling(20, min_periods=10).mean()
        vol_std = grp['sector_article_count'].rolling(20, min_periods=10).std(ddof=0)
        sector_daily.loc[idx, 'sector_news_volume_zscore'] = ((grp['sector_article_count'] - vol_ma) / vol_std.replace(0, np.nan)).values
    
    return sector_daily


def fetch_news_for_symbols(
    symbols: List[str],
    start_date: str,
    end_date: str,
    api_keys: Dict[str, str] = None,
    use_rss: bool = True,
    cache_dir: str = "data/external/news",
) -> pd.DataFrame:
    """
    Fetch news for a list of symbols from multiple sources.
    
    Parameters
    ----------
    symbols : List[str]
        List of stock symbols
    start_date : str
        Start date in YYYY-MM-DD
    end_date : str
        End date in YYYY-MM-DD
    api_keys : Dict[str, str]
        Dictionary with API keys: {'newsapi': '...', 'gnews': '...'}
    use_rss : bool
        Whether to use free RSS feeds
    cache_dir : str
        Cache directory
    
    Returns
    -------
    DataFrame with processed news articles
    """
    cache_path = Path(cache_dir) / f"news_{start_date}_to_{end_date}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded news from cache: {len(df)} articles")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    all_articles = []
    
    # RSS feeds (free)
    if use_rss:
        logger.info("Fetching Indian financial news RSS feeds...")
        rss_client = RSSFeedClient()
        rss_articles = rss_client.fetch_all_indian_feeds(max_items_per_feed=100)
        all_articles.extend(rss_articles)
        logger.info(f"Fetched {len(rss_articles)} articles from RSS feeds")
    
    # NewsAPI (requires key)
    if api_keys and 'newsapi' in api_keys:
        logger.info("Fetching from NewsAPI...")
        newsapi = NewsAPIClient(api_keys['newsapi'])
        
        # Query for Indian market
        query = " OR ".join(symbols[:10]) + " OR India stock market OR NSE OR BSE"
        articles = newsapi.fetch_everything(query, start_date, end_date, page_size=100)
        all_articles.extend(articles)
        logger.info(f"Fetched {len(articles)} articles from NewsAPI")
    
    # GNews (requires key)
    if api_keys and 'gnews' in api_keys:
        logger.info("Fetching from GNews...")
        gnews = GNewsClient(api_keys['gnews'])
        
        query = " OR ".join(symbols[:10]) + " OR India stock market"
        articles = gnews.fetch_news(query, start_date, end_date, max_results=100)
        all_articles.extend(articles)
        logger.info(f"Fetched {len(articles)} articles from GNews")
    
    if not all_articles:
        logger.warning("No articles fetched from any source")
        return pd.DataFrame()
    
    # Process articles
    logger.info("Processing articles...")
    processed_df = process_news_articles(all_articles, sentiment_method='vader')
    
    if processed_df.empty:
        logger.warning("No articles after processing")
        return pd.DataFrame()
    
    # Save cache
    processed_df.to_parquet(cache_path, index=False)
    logger.info(f"Saved {len(processed_df)} processed articles to cache")
    
    return processed_df


def compute_market_sentiment_features(
    news_df: pd.DataFrame,
    symbols: List[str],
) -> pd.DataFrame:
    """
    Compute market-wide sentiment features.
    
    Features per date:
    - market_sentiment_mean: Mean sentiment across all symbols
    - market_sentiment_breadth: % of symbols with positive sentiment
    - market_news_volume: Total articles across all symbols
    - market_positive_ratio: Overall positive article ratio
    - sector_sentiment_dispersion: Std of sector sentiments
    """
    if news_df.empty:
        return pd.DataFrame()
    
    df = news_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Daily market-level aggregation
    market_daily = df.groupby('date').agg(
        market_sentiment_mean=('sentiment', 'mean'),
        market_sentiment_median=('sentiment', 'median'),
        market_article_count=('sentiment', 'count'),
        market_positive_ratio=('sentiment', lambda x: (x > 0.1).mean()),
        market_negative_ratio=('sentiment', lambda x: (x < -0.1).mean()),
        unique_symbols=('symbol', 'nunique'),
    ).reset_index()
    
    # Breadth: % of tracked symbols with positive sentiment
    symbol_daily = df.groupby(['date', 'symbol'])['sentiment'].mean().reset_index()
    symbol_daily['is_positive'] = symbol_daily['sentiment'] > 0.1
    
    breadth = symbol_daily.groupby('date').agg(
        market_breadth_positive=('is_positive', 'mean'),
        symbols_with_news=('symbol', 'nunique'),
    ).reset_index()
    
    market_daily = market_daily.merge(breadth, on='date', how='left')
    
    # Rolling features
    market_daily = market_daily.sort_values('date').reset_index(drop=True)
    
    market_daily['market_sentiment_5d_ma'] = market_daily['market_sentiment_mean'].rolling(5, min_periods=3).mean()
    market_daily['market_sentiment_20d_ma'] = market_daily['market_sentiment_mean'].rolling(20, min_periods=10).mean()
    market_daily['market_breadth_5d_ma'] = market_daily['market_breadth_positive'].rolling(5, min_periods=3).mean()
    
    # Z-scores
    for w in [20, 60]:
        ma = market_daily['market_sentiment_mean'].rolling(w, min_periods=w//2).mean()
        std = market_daily['market_sentiment_mean'].rolling(w, min_periods=w//2).std(ddof=0)
        market_daily[f'market_sentiment_zscore_{w}d'] = (market_daily['market_sentiment_mean'] - ma) / std.replace(0, np.nan)
    
    return market_daily


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("News/Sentiment data pipeline")
    logger.warning("This requires API keys for NewsAPI/GNews or uses free RSS feeds")
    
    # Example symbols
    symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN']
    
    # Date range (last 30 days for testing)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # API keys (set these in environment or config)
    api_keys = {
        # 'newsapi': os.environ.get('NEWSAPI_KEY'),
        # 'gnews': os.environ.get('GNEWS_KEY'),
    }
    
    # Fetch news (will use RSS if no API keys)
    logger.info(f"Fetching news from {start_date} to {end_date}")
    news_df = fetch_news_for_symbols(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        api_keys=api_keys,
        use_rss=True,
    )
    
    if news_df.empty:
        logger.warning("No news data fetched")
        logger.info("Create synthetic data for testing...")
        
        # Create synthetic news data for testing
        dates = pd.date_range(start_date, end_date, freq='B')
        synthetic_articles = []
        
        for date in dates:
            for symbol in symbols:
                n_articles = np.random.poisson(2)
                for _ in range(n_articles):
                    sentiment = np.random.normal(0, 0.3)
                    synthetic_articles.append({
                        'date': date.date(),
                        'title': f'{symbol} news article',
                        'description': f'Some news about {symbol}',
                        'source': 'synthetic',
                        'url': '',
                        'sentiment': sentiment,
                        'symbols': [symbol],
                        'sector': np.random.choice(['Banks', 'IT', 'Energy', 'FMCG', 'Auto']),
                    })
        
        news_df = pd.DataFrame(synthetic_articles)
        news_df = news_df.explode('symbols').rename(columns={'symbols': 'symbol'})
    
    # Aggregate to daily symbol-level features
    logger.info("Aggregating daily sentiment features...")
    daily_sentiment = aggregate_daily_sentiment(news_df, symbols)
    
    if not daily_sentiment.empty:
        output_dir = ROOT / "data" / "external"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        features_path = output_dir / "news_sentiment_features.parquet"
        daily_sentiment.to_parquet(features_path, index=False)
        logger.info(f"Saved daily sentiment features to {features_path}")
        logger.info(f"Shape: {daily_sentiment.shape}")
        print(daily_sentiment.tail(10).to_string())
    
    # Sector sentiment
    logger.info("Aggregating sector sentiment...")
    sector_sentiment = aggregate_sector_sentiment(news_df)
    
    if not sector_sentiment.empty:
        sector_path = output_dir / "sector_sentiment_features.parquet"
        sector_sentiment.to_parquet(sector_path, index=False)
        logger.info(f"Saved sector sentiment features to {sector_path}")
    
    # Market sentiment
    logger.info("Computing market sentiment...")
    market_sentiment = compute_market_sentiment_features(news_df, symbols)
    
    if not market_sentiment.empty:
        market_path = output_dir / "market_sentiment_features.parquet"
        market_sentiment.to_parquet(market_path, index=False)
        logger.info(f"Saved market sentiment features to {market_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())