import os
from dotenv import load_dotenv
from newsapi import NewsApiClient

load_dotenv()
api_key = os.getenv("NEWS_API_KEY")
newsapi = NewsApiClient(api_key=api_key)

def fetch_news(ticker: str):
    # Fetch news articles related to the stock ticker
    news = newsapi.get_everything(q=ticker,
                                  language='en',
                                  sort_by='relevancy',
                                  page_size=5)  # Fetch top 5 relevant news articles
    return news['articles']