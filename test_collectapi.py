"""CollectAPI test script"""
from src.data.news_fetcher import NewsFetcher

fetcher = NewsFetcher()

print(f"NEWS_API_KEY: {'Var' if fetcher.api_key else 'Yok'}")
print(f"COLLECTAPI_KEY: {'Var' if fetcher.collectapi_key else 'Yok'}")

# CollectAPI test
if fetcher.collectapi_key:
    print("\n--- CollectAPI Test ---")
    articles = fetcher.fetch_collectapi_news("economy")
    print(f"Economy haberleri: {len(articles)}")
    if articles:
        print(f"İlk haber: {articles[0].get('name', '')[:80]}...")
else:
    print("CollectAPI key yok - .env dosyasını kontrol et")