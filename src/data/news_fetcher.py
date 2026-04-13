"""QAU - Haber Çekme ve Filtreleme Modülü
NewsAPI ve CollectAPI'den geniş kapsamlı haberler çeker, LLM ile altın-relevance filtreleme yapar
"""

import http.client
import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from loguru import logger
import time
import hashlib

from src.config import data_config, db_config
from src.data.db import get_db_connection


class NewsFetcher:
    """NewsAPI ve CollectAPI'den haber çekme ve LLM filtreleme"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or data_config.NEWS_API_KEY
        self.base_url = data_config.NEWS_API_BASE_URL
        self.queries = data_config.NEWS_QUERIES
        self.keywords = data_config.NEWS_KEYWORDS
        self.language = data_config.NEWS_LANGUAGE
        self.page_size = data_config.NEWS_PAGE_SIZE
        
        # CollectAPI ayarları
        self.collectapi_key = data_config.COLLECTAPI_KEY
        self.collectapi_base_url = data_config.COLLECTAPI_BASE_URL
        self.collectapi_tags = data_config.COLLECTAPI_NEWS_TAGS
        
    def fetch_news(self, days_back: int = 7) -> List[Dict]:
        """Tüm query'ler için haberleri çek"""
        all_articles = []
        seen_hashes = set()
        
        for query in self.queries:
            try:
                articles = self._fetch_query_articles(query, days_back)
                
                # Deduplication
                for article in articles:
                    article_hash = hashlib.md5(
                        f"{article.get('title', '')}{article.get('publishedAt', '')}".encode()
                    ).hexdigest()
                    
                    if article_hash not in seen_hashes:
                        seen_hashes.add(article_hash)
                        all_articles.append(article)
                
                logger.info(f"Query '{query}': {len(articles)} haber çekildi")
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Query '{query}' hatası: {e}")
                continue
        
        logger.info(f"Toplam {len(all_articles)} benzersiz haber çekildi")
        return all_articles
    
    def fetch_collectapi_news(self, tag: str = "economy") -> List[Dict]:
        """CollectAPI'den Türk haber kaynaklarından haber çek
        
        Tag options: economy, politics, world, business, sport, technology, health, entertainment
        """
        conn = None
        try:
            conn = http.client.HTTPSConnection(self.collectapi_base_url.replace("https://", ""))
            
            headers = {
                'content-type': "application/json",
                'authorization': f"apikey {self.collectapi_key}"
            }
            
            endpoint = f"/news/getNews?country=tr&tag={tag}"
            conn.request("GET", endpoint, headers=headers)
            
            res = conn.getresponse()
            data = res.read()
            
            if res.status == 200:
                result = json.loads(data.decode("utf-8"))
                if result.get("success"):
                    articles = result.get("result", [])
                    logger.info(f"CollectAPI {tag}: {len(articles)} haber çekildi")
                    return articles
                else:
                    logger.warning(f"CollectAPI başarısız: {result}")
                    return []
            else:
                logger.error(f"CollectAPI HTTP {res.status}")
                return []
                
        except Exception as e:
            logger.error(f"CollectAPI hatası: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def fetch_all_collectapi_news(self) -> List[Dict]:
        """Tüm CollectAPI tag'lerinden haber çek"""
        all_articles = []
        seen_hashes = set()
        
        for tag in self.collectapi_tags:
            articles = self.fetch_collectapi_news(tag)
            for article in articles:
                article_hash = hashlib.md5(
                    (article.get("url", "") + article.get("name", "")).encode()
                ).hexdigest()
                
                if article_hash not in seen_hashes:
                    seen_hashes.add(article_hash)
                    # CollectAPI formatını standardize et
                    standardized = {
                        "title": article.get("name", ""),
                        "description": article.get("description", ""),
                        "url": article.get("url", ""),
                        "source": {"name": article.get("source", "CollectAPI")},
                        "publishedAt": article.get("date", ""),
                        "urlToImage": article.get("image", ""),
                        "provider": "collectapi",
                        "tag": tag
                    }
                    all_articles.append(standardized)
            
            time.sleep(0.5)  # Rate limiting
        
        logger.info(f"CollectAPI toplam: {len(all_articles)} benzersiz haber")
        return all_articles
    
    def _fetch_query_articles(self, query: str, days_back: int) -> List[Dict]:
        """Tek bir query için haber çek"""
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        # Query ve keywords kombinasyonu
        full_query = f"({query}) AND ({self.keywords})"
        
        params = {
            "q": full_query,
            "language": self.language,
            "from": from_date,
            "sortBy": "relevancy",
            "pageSize": self.page_size,
            "apiKey": self.api_key,
        }
        
        response = requests.get(f"{self.base_url}/everything", params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("articles", [])
        elif response.status_code == 429:
            logger.warning("NewsAPI rate limit - beklemede...")
            time.sleep(60)
            return []
        else:
            logger.error(f"NewsAPI error: {response.status_code} - {response.text}")
            return []
    
    def save_to_db(self, articles: List[Dict], date: Optional[str] = None) -> int:
        """Haberleri veritabanına kaydet"""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        saved_count = 0
        
        for article in articles:
            try:
                # Zaten var mı kontrol et
                cursor.execute("""
                    SELECT id FROM news_sentiment 
                    WHERE title = %s AND source = %s
                """, (article.get("title"), article.get("source", {}).get("name")))
                
                if cursor.fetchone():
                    continue
                
                cursor.execute("""
                    INSERT INTO news_sentiment 
                    (date, source, title, summary, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    date,
                    article.get("source", {}).get("name", "Unknown"),
                    article.get("title"),
                    article.get("description") or article.get("content"),
                    datetime.now()
                ))
                saved_count += 1
                
            except Exception as e:
                logger.debug(f"Haber kaydetme hatası: {e}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"{saved_count} yeni haber veritabanına kaydedildi")
        return saved_count


class GoldRelevanceFilter:
    """LLM kullanarak haberlerin altın ile ilgisini kontrol eder"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
        
    def filter_articles(self, articles: List[Dict], batch_size: int = 10) -> List[Dict]:
        """Haberleri LLM ile filtrele - sadece altınla ilgili olanları döndür"""
        relevant_articles = []
        
        prompt_template = """Sen bir finans ve emtia uzmanısın. Aşağıdaki haberin ALTIN FİYATINI etkileyip etkilemeyeceğini değerlendir.

Haber Başlığı: {title}
Haber Özeti: {description}

Yanıtını şu JSON formatında ver (sadece JSON, açıklama yok):
{{
    "is_relevant": true/false,
    "impact_direction": "positive/negative/neutral",
    "reason": "kısa açıklama (1-2 cümle)",
    "confidence": 0.0-1.0 arası güven skoru
}}

Altın fiyatını etkileyebilecek faktörler:
- Merkez bankası politikaları (faiz kararları, para politikası)
- Enflasyon verileri
- Jeopolitik riskler (savaş, çatışma, gerilimler)
- Ekonomik veriler (GDP, istihdam, ticaret)
- Doların değeri
- Petrol ve emtia fiyatları
- Türkiye ekonomik/siyasi durumu
- Küresel piyasa hareketleri

DOĞRUDAN altın ile ilgili olmasa bile dolaylı olarak etkileyebilecek haberleri de "relevant" işaretle."""
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i+batch_size]
            
            for article in batch:
                title = article.get("title", "")
                description = article.get("description") or article.get("content", "") or ""
                
                if not title:
                    continue
                
                try:
                    result = self._check_relevance(title, description, prompt_template)
                    
                    if result and result.get("is_relevant"):
                        article["gold_relevance"] = True
                        article["impact_direction"] = result.get("impact_direction", "neutral")
                        article["impact_reason"] = result.get("reason", "")
                        article["relevance_confidence"] = result.get("confidence", 0.5)
                        relevant_articles.append(article)
                        
                except Exception as e:
                    logger.debug(f"LLM filtreleme hatası: {e}")
                    # Hata durumunda conservative yaklaşım - dahil etme
                    continue
            
            logger.info(f"Filtreleme: {len(relevant_articles)}/{len(articles)} haber altın ile ilgili")
            time.sleep(1)  # Rate limiting
        
        return relevant_articles
    
    def _check_relevance(self, title: str, description: str, prompt: str) -> Optional[Dict]:
        """LLM'e tek haber için relevance kontrolü"""
        formatted_prompt = prompt.format(title=title, description=description[:500])
        
        response = self.llm.generate(formatted_prompt)
        
        if not response:
            return None
        
        # JSON parse et
        import json
        try:
            # JSON bloğunu çıkar
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            return json.loads(response.strip())
        except json.JSONDecodeError:
            logger.debug(f"JSON parse hatası: {response[:100]}")
            return None


def run_news_pipeline(llm_client=None, days_back: int = 3) -> Tuple[int, int]:
    """Haber çekme ve filtreleme pipeline'ını çalıştır
    
    Returns:
        (toplam_haber_sayisi, ilgili_haber_sayisi)
    """
    logger.info("=== Haber Pipeline Başladı ===")
    
    # 1. Haberleri çek
    fetcher = NewsFetcher()
    
    if not fetcher.api_key:
        logger.error("NEWS_API_KEY bulunamadı!")
        return 0, 0
    
    articles = fetcher.fetch_news(days_back=days_back)
    
    if not articles:
        logger.warning("Haber çekilemedi")
        return 0, 0
    
    # 2. Veritabanına kaydet (ham halleriyle)
    saved = fetcher.save_to_db(articles)
    
    # 3. LLM ile filtrele
    if llm_client:
        filter_obj = GoldRelevanceFilter(llm_client)
        relevant = filter_obj.filter_articles(articles)
        logger.info(f"LLM filtreleme: {len(relevant)}/{len(articles)} haber ilgili")
    else:
        relevant = articles
        logger.info("LLM yok - tüm haberler dahil")
    
    # 4. İlgili haberleri güncelle
    conn = get_db_connection()
    cursor = conn.cursor()
    
    for article in relevant:
        try:
            cursor.execute("""
                UPDATE news_sentiment 
                SET relevance_score = %s
                WHERE title = %s AND source = %s
            """, (
                article.get("relevance_confidence", 0.5),
                article.get("title"),
                article.get("source", {}).get("name")
            ))
        except Exception as e:
            logger.debug(f"Güncelleme hatası: {e}")
    
    conn.commit()
    cursor.close()
    conn.close()
    
    logger.info(f"=== Haber Pipeline Tamamlandı: {saved} kaydedildi, {len(relevant)} ilgili ===")
    return len(articles), len(relevant)


if __name__ == "__main__":
    # Test
    from src.llm.assistant import GeminiAssistant
    
    llm = GeminiAssistant()
    total, relevant = run_news_pipeline(llm, days_back=3)
    print(f"\nSonuç: {relevant}/{total} haber altın ile ilgili bulundu")