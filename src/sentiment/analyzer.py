"""QAU - Sentiment Analiz Motoru

Haber başlıkları ve metinlerinden altın piyasası sentiment'i üretir.
Gemini Flash LLM + anahtar kelime bazlı analiz.
"""

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
from loguru import logger

from src.config import api_config


@dataclass
class SentimentResult:
    """Sentiment analiz sonucu"""
    source: str
    title: str
    sentiment_score: float  # -1.0 (çok negatif) → +1.0 (çok pozitif)
    gold_impact: float  # -1.0 (altın düşer) → +1.0 (altın yükselir)
    confidence: float  # 0.0 → 1.0
    summary: str
    date: date


class SentimentAnalyzer:
    """Altın piyasası sentiment analizi"""

    # Altın fiyatını etkileyen anahtar kelimeler
    POSITIVE_KEYWORDS = {
        "enflasyon": 0.6, "fiyat artışı": 0.7, "kur artışı": 0.8,
        "jeopolitik risk": 0.8, "savaş": 0.9, "çatışma": 0.7,
        "merkez bankası alım": 0.9, "altın rezerv": 0.7,
        "güvenli liman": 0.6, "dolar düşüşü": 0.7, "faiz indirim": 0.8,
        "küresel belirsizlik": 0.7, "kriz": 0.8, "devalüasyon": 0.9,
    }

    NEGATIVE_KEYWORDS = {
        "faiz artışı": 0.7, "dolar güçlenmesi": 0.6, "risk iştahı": 0.5,
        "ekonomik iyileşme": 0.4, "altın satışı": 0.6, "merkez bankası satım": 0.8,
        "enflasyon düşüş": 0.5, "barış": 0.6, "anlaşma": 0.5,
    }

    def __init__(self):
        self.llm_available = False
        self._setup_llm()

    def _setup_llm(self):
        """LLM bağlantısını kur"""
        try:
            import google.generativeai as genai
            if api_config.GEMINI_API_KEY:
                genai.configure(api_key=api_config.GEMINI_API_KEY)
                self.model = genai.GenerativeModel("gemini-2.0-flash")
                self.llm_available = True
                logger.info("Gemini Flash connected for sentiment analysis")
            else:
                logger.warning("No GEMINI_API_KEY set. Using keyword-only sentiment.")
        except ImportError:
            logger.warning("google-generativeai not installed. Run: pip install google-generativeai")
        except Exception as e:
            logger.warning(f"LLM setup failed: {e}")

    def analyze_text(self, text: str, source: str = "unknown") -> SentimentResult:
        """Tek bir metni analiz et"""
        if self.llm_available:
            try:
                return self._analyze_with_llm(text, source)
            except Exception as e:
                logger.warning(f"LLM analysis failed, falling back to keywords: {e}")

        return self._analyze_with_keywords(text, source)

    def _analyze_with_llm(self, text: str, source: str) -> SentimentResult:
        """Gemini Flash ile sentiment analizi"""
        prompt = f"""Sen bir altın piyasası uzmanısın. Aşağıdaki haberi analiz et ve Türkiye'de gram altın fiyatına etkisini değerlendir.

HABER: {text}

Yanıtını SADECE şu JSON formatında ver (başka hiçbir şey yazma):
{{
    "sentiment_score": <float -1.0 ile 1.0 arası, haberin genel duygusu>,
    "gold_impact": <float -1.0 ile 1.0 arası, gram altına etkisi, pozitif=altın yükselir>,
    "confidence": <float 0.0 ile 1.0 arası, tahmin güveni>,
    "summary": <string, tek cümlelik özet>
}}"""

        response = self.model.generate_content(prompt)
        import json
        result = json.loads(response.text.strip().replace("```json", "").replace("```", ""))

        return SentimentResult(
            source=source,
            title=text[:100],
            sentiment_score=float(result.get("sentiment_score", 0)),
            gold_impact=float(result.get("gold_impact", 0)),
            confidence=float(result.get("confidence", 0.5)),
            summary=result.get("summary", ""),
            date=date.today(),
        )

    def _analyze_with_keywords(self, text: str, source: str) -> SentimentResult:
        """Anahtar kelime bazlı sentiment analizi (fallback)"""
        text_lower = text.lower()

        positive_score = 0.0
        negative_score = 0.0
        matched_keywords = []

        for keyword, weight in self.POSITIVE_KEYWORDS.items():
            if keyword in text_lower:
                positive_score += weight
                matched_keywords.append(keyword)

        for keyword, weight in self.NEGATIVE_KEYWORDS.items():
            if keyword in text_lower:
                negative_score += weight
                matched_keywords.append(keyword)

        total = positive_score + negative_score
        if total == 0:
            gold_impact = 0.0
            sentiment_score = 0.0
            confidence = 0.2
        else:
            gold_impact = (positive_score - negative_score) / total
            sentiment_score = gold_impact * 0.8
            confidence = min(total / 3.0, 1.0)

        return SentimentResult(
            source=source,
            title=text[:100],
            sentiment_score=round(sentiment_score, 4),
            gold_impact=round(gold_impact, 4),
            confidence=round(confidence, 4),
            summary=f"Eşleşen anahtarlar: {', '.join(matched_keywords[:5]) or 'yok'}",
            date=date.today(),
        )

    def analyze_batch(self, news_items: list[dict]) -> pd.DataFrame:
        """Birden fazla haberi analiz et

        Args:
            news_items: [{"title": ..., "source": ..., "date": ...}, ...]
        """
        results = []
        for item in news_items:
            try:
                result = self.analyze_text(
                    text=item.get("title", ""),
                    source=item.get("source", "unknown"),
                )
                result.date = item.get("date", date.today())
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to analyze: {item.get('title', '')[:50]}: {e}")

        df = pd.DataFrame([{
            "date": r.date,
            "source": r.source,
            "title": r.title,
            "sentiment_score": r.sentiment_score,
            "gold_impact": r.gold_impact,
            "confidence": r.confidence,
            "summary": r.summary,
        } for r in results])

        return df

    def get_daily_sentiment(self, news_items: list[dict]) -> dict:
        """Günlük toplu sentiment skoru üret"""
        if not news_items:
            return {
                "sentiment_score": 0.0,
                "gold_impact": 0.0,
                "confidence": 0.0,
                "news_count": 0,
                "summary": "Haber bulunamadı",
            }

        df = self.analyze_batch(news_items)
        if df.empty:
            return {
                "sentiment_score": 0.0,
                "gold_impact": 0.0,
                "confidence": 0.0,
                "news_count": 0,
                "summary": "Analiz yapılamadı",
            }

        # Güven bazlı ağırlıklı ortalama
        total_confidence = df["confidence"].sum()
        if total_confidence > 0:
            weighted_impact = (df["gold_impact"] * df["confidence"]).sum() / total_confidence
            weighted_sentiment = (df["sentiment_score"] * df["confidence"]).sum() / total_confidence
        else:
            weighted_impact = df["gold_impact"].mean()
            weighted_sentiment = df["sentiment_score"].mean()

        # LLM ile günlük özet
        daily_summary = ""
        if self.llm_available and len(df) > 0:
            try:
                headlines = "\n".join(f"- {t}" for t in df["title"].head(10))
                prompt = f"""Bugünkü altın piyasası haberlerini özetle ve gram altın için beklentini belirt:

{headlines}

Tek cümlelik özet:"""
                response = self.model.generate_content(prompt)
                daily_summary = response.text.strip()
            except:
                daily_summary = f"{len(df)} haber analiz edildi"

        return {
            "sentiment_score": round(weighted_sentiment, 4),
            "gold_impact": round(weighted_impact, 4),
            "confidence": round(float(df["confidence"].mean()), 4),
            "news_count": len(df),
            "summary": daily_summary or f"{len(df)} haber analiz edildi",
        }