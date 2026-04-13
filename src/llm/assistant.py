"""QAU - LLM Investment Assistant

GPT-4 ile altın analizi, yatırım önerisi, risk değerlendirmesi.
"""

import json
from datetime import datetime

import pandas as pd
from loguru import logger

from src.config import settings


class GoldAssistant:
    """LLM destekli altın yatırım asistanı"""

    SYSTEM_PROMPT = """Sen QAU - profesyonel bir altın yatırım asistanısın. 
Görevin gram altın fiyatı tahminlerini, piyasa analizini ve yatırım önerilerini sunmak.

Kurallar:
1. Her zaman veriye dayalı analiz yap
2. Riskleri açıkça belirt
3. Kesin garanti verme, olasılık tabanlı konuş
4. Teknik + temel + sentiment analizini birleştir
5. Türkçe yanıt ver
6. Yatırım tavsiyesi değil, analiz sun (disclaimer ekle)

Çıktı formatı:
- 📊 Güncel Durum
- 📈 Teknik Analiz  
- 📰 Sentiment Analizi
- 🎯 Tahmin & Beklenti
- ⚠️ Risk Değerlendirmesi
- 💡 Öneri
"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = None
        self._init_client()

    def _init_client(self):
        """OpenAI client başlat"""
        try:
            from openai import OpenAI
            api_key = settings.OPENAI_API_KEY
            if api_key and api_key != "sk-xxx":
                self.client = OpenAI(api_key=api_key)
                logger.info(f"OpenAI client initialized: {self.model}")
            else:
                logger.warning("OpenAI API key not set, LLM features disabled")
        except ImportError:
            logger.warning("openai not installed")

    def analyze(self, market_data: dict, predictions: dict,
                sentiment: dict = None, portfolio: dict = None) -> str:
        """Tam analiz ve öneri oluştur"""
        if not self.client:
            return self._generate_rule_based_analysis(market_data, predictions, sentiment)

        prompt = self._build_analysis_prompt(market_data, predictions, sentiment, portfolio)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._generate_rule_based_analysis(market_data, predictions, sentiment)

    def _build_analysis_prompt(self, market_data: dict, predictions: dict,
                                sentiment: dict = None, portfolio: dict = None) -> str:
        """Analiz promptu oluştur"""
        prompt = f"""Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}

📊 GÜNCEL PİYASA VERİLERİ:
{json.dumps(market_data, indent=2, ensure_ascii=False, default=str)}

🎯 MODEL TAHMİNLERİ:
{json.dumps(predictions, indent=2, ensure_ascii=False, default=str)}
"""
        if sentiment:
            prompt += f"""
📰 SENTİMENT ANALİZİ:
{json.dumps(sentiment, indent=2, ensure_ascii=False, default=str)}
"""
        if portfolio:
            prompt += f"""
💼 MEVCUT PORTFÖY:
{json.dumps(portfolio, indent=2, ensure_ascii=False, default=str)}
"""
        prompt += """
Yukarıdaki verileri analiz ederek günlük altın yatırım raporu oluştur.
Tahminlerin güvenilirlik seviyesini belirt.
"""
        return prompt

    def _generate_rule_based_analysis(self, market_data: dict, predictions: dict,
                                       sentiment: dict = None) -> str:
        """Kural tabanlı analiz (LLM yoksa)"""
        current = market_data.get("current_price", 0)
        pred = predictions.get("ensemble_prediction", 0)
        change = ((pred - current) / current * 100) if current > 0 else 0

        direction = "📈 YÜKSELİŞ" if change > 0 else "📉 DÜŞÜŞ"
        strength = "güçlü" if abs(change) > 1 else "zayıf"

        sentiment_text = ""
        if sentiment:
            score = sentiment.get("composite_score", 0)
            sentiment_text = f"""
📰 **Sentiment Analizi:**
- Haber Skoru: {score:.2f} ({'pozitif' if score > 0 else 'negatif'})
- Haber Sayısı: {sentiment.get('news_count', 0)}
"""

        return f"""
🏛️ **QAU Günlük Altın Analizi**
📅 {datetime.now().strftime('%d.%m.%Y')}

📊 **Güncel Durum:**
- Gram Altın: ₺{current:,.2f}
- Tahmin: ₺{pred:,.2f} ({change:+.2f}%)

📈 **Teknik Görünüm:**
- Beklenti: {direction} ({strength})
- Değişim: {change:+.2f}%
{sentiment_text}
🎯 **Tahmin:**
- Yarın: ₺{pred:,.2f}
- yön: {direction}

⚠️ **Risk:** Orta-Yüksek
💡 **Öneri:** {'Dikkatli alım fırsatı' if change > 0.5 else 'Bekleme pozisyonu' if abs(change) < 0.3 else 'Kâr realizede düşünebilir'}

---
⚠️ Bu analiz yatırım tavsiyesi değildir. Yatırım kararlarıınızı kendi araştırmanıza dayandırın.
"""

    def generate_signal(self, predictions: dict, sentiment: dict = None) -> dict:
        """Trading sinyali oluştur"""
        pred = predictions.get("ensemble_prediction", 0)
        current = predictions.get("current_price", 0)

        if current == 0:
            return {"signal": "HOLD", "confidence": 0, "reason": "Veri yok"}

        change_pct = (pred - current) / current * 100
        confidence = min(abs(change_pct) * 20, 95)  # Basit güven skoru

        if sentiment:
            sentiment_score = sentiment.get("composite_score", 0)
            if sentiment_score > 0.3:
                confidence = min(confidence + 10, 95)
            elif sentiment_score < -0.3:
                confidence = min(confidence + 10, 95)

        if change_pct > 0.5:
            signal = "BUY"
            reason = f"Tahmin %{change_pct:+.2f} artış"
        elif change_pct < -0.5:
            signal = "SELL"
            reason = f"Tahmin %{change_pct:+.2f} düşüş"
        else:
            signal = "HOLD"
            reason = f"Tahmin %{change_pct:+.2f} (belirsiz)"
            confidence *= 0.5

        return {
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "predicted_price": pred,
            "current_price": current,
            "change_pct": change_pct,
        }