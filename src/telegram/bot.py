"""Telegram Bot - Günlük altın sinyalleri ve sohbet"""

import os
import logging
from datetime import datetime
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Bot token
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USERS = os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",")


def is_allowed(user_id: int) -> bool:
    """Kullanıcı yetkilendirme"""
    if not ALLOWED_USERS or ALLOWED_USERS == [""]:
        return True  # Herkes kullanabilir
    return str(user_id) in ALLOWED_USERS


class GoldTelegramBot:
    """Telegram bot interface"""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.app = None

    async def start(self):
        """Bot'u başlat"""
        if not TELEGRAM_BOT_TOKEN:
            logger.error("TELEGRAM_BOT_TOKEN not set!")
            return

        self.app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

        # Komutlar
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("signal", self.cmd_signal))
        self.app.add_handler(CommandHandler("price", self.cmd_price))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("portfolio", self.cmd_portfolio))
        self.app.add_handler(CommandHandler("report", self.cmd_report))
        self.app.add_handler(CommandHandler("backtest", self.cmd_backtest))

        # Mesajlar
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

        logger.info("Telegram bot başlatılıyor...")
        await self.app.run_polling(allowed_updates=Update.ALL_TYPES)

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Hoşgeldin mesajı"""
        if not is_allowed(update.effective_user.id):
            await update.message.reply_text("⛔ Yetkiniz yok.")
            return

        await update.message.reply_text(
            "🏆 *QAU Altın Asistanı'na Hoşgeldiniz!*\n\n"
            "📊 *Komutlar:*\n"
            "/signal - Günlük altın sinyali\n"
            "/price - Güncel fiyat\n"
            "/report - Detaylı analiz raporu\n"
            "/portfolio - Portföy durumu\n"
            "/backtest - Backtest sonuçları\n"
            "/status - Sistem durumu\n"
            "/help - Yardım\n\n"
            "💬 Sorularınızı direkt yazabilirsiniz!",
            parse_mode="Markdown",
        )

    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Yardım"""
        if not is_allowed(update.effective_user.id):
            await update.message.reply_text("⛔ Yetkiniz yok.")
            return

        await update.message.reply_text(
            "📖 *QAU Yardım*\n\n"
            "*Tahmin Modelleri:*\n"
            "• XGBoost (Classification) - Yön tahmini\n"
            "• ARIMA - Zaman serisi\n\n"
            "*Veri Kaynakları:*\n"
            "• XAU/USD (Altın/Dolar)\n"
            "• USD/TRY (Dolar/TL)\n"
            "• Brent Oil\n"
            "• Türkiye Altın Fiyatları\n"
            "• Ekonomik Haberler\n\n"
            "*Sinyal Gücü:*\n"
            "• 🟢 %70+ = Güçlü sinyal\n"
            "• 🟡 %50-70 = Orta sinyal\n"
            "• 🔴 %50- = Zayıf sinyal",
            parse_mode="Markdown",
        )

    async def cmd_signal(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Günlük sinyal"""
        if not is_allowed(update.effective_user.id):
            await update.message.reply_text("⛔ Yetkiniz yok.")
            return

        await update.message.reply_text("📡 *Analiz yapılıyor...*", parse_mode="Markdown")

        try:
            result = self.pipeline.run_daily()
            signal = result["signal"]
            pred = result["predictions"]

            direction = pred.get("xgboost_direction", "NEUTRAL")
            confidence = pred.get("xgboost_confidence", 0)
            current_price = result["current_price"]

            # Emoji seçimi
            if signal["signal"] == "BUY":
                emoji = "🟢"
                signal_text = "AL"
            elif signal["signal"] == "SELL":
                emoji = "🔴"
                signal_text = "SAT"
            else:
                emoji = "🟡"
                signal_text = "BEKLE"

            # Güç değerlendirmesi
            if confidence >= 0.7:
                strength = "Güçlü"
            elif confidence >= 0.5:
                strength = "Orta"
            else:
                strength = "Zayıf"

            msg = (
                f"{emoji} *Günlük Altın Sinyali*\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"📌 Sinyal: *{signal_text}*\n"
                f"📊 Yön: *{direction}*\n"
                f"🎯 Güven: *{confidence:.0%}* ({strength})\n"
                f"💰 Fiyat: *₺{current_price:,.2f}*\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"📝 {signal.get('reasoning', 'Analiz tamamlandı.')[:200]}"
            )

            await update.message.reply_text(msg, parse_mode="Markdown")

        except Exception as e:
            logger.error(f"Signal error: {e}")
            await update.message.reply_text(f"❌ Hata: {str(e)}")

    async def cmd_price(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Güncel fiyat"""
        if not is_allowed(update.effective_user.id):
            await update.message.reply_text("⛔ Yetkiniz yok.")
            return

        try:
            result = self.pipeline.run_daily()
            current = result["current_price"]

            await update.message.reply_text(
                f"💰 *Gram Altın*\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📍 Güncel: *₺{current:,.2f}*\n"
                f"🕐 Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
                parse_mode="Markdown",
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Hata: {str(e)}")

    async def cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Sistem durumu"""
        if not is_allowed(update.effective_user.id):
            await update.message.reply_text("⛔ Yetkiniz yok.")
            return

        try:
            status = self.pipeline.get_status()

            models = "\n".join(
                f"  • {name}: {'✅' if fitted else '❌'}"
                for name, fitted in status["models_trained"].items()
            )

            portfolio = status["portfolio"]
            msg = (
                f"🔧 *Sistem Durumu*\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"*Modeller:*\n{models}\n\n"
                f"*Portföy:*\n"
                f"  • Pozisyon: {portfolio.get('position', 'N/A')}\n"
                f"  • Bakiye: ₺{portfolio.get('balance', 0):,.2f}\n"
                f"  • P&L: ₺{portfolio.get('pnl', 0):,.2f}"
            )

            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Hata: {str(e)}")

    async def cmd_portfolio(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Portföy durumu"""
        if not is_allowed(update.effective_user.id):
            await update.message.reply_text("⛔ Yetkiniz yok.")
            return

        try:
            portfolio = self.pipeline.portfolio.get_status()

            position_emoji = "📈" if portfolio.get("position") == "LONG" else "📉" if portfolio.get("position") == "SHORT" else "⏳"

            msg = (
                f"💼 *Portföy Durumu*\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"{position_emoji} Pozisyon: *{portfolio.get('position', 'YOK')}*\n"
                f"💰 Bakiye: *₺{portfolio.get('balance', 0):,.2f}*\n"
                f"📊 P&L: *₺{portfolio.get('pnl', 0):,.2f}*\n"
                f"📈 Toplam İşlem: *{portfolio.get('total_trades', 0)}*"
            )

            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Hata: {str(e)}")

    async def cmd_report(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Detaylı rapor"""
        if not is_allowed(update.effective_user.id):
            await update.message.reply_text("⛔ Yetkiniz yok.")
            return

        await update.message.reply_text("📊 *Rapor hazırlanıyor...*", parse_mode="Markdown")

        try:
            result = self.pipeline.run_daily()
            analysis = result.get("analysis", "Analiz yok")
            sentiment = result.get("sentiment", {})

            msg = (
                f"📊 *QAU Günlük Rapor*\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"{analysis[:1000]}\n\n"
                f"*Sentiment:* {sentiment.get('overall', 'N/A')}"
            )

            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Hata: {str(e)}")

    async def cmd_backtest(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Backtest sonuçları"""
        if not is_allowed(update.effective_user.id):
            await update.message.reply_text("⛔ Yetkiniz yok.")
            return

        await update.message.reply_text("🔄 *Backtest çalıştırılıyor...*", parse_mode="Markdown")

        try:
            result = self.pipeline.run_backtest(days=180)
            
            # Format results
            lines = ["📈 *Backtest Sonuçları (180 gün)*\n━━━━━━━━━━━━━━━━━━━━"]
            for strategy, metrics in result.items():
                total_return = metrics.get("total_return", 0) * 100
                sharpe = metrics.get("sharpe_ratio", 0)
                trades = metrics.get("num_trades", 0)
                lines.append(
                    f"*{strategy}:*\n"
                    f"  Return: {total_return:.1f}%\n"
                    f"  Sharpe: {sharpe:.2f}\n"
                    f"  İşlem: {trades}"
                )

            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Hata: {str(e)}")

    async def handle_message(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Serbest mesajlara cevap"""
        if not is_allowed(update.effective_user.id):
            await update.message.reply_text("⛔ Yetkiniz yok.")
            return

        text = update.message.text.lower()

        # Basit keyword yanıtları
        if any(k in text for k in ["merhaba", "selam", "selamlar"]):
            await update.message.reply_text(
                "Merhaba! 👋 Size nasıl yardımcı olabilirim?\n"
                "/signal - Günlük altın sinyali\n"
                "/price - Güncel fiyat\n"
                "/help - Tüm komutlar"
            )
        elif any(k in text for k in ["altın", "gram", "gümüş", "yatırım"]):
            await self.cmd_signal(update, ctx)
        elif any(k in text for k in ["ne haber", "nasılsın"]):
            await update.message.reply_text(
                "İyiyim teşekkürler! 😊\nBugünkü altın sinyali için /signal yazabilirsiniz."
            )
        else:
            await update.message.reply_text(
                "🤔 Bunu anlayamadım.\n"
                "/help yazarak tüm komutları görebilirsiniz."
            )


async def run_bot(pipeline):
    """Bot'u çalıştır"""
    bot = GoldTelegramBot(pipeline)
    await bot.start()