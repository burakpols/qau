"""QAU - Portfolio Manager

Altın portföy yönetimi, pozisyon takibi, risk kontrolü.
"""

from datetime import datetime
from loguru import logger


class PortfolioManager:
    """Altın portföy yöneticisi"""

    def __init__(self, initial_capital: float = 100_000,
                 max_position_pct: float = 0.8,
                 stop_loss_pct: float = 0.03,
                 take_profit_pct: float = 0.06):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.gold_grams = 0.0
        self.avg_buy_price = 0.0
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trades = []
        self.total_commission = 0.0

    @property
    def total_value(self) -> float:
        """Toplam portföy değeri"""
        return self.cash + self.gold_grams * self.current_price

    @property
    def current_price(self) -> float:
        """Son bilinen fiyat"""
        if self.trades:
            return self.trades[-1].get("price", 0)
        return 0

    @property
    def pnl(self) -> float:
        """Toplam kar/zarar"""
        return self.total_value - self.initial_capital

    @property
    def pnl_pct(self) -> float:
        """Kar/zarar yüzdesi"""
        return (self.pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0

    @property
    def position_value(self) -> float:
        """Altın pozisyon değeri"""
        return self.gold_grams * self.current_price

    @property
    def position_pct(self) -> float:
        """Portföyde altın yüzdesi"""
        return (self.position_value / self.total_value * 100) if self.total_value > 0 else 0

    def execute_signal(self, signal: str, price: float,
                       confidence: float = 0.5,
                       commission_rate: float = 0.001) -> dict | None:
        """Sinyali işle"""
        action = None

        if signal == "BUY" and self.gold_grams == 0:
            # Pozisyon boyutu: güven skoruyla ölçekle
            position_size = self.cash * self.max_position_pct * min(confidence / 50, 1.0)
            grams = position_size / price
            commission = position_size * commission_rate
            total_cost = position_size + commission

            if total_cost <= self.cash:
                self.gold_grams += grams
                self.avg_buy_price = (self.avg_buy_price * (self.gold_grams - grams) + price * grams) / self.gold_grams if self.gold_grams > 0 else price
                self.cash -= total_cost
                self.total_commission += commission
                action = "BUY"

                self.trades.append({
                    "date": datetime.now().isoformat(),
                    "action": "BUY", "price": price,
                    "grams": grams, "commission": commission,
                })
                logger.info(f"BUY {grams:.2f}g @ ₺{price:,.2f}")

        elif signal == "SELL" and self.gold_grams > 0:
            # Stop loss / take profit kontrol
            if self._should_stop_loss(price) or self._should_take_profit(price) or confidence > 30:
                commission = self.gold_grams * price * commission_rate
                revenue = self.gold_grams * price - commission
                self.cash += revenue
                self.total_commission += commission

                self.trades.append({
                    "date": datetime.now().isoformat(),
                    "action": "SELL", "price": price,
                    "grams": self.gold_grams, "commission": commission,
                })
                logger.info(f"SELL {self.gold_grams:.2f}g @ ₺{price:,.2f}")
                self.gold_grams = 0
                self.avg_buy_price = 0
                action = "SELL"

        return self.trades[-1] if action else None

    def _should_stop_loss(self, current_price: float) -> bool:
        """Stop loss kontrolü"""
        if self.avg_buy_price > 0:
            loss = (current_price - self.avg_buy_price) / self.avg_buy_price
            return loss < -self.stop_loss_pct
        return False

    def _should_take_profit(self, current_price: float) -> bool:
        """Take profit kontrolü"""
        if self.avg_buy_price > 0:
            gain = (current_price - self.avg_buy_price) / self.avg_buy_price
            return gain > self.take_profit_pct
        return False

    def get_status(self) -> dict:
        """Portföy durumu"""
        return {
            "total_value": self.total_value,
            "cash": self.cash,
            "gold_grams": self.gold_grams,
            "avg_buy_price": self.avg_buy_price,
            "current_price": self.current_price,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "position_pct": self.position_pct,
            "total_trades": len(self.trades),
            "total_commission": self.total_commission,
        }

    def get_report(self) -> str:
        """Okunabilir portföy raporu"""
        status = self.get_status()
        pos_val = status['gold_grams'] * status.get('current_price', 0)
        emoji = "🟢" if status["pnl"] >= 0 else "🔴"

        return f"""
💼 **Portföy Durumu**
{emoji} Toplam: ₺{status['total_value']:,.2f}
💵 Nakit: ₺{status['cash']:,.2f}
🪙 Altın: {status['gold_grams']:.2f}g (₺{pos_val:,.2f})
📊 Ort. Maliyet: ₺{status['avg_buy_price']:,.2f}
{emoji} K/Z: ₺{status['pnl']:,.2f} ({status['pnl_pct']:+.2f}%)
📝 İşlem: {status['total_trades']} | Komisyon: ₺{status['total_commission']:,.2f}
"""
