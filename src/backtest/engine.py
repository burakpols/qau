"""QAU - Backtest Engine

Walk-forward backtest, işlem maliyetleri, Sharpe/Sortino/CALMAR metrikleri.
"""

import numpy as np
import pandas as pd
from loguru import logger


class BacktestEngine:
    """Altın trading stratejisi backtest motoru"""

    def __init__(self, initial_capital: float = 100_000,
                 commission_rate: float = 0.001,
                 slippage: float = 0.0005,
                 risk_free_rate: float = 0.15):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate  # Yıllık risksiz getiri (TL)

    def run(self, prices: pd.Series, signals: pd.Series,
            position_size: float = 0.9) -> dict:
        """Backtest çalıştır

        Args:
            prices: Günlük kapanış fiyatları
            signals: 1=AL, -1=SAT, 0=Bekle
            position_size: Her işlemde kullanılacak sermaye oranı
        """
        # Align
        common_idx = prices.index.intersection(signals.index)
        prices = prices.loc[common_idx]
        signals = signals.loc[common_idx]

        capital = self.initial_capital
        position = 0.0  # Gram altın miktarı
        trades = []
        equity_curve = []
        daily_returns = []

        for i in range(1, len(prices)):
            price = prices.iloc[i]
            prev_price = prices.iloc[i-1]
            signal = signals.iloc[i-1]  # Önceki günün sinyali

            # Mevcut pozisyonun P&L
            if position > 0:
                daily_pnl = position * (price - prev_price)
            else:
                daily_pnl = 0

            # Sinyal işle
            if signal == 1 and position == 0:  # AL
                # Slippage dahil alış fiyatı
                buy_price = price * (1 + self.slippage)
                amount = (capital * position_size) / buy_price
                commission = amount * buy_price * self.commission_rate
                position = amount - (commission / buy_price)
                capital -= (amount * buy_price + commission)
                trades.append({
                    "date": prices.index[i], "type": "BUY",
                    "price": buy_price, "amount": position,
                    "commission": commission,
                })

            elif signal == -1 and position > 0:  # SAT
                sell_price = price * (1 - self.slippage)
                commission = position * sell_price * self.commission_rate
                capital += (position * sell_price - commission)
                trades.append({
                    "date": prices.index[i], "type": "SELL",
                    "price": sell_price, "amount": position,
                    "commission": commission,
                })
                position = 0

            # Equity
            equity = capital + (position * price)
            equity_curve.append(equity)
            daily_ret = (equity_curve[-1] / equity_curve[-2] - 1) if len(equity_curve) > 1 else 0
            daily_returns.append(daily_ret)

        # Son equity
        final_equity = capital + (position * prices.iloc[-1])

        # Metrikler
        metrics = self._calculate_metrics(
            np.array(daily_returns), final_equity, trades
        )
        metrics["trades"] = trades
        metrics["equity_curve"] = equity_curve
        metrics["final_equity"] = final_equity
        metrics["total_return"] = (final_equity / self.initial_capital - 1) * 100

        return metrics

    def _calculate_metrics(self, daily_returns: np.ndarray,
                           final_equity: float, trades: list) -> dict:
        """Performans metrikleri hesapla"""
        if len(daily_returns) == 0:
            return {}

        # Yıllık getiri
        total_days = len(daily_returns)
        annual_return = ((final_equity / self.initial_capital) ** (252 / max(total_days, 1)) - 1)

        # Volatilite
        annual_vol = np.std(daily_returns) * np.sqrt(252)

        # Sharpe Ratio
        daily_rf = self.risk_free_rate / 252
        excess_returns = daily_returns - daily_rf
        sharpe = np.mean(excess_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)

        # Sortino Ratio
        downside = daily_returns[daily_returns < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 1e-8
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)

        # Max Drawdown
        cumulative = np.cumprod(1 + daily_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(np.min(drawdown))

        # Calmar Ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win/Loss
        winning = [t for t in trades if t["type"] == "SELL"]
        # Simplified win rate
        win_trades = 0
        for i in range(0, len(trades) - 1, 2):
            if i + 1 < len(trades):
                if trades[i+1]["price"] > trades[i]["price"]:
                    win_trades += 1
        total_pairs = len(trades) // 2
        win_rate = win_trades / total_pairs * 100 if total_pairs > 0 else 0

        return {
            "annual_return": annual_return * 100,
            "annual_volatility": annual_vol * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_drawdown * 100,
            "total_trades": len(trades),
            "win_rate": win_rate,
            "profit_factor": 0,  # TODO: calculate properly
        }

    def generate_signals_from_predictions(self, actual: pd.Series,
                                          predicted: pd.Series,
                                          threshold: float = 0.005) -> pd.Series:
        """Tahminlerden trading sinyali üret

        threshold: %0.5'den fazla değişim bekleniyorsa işlem yap
        """
        signals = pd.Series(0, index=actual.index)

        for i in range(1, len(predicted)):
            if i < len(actual):
                expected_change = (predicted.iloc[i] - actual.iloc[i-1]) / actual.iloc[i-1]
                if expected_change > threshold:
                    signals.iloc[i] = 1   # AL
                elif expected_change < -threshold:
                    signals.iloc[i] = -1  # SAT
                else:
                    signals.iloc[i] = 0   # Bekle

        return signals

    def compare_strategies(self, prices: pd.Series,
                           strategies: dict[str, pd.Series]) -> pd.DataFrame:
        """Birden fazla stratejiyi karşılaştır"""
        results = {}

        # Buy & Hold
        bh_returns = prices.pct_change().dropna()
        bh_metrics = {
            "total_return": ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100,
            "sharpe": (bh_returns.mean() / (bh_returns.std() + 1e-8)) * np.sqrt(252),
            "max_drawdown": self._max_drawdown(bh_returns),
        }
        results["Buy&Hold"] = bh_metrics

        for name, signals in strategies.items():
            try:
                result = self.run(prices, signals)
                results[name] = {
                    "total_return": result["total_return"],
                    "sharpe": result["sharpe_ratio"],
                    "max_drawdown": result["max_drawdown"],
                    "win_rate": result["win_rate"],
                    "total_trades": result["total_trades"],
                }
            except Exception as e:
                logger.error(f"Strategy {name} backtest failed: {e}")
                results[name] = {"error": str(e)}

        return pd.DataFrame(results).T

    @staticmethod
    def _max_drawdown(returns: pd.Series) -> float:
        """Max drawdown hesapla"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min()) * 100