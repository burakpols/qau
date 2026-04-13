"""QAU - Gram Altın Tahmin & Yatırım Asistanı

CLI entry point.
"""

import argparse
import sys

from loguru import logger


def setup_logging(level: str = "INFO"):
    logger.remove()
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss}|{level}|{message}")
    logger.add("logs/qau_{time:YYYY-MM-DD}.log", level="DEBUG", rotation="1 day")


def cmd_init(args):
    """Veritabanı oluştur + veri çek"""
    from src.pipeline import pipeline
    setup_logging(args.log)

    logger.info("QAU Initializing...")
    df = pipeline.initialize()
    logger.info(f"✅ {len(df)} satır veri yüklendi, {len(df.columns)} feature")


def cmd_train(args):
    """Modelleri eğit"""
    from src.pipeline import pipeline
    setup_logging(args.log)

    logger.info("Training models...")
    metrics = pipeline.train_models(days=args.days)
    for name, m in metrics.items():
        logger.info(f"  {name}: {m}")
    logger.info("✅ Modeller eğitildi")


def cmd_daily(args):
    """Günlük analiz çalıştır"""
    from src.pipeline import pipeline
    setup_logging(args.log)

    # Model eğitimi yap (yoksa)
    if not pipeline.ensemble.is_fitted:
        logger.info("Models not trained, initializing...")
        pipeline.initialize()
        pipeline.train_models(days=365)
    
    result = pipeline.run_daily()
    print("\n" + "=" * 60)
    print(result.get("analysis", "Analiz oluşturulamadı"))
    print("=" * 60)
    print(pipeline.portfolio.get_report())


def cmd_backtest(args):
    """Backtest çalıştır"""
    from src.pipeline import pipeline
    setup_logging(args.log)

    pipeline.initialize()
    pipeline.train_models(days=args.train_days)
    result = pipeline.run_backtest(days=args.days)
    print("\n📊 Backtest Sonuçları:")
    for key, val in result.items():
        print(f"  {key}: {val}")


def cmd_status(args):
    """Sistem durumu"""
    from src.pipeline import pipeline
    setup_logging(args.log)

    status = pipeline.get_status()
    print("\n🔧 QAU Sistem Durumu:")
    print(f"  Modeller: {status['models_trained']}")
    print(f"  Ensemble: {'✅' if status['ensemble_fitted'] else '❌'}")
    if status['last_predictions']:
        print(f"  Son Tahmin: {status['last_predictions']}")
    print(f"  Portföy: {status['portfolio']}")


def main():
    parser = argparse.ArgumentParser(description="QAU - Gram Altın Tahmin Asistanı")
    parser.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    subparsers = parser.add_subparsers(dest="command")

    # init
    p_init = subparsers.add_parser("init", help="Veritabanı oluştur + veri çek")
    p_init.set_defaults(func=cmd_init)

    # train
    p_train = subparsers.add_parser("train", help="Modelleri eğit")
    p_train.add_argument("--days", type=int, default=365, help="Eğitim gün sayısı")
    p_train.set_defaults(func=cmd_train)

    # daily
    p_daily = subparsers.add_parser("daily", help="Günlük analiz")
    p_daily.set_defaults(func=cmd_daily)

    # backtest
    p_bt = subparsers.add_parser("backtest", help="Backtest")
    p_bt.add_argument("--days", type=int, default=180, help="Backtest gün sayısı")
    p_bt.add_argument("--train-days", type=int, default=365, help="Eğitim gün sayısı")
    p_bt.set_defaults(func=cmd_backtest)

    # status
    p_status = subparsers.add_parser("status", help="Sistem durumu")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()