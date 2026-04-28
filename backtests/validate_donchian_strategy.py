from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_multi_tf_lorentzian_andean as bt

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
REPORT_DIR = ROOT_DIR / "reports"


def fixed_config() -> bt.Config:
    return bt.Config(
        m1_file=str(DATA_DIR / "bitcoin_365d_1m.csv"),
        m5_file=str(DATA_DIR / "bitcoin_365d_5m.csv"),
        m15_file=str(DATA_DIR / "bitcoin_365d_15m.csv"),
        h1_file=str(DATA_DIR / "bitcoin_365d_1h.csv"),
        ema_fast=50,
        ema_slow=200,
        lorentzian_k=8,
        lorentzian_horizon=8,
        andean_length=50,
        min_atr_pct=0.002,
        risk_per_trade=0.005,
        rr=2.0,
        exit_mode="fixed",
        m15_adx_min=20,
        require_m15_adx_rising=True,
        use_donchian=True,
        donchian_n=20,
        regime_name="donchian_20",
    )


def prepare_data(config: bt.Config):
    m1 = bt.load_csv(config.m1_file)
    m5 = bt.load_csv(config.m5_file)
    m15 = bt.load_csv(config.m15_file)
    h1 = bt.load_csv(config.h1_file)
    m1, m5, m15 = bt.calculate_indicators(m1, m5, m15, config)
    h1 = bt.calculate_h1_indicators(h1)
    m5 = bt.align_timeframes(m5, m15)
    m5 = bt.align_h1_timeframe(m5, h1)
    m5 = bt.calculate_lorentzian(m5, config)
    return m1, m5


def period_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return max((end.date() - start.date()).days + 1, 1)


def run_period(m1: pd.DataFrame, m5: pd.DataFrame, config: bt.Config, start, end, label: str):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    cfg = bt.Config(**asdict(config))
    cfg.total_days = period_days(start, end)
    m5_period = m5[(m5.index >= start) & (m5.index <= end)]
    m1_period = m1[(m1.index >= start) & (m1.index <= end + pd.Timedelta(minutes=cfg.max_m1_confirm_candles))]
    if m5_period.empty or m1_period.empty:
        return empty_metrics(cfg, label, start, end)
    candidates = bt.signal_candidates(m5_period, cfg)
    if candidates.empty:
        metrics = empty_metrics(cfg, label, start, end)
        metrics["signal_candidates"] = 0
        return metrics
    metrics, _ = bt.run_backtest(m1_period, candidates, cfg, write_logs=False)
    metrics["label"] = label
    metrics["start"] = start
    metrics["end"] = end
    metrics["signal_candidates"] = len(candidates)
    return metrics


def empty_metrics(config: bt.Config, label: str, start, end):
    return {
        "label": label,
        "start": start,
        "end": end,
        "profit_pct": 0.0,
        "final_balance": config.initial_balance,
        "total_trades": 0,
        "trades_per_day": 0.0,
        "winrate": 0.0,
        "profit_factor": 0.0,
        "max_drawdown_pct": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "expectancy": 0.0,
        "negative_months": 0,
        "signal_candidates": 0,
        "rr": config.rr,
        "fee_rate": config.fee_rate,
        "slippage_atr_mult": config.slippage_atr_mult,
        "donchian_n": config.donchian_n,
        "m15_adx_min": config.m15_adx_min,
    }


def out_of_sample_test(m1: pd.DataFrame, m5: pd.DataFrame, config: bt.Config):
    split_idx = int(len(m5) * 0.70)
    split_time = m5.index[split_idx]
    rows = [
        run_period(m1, m5, config, m5.index[0], m5.index[split_idx - 1], "in_sample_70pct"),
        run_period(m1, m5, config, split_time, m5.index[-1], "out_of_sample_30pct"),
    ]
    return pd.DataFrame(rows)


def walk_forward_test(m1: pd.DataFrame, m5: pd.DataFrame, config: bt.Config):
    months = pd.period_range(m5.index.min().to_period("M"), m5.index.max().to_period("M"), freq="M")
    rows = []
    for i in range(3, len(months)):
        train_start = months[i - 3].start_time
        train_end = months[i - 1].end_time
        test_start = months[i].start_time
        test_end = min(months[i].end_time, m5.index.max())
        row = run_period(m1, m5, config, test_start, test_end, str(months[i]))
        row["train_start"] = train_start
        row["train_end"] = train_end
        row["test_month"] = str(months[i])
        rows.append(row)
    return pd.DataFrame(rows)


def stress_test(m1: pd.DataFrame, m5: pd.DataFrame, config: bt.Config):
    rows = []
    for multiplier in [1, 2, 3]:
        cfg = bt.Config(**asdict(config))
        cfg.fee_rate *= multiplier
        cfg.slippage_atr_mult *= multiplier
        row = run_period(m1, m5, cfg, m5.index[0], m5.index[-1], f"{multiplier}x_cost")
        row["cost_multiplier"] = multiplier
        rows.append(row)
    return pd.DataFrame(rows)


def stability_test(m1: pd.DataFrame, m5: pd.DataFrame, config: bt.Config):
    rows = []
    for donchian_n in [15, 20, 25, 30]:
        for adx_min in [18, 20, 22, 25]:
            cfg = bt.Config(**asdict(config))
            cfg.donchian_n = donchian_n
            cfg.m15_adx_min = adx_min
            row = run_period(m1, m5, cfg, m5.index[0], m5.index[-1], f"donchian_{donchian_n}_adx_{adx_min}")
            row["donchian_n"] = donchian_n
            row["m15_adx_min"] = adx_min
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    config = fixed_config()
    m1, m5 = prepare_data(config)

    outputs = {
        str(REPORT_DIR / "out_of_sample_results.csv"): out_of_sample_test(m1, m5, config),
        str(REPORT_DIR / "walk_forward_results.csv"): walk_forward_test(m1, m5, config),
        str(REPORT_DIR / "stress_test_results.csv"): stress_test(m1, m5, config),
        str(REPORT_DIR / "parameter_stability_results.csv"): stability_test(m1, m5, config),
    }
    for path, df in outputs.items():
        df.to_csv(path, index=False)
        print(f"\n{path}")
        display_cols = [
            col
            for col in [
                "label",
                "test_month",
                "cost_multiplier",
                "donchian_n",
                "m15_adx_min",
                "profit_pct",
                "total_trades",
                "trades_per_day",
                "winrate",
                "profit_factor",
                "max_drawdown_pct",
                "expectancy",
                "negative_months",
            ]
            if col in df.columns
        ]
        print(df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
