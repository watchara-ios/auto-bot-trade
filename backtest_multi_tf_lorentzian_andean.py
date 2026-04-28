from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "backtests" / "backtest_multi_tf_lorentzian_andean.py"), run_name="__main__")
