from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "backtests" / "run_forex_donchian.py"), run_name="__main__")
