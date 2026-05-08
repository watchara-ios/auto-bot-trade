from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "live" / "gold_v2_sell_live.py"),
    run_name="__main__",
)
