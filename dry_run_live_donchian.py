from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "bots" / "dry_run_live_donchian.py"), run_name="__main__")
