from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "bots" / "gold_bot.py"), run_name="__main__")
