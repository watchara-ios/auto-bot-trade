from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "bots" / "bitcoin_bot_deepseek.py"), run_name="__main__")
