from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "bots" / "bitcoin_bot_hybrid.py"), run_name="__main__")
