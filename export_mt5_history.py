from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "export_mt5_history.py"), run_name="__main__")
