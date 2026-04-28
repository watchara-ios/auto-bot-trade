from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "load_history_md5.py"), run_name="__main__")
