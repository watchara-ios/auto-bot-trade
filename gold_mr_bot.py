import runpy, sys, os
sys.path.insert(0, os.path.dirname(__file__))
runpy.run_module("bots.gold_mr_bot", run_name="__main__", alter_sys=True)
