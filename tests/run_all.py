"""Run every test_* function without pytest:  python tests/run_all.py"""
import importlib.util
import pathlib
import sys
import traceback

ok = True
for f in sorted(pathlib.Path(__file__).parent.glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in [n for n in dir(mod) if n.startswith("test_")]:
        try:
            getattr(mod, name)()
            print(f"PASS {f.stem}.{name}")
        except Exception:
            ok = False
            print(f"FAIL {f.stem}.{name}")
            traceback.print_exc()
sys.exit(0 if ok else 1)
