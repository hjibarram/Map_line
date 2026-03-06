from importlib.machinery import SourceFileLoader
from pathlib import Path
import types

_cli_path = Path(__file__).resolve().parents[2] / "bin" / "run_mapline"

module = types.ModuleType("maplines_cli")
loader = SourceFileLoader("maplines_cli", str(_cli_path))
loader.exec_module(module)

cli = module.cli