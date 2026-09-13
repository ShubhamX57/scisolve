"""scisolve — solve scientific computing problems by running code, not by recalling it."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .executor import ExecResult, Executor

try:
    __version__ = _version("scisolve")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

# Agent and Solution are exported here once agent.py lands (milestone 3).
__all__ = ["ExecResult", "Executor", "__version__"]
