"""scisolve — solve scientific computing problems by running code, not by recalling it."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .agent import Agent, Solution, Verification
from .executor import ExecResult, Executor

try:
    __version__ = _version("scisolve")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = [
    "Agent",
    "ExecResult",
    "Executor",
    "Solution",
    "Verification",
    "__version__",
]
