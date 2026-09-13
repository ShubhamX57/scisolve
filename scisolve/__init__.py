"""scisolve — solve scientific computing problems by running code, not by recalling it."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("scisolve")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

# Executor, ExecResult, Agent and Solution are exported here as the modules land.
__all__ = ["__version__"]
