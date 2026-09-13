"""A persistent Python session for running model-generated code.

THIS IS NOT A SANDBOX. Code passed to :meth:`Executor.run_cell` is executed with
``exec`` in this process, with full access to the filesystem, the network, and
the environment of whoever started scisolve. It is exactly as dangerous as
running a script a stranger emailed you. The persistent namespace is a feature
of the agent loop; isolating it properly is a different project.

Known limits, stated here so nothing downstream has to pretend otherwise:

* The timeout uses ``signal.setitimer(ITIMER_REAL)``. It only works on Unix, on
  the main thread, and it can only interrupt Python bytecode. A long call inside
  compiled code (``np.linalg.eig`` on a huge matrix, say) will not be
  interrupted until it returns to the interpreter.
* Model code can swallow the timeout with a bare ``except Exception``, because
  the alarm surfaces as an ordinary exception inside the cell.
* There is no memory limit. ``setrlimit(RLIMIT_AS)`` would cap the agent process
  itself and turn a runaway cell into a dead process.
* There is no network blocking. Patching ``socket`` in-process is bypassable,
  and a guard that can be stepped around is worse than a documented absence.
"""

from __future__ import annotations

import contextlib
import io
import math
import signal
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must happen before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402

CELL_FILENAME = "<cell>"

#: True when this platform can install a wall-clock alarm at all.
SIGALRM_AVAILABLE = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")

#: Set to True the first time a timeout was requested but could not be installed
#: (Windows, or a non-main thread). Read by the CLI so it can warn.
TIMEOUT_SKIPPED = False

TimeLimit = Callable[[float | None], "contextlib.AbstractContextManager[None]"]


class CellTimeout(Exception):
    """Raised inside a cell when its wall-clock limit expires."""


@dataclass(frozen=True)
class ExecResult:
    """The outcome of one ``run_cell`` call."""

    code: str
    stdout: str
    error: str | None
    figures: tuple[Path, ...]
    duration_s: float
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.error is None and not self.timed_out


@contextlib.contextmanager
def time_limit(seconds: float | None) -> Iterator[None]:
    """Raise :class:`CellTimeout` in the calling thread after ``seconds``.

    Silently does nothing when there is no alarm to install; sets the module
    flag :data:`TIMEOUT_SKIPPED` so callers can tell the user.
    """
    global TIMEOUT_SKIPPED

    if seconds is None or seconds <= 0:
        yield
        return

    on_main_thread = threading.current_thread() is threading.main_thread()
    if not (SIGALRM_AVAILABLE and on_main_thread):
        TIMEOUT_SKIPPED = True
        yield
        return

    def _fire(signum: int, frame: object) -> None:
        raise CellTimeout(f"execution exceeded {seconds:g}s")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _truncate(text: str, limit: int) -> str:
    """Cap ``text`` at ``limit`` characters with an explicit marker."""
    if limit <= 0 or len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n... [truncated, {omitted} chars omitted]"


def _format_traceback(exc: BaseException) -> str:
    """Format ``exc`` starting at the first frame belonging to the cell.

    Frames from scisolve itself are dropped so the model only ever sees its own
    code. A SyntaxError has no cell frame at all; it still prints its own source
    line and caret.
    """
    tb = exc.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_filename != CELL_FILENAME:
        tb = tb.tb_next
    return "".join(traceback.format_exception(type(exc), exc, tb)).rstrip()


class Executor:
    """One long-lived namespace. Variables defined in a cell survive to the next."""

    def __init__(
        self,
        *,
        figure_dir: Path | None = None,
        timeout: float | None = 30.0,
        max_output_chars: int = 8_000,
        time_limit_factory: TimeLimit = time_limit,
    ) -> None:
        self.figure_dir = (
            Path(figure_dir) if figure_dir else Path(tempfile.mkdtemp(prefix="scisolve-"))
        )
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self._time_limit = time_limit_factory
        self._figure_count = 0
        self._namespace: dict[str, object] = {
            "__name__": "__main__",
            "np": np,
            "numpy": np,
            "scipy": scipy,
            "plt": plt,
            "math": math,
        }

    @property
    def namespace(self) -> dict[str, object]:
        """The live cell namespace. Exposed for tests and debugging."""
        return self._namespace

    def run_cell(self, code: str) -> ExecResult:
        """Execute ``code``. Never raises; failures come back as ``error`` text."""
        buffer = io.StringIO()
        error: str | None = None
        timed_out = False
        started = time.perf_counter()

        try:
            with self._time_limit(self.timeout), contextlib.redirect_stdout(buffer):
                exec(compile(code, CELL_FILENAME, "exec"), self._namespace)  # noqa: S102
        except CellTimeout as exc:
            timed_out = True
            error = _format_traceback(exc)
        except (Exception, SystemExit) as exc:  # KeyboardInterrupt still propagates
            error = _format_traceback(exc)

        duration = time.perf_counter() - started
        figures = self._save_figures()

        return ExecResult(
            code=code,
            stdout=_truncate(buffer.getvalue(), self.max_output_chars),
            error=_truncate(error, self.max_output_chars) if error is not None else None,
            figures=figures,
            duration_s=duration,
            timed_out=timed_out,
        )

    def _save_figures(self) -> tuple[Path, ...]:
        """Save and close every open figure. The counter never resets."""
        saved: list[Path] = []
        for num in plt.get_fignums():
            self._figure_count += 1
            path = self.figure_dir / f"fig_{self._figure_count:03d}.png"
            try:
                plt.figure(num).savefig(path)
            except Exception:  # a broken figure must not sink the cell result
                self._figure_count -= 1
                continue
            saved.append(path)
        plt.close("all")
        return tuple(saved)
