"""Executor tests. No API key, no network."""

from __future__ import annotations

import contextlib

import pytest

from scisolve import executor as executor_mod
from scisolve.executor import SIGALRM_AVAILABLE, ExecResult, Executor


def make(tmp_path, **kw) -> Executor:
    return Executor(figure_dir=tmp_path, **kw)


def test_namespace_persists_across_cells(tmp_path):
    ex = make(tmp_path)
    assert ex.run_cell("x = 41").ok
    result = ex.run_cell("print(x + 1)")
    assert result.ok, result.error
    assert result.stdout.strip() == "42"


def test_preloaded_modules_are_available(tmp_path):
    ex = make(tmp_path)
    result = ex.run_cell("print(np.arange(4).sum(), math.floor(1.7), scipy.__name__)")
    assert result.ok, result.error
    assert result.stdout.strip() == "6 1 scipy"


def test_stdout_is_captured_not_printed(tmp_path, capsys):
    ex = make(tmp_path)
    result = ex.run_cell("print('hello from the cell')")
    assert result.stdout.strip() == "hello from the cell"
    assert "hello from the cell" not in capsys.readouterr().out


def test_exception_is_returned_as_text_not_raised(tmp_path):
    ex = make(tmp_path)
    result = ex.run_cell("1 / 0")
    assert isinstance(result, ExecResult)
    assert not result.ok
    assert result.error is not None
    assert "ZeroDivisionError" in result.error
    assert not result.timed_out
    # the session survives a failed cell
    assert ex.run_cell("print('still alive')").stdout.strip() == "still alive"


def test_traceback_hides_scisolve_frames(tmp_path):
    ex = make(tmp_path)
    result = ex.run_cell("def f():\n    raise ValueError('boom')\n\nf()")
    assert result.error is not None
    assert "executor.py" not in result.error
    assert "run_cell" not in result.error
    assert '"<cell>"' in result.error
    assert result.error.endswith("ValueError: boom")


def test_syntax_error_is_reported(tmp_path):
    ex = make(tmp_path)
    result = ex.run_cell("for i in range(3)\n    print(i)")
    assert not result.ok
    assert "SyntaxError" in result.error


def test_stdout_before_an_error_is_kept(tmp_path):
    ex = make(tmp_path)
    result = ex.run_cell("print('partial')\nraise RuntimeError('later')")
    assert result.stdout.strip() == "partial"
    assert "RuntimeError: later" in result.error


def test_system_exit_does_not_kill_the_session(tmp_path):
    ex = make(tmp_path)
    result = ex.run_cell("raise SystemExit(1)")
    assert not result.ok
    assert ex.run_cell("print('alive')").stdout.strip() == "alive"


def test_figure_is_saved_and_path_returned(tmp_path):
    ex = make(tmp_path)
    result = ex.run_cell("plt.plot([0, 1], [0, 1])")
    assert result.ok, result.error
    assert len(result.figures) == 1
    assert result.figures[0].exists()
    assert result.figures[0].name == "fig_001.png"


def test_figure_counter_never_resets(tmp_path):
    ex = make(tmp_path)
    first = ex.run_cell("plt.figure()\nplt.plot([1, 2])")
    second = ex.run_cell("plt.figure()\nplt.plot([2, 1])\nplt.figure()\nplt.plot([3, 1])")
    assert [p.name for p in first.figures] == ["fig_001.png"]
    assert [p.name for p in second.figures] == ["fig_002.png", "fig_003.png"]
    assert all(p.exists() for p in first.figures + second.figures)


def test_figures_are_closed_between_cells(tmp_path):
    ex = make(tmp_path)
    ex.run_cell("plt.plot([1, 2])")
    result = ex.run_cell("print(len(plt.get_fignums()))")
    assert result.stdout.strip() == "0"


def test_figure_saved_even_when_the_cell_fails(tmp_path):
    ex = make(tmp_path)
    result = ex.run_cell("plt.plot([1, 2])\nraise ValueError('after plotting')")
    assert not result.ok
    assert len(result.figures) == 1
    assert result.figures[0].exists()


def test_stdout_is_truncated_at_the_cap(tmp_path):
    ex = make(tmp_path, max_output_chars=100)
    result = ex.run_cell("print('x' * 5000)")
    assert result.ok, result.error
    assert len(result.stdout) < 200
    assert "[truncated," in result.stdout
    assert "chars omitted]" in result.stdout
    assert result.stdout.startswith("x" * 100)


def test_short_output_is_not_truncated(tmp_path):
    ex = make(tmp_path, max_output_chars=100)
    result = ex.run_cell("print('short')")
    assert result.stdout == "short\n"


def test_error_text_is_truncated_at_the_cap(tmp_path):
    ex = make(tmp_path, max_output_chars=200)
    result = ex.run_cell("raise ValueError('y' * 5000)")
    assert not result.ok
    assert "[truncated," in result.error


def test_duration_is_recorded(tmp_path):
    ex = make(tmp_path)
    result = ex.run_cell("import time as _t; _t.sleep(0.05)")
    assert result.ok, result.error
    assert result.duration_s >= 0.05


@pytest.mark.skipif(not SIGALRM_AVAILABLE, reason="no SIGALRM on this platform")
def test_timeout_fires_on_a_python_loop(tmp_path):
    ex = make(tmp_path, timeout=0.2)
    result = ex.run_cell("while True:\n    pass")
    assert result.timed_out
    assert not result.ok
    assert "CellTimeout" in result.error
    assert result.duration_s < 5.0
    # the session is usable afterwards
    assert ex.run_cell("print('recovered')").stdout.strip() == "recovered"


@pytest.mark.skipif(not SIGALRM_AVAILABLE, reason="no SIGALRM on this platform")
def test_timer_is_cleared_after_a_successful_cell(tmp_path):
    ex = make(tmp_path, timeout=0.3)
    assert ex.run_cell("print('fast')").ok
    import time as _t

    _t.sleep(0.4)  # a leaked timer would raise here, in the test's own frame
    assert ex.run_cell("print('no stray alarm')").ok


def test_timeout_is_skipped_and_flagged_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(executor_mod, "SIGALRM_AVAILABLE", False)
    monkeypatch.setattr(executor_mod, "TIMEOUT_SKIPPED", False)
    ex = make(tmp_path, timeout=0.2)
    assert ex.run_cell("print('ran without a limit')").ok
    assert executor_mod.TIMEOUT_SKIPPED is True


def test_time_limit_strategy_is_injectable(tmp_path):
    calls: list[float | None] = []

    @contextlib.contextmanager
    def recording_limit(seconds):
        calls.append(seconds)
        yield

    ex = Executor(figure_dir=tmp_path, timeout=7.5, time_limit_factory=recording_limit)
    assert ex.run_cell("print(1)").ok
    assert calls == [7.5]


def test_no_timeout_requested_means_no_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(executor_mod, "TIMEOUT_SKIPPED", False)
    ex = make(tmp_path, timeout=None)
    assert ex.run_cell("print('unbounded')").ok
    assert executor_mod.TIMEOUT_SKIPPED is False


def test_default_figure_dir_is_created(tmp_path):
    ex = Executor()
    assert ex.figure_dir.exists()
    result = ex.run_cell("plt.plot([1, 2])")
    assert result.figures[0].parent == ex.figure_dir


def test_result_carries_the_code_back(tmp_path):
    ex = make(tmp_path)
    code = "print('echo')"
    assert ex.run_cell(code).code == code
