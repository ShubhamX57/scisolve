"""Every example is re-run through the Executor. No API key, no network.

These are the tests that would catch a wrong answer rather than a crash: each
example asserts its own verification, so a silent numerical regression in numpy
or scipy surfaces here as a failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scisolve.executor import Executor

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SCRIPTS = sorted(EXAMPLES.glob("*.py"))
FAILURES = sorted((EXAMPLES / "failures").glob("*.py"))


def run_script(path: Path, figure_dir: Path):
    executor = Executor(figure_dir=figure_dir, timeout=180.0, max_output_chars=40_000)
    return executor.run_cell(path.read_text())


def test_examples_are_present():
    assert len(SCRIPTS) >= 6, f"expected at least six examples, found {[p.name for p in SCRIPTS]}"
    assert len(FAILURES) >= 2, "the failure gallery needs at least two entries"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_example_verifies_itself(script, tmp_path):
    result = run_script(script, tmp_path)
    assert result.ok, f"{script.name} failed:\n{result.error}"
    assert "VERIFIED" in result.stdout, f"{script.name} ran but never confirmed its check"


@pytest.mark.parametrize("script", FAILURES, ids=lambda p: p.stem)
def test_recorded_failure_still_fails(script, tmp_path):
    """If one of these starts passing, the gallery entry is stale, not the code."""
    result = run_script(script, tmp_path)
    assert result.ok, f"{script.name} could not run:\n{result.error}"
    assert "STILL FAILING" in result.stdout, (
        f"{script.name} no longer demonstrates its failure; update the entry"
    )
