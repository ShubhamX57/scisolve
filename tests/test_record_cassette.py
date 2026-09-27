"""Tests for the cassette recorder. It is the one script that costs money, so
it should not be the one script nobody tested.

Everything here injects a transport; nothing touches the network or a key.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from fakes import FakeAnthropic, finish, http_failure, run_python

from scisolve.transport import with_retries

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "record_cassette.py"


def load_script():
    spec = importlib.util.spec_from_file_location("record_cassette", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rc(tmp_path, monkeypatch):
    module = load_script()
    monkeypatch.setattr(module, "CASSETTE_DIR", tmp_path)
    return module


def solved_api() -> FakeAnthropic:
    return FakeAnthropic([
        run_python("import numpy as np\nprint(repr(float(np.log(2))))"),
        finish("ln(2) = 0.6931471805599453", evidence_turn=0),
    ])


def test_scrub_removes_key_shaped_text(rc):
    assert rc.scrub("key sk-ant-api03-AbC_123-xyz here", "") == "key <scrubbed> here"
    assert rc.scrub("literal secret", "secret") == "literal <scrubbed>"


def test_placeholder_key_is_rejected_before_spending_anything(rc, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-...")
    assert rc.main([]) == 2
    assert "does not look like a key" in capsys.readouterr().err


def test_missing_key_is_rejected(rc, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert rc.main([]) == 2
    assert "is not set" in capsys.readouterr().err


def test_successful_run_writes_a_replayable_cassette(rc, tmp_path, capsys):
    code = rc.main(["--name", "demo", "--problem", "what is ln(2)?"], transport=solved_api())
    assert code == 0

    cassette = json.loads((tmp_path / "demo.json").read_text())
    assert cassette["problem"] == "what is ln(2)?"
    assert cassette["solution"]["grounded"] is True
    assert len(cassette["turns"]) == 2
    assert cassette["turns"][0]["request"]["messages"][0]["role"] == "user"
    assert cassette["turns"][-1]["response"]["content"][-1]["name"] == "finish_solution"
    assert "transcript" in cassette
    assert "sk-ant-" not in json.dumps(cassette)


def test_api_failure_writes_nothing_and_says_why(rc, tmp_path, capsys):
    api = FakeAnthropic([http_failure(401, err_type="authentication_error")])
    code = rc.main(["--name", "broken"], transport=with_retries(api, sleep=lambda s: None))

    assert code == 1
    assert not (tmp_path / "broken.json").exists(), "an empty cassette is worse than none"
    err = capsys.readouterr().err
    assert "No response was received" in err
    assert "authentication_error" in err


def test_ungrounded_run_is_written_but_flagged(rc, tmp_path, capsys):
    api = FakeAnthropic([
        run_python("print(1.0)"),
        finish("the period is 6.2832", evidence_turn=0),
        finish("the period is 6.2832", evidence_turn=0),
    ])
    code = rc.main(["--name", "ungrounded"], transport=api)

    assert code == 1, "a run that was not grounded must not exit 0"
    assert (tmp_path / "ungrounded.json").exists(), "there were real turns; keep them"
    assert "belongs in examples/failures/" in capsys.readouterr().err


def test_the_run_is_narrated_as_it_happens(rc, capsys):
    rc.main(["--name", "narrated"], transport=solved_api())
    out = capsys.readouterr().out

    assert "asking the model (turn 0)" in out
    assert "tool: run_python" in out
    assert "tool: finish_solution" in out
    assert "[0] run_python" in out, "the code cell should stream like it does in the CLI"
    assert "out 0.6931471805599453" in out, "and so should its output"
    assert "stop_reason=tool_use" in out


def test_quiet_suppresses_narration_but_keeps_the_verdict(rc, capsys):
    rc.main(["--name", "quiet", "--quiet"], transport=solved_api())
    out = capsys.readouterr().out

    assert "asking the model" not in out
    assert "[0] run_python" not in out
    assert "grounded=True" in out
    assert "wrote" in out
