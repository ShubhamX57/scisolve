"""CLI tests. End-to-end through a scripted API; no key, no network."""

from __future__ import annotations

import json

import pytest
from fakes import FakeAnthropic, finish, http_failure, run_python

from scisolve.cli import EXIT_ERROR, EXIT_OK, EXIT_UNVERIFIED, main

LN2 = "print(0.6931471805599453)"


def solved() -> FakeAnthropic:
    return FakeAnthropic([run_python(LN2), finish("ln(2) = 0.6931", evidence_turn=0)])


def only(tmp_path):
    runs = list(tmp_path.iterdir())
    assert len(runs) == 1, f"expected one run directory, found {runs}"
    return runs[0]


def test_version_prints_and_exits_zero(capsys):
    assert main(["--version"]) == EXIT_OK
    assert capsys.readouterr().out.startswith("scisolve ")


def test_missing_problem_is_an_error(capsys):
    assert main([]) == EXIT_ERROR
    assert "problem statement is required" in capsys.readouterr().err


def test_no_api_key_exits_two(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["what is ln(2)?", "--out", str(tmp_path)]) == EXIT_ERROR
    assert "no API key" in capsys.readouterr().err


def test_solved_and_grounded_exits_zero(tmp_path, capsys):
    code = main(["what is ln(2)?", "--out", str(tmp_path)], transport=solved())
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "answer: ln(2) = 0.6931" in out
    assert "checked: analytic" in out


def test_turns_are_streamed_as_they_happen(tmp_path, capsys):
    main(["what is ln(2)?", "--out", str(tmp_path)], transport=solved())
    out = capsys.readouterr().out

    assert "[0] run_python" in out
    assert LN2 in out
    assert "out 0.6931471805599453" in out


def test_quiet_suppresses_the_stream_but_keeps_the_answer(tmp_path, capsys):
    main(["what is ln(2)?", "--out", str(tmp_path), "--quiet"], transport=solved())
    out = capsys.readouterr().out

    assert "[0] run_python" not in out
    assert "answer: ln(2) = 0.6931" in out


def test_run_directory_is_complete(tmp_path):
    main(
        ["what is ln(2)?", "--out", str(tmp_path)],
        transport=FakeAnthropic([
            run_python("plt.plot([0, 1], [0, 1])\n" + LN2),
            finish("ln(2) = 0.6931", evidence_turn=0),
        ]),
    )
    run_dir = only(tmp_path)

    solution = json.loads((run_dir / "solution.json").read_text())
    assert solution["answer"] == "ln(2) = 0.6931"
    assert solution["grounded"] is True
    assert solution["verification"]["kind"] == "analytic"

    transcript = json.loads((run_dir / "transcript.json").read_text())
    assert transcript[0]["role"] == "user"
    assert any(m["role"] == "assistant" for m in transcript)

    cell = (run_dir / "cells" / "cell_000.py").read_text()
    assert LN2 in cell
    assert "run_python call 0" in cell

    figures = list((run_dir / "figures").glob("*.png"))
    assert len(figures) == 1


def test_every_cell_is_written_including_failed_ones(tmp_path):
    main(
        ["what is ln(2)?", "--out", str(tmp_path)],
        transport=FakeAnthropic([
            run_python("print(1 / 0)"),
            run_python(LN2),
            finish("ln(2) = 0.6931", evidence_turn=1),
        ]),
    )
    cells = sorted((only(tmp_path) / "cells").iterdir())

    assert [c.name for c in cells] == ["cell_000.py", "cell_001.py"]
    assert "ok=False" in cells[0].read_text()
    assert "ok=True" in cells[1].read_text()


def test_ungrounded_answer_warns_and_exits_one(tmp_path, capsys):
    api = FakeAnthropic([
        run_python(LN2),
        finish("the period is 6.2832", evidence_turn=0),
        finish("the period is 6.2832", evidence_turn=0),
    ])
    code = main(["what is ln(2)?", "--out", str(tmp_path)], transport=api)
    captured = capsys.readouterr()

    assert code == EXIT_UNVERIFIED
    assert "NOT grounded" in captured.err
    assert "6.2832" in captured.err
    assert json.loads((only(tmp_path) / "solution.json").read_text())["ungrounded"] == ["6.2832"]


def test_unfinished_run_exits_one(tmp_path, capsys):
    api = FakeAnthropic([run_python("x = 1") for _ in range(4)])
    code = main(["spin", "--out", str(tmp_path), "--max-turns", "2"], transport=api)

    assert code == EXIT_UNVERIFIED
    assert "stopped on max_turns" in capsys.readouterr().out


def test_api_error_still_writes_a_run_directory(tmp_path):
    api = FakeAnthropic([http_failure(400, err_type="invalid_request_error")])
    code = main(["what is ln(2)?", "--out", str(tmp_path)], transport=api)

    assert code == EXIT_UNVERIFIED
    solution = json.loads((only(tmp_path) / "solution.json").read_text())
    assert solution["stop_reason"] == "api_error"


def test_json_output_is_machine_readable(tmp_path, capsys):
    code = main(["what is ln(2)?", "--out", str(tmp_path), "--json"], transport=solved())
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["grounded"] is True
    assert payload["stop_reason"] == "finished"
    assert payload["turns"] == 2


def test_max_turns_is_passed_through(tmp_path):
    api = FakeAnthropic([run_python("x = 1") for _ in range(9)])
    main(["spin", "--out", str(tmp_path), "--max-turns", "3"], transport=api)
    assert len(api.requests) == 3


def test_two_runs_get_separate_directories(tmp_path):
    main(["what is ln(2)?", "--out", str(tmp_path)], transport=solved())
    main(["what is ln(2)?", "--out", str(tmp_path)], transport=solved())
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize("flag", ["--help"])
def test_help_exits_cleanly(flag, capsys):
    with pytest.raises(SystemExit) as caught:
        main([flag])
    assert caught.value.code == 0
    assert "scisolve" in capsys.readouterr().out
