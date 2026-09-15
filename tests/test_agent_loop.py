"""Agent loop tests. Every one runs against a scripted API; none needs a key."""

from __future__ import annotations

import pytest
from fakes import (
    FakeAnthropic,
    finish,
    http_failure,
    last_user_content,
    msg_text,
    msg_tool_use,
    msg_tool_uses,
    run_python,
    tool_use_block,
)

from scisolve.agent import Agent
from scisolve.executor import Executor
from scisolve.transport import RetryPolicy, with_retries

LN2 = "print(0.6931471805599453)"


@pytest.fixture
def executor(tmp_path) -> Executor:
    return Executor(figure_dir=tmp_path, timeout=5.0)


def build(api: FakeAnthropic, executor: Executor, **kw) -> Agent:
    return Agent(transport=api, executor=executor, **kw)


def test_single_run_then_finish(executor):
    api = FakeAnthropic([
        run_python(LN2),
        finish("ln(2) = 0.6931", evidence_turn=0),
    ])
    sol = build(api, executor).solve("what is ln(2)?")

    assert sol.finished and sol.grounded
    assert sol.stop_reason == "finished"
    assert sol.answer == "ln(2) = 0.6931"
    assert sol.verification.kind == "analytic"
    assert sol.turns == 2
    assert api.requests[1]["messages"][-1]["content"][0]["type"] == "tool_result"


def test_execution_error_is_fed_back_and_recovered(executor):
    api = FakeAnthropic([
        run_python("print(1 / 0)"),
        run_python(LN2),
        finish("ln(2) = 0.6931", evidence_turn=1),
    ])
    sol = build(api, executor).solve("what is ln(2)?")

    fed_back = last_user_content(api, 1)[0]
    assert fed_back["is_error"] is True
    assert "ZeroDivisionError" in fed_back["content"]
    assert sol.finished and sol.grounded


def test_max_turns_returns_unfinished_solution(executor):
    api = FakeAnthropic([run_python("x = 1") for _ in range(5)])
    sol = build(api, executor, max_turns=3).solve("spin forever")

    assert not sol.finished
    assert sol.stop_reason == "max_turns"
    assert sol.turns == 3
    assert sol.answer is None
    assert len(api.requests) == 3


def test_model_returns_no_tool_calls(executor):
    api = FakeAnthropic([msg_text("the answer is about 0.69"), msg_text("still prose")])
    sol = build(api, executor).solve("what is ln(2)?")

    assert not sol.finished
    assert sol.stop_reason == "no_tool_calls"
    assert api.requests[1]["messages"][-1]["role"] == "user"
    assert "did not call a tool" in api.requests[1]["messages"][-1]["content"]


def test_unknown_tool_name_is_reported_not_crashed(executor):
    api = FakeAnthropic([
        msg_tool_use("solve_it_for_me", {"problem": "ln 2"}, id="toolu_bad"),
        run_python(LN2),
        finish("ln(2) = 0.6931", evidence_turn=0),
    ])
    sol = build(api, executor).solve("what is ln(2)?")

    complaint = last_user_content(api, 1)[0]
    assert complaint["is_error"] is True
    assert "solve_it_for_me" in complaint["content"]
    assert sol.finished and sol.grounded


def test_malformed_tool_input(executor):
    api = FakeAnthropic([
        msg_tool_use("run_python", {"source": "print(1)"}, id="toolu_malformed"),
        run_python(LN2),
        finish("ln(2) = 0.6931", evidence_turn=0),
    ])
    sol = build(api, executor).solve("what is ln(2)?")

    complaint = last_user_content(api, 1)[0]
    assert complaint["is_error"] is True
    assert "code" in complaint["content"]
    assert sol.finished


def test_tool_result_ids_match_tool_use_ids(executor):
    api = FakeAnthropic([
        run_python(LN2, id="toolu_xyz789"),
        finish("ln(2) = 0.6931", evidence_turn=0),
    ])
    build(api, executor).solve("what is ln(2)?")

    assert last_user_content(api, 1)[0]["tool_use_id"] == "toolu_xyz789"


def test_multiple_tool_uses_in_one_turn_all_get_results(executor):
    api = FakeAnthropic([
        msg_tool_uses(
            tool_use_block("run_python", {"code": "a = 2"}, id="toolu_a"),
            tool_use_block("run_python", {"code": "print(a * 1.5)"}, id="toolu_b"),
        ),
        finish("the product is 3.0", evidence_turn=1),
    ])
    sol = build(api, executor).solve("multiply")

    results = last_user_content(api, 1)
    assert [r["tool_use_id"] for r in results] == ["toolu_a", "toolu_b"]
    assert all(r["type"] == "tool_result" for r in results)
    assert "3.0" in results[1]["content"]
    assert sol.finished and sol.grounded


def test_transient_429_is_retried_then_succeeds(executor):
    api = FakeAnthropic([
        http_failure(429, retry_after=0.0),
        run_python(LN2),
        finish("ln(2) = 0.6931", evidence_turn=0),
    ])
    slept: list[float] = []
    transport = with_retries(api, sleep=slept.append)
    sol = Agent(transport=transport, executor=executor).solve("what is ln(2)?")

    assert slept == [0.0]
    assert sol.finished and sol.grounded
    assert len(api.requests) == 3


def test_persistent_500_gives_up_with_clear_error(executor):
    api = FakeAnthropic([http_failure(500, err_type="api_error") for _ in range(6)])
    slept: list[float] = []
    transport = with_retries(
        api, policy=RetryPolicy(max_attempts=3), sleep=slept.append
    )
    sol = Agent(transport=transport, executor=executor).solve("what is ln(2)?")

    assert sol.stop_reason == "api_error"
    assert not sol.finished
    assert len(api.requests) == 3
    assert len(slept) == 2
    note = sol.transcript[-1]["content"]
    assert "gave up after 3 attempts" in note
    assert "HTTP 500" in note


def test_400_is_not_retried(executor):
    api = FakeAnthropic([http_failure(400, err_type="invalid_request_error")])
    slept: list[float] = []
    transport = with_retries(api, sleep=slept.append)
    sol = Agent(transport=transport, executor=executor).solve("what is ln(2)?")

    assert sol.stop_reason == "api_error"
    assert len(api.requests) == 1
    assert slept == []
    assert "invalid_request_error" in sol.transcript[-1]["content"]


def test_finish_with_ungrounded_number_reprompts(executor):
    api = FakeAnthropic([
        run_python(LN2),
        finish("ln(2) = 0.6931 and the period is 6.2832", evidence_turn=0),
        run_python("print(6.283185307179586)"),
        finish("ln(2) = 0.6931 and the period is 6.2832", evidence_turn=1),
    ])
    sol = build(api, executor).solve("what is ln(2)?")

    complaint = last_user_content(api, 2)[0]
    assert complaint["is_error"] is True
    assert "6.2832" in complaint["content"]
    assert "0.6931" not in complaint["content"]
    assert sol.finished and sol.grounded


def test_second_ungrounded_finish_is_reported_not_argued_with(executor):
    api = FakeAnthropic([
        run_python(LN2),
        finish("the period is 6.2832", evidence_turn=0),
        finish("the period is 6.2832", evidence_turn=0),
    ])
    sol = build(api, executor).solve("what is ln(2)?")

    assert sol.finished
    assert not sol.grounded
    assert sol.ungrounded == ("6.2832",)
    assert sol.answer == "the period is 6.2832"


def test_finish_with_rounded_number_is_accepted(executor):
    api = FakeAnthropic([
        run_python(LN2),
        finish("ln(2) = 0.6931", evidence_turn=0),
    ])
    sol = build(api, executor).solve("what is ln(2)?")

    assert sol.grounded
    assert sol.ungrounded == ()


def test_finish_with_unknown_verification_kind_is_rejected(executor):
    api = FakeAnthropic([
        run_python(LN2),
        finish("ln(2) = 0.6931", kind="looks_right"),
        finish("ln(2) = 0.6931", kind="analytic", evidence_turn=0),
    ])
    sol = build(api, executor).solve("what is ln(2)?")

    complaint = last_user_content(api, 2)[0]
    assert complaint["is_error"] is True
    assert "looks_right" in complaint["content"]
    assert "independent_method" in complaint["content"]
    assert sol.finished and sol.grounded


def test_finish_with_out_of_range_evidence_turn_is_rejected(executor):
    api = FakeAnthropic([
        run_python(LN2),
        finish("ln(2) = 0.6931", evidence_turn=7),
        finish("ln(2) = 0.6931", evidence_turn=0),
    ])
    sol = build(api, executor).solve("what is ln(2)?")

    complaint = last_user_content(api, 2)[0]
    assert "evidence_turn 7" in complaint["content"]
    assert sol.finished and sol.grounded


def test_assistant_content_is_appended_verbatim(executor):
    turn = run_python(LN2, text="Let me compute that.")
    api = FakeAnthropic([turn, finish("ln(2) = 0.6931", evidence_turn=0)])
    build(api, executor).solve("what is ln(2)?")

    sent = api.requests[1]["messages"][1]
    assert sent["role"] == "assistant"
    assert sent["content"] == turn["content"]


def test_figures_are_collected_onto_the_solution(executor):
    api = FakeAnthropic([
        run_python("plt.plot([0, 1], [0, 1])\nprint(0.6931471805599453)"),
        finish("ln(2) = 0.6931", evidence_turn=0),
    ])
    sol = build(api, executor).solve("plot it")

    assert len(sol.figures) == 1
    assert sol.figures[0].exists()


def test_no_api_key_and_no_transport_fails_loudly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="no API key"):
        Agent()
