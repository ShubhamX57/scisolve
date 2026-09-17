"""Replay recorded real API runs. Skips until a cassette exists (milestone 6).

The scripted fakes in fakes.py were written from the same assumptions as the
code, so they cannot catch the case where the real API's response shape differs
from what agent.py expects. A cassette can: it is a real response, frozen, and
these tests re-run the loop against it with no network and no key.

What this does not do is re-check the API today. A cassette goes stale silently;
re-record it when something looks wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scisolve.agent import Agent
from scisolve.executor import Executor
from scisolve.prompts import VERIFICATION_KINDS

CASSETTE_DIR = Path(__file__).resolve().parent / "cassettes"
CASSETTES = sorted(CASSETTE_DIR.glob("*.json"))
NO_CASSETTE = "no cassette recorded yet; see scripts/record_cassette.py (milestone 6)"


class CassettePlayer:
    """Replays recorded responses in order, recording what was asked."""

    def __init__(self, turns: list[dict]) -> None:
        self._responses = [turn["response"] for turn in turns]
        self.requests: list[dict] = []

    def __call__(self, payload: dict) -> dict:
        self.requests.append(payload)
        if not self._responses:
            raise AssertionError(
                "the loop asked for more turns than were recorded; either the "
                "cassette is truncated or agent.py now behaves differently"
            )
        return self._responses.pop(0)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.mark.skipif(not CASSETTES, reason=NO_CASSETTE)
def test_no_cassette_contains_key_material():
    for path in CASSETTES:
        text = path.read_text()
        assert "sk-ant-" not in text, f"{path.name} still contains an API key"
        assert "x-api-key" not in text, f"{path.name} contains request headers"


@pytest.mark.skipif(not CASSETTES, reason=NO_CASSETTE)
def test_recorded_responses_have_the_shape_agent_expects():
    for path in CASSETTES:
        cassette = load(path)
        assert cassette["turns"], f"{path.name} recorded no turns"
        for index, turn in enumerate(cassette["turns"]):
            response = turn["response"]
            assert isinstance(response.get("content"), list), f"{path.name} turn {index}"
            assert response.get("role") == "assistant", f"{path.name} turn {index}"
            assert response.get("stop_reason"), f"{path.name} turn {index}"
        last = cassette["turns"][-1]["response"]["content"]
        names = [b.get("name") for b in last if b.get("type") == "tool_use"]
        assert "finish_solution" in names, f"{path.name} does not end in finish_solution"


@pytest.mark.skipif(not CASSETTES, reason=NO_CASSETTE)
def test_cassette_replays_to_the_same_solution(tmp_path):
    for path in CASSETTES:
        cassette = load(path)
        player = CassettePlayer(cassette["turns"])
        agent = Agent(
            transport=player,
            executor=Executor(figure_dir=tmp_path / path.stem, timeout=60.0),
        )
        solution = agent.solve(cassette["problem"])
        recorded = cassette["solution"]

        assert solution.answer == recorded["answer"], path.name
        assert solution.finished is recorded["finished"], path.name
        assert solution.grounded is recorded["grounded"], path.name
        assert solution.turns == recorded["turns"], path.name
        assert len(player.requests) == len(cassette["turns"]), path.name
        if solution.verification:
            assert solution.verification.kind in VERIFICATION_KINDS, path.name


@pytest.mark.skipif(not CASSETTES, reason=NO_CASSETTE)
def test_replayed_answer_is_grounded_in_freshly_computed_output(tmp_path):
    """The code is re-executed on replay, so grounding is checked against output
    produced now -- not against the numbers the recording happened to contain."""
    for path in CASSETTES:
        cassette = load(path)
        if not cassette["solution"]["grounded"]:
            continue
        agent = Agent(
            transport=CassettePlayer(cassette["turns"]),
            executor=Executor(figure_dir=tmp_path / f"{path.stem}_fresh", timeout=60.0),
        )
        solution = agent.solve(cassette["problem"])
        assert solution.grounded, (
            f"{path.name} was grounded when recorded but not on replay; the model's "
            "code is not deterministic, or numpy/scipy changed under it"
        )
