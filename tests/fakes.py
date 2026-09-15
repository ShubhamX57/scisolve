"""A scripted Messages API. No network, no key, no monkeypatching."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from scisolve.transport import APIStatusError


@dataclass(frozen=True)
class HTTPFailure:
    """A non-200 the fake should raise instead of answering."""

    status: int
    retry_after: float | None = None
    err_type: str = "rate_limit_error"
    message: str = "scripted failure"
    request_id: str = "req_fake_01"


@dataclass
class FakeAnthropic:
    """Pops one scripted turn per call, records what was sent."""

    turns: list
    requests: list[dict] = field(default_factory=list)

    def __call__(self, payload: dict) -> dict:
        self.requests.append(copy.deepcopy(payload))
        if not self.turns:
            raise AssertionError(
                f"FakeAnthropic ran out of scripted turns after {len(self.requests)} "
                "requests; the agent asked for one more than the test scripted."
            )
        turn = self.turns.pop(0)
        if isinstance(turn, HTTPFailure):
            raise APIStatusError(
                turn.status,
                message=turn.message,
                error_type=turn.err_type,
                request_id=turn.request_id,
                retry_after=turn.retry_after,
            )
        return turn


def _message(content: list[dict], stop_reason: str) -> dict:
    return {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": "fake-model",
        "content": content,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def msg_tool_use(
    name: str, tool_input: dict, *, id: str = "toolu_01", text: str | None = None
) -> dict:
    content: list[dict] = []
    if text:
        content.append({"type": "text", "text": text})
    content.append({"type": "tool_use", "id": id, "name": name, "input": tool_input})
    return _message(content, "tool_use")


def msg_tool_uses(*calls: dict, text: str | None = None) -> dict:
    """Several tool_use blocks in one assistant turn."""
    content: list[dict] = []
    if text:
        content.append({"type": "text", "text": text})
    content.extend(calls)
    return _message(content, "tool_use")


def tool_use_block(name: str, tool_input: dict, *, id: str) -> dict:
    return {"type": "tool_use", "id": id, "name": name, "input": tool_input}


def msg_text(text: str) -> dict:
    return _message([{"type": "text", "text": text}], "end_turn")


def run_python(code: str, *, id: str = "toolu_run", text: str | None = None) -> dict:
    return msg_tool_use("run_python", {"code": code}, id=id, text=text)


def finish(
    answer: str,
    *,
    kind: str = "analytic",
    description: str = "compared against the closed form",
    evidence_turn: int = 0,
    id: str = "toolu_fin",
    verification: dict | None = None,
) -> dict:
    if verification is None:
        verification = {
            "kind": kind,
            "description": description,
            "evidence_turn": evidence_turn,
        }
    return msg_tool_use(
        "finish_solution", {"answer": answer, "verification": verification}, id=id
    )


def http_failure(
    status: int, *, retry_after: float | None = None, err_type: str = "rate_limit_error"
) -> HTTPFailure:
    return HTTPFailure(status=status, retry_after=retry_after, err_type=err_type)


def last_user_content(api: FakeAnthropic, request_index: int) -> list[dict]:
    """The tool_result blocks the agent sent in a given request."""
    return api.requests[request_index]["messages"][-1]["content"]
