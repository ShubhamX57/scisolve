"""The agent loop: call the model, run its code, refuse ungrounded answers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .executor import ExecResult, Executor
from .grounding import check_grounding
from .prompts import SYSTEM_PROMPT, TOOLS, VERIFICATION_KINDS
from .transport import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    HTTPTransport,
    Transport,
    TransportError,
    with_retries,
)

#: One rejected finish_solution earns a re-prompt. The second is reported as-is
#: rather than argued with -- the loop is not a negotiation.
MAX_FINISH_ATTEMPTS = 2
#: Two consecutive turns with no tool call and we stop.
MAX_EMPTY_TURNS = 2

NUDGE = (
    "You did not call a tool. Use run_python to compute, then finish_solution "
    "to report. Do not answer in prose."
)


@dataclass(frozen=True)
class Verification:
    kind: str
    description: str
    evidence_turn: int


@dataclass(frozen=True)
class Solution:
    """The outcome of one ``solve``.

    ``finished`` and ``grounded`` are independent on purpose. A model can finish
    with an answer whose numbers were never computed; that is
    ``finished=True, grounded=False``, with the offending literals in
    ``ungrounded``. Only ``finished and grounded`` means the answer is safe to
    quote.
    """

    problem: str
    answer: str | None
    verification: Verification | None
    finished: bool
    grounded: bool
    ungrounded: tuple[str, ...]
    turns: int
    stop_reason: str
    transcript: tuple[dict, ...]
    figures: tuple[Path, ...]


@dataclass(frozen=True)
class _FinishAttempt:
    """A finish_solution call, validated. ``complaint`` is None when accepted."""

    answer: str | None
    verification: Verification | None
    grounded: bool
    ungrounded: tuple[str, ...]
    complaint: str | None


class Agent:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_turns: int = 12,
        api_key: str | None = None,
        executor: Executor | None = None,
        transport: Transport | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.max_turns = max_turns
        self.executor = executor or Executor()
        self.system_prompt = system_prompt or SYSTEM_PROMPT

        if transport is not None:
            # Supplied transports are used as given. Wrap your own with
            # transport.with_retries if you want the retry policy.
            self._transport = transport
        else:
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "no API key: pass api_key=, set ANTHROPIC_API_KEY, or inject "
                    "transport= (tests do the last one)"
                )
            self._transport = with_retries(HTTPTransport(api_key=key))

    def solve(self, problem: str) -> Solution:
        messages: list[dict] = [{"role": "user", "content": problem}]
        runs: list[ExecResult] = []
        turns = 0
        empty_turns = 0
        finish_attempts = 0

        def result(
            *,
            stop_reason: str,
            finished: bool = False,
            answer: str | None = None,
            verification: Verification | None = None,
            grounded: bool = False,
            ungrounded: tuple[str, ...] = (),
        ) -> Solution:
            return Solution(
                problem=problem,
                answer=answer,
                verification=verification,
                finished=finished,
                grounded=grounded,
                ungrounded=ungrounded,
                turns=turns,
                stop_reason=stop_reason,
                transcript=tuple(messages),
                figures=tuple(f for run in runs for f in run.figures),
            )

        while turns < self.max_turns:
            try:
                response = self._transport(self._payload(messages))
            except TransportError as exc:
                messages.append({"role": "system_note", "content": f"api_error: {exc}"})
                return result(stop_reason="api_error")

            turns += 1
            content = response.get("content", [])
            # Verbatim. Filtering or rebuilding blocks here is how you get a 400
            # on the next call.
            messages.append({"role": "assistant", "content": content})

            tool_uses = [
                block
                for block in content
                if isinstance(block, dict) and block.get("type") == "tool_use"
            ]
            if not tool_uses:
                empty_turns += 1
                if empty_turns >= MAX_EMPTY_TURNS:
                    return result(stop_reason="no_tool_calls")
                messages.append({"role": "user", "content": NUDGE})
                continue

            tool_results: list[dict] = []
            for block in tool_uses:
                name = block.get("name")
                block_id = block.get("id", "")
                tool_input = block.get("input")
                if not isinstance(tool_input, dict):
                    tool_input = {}

                if name == "run_python":
                    code = tool_input.get("code")
                    if not isinstance(code, str):
                        tool_results.append(
                            _tool_result(
                                block_id,
                                "run_python needs a string `code` field; got "
                                f"{sorted(tool_input) or 'nothing'}.",
                                is_error=True,
                            )
                        )
                        continue
                    run = self.executor.run_cell(code)
                    runs.append(run)
                    tool_results.append(
                        _tool_result(
                            block_id,
                            _render_run(run, len(runs) - 1),
                            is_error=not run.ok,
                        )
                    )

                elif name == "finish_solution":
                    attempt = self._validate_finish(tool_input, runs, problem)
                    if attempt.complaint is None:
                        return result(
                            stop_reason="finished",
                            finished=True,
                            answer=attempt.answer,
                            verification=attempt.verification,
                            grounded=True,
                        )
                    finish_attempts += 1
                    if finish_attempts >= MAX_FINISH_ATTEMPTS:
                        return result(
                            stop_reason="finished",
                            finished=True,
                            answer=attempt.answer,
                            verification=attempt.verification,
                            grounded=attempt.grounded,
                            ungrounded=attempt.ungrounded,
                        )
                    tool_results.append(
                        _tool_result(block_id, attempt.complaint, is_error=True)
                    )

                else:
                    tool_results.append(
                        _tool_result(
                            block_id,
                            f"unknown tool {name!r}; this session has run_python "
                            "and finish_solution.",
                            is_error=True,
                        )
                    )

            messages.append({"role": "user", "content": tool_results})

        return result(stop_reason="max_turns")

    def _payload(self, messages: list[dict]) -> dict:
        return {
            "model": self.model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system": self.system_prompt,
            "messages": [m for m in messages if m["role"] in ("user", "assistant")],
            "tools": TOOLS,
            "tool_choice": {"type": "auto"},
        }

    def _validate_finish(
        self, tool_input: dict, runs: list[ExecResult], problem: str
    ) -> _FinishAttempt:
        answer = tool_input.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return _FinishAttempt(
                None, None, False, (), "finish_solution needs a non-empty `answer`."
            )

        report = check_grounding(
            answer, [run.stdout for run in runs if run.ok], problem=problem
        )
        grounded = report.grounded
        ungrounded = report.unmatched

        raw = tool_input.get("verification")
        if not isinstance(raw, dict):
            return _FinishAttempt(
                answer, None, grounded, ungrounded,
                "finish_solution needs a `verification` object with kind, "
                "description and evidence_turn.",
            )

        kind = raw.get("kind")
        if kind not in VERIFICATION_KINDS:
            return _FinishAttempt(
                answer, None, grounded, ungrounded,
                f"{kind!r} is not a verification kind. Use one of: "
                f"{', '.join(VERIFICATION_KINDS)}.",
            )

        evidence_turn = raw.get("evidence_turn")
        if not isinstance(evidence_turn, int) or isinstance(evidence_turn, bool):
            return _FinishAttempt(
                answer, None, grounded, ungrounded,
                "`evidence_turn` must be an integer index of a run_python call.",
            )
        if not 0 <= evidence_turn < len(runs):
            return _FinishAttempt(
                answer, None, grounded, ungrounded,
                f"evidence_turn {evidence_turn} does not index a run_python call; "
                f"there {'has' if len(runs) == 1 else 'have'} been {len(runs)} "
                f"so far (0 to {len(runs) - 1})." if runs else
                "evidence_turn cannot point anywhere: you have not run any code yet.",
            )

        verification = Verification(
            kind=kind,
            description=str(raw.get("description", "")),
            evidence_turn=evidence_turn,
        )

        if not grounded:
            listed = ", ".join(ungrounded)
            return _FinishAttempt(
                answer, verification, grounded, ungrounded,
                f"these values in your answer do not appear in any output: {listed}. "
                "Run code that produces them, or remove them from the answer.",
            )

        return _FinishAttempt(answer, verification, True, (), None)


def _tool_result(tool_use_id: str, content: str, *, is_error: bool = False) -> dict:
    block: dict = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        block["is_error"] = True
    return block


def _render_run(run: ExecResult, index: int) -> str:
    parts = [f"[run_python call {index}]"]
    parts.append(run.stdout if run.stdout.strip() else "(no output printed)")
    if run.error:
        parts.append(run.error)
    if run.timed_out:
        parts.append("the cell hit its time limit and was interrupted")
    if run.figures:
        parts.append("figures: " + ", ".join(str(p) for p in run.figures))
    return "\n".join(parts)
