#!/usr/bin/env python
"""Record one real API run into tests/cassettes/.

Hand-written fakes prove the loop does what we think it does. They cannot tell
you that the API's response shape changed under you, because the fake was
written from the same assumption as the code. This does one real run, saves
every request/response pair verbatim, scrubs the key, and leaves something
`tests/test_cassette.py` can replay forever afterwards with no network.

This is the only script in the repo that spends money. It needs
ANTHROPIC_API_KEY.

    python scripts/record_cassette.py
    python scripts/record_cassette.py --name oscillator --problem "..."

It exits non-zero if the run came back ungrounded or unfinished, so a bad run
is not quietly committed as a reference recording.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scisolve.agent import Agent  # noqa: E402
from scisolve.cli import solution_dict  # noqa: E402
from scisolve.executor import Executor  # noqa: E402
from scisolve.transport import DEFAULT_MODEL, HTTPTransport, with_retries  # noqa: E402

CASSETTE_DIR = Path(__file__).resolve().parent.parent / "tests" / "cassettes"
KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")

DEFAULT_PROBLEM = (
    "A damped oscillator satisfies x'' + 2*0.08*3.0*x' + 3.0**2 * x = 0 with "
    "x(0) = 1 and x'(0) = 0. Find x(4.0), and verify it against the closed-form "
    "underdamped solution."
)


class Recorder:
    """Passes calls through to the real transport, keeping both sides."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.turns: list[dict] = []

    def __call__(self, payload: dict) -> dict:
        response = self._inner(payload)
        self.turns.append({"request": payload, "response": response})
        return response


def scrub(text: str, api_key: str) -> str:
    """Remove anything key-shaped. The key rides in a header, not the payload,
    but a recording that is wrong about that is expensive, so check anyway."""
    if api_key:
        text = text.replace(api_key, "<scrubbed>")
    return KEY_PATTERN.sub("<scrubbed>", text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="damped_oscillator")
    parser.add_argument("--problem", default=DEFAULT_PROBLEM)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-turns", type=int, default=12)
    args = parser.parse_args(argv)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ANTHROPIC_API_KEY is not set; this script makes a real call.", file=sys.stderr)
        return 2

    CASSETTE_DIR.mkdir(parents=True, exist_ok=True)
    figure_dir = CASSETTE_DIR / "_figures"
    recorder = Recorder(with_retries(HTTPTransport(api_key=api_key)))

    print(f"model: {args.model}")
    print(f"problem: {args.problem}\n")

    solution = Agent(
        model=args.model,
        max_turns=args.max_turns,
        transport=recorder,
        executor=Executor(figure_dir=figure_dir, timeout=60.0),
    ).solve(args.problem)

    for index, turn in enumerate(recorder.turns):
        blocks = turn["response"].get("content", [])
        kinds = [block.get("type") for block in blocks if isinstance(block, dict)]
        print(f"turn {index}: stop_reason={turn['response'].get('stop_reason')} blocks={kinds}")

    print(f"\nanswer: {solution.answer}")
    print(f"finished={solution.finished} grounded={solution.grounded} turns={solution.turns}")
    if solution.ungrounded:
        print(f"ungrounded: {', '.join(solution.ungrounded)}")

    cassette = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "problem": args.problem,
        "solution": solution_dict(solution),
        "turns": recorder.turns,
    }
    path = CASSETTE_DIR / f"{args.name}.json"
    path.write_text(scrub(json.dumps(cassette, indent=2), api_key))
    print(f"\nwrote {path}")

    if not (solution.finished and solution.grounded):
        print(
            "\nThis run was not grounded and finished. Look at it before committing "
            "it as a reference recording -- and if the agent genuinely got it wrong, "
            "it belongs in examples/failures/, not here.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
