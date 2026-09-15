"""`scisolve "problem"` — run the agent and leave an auditable trail behind.

Every run writes `runs/<utc-timestamp>/` with the transcript, each code cell as
a file, the figures, and the solution. That is not logging; it is the point. An
answer you cannot retrace is the thing this project exists to avoid.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .agent import Agent, Solution
from .executor import TIMEOUT_SKIPPED, ExecResult, Executor
from .transport import DEFAULT_MODEL, Transport

EXIT_OK = 0
EXIT_UNVERIFIED = 1
EXIT_ERROR = 2


class RecordingExecutor(Executor):
    """An Executor that keeps every result and can narrate as it goes.

    Subclassing rather than adding a callback to Agent keeps the public API in
    one piece: the agent still just calls run_cell.
    """

    def __init__(self, *, stream: bool = True, out=None, **kw) -> None:
        super().__init__(**kw)
        self.runs: list[ExecResult] = []
        self._stream = stream
        self._out = out or sys.stdout

    def run_cell(self, code: str) -> ExecResult:
        index = len(self.runs)
        if self._stream:
            print(f"\n[{index}] run_python", file=self._out)
            for line in code.splitlines():
                print(f"    {line}", file=self._out)
        result = super().run_cell(code)
        self.runs.append(result)
        if self._stream:
            self._narrate(result)
        return result

    def _narrate(self, result: ExecResult) -> None:
        body = result.stdout.rstrip()
        if body:
            for line in body.splitlines():
                print(f"  out {line}", file=self._out)
        elif result.ok:
            print("  out (nothing printed)", file=self._out)
        if result.error:
            for line in result.error.splitlines():
                print(f"  err {line}", file=self._out)
        for figure in result.figures:
            print(f"  fig {figure.name}", file=self._out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scisolve",
        description="Solve a scientific computing problem by running code and checking the result.",
    )
    parser.add_argument("problem", nargs="?", help="the problem, in plain English")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    parser.add_argument("--max-turns", type=int, default=12, metavar="N")
    parser.add_argument(
        "--timeout", type=float, default=30.0, metavar="S", help="per-cell wall clock limit"
    )
    parser.add_argument("--out", default="runs", metavar="DIR", help="where run directories go")
    parser.add_argument("--json", action="store_true", help="print the solution as JSON")
    parser.add_argument("--quiet", action="store_true", help="don't stream turns")
    parser.add_argument("--version", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, transport: Transport | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.version:
        from . import __version__

        print(f"scisolve {__version__}")
        return EXIT_OK

    if not args.problem:
        build_parser().print_usage(sys.stderr)
        print("scisolve: a problem statement is required", file=sys.stderr)
        return EXIT_ERROR

    stream = not (args.quiet or args.json)
    run_dir = _make_run_dir(Path(args.out))

    executor = RecordingExecutor(
        stream=stream,
        figure_dir=run_dir / "figures",
        timeout=args.timeout,
    )

    try:
        agent = Agent(
            model=args.model,
            max_turns=args.max_turns,
            executor=executor,
            transport=transport,
        )
    except ValueError as exc:
        print(f"scisolve: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if stream:
        print(f"problem: {args.problem}")
        if args.timeout and TIMEOUT_SKIPPED:
            print("note: cell timeouts are unavailable on this platform", file=sys.stderr)

    try:
        solution = agent.solve(args.problem)
    except Exception as exc:  # the loop is meant to absorb failures; this is a bug
        print(f"scisolve: unexpected failure: {exc!r}", file=sys.stderr)
        return EXIT_ERROR

    write_run(run_dir, solution, executor.runs)

    if args.json:
        print(json.dumps(solution_dict(solution), indent=2))
    else:
        _report(solution, run_dir)

    return EXIT_OK if (solution.finished and solution.grounded) else EXIT_UNVERIFIED


def _make_run_dir(base: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = base / stamp
    suffix = 1
    while run_dir.exists():  # two runs inside one second
        run_dir = base / f"{stamp}-{suffix}"
        suffix += 1
    (run_dir / "cells").mkdir(parents=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir


def solution_dict(solution: Solution) -> dict:
    verification = (
        dataclasses.asdict(solution.verification) if solution.verification else None
    )
    return {
        "problem": solution.problem,
        "answer": solution.answer,
        "verification": verification,
        "finished": solution.finished,
        "grounded": solution.grounded,
        "ungrounded": list(solution.ungrounded),
        "turns": solution.turns,
        "stop_reason": solution.stop_reason,
        "figures": [str(f) for f in solution.figures],
    }


def write_run(run_dir: Path, solution: Solution, runs: list[ExecResult]) -> None:
    """Write the full audit trail. Nothing here is derived; it is what happened."""
    (run_dir / "solution.json").write_text(json.dumps(solution_dict(solution), indent=2))
    (run_dir / "transcript.json").write_text(json.dumps(list(solution.transcript), indent=2))
    for index, result in enumerate(runs):
        header = [
            f"# run_python call {index}",
            f"# ok={result.ok} timed_out={result.timed_out} duration={result.duration_s:.3f}s",
            "",
        ]
        (run_dir / "cells" / f"cell_{index:03d}.py").write_text(
            "\n".join(header) + result.code + "\n"
        )


def _report(solution: Solution, run_dir: Path) -> None:
    print()
    if solution.answer:
        print(f"answer: {solution.answer}")
    else:
        print(f"no answer: stopped on {solution.stop_reason} after {solution.turns} turns")

    if solution.verification:
        v = solution.verification
        print(f"checked: {v.kind} — {v.description} (evidence: run_python call {v.evidence_turn})")

    if solution.answer and not solution.grounded:
        listed = ", ".join(solution.ungrounded) or "(none identified)"
        print()
        print("WARNING: this answer is NOT grounded.", file=sys.stderr)
        print(
            f"  these values were never printed by any code in this run: {listed}",
            file=sys.stderr,
        )
        print("  treat the answer as unverified.", file=sys.stderr)

    print(f"run: {run_dir}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
