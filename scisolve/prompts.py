"""The system prompt and the two tool schemas the agent exposes."""

from __future__ import annotations

VERIFICATION_KINDS = (
    "analytic",  # matched a closed form
    "independent_method",  # a second algorithm agrees
    "limiting_case",  # reduces to a known result in a limit
    "invariant",  # conservation law, residual norm, KKT, orthogonality
    "convergence",  # refine grid/tolerance, answer is stable
    "dimensional",  # units are consistent
)

RUN_PYTHON = {
    "name": "run_python",
    "description": (
        "Execute Python in a persistent session with numpy (np), scipy, and "
        "matplotlib.pyplot (plt) available. Variables persist between calls. "
        "Returns stdout, any traceback, and paths to figures you create. "
        "Print anything you intend to reference in your answer."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    },
}

FINISH_SOLUTION = {
    "name": "finish_solution",
    "description": (
        "Report the final answer. Every number in `answer` must have been "
        "printed by a previous run_python call. Name the verification you ran."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "verification": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(VERIFICATION_KINDS)},
                    "description": {"type": "string"},
                    "evidence_turn": {"type": "integer"},
                },
                "required": ["kind", "description", "evidence_turn"],
            },
        },
        "required": ["answer", "verification"],
    },
}

TOOLS = [RUN_PYTHON, FINISH_SOLUTION]

SYSTEM_PROMPT = """\
You solve scientific computing problems by running code, not by recalling results.

Rules that are enforced, not merely requested:

1. Compute, don't recall. If you know an answer, run code that produces it anyway.
   A remembered value and a computed value are indistinguishable in prose, which
   is why only the computed one is allowed.

2. Print every value you intend to cite. Your answer is checked against captured
   stdout. A number that was computed but never printed will be rejected exactly
   like an invented one, so print intermediate values you plan to quote. Print
   at full precision — numpy truncates arrays by default, and the check matches
   to a relative tolerance, so an answer quoting more digits than you printed is
   only verified to the digits that actually appeared. `print(repr(x))` or
   `np.set_printoptions(precision=17)` if you intend to quote many digits.

3. Before finishing, run a check that could actually fail. Prefer a check that is
   a different computation: compare against a closed form, a second algorithm, a
   conservation law or residual, the limiting case, or the same problem at a
   finer tolerance. Re-running the same computation and finding the same number
   verifies nothing.

4. Then call finish_solution and name the check you ran. Its `evidence_turn` is
   the index of the run_python call that produced the check, counting from 0 in
   the order you made them; each tool result tells you its own index.

Errors come back as tracebacks. Read them and fix the code. The session is
persistent, so variables and imports survive between calls.
"""
