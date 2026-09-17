# Contributing

## The rule that is not negotiable

Every number in a scisolve answer must be traceable to output captured from a
`run_python` call in the same session.

It is enforced in `scisolve/grounding.py`, not requested in a prompt, and it is
the only thing here that a weekend API wrapper doesn't have. A change that
weakens or bypasses it needs to say so out loud in the pull request. That is
not a veto — the check has real holes and closing or widening them is welcome
work — it is a requirement that the trade-off is stated rather than arriving
inside a refactor.

Between "make the output look more confident" and "make the output only claim
what was verified", take the second one.

## Running the tests

```bash
pip install -e ".[dev]"
python -m pytest
ruff check .
```

Use `python -m pytest`, not `pytest`. The bare command can resolve to a
different interpreter than the one your virtualenv installed into, and you get
a confusing `ModuleNotFoundError: No module named 'scisolve'` that has nothing
to do with your change.

No test may require an API key or touch the network. The loop is tested by
injecting a scripted transport (`tests/fakes.py`); if you find yourself wanting
to monkeypatch `requests`, inject a transport instead.

A test goes alongside any change to `executor.py`, `agent.py` or
`grounding.py`.

## Adding an example

Examples live in `examples/`, one file each, and end in `assert`s so
`tests/test_examples.py` can re-run them in CI and fail the build on a wrong
answer rather than printing quietly.

The bar is a real check: a closed form, a conservation law, an independent
algorithm, or a measured rate against a predicted one. **If the only way you can
verify it is to run the same computation twice, it is not an example yet.**
Solving an ODE with two calls to the same solver and getting the same number
verifies nothing.

Print what you check, print the thing you are checking it against, and end with
a line starting `VERIFIED:`.

Two of the seven examples in the repo were wrong when first written — one
converged to the wrong root, one compared two tolerances that both hit machine
precision so the comparison was vacuous. Run yours, read the numbers, and be
suspicious of a check that passes immediately.

## Adding a failure

`examples/failures/` holds cases where scisolve is wrong, or where a check
passes that should not. Entries are executable and assert that the failure
**still happens**, so closing a hole breaks the test and forces the entry to be
rewritten rather than silently deleted.

This directory is not an embarrassment to be kept short. For a project whose
claim is about verification, it is load-bearing.

## Things that need a decision, not a commit

Open an issue first:

- adding a runtime dependency (there are four: requests, numpy, scipy, matplotlib)
- replacing the `exec`-based executor with subprocess or container isolation
- changing the signature of `Agent.solve` or `Executor.run_cell` (additive
  keyword-only parameters are fine; anything else updates every call site and
  the README in the same change)
- anything that weakens the grounding check
- supporting a second model provider

## Style

Dataclasses for structured results, type hints, short docstrings, minimal
abstraction. No framework, no plugin system, no config layer. Prefer boring and
well-tested over clever.

Don't hand back code you haven't run.
