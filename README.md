# scisolve

[![ci](https://github.com/ShubhamX57/scisolve/actions/workflows/ci.yml/badge.svg)](https://github.com/ShubhamX57/scisolve/actions/workflows/ci.yml)

Solves scientific computing problems — ODEs, optimization, curve fitting, linear
algebra, statistics — by writing and running NumPy/SciPy code in a loop, and
refuses to report a number it did not compute in that session.

> **scisolve executes model-generated code in your own Python process.** Treat it
> like running a script a stranger emailed you. Don't point it at untrusted
> problem statements.

## The one rule

Every number in a scisolve answer must be traceable to output captured from a
`run_python` call in the same session.

This is enforced in `scisolve/grounding.py`, not requested in a prompt. Before an
answer is accepted, every numeric literal in it is matched against stdout from
the cells that actually succeeded. Anything unmatched goes back to the model
naming the specific value, and an answer that fails twice is returned marked
ungrounded with the offending numbers listed rather than presented as a result.

Each solution also has to name the check it ran, from a fixed list:

| Kind | Means |
|---|---|
| `analytic` | matched a closed form |
| `independent_method` | a second algorithm agrees |
| `limiting_case` | reduces to a known result in a limit |
| `invariant` | conservation law, residual norm, KKT, orthogonality |
| `convergence` | refine the grid or tolerance, the answer is stable |
| `dimensional` | the units are consistent |

## Install

```bash
git clone https://github.com/ShubhamX57/scisolve.git
cd scisolve
pip install -e ".[dev]"
python -m pytest
```

Python 3.10+. Runtime dependencies are `requests`, `numpy`, `scipy`,
`matplotlib`. The test suite needs no API key and makes no network calls.

## Use

```bash
export ANTHROPIC_API_KEY=sk-ant-...
scisolve "find the eigenvalues of [[2,1],[1,3]] and verify them"
```

Turns stream as they happen — the code, then its output:

```
problem: find the eigenvalues of [[2,1],[1,3]] and verify them

[0] run_python
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    w = np.linalg.eigvalsh(A)
    print('eigenvalues', repr(w.tolist()))
  out eigenvalues [1.381966011250105, 3.618033988749895]

[1] run_python
    print('trace check', A.trace(), w.sum())
    print('det check', np.linalg.det(A), w.prod())
  out trace check 5.0 5.0
  out det check 5.000000000000001 5.0

answer: The eigenvalues are 1.381966011250105 and 3.618033988749895.
checked: invariant — trace and determinant match the sum and product of the
eigenvalues (evidence: run_python call 1)
run: runs/2026-09-16T10-23-45Z
```

That transcript came from the test harness driving the loop with scripted model
turns: the code ran and the numbers are real, but the model's side was written
rather than generated. See Status, below.

From Python:

```python
from scisolve import Agent

solution = Agent().solve("integrate ln(sin x) from 0 to pi/2")
print(solution.answer)
print(solution.grounded, solution.verification.kind)
print(solution.ungrounded)   # values that matched nothing, if any
```

### Options

```
scisolve "problem" [--model M] [--max-turns N] [--timeout S]
                   [--out DIR] [--json] [--quiet] [--version]
```

Exit codes: `0` solved and grounded, `1` unfinished or ungrounded, `2` could not
run at all.

### The run directory

Every run writes `runs/<utc-timestamp>/`:

```
solution.json     answer, verification, grounded, ungrounded, stop reason
transcript.json   every message, verbatim
cells/            each code cell as a numbered file, failed ones included
figures/          anything the session plotted
```

This is written even when the run fails — an API error or a `max_turns` exit
still leaves a complete directory. An answer you cannot retrace is the thing
this project exists to avoid.

## Status

Everything but the live run: the executor, the grounding check, the agent loop,
the CLI and its audit trail, seven self-verifying examples, a failure gallery,
and the docs. 139 tests on Python 3.10 through 3.13, none of them requiring an
API key. Four more are written and skipped, waiting on a recording.

**The loop has not yet run against the live API.** Every test drives it through a
scripted transport, and a fake cannot tell you that the wire format it was
written from is still right. Until a recording exists, `transport.py` is verified
against the documentation rather than against the service, and the transcript
above shows the shape of a run rather than evidence that a model produces one.

A recorded demo goes here once there is a real run to record.

## How the grounding check works

Numeric literals are pulled out of the answer (decimals, scientific notation,
thousands separators, percentages) and each one has to match a number from:

- stdout of cells that **succeeded** — tracebacks are excluded, since line
  numbers and array shapes would ground almost anything
- the problem statement itself
- or be a bare integer from 0 to 10, which are usually structural ("three
  roots", "second order")

A literal matches when `abs(a - g) <= 1e-3 * max(abs(a), abs(g))`, so display
rounding passes: `0.6931` matches `0.6931471805599453`. Anything unmatched comes
back to the model naming the specific value, once. If the second attempt is also
ungrounded, the answer is returned with `grounded=False` and the offending
literals listed, rather than argued with further.

**What it does not catch**, and cannot:

- **Correctness.** A number printed by wrong code is grounded. The check says
  "this was computed here", never "this is right".
- **Coincidence.** An invented value within `rtol` of some unrelated number in
  the output passes.
- **Prose.** "The solution is stable" carries no digits, so nothing is verified.
- **The right variable.** It matches values, not names. Quoting the residual
  where you meant the root still grounds.

A literal within `rtol` of a printed number that claims more significant
figures than that number was printed to is rejected separately, in
`overprecise`: the value is traceable, the extra digits are not.

[`examples/failures/`](examples/failures/) holds these as running code —
including the case where an answer made entirely of small integers passes with
no computation behind it. Each asserts that its failure still happens, so
closing a hole breaks the build and forces the entry to be rewritten instead of
quietly deleted. `failures/closed/` is where they go once fixed, as regression
guards.

## What it isn't

- Not a computer algebra system. No symbolic proofs.
- **Not a security sandbox.** It runs model-generated code in your own process,
  with your filesystem, your network and your environment. There is no memory
  limit and no network block, because an in-process guard against code running
  in the same process is bypassable, and a guard this README would have to lie
  about is worse than a documented absence. The cell timeout uses
  `signal.setitimer`, so it is Unix-and-main-thread only, it cannot interrupt a
  long call inside compiled code, and model code can swallow it with a bare
  `except Exception`.
- Not multi-provider. One model API, deliberately.
- Not a substitute for domain review. It checks its arithmetic, not your physics.
- No success-rate claims. There is no fixed problem set, no pinned commit and no
  published transcripts yet, so any number here would be decoration.

## Isn't this just an API wrapper?

It calls one API, and two things sit on top of it that change what comes back.

The session is persistent. Variables, imports and fitted models survive between
calls, so the model builds on its own state instead of re-deriving it in every
cell — and the verification step can use what the solution step left behind.

The grounding check runs in code rather than in the prompt. A `finish_solution`
whose numbers are not in the session's captured output is rejected and sent back
naming the specific value; if the second attempt still fails, the answer comes
back `grounded=False` with the offending literals listed and the CLI exits 1.
Each solution also has to name which check it ran and point at the cell that
produced it, and that is validated too.

So what you get back is not the model's answer, it is the model's answer plus a
machine-checked claim about where each number in it came from. That describes
what the code does, which the test suite covers. How much it helps in practice
is a separate question and needs real runs.

## Why not just use Jupyter?

For most work, Jupyter is fine, and if you are going to read the output and check
it yourself then Jupyter plus your own judgement is strictly better than this.

scisolve is for when you want the checking to be mechanical rather than
attentive: when the thing producing the answer is a language model that will
write a plausible number if you let it, and you want that failure mode closed off
by code rather than by remembering to look. The audit trail is the other half —
every claim in `runs/` has its cell, its output and its check sitting next to it.

## Examples

Seven, across genuinely different domains, each one checked by a computation
different from the one it is checking — a closed form, a conservation law, an
independent algorithm, or a measured convergence rate against a predicted one.
`tests/test_examples.py` re-runs all of them in CI with no API key, so a wrong
answer fails the build.

See [`examples/README.md`](examples/README.md).

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) covers running the tests, the bar a new
example has to clear, and the changes that need a decision rather than a commit.

The one rule is not negotiable. A change that weakens the grounding check has to
say so out loud in the pull request rather than arriving as a refactor.

## License

MIT.
