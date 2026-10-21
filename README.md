# scisolve

[![ci](https://github.com/ShubhamX57/scisolve/actions/workflows/ci.yml/badge.svg)](https://github.com/ShubhamX57/scisolve/actions/workflows/ci.yml)

Solves scientific computing problems - ODEs, optimization, curve fitting, linear
algebra, statistics - by writing and running NumPy/SciPy code in a loop, and
refuses to report a number it did not compute in that session.

> **scisolve executes model-generated code in your own Python process.** Treat it
> like running a script a stranger emailed you. Don't point it at untrusted
> problem statements.

## Status

Milestones 0–5 of 7 are done: the executor, the grounding check, the agent loop,
the CLI, and seven self-verifying examples. 122 tests, all passing on Python
3.10 through 3.13, none of them requiring an API key.

**The loop has never run against the live API.** Every test drives it through a
scripted transport. The wire format is written from the current Messages API
documentation, and a fake cannot tell you the documentation is right. Until that
smoke test happens, treat `transport.py` as unverified against the real thing,
and treat this README's claims about end-to-end behaviour as claims about code
that passes its own tests rather than code anyone has watched work.

A recorded demo goes here once there is a real run to record.

## The one rule

Every number in a scisolve answer must be traceable to output captured from a
`run_python` call in the same session.

This is enforced in `scisolve/grounding.py`, not requested in a prompt. It is the
only thing here that a weekend API wrapper doesn't have.

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

Turns stream as they happen - the code, then its output:

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
checked: invariant - trace and determinant match the sum and product of the
eigenvalues (evidence: run_python call 1)
run: runs/2026-09-16T10-23-45Z
```

That transcript came from the test harness driving the loop with scripted model
turns. The code ran and the numbers are real; the model's side was scripted, not
generated. It shows the shape of a run, not evidence that a model produces one.

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

This is written even when the run fails - an API error or a `max_turns` exit
still leaves a complete directory. An answer you cannot retrace is the thing
this project exists to avoid.

## How the grounding check works

Numeric literals are pulled out of the answer (decimals, scientific notation,
thousands separators, percentages) and each one has to match a number from:

- stdout of cells that **succeeded** - tracebacks are excluded, since line
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
- **Precision.** An answer can quote more digits than were printed and still
  match within `rtol`.
- **The right variable.** It matches values, not names. Quoting the residual
  where you meant the root still grounds.

The last two are demonstrated as running code in
[`examples/failures/`](examples/failures/), along with the case where an answer
made entirely of small integers passes with no computation behind it. Those
scripts assert that the failure still happens, so closing a hole breaks the
build and forces the entry to be rewritten instead of quietly deleted.

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

Mostly, plus two things. The session is persistent, so variables, imports and
fitted models survive between calls and the model builds on its own state rather
than re-deriving it in every cell. And the grounding check is enforced in code:
a `finish_solution` whose numbers are not in the session's captured output is
rejected and sent back, and if it fails twice the answer is returned marked
ungrounded rather than presented as a result.

Take that as a description of what the code does, which the test suite covers,
not as a claim about how often it helps in practice. That claim needs the real
runs, and they haven't happened.

## Why not just use Jupyter?

For most work, Jupyter is fine, and if you are going to read the output and check
it yourself then Jupyter plus your own judgement is strictly better than this.

scisolve is for when you want the checking to be mechanical rather than
attentive: when the thing producing the answer is a language model that will
write a plausible number if you let it, and you want that failure mode closed off
by code rather than by remembering to look. The audit trail is the other half -
every claim in `runs/` has its cell, its output and its check sitting next to it.

## Examples

Seven, across genuinely different domains, each one checked by a computation
different from the one it is checking - a closed form, a conservation law, an
independent algorithm, or a measured convergence rate against a predicted one.
`tests/test_examples.py` re-runs all of them in CI with no API key, so a wrong
answer fails the build.

See [`examples/README.md`](examples/README.md).

## Contributing

`CONTRIBUTING.md` lands with the milestone 7 docs pass. Until then: run
`python -m pytest` and `ruff check .` before opening anything, put a test
alongside any change to `executor.py`, `agent.py` or `grounding.py`, and give a
new example a real check - a closed form, a conservation law, or an independent
method. If the only way you can verify it is to run the same computation twice,
it is not an example yet.

The one rule is not negotiable. A change that weakens the grounding check has to
say so out loud in the pull request rather than arriving as a refactor.

## License

MIT.
