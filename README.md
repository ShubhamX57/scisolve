# scisolve

Solves scientific computing problems by writing and running NumPy/SciPy code in
a loop, and only reports an answer it has checked in that same session.

**Status: in development.** Milestones 0 and 1 of 7 are done (package skeleton
and the persistent executor). There is no agent yet. This README is a
placeholder; the real one lands at milestone 7.

> scisolve executes model-generated code in your own Python process. Treat it
> like running a script a stranger emailed you. Don't point it at untrusted
> problem statements.

## The one rule

Every number in a scisolve answer must be traceable to captured output from a
`run_python` call in the same session. This is enforced in code
(`scisolve/grounding.py`), not just asked for in a prompt.

## What it isn't

- Not a computer algebra system. No symbolic proofs.
- **Not a security sandbox.** It runs model-generated code in your own process.
- Not multi-provider. One model API, deliberately.
- Not a substitute for domain review. It checks its arithmetic, not your physics.
- No success-rate claims without a fixed problem set, a pinned commit, and
  published transcripts.

## Install (development)

```bash
pip install -e ".[dev]"
pytest
```

Python 3.10+.

## License

MIT.
