# Failure gallery

Cases where scisolve gets it wrong, or where a check passes that should not.
This directory exists because a project whose entire claim is "it verifies
before it reports" has to show where that verification is thin.

Every file here is executable and asserts that the failure **still happens**.
If you close one of these holes, the corresponding script fails, and that is the
signal to rewrite the entry rather than quietly delete it.

## What is here now

- `grounding_precision_overrun.py` — an answer quoting more digits than were
  ever printed passes the grounding check.
- `grounding_structural_integers.py` — an answer made entirely of small
  integers passes with no computation behind it.

Both were found by running the thing, not by reading the code.

## What is not here yet

Recorded transcripts of the agent itself failing — giving a wrong answer or
giving up on a solvable problem. Those need real runs against the real API,
which is milestone 6. Writing plausible-looking transcripts before then would
be inventing evidence, which is the exact failure mode this project is built
to prevent.
