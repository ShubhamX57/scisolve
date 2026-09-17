# Changelog

Notable changes. Format follows [Keep a Changelog](https://keepachangelog.com/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `Executor`: persistent in-process Python session with stdout capture,
  tracebacks returned as text with scisolve's own frames stripped, figure
  capture, output truncation, and a `signal.setitimer` cell timeout behind an
  injectable strategy.
- `check_grounding`: numeric provenance check over an answer, matching literals
  against stdout of successful cells, the problem statement, or bare integers
  0–10, at a relative tolerance of 1e-3.
- `Agent`: the loop. Handles every `tool_use` block, replies with one
  `tool_result` per block in order, feeds tracebacks back, and validates
  `finish_solution` on verification kind, evidence index and grounding —
  re-prompting once, then returning the answer marked `grounded=False`.
- `transport.py`: raw `requests` against the Messages API, with the retry
  decision isolated in one pure function. Honours `retry-after`, backs off
  further on 529 than on 429, never retries 400/401/403/404/413, and caps both
  attempts and total wait so a spend-cap 429 fails with a message saying so.
- `scisolve` CLI: streams turns as they happen and writes
  `runs/<utc-timestamp>/` with the solution, the full transcript, every code
  cell and the figures. Exit codes 0 solved and grounded, 1 unfinished or
  ungrounded, 2 could not run.
- Seven self-verifying examples across ODEs, random matrices, constrained
  optimization, root finding, statistics, quadrature and PDEs, re-run in CI
  with no API key.
- `examples/failures/`: executable demonstrations of holes in the grounding
  check, each asserting the failure still happens.
- `scripts/record_cassette.py` and `tests/test_cassette.py`: record one real API
  run, scrub the key, replay it forever with no network.
- CI on Python 3.10–3.13.

### Known limitations
- The loop has not yet run against the live API; `transport.py` is verified
  against the documentation and hand-written fakes only.
- An answer can quote more significant digits than were printed and still pass
  the grounding check within `rtol`.
- An answer composed entirely of bare integers 0–10 passes with no computation
  behind it.
