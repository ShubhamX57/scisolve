"""Check that every number in an answer came from output the session actually produced.

What this checks: **provenance**. Each numeric literal in the answer is matched
against numbers that appeared in captured stdout, in the problem statement, or
against a small set of structural integers.

What it does not check, and cannot:

* **Correctness.** A number printed by wrong code is grounded. The check says
  "this was computed here", never "this is right".
* **Coincidence.** An invented value that happens to sit within ``rtol`` of some
  unrelated number in the output passes. Longer outputs make this likelier.
* **Prose.** "The solution is stable" carries no digits, so nothing is verified.
  Claims without numbers pass through untouched.
* **Provenance of the right number.** It matches values, not variables. An
  answer quoting the residual where it meant the root still grounds.

A literal that sits within ``rtol`` of a printed number but claims more
significant figures than that number was printed to is reported in
``overprecise`` rather than ``matched``, and the answer is not grounded. The
value is traceable; the extra digits are not.

The caller is responsible for passing **stdout from successful cells only**.
Tracebacks must not be included: line numbers, memory addresses and array shapes
in a traceback would ground almost anything. See
``test_traceback_line_numbers_do_not_ground_anything``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

#: Answers routinely contain small counting integers ("three roots", "second
#: order"). Requiring those to be printed produces noise, so they pass. Only
#: bare integer literals qualify: "3" is structural, "3.0" and "3%" are claims.
STRUCTURAL_INT_MAX = 10

_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])
    (?P<num>
        [-+]?
        (?:
            \d{1,3}(?:,\d{3})+(?:\.\d+)?   # 1,234  1,234.56
          | \d+\.\d+                        # 1.5
          | \.\d+                           # .5
          | \d+                             # 42
        )
        (?:[eE][-+]?\d+)?                   # 1.23e-4
    )
    (?P<pct>\s*%)?
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class GroundingReport:
    """Result of a provenance check.

    ``matched`` pairs each grounded literal with the output value it matched
    (for structural integers, with itself). ``unmatched`` holds the literals, as
    written, that nothing in the session supports. Repeated literals are
    reported once.
    """

    grounded: bool
    unmatched: tuple[str, ...]
    matched: tuple[tuple[str, float], ...]
    overprecise: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Literal:
    text: str
    value: float
    is_percent: bool
    is_plain_int: bool
    digits: int

    @property
    def candidates(self) -> tuple[float, ...]:
        """Values that would count as a match for this literal."""
        if self.is_percent:
            return (self.value, self.value / 100.0)
        return (self.value,)


def _significant_digits(raw: str) -> int:
    """Significant figures in a literal as written.

    Trailing zeros after a decimal point are formatting, not precision, so
    "5.00" counts as one. Zeros before the point are significant, so "100.0"
    counts as three.
    """
    text = raw.replace(",", "").lstrip("+-").lower()
    if "e" in text:
        text = text.split("e")[0]
    if "." in text:
        whole, _, frac = text.partition(".")
        digits = (whole + frac.rstrip("0")).lstrip("0")
    else:
        digits = text.lstrip("0")
    return max(len(digits), 1)


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "").replace("+", "", 1))
    except ValueError:
        return None


def _literals(text: str) -> list[_Literal]:
    """Every numeric literal in ``text``, in order of appearance."""
    found: list[_Literal] = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group("num")
        value = _to_float(raw)
        if value is None:
            continue
        is_percent = match.group("pct") is not None
        found.append(
            _Literal(
                text=(raw + "%") if is_percent else raw,
                value=value,
                is_percent=is_percent,
                is_plain_int=not is_percent and re.fullmatch(r"[-+]?\d+", raw) is not None,
                digits=_significant_digits(raw),
            )
        )
    return found


def _values(text: str) -> list[tuple[float, int]]:
    """Numbers available for matching, each with the precision it was printed to."""
    pool: list[tuple[float, int]] = []
    for lit in _literals(text):
        pool.extend((candidate, lit.digits) for candidate in lit.candidates)
    return pool


def _close(a: float, b: float, rtol: float) -> bool:
    if abs(a) <= 1e-12 and abs(b) <= 1e-12:
        return True
    return abs(a - b) <= rtol * max(abs(a), abs(b))


def check_grounding(
    answer: str,
    outputs: Sequence[str],
    *,
    problem: str = "",
    rtol: float = 1e-3,
) -> GroundingReport:
    """Report which numbers in ``answer`` are traceable to ``outputs``.

    ``outputs`` must be stdout from successful cells only. ``rtol`` defaults to
    1e-3 so that display rounding passes: ``0.6931`` matches
    ``0.6931471805599453``.
    """
    pool: list[tuple[float, int]] = []
    for out in outputs:
        pool.extend(_values(out))
    pool.extend(_values(problem))

    matched: list[tuple[str, float]] = []
    unmatched: list[str] = []
    overprecise: list[str] = []
    decided: set[str] = set()

    for lit in _literals(answer):
        if lit.text in decided:
            continue
        decided.add(lit.text)

        if lit.is_plain_int and 0 <= lit.value <= STRUCTURAL_INT_MAX:
            matched.append((lit.text, lit.value))
            continue

        near = [
            (value, digits)
            for value, digits in pool
            for candidate in lit.candidates
            if _close(candidate, value, rtol)
        ]
        if not near:
            unmatched.append(lit.text)
            continue

        # Of the numbers this literal is close to, was any printed to at least
        # the precision the literal claims? Matching 1.381966011250105 against a
        # printed 1.38196601 leaves seven digits that nothing computed.
        precise_enough = [pair for pair in near if lit.digits <= pair[1]]
        if precise_enough:
            matched.append((lit.text, precise_enough[0][0]))
        else:
            overprecise.append(lit.text)

    return GroundingReport(
        grounded=not unmatched and not overprecise,
        unmatched=tuple(unmatched),
        matched=tuple(matched),
        overprecise=tuple(overprecise),
    )
