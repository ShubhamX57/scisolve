"""KNOWN FAILURE: an answer of small integers grounds with no computation.

check_grounding treats bare integers 0 to 10 as structural -- "three roots",
"second order" -- because requiring them to be printed produces noise. The cost
is that an answer whose numbers all happen to be small integers is accepted
without a single line of code having produced them.

This is a real hole in the one rule, not a rounding detail. It is narrowed as
far as it can be without a decision: only bare integer literals qualify, so
3.0, 3% and 3.1 are claims that must be printed. Reporting structural matches
separately in GroundingReport would surface it to the user; that needs a third
field on the dataclass.
"""

from scisolve.grounding import check_grounding

invented = "The system has 3 equilibria, at x = 0, 1 and 2."
report = check_grounding(invented, outputs=[])

print("answer:", invented)
print("outputs the session produced: none")
print("grounded =", report.grounded)
print("matched =", report.matched)
print("unmatched =", report.unmatched)

honest = "The system has 3 equilibria, at x = 0, 1 and 2.5."
honest_report = check_grounding(honest, outputs=[])
print()
print("change one value to 2.5 and the same answer is caught:")
print("grounded =", honest_report.grounded, " unmatched =", honest_report.unmatched)

assert report.grounded, "this entry is out of date: the hole appears to be closed"
assert not honest_report.grounded
print("STILL FAILING: an answer with no computation behind it was accepted")
