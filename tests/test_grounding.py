"""Grounding tests. Pure functions, no API key, no network."""

from __future__ import annotations

from scisolve.grounding import check_grounding


def test_exact_match():
    report = check_grounding("The root is 1.4142135624.", ["root = 1.4142135624"])
    assert report.grounded
    assert report.unmatched == ()
    assert report.matched == (("1.4142135624", 1.4142135624),)


def test_rounded_display_is_accepted():
    report = check_grounding("ln(2) = 0.6931", ["0.6931471805599453"])
    assert report.grounded


def test_rounding_beyond_rtol_is_rejected():
    report = check_grounding("the value is 0.69", ["0.6931471805599453"])
    assert not report.grounded
    assert report.unmatched == ("0.69",)


def test_rtol_is_configurable():
    assert check_grounding("0.69", ["0.6931471805599453"], rtol=1e-2).grounded


def test_scientific_notation_matches_decimal():
    report = check_grounding("residual 1.23e-4", ["residual = 0.000123"])
    assert report.grounded


def test_decimal_answer_matches_scientific_output():
    assert check_grounding("the error is 0.000123", ["1.23e-04"]).grounded


def test_percentage_matches_a_fraction_in_output():
    report = check_grounding("relative error 45%", ["rel_err = 0.4503"])
    assert report.grounded
    assert report.matched[0][0] == "45%"


def test_percentage_matches_a_percentage_in_output():
    assert check_grounding("45%", ["45.01 percent"]).grounded


def test_thousands_separator_is_stripped():
    assert check_grounding("n = 1,048,576 samples", ["1048576"]).grounded


def test_negative_number():
    report = check_grounding("lambda_min = -3.7412", ["eigenvalue: -3.74119"])
    assert report.grounded


def test_number_from_the_problem_statement_is_allowed():
    report = check_grounding(
        "with damping ratio 0.15 the amplitude decays",
        [],
        problem="a damped oscillator with zeta = 0.15",
    )
    assert report.grounded


def test_invented_number_is_caught():
    report = check_grounding(
        "at frequency 1.0 Hz the period is 6.2832 s",
        ["frequency = 1.0"],
    )
    assert not report.grounded
    assert report.unmatched == ("6.2832",)
    assert report.matched == (("1.0", 1.0),)


def test_several_invented_numbers_are_all_reported_in_order():
    report = check_grounding("x = 12.5, y = 88.25", ["nothing useful here"])
    assert report.unmatched == ("12.5", "88.25")


def test_repeated_literal_is_reported_once():
    report = check_grounding("12.5 and again 12.5", [])
    assert report.unmatched == ("12.5",)


def test_traceback_line_numbers_do_not_ground_anything():
    # agent.py passes stdout from successful cells only, so a value whose only
    # support is a traceback is ungrounded:
    assert not check_grounding("the root is 37.5", outputs=[]).grounded

    # this is exactly why the filtering is the caller's job -- pass a traceback
    # in and line numbers will ground almost anything:
    leaky = 'File "<cell>", line 37, in <module>\n  x = 5 / 0\nZeroDivisionError'
    assert check_grounding("the root is 37.5", outputs=[leaky], rtol=2e-2).grounded


def test_figure_paths_are_not_offered_as_outputs():
    # same contract: only stdout goes in. fig_002.png must not ground "2".
    assert not check_grounding("amplitude 0.002", outputs=[]).grounded


def test_small_integers_are_structural():
    report = check_grounding("there are 3 roots and the method is 2nd order", [])
    assert report.grounded
    assert dict(report.matched) == {"3": 3.0, "2": 2.0}


def test_integers_above_the_structural_cap_must_be_computed():
    assert not check_grounding("there are 11 roots", []).grounded
    assert check_grounding("there are 11 roots", ["n_roots = 11"]).grounded


def test_a_decimal_is_never_structural():
    assert not check_grounding("the answer is 3.0", []).grounded
    assert check_grounding("the answer is 3.0", ["x = 3.0"]).grounded


def test_a_small_percentage_is_never_structural():
    assert not check_grounding("it improved by 5%", []).grounded


def test_zero_matches_numerical_zero():
    assert check_grounding("the residual is 0.0", ["residual = 0.0"]).grounded


def test_near_zero_values_both_count_as_zero():
    assert check_grounding("residual 0.0", ["residual = 1e-15"]).grounded


def test_nonzero_does_not_match_zero():
    assert not check_grounding("the residual is 0.5", ["residual = 0.0"]).grounded


def test_empty_answer_is_vacuously_grounded():
    report = check_grounding("", ["1.0"])
    assert report.grounded
    assert report.matched == ()
    assert report.unmatched == ()


def test_prose_without_numbers_is_vacuously_grounded():
    # documented limitation: an unchecked prose claim passes
    assert check_grounding("the system is asymptotically stable", []).grounded


def test_identifiers_containing_digits_are_not_treated_as_numbers():
    report = check_grounding("stored in fig_001 and x2 and h5py", [])
    assert report.grounded
    assert report.matched == ()


def test_multiple_outputs_are_all_searched():
    report = check_grounding("x = 7.25", ["first cell: 1.0", "second cell: 7.25"])
    assert report.grounded


def test_matched_reports_the_output_value_not_the_literal():
    report = check_grounding("0.6931", ["0.6931471805599453"])
    assert report.matched[0][1] == 0.6931471805599453
