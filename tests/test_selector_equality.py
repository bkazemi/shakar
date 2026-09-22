from __future__ import annotations

import pytest

from tests.support.harness import run_runtime_case


# Operator-level coverage. `shk_equals` has its own unit tests in
# test_collections.py, but those call the function directly and so never
# exercised the `==` dispatch path, which used to route selector-vs-selector
# comparisons through scalar membership and report every selector unequal to
# itself.
SCENARIOS = [
    pytest.param(
        "s := `1:3`; s == s",
        ("bool", True),
        None,
        id="selector-equals-itself",
    ),
    pytest.param(
        "s := `1:3`; s != s",
        ("bool", False),
        None,
        id="selector-not-unequal-to-itself",
    ),
    pytest.param(
        "`1:3` == `1:3`",
        ("bool", True),
        None,
        id="identical-literals-equal",
    ),
    # Spellings that visit the same positions compare equal: the exclusive-stop
    # marker and an explicit unit step are notation, not content.
    pytest.param(
        "`1:3` == `1:<4`",
        ("bool", True),
        None,
        id="inclusive-equals-exclusive-spelling",
    ),
    pytest.param(
        "`1:3` == `1:3:1`",
        ("bool", True),
        None,
        id="implicit-equals-explicit-unit-step",
    ),
    pytest.param(
        "`0:<10:2` == `0:8:2`",
        ("bool", True),
        None,
        id="stepped-slack-in-stop-normalized",
    ),
    pytest.param(
        "`3:2` == `3:<3`",
        ("bool", True),
        None,
        id="empty-ranges-at-same-start-equal",
    ),
    # Negative bounds resolve against the indexed collection's length, so a
    # mixed-sign span is not a fixed position count and must not be trimmed
    # or collapsed. `0:<{-1}` selects [10,20] from [10,20,30]; `0:<0` is empty.
    pytest.param(
        "`0:<{-1}` == `0:<0`",
        ("bool", False),
        None,
        id="mixed-sign-bounds-not-collapsed-to-empty",
    ),
    pytest.param(
        "`0:-2:2` == `0:<0:2`",
        ("bool", False),
        None,
        id="stepped-mixed-sign-bounds-not-collapsed",
    ),
    pytest.param(
        "`0:<{-1}` == `0:<{-2}`",
        ("bool", False),
        None,
        id="mixed-sign-stops-kept-distinct",
    ),
    # `<-1` lexes as RECV + NUMBER; the selector parser splits it back into an
    # exclusive marker and a negative stop, matching the `<{-1}` spelling.
    pytest.param(
        "`0:<-1` == `0:<{-1}`",
        ("bool", True),
        None,
        id="exclusive-negative-stop-without-braces",
    ),
    pytest.param(
        "[10, 20, 30][`0:<-1`] == [10, 20]",
        ("bool", True),
        None,
        id="exclusive-negative-stop-indexes-correctly",
    ),
    pytest.param(
        "[10, 20, 30, 40][`0:<-1:2`] == [10, 30]",
        ("bool", True),
        None,
        id="exclusive-negative-stop-with-step",
    ),
    # Stop trimming is only length-independent for forward slices with
    # non-negative bounds. Elsewhere clamping or negative-index resolution can
    # shift the step origin, so slack in the stop stays significant.
    pytest.param(
        "`10:2:-3` == `10:4:-3`",
        ("bool", False),
        None,
        id="clamped-reverse-start-keeps-stop-distinct",
    ),
    pytest.param(
        "[0, 1, 2, 3, 4, 5][`10:2:-3`] == [5, 2]",
        ("bool", True),
        None,
        id="clamped-reverse-start-indexes-from-clamped-origin",
    ),
    pytest.param(
        "`-9:<-8:2` == `-9:<-7:2`",
        ("bool", False),
        None,
        id="negative-forward-slack-kept-distinct",
    ),
    # Inclusive-to-exclusive conversion is spelling only, so it still applies
    # to negative bounds.
    pytest.param(
        "`-4:<-1:2` == `-4:-2:2`",
        ("bool", True),
        None,
        id="negative-inclusive-equals-exclusive-spelling",
    ),
    # Index parts are never folded into slices: indices throw on out-of-bounds
    # and yield scalars, slices clamp and yield arrays, so the kinds differ
    # observably even when they select the same positions on a long enough base.
    pytest.param(
        "`1:3` == `1,2,3`",
        ("bool", False),
        None,
        id="slice-not-equal-to-index-list",
    ),
    pytest.param(
        "`2` == `2:2`",
        ("bool", False),
        None,
        id="index-not-equal-to-singleton-slice",
    ),
    pytest.param(
        "`1:3` == `1:4`",
        ("bool", False),
        None,
        id="different-extents-unequal",
    ),
    # Selection order is meaningful, so part order is part of identity.
    pytest.param(
        "`1,2` == `2,1`",
        ("bool", False),
        None,
        id="part-order-significant",
    ),
    # Scalar-vs-selector comparisons still mean membership.
    pytest.param(
        "2 == `1:3`",
        ("bool", True),
        None,
        id="scalar-membership-hit",
    ),
    pytest.param(
        "9 == `1:3`",
        ("bool", False),
        None,
        id="scalar-membership-miss",
    ),
    pytest.param(
        "9 != `1:3`",
        ("bool", True),
        None,
        id="scalar-membership-negated",
    ),
    # Equality through a container reaches the same comparison.
    pytest.param(
        "s := `1:3`; [s] == [s]",
        ("bool", True),
        None,
        id="selector-equality-inside-array",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_selector_equality(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)
