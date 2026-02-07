from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    ParseError,
    run_runtime_case,
)

SCENARIOS = [
    pytest.param(
        "7 // 2",
        ("number", 3),
        None,
        id="floor-div-basic",
    ),
    pytest.param(
        "-7 // 2",
        ("number", -4),
        None,
        id="floor-div-negative",
    ),
    pytest.param(
        "a := 1; a += 2; a",
        ("number", 3),
        None,
        id="compound-assign-number",
    ),
    pytest.param(
        's := "a"; s += "b"; s',
        ("string", "ab"),
        None,
        id="compound-assign-string",
    ),
    pytest.param(
        "a := 10; a %= 3; a",
        ("number", 1),
        None,
        id="compound-assign-mod",
    ),
    pytest.param(
        "a := 10; a -= 4; a",
        ("number", 6),
        None,
        id="compound-assign-minus",
    ),
    pytest.param(
        "a := 3; a *= 4; a",
        ("number", 12),
        None,
        id="compound-assign-mul",
    ),
    pytest.param(
        "a := 9; a /= 2; a",
        ("number", 4.5),
        None,
        id="compound-assign-div",
    ),
    pytest.param(
        "a := 9; a //= 2; a",
        ("number", 4),
        None,
        id="compound-assign-floordiv",
    ),
    pytest.param(
        dedent(
            """\
            profile := { name: "  Ada " }
            profile.name .= .trim()
            profile.name
        """
        ),
        ("string", "Ada"),
        None,
        id="applyassign-chain",
    ),
    pytest.param(
        "2 ** 3",
        ("number", 8),
        None,
        id="power-basic",
    ),
    pytest.param(
        "2 ** 3 ** 2",
        ("number", 512),
        None,
        id="power-precedence",
    ),
    pytest.param(
        "(-2) ** 3",
        ("number", -8),
        None,
        id="power-negative",
    ),
    pytest.param(
        "x := 2; x **= 3; x",
        ("number", 8),
        None,
        id="power-assign",
    ),
    pytest.param(
        "a := 5; a++; a",
        ("number", 6),
        None,
        id="postfix-incr-basic",
    ),
    pytest.param(
        "a := 5; a--; a",
        ("number", 4),
        None,
        id="postfix-decr-basic",
    ),
    pytest.param(
        "a := 5; ++a; a",
        ("number", 6),
        None,
        id="prefix-incr-basic",
    ),
    pytest.param(
        "a := 5; --a; a",
        ("number", 4),
        None,
        id="prefix-decr-basic",
    ),
    pytest.param(
        "~5",
        None,
        ParseError,
        id="unary-tilde-rejected",
    ),
    pytest.param(
        "a := 10; a %= 3; a",
        ("number", 1),
        None,
        id="compound-mod",
    ),
    pytest.param(
        "a := 10; a -= 4; a",
        ("number", 6),
        None,
        id="compound-minus",
    ),
    pytest.param(
        "a := 3; a *= 4; a",
        ("number", 12),
        None,
        id="compound-mul",
    ),
    pytest.param(
        "a := 9; a /= 2; a",
        ("number", 4.5),
        None,
        id="compound-div",
    ),
    pytest.param(
        "a := 9; a //= 2; a",
        ("number", 4),
        None,
        id="compound-floordiv",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_operators(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)
