from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    ShakarMatchError,
    run_runtime_case,
)

SCENARIOS = [
    pytest.param(
        dedent(
            """\
            x := 2
            match x:
              1: "one"
              2: "two"
              else: "other"
            
        """
        ),
        ("string", "two"),
        None,
        id="match-basic",
    ),
    pytest.param(
        dedent(
            """\
            obj := { val: 3 }
            match obj:
              obj: .val
              else: 0
            
        """
        ),
        ("number", 3),
        None,
        id="match-dot",
    ),
    pytest.param(
        dedent(
            """\
            score := 85
            match score:
              `90:100`: "A"
              `80:<90`: "B"
              else: "F"
            
        """
        ),
        ("string", "B"),
        None,
        id="match-selector",
    ),
    pytest.param(
        dedent(
            """\
            score := 85
            match[lt] score:
              90: "A"
              80: "B"
              else: "F"
            
        """
        ),
        ("string", "B"),
        None,
        id="match-cmp-lt",
    ),
    pytest.param(
        dedent(
            """\
            x := 2
            match[==] x:
              2: "hit"
              else: "miss"
            
        """
        ),
        ("string", "hit"),
        None,
        id="match-cmp-eq-symbol",
    ),
    pytest.param(
        dedent(
            """\
            x := 2
            match[ne] x:
              2: "nope"
              3: "hit"
              else: "miss"
            
        """
        ),
        ("string", "hit"),
        None,
        id="match-cmp-ne-roman",
    ),
    pytest.param(
        dedent(
            """\
            score := 85
            match[gt] score:
              60: "low"
              90: "high"
              else: "mid"
            
        """
        ),
        ("string", "high"),
        None,
        id="match-cmp-gt",
    ),
    pytest.param(
        dedent(
            """\
            score := 90
            match[ge] score:
              90: "ok"
              else: "no"
            
        """
        ),
        ("string", "ok"),
        None,
        id="match-cmp-ge",
    ),
    pytest.param(
        dedent(
            """\
            score := 70
            match[le] score:
              70: "ok"
              else: "no"
            
        """
        ),
        ("string", "ok"),
        None,
        id="match-cmp-le",
    ),
    pytest.param(
        dedent(
            """\
            ch := "a"
            match[in] ch:
              "aeiou": "vowel"
              "0123456789": "digit"
              else: "other"
            
        """
        ),
        ("string", "vowel"),
        None,
        id="match-cmp-in",
    ),
    pytest.param(
        dedent(
            """\
            ch := "b"
            match[!in] ch:
              "aeiou": "consonant"
              else: "vowel"
            
        """
        ),
        ("string", "consonant"),
        None,
        id="match-cmp-not-in",
    ),
    pytest.param(
        dedent(
            """\
            ch := "b"
            match[not in] ch:
              "aeiou": "consonant"
              else: "vowel"
            
        """
        ),
        ("string", "consonant"),
        None,
        id="match-cmp-not-in-words",
    ),
    pytest.param(
        dedent(
            """\
            n := 3
            match[in] n:
              `1:5`: "in"
              else: "out"
            
        """
        ),
        ("string", "in"),
        None,
        id="match-cmp-in-selector",
    ),
    pytest.param(
        dedent(
            """\
            path := "main.py"
            match[~~] path:
              r"\\.py$": "python"
              r"\\.rs$": "rust"
              else: "other"
            
        """
        ),
        ("string", "python"),
        None,
        id="match-cmp-regex",
    ),
    pytest.param(
        dedent(
            """\
            match 1:
              2: "no"
            
        """
        ),
        None,
        ShakarMatchError,
        id="match-no-else",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_match(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)
