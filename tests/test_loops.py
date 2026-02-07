from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    ShakarTypeError,
    run_runtime_case,
)

SCENARIOS = [
    pytest.param(
        dedent(
            """\
            src := [1, 2, 3, 4]
            odds := [ n over src if n % 2 == 1 ]
            odds[1]
        """
        ),
        ("number", 3),
        None,
        id="listcomp-filter",
    ),
    pytest.param(
        dedent(
            """\
            pairs := [[i, v] over [i, v] [[0, "a"], [1, "b"]]]
            pairs[1][1]
        """
        ),
        ("string", "b"),
        None,
        id="listcomp-binder",
    ),
    pytest.param(
        dedent(
            """\
            fn make():
              [x over [1, 2]]
            x := 99
            make()[0]
        """
        ),
        ("number", 1),
        None,
        id="listcomp-implicit-binder-freeze",
    ),
    pytest.param(
        dedent(
            """\
            fn board_lines(board):
              [row_str(row) over board]
            fn row_str(row): row[0]
            board_lines([[1]])[0]
        """
        ),
        ("number", 1),
        None,
        id="listcomp-forward-fn",
    ),
    pytest.param(
        dedent(
            """\
            fn make():
              a := 10
              [a over [1, 2]][0]
            make()
        """
        ),
        ("number", 10),
        None,
        id="listcomp-local-capture",
    ),
    pytest.param(
        dedent(
            """\
            src := [[0, 2], [1, 3]]
            sums := set{ i + v over [i, v] src }
            total := 0
            for v in sums: total += v
            total
        """
        ),
        ("number", 6),
        None,
        id="setcomp-overspec",
    ),
    pytest.param(
        dedent(
            """\
            items := ["a", "b"]
            obj := { k: k + "!" over items bind k }
            obj["b"]
        """
        ),
        ("string", "b!"),
        None,
        id="dictcomp-basic",
    ),
    pytest.param(
        dedent(
            """\
            i := 0
            while i < 3: { i += 1 }
            i
        """
        ),
        ("number", 3),
        None,
        id="while-basic",
    ),
    pytest.param(
        dedent(
            """\
            i := 0
            acc := 0
            while true:
              i += 1
              if i == 5: break
              if i % 2 == 0: continue
              acc += i
            acc
        """
        ),
        ("number", 4),
        None,
        id="while-break-continue",
    ),
    pytest.param(
        dedent(
            """\
            sum := 0
            for x in [1, 2, 3]: sum = sum + x
            sum
        """
        ),
        ("number", 6),
        None,
        id="for-in-sum",
    ),
    pytest.param(
        dedent(
            """\
            acc := ""
            for ["a", "b"]: acc += .upper()
            acc
        """
        ),
        ("string", "AB"),
        None,
        id="for-subject-dot",
    ),
    pytest.param(
        dedent(
            """\
            acc := ""
            items := ["x", "y"]
            for items: acc += .
            acc
        """
        ),
        ("string", "xy"),
        None,
        id="for-subject-bare-ident",
    ),
    pytest.param(
        dedent(
            """\
            sum := 0
            for 3: sum += .
            sum
        """
        ),
        ("number", 3),
        None,
        id="for-subject-number-int",
    ),
    pytest.param(
        "for 2.5: print(.)",
        None,
        ShakarTypeError,
        id="for-subject-number-float",
    ),
    pytest.param(
        "for -2: print(.)",
        None,
        ShakarTypeError,
        id="for-subject-number-negative",
    ),
    pytest.param(
        dedent(
            """\
            log := ""
            fn out(s): log = log + s
            items := ["a", "b"]
            call out:
              for items: > .
            log
        """
        ),
        ("string", "ab"),
        None,
        id="for-subject-in-call-block",
    ),
    pytest.param(
        dedent(
            """\
            logs := ""
            items := ["a", "b"]
            for[i] items: logs = logs + ("" + i)
            logs
        """
        ),
        ("string", "01"),
        None,
        id="for-indexed",
    ),
    pytest.param(
        dedent(
            """\
            obj := { "a": 1, "b": 2 }
            keys := ""
            sum := 0
            for[k, v] obj: { keys = keys + k; sum = sum + v }
            keys + ":" + ("" + sum)
        """
        ),
        ("string", "ab:3"),
        None,
        id="for-map-key-value",
    ),
    pytest.param(
        dedent(
            """\
            obj := { "a": 1, "b": 2 }
            keys := ""
            sum := 0
            for k, v in obj: { keys = keys + k; sum = sum + v }
            keys + ":" + ("" + sum)
        """
        ),
        ("string", "ab:3"),
        None,
        id="for-map-destructure",
    ),
    pytest.param(
        dedent(
            """\
            for[^idx] [10, 20]: idx = idx
            idx
        """
        ),
        ("number", 1),
        None,
        id="for-hoist-index",
    ),
    pytest.param(
        dedent(
            """\
            sum := 0
            for x in [1, 2, 3]: { if x == 2: break; sum = sum + x }
            sum
        """
        ),
        ("number", 1),
        None,
        id="for-break",
    ),
    pytest.param(
        dedent(
            """\
            sum := 0
            for x in [1, 2, 3]: { if x == 2: continue; sum = sum + x }
            sum
        """
        ),
        ("number", 4),
        None,
        id="for-continue",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_loops(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)
