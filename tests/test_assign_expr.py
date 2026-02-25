"""Tests for value-producing assignment expressions (Phase 1 + Phase 2)."""

from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    ParseError,
    ShakarRuntimeError,
    run_runtime_case,
)


# ---- Phase 1: value-producing assignment statements ----

PHASE1_SCENARIOS = [
    # --- Simple assignment returns assigned value ---
    pytest.param(
        dedent(
            """\
            x := 0
            x = 42
        """
        ),
        ("number", 42),
        None,
        id="assign-returns-value",
    ),
    # --- Compound assignment returns updated value ---
    pytest.param(
        dedent(
            """\
            count := 5
            count += 1
        """
        ),
        ("number", 6),
        None,
        id="compound-add-returns-value",
    ),
    pytest.param(
        dedent(
            """\
            x := 10
            x -= 3
        """
        ),
        ("number", 7),
        None,
        id="compound-sub-returns-value",
    ),
    pytest.param(
        dedent(
            """\
            x := 4
            x *= 5
        """
        ),
        ("number", 20),
        None,
        id="compound-mul-returns-value",
    ),
    pytest.param(
        dedent(
            """\
            x := 10
            x //= 3
        """
        ),
        ("number", 3),
        None,
        id="compound-floordiv-returns-value",
    ),
    pytest.param(
        dedent(
            """\
            x := 2
            x **= 8
        """
        ),
        ("number", 256),
        None,
        id="compound-pow-returns-value",
    ),
    # --- Implicit return from one-line function body ---
    pytest.param(
        dedent(
            """\
            count := 0
            fn inc(): count += 1
            inc()
            inc()
            inc()
        """
        ),
        ("number", 3),
        None,
        id="fn-implicit-return-compound",
    ),
    pytest.param(
        dedent(
            """\
            state := {val: 0}
            fn put(v): state.val = v
            put(99)
        """
        ),
        ("number", 99),
        None,
        id="fn-implicit-return-assign",
    ),
    # --- Fan apply-assign already returns ShkArray (unchanged, regression) ---
    pytest.param(
        dedent(
            """\
            state := {a: 1, b: 2}
            result := state.{a, b} .= . + 10
            result[0] + result[1]
        """
        ),
        ("number", 23),
        None,
        id="fan-apply-assign-returns-array",
    ),
    # --- Non-final assignment in block doesn't break ---
    pytest.param(
        dedent(
            """\
            x := 0
            x = 5
            x + 1
        """
        ),
        ("number", 6),
        None,
        id="assign-non-final-no-regression",
    ),
    pytest.param(
        dedent(
            """\
            x := 0
            x += 5
            x + 1
        """
        ),
        ("number", 6),
        None,
        id="compound-non-final-no-regression",
    ),
    # --- Field assignment returns value ---
    pytest.param(
        dedent(
            """\
            obj := {x: 0}
            obj.x = 42
        """
        ),
        ("number", 42),
        None,
        id="field-assign-returns-value",
    ),
    # --- Field compound assignment returns value ---
    pytest.param(
        dedent(
            """\
            obj := {x: 10}
            obj.x += 5
        """
        ),
        ("number", 15),
        None,
        id="field-compound-returns-value",
    ),
    # --- Index assignment returns value ---
    pytest.param(
        dedent(
            """\
            arr := [1, 2, 3]
            arr[1] = 99
        """
        ),
        ("number", 99),
        None,
        id="index-assign-returns-value",
    ),
]

# ---- Phase 2: expression-position assignment ----

PHASE2_SCENARIOS = [
    # --- Assignment in parenthesized expression ---
    pytest.param(
        dedent(
            """\
            x := 0
            y := (x = 10) + 5
            y
        """
        ),
        ("number", 15),
        None,
        id="assign-in-parens",
    ),
    # --- Compound assignment in parenthesized expression ---
    pytest.param(
        dedent(
            """\
            x := 3
            y := (x += 7) * 2
            y
        """
        ),
        ("number", 20),
        None,
        id="compound-in-parens",
    ),
    # --- Right-associative chained assignment ---
    pytest.param(
        dedent(
            """\
            a := 0
            b := 0
            a = b = 42
            a + b
        """
        ),
        ("number", 84),
        None,
        id="chained-assign-right-assoc",
    ),
    # --- Triple chain ---
    pytest.param(
        dedent(
            """\
            a := 0
            b := 0
            c := 0
            a = b = c = 7
            a + b + c
        """
        ),
        ("number", 21),
        None,
        id="triple-chained-assign",
    ),
    # --- Assignment in walrus RHS ---
    pytest.param(
        dedent(
            """\
            x := 0
            y := (x = 5)
            x + y
        """
        ),
        ("number", 10),
        None,
        id="assign-in-walrus-rhs",
    ),
    # --- Assignment in function argument ---
    pytest.param(
        dedent(
            """\
            x := 0
            fn id(v): v
            id(x = 99)
        """
        ),
        ("number", 99),
        None,
        id="assign-in-arg",
    ),
    # --- Compound assignment in function argument ---
    pytest.param(
        dedent(
            """\
            x := 10
            fn id(v): v
            id(x -= 3)
        """
        ),
        ("number", 7),
        None,
        id="compound-in-arg",
    ),
    # --- Assignment in index expression ---
    pytest.param(
        dedent(
            """\
            arr := [10, 20, 30]
            i := 0
            arr[i = 2]
        """
        ),
        ("number", 30),
        None,
        id="assign-in-index",
    ),
    # --- Assignment in ternary condition ---
    pytest.param(
        dedent(
            """\
            x := 0
            (x = 5) > 3 ? "yes" : "no"
        """
        ),
        ("string", "yes"),
        None,
        id="assign-in-ternary-cond",
    ),
    # --- Defaulting via ?? instead of = ---
    pytest.param(
        dedent(
            """\
            x := nil
            x = x ?? 42
        """
        ),
        ("number", 42),
        None,
        id="default-via-nullish",
    ),
    # --- Chained assignment with field targets ---
    pytest.param(
        dedent(
            """\
            a := {x: 0}
            b := {x: 0}
            a.x = b.x = 9
            a.x + b.x
        """
        ),
        ("number", 18),
        None,
        id="chained-field-assign",
    ),
    # --- let with assignment expression ---
    pytest.param(
        dedent(
            """\
            x := 0
            let x = 10
            x
        """
        ),
        ("number", 10),
        None,
        id="let-assign-expr",
    ),
    # --- let with chained assignment (P1 regression) ---
    pytest.param(
        dedent(
            """\
            x := 0
            y := 0
            let x = y = 5
            x + y
        """
        ),
        ("number", 10),
        None,
        id="let-chained-assign",
    ),
    # --- Fan broadcast assignment returns per-target array (P2 regression) ---
    pytest.param(
        dedent(
            """\
            obj := {a: 0, b: 0}
            obj.{a, b} = 1
        """
        ),
        ("array", [1.0, 1.0]),
        None,
        id="fan-broadcast-assign-return-shape",
    ),
    # --- Index-create assignment (write-only, no prior key) ---
    pytest.param(
        dedent(
            """\
            obj := {}
            obj["key"] = 42
        """
        ),
        ("number", 42),
        None,
        id="index-create-assign",
    ),
    # --- Aliased fan assignment returns ShkArray ---
    pytest.param(
        dedent(
            """\
            a := {x: 0}
            b := {x: 0}
            f := fan { a, b }
            f.x = 9
        """
        ),
        ("array", [9.0, 9.0]),
        None,
        id="aliased-fan-assign-return-shape",
    ),
    # --- Nested fan assignment returns one result per updated target ---
    pytest.param(
        dedent(
            """\
            a := {x: 0, y: 0}
            b := {x: 0, y: 0}
            f := fan { a, b }
            f.{x, y} = 1
        """
        ),
        ("array", [1.0, 1.0, 1.0, 1.0]),
        None,
        id="nested-fan-assign-return-shape",
    ),
    # --- Fan assignment preserves array values per target ---
    pytest.param(
        dedent(
            """\
            a := {x: nil}
            b := {x: nil}
            f := fan { a, b }
            res := f.x = [1]
            res[0][0] + res[1][0]
        """
        ),
        ("number", 2),
        None,
        id="fan-assign-preserves-array-values",
    ),
    # --- Fan assignment in arg list must keep comma as separator ---
    pytest.param(
        dedent(
            """\
            fn snd(a, b): b
            obj := {x: 0, y: 0}
            snd(obj.{x, y} = 1, 2)
        """
        ),
        ("number", 2),
        None,
        id="fan-assign-arg-comma-separator",
    ),
    # --- Fan assignment in array elements must keep comma as separator ---
    pytest.param(
        dedent(
            """\
            obj := {x: 0, y: 0}
            arr := [obj.{x, y} = 1, 2]
            arr[1]
        """
        ),
        ("number", 2),
        None,
        id="fan-assign-array-comma-separator",
    ),
    # --- Fan assignment with chained RHS ---
    pytest.param(
        dedent(
            """\
            obj := {x: 0, y: 0}
            b := 0
            obj.{x, y} = b = 3
            obj.x + obj.y + b
        """
        ),
        ("number", 9),
        None,
        id="fan-assign-chained-rhs",
    ),
    # --- Fan compound with chained RHS (statement level) ---
    pytest.param(
        dedent(
            """\
            obj := {x: 1, y: 2}
            obj.{x, y} = 10, 20
            obj.x + obj.y
        """
        ),
        ("number", 30),
        None,
        id="fan-assign-pack-rhs",
    ),
    # --- Single-target fan assignment returns scalar by write-count ---
    pytest.param(
        dedent(
            """\
            a := {x: 0}
            f := fan { a }
            f.x = 9
        """
        ),
        ("number", 9),
        None,
        id="fan-assign-single-target-scalar",
    ),
    # --- Fan assignment resolves targets after RHS side effects ---
    pytest.param(
        dedent(
            """\
            arr := [{x: 0, y: 0}, {x: 0, y: 0}]
            i := 0
            arr[i].{x, y} = i = 1
            arr[0].x + arr[1].x * 10 + i * 100
        """
        ),
        ("number", 110),
        None,
        id="fan-assign-rhs-before-target-resolution",
    ),
    # --- Destructure defaults still work (with ?? for defaults) ---
    pytest.param(
        dedent(
            """\
            a, b = 0 := [1]
            a + b
        """
        ),
        ("number", 1),
        None,
        id="destructure-default-still-works",
    ),
    # --- Nested assignment in default expression (inside array/object) ---
    pytest.param(
        dedent(
            """\
            x := 0
            a = [x = 1], b = 2 := []
            a[0] + x + b
        """
        ),
        ("number", 4),
        None,
        id="destructure-default-nested-assign-expr",
    ),
    # --- Destructure with contracts still works ---
    pytest.param(
        dedent(
            """\
            a ~ Int, b ~ Int := 3, 7
            a + b
        """
        ),
        ("number", 10),
        None,
        id="destructure-contracts-still-work",
    ),
]


@pytest.mark.parametrize("source, expected, error", PHASE1_SCENARIOS + PHASE2_SCENARIOS)
def test_assign_expr(
    source: str,
    expected: object,
    error: object,
) -> None:
    run_runtime_case(source, expected, error)
