from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    ParseError,
    ShakarRuntimeError,
    ShakarTypeError,
    run_runtime_case,
)
from shakar_ref.parser_rd import parse_source
from shakar_ref.runner import run as run_program

SCENARIOS = [
    pytest.param(
        "state := {cur: 1, next: 2, x: 0}; state{ .cur = .next; .x += 5 }; state.cur + state.x",
        ("number", 7),
        None,
        id="fanout-block-basic",
    ),
    pytest.param(
        dedent(
            """\
            state := {cur: 1, next: 2, x: 0}
            state{
              .cur = .next
              .x += 5
            }
            state.cur + state.x
        """
        ),
        ("number", 7),
        None,
        id="fanout-block-indented",
    ),
    pytest.param(
        "state := {a: {c: 0}, b: {c: 1}}; state.{a, b}.c = 5; state.a.c + state.b.c",
        ("number", 10),
        None,
        id="fieldfan-chain-assign",
    ),
    pytest.param(
        "state := {a: {c: 1}, b: {c: 3}}; state.{a, b}.c .= . + 1; state.a.c + state.b.c",
        ("number", 6),
        None,
        id="fieldfan-chain-apply",
    ),
    pytest.param(
        'state := {name: " Ada ", email: " ADA@EXAMPLE.COM "}; state.{name, email} .= .trim(); state.name + ":" + state.email',
        ("string", "Ada:ADA@EXAMPLE.COM"),
        None,
        id="fieldfan-apply-assign",
    ),
    pytest.param(
        "state := {a: 1, b: 2}; =(state).{a, b} = 10; state.a + state.b",
        ("number", 20),
        None,
        id="rebind-fieldfan-assign",
    ),
    pytest.param(
        "state := {a: 1, b: 2}; =(state).{a, b} .= . + 5; state.a + state.b",
        ("number", 13),
        None,
        id="rebind-fieldfan-apply",
    ),
    pytest.param(
        "state := {cur: 1, next: 2}; state{ .cur = .next }; state.cur",
        ("number", 2),
        None,
        id="fanout-single-clause-implicit",
    ),
    pytest.param(
        "state := {a: 1}; state{ .a = 5 }",
        None,
        ParseError,
        id="fanout-single-clause-literal-error",
    ),
    pytest.param(
        "state := {a: {c: 1}, b: {c: 3}}; result := state.{a, b}.c .= . + 1; result[0] + result[1]",
        ("number", 6),
        None,
        id="fieldfan-chain-apply-return",
    ),
    pytest.param(
        "state := {rows: [{v: 1}, {v: 3}, {v: 5}]}; state{ .rows[1:3].v = 0 }; state.rows[0].v + state.rows[1].v + state.rows[2].v",
        ("number", 1),
        None,
        id="fanout-block-slice-selector",
    ),
    pytest.param(
        dedent(
            """\
            state := {rows: set{{v: 1}, {v: 2}, {v: 3}}}
            state{ .rows[0:2].v .= . + 10 }
            total := 0
            for row in state.rows: total += row.v
            total
        """
        ),
        ("number", 26),
        None,
        id="fanout-block-set-multi-selector",
    ),
    pytest.param(
        "state := {arr: [[0, 0], [0, 0]], idx: 1, x: 0}; state{ .x += 0; .arr[0][.idx] = 9 }",
        None,
        ShakarTypeError,
        id="fanout-selector-local-dot-single-target-error",
    ),
    pytest.param(
        "state := {arr: [[0, 0], [0, 0]], idx: 1, x: 0}; state{ .x += 0; .arr[0:2][.idx] = 9 }",
        None,
        ShakarTypeError,
        id="fanout-selector-local-dot-multi-target-error",
    ),
    pytest.param(
        "state := {arr: [[0, 0], [0, 0]], idx: 1, x: 0}; state{ .x += 0; .arr[0:2][state.idx] = 9 }; state.arr[0][1] + state.arr[1][1]",
        ("number", 18),
        None,
        id="fanout-selector-explicit-outer-reference",
    ),
    pytest.param(
        "state := {rows: [{v: 1}, {v: 3}, {v: 5}]}; state{ .rows[1].v += 4; .rows[0] = {v: state.rows[0].v + 2} }; state.rows[0].v + state.rows[1].v",
        ("number", 10),
        None,
        id="fanout-block-bracketed",
    ),
    pytest.param(
        "state := {rows: [[{v: 1}], [{v: 3}]]}; state{ .rows[1][0].v = 8 }; state.rows[0][0].v + state.rows[1][0].v",
        None,
        ParseError,
        id="fanout-block-multi-index-selector",
    ),
    pytest.param(
        's := {name: " Ada ", greet: ""}; s{ .name .= .trim(); .greet = .name }; s.greet',
        ("string", "Ada"),
        None,
        id="fanout-block-apply",
    ),
    pytest.param(
        "state := {a: 1}; state{ .a = 1; .a = 2 }",
        None,
        ShakarRuntimeError,
        id="fanout-block-dup-error",
    ),
    pytest.param(
        "state := {a: 1, b: 2}; arr := state.{a, b}; arr[0] + arr[1]",
        ("number", 3),
        None,
        id="fanout-value-array",
    ),
    pytest.param(
        "state := {a: 1, b: 2}; fn add(x, y): x + y; add(state.{a, b})",
        ("number", 3),
        None,
        id="fanout-call-spread",
    ),
    pytest.param(
        "state := {a: fn():3, b: 2}; vals := state.{a(), b}; vals[0] + vals[1]",
        ("number", 5),
        None,
        id="fanout-value-call-item",
    ),
    pytest.param(
        "state := {a: 1, b: 2}; fn wrap(x): x[1]; wrap(x: state.{a, b})",
        ("number", 2),
        None,
        id="fanout-named-arg-no-spread",
    ),
    pytest.param(
        'fan { "a", "bb" }',
        ("fan", ["a", "bb"]),
        None,
        id="fan-literal-basic",
    ),
    pytest.param(
        'fan { "a", "bb" }.len',
        ("fan", [1.0, 2.0]),
        None,
        id="fan-broadcast-field",
    ),
    pytest.param(
        dedent(
            """\
            a := { x: 1 }
            b := { x: 2 }
            fan { a, b }.x = 5
            [a.x, b.x]
        """
        ),
        ("array", [5.0, 5.0]),
        None,
        id="fan-assign-field",
    ),
    pytest.param(
        dedent(
            """\
            a := 1
            b := 2
            { a, b } = 5
            [a, b]
        """
        ),
        ("array", [5.0, 5.0]),
        None,
        id="obj-to-fan-assign-identifiers",
    ),
    pytest.param(
        dedent(
            """\
            a := 1
            b := 2
            { a, b } = 10, 20
            [a, b]
        """
        ),
        ("array", [10.0, 20.0]),
        None,
        id="obj-to-fan-assign-identifiers-pack-rhs",
    ),
    pytest.param(
        dedent(
            """\
            a := 1
            b := 2
            ({ a, b }) += 5
            [a, b]
        """
        ),
        ("array", [6.0, 7.0]),
        None,
        id="obj-to-fan-compound-assign-grouped-head",
    ),
    pytest.param(
        dedent(
            """\
            a := " Ada "
            b := " BOB "
            { a, b } .= .trim()
            [a, b]
        """
        ),
        ("array", ["Ada", "BOB"]),
        None,
        id="obj-to-fan-apply-assign-identifiers",
    ),
    pytest.param(
        dedent(
            """\
            a := {x: 1}
            b := {x: 2}
            { a, b }.x += 5
            [a.x, b.x]
        """
        ),
        ("array", [6.0, 7.0]),
        None,
        id="obj-to-fan-chain-compound-assign-field",
    ),
    pytest.param(
        dedent(
            """\
            a := 1
            b := 2
            { a: a, b: b } = 5
        """
        ),
        None,
        ShakarRuntimeError,
        id="obj-to-fan-explicit-self-map-not-promoted",
    ),
    pytest.param(
        dedent(
            """\
            a := 1
            b := 2
            fan { a, b } = 5
            [a, b]
        """
        ),
        ("array", [5.0, 5.0]),
        None,
        id="fan-assign-identifiers",
    ),
    pytest.param(
        dedent(
            """\
            a := 1
            b := 2
            fan { a, b } += 5
            [a, b]
        """
        ),
        ("array", [6.0, 7.0]),
        None,
        id="fan-compound-assign-identifiers",
    ),
    pytest.param(
        dedent(
            """\
            a := 1
            b := 2
            (fan { a, b }) += 5
            [a, b]
        """
        ),
        ("array", [6.0, 7.0]),
        None,
        id="fan-compound-assign-identifiers-grouped-head",
    ),
    pytest.param(
        dedent(
            """\
            a := 1
            b := 2
            fan { a, b } += [10, 20]
            [a, b]
        """
        ),
        ("array", [11.0, 22.0]),
        None,
        id="fan-compound-assign-identifiers-zipped-rhs",
    ),
    pytest.param(
        dedent(
            """\
            a := 1
            b := 2
            fan { a, b } += 1, 2
            [a, b]
        """
        ),
        ("array", [2.0, 4.0]),
        None,
        id="fan-compound-assign-identifiers-pack-rhs",
    ),
    pytest.param(
        dedent(
            """\
            a := 1
            b := 2
            fan { a, b } = 10, 20
            [a, b]
        """
        ),
        ("array", [10.0, 20.0]),
        None,
        id="fan-assign-identifiers-pack-rhs",
    ),
    pytest.param(
        dedent(
            """\
            a := false
            b := false
            fan { a, b } = 1 < 2, 3 < 4
            [a, b]
        """
        ),
        ("array", [True, True]),
        None,
        id="fan-assign-identifiers-pack-rhs-comparisons",
    ),
    pytest.param(
        dedent(
            """\
            a := 1
            b := 2
            (fan { a, b }) = 5
            [a, b]
        """
        ),
        ("array", [5.0, 5.0]),
        None,
        id="fan-assign-identifiers-grouped-head",
    ),
    pytest.param(
        dedent(
            """\
            a := 1
            b := 2
            fan { a, b } .= . + 5
            [a, b]
        """
        ),
        ("array", [6.0, 7.0]),
        None,
        id="fan-apply-assign-identifiers",
    ),
    pytest.param(
        dedent(
            """\
            a := {x: 1}
            b := {x: 2}
            fan { a, b }.x += 5
            [a.x, b.x]
        """
        ),
        ("array", [6.0, 7.0]),
        None,
        id="fan-chain-compound-assign-field",
    ),
    pytest.param(
        dedent(
            """\
            a := {x: 1}
            b := {x: 2}
            fan { a, b }.x .= . + 5
            [a.x, b.x]
        """
        ),
        ("array", [6.0, 7.0]),
        None,
        id="fan-chain-apply-assign-field",
    ),
    pytest.param(
        dedent(
            """\
            a := {x: 1}
            result := (fan { a }).x .= . + 1
            result[0] + a.x
        """
        ),
        ("number", 4),
        None,
        id="fan-chain-apply-assign-grouped-head-return-array",
    ),
    pytest.param(
        "state := {a: 1, b: 2}; state.{a, b} += 5; [state.a, state.b]",
        ("array", [6.0, 7.0]),
        None,
        id="fieldfan-compound-assign",
    ),
    pytest.param(
        "state := {a: 1, b: 2}; state.{a, b} += 1, 2; [state.a, state.b]",
        ("array", [2.0, 4.0]),
        None,
        id="fieldfan-compound-assign-pack-rhs",
    ),
    pytest.param(
        "obj := {a: {x: 1, y: 2}, b: {x: 3, y: 4}}; obj.{a, b}.{x, y} += 10; [obj.a.x, obj.a.y, obj.b.x, obj.b.y]",
        ("array", [11.0, 12.0, 13.0, 14.0]),
        None,
        id="nested-fieldfan-compound-assign",
    ),
    pytest.param(
        dedent(
            """\
            a := 1
            fan { a + 1 } += 5
        """
        ),
        None,
        ShakarRuntimeError,
        id="fan-compound-assign-nonassignable-item-error",
    ),
    pytest.param(
        "fan { 1, 2 } == fan { 1, 2 }",
        ("bool", True),
        None,
        id="fan-equality-basic",
    ),
    pytest.param(
        "fan { fan { 1 }, fan { 2 } } == fan { fan { 1 }, fan { 2 } }",
        ("bool", True),
        None,
        id="fan-equality-nested",
    ),
    pytest.param(
        "[...fan { 1, 2 }]",
        ("array", [1.0, 2.0]),
        None,
        id="fan-spread-array",
    ),
    pytest.param(
        dedent(
            """\
            fn count(...items): items.len
            count(...fan { 1, 2, 3 })
        """
        ),
        ("number", 3),
        None,
        id="fan-spread-args",
    ),
    pytest.param(
        "[x for x in fan { 1, 2, 3 }]",
        ("array", [1.0, 2.0, 3.0]),
        None,
        id="fan-iter-comprehension",
    ),
    pytest.param(
        "fan[par] { 1 }",
        None,
        ShakarRuntimeError,
        id="fan-modifier-unsupported",
    ),
    # ---- Indented fanout: clause vs chain-continuation disambiguation ----
    pytest.param(
        dedent(
            """\
            state := {x: 0, y: 0}
            state{
              .x = 1
              .y = 2
            }
            state.x + state.y
        """
        ),
        ("number", 3),
        None,
        id="fanout-indented-plain-values",
    ),
    pytest.param(
        dedent(
            """\
            state := {x: 0, y: 0}
            state{
              .x = 1
                + 2
              .y = 3
            }
            state.x + state.y
        """
        ),
        ("number", 6),
        None,
        id="fanout-indented-rhs-op-continuation",
    ),
    pytest.param(
        dedent(
            """\
            state := {x: "hello"}
            state{
              .x = .x
                .len
            }
            state.x
        """
        ),
        ("number", 5),
        None,
        id="fanout-indented-rhs-chain-deeper",
    ),
    pytest.param(
        dedent(
            """\
            s := {a: "hello", b: "world"}
            s{
              .a = .b
                .trim()
            }
            s.a
        """
        ),
        ("string", "world"),
        None,
        id="fanout-indented-rhs-chain-continuation",
    ),
    pytest.param(
        dedent(
            """\
            state := {rows: [{v: 1}, {v: 2}]}
            state{
              .rows[0].v = 10
              .rows[1].v = 20
            }
            state.rows[0].v + state.rows[1].v
        """
        ),
        ("number", 30),
        None,
        id="fanout-indented-selector-path-clauses",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_fan(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)


def test_fanout_indented_selector_path_indenter_mode() -> None:
    """Adjacent-selector fanpath clauses must parse under use_indenter=True.

    Regression: _lookahead_is_fanout_clause broke on .rows[0].v because the
    lookahead only continued across DOT, not adjacent LSQB.  Without this
    test the failure is masked by run()'s silent fallback to use_indenter=False.
    """
    code = dedent(
        """\
        state := {rows: [{v: 1}, {v: 2}]}
        state{
          .rows[0].v = 10
          .rows[1].v = 20
        }
        state.rows[0].v + state.rows[1].v
    """
    )
    # Must not raise — parse_source with use_indenter=True directly,
    # no fallback to non-indenter mode.
    parse_source(code, use_indenter=True)


def test_fan_unknown_modifier_runtime_error() -> None:
    with pytest.raises(
        ShakarRuntimeError,
        match="unknown fan modifier 'group'; expected one of: par",
    ):
        run_program("fan[group] { 1 }")
