from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    ParseError,
    ShakarRuntimeError,
    run_runtime_case,
)

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
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_fan(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)
