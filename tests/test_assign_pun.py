from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    ShakarRuntimeError,
    run_runtime_case,
)

SCENARIOS = [
    pytest.param(
        dedent(
            """\
            obj := {x: 0, y: 0}
            x := 10
            y := 20
            =obj.{x, y}
            obj.x + obj.y
        """
        ),
        ("number", 30),
        None,
        id="basic-assign-pun",
    ),
    pytest.param(
        dedent(
            """\
            outer := {inner: {a: 0, b: 0}}
            a := 1
            b := 2
            =outer.inner.{a, b}
            outer.inner.a + outer.inner.b
        """
        ),
        ("number", 3),
        None,
        id="nested-target",
    ),
    pytest.param(
        dedent(
            """\
            arr := [{a: 0, b: 0}]
            a := 5
            b := 7
            =arr[0].{a, b}
            arr[0].a + arr[0].b
        """
        ),
        ("number", 12),
        None,
        id="indexed-target",
    ),
    pytest.param(
        dedent(
            """\
            obj := {x: 0}
            x := 42
            =obj.{x}
            obj.x
        """
        ),
        ("number", 42),
        None,
        id="single-field",
    ),
    pytest.param(
        dedent(
            """\
            obj := {x: 0, y: 0}
            x := 3
            y := 4
            =obj.{x, y,}
            obj.x + obj.y
        """
        ),
        ("number", 7),
        None,
        id="trailing-comma",
    ),
    pytest.param(
        dedent(
            """\
            obj := {x: 0}
            =obj.{noexist}
        """
        ),
        None,
        ShakarRuntimeError,
        id="error-name-not-in-scope",
    ),
    pytest.param(
        dedent(
            """\
            obj := {x: 0, y: 0}
            x := 10
            y := 20
            result := =obj.{x, y}
            result == nil
        """
        ),
        ("bool", True),
        None,
        id="returns-nil",
    ),
    # Disambiguation: postfix continues after fan => rebind + fan-access, not assign-pun.
    # =state.{name, email}.trim() is rebind + valuefan + method call,
    # which trims each fanned value and writes back via rebind.
    pytest.param(
        dedent(
            """\
            state := {name: " Ada ", email: " BOB "}
            =state.{name, email}.trim()
            state.name + ":" + state.email
        """
        ),
        ("string", "Ada:BOB"),
        None,
        id="disambig-fan-access-with-method",
    ),
    # Identchain items in fan (e.g. name.upper()) => not assign-pun,
    # remains rebind + fan-access
    pytest.param(
        dedent(
            """\
            state := {name: "hello", email: "world"}
            =state.{name, email}.upper()
            state.name + ":" + state.email
        """
        ),
        ("string", "HELLO:WORLD"),
        None,
        id="disambig-identchain-fan-items",
    ),
]


@pytest.mark.parametrize("source, expected, error", SCENARIOS)
def test_assign_pun(source: str, expected, error) -> None:
    run_runtime_case(source, expected, error)


# ------------------------------------------------------------------
# Parser-level: verify intermediate-op restriction produces the
# correct AST node (assign_pun vs explicit_chain).
# ------------------------------------------------------------------

from shakar_ref.parser_rd import parse_source
from shakar_ref.tree import tree_label, tree_children


def _top_expr_label(source: str) -> str:
    """Return the label of the innermost expression node for a one-liner."""
    node = parse_source(source)
    # Drill through structural wrappers
    while tree_label(node) in {
        "start_indented",
        "start_noindent",
        "stmtlist",
        "stmt",
        "expr",
        "primary",
    }:
        children = tree_children(node)
        if children:
            node = children[0]
        else:
            break

    return tree_label(node)


@pytest.mark.parametrize(
    "source, expected_label",
    [
        # Should be assign_pun — field/index intermediates only
        pytest.param("=obj.{x, y}", "assign_pun", id="pun-bare"),
        pytest.param("=obj.nested.{x}", "assign_pun", id="pun-field-intermediate"),
        pytest.param("=arr[0].{a, b}", "assign_pun", id="pun-index-intermediate"),
        pytest.param("=a.b[0].c.{x}", "assign_pun", id="pun-mixed-field-index"),
        # Should remain explicit_chain — call in chain
        pytest.param("=obj.f().{x}", "explicit_chain", id="no-pun-call-in-chain"),
        pytest.param(
            "=obj.fetch().nested.{a, b}",
            "explicit_chain",
            id="no-pun-call-then-field",
        ),
        # Should remain explicit_chain — method call after fan
        pytest.param(
            "=obj.{a,b}.trim()", "explicit_chain", id="no-pun-method-after-fan"
        ),
    ],
)
def test_assign_pun_ast_node(source: str, expected_label: str) -> None:
    assert _top_expr_label(source) == expected_label
