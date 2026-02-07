from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    ParseError,
    ShakarRuntimeError,
    ShakarTypeError,
    run_runtime_case,
)

SCENARIOS = [
    pytest.param(
        dedent(
            """\
            state := { lines: 6, level: 2 }
            state.$lines >= .level * 3
        """
        ),
        ("bool", True),
        None,
        id="noanchor-segment-field",
    ),
    pytest.param(
        dedent(
            """\
            arr := [{x: 1}, {x: 2}]
            arr$[0].x + .len
        """
        ),
        ("number", 3),
        None,
        id="noanchor-segment-index",
    ),
    pytest.param(
        "state.$foo.$bar",
        None,
        ParseError,
        id="noanchor-segment-multiple-error",
    ),
    pytest.param(
        "$state.$lines",
        None,
        SyntaxError,
        id="noanchor-segment-in-noanchor-error",
    ),
    pytest.param(
        dedent(
            """\
            state := { lines: 6 }
            x := state.$lines
            obj := { val: 10 }
            obj and .val
        """
        ),
        ("number", 10),
        None,
        id="noanchor-segment-no-leak",
    ),
    pytest.param(
        dedent(
            """\
            state := { lines: 6, val: 99 }
            obj := { val: 10 }
            (state.$lines) and obj and .val
        """
        ),
        ("number", 10),
        None,
        id="noanchor-segment-no-leak-grouped",
    ),
    pytest.param(
        dedent(
            """\
            state := { lines: 6, val: 99 }
            fn make(x): ({ val: x })
            make(state.$lines) and .val
        """
        ),
        ("number", 6),
        None,
        id="noanchor-segment-no-leak-call-arg",
    ),
    pytest.param(
        dedent(
            """\
            state := { lines: 6 }
            [state.$lines, 1] and .len
        """
        ),
        ("number", 2),
        None,
        id="noanchor-segment-no-leak-array-literal",
    ),
    pytest.param(
        dedent(
            """\
            state := { lines: 6, val: 99 }
            { val: 9, x: state.$lines } and .val
        """
        ),
        ("number", 9),
        None,
        id="noanchor-segment-no-leak-object-literal",
    ),
    pytest.param(
        dedent(
            """\
            state := { obj: { val: 5 }, val: 99 }
            state.$obj ?? { val: 0 } and .val
        """
        ),
        ("number", 5),
        None,
        id="noanchor-segment-no-leak-nullish",
    ),
    pytest.param(
        dedent(
            """\
            state := { obj: { val: 5 }, val: 99 }
            [true ? state.$obj : { val: 0 }] and .len
        """
        ),
        ("number", 1),
        None,
        id="noanchor-segment-no-leak-ternary",
    ),
    pytest.param(
        dedent(
            """\
            state := { start: 1, val: 99 }
            stop := 3
            arr := [1,2,3,4,5,6,7]
            arr[`{state.$start}:{stop}`] and .len
        """
        ),
        ("number", 3),
        None,
        id="noanchor-segment-no-leak-selector",
    ),
    pytest.param(
        dedent(
            """\
            obj := { count: 5, val: 10 }
            obj.$count++ and .val
        """
        ),
        ("number", 10),
        None,
        id="noanchor-segment-postfix-incr",
    ),
    pytest.param(
        dedent(
            """\
            if true:
              let b := 2
            b
        """
        ),
        None,
        ShakarRuntimeError,
        id="let-scope-noleak",
    ),
    pytest.param(
        dedent(
            """\
            a := 1
            if true:
              let a = 2
            a
        """
        ),
        ("number", 2),
        None,
        id="let-rebind-outer",
    ),
    pytest.param(
        dedent(
            """\
            a := 1
            if true:
              let a := 2
        """
        ),
        None,
        ShakarRuntimeError,
        id="let-shadow-error",
    ),
    pytest.param(
        dedent(
            """\
            if true:
              let a, b := 1, 2
              a + b
        """
        ),
        ("number", 3),
        None,
        id="let-destructure-local",
    ),
    pytest.param(
        dedent(
            """\
            a := 1
            let a += 1
        """
        ),
        None,
        ParseError,
        id="let-compound-assign-rejected",
    ),
    pytest.param(
        dedent(
            """\
            obj := {x: 1}
            let obj.x .= 2
        """
        ),
        None,
        ParseError,
        id="let-apply-assign-rejected",
    ),
    pytest.param(
        dedent(
            """\
            user := { name: "Ada" }
            user and $"skip" and .name
        """
        ),
        ("string", "Ada"),
        None,
        id="no-anchor-preserves-dot",
    ),
    pytest.param(
        dedent(
            """\
            a := "a"
            a and 1 and .upper()
        """
        ),
        ("string", "A"),
        None,
        id="and-literal-anchor",
    ),
    pytest.param(
        dedent(
            """\
            user := { id: "outer" }
            other := { id: "inner" }
            user and (other and .id) and .id
        """
        ),
        ("string", "outer"),
        None,
        id="anchor-law",
    ),
    pytest.param(
        dedent(
            """\
            user := { friend: { name: "bob" } }
            (user.friend and .name)
        """
        ),
        ("string", "bob"),
        None,
        id="group-anchor",
    ),
    pytest.param(
        dedent(
            """\
            user := { profile: { name: "  Ada " }, id: "ID" }
            user and (.profile.name.trim()) and .id
        """
        ),
        ("string", "ID"),
        None,
        id="leading-dot-chain-law",
    ),
    pytest.param(
        dedent(
            """\
            outer := { size: 42 }
            u := "  hi  "
            outer and (=u .trim()) and .size
        """
        ),
        ("number", 42),
        None,
        id="statement-subject-locality",
    ),
    pytest.param(
        dedent(
            """\
            user := { profile: { contact: { name: "  Ada " } } }
            =(user.profile.contact.name).trim()
            user.profile.contact.name
        """
        ),
        ("string", "Ada"),
        None,
        id="statement-subject-grouped-anchor",
    ),
    pytest.param(
        dedent(
            """\
            user := { profile: { contact: { name: "  Ada " } } }
            =(user).profile.contact.name.trim()
            user
        """
        ),
        ("string", "Ada"),
        None,
        id="statement-subject-retarget-ident",
    ),
    pytest.param(
        dedent(
            """\
            a := { b: "s" }
            =(a).b
            a
        """
        ),
        ("string", "s"),
        None,
        id="statement-subject-grouped-ident-tail",
    ),
    pytest.param(
        dedent(
            """\
            user := { name: "Ada" }
            =user.name
        """
        ),
        None,
        ShakarRuntimeError,
        id="statement-subject-missing-tail",
    ),
    pytest.param(
        'nil ?? "guest" ?? "fallback" ',
        ("string", "guest"),
        None,
        id="nullish-chain",
    ),
    pytest.param(
        "a := {b: 1, c: 2}; a >= .b",
        None,
        ShakarTypeError,
        id="compare-anchor-object",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_scoping(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)
