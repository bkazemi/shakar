from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    LexError,
    ShakarRuntimeError,
    ShakarTypeError,
    run_runtime_case,
)

SCENARIOS = [
    pytest.param(
        dedent(
            """\
            o := {}
            o.x = 1
        """
        ),
        None,
        ShakarRuntimeError,
        id="object-assign-requires-existing-field",
    ),
    pytest.param(
        "([1,2] + [3]).len",
        ("number", 3),
        None,
        id="array-concat",
    ),
    pytest.param(
        "arr := [1]; arr.push(2); arr.len",
        ("number", 2),
        None,
        id="array-push",
    ),
    pytest.param(
        "arr := [1]; arr.append(2); arr[1]",
        ("number", 2),
        None,
        id="array-append",
    ),
    pytest.param(
        "arr := [1, 2, 3]; arr.pop() + arr.len",
        ("number", 5),
        None,
        id="array-pop",
    ),
    pytest.param(
        "[].pop()",
        None,
        ShakarRuntimeError,
        id="array-pop-empty",
    ),
    pytest.param(
        dedent(
            """\
            xs := [10, 20, 30]
            xs.high
        """
        ),
        ("number", 2),
        None,
        id="array-high",
    ),
    pytest.param(
        " [].high",
        ("number", -1),
        None,
        id="array-high-empty",
    ),
    pytest.param(
        "obj := {a: 1, b: 2}; obj.len",
        ("number", 2),
        None,
        id="object-len",
    ),
    pytest.param(
        "{a: 1, b: 2}.keys()",
        ("array", ["a", "b"]),
        None,
        id="object-keys",
    ),
    pytest.param(
        "{a: 1, b: 2}.values()",
        ("array", [1, 2]),
        None,
        id="object-values",
    ),
    pytest.param(
        "[1, 2, 3].map&(. * 2)",
        ("array", [2, 4, 6]),
        None,
        id="array-map-basic",
    ),
    pytest.param(
        "[1, 2, 3, 4].filter&(. > 1)",
        ("array", [2, 3, 4]),
        None,
        id="array-filter-basic",
    ),
    pytest.param(
        '[0, 1, "", "hi", nil].filter&(.)',
        ("array", [1, "hi"]),
        None,
        id="array-filter-truthy",
    ),
    pytest.param(
        "arr := [0, 1]; o := {a: arr}; o.a[0,1] = 2; o.a[0] + o.a[1]",
        ("number", 4),
        None,
        id="selector-assign-broadcast",
    ),
    pytest.param(
        "arr := [0, 1, 2]; arr[`0:1`] = 5; arr[0] + arr[1] + arr[2]",
        ("number", 12),
        None,
        id="selectorliteral-assign-broadcast",
    ),
    pytest.param(
        "arr := [0, 1, 2]; arr[-1:2]",
        None,
        ShakarRuntimeError,
        id="slice-negative-start-positive-stop-error",
    ),
    pytest.param(
        "arr := [0, 1, 2, 3, 4]; result := arr[0:-1]; result[0] + result[3]",
        ("number", 3),
        None,
        id="slice-positive-start-negative-stop",
    ),
    pytest.param(
        "arr := [0, 1, 2, 3, 4]; result := arr[-3:-1]; result[0] + result[1]",
        ("number", 5),
        None,
        id="slice-both-negative",
    ),
    pytest.param(
        dedent(
            """\
            vals := set{1, 2, 1}
            total := 0
            for v in vals: total += v
            total
        """
        ),
        ("number", 3),
        None,
        id="setliteral-sum",
    ),
    pytest.param(
        dedent(
            """\
            users := ["aa", "bb"]
            seen := users.len
            users[seen - 1]
            "" + seen
        """
        ),
        ("string", "2"),
        None,
        id="selector-base-anchor",
    ),
    pytest.param(
        dedent(
            """\
            v1 := 1 == `1, 2`
            v2 := 1 != `2, 3`
            v3 := 1 != `1, 2`
            [v1, v2, v3]
        """
        ),
        ("array", [True, True, False]),
        None,
        id="selector-compare-eq-any",
    ),
    pytest.param(
        dedent(
            """\
            sel := `0, 2`
            arr := [10, 20, 30]
            picked := arr[sel]
            picked[0] + picked[1]
        """
        ),
        ("number", 40),
        None,
        id="selector-literal-pick",
    ),
    pytest.param(
        dedent(
            """\
            sel := `:3`
            arr := [10, 20, 30, 40]
            picked := arr[sel]
            picked[0] + picked[1] + picked[2]
        """
        ),
        ("number", 60),
        None,
        id="selector-literal-open-start",
    ),
    pytest.param(
        dedent(
            """\
            sel := `2:`
            arr := [1, 2, 3, 4, 5]
            picked := arr[sel]
            picked[0] + picked[1] + picked[2]
        """
        ),
        ("number", 12),
        None,
        id="selector-literal-open-stop",
    ),
    pytest.param(
        dedent(
            """\
            sel := `5::-2`
            arr := [0, 1, 2, 3, 4, 5]
            picked := arr[sel]
            picked.len
        """
        ),
        ("number", 0),
        None,
        id="selector-literal-open-stop-neg-step",
    ),
    pytest.param(
        dedent(
            """\
            sel := `:1:-2`
            arr := [0, 1, 2, 3, 4, 5]
            picked := arr[sel]
            picked[0] + picked[1] + picked[2]
        """
        ),
        ("number", 9),
        None,
        id="selector-literal-open-start-neg-step",
    ),
    pytest.param(
        dedent(
            """\
            sel := `1:7:2`
            arr := [0, 1, 2, 3, 4, 5, 6, 7]
            picked := arr[sel]
            picked[0] + picked[1] + picked[2]
        """
        ),
        ("number", 9),
        None,
        id="selector-literal-slice-step",
    ),
    pytest.param(
        dedent(
            """\
            step := -1
            sel := `5:1:{step}`
            arr := [0, 1, 2, 3, 4, 5]
            picked := arr[sel]
            picked[0] + picked[1] + picked[2] + picked[3]
        """
        ),
        ("number", 14),
        None,
        id="selector-literal-slice-step-interp-neg",
    ),
    pytest.param(
        dedent(
            """\
            arr := [0]
            arr[`0x8000000000000000`]
        """
        ),
        None,
        LexError,
        id="selector-literal-hex-overflow",
    ),
    pytest.param(
        dedent(
            """\
            arr := [0]
            arr[`9223372036854775808`]
        """
        ),
        None,
        LexError,
        id="selector-literal-dec-overflow",
    ),
    pytest.param(
        dedent(
            """\
            arr := [0]
            arr[`0:0x8000000000000000`]
        """
        ),
        None,
        LexError,
        id="selector-literal-slice-stop-overflow",
    ),
    pytest.param(
        dedent(
            """\
            arr := [0]
            arr[`0:1:0x8000000000000000`]
        """
        ),
        None,
        LexError,
        id="selector-literal-slice-step-overflow",
    ),
    pytest.param(
        dedent(
            """\
            arr := [0]
            arr[`0:1:-0x8000000000000001`]
        """
        ),
        None,
        LexError,
        id="selector-literal-slice-step-neg-overflow",
    ),
    pytest.param(
        dedent(
            """\
            arr := [10, 20, 30, 40]
            slice := arr[1:3]
            slice[0] + slice[1]
        """
        ),
        ("number", 50),
        None,
        id="selector-slice",
    ),
    pytest.param(
        dedent(
            """\
            start := 1
            stop := 3
            sel := `{start}:{stop}`
            arr := [4, 5, 6, 7]
            sum := arr[sel]
            sum[0] + sum[1]
        """
        ),
        ("number", 11),
        None,
        id="selector-literal-interp",
    ),
    pytest.param(
        dedent(
            """\
            cfg := { db: { host: "db" } }
            calls := 0
            fn fallback():
              calls += 1
              return "localhost"
            found := cfg["db", default: {}]["host", default: fallback()]
            missing := cfg["port", default: fallback()]
            assert calls == 1
            "{found}-{missing}"
            
        """
        ),
        ("string", "db-localhost"),
        None,
        id="object-index-default",
    ),
    pytest.param(
        dedent(
            """\
            arr := [1, 2, 3]
            arr[0, default: 9]
        """
        ),
        None,
        ShakarTypeError,
        id="array-index-default-error",
    ),
    pytest.param(
        dedent(
            """\
            obj := {
              name: "Ada"
              total: 12
              get label(): .name
              set label(next):
                owner := .
                owner.name = next.trim().upper()
              get double(): .total * 2
              set double(next):
                owner := .
                owner.total = next / 2
              greet(name): "hi " + name
              ("dyn" + "Key"): 42
            }
            before := obj.double
            obj.double = 10
            after_dot := obj.double
            obj["double"] = 18
            after_index := obj.double
            label_before := obj.label
            obj.label = "  grace  "
            label_after := obj.label
            greet := obj.greet(label_after)
            dyn := obj["dynKey"]
            label_before + "|" + greet + "|" + ("" + dyn) + "|" + ("" + before) + "|" + ("" + after_dot) + "|" + ("" + after_index) + "|" + label_after
        """
        ),
        ("string", "Ada|hi GRACE|42|24|10|18|GRACE"),
        None,
        id="object-getter-setter",
    ),
    pytest.param(
        "o := {(sum(a, b)): a + b}; o.sum(2, 3)",
        ("number", 5),
        None,
        id="object-key-expr-method-sugar-simple-ident-args",
    ),
    pytest.param(
        dedent(
            """\
            fn g(x): "k" + x
            fn f(v): v
            x := "1"
            o := {(f(g(x))): 7}
            o["k1"]
        """
        ),
        ("number", 7),
        None,
        id="object-key-expr-nested-call-not-method-sugar",
    ),
    pytest.param(
        dedent(
            """\
            arr := [1, 2]
            arr.repeat(2)
        """
        ),
        ("array", [1, 2, 1, 2]),
        None,
        id="array-repeat",
    ),
    pytest.param(
        dedent(
            """\
            arr := [1, 2]
            arr.repeat(0)
        """
        ),
        ("array", []),
        None,
        id="array-repeat-zero",
    ),
    pytest.param(
        "a := [1, 2, 3]; a.update&(. * 2); a",
        ("array", [2, 4, 6]),
        None,
        id="array-update-basic",
    ),
    pytest.param(
        "a := [1, 2].update&(. + 1); a",
        ("array", [2, 3]),
        None,
        id="array-update-chaining",
    ),
    pytest.param(
        "a := [1, 2, 3, 4]; a.keep&(. > 2); a",
        ("array", [3, 4]),
        None,
        id="array-keep-basic",
    ),
    pytest.param(
        "o := {a: 1, b: 2}; o.update&(. + 10); o.a",
        ("number", 11),
        None,
        id="object-update-basic",
    ),
    # Object punning
    pytest.param(
        "x := 42; {x}.x",
        ("number", 42),
        None,
        id="obj-pun-single",
    ),
    pytest.param(
        "x := 1; y := 2; o := {x, y}; o.x + o.y",
        ("number", 3),
        None,
        id="obj-pun-multi",
    ),
    pytest.param(
        "x := 1; {x, y: 2}.y",
        ("number", 2),
        None,
        id="obj-pun-mixed",
    ),
    pytest.param(
        "x := 1; base := {y: 2}; {x, ...base}.y",
        ("number", 2),
        None,
        id="obj-pun-with-spread",
    ),
    pytest.param(
        dedent(
            """\
            x := 1
            y := 2
            {
              x
              y
            }.x
        """
        ),
        ("number", 1),
        None,
        id="obj-pun-newline-sep",
    ),
    pytest.param(
        "x := 1; {x,}.x",
        ("number", 1),
        None,
        id="obj-pun-trailing-comma",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_collections(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)
