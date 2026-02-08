from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    ShakarArityError,
    ShakarRuntimeError,
    ShakarTypeError,
    run_runtime_case,
)

SCENARIOS = [
    pytest.param(
        "x = 1",
        None,
        ShakarRuntimeError,
        id="assign-requires-existing-var",
    ),
    pytest.param(
        dedent(
            """\
            user := {profile: {name: "Ada"}}
            name := user.profile
              .name
            name
        """
        ),
        ("string", "Ada"),
        None,
        id="dot-continuation",
    ),
    pytest.param(
        'fn greet(name, greeting): greeting + " " + name; greet(greeting: "Hi", name: "Bob")',
        ("string", "Hi Bob"),
        None,
        id="named-arg-binding-by-name",
    ),
    pytest.param(
        "fn add(a, b, c): a * 100 + b * 10 + c; add(b: 5, a: 1, c: 3)",
        ("number", 153),
        None,
        id="named-arg-out-of-order",
    ),
    pytest.param(
        "fn f(a, b): a + b; f(a: 10, 20)",
        ("number", 30),
        None,
        id="named-arg-before-positional",
    ),
    pytest.param(
        "fn f(a, b): a + b; f(10, b: 20)",
        ("number", 30),
        None,
        id="named-arg-after-positional",
    ),
    pytest.param(
        "fn f(a, b, c): a * 100 + b * 10 + c; f(1, b: 2, 3)",
        ("number", 123),
        None,
        id="named-arg-interleaved-userfn",
    ),
    pytest.param(
        'print("a", sep: "\\n", "b")',
        None,
        ShakarTypeError,
        id="named-arg-interleaved-stdlib-error",
    ),
    pytest.param(
        'print("a", "b", sep: "\\n")',
        ("null", None),
        None,
        id="named-arg-stdlib-after-positional",
    ),
    pytest.param(
        "fn f(a, b): a; f(a: 1, c: 2)",
        None,
        ShakarTypeError,
        id="named-arg-unknown-param-error",
    ),
    pytest.param(
        "fn f(a, b): a; f(1, 2, a: 3)",
        None,
        ShakarArityError,
        id="named-arg-duplicate-slot-error",
    ),
    pytest.param(
        "fn f(a, b): a + b; f(b: 2)",
        None,
        ShakarArityError,
        id="named-arg-missing-required-error",
    ),
    pytest.param(
        "fn f(a, b = 10): a + b; f(a: 5)",
        ("number", 15),
        None,
        id="named-arg-with-default",
    ),
    pytest.param(
        "fn f(a, b = a + 1): b; f(a: 5)",
        ("number", 6),
        None,
        id="named-arg-default-refs-earlier-param",
    ),
    pytest.param(
        'print(sep: "\\n", "a")',
        ("null", None),
        None,
        id="named-arg-stdlib-named-first",
    ),
    pytest.param(
        "fn f(a, ...rest): a + rest.len; f(a: 1, 2, 3)",
        ("number", 3),
        None,
        id="named-arg-vararg",
    ),
    pytest.param(
        "arr := [1, ...[2, 3], 4]; arr[0] + arr[3]",
        ("number", 5),
        None,
        id="spread-array-literal",
    ),
    pytest.param(
        "base := {a: 1}; obj := { ...base, b: 2, a: 3 }; obj.a + obj.b",
        ("number", 5),
        None,
        id="spread-object-literal",
    ),
    pytest.param(
        "fn add(a, b, c): a + b + c; add(...[1, 2, 3])",
        ("number", 6),
        None,
        id="spread-call-array",
    ),
    pytest.param(
        "fn pair(a, b): a * 10 + b; pair(...{a: 1, b: 2})",
        ("number", 12),
        None,
        id="spread-call-object",
    ),
    pytest.param(
        "fn capture(a, ...mid, b, ...tail, c): [a, mid.len, b, tail.len, c]; res := capture(1, 2, 3, 4, 5, 6, 7); res[0] + res[1] + res[2] + res[3] + res[4]",
        None,
        SyntaxError,
        id="spread-params-multi",
    ),
    pytest.param(
        "state := {config: {a: 1}, cfg: {}}; state{ .cfg = { ... .config } }; state.cfg.a",
        ("number", 1),
        None,
        id="spread-object-dot-space",
    ),
    pytest.param(
        dedent(
            """\
            decorator bump(...xs):
              args[0] = args[0] + xs.len
            @bump(1, 2, 3)
            fn id(x): x
            id(10)
        """
        ),
        ("number", 13),
        None,
        id="spread-decorator-params",
    ),
    pytest.param(
        dedent(
            """\
            flag := true
            flag ? "yes" : "no" 
        """
        ),
        ("string", "yes"),
        None,
        id="ternary-expr",
    ),
    pytest.param(
        dedent(
            """\
            temp := 50
            verdict := "fail"
            if temp > 40, < 60, and != 55: verdict = "ok"
            verdict
        """
        ),
        ("string", "ok"),
        None,
        id="ccc-runtime",
    ),
    pytest.param(
        dedent(
            """\
            calls := 0
            fn bump(): { calls += 1; "run" }
            result := bump() if true
            assert calls == 1
            result
        """
        ),
        ("string", "run"),
        None,
        id="postfix-if-walrus-true",
    ),
    pytest.param(
        dedent(
            """\
            calls := 0
            fn bump(): { calls += 1; "run" }
            result := bump() if false
            assert calls == 0
            assert result == nil
            result
        """
        ),
        ("null", None),
        None,
        id="postfix-if-walrus-false",
    ),
    pytest.param(
        dedent(
            """\
            value := 1
            value = 2 if false
            assert value == 1
            value
        """
        ),
        ("number", 1),
        None,
        id="postfix-if-assign-noop",
    ),
    pytest.param(
        dedent(
            """\
            calls := 0
            fn bump(): { calls += 1; "run" }
            result := bump() unless true
            assert calls == 0
            fallback := bump() unless false
            assert calls == 1
            fallback
        """
        ),
        ("string", "run"),
        None,
        id="postfix-unless-walrus",
    ),
    pytest.param(
        dedent(
            """\
            result := "unset"
            true: result = "hit" | false: result = "miss" |: result = "else"
            result
        """
        ),
        ("string", "hit"),
        None,
        id="guard-oneline",
    ),
    pytest.param(
        dedent(
            """\
            fn rotate(shape):
              ??(!shape[0]): return []
              return shape
            
            rotate([])
        """
        ),
        None,
        None,
        id="guard-nullsafe-index",
    ),
    pytest.param(
        dedent(
            """\
            a := 1
            a := 2
        """
        ),
        None,
        ShakarRuntimeError,
        id="walrus-duplicate-name",
    ),
    pytest.param(
        dedent(
            """\
            fn first_even(xs): { for n in xs: { if n % 2 == 0: { ?ret n } }; "none" }
            first_even([1, 3, 5, 6, 8])
        """
        ),
        ("number", 6),
        None,
        id="return-if",
    ),
    pytest.param(
        "return 1",
        None,
        ShakarRuntimeError,
        id="return-outside-fn",
    ),
    pytest.param(
        "fn f(a, b, c): a + b + c; f(1, 2, 3)",
        ("number", 6),
        None,
        id="ccc-function-args",
    ),
    pytest.param(
        "a := [1, 2, 3]; a[0] + a[1] + a[2]",
        ("number", 6),
        None,
        id="ccc-array-elements",
    ),
    pytest.param(
        "a := [(5 == 5, == 5)]; a[0]",
        ("bool", True),
        None,
        id="ccc-array-with-parens",
    ),
    pytest.param(
        "a, b := 10, 20; a + b",
        ("number", 30),
        None,
        id="ccc-destructure-pack",
    ),
    pytest.param(
        "x := 5; assert x == 5, < 10; x",
        ("number", 5),
        None,
        id="ccc-statement-allowed",
    ),
    pytest.param(
        "data := [1, 5, 10]; [x for x in data if x == 1, < 6]",
        ("array", [1.0]),
        None,
        id="ccc-comprehension-filter",
    ),
    pytest.param(
        "data := [1, 5, 10]; [x for x in data if x > 0, and < 8]",
        ("array", [1.0, 5.0]),
        None,
        id="ccc-comprehension-explicit",
    ),
    pytest.param(
        "data := [1, 5, 10]; [x for x in data if x == 1, or == 10]",
        ("array", [1.0, 10.0]),
        None,
        id="ccc-comprehension-or",
    ),
    pytest.param(
        dedent(
            """\
            a, (b, (c, d)) := [1, [2, [3, 4]]]
            a + b + c + d
        """
        ),
        ("number", 10),
        None,
        id="destructure-nested",
    ),
    pytest.param(
        dedent(
            """\
            a ~ (Int), b := 10, 20
            x := 1
            y := 2
            x + y
        """
        ),
        ("number", 3),
        None,
        id="lookahead-destructure-paren",
    ),
    pytest.param(
        dedent(
            """\
            sum := 0
            for x in [1, 2, 3]:
                sum = sum + x
            sum
        """
        ),
        ("number", 6),
        None,
        id="lookahead-forin-paren",
    ),
    pytest.param(
        dedent(
            """\
            x := 5
            x > 3:
                print("big")
            | x > 0:
                print("small")
            y := 42
            y
        """
        ),
        ("number", 42),
        None,
        id="lookahead-guard-continue",
    ),
    pytest.param(
        dedent(
            """\
            a, b := (1 > 0), (2 > 1)
            x := 99
            x
        """
        ),
        ("number", 99),
        None,
        id="lookahead-ccc-paren",
    ),
    pytest.param(
        dedent(
            """\
            d := {x: x * 2 for x in [1, 2, 3]}
            y := 42
            y
        """
        ),
        ("number", 42),
        None,
        id="lookahead-dictcomp-paren",
    ),
    pytest.param(
        dedent(
            """\
            arr := [10, 20, 30, 40]
            s := arr`1:3`
            z := 100
            z
        """
        ),
        ("number", 100),
        None,
        id="lookahead-slice-literal",
    ),
    # --- Object destructuring (contextual keyed extraction) ---
    pytest.param(
        "a, b := {a: 10, b: 20}; a + b",
        ("number", 30),
        None,
        id="obj-destruct-walrus",
    ),
    pytest.param(
        "a := 0; b := 0; a, b = {a: 10, b: 20}; a + b",
        ("number", 30),
        None,
        id="obj-destruct-update",
    ),
    pytest.param(
        "a, b := {a: 1, b: 2, c: 3}; a + b",
        ("number", 3),
        None,
        id="obj-destruct-extra-keys",
    ),
    pytest.param(
        "a, b := {a: 1, c: 2}",
        None,
        ShakarRuntimeError,
        id="obj-destruct-missing-key",
    ),
    pytest.param(
        "a, b := [10, 20]; a + b",
        ("number", 30),
        None,
        id="obj-destruct-not-object",
    ),
    pytest.param(
        'a ~ Int, b := {a: 42, b: "hi"}; a',
        ("number", 42),
        None,
        id="obj-destruct-contract",
    ),
    pytest.param(
        "let a, b := {a: 5, b: 6}; a + b",
        ("number", 11),
        None,
        id="let-obj-destruct",
    ),
    pytest.param(
        "a := {a: 1, b: 2}; a.b",
        ("number", 2),
        None,
        id="obj-destruct-single-pattern",
    ),
    pytest.param(
        "a, b := 0; a + b",
        ("number", 0),
        None,
        id="obj-destruct-scalar-broadcast",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_control_flow(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)
