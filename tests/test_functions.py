from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    ShakarArityError,
    ShakarAssertionError,
    ShakarRuntimeError,
    ShakarTypeError,
    run_runtime_case,
)

SCENARIOS = [
    pytest.param(
        dedent(
            """\
            fn f(a = 1, b = 2): a + b
            f()
        """
        ),
        ("number", 3),
        None,
        id="param-default-basic",
    ),
    pytest.param(
        dedent(
            """\
            fn f(a, b = 2): a + b
            f(1)
        """
        ),
        ("number", 3),
        None,
        id="param-default-partial",
    ),
    pytest.param(
        dedent(
            """\
            fn f({a, b}): a + b
            f({a: 1, b: 2})
        """
        ),
        ("number", 3),
        None,
        id="param-destruct-basic",
    ),
    pytest.param(
        dedent(
            """\
            fn f({host, port = 5432}): port
            f({host: "x"})
        """
        ),
        ("number", 5432),
        None,
        id="param-destruct-default",
    ),
    pytest.param(
        dedent(
            """\
            fn f({host, port = 5432}): port
            f({host: "x", port: 3000})
        """
        ),
        ("number", 3000),
        None,
        id="param-destruct-default-override",
    ),
    pytest.param(
        dedent(
            """\
            fn f({a ~ Int}): a
            f({a: 3})
        """
        ),
        ("number", 3),
        None,
        id="param-destruct-field-contract-ok",
    ),
    pytest.param(
        dedent(
            """\
            fn f({a ~ Int}): a
            f({a: "x"})
        """
        ),
        None,
        ShakarAssertionError,
        id="param-destruct-field-contract-fail",
    ),
    pytest.param(
        dedent(
            """\
            fn f({a = 1 ~ Int}): a
            f({})
        """
        ),
        ("number", 1),
        None,
        id="param-destruct-field-default-contract-ok",
    ),
    pytest.param(
        dedent(
            """\
            fn f(mode, {host, port}): mode + host
            f("tcp-", {host: "db", port: 5432})
        """
        ),
        ("string", "tcp-db"),
        None,
        id="param-destruct-mixed-positional",
    ),
    pytest.param(
        dedent(
            """\
            fn f({host, port} ~ {host: Str, port: Int}): port
            f({host: "db", port: 5432})
        """
        ),
        ("number", 5432),
        None,
        id="param-destruct-contract-ok",
    ),
    pytest.param(
        dedent(
            """\
            fn f({host, port} ~ {host: Str, port: Int}): port
            f({host: "db", port: "bad"})
        """
        ),
        None,
        ShakarAssertionError,
        id="param-destruct-contract-fail",
    ),
    pytest.param(
        dedent(
            """\
            fn f({host, port}): host
            f({host: "db"})
        """
        ),
        None,
        ShakarRuntimeError,
        id="param-destruct-missing-required",
    ),
    pytest.param(
        dedent(
            """\
            fn f({a}): a
            f(1)
        """
        ),
        None,
        ShakarTypeError,
        id="param-destruct-non-object",
    ),
    pytest.param(
        dedent(
            """\
            mul := fn({x, y}): x * y
            mul({x: 3, y: 4})
        """
        ),
        ("number", 12),
        None,
        id="param-destruct-anon",
    ),
    pytest.param(
        dedent(
            """\
            fn f({a}, {b}): a + b
            f({a: 1}, {b: 2})
        """
        ),
        ("number", 3),
        None,
        id="param-destruct-multiple",
    ),
    pytest.param(
        dedent(
            """\
            Schema := {a: Int, b: Int, c: Array}
            fn f(({a}, obj) ~ Schema): a + obj.b
            f({a: 1, b: 2, c: []}, {a: 3, b: 4, c: [0]})
        """
        ),
        ("number", 5),
        None,
        id="param-group-destruct-contract-propagate",
    ),
    pytest.param(
        dedent(
            """\
            Schema := {a: Int, b: Int, c: Array}
            fn f(({a}, obj) ~ Schema): a + obj.b
            f({a: 1, b: 2}, {a: 3, b: 4, c: [0]})
        """
        ),
        None,
        ShakarAssertionError,
        id="param-group-destruct-contract-fail",
    ),
    pytest.param(
        dedent(
            """\
            fn f({a}, b ~ {a: Int}): a
            f({a: "x"}, {a: 1})
        """
        ),
        None,
        ShakarAssertionError,
        id="param-destruct-trailing-contract-propagates",
    ),
    pytest.param(
        dedent(
            """\
            fn f(({a}), b ~ {a: Int}): a
            f({a: "x"}, {a: 1})
        """
        ),
        ("string", "x"),
        None,
        id="param-destruct-isolated-no-trailing-contract",
    ),
    pytest.param(
        dedent(
            """\
            fn f(a, {a}): a
            f(1, {a: 2})
        """
        ),
        None,
        SyntaxError,
        id="param-destruct-collision-regular",
    ),
    pytest.param(
        dedent(
            """\
            fn f({a, b}, {b, c}): b
            f({a: 1, b: 2}, {b: 3, c: 4})
        """
        ),
        None,
        SyntaxError,
        id="param-destruct-collision-between-destructs",
    ),
    pytest.param(
        dedent(
            """\
            f := &[a, b ~ Int](a + b)
            f(1, 2)
        """
        ),
        ("number", 3),
        None,
        id="amp-lambda-contract-ok",
    ),
    pytest.param(
        dedent(
            """\
            f := &[a, b ~ Int](a + b)
            f("x", 2)
        """
        ),
        None,
        ShakarAssertionError,
        id="amp-lambda-contract-fail",
    ),
    pytest.param(
        'a := &(.trim()); a(" B")',
        ("string", "B"),
        None,
        id="lambda-subject-direct-call",
    ),
    pytest.param(
        "fn add(x, y): x + y; add(2, 3)",
        ("number", 5),
        None,
        id="fn-definition",
    ),
    pytest.param(
        dedent(
            """\
            fn b():
              a()
            fn a(): 42
            b()
        """
        ),
        ("number", 42),
        None,
        id="fn-forward-ref",
    ),
    pytest.param(
        dedent(
            """\
            fn a():
              b()
            a()
            fn b(): 1
        """
        ),
        None,
        ShakarRuntimeError,
        id="fn-forward-call-error",
    ),
    pytest.param(
        "y := 5; fn addY(x): x + y; addY(2)",
        ("number", 7),
        None,
        id="fn-closure",
    ),
    pytest.param(
        dedent(
            """\
            fn addOne(x): { return x + 1 }
            addOne(4)
        """
        ),
        ("number", 5),
        None,
        id="fn-return-value",
    ),
    pytest.param(
        dedent(
            """\
            fn noop(): { return }
            noop()
        """
        ),
        ("null", None),
        None,
        id="fn-return-default-null",
    ),
    pytest.param(
        "inc := fn(x): { x + 1 }; inc(5)",
        ("number", 6),
        None,
        id="anon-fn-expression",
    ),
    pytest.param(
        dedent(
            """\
            make := fn(): { value := 2; value + 3 }
            make()
        """
        ),
        ("number", 5),
        None,
        id="anon-fn-block",
    ),
    pytest.param(
        dedent(
            """\
            result := fn(()) : { tmp := 4; tmp + 1 }
            result
        """
        ),
        ("number", 5),
        None,
        id="anon-fn-auto",
    ),
    pytest.param(
        dedent(
            """\
            inline := fn(): 40 + 2
            inline()
        """
        ),
        ("number", 42),
        None,
        id="anon-fn-inline",
    ),
    pytest.param(
        dedent(
            """\
            fnRef := fn(): print("awd")
            "done"
        """
        ),
        ("string", "done"),
        None,
        id="anon-fn-inline-store",
    ),
    pytest.param(
        dedent(
            """\
            fn choose(): { defer finish: { return 2 }; return 1 }
            choose()
        """
        ),
        ("number", 2),
        None,
        id="fn-return-in-defer",
    ),
    pytest.param(
        'print(1, "a")',
        ("null", None),
        None,
        id="stdlib-print",
    ),
    pytest.param(
        'tap("abc", print).upper()',
        ("string", "ABC"),
        None,
        id="stdlib-tap-direct",
    ),
    pytest.param(
        '"abc".tap(print).upper()',
        ("string", "ABC"),
        None,
        id="stdlib-tap-ufcs",
    ),
    pytest.param(
        dedent(
            """\
            fn needs_sep(v, sep): v + sep + v
            "abc".tap(needs_sep, sep: "-")
        """
        ),
        ("string", "abc"),
        None,
        id="stdlib-tap-forward-named",
    ),
    pytest.param(
        '"abc".tap()',
        None,
        ShakarTypeError,
        id="stdlib-tap-arity-error",
    ),
    pytest.param(
        "a := &(.trim()); a()",
        None,
        ShakarArityError,
        id="lambda-subject-missing-arg",
    ),
    pytest.param(
        dedent(
            """\
            fn blend(a, b, ratio): a + (b - a) * ratio
            partial := blend(?, ?, 0.25)
            partial(0, 16)
        """
        ),
        ("number", 4),
        None,
        id="lambda-hole-runtime",
    ),
    pytest.param(
        dedent(
            """\
            _hole0 := 100
            fn add(a, b): a + b
            partial := add(?, _hole0)
            partial(1)
        """
        ),
        ("number", 101),
        None,
        id="lambda-hole-collision-free",
    ),
    pytest.param(
        "blend(?, ?, 0.25)()",
        None,
        SyntaxError,
        id="lambda-hole-iifc",
    ),
    pytest.param(
        dedent(
            """\
            f := fn: 42
            f()
        """
        ),
        ("number", 42),
        None,
        id="fn-sugar-simple",
    ),
    pytest.param(
        dedent(
            """\
            f := fn ~ Int: 100
            f()
        """
        ),
        ("number", 100),
        None,
        id="fn-sugar-contract",
    ),
    pytest.param(
        dedent(
            """\
            f := fn:
                x := 10
                x * 2
            f()
        """
        ),
        ("number", 20),
        None,
        id="fn-sugar-block",
    ),
    pytest.param(
        dedent(
            """\
            call_wrapper := fn(f, val): f() == val
            call_wrapper(fn: 5, 5)
        """
        ),
        ("bool", True),
        None,
        id="fn-sugar-arglist",
    ),
    pytest.param(
        dedent(
            """\
            f := fn ~ Int: "bad"
            f()
        """
        ),
        None,
        ShakarTypeError,
        id="fn-sugar-contract-fail",
    ),
    pytest.param(
        '"42".int()',
        ("number", 42),
        None,
        id="ufcs-stdlib-int",
    ),
    pytest.param(
        '"3.5".float()',
        ("number", 3.5),
        None,
        id="ufcs-stdlib-float",
    ),
    pytest.param(
        '"".bool()',
        ("bool", False),
        None,
        id="ufcs-stdlib-bool",
    ),
    pytest.param(
        "123.str()",
        ("string", "123"),
        None,
        id="ufcs-stdlib-str",
    ),
    pytest.param(
        dedent(
            """\
            fn double(x): x * 2
            5.double()
        """
        ),
        ("number", 10),
        None,
        id="ufcs-user-fn-prepend",
    ),
    pytest.param(
        dedent(
            """\
            fn add(a, b): a + b
            10.add(5)
        """
        ),
        ("number", 15),
        None,
        id="ufcs-user-fn-multi-arg",
    ),
    pytest.param(
        dedent(
            """\
            fn len(x): 999
            [1, 2, 3].len
        """
        ),
        ("number", 3),
        None,
        id="ufcs-method-shadows-global",
    ),
    pytest.param(
        dedent(
            """\
            fn inc(x): x + 1
            fn double(x): x * 2
            5.inc().double()
        """
        ),
        ("number", 12),
        None,
        id="ufcs-chain",
    ),
    pytest.param(
        dedent(
            """\
            fn double(x): x * 2
            [.double() over [3]][0]
        """
        ),
        ("number", 6),
        None,
        id="ufcs-implicit-subject",
    ),
    pytest.param(
        dedent(
            """\
            fn double(x): x * 2
            [1, 2].map&(.double())[0]
        """
        ),
        ("number", 2),
        None,
        id="ufcs-amp-lambda-implicit-subject",
    ),
    pytest.param(
        dedent(
            """\
            notfn := 42
            5.notfn()
        """
        ),
        None,
        ShakarTypeError,
        id="ufcs-non-callable-error",
    ),
    pytest.param(
        "5.nonexistent()",
        None,
        ShakarRuntimeError,
        id="ufcs-missing-name-error",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_functions(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)
