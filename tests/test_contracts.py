from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    ShakarAssertionError,
    ShakarRuntimeError,
    ShakarTypeError,
    run_runtime_case,
)

SCENARIOS = [
    pytest.param(
        dedent(
            """\
            fn f(a, b ~ Int): a + b
            f(1, 2)
        """
        ),
        ("number", 3),
        None,
        id="param-contract-grouped-ok",
    ),
    pytest.param(
        dedent(
            """\
            fn f(a, b ~ Int): a + b
            f("x", 2)
        """
        ),
        None,
        ShakarAssertionError,
        id="param-contract-grouped-fail",
    ),
    pytest.param(
        dedent(
            """\
            fn f(a, (b ~ Int)): a
            f("x", 2)
        """
        ),
        ("string", "x"),
        None,
        id="param-contract-isolated-inner",
    ),
    pytest.param(
        dedent(
            """\
            fn f((a), b ~ Int): a
            f("x", 2)
        """
        ),
        ("string", "x"),
        None,
        id="param-contract-isolated-bare",
    ),
    pytest.param(
        dedent(
            """\
            fn f(...rest ~ Int): rest.len
            f(1, "x", 3)
        """
        ),
        None,
        ShakarAssertionError,
        id="param-contract-spread",
    ),
    pytest.param(
        'all(true, 1, "x")',
        ("bool", True),
        None,
        id="all-varargs",
    ),
    pytest.param(
        "all(true, false, 1)",
        ("bool", False),
        None,
        id="all-varargs-short",
    ),
    pytest.param(
        'all([true, 1, "x"])',
        ("bool", True),
        None,
        id="all-iterable",
    ),
    pytest.param(
        "all([])",
        ("bool", True),
        None,
        id="all-empty-iterable",
    ),
    pytest.param(
        "all()",
        None,
        ShakarRuntimeError,
        id="all-zero-args-error",
    ),
    pytest.param(
        'any(false, 0, "x")',
        ("bool", True),
        None,
        id="any-varargs",
    ),
    pytest.param(
        'any([false, 0, ""])',
        ("bool", False),
        None,
        id="any-iterable",
    ),
    pytest.param(
        "5 ~ Int",
        ("bool", True),
        None,
        id="struct-match-type-int",
    ),
    pytest.param(
        '"hello" ~ Str',
        ("bool", True),
        None,
        id="struct-match-type-str",
    ),
    pytest.param(
        "true ~ Bool",
        ("bool", True),
        None,
        id="struct-match-type-bool",
    ),
    pytest.param(
        "[1, 2] ~ Array",
        ("bool", True),
        None,
        id="struct-match-type-array",
    ),
    pytest.param(
        "{a: 1} ~ Object",
        ("bool", True),
        None,
        id="struct-match-type-object",
    ),
    pytest.param(
        "obj := {a: 1, b: 2, c: 3}; obj ~ {a: Int}",
        ("bool", True),
        None,
        id="struct-match-object-subset",
    ),
    pytest.param(
        "obj := {a: 1, b: 2}; obj ~ {a: Int, b: Int}",
        ("bool", True),
        None,
        id="struct-match-object-multi",
    ),
    pytest.param(
        "{a: 1} ~ {a: Int, b: Int}",
        ("bool", False),
        None,
        id="struct-match-object-missing",
    ),
    pytest.param(
        'u := {name: "Alice", profile: {role: "admin"}}; u ~ {profile: {role: Str}}',
        ("bool", True),
        None,
        id="struct-match-nested",
    ),
    pytest.param(
        "5 ~ 5",
        ("bool", True),
        None,
        id="struct-match-value",
    ),
    pytest.param(
        dedent(
            """\
            fn add(a ~ Int, b ~ Int):
                a + b
            add(1, 2)
        """
        ),
        ("number", 3),
        None,
        id="contract-int-valid",
    ),
    pytest.param(
        dedent(
            """\
            fn greet(name ~ Str):
                "Hello, " + name
            greet("Bob")
        """
        ),
        ("string", "Hello, Bob"),
        None,
        id="contract-str-valid",
    ),
    pytest.param(
        dedent(
            """\
            fn getname(u ~ {name: Str}):
                u.name
            getname({name: "Alice", age: 30})
        """
        ),
        ("string", "Alice"),
        None,
        id="contract-object-valid",
    ),
    pytest.param(
        dedent(
            """\
            fn add(a ~ Int):
                a
            add("wrong")
        """
        ),
        None,
        ShakarAssertionError,
        id="contract-int-invalid",
    ),
    pytest.param(
        dedent(
            """\
            fn f(u ~ {x: Int}):
                u
            f({y: 1})
        """
        ),
        None,
        ShakarAssertionError,
        id="contract-object-invalid",
    ),
    pytest.param(
        "fn inc(x ~ Int): x + 1; inc(5)",
        ("number", 6),
        None,
        id="contract-inline-fn",
    ),
    pytest.param(
        'fn inc(x ~ Int): x + 1; inc("bad")',
        None,
        ShakarAssertionError,
        id="contract-inline-fn-invalid",
    ),
    pytest.param(
        dedent(
            """\
            fn test(x ~ Int):
                defer: x + 1
                x * 2
            test(5)
        """
        ),
        ("number", 10),
        None,
        id="contract-inline-fn-defer",
    ),
    pytest.param(
        "fn add(a ~ Int, b ~ Int): a + b; add(3, 4)",
        ("number", 7),
        None,
        id="contract-inline-multiple",
    ),
    pytest.param(
        "{} ~ {a: Optional(Int)}",
        ("bool", True),
        None,
        id="optional-fn-missing-ok",
    ),
    pytest.param(
        "{a: 1} ~ {a: Optional(Int)}",
        ("bool", True),
        None,
        id="optional-fn-present-valid",
    ),
    pytest.param(
        '{a: "no"} ~ {a: Optional(Int)}',
        ("bool", False),
        None,
        id="optional-fn-present-invalid",
    ),
    pytest.param(
        "{} ~ {a?: Int}",
        ("bool", True),
        None,
        id="optional-syntax-missing-ok",
    ),
    pytest.param(
        "{a: 1, b: 2} ~ {a: Int, b?: Int}",
        ("bool", True),
        None,
        id="optional-syntax-present-valid",
    ),
    pytest.param(
        '{a: "bad"} ~ {a?: Int}',
        ("bool", False),
        None,
        id="optional-syntax-present-invalid",
    ),
    pytest.param(
        '{name: "Alice"} ~ {name: Str, age?: Int}',
        ("bool", True),
        None,
        id="optional-syntax-mixed",
    ),
    pytest.param(
        "{age: 30} ~ {name: Str, age?: Int}",
        ("bool", False),
        None,
        id="optional-syntax-required-missing",
    ),
    pytest.param(
        "5 ~ Union(Int, Str)",
        ("bool", True),
        None,
        id="union-basic-int",
    ),
    pytest.param(
        '"hi" ~ Union(Int, Str)',
        ("bool", True),
        None,
        id="union-basic-str",
    ),
    pytest.param(
        "true ~ Union(Int, Str)",
        ("bool", False),
        None,
        id="union-basic-fail",
    ),
    pytest.param(
        "nil ~ Union(Int, Nil)",
        ("bool", True),
        None,
        id="union-with-nil",
    ),
    pytest.param(
        'Schema := Union(Int, Str); x := 5; x = "hello"; x ~ Schema',
        ("bool", True),
        None,
        id="union-reassign",
    ),
    pytest.param(
        "{age: 30} ~ {age: Union(Int, Str)}",
        ("bool", True),
        None,
        id="union-in-object",
    ),
    pytest.param(
        '{age: "30"} ~ {age: Union(Int, Str)}',
        ("bool", True),
        None,
        id="union-in-object-str",
    ),
    pytest.param(
        "{age: true} ~ {age: Union(Int, Str)}",
        ("bool", False),
        None,
        id="union-in-object-fail",
    ),
    pytest.param(
        '{name: "Alice"} ~ {name: Str, age?: Union(Int, Str)}',
        ("bool", True),
        None,
        id="union-with-optional",
    ),
    pytest.param(
        dedent(
            """\
            fn process(value ~ Union(Int, Str)):
                value
            process(42)
        """
        ),
        ("number", 42),
        None,
        id="union-contract-valid",
    ),
    pytest.param(
        dedent(
            """\
            fn process(value ~ Union(Int, Str)):
                value
            process(true)
        """
        ),
        None,
        ShakarAssertionError,
        id="union-contract-invalid",
    ),
    pytest.param(
        dedent(
            """\
            fn double(x ~ Int) ~ Int:
                x * 2
            double(5)
        """
        ),
        ("number", 10),
        None,
        id="return-contract-basic",
    ),
    pytest.param(
        dedent(
            """\
            fn safe_divide(a ~ Int, b ~ Int) ~ Union(Float, Nil):
                if b == 0:
                    nil
                else:
                    a / b
            safe_divide(10, 2)
        """
        ),
        ("number", 5.0),
        None,
        id="return-contract-union",
    ),
    pytest.param(
        dedent(
            """\
            fn safe_divide(a ~ Int, b ~ Int) ~ Union(Float, Nil):
                if b == 0:
                    nil
                else:
                    a / b
            safe_divide(10, 0)
        """
        ),
        ("null", None),
        None,
        id="return-contract-union-nil",
    ),
    pytest.param(
        dedent(
            """\
            fn inc(x ~ Int) ~ Int: x + 1
            inc(5)
        """
        ),
        ("number", 6),
        None,
        id="return-contract-inline",
    ),
    pytest.param(
        dedent(
            """\
            square := fn(x ~ Int) ~ Int: x * x
            square(4)
        """
        ),
        ("number", 16),
        None,
        id="return-contract-anon",
    ),
    pytest.param(
        dedent(
            """\
            PersonSchema := {name: Str, age: Int}
            fn make_person(name ~ Str, age ~ Int) ~ PersonSchema:
                {name: name, age: age}
            p := make_person("Bob", 30)
            p.age
        """
        ),
        ("number", 30),
        None,
        id="return-contract-object",
    ),
    pytest.param(
        dedent(
            """\
            fn bad_return(x ~ Int) ~ Str:
                x * 2
            bad_return(5)
        """
        ),
        None,
        ShakarTypeError,
        id="return-contract-fail",
    ),
    pytest.param(
        dedent(
            """\
            a ~ Int := 42
            a
        """
        ),
        ("number", 42),
        None,
        id="destructure-contract-single",
    ),
    pytest.param(
        dedent(
            """\
            a ~ Int, (b, c) := [10, [20, 30]]
            a + b + c
        """
        ),
        ("number", 60),
        None,
        id="destructure-mixed-contract-nested",
    ),
    pytest.param(
        dedent(
            """\
            a ~ Int, b ~ Str := 10, "hello"
            a
        """
        ),
        ("number", 10),
        None,
        id="destructure-contract-basic",
    ),
    pytest.param(
        dedent(
            """\
            x ~ Int, y, z ~ Int := 5, "mid", 15
            z - x
        """
        ),
        ("number", 10),
        None,
        id="destructure-contract-partial",
    ),
    pytest.param(
        dedent(
            """\
            m ~ Int, n ~ Int := 42
            m + n
        """
        ),
        ("number", 84),
        None,
        id="destructure-contract-broadcast",
    ),
    pytest.param(
        dedent(
            """\
            id ~ Int, name ~ Str := [100, "Alice"]
            id
        """
        ),
        ("number", 100),
        None,
        id="destructure-contract-array",
    ),
    pytest.param(
        dedent(
            """\
            a ~ Int, b ~ Str := 10, 20
            a
        """
        ),
        None,
        ShakarAssertionError,
        id="destructure-contract-fail",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_contracts(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)
