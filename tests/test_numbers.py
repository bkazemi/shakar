from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    LexError,
    ShakarTypeError,
    run_runtime_case,
)

SCENARIOS = [
    pytest.param(
        dedent(
            """\
            x := 1
            x.len
        """
        ),
        None,
        ShakarTypeError,
        id="number-len-typeerror",
    ),
    pytest.param(
        'int(3) + int("4")',
        ("number", 7),
        None,
        id="int-builtin",
    ),
    pytest.param(
        'float(3) + float("4.5")',
        ("number", 7.5),
        None,
        id="float-builtin",
    ),
    pytest.param(
        "bool(0)",
        ("bool", False),
        None,
        id="bool-builtin-false",
    ),
    pytest.param(
        "bool([1])",
        ("bool", True),
        None,
        id="bool-builtin-true",
    ),
    pytest.param(
        'bool(r"abc")',
        None,
        ShakarTypeError,
        id="bool-builtin-regex-typeerror",
    ),
    pytest.param(
        "str(nil)",
        ("string", "nil"),
        None,
        id="str-builtin-nil",
    ),
    pytest.param(
        "str(true)",
        ("string", "true"),
        None,
        id="str-builtin-bool",
    ),
    pytest.param(
        'float("x")',
        None,
        ShakarTypeError,
        id="float-builtin-typeerror",
    ),
    pytest.param(
        "0b1010_0011",
        ("number", 163),
        None,
        id="base-prefix-binary",
    ),
    pytest.param(
        "0o755",
        ("number", 493),
        None,
        id="base-prefix-octal",
    ),
    pytest.param(
        "0xdead_beef",
        ("number", 3735928559),
        None,
        id="base-prefix-hex",
    ),
    pytest.param(
        "0b111111111111111111111111111111111111111111111111111111111111111",
        ("number", 9223372036854775807),
        None,
        id="base-prefix-binary-63bit",
    ),
    pytest.param(
        "0x7fffffffffffffff",
        ("number", 9223372036854775807),
        None,
        id="base-prefix-hex-max",
    ),
    pytest.param(
        "-0x8000000000000000",
        ("number", -9223372036854775808),
        None,
        id="base-prefix-hex-min-neg",
    ),
    pytest.param(
        "-9223372036854775808",
        ("number", -9223372036854775808),
        None,
        id="decimal-int64-min-neg",
    ),
    pytest.param(
        "9223372036854775808",
        None,
        LexError,
        id="decimal-int64-overflow",
    ),
    pytest.param(
        "0b1000000000000000000000000000000000000000000000000000000000000000",
        None,
        LexError,
        id="base-prefix-binary-overflow",
    ),
    pytest.param(
        "0o1000000000000000000000",
        None,
        LexError,
        id="base-prefix-octal-overflow",
    ),
    pytest.param(
        "0x8000000000000000",
        None,
        LexError,
        id="base-prefix-hex-overflow",
    ),
    pytest.param(
        "0b102",
        None,
        LexError,
        id="base-prefix-invalid-bin",
    ),
    pytest.param(
        "0o9",
        None,
        LexError,
        id="base-prefix-invalid-oct",
    ),
    pytest.param(
        "0xG",
        None,
        LexError,
        id="base-prefix-invalid-hex",
    ),
    pytest.param(
        "0b",
        None,
        LexError,
        id="base-prefix-incomplete",
    ),
    pytest.param(
        "0X10",
        None,
        LexError,
        id="base-prefix-uppercase",
    ),
    pytest.param(
        "0b_101",
        None,
        LexError,
        id="base-prefix-underscore-start",
    ),
    pytest.param(
        "0b101_",
        None,
        LexError,
        id="base-prefix-underscore-end",
    ),
    pytest.param(
        "0b10__01",
        None,
        LexError,
        id="base-prefix-underscore-double",
    ),
    pytest.param(
        "0x10ms",
        None,
        LexError,
        id="base-prefix-duration",
    ),
    pytest.param(
        "0b10s",
        None,
        LexError,
        id="base-prefix-duration-bin",
    ),
    pytest.param(
        "0o755kb",
        None,
        LexError,
        id="base-prefix-size",
    ),
    pytest.param(
        "0x10 * 1msec",
        ("duration", 16000000),
        None,
        id="base-prefix-duration-mul",
    ),
    pytest.param(
        "1_000msec",
        ("duration", 1000000000),
        None,
        id="duration-underscore",
    ),
    pytest.param(
        "1sec500_000usec",
        ("duration", 1500000000),
        None,
        id="duration-compound-underscore",
    ),
    pytest.param(
        "1_000kb",
        ("size", 1000000),
        None,
        id="size-underscore",
    ),
    pytest.param(
        "1mb500_000b",
        ("size", 1500000),
        None,
        id="size-compound-underscore",
    ),
    pytest.param(
        "1_000",
        ("number", 1000),
        None,
        id="decimal-underscore",
    ),
    pytest.param(
        "1_000_000",
        ("number", 1000000),
        None,
        id="decimal-underscore-multi",
    ),
    pytest.param(
        "100_",
        None,
        LexError,
        id="decimal-underscore-trailing",
    ),
    pytest.param(
        "1__0",
        None,
        LexError,
        id="decimal-underscore-double",
    ),
    pytest.param(
        "1_000.5",
        ("number", 1000.5),
        None,
        id="float-underscore-int",
    ),
    pytest.param(
        "1.000_001",
        ("number", 1.000001),
        None,
        id="float-underscore-frac",
    ),
    pytest.param(
        "1_000.000_5",
        ("number", 1000.0005),
        None,
        id="float-underscore-both",
    ),
    pytest.param(
        "1._5",
        None,
        LexError,
        id="float-underscore-leading-frac",
    ),
    pytest.param(
        "1.5_",
        None,
        LexError,
        id="float-underscore-trailing-frac",
    ),
    pytest.param(
        "1e10",
        ("number", 10000000000.0),
        None,
        id="float-exponent",
    ),
    pytest.param(
        "1E10",
        ("number", 10000000000.0),
        None,
        id="float-exponent-upper",
    ),
    pytest.param(
        "1.5e-3",
        ("number", 0.0015),
        None,
        id="float-exponent-neg",
    ),
    pytest.param(
        "1E+5",
        ("number", 100000.0),
        None,
        id="float-exponent-pos",
    ),
    pytest.param(
        "1_000e2",
        ("number", 100000.0),
        None,
        id="float-exponent-underscore",
    ),
    pytest.param(
        "1e1_0",
        ("number", 10000000000.0),
        None,
        id="float-exponent-underscore-exp",
    ),
    pytest.param(
        "1e_5",
        None,
        LexError,
        id="float-exponent-leading-underscore",
    ),
    pytest.param(
        "1.e5",
        None,
        LexError,
        id="float-dot-no-frac",
    ),
    pytest.param(
        "1e5.",
        None,
        LexError,
        id="float-trailing-dot",
    ),
    pytest.param(
        "1scec",
        None,
        LexError,
        id="number-invalid-suffix",
    ),
    pytest.param(
        "1_foo",
        None,
        LexError,
        id="number-invalid-suffix-underscore",
    ),
    pytest.param(
        "5min30sec.total_nsec",
        ("number", 330000000000),
        None,
        id="duration-total-nsec",
    ),
    pytest.param(
        "(1min + 30sec).sec",
        ("number", 90),
        None,
        id="duration-unit-sec",
    ),
    pytest.param(
        "2gb500mb.total_bytes",
        ("number", 2500000000),
        None,
        id="size-total-bytes",
    ),
    pytest.param(
        "(1gb * 2).gb",
        ("number", 2),
        None,
        id="size-unit-gb",
    ),
    pytest.param(
        "1hr30min15sec.sec",
        ("number", 5415),
        None,
        id="duration-compound-multi",
    ),
    pytest.param(
        "5sec / 2sec",
        ("number", 2.5),
        None,
        id="duration-div-ratio",
    ),
    pytest.param(
        "6gb / 2gb",
        ("number", 3),
        None,
        id="size-div-ratio",
    ),
    pytest.param(
        "5min > 3min",
        ("bool", True),
        None,
        id="duration-compare-gt",
    ),
    pytest.param(
        "2sec <= 2sec",
        ("bool", True),
        None,
        id="duration-compare-lte",
    ),
    pytest.param(
        "1mb < 2mb",
        ("bool", True),
        None,
        id="size-compare-lt",
    ),
    pytest.param(
        "5gb >= 4gb",
        ("bool", True),
        None,
        id="size-compare-gte",
    ),
    pytest.param(
        "(-5sec).total_nsec",
        ("number", -5000000000),
        None,
        id="duration-negate",
    ),
    pytest.param(
        "(-2kb).total_bytes",
        ("number", -2000),
        None,
        id="size-negate",
    ),
    pytest.param(
        "(10sec - 3sec).sec",
        ("number", 7),
        None,
        id="duration-sub",
    ),
    pytest.param(
        "(5mb - 2mb).mb",
        ("number", 3),
        None,
        id="size-sub",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_numbers(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)
