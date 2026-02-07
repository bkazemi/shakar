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
            count := 0
            fn bump(x): count += x
            call bump:
              > 1
              > 2
            count
        """
        ),
        ("number", 3),
        None,
        id="call-basic-emit",
    ),
    pytest.param(
        dedent(
            """\
            log := ""
            fn out(x): log += x
            fn wrap(x): log += "[" + x + "]"
            call out:
              helper := fn(): > "a"
              call wrap:
                > "b"
                helper()
            log
        """
        ),
        ("string", "[b]a"),
        None,
        id="call-emit-capture",
    ),
    pytest.param(
        dedent(
            """\
            log := ""
            fn emit(x): log += x
            call emit:
              > "a" unless false
              > "b" unless true
            log
        """
        ),
        ("string", "a"),
        None,
        id="call-emit-postfix-unless",
    ),
    pytest.param(
        dedent(
            """\
            log := ""
            fn emit(a, b): log += a + ":" + b
            call emit:
              > a: "x", b: "y"
            log
        """
        ),
        ("string", "x:y"),
        None,
        id="call-emit-named-args",
    ),
    pytest.param(
        dedent(
            """\
            count := 0
            fn emit(...args): count += args.len
            call emit:
              > ...[1, 2, 3]
            count
        """
        ),
        ("number", 3),
        None,
        id="call-emit-spread",
    ),
    pytest.param(
        dedent(
            """\
            log := ""
            fn outer(x): log += "o" + x
            fn inner(x): log += "i" + x
            call outer:
              > "1"
              call inner:
                > "2"
              > "3"
            log
        """
        ),
        ("string", "o1i2o3"),
        None,
        id="call-nested-shadow",
    ),
    pytest.param(
        dedent(
            """\
            x := 42
            call x:
              > 1
        """
        ),
        None,
        ShakarRuntimeError,
        id="call-non-callable-error",
    ),
    pytest.param(
        dedent(
            """\
            log := ""
            fn emit(a, b): log += a + b
            call emit:
              > "x", "y",
            log
        """
        ),
        ("string", "xy"),
        None,
        id="call-emit-trailing-comma",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_call_blocks(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)
