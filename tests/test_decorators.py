from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import run_runtime_case

SCENARIOS = [
    pytest.param(
        dedent(
            """\
            decorator noop(): args
            @noop
            fn hi(name): name
            hi("Ada")
        """
        ),
        ("string", "Ada"),
        None,
        id="decorator-pass-through",
    ),
    pytest.param(
        dedent(
            """\
            decorator double(): args[0] = args[0] * 2
            @double
            fn sum(x, y): x + y
            sum(3, 4)
        """
        ),
        ("number", 10),
        None,
        id="decorator-arg-mutate",
    ),
    pytest.param(
        dedent(
            """\
            decorator always_nil(): return nil
            @always_nil
            fn value(): 42
            value()
        """
        ),
        ("null", None),
        None,
        id="decorator-return-shortcut",
    ),
    pytest.param(
        dedent(
            """\
            decorator mark(label): args[0] = args[0] * 10 + label
            @mark(3)
            @mark(2)
            fn encode(x): x
            encode(1)
        """
        ),
        ("number", 123),
        None,
        id="decorator-chain-order",
    ),
    pytest.param(
        dedent(
            """\
            decorator double_call(): return f(args) + f(args)
            @double_call
            fn point(): 2
            point()
        """
        ),
        ("number", 4),
        None,
        id="decorator-call-twice",
    ),
    pytest.param(
        dedent(
            """\
            decorator add_offset(delta): args[0] = args[0] + delta
            @add_offset(5)
            fn bump(x): x
            bump(3)
        """
        ),
        ("number", 8),
        None,
        id="decorator-params",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_decorators(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)
