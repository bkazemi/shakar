from __future__ import annotations

from textwrap import dedent

from typing import Optional

import pytest

from tests.support.harness import (
    ParseError,
    check_decorated_fn,
    check_decorator_def,
    check_holes,
    check_hook_inline,
    check_map,
    check_zipwith,
    parse_pipeline,
)

CHECKERS = {
    "check_zipwith": check_zipwith,
    "check_map": check_map,
    "check_holes": check_holes,
    "check_hook_inline": check_hook_inline,
    "check_decorator_def": check_decorator_def,
    "check_decorated_fn": check_decorated_fn,
    None: None,
}

EXCEPTIONS = {
    "ParseError": ParseError,
    "SyntaxError": SyntaxError,
    None: None,
}

AST_CASES = [
    ("lambda-infer-zipwith", "zipWith&(left + right)(xs, ys)", "check_zipwith", None),
    ("lambda-respect-subject", "map&(.trim())", "check_map", None),
    ("lambda-hole-desugar", "blend(?, ?, 0.25)", "check_holes", None),
    ("lambda-dot-mix-error", "map&(value + .trim())", None, "SyntaxError"),
    ("hook-inline-body", 'hook "warn": .trim()', "check_hook_inline", None),
    ("decorator-ast-def", "decorator logger(msg): args", "check_decorator_def", None),
    (
        "decorated-fn-ast",
        dedent(
            """\
            @noop
            fn hi(): 1
        """
        ),
        "check_decorated_fn",
        None,
    ),
    ("emit-outside-call", "> 1", None, "ParseError"),
    ("reserved-fan-field", "obj.fan", None, "ParseError"),
    ("param-destruct-outer-default", "fn f({a} = {}): a", None, "ParseError"),
    (
        "param-group-destruct-inner-contract",
        "fn f(({a ~ Int}, b) ~ {a: Int}): a",
        None,
        "ParseError",
    ),
    (
        "match-empty-body",
        dedent(
            """\
            match x:
            
        """
        ),
        None,
        "ParseError",
    ),
    (
        "match-pattern-dot-error",
        dedent(
            """\
            match x:
              .: 1
            
        """
        ),
        None,
        "ParseError",
    ),
    ("param-group-nested-contract", "fn f((a ~ Int, b) ~ Int): a", None, "ParseError"),
]


@pytest.mark.parametrize(
    "code, checker_name, expected_exc_name",
    [
        pytest.param(code, checker_name, expected_exc_name, id=name)
        for name, code, checker_name, expected_exc_name in AST_CASES
    ],
)
def test_ast_cases(
    code: str,
    checker_name: Optional[str],
    expected_exc_name: Optional[str],
) -> None:
    checker = CHECKERS[checker_name]
    expected_exc = EXCEPTIONS[expected_exc_name]

    if expected_exc is not None:
        with pytest.raises(expected_exc):
            parse_pipeline(code, use_indenter=False)
        return

    ast = parse_pipeline(code, use_indenter=False)
    if checker is not None:
        err = checker(ast)
        assert err is None, err
