from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    ParseError,
    ShakarRuntimeError,
    parse_pipeline,
    run_runtime_case,
    verify_result,
)
from shakar_ref.runner import eval_in_env, run as run_program, run_with_env
from shakar_ref.tree import Tree

RUNTIME_SCENARIOS = [
    # --- Expression-form once: <expr> ---
    pytest.param(
        "once: 1",
        ("number", 1),
        None,
        id="once-expression-resolves-to-value",
    ),
    pytest.param(
        "[once: 1][0] + {x: once: 2}.x",
        ("number", 3),
        None,
        id="once-in-literals-resolves",
    ),
    pytest.param(
        dedent(
            """\
            now := 1
            static := 2
            now + static
        """
        ),
        ("number", 3),
        None,
        id="now-and-static-remain-identifiers-outside-modifiers",
    ),
    pytest.param(
        dedent(
            """\
            acc := 0
            for [1]:
              acc += once: .
            acc
        """
        ),
        ("number", 1),
        None,
        id="once-captures-subject-anchor-at-binding-site",
    ),
    # --- once caching: shared across loop iterations ---
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            for i in 3:
              once: bump()
            hits
        """
        ),
        ("number", 1),
        None,
        id="once-shared-across-for-iterations-discard",
    ),
    # --- Bare once: is always eager (runs and caches immediately) ---
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            fn f():
              once: bump()
              once: bump()
              hits
            f()
        """
        ),
        ("number", 2),
        None,
        id="bare-once-always-eager-distinct-sites",
    ),
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            fn f():
              for i in 3:
                once: bump()
                nil
              hits
            f()
        """
        ),
        ("number", 1),
        None,
        id="bare-once-fires-once-even-in-non-final-position",
    ),
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            total := 0
            for i in 3:
              total += once: bump()
            [total, hits]
        """
        ),
        ("array", [3, 1]),
        None,
        id="once-shared-across-for-iterations",
    ),
    # --- once scoping: per-call vs static ---
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            fn f():
              once: bump()
            [f(), f(), hits]
        """
        ),
        ("array", [1, 2, 2]),
        None,
        id="non-static-is-per-function-call",
    ),
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            fn f():
              once[static]: bump()
            [f(), f(), hits]
        """
        ),
        ("array", [1, 1, 1]),
        None,
        id="static-is-shared-across-function-calls",
    ),
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            fn f():
              once[static]: bump()
              hits
            [f(), f(), hits]
        """
        ),
        ("array", [1.0, 1.0, 1.0]),
        None,
        id="static-is-eager-and-shared",
    ),
    # --- Error caching ---
    pytest.param(
        dedent(
            """\
            hits := 0
            fn fail():
              hits += 1
              throw "boom"
            fn f():
              once: fail()
            f() catch err: nil
            f() catch err: nil
            hits
        """
        ),
        ("number", 2),
        None,
        id="runtime-error-is-per-call-nonstatic",
    ),
    pytest.param(
        dedent(
            """\
            hits := 0
            fn fail():
              hits += 1
              throw "boom"
            fn f():
              once[static]: fail()
            f() catch err: nil
            f() catch err: nil
            hits
        """
        ),
        ("number", 1),
        None,
        id="runtime-error-is-cached-static",
    ),
    # --- Prefix once with walrus body (now eager by default) ---
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            once: x := bump()
            hits
        """
        ),
        ("number", 1),
        None,
        id="eager-walrus-fires-immediately",
    ),
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            once: x := bump()
            x + x + hits
        """
        ),
        ("number", 3),
        None,
        id="eager-walrus-value-cached",
    ),
    # Eager walrus: evaluated and cached immediately.
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            total := 0
            for i in 3:
              once: x := bump()
              total += x
            [total, hits]
        """
        ),
        ("array", [3, 1]),
        None,
        id="prefix-once-walrus-cached-in-loop",
    ),
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            fn sink(v):
              nil
            sink(once: x := bump())
            [x, hits]
        """
        ),
        ("array", [1, 1]),
        None,
        id="prefix-once-walrus-in-call-arg",
    ),
    pytest.param(
        "((once: x := nil) ?? 1) + 0",
        ("number", 1),
        None,
        id="prefix-once-nullish-coalesces-nil",
    ),
    pytest.param(
        dedent(
            """\
            once: f := fn(x): x + 1
            1.f()
        """
        ),
        ("number", 2),
        None,
        id="prefix-once-walrus-ufcs-eager",
    ),
    pytest.param(
        dedent(
            """\
            once[lazy]: f := fn(x): x + 1
            1.f()
        """
        ),
        ("number", 2),
        None,
        id="prefix-once-walrus-ufcs-lazy",
    ),
    pytest.param(
        dedent(
            """\
            fn id(v): v
            id(once: x := 1)
        """
        ),
        ("number", 1),
        None,
        id="prefix-once-walrus-arg-eval",
    ),
    # --- RHS form: x := once: expr ---
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            total := 0
            for i in 3:
              v := once: bump()
              total += v
            [total, hits]
        """
        ),
        ("array", [3, 1]),
        None,
        id="rhs-once-walrus-caches-value",
    ),
    # --- Lazy modifier: once[lazy]: x := expr ---
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            once[lazy]: x := bump()
            hits
        """
        ),
        ("number", 0),
        None,
        id="lazy-walrus-defers-until-read",
    ),
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            fn sink(v):
              nil
            sink(once[lazy]: x := bump())
            [x, hits]
        """
        ),
        ("array", [1, 1]),
        None,
        id="lazy-walrus-in-call-arg-defers-then-reads",
    ),
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            once[lazy]: x := bump()
            x = 2
            [x, hits]
        """
        ),
        ("array", [2, 0]),
        None,
        id="lazy-walrus-assign-before-read-overrides-thunk",
    ),
    pytest.param(
        dedent(
            """\
            once[lazy]: x := 1
            mixin({x: 2})
            x
        """
        ),
        ("number", 2),
        None,
        id="lazy-walrus-stale-thunk-does-not-shadow-define",
    ),
    pytest.param(
        dedent(
            """\
            once[lazy]: x := 1
            [x for [2, 3]]
        """
        ),
        ("array", [1, 1]),
        None,
        id="lazy-walrus-name-visible-to-implicit-binder-inference",
    ),
    pytest.param(
        dedent(
            """\
            arr := [7]
            arr[(once[lazy]: i := 0)]
            marker := 1
            [i, marker]
        """
        ),
        ("array", [0, 1]),
        None,
        id="lazy-walrus-index-subexpr-non-final-still-value",
    ),
    pytest.param(
        dedent(
            """\
            obj := {a: 0, b: 0}
            obj{ .a = once[lazy]: x := 2; .b = 3 }
            [obj.a, obj.b, x]
        """
        ),
        ("array", [2, 3, 2]),
        None,
        id="lazy-walrus-fanout-rhs-non-final-still-value",
    ),
    pytest.param(
        dedent(
            """\
            fn mk(): ({a: 0})
            (once[lazy]: o := mk()).a = 1
            marker := 0
            [o.a, marker]
        """
        ),
        ("array", [1, 0]),
        None,
        id="lazy-walrus-lvalue-head-non-final-still-value",
    ),
    pytest.param(
        dedent(
            """\
            a, b = (once[lazy]: x := 2) := [1]
            marker := 0
            [b, x, marker]
        """
        ),
        ("array", [2, 2, 0]),
        None,
        id="lazy-walrus-destructure-default-non-final-still-value",
    ),
    pytest.param(
        dedent(
            """\
            a ~ (once[lazy]: c := 1) := 1
            marker := 0
            [a, c, marker]
        """
        ),
        ("array", [1, 1, 0]),
        None,
        id="lazy-walrus-destructure-contract-non-final-still-value",
    ),
    pytest.param(
        dedent(
            """\
            total := 0
            for i in 3:
              once[lazy]: x := 1
              mixin({x: 2})
              total += x
            total
        """
        ),
        ("number", 6),
        None,
        id="lazy-walrus-site-id-persists-through-define-overrides",
    ),
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            fn f():
              once[lazy, static]: x := bump()
              x
            [f(), f(), hits]
        """
        ),
        ("array", [1, 1, 1]),
        None,
        id="lazy-static-walrus-shared-across-calls",
    ),
    # --- Lazy once in loops: must not raise duplicate-definition ---
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            total := 0
            for i in 3:
              once[lazy]: x := bump()
              total += x
            [total, hits]
        """
        ),
        ("array", [3, 1]),
        None,
        id="lazy-walrus-in-loop-no-duplicate-error",
    ),
    # --- Lazy once as RHS of walrus in non-final statement ---
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            once[lazy]: x := bump()
            y := x
            [y, hits]
        """
        ),
        ("array", [1, 1]),
        None,
        id="lazy-walrus-rhs-non-final-resolves-on-read",
    ),
    pytest.param(
        dedent(
            """\
            y := once[lazy]: x := 1
            marker := 0
            [y, x, marker]
        """
        ),
        ("array", [1, 1, 0]),
        None,
        id="lazy-walrus-rhs-value-context-in-non-final-stmt",
    ),
    pytest.param(
        dedent(
            """\
            fn f():
              once[lazy]: x := 42
              y := x + 1
              y
            assert f() == 43
        """
        ),
        None,
        None,
        id="lazy-walrus-nested-subexpr-always-resolves",
    ),
    pytest.param(
        dedent(
            """\
            let y := once[lazy]: x := 1
            marker := 0
            [y, x, marker]
        """
        ),
        ("array", [1, 1, 0]),
        None,
        id="lazy-walrus-rhs-value-context-in-let-non-final-stmt",
    ),
    pytest.param(
        dedent(
            """\
            x := 0
            once[lazy]: x := 1
        """
        ),
        None,
        ShakarRuntimeError,
        id="lazy-walrus-duplicate-with-existing-name-errors",
    ),
    pytest.param(
        dedent(
            """\
            once[lazy]: x := 1
            once[lazy]: x := 2
        """
        ),
        None,
        ShakarRuntimeError,
        id="lazy-walrus-duplicate-across-sites-errors",
    ),
    pytest.param(
        dedent(
            """\
            once[lazy]: x := 1
            let x := 2
        """
        ),
        None,
        ShakarRuntimeError,
        id="lazy-walrus-conflicts-with-let-shadow-check",
    ),
    pytest.param(
        dedent(
            """\
            v := once: {a: 1}
            v.a
        """
        ),
        ("number", 1),
        None,
        id="once-object-literal-body-stays-expression",
    ),
    # --- Block form: once:\n  stmts ---
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            result := 0
            for i in 3:
              result = once:
                bump()
                42
            [result, hits]
        """
        ),
        ("array", [42, 1]),
        None,
        id="block-once-caches-last-value",
    ),
    pytest.param(
        dedent(
            """\
            out := nil
            for [5]:
              out = once:
                .
            out
        """
        ),
        ("number", 5),
        None,
        id="block-once-preserves-subject-anchor",
    ),
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            v := once:
              bump()
              42
            [v, hits]
        """
        ),
        ("array", [42, 1]),
        None,
        id="block-once-walrus-rhs",
    ),
    pytest.param(
        dedent(
            """\
            result := once:
              inner := 99
              inner
            leaked := false
            inner catch err:
              leaked = true
            [result, leaked]
        """
        ),
        ("array", [99, True]),
        None,
        id="block-once-scoping-no-leak",
    ),
    # --- Hole partial + once ---
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            fn blend(a, b): a + b
            f := blend(once: bump(), ?)
            [f(1), hits]
        """
        ),
        ("array", [2, 1]),
        None,
        id="hole-partial-once-does-not-require-parser-once-id",
    ),
    pytest.param(
        dedent(
            """\
            hits := 0
            fn bump():
              hits += 1
              hits
            fn blend(a, b): a + b
            f := blend(once[static]: bump(), ?)
            [f(1), f(2), hits]
        """
        ),
        ("array", [2, 3, 1]),
        None,
        id="hole-partial-once-static-repeat-invocation-stable",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", RUNTIME_SCENARIOS)
def test_once_runtime(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)


def test_once_cache_keys_do_not_collide_across_eval_in_env() -> None:
    frame = run_with_env("seed := 0")

    for index in range(64):
        result = eval_in_env(f"once: {index}", frame)
        verify_result(result, "number", index)


# --- Parse error tests ---
@pytest.mark.parametrize(
    "source, message",
    [
        pytest.param(
            "once[static, static]: 1",
            "duplicate once modifier 'static'",
            id="duplicate-static",
        ),
        pytest.param(
            "once[later]: 1",
            "unknown once modifier 'later'",
            id="unknown-modifier",
        ),
        pytest.param(
            "once[now]: 1",
            "unknown once modifier 'now'",
            id="now-modifier-removed",
        ),
        pytest.param(
            "once[lazy]: 1",
            "lazy requires prefix walrus binding",
            id="lazy-without-walrus",
        ),
    ],
)
def test_once_parse_errors(source: str, message: str) -> None:
    for use_indenter in (False, True):
        with pytest.raises(ParseError, match=message):
            parse_pipeline(source, use_indenter=use_indenter)


def _collect_tree_labels(node: object) -> set[str]:
    labels: set[str] = set()
    if not isinstance(node, Tree):
        return labels

    stack = [node]
    while stack:
        current = stack.pop()
        labels.add(current.data)
        for child in current.children:
            if isinstance(child, Tree):
                stack.append(child)
    return labels


def test_once_parse_shape() -> None:
    ast = parse_pipeline("once[static]: 1", use_indenter=False)
    labels = _collect_tree_labels(ast)
    assert "once_expr" in labels
    assert "once_modifiers" in labels


def test_once_object_literal_parse_shape() -> None:
    source = "v := once: {a: 1}"

    for use_indenter in (False, True):
        ast = parse_pipeline(source, use_indenter=use_indenter)
        labels = _collect_tree_labels(ast)
        assert "once_expr" in labels
        assert "object" in labels


def test_lazy_walrus_ufcs_initializer_error_propagates() -> None:
    source = dedent(
        """\
        fn fail():
          throw "boom"
        once[lazy]: f := fail()
        1.f()
    """
    )

    with pytest.raises(ShakarRuntimeError) as excinfo:
        run_program(source)

    assert "boom" in str(excinfo.value)
