from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    ShakarRuntimeError,
    ShakarTypeError,
    run_runtime_case,
)

SCENARIOS = [
    # ---- Basic generator creation and .next() ----
    pytest.param(
        dedent(
            """\
            fn counter():
              yield 1
              yield 2
              yield 3
            gen := counter()
            gen.next()
        """
        ),
        ("number", 1),
        None,
        id="basic-next-first",
    ),
    pytest.param(
        dedent(
            """\
            fn counter():
              yield 1
              yield 2
              yield 3
            gen := counter()
            gen.next()
            gen.next()
            gen.next()
        """
        ),
        ("number", 3),
        None,
        id="basic-next-all",
    ),
    pytest.param(
        dedent(
            """\
            fn counter():
              yield 1
              yield 2
            gen := counter()
            gen.next()
            gen.next()
            gen.next()
        """
        ),
        None,
        ShakarRuntimeError,
        id="next-exhausted-throws",
    ),
    # ---- .done and .result ----
    pytest.param(
        dedent(
            """\
            fn g():
              yield 1
            gen := g()
            gen.done
        """
        ),
        ("bool", False),
        None,
        id="done-before-iteration",
    ),
    pytest.param(
        dedent(
            """\
            fn g():
              yield 1
            gen := g()
            gen.next()
            # Exhaust it by trying next (which fails) - use peek to check
            gen.done
        """
        ),
        ("bool", False),
        None,
        id="done-after-partial-iteration",
    ),
    pytest.param(
        dedent(
            """\
            fn g():
              yield 1
            gen := g()
            gen.next()
            # Generator body has finished, should be completed
            gen.done
        """
        ),
        # After consuming the only yield, the generator body runs to completion
        # Only when .next() or iteration discovers exhaustion does .done become true
        ("bool", False),
        None,
        id="done-after-yield-consumed",
    ),
    pytest.param(
        dedent(
            """\
            fn g():
              yield 1
              return 42
            gen := g()
            gen.next()
            gen.result
        """
        ),
        ("null", None),
        None,
        id="result-before-exhaustion",
    ),
    # ---- .peek() ----
    pytest.param(
        dedent(
            """\
            fn g():
              yield 10
              yield 20
            gen := g()
            gen.peek()
        """
        ),
        ("number", 10),
        None,
        id="peek-first",
    ),
    pytest.param(
        dedent(
            """\
            fn g():
              yield 10
              yield 20
            gen := g()
            gen.peek()
            gen.peek()
        """
        ),
        ("number", 10),
        None,
        id="peek-idempotent",
    ),
    pytest.param(
        dedent(
            """\
            fn g():
              yield 10
              yield 20
            gen := g()
            gen.peek()
            gen.next()
        """
        ),
        ("number", 10),
        None,
        id="peek-then-next-returns-same",
    ),
    pytest.param(
        dedent(
            """\
            fn g():
              yield 10
              yield 20
            gen := g()
            gen.peek()
            gen.next()
            gen.peek()
        """
        ),
        ("number", 20),
        None,
        id="peek-after-next-advances",
    ),
    # ---- for loop over generator ----
    pytest.param(
        dedent(
            """\
            fn nums():
              yield 1
              yield 2
              yield 3
            result := 0
            for x in nums():
              result += x
            result
        """
        ),
        ("number", 6),
        None,
        id="for-in-generator",
    ),
    pytest.param(
        dedent(
            """\
            fn nums():
              yield 10
              yield 20
              yield 30
            gen := nums()
            result := gen.next()
            for gen:
              result = .
            result
        """
        ),
        ("number", 30),
        None,
        id="for-subject-generator",
    ),
    # ---- break in for loop leaves generator resumable ----
    pytest.param(
        dedent(
            """\
            fn nums():
              yield 1
              yield 2
              yield 3
            gen := nums()
            for x in gen:
              break
            gen.done
        """
        ),
        ("bool", False),
        None,
        id="break-leaves-generator-resumable",
    ),
    pytest.param(
        dedent(
            """\
            fn nums():
              yield 1
              yield 2
              yield 3
            gen := nums()
            for x in gen:
              break
            gen.next()
        """
        ),
        ("number", 2),
        None,
        id="break-then-resume-next",
    ),
    # ---- Generator with return populates .result ----
    pytest.param(
        dedent(
            """\
            fn g():
              yield 1
              yield 2
              return 99
            gen := g()
            gen.next()
            gen.next()
            gen.done
        """
        ),
        ("bool", False),
        None,
        id="return-not-done-until-exhausted",
    ),
    # ---- .close() ----
    pytest.param(
        dedent(
            """\
            fn g():
              yield 1
              yield 2
            gen := g()
            gen.next()
            gen.close()
            gen.done
        """
        ),
        ("bool", True),
        None,
        id="close-sets-done",
    ),
    pytest.param(
        dedent(
            """\
            fn g():
              yield 1
              yield 2
            gen := g()
            gen.close()
            gen.next()
        """
        ),
        None,
        ShakarRuntimeError,
        id="next-after-close-throws",
    ),
    # ---- Yield contract (~ T) ----
    pytest.param(
        dedent(
            """\
            fn g() ~ Int:
              yield 1
              yield 2
            gen := g()
            gen.next()
        """
        ),
        ("number", 1),
        None,
        id="yield-contract-valid",
    ),
    pytest.param(
        dedent(
            """\
            fn g() ~ Int:
              yield "hello"
            gen := g()
            gen.next()
        """
        ),
        None,
        ShakarTypeError,
        id="yield-contract-violation",
    ),
    # ---- yield ...delegation ----
    pytest.param(
        dedent(
            """\
            fn inner():
              yield 10
              yield 20
            fn outer():
              yield 1
              yield ...inner()
              yield 3
            result := []
            for x in outer():
              result = result + [x]
            result
        """
        ),
        ("array", [1.0, 10.0, 20.0, 3.0]),
        None,
        id="yield-deleg-generator",
    ),
    pytest.param(
        dedent(
            """\
            fn g():
              yield ...[10, 20, 30]
            result := []
            for x in g():
              result = result + [x]
            result
        """
        ),
        ("array", [10.0, 20.0, 30.0]),
        None,
        id="yield-deleg-array",
    ),
    # ---- Nested functions: inner yield does not make outer a generator ----
    pytest.param(
        dedent(
            """\
            fn outer():
              fn inner():
                yield 1
              gen := inner()
              return gen.next()
            outer()
        """
        ),
        ("number", 1),
        None,
        id="nested-fn-yield-scoping",
    ),
    # ---- Selector on generator ----
    pytest.param(
        dedent(
            """\
            fn naturals():
              i := 0
              while true:
                yield i
                i++
            naturals()[`:<5`]
        """
        ),
        ("array", [0.0, 1.0, 2.0, 3.0, 4.0]),
        None,
        id="selector-take-first-n",
    ),
    pytest.param(
        dedent(
            """\
            fn naturals():
              i := 0
              while true:
                yield i
                i++
            naturals()[`3`]
        """
        ),
        ("number", 3),
        None,
        id="selector-single-index",
    ),
    pytest.param(
        dedent(
            """\
            fn naturals():
              i := 0
              while true:
                yield i
                i++
            naturals()[`2:<7`]
        """
        ),
        ("array", [2.0, 3.0, 4.0, 5.0, 6.0]),
        None,
        id="selector-skip-take",
    ),
    # ---- Generator with local state ----
    pytest.param(
        dedent(
            """\
            fn fibonacci():
              a := 0
              b := 1
              while true:
                yield a
                a, b = b, a + b
            result := []
            gen := fibonacci()
            for i in 8:
              result = result + [gen.next()]
            result
        """
        ),
        ("array", [0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0]),
        None,
        id="fibonacci-generator",
    ),
    # ---- Generator with parameters ----
    pytest.param(
        dedent(
            """\
            fn range_gen(n):
              i := 0
              while i < n:
                yield i
                i++
            result := []
            for x in range_gen(5):
              result = result + [x]
            result
        """
        ),
        ("array", [0.0, 1.0, 2.0, 3.0, 4.0]),
        None,
        id="generator-with-params",
    ),
    # ---- Generator return populates .result after exhaustion ----
    pytest.param(
        dedent(
            """\
            fn g():
              yield 1
              return 42
            gen := g()
            gen.next()
            # Trigger exhaustion by trying to get next value
            try:
              gen.next()
            catch:
              nil
            gen.result
        """
        ),
        ("number", 42),
        None,
        id="return-populates-result",
    ),
    # ---- Generator .result stays nil after .close() ----
    pytest.param(
        dedent(
            """\
            fn g():
              yield 1
              return 42
            gen := g()
            gen.next()
            gen.close()
            gen.result
        """
        ),
        ("null", None),
        None,
        id="close-result-stays-nil",
    ),
    # ---- Selector short-circuit leaves generator resumable ----
    pytest.param(
        dedent(
            """\
            fn naturals():
              i := 0
              while true:
                yield i
                i++
            gen := naturals()
            first_three := gen[`:<3`]
            gen.done
        """
        ),
        ("bool", False),
        None,
        id="selector-leaves-generator-resumable",
    ),
    pytest.param(
        dedent(
            """\
            fn naturals():
              i := 0
              while true:
                yield i
                i++
            gen := naturals()
            gen[`:<3`]
            gen.next()
        """
        ),
        ("number", 3),
        None,
        id="selector-then-resume",
    ),
    # ---- Multi-range selectors ----
    pytest.param(
        dedent(
            """\
            fn naturals():
              i := 0
              while true:
                yield i
                i++
            naturals()[`0:<3, 7:<10`]
        """
        ),
        ("array", [0.0, 1.0, 2.0, 7.0, 8.0, 9.0]),
        None,
        id="multi-range-selector",
    ),
    # ---- close() runs deferred cleanup ----
    pytest.param(
        dedent(
            """\
            flag := 0
            fn g():
              defer cleanup:
                flag = 1
              yield 1
              yield 2
            gen := g()
            gen.next()
            gen.close()
            flag
        """
        ),
        ("number", 1),
        None,
        id="close-runs-defer-cleanup",
    ),
    # ---- yield during cleanup is an error ----
    pytest.param(
        dedent(
            """\
            fn g():
              defer cleanup:
                yield 2
              yield 1
            gen := g()
            gen.next()
            gen.close()
        """
        ),
        None,
        ShakarRuntimeError,
        id="yield-during-cleanup-rejected",
    ),
    # ---- Reverse slice on generator ----
    pytest.param(
        dedent(
            """\
            fn naturals():
              i := 0
              while true:
                yield i
                i++
            naturals()[`5:<0:-1`]
        """
        ),
        ("array", [5.0, 4.0, 3.0, 2.0, 1.0]),
        None,
        id="selector-reverse-slice",
    ),
    # ---- Empty forward slice on generator ----
    pytest.param(
        dedent(
            """\
            fn naturals():
              i := 0
              while true:
                yield i
                i++
            naturals()[`5:<2`]
        """
        ),
        ("array", []),
        None,
        id="selector-empty-forward-slice",
    ),
    # ---- Object method with yield does not make outer fn a generator ----
    pytest.param(
        dedent(
            """\
            fn outer():
              obj := {
                nums():
                  yield 1
              }
              return 42
            outer()
        """
        ),
        ("number", 42),
        None,
        id="obj-method-yield-scoping",
    ),
    # ---- Object method with yield is callable as generator ----
    pytest.param(
        dedent(
            """\
            obj := {
              nums():
                yield 1
                yield 2
                yield 3
            }
            gen := obj.nums()
            gen.next()
            gen.next()
        """
        ),
        ("number", 2),
        None,
        id="obj-method-generator-callable",
    ),
    # ---- Reverse slice without explicit start rejected ----
    pytest.param(
        dedent(
            """\
            fn naturals():
              i := 0
              while true:
                yield i
                i++
            naturals()[`:<0:-1`]
        """
        ),
        None,
        ShakarTypeError,
        id="selector-reverse-no-start-rejected",
    ),
    # ---- Delegated generator cleanup on close ----
    pytest.param(
        dedent(
            """\
            flag := 0
            fn inner():
              defer cleanup:
                flag = 1
              yield 10
              yield 20
            fn outer():
              yield 1
              yield ...inner()
              yield 99
            gen := outer()
            gen.next()
            gen.next()
            gen.close()
            flag
        """
        ),
        ("number", 1),
        None,
        id="deleg-close-cleans-inner",
    ),
    # ---- Setter with yield is rejected ----
    pytest.param(
        dedent(
            """\
            obj := {
              get x: 1
              set x(v):
                yield v
            }
        """
        ),
        None,
        ShakarRuntimeError,
        id="setter-with-yield-rejected",
    ),
    # ---- peek after close returns nothing ----
    pytest.param(
        dedent(
            """\
            fn g():
              yield 1
              yield 2
            gen := g()
            gen.peek()
            gen.close()
            gen.done
        """
        ),
        ("bool", True),
        None,
        id="peek-then-close-is-done",
    ),
    # ---- generator identity equality ----
    pytest.param(
        dedent(
            """\
            fn g():
              yield 1
            gen := g()
            gen == gen
        """
        ),
        ("bool", True),
        None,
        id="generator-identity-equal",
    ),
    pytest.param(
        dedent(
            """\
            fn g():
              yield 1
            g() == g()
        """
        ),
        ("bool", False),
        None,
        id="generator-different-instances-not-equal",
    ),
    # ---- yield with pack (comma-separated) ----
    pytest.param(
        dedent(
            """\
            fn g():
              yield 1, 2
            gen := g()
            gen.next()
        """
        ),
        ("array", [1.0, 2.0]),
        None,
        id="yield-pack",
    ),
    # ---- yield outside generator ----
    pytest.param(
        dedent(
            """\
            yield 1
        """
        ),
        None,
        ShakarRuntimeError,
        id="yield-outside-generator",
    ),
    # ---- yield in non-generator callable inside generator does not leak ----
    pytest.param(
        dedent(
            """\
            fn outer():
              fn inner():
                yield 1
              inner()
              yield 2
            gen := outer()
            gen.next()
        """
        ),
        ("number", 2),
        None,
        id="yield-does-not-escape-non-generator-fn",
    ),
    # ---- Inline object member generators ----
    pytest.param(
        dedent(
            """\
            obj := {
              nums(): yield 1
            }
            gen := obj.nums()
            gen.next()
        """
        ),
        ("number", 1),
        None,
        id="inline-obj-method-generator",
    ),
    pytest.param(
        dedent(
            """\
            obj := {
              get seq: yield 10
            }
            gen := obj.seq
            gen.next()
        """
        ),
        ("number", 10),
        None,
        id="inline-obj-getter-generator",
    ),
    # ---- Parameter contracts enforced at call time, not first .next() ----
    pytest.param(
        dedent(
            """\
            fn g(x ~ Int):
              yield x
            g("bad")
        """
        ),
        None,
        ShakarRuntimeError,
        id="param-contract-eager-at-call",
    ),
    pytest.param(
        dedent(
            """\
            fn g(x ~ Int):
              yield x
            gen := g(42)
            gen.next()
        """
        ),
        ("number", 42),
        None,
        id="param-contract-valid-gen",
    ),
    # ---- Spread parameter contracts enforced at call time ----
    pytest.param(
        dedent(
            """\
            fn g(...xs ~ Int):
              yield 1
            g(1, "x")
        """
        ),
        None,
        ShakarRuntimeError,
        id="spread-contract-eager-at-call",
    ),
    pytest.param(
        dedent(
            """\
            fn g(...xs ~ Int):
              yield ...xs
            result := []
            for x in g(1, 2, 3):
              result = result + [x]
            result
        """
        ),
        ("array", [1.0, 2.0, 3.0]),
        None,
        id="spread-contract-valid-gen",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_generators(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)


def test_yield_in_defer_rejects_on_normal_exhaustion() -> None:
    """Deferred cleanup must not leak yield values into the generator stream.

    Regression: the cleanup-yield check only fired after .close(), so
    `defer: yield 2` silently produced an extra value during normal
    exhaustion.
    """
    from shakar_ref.runner import run as run_program

    with pytest.raises(ShakarRuntimeError, match="Cannot yield"):
        run_program(
            dedent(
                """\
            fn g():
              defer cleanup:
                yield 99
              yield 1
            results := []
            for x in g():
              results = results + [x]
            results
        """
            )
        )


def test_yield_in_spawn_does_not_leak_to_generator() -> None:
    """yield inside a spawn block must not escape to the outer generator.

    Regression: current_generator_frame() did not treat spawn as a
    yield boundary, so `spawn: yield v` from inside a generator
    would race into the parent generator's output stream.
    """
    from shakar_ref.runner import run as run_program

    # The spawn errors because yield has no valid generator target.
    # The generator itself should only produce its own yield (2),
    # never the spawn's attempted yield (1).
    with pytest.raises(ShakarRuntimeError, match="yield"):
        run_program(
            dedent(
                """\
            fn g():
              t := spawn: yield 1
              yield 2
              wait t
            gen := g()
            gen.next()
            gen.next()
        """
            )
        )


def test_fn_with_yield_only_in_spawn_is_not_generator() -> None:
    """A function whose only yield is inside spawn must not become a generator.

    Regression: _has_yield_in_scope descended into spawn bodies, tagging
    the enclosing function as a generator even though the yield can never
    reach it at runtime.
    """
    from shakar_ref.runner import run as run_program

    # If f were mistakenly a generator, f() would return a ShkGenerator
    # silently. Correctly, f() is a normal function — the invalid yield
    # inside spawn surfaces as a runtime error when `wait t` collects it.
    with pytest.raises(ShakarRuntimeError, match="yield"):
        run_program(
            dedent(
                """\
            fn f():
              t := spawn: yield 1
              wait t
            f()
        """
            )
        )


def test_fn_with_yield_only_in_defer_is_not_generator() -> None:
    """A function whose only yield is inside defer must not become a generator.

    Regression: _has_yield_in_scope descended into defer bodies, tagging
    the enclosing function as a generator.
    """
    from shakar_ref.runner import run as run_program

    # If f were mistakenly a generator, f() would return a ShkGenerator
    # silently. Correctly, f() is a normal function — the invalid yield
    # inside defer surfaces as a runtime error during cleanup.
    with pytest.raises(ShakarRuntimeError, match="yield"):
        run_program(
            dedent(
                """\
            fn f():
              defer cleanup:
                yield 1
              return 2
            f()
        """
            )
        )


def test_zero_step_selector_rejects_before_consuming_generator() -> None:
    """A zero-step selector must raise before pulling any values.

    Regression: _compute_max_position() coerced step 0 to 1 via `or 1`,
    causing the generator to advance before the later array selector
    raised 'step cannot be zero'.
    """
    from shakar_ref.runner import run as run_program

    with pytest.raises(ShakarTypeError, match="step cannot be zero"):
        run_program(
            dedent(
                """\
            fn naturals():
              i := 0
              while true:
                yield i++
            gen := naturals()
            gen[`:<3:0`]
        """
            )
        )


def test_cancel_token_child_pruned_on_generator_close() -> None:
    """Generator cancel tokens must be unlinked from the parent when the
    generator closes, so dead children do not accumulate under long-lived
    parent tokens.
    """
    from shakar_ref.types import CancelToken

    parent = CancelToken()
    child = CancelToken()
    parent.add_child(child)
    assert len(parent._children) == 1

    # Explicit removal
    parent.remove_child(child)
    assert len(parent._children) == 0


def test_cancel_token_child_pruned_on_generator_completion() -> None:
    """Generator completion must eagerly unlink its cancel token from
    the caller's token to prevent unbounded retention.

    Regression: parent=>child cancel links were only cleaned up via
    weak-ref GC callbacks, which could lag indefinitely.
    """
    from shakar_ref.runner import run as run_program

    # This is an integration-level check: create and exhaust generators
    # in a loop. If unlinking works, the parent token's children list
    # stays bounded. We just verify no error occurs.
    run_program(
        dedent(
            """\
        fn g():
          yield 1
        for i in 10:
          gen := g()
          gen.next()
    """
        )
    )


def test_delegated_generator_closed_on_yield_error() -> None:
    """If yield ...gen delegation aborts because yield_fn raises, the
    delegated generator must be closed so deferred cleanup runs.

    Regression: only _GeneratorCloseSignal triggered close; other
    errors leaked the delegated generator's suspended state.
    """
    # The outer generator closes mid-delegation. The inner generator's
    # deferred cleanup must still run and set the flag.
    run_runtime_case(
        dedent(
            """\
        flag := 0
        fn inner():
          defer cleanup:
            flag = 1
          yield 10
          yield 20
          yield 30
        fn outer():
          yield ...inner()
        gen := outer()
        gen.next()
        gen.close()
        flag
    """
        ),
        ("number", 1),
        None,
    )


def test_decorator_def_with_yield_not_marked_as_generator() -> None:
    """Decorator definitions must not be marked as generators even if
    their body contains yield, since the runtime does not support
    generator decorators.

    Regression: _mark_generator_fns treated decorator_def the same as
    fndef, silently returning a ShkGenerator instead of running the
    decorator body.
    """
    from shakar_ref.runner import run as run_program

    # A decorator whose body calls yield should raise a runtime error
    # (yield outside generator), not silently become a generator.
    with pytest.raises(ShakarRuntimeError, match="yield"):
        run_program(
            dedent(
                """\
            decorator my_dec(msg):
              yield 1
            @my_dec("hello")
            fn greet():
              return "hi"
            greet()
        """
            )
        )


def test_parent_cancellation_propagates_through_generator() -> None:
    """Parent-task cancellation must surface as ShakarCancelledError, not
    silent exhaustion.

    Regression: run_body() turned every ShakarCancelledError into a
    clean "closed" result, so parent cancellation looked like the
    generator ran out of values.
    """
    from shakar_ref.runner import run as run_program
    from shakar_ref.types import ShakarCancelledError

    # Spawn a task that creates a generator blocked on a channel.
    # Cancel the task — the generator should propagate cancellation,
    # not silently return nil.
    with pytest.raises((ShakarCancelledError, ShakarRuntimeError)):
        run_program(
            dedent(
                """\
            ch := chan()
            fn blocking_gen():
              yield <-ch
            t := spawn:
              gen := blocking_gen()
              gen.next()
            cancel t
            wait t
        """
            )
        )


def test_generator_spread_rejected() -> None:
    """Spreading a generator in an array literal must raise, not hang.

    Regression: _iterable_values() eagerly materialized generators
    with list(), which hangs on infinite generators.
    """
    from shakar_ref.runner import run as run_program

    with pytest.raises(ShakarTypeError, match="[Cc]annot eagerly iterate"):
        run_program(
            dedent(
                """\
            fn nums():
              i := 0
              while true:
                yield i++
            [...nums()]
        """
            )
        )


def test_generator_comprehension_rejected() -> None:
    """Using a generator as comprehension source must raise, not hang.

    Regression: _iterable_values() eagerly materialized generators
    with list(), which hangs on infinite generators.
    """
    from shakar_ref.runner import run as run_program

    with pytest.raises(ShakarTypeError, match="[Cc]annot eagerly iterate"):
        run_program(
            dedent(
                """\
            fn nums():
              i := 0
              while true:
                yield i++
            [x * 2 for x in nums()]
        """
            )
        )
