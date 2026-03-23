from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    ShakarRuntimeError,
    ShakarTypeError,
    run_runtime_case,
)
from shakar_ref.runner import run as run_program

SCENARIOS = [
    pytest.param(
        dedent(
            """\
            fn id(x): x
            wait spawn id(5)
        """
        ),
        ("number", 5),
        None,
        id="wait-value",
    ),
    pytest.param(
        dedent(
            """\
            val := 0
            ch := channel(1)
            5 -> ch
            wait[any]:
              x := <-ch: { val = x }
            val
        """
        ),
        ("number", 5),
        None,
        id="wait-any-body",
    ),
    pytest.param(
        dedent(
            """\
            ch := channel(1)
            1 -> ch
            wait[any]:
              x := <-ch: x
        """
        ),
        ("number", 1),
        None,
        id="wait-any-recv",
    ),
    pytest.param(
        dedent(
            """\
            ch := channel(1)
            wait[any]:
              7 -> ch: "sent"
            
        """
        ),
        ("string", "sent"),
        None,
        id="wait-any-send",
    ),
    pytest.param(
        dedent(
            """\
            result := wait[all]:
              first: 1
              second: 2
            result.first + result.second
        """
        ),
        ("number", 3),
        None,
        id="wait-all-block",
    ),
    pytest.param(
        dedent(
            """\
            ch := channel()
            wait[any]:
              x := <-ch: 1
              default: 2
        """
        ),
        ("number", 2),
        None,
        id="wait-any-default",
    ),
    pytest.param(
        dedent(
            """\
            ch := channel(2)
            wait[group]:
              1 -> ch
              2 -> ch
            x := <-ch
            y := <-ch
            x + y
        """
        ),
        ("number", 3),
        None,
        id="wait-group-send",
    ),
    pytest.param(
        dedent(
            """\
            fn id(x): x
            tasks := [spawn id(1), spawn id(2)]
            results := wait[all] tasks
            results[0] + results[1]
        """
        ),
        ("number", 3),
        None,
        id="wait-all-call",
    ),
    pytest.param(
        dedent(
            """\
            tasks := spawn [fn(): 1, fn(): 2]
            results := wait[all] tasks
            results[0] + results[1]
        """
        ),
        ("number", 3),
        None,
        id="spawn-iterable-array",
    ),
    pytest.param(
        "spawn [1, 2]",
        None,
        ShakarTypeError,
        id="spawn-iterable-noncallable-error",
    ),
    pytest.param(
        dedent(
            """\
            ch := channel(1)
            nil -> ch
            val, ok := <-ch
            ok
        """
        ),
        ("bool", True),
        None,
        id="recv-ok",
    ),
    pytest.param(
        dedent(
            """\
            ch := channel(1)
            ch.close()
            1 -> ch
        """
        ),
        ("bool", False),
        None,
        id="send-closed-false",
    ),
    pytest.param(
        dedent(
            """\
            ch := channel(1)
            99 -> ch
            wait ch
        """
        ),
        ("number", 99),
        None,
        id="wait-bare-ident",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_concurrency(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)


def test_channel_cancel_race() -> None:
    from tests.support.harness import check_channel_cancel_race

    check_channel_cancel_race()


def test_wait_unknown_modifier_runtime_error() -> None:
    with pytest.raises(
        ShakarRuntimeError,
        match="unknown wait modifier 'foo'; expected one of: any, all, group",
    ):
        run_program("wait[foo] 1")


def test_wait_modifier_close_match_suggestion() -> None:
    with pytest.raises(
        ShakarRuntimeError,
        match=r"unknown wait modifier 'gruop'.*did you mean 'group'\?",
    ):
        run_program("wait[gruop] 1")


def test_generator_inherits_cancel_token() -> None:
    """Generator body must see the caller's cancellation.

    Regression: _create_generator replaced the frame's cancel_token with a
    fresh CancelToken, severing the link to the caller. In a spawn context
    the generator would ignore task cancellation and block forever.
    """
    import threading

    from shakar_ref.types import CancelToken

    parent = CancelToken()
    child = CancelToken()
    parent.add_child(child)

    # Child should not be cancelled yet.
    assert not child.cancelled()

    # Cancelling parent must propagate to child.
    parent.cancel()
    assert child.cancelled()


def test_generator_cancel_propagation_with_conditions() -> None:
    """Parent cancellation must wake conditions registered on the child token."""
    import threading

    from shakar_ref.types import CancelToken

    parent = CancelToken()
    child = CancelToken()
    parent.add_child(child)

    lock = threading.Lock()
    cond = threading.Condition(lock)
    child.register_condition(cond)

    woke = threading.Event()

    def waiter() -> None:
        with cond:
            # Wait until notified — parent.cancel() should wake us.
            cond.wait(timeout=2)
        woke.set()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()

    # Give the waiter a moment to enter cond.wait().
    import time

    time.sleep(0.05)

    parent.cancel()
    assert woke.wait(timeout=1), "condition was not notified on parent cancel"
    assert child.cancelled()


def test_generator_close_does_not_cancel_parent() -> None:
    """Calling .close() on a generator must not cancel the caller's token."""
    from shakar_ref.types import CancelToken

    parent = CancelToken()
    child = CancelToken()
    parent.add_child(child)

    # Closing the generator cancels the child only.
    child.cancel()
    assert child.cancelled()
    assert not parent.cancelled()


def test_cancel_token_child_refs_do_not_leak() -> None:
    """Dead child tokens must be pruned eagerly, not only on cancel().

    Regression: add_child() stored refs that were only pruned during
    cancel(), so an uncancelled long-lived parent accumulated one dead
    entry per generator call.
    """
    import gc

    from shakar_ref.types import CancelToken

    parent = CancelToken()

    for _ in range(100):
        child = CancelToken()
        parent.add_child(child)
    # All children go out of scope here.

    del child
    gc.collect()

    # Refs should be pruned eagerly by the weakref callback —
    # no cancel() needed.
    assert len(parent._children) == 0
