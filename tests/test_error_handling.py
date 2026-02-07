from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    ShakarAssertionError,
    ShakarRuntimeError,
    run_runtime_case,
)

SCENARIOS = [
    pytest.param(
        dedent(
            """\
            flag := { value: 0 }
            resource := {
              enter: fn():
                flag.value = flag.value + 1
                return "ok"
              exit: fn(err):
                flag.value = flag.value + 1
            }
            using[f] resource:
              flag.value = flag.value + 1
            flag.value
        """
        ),
        ("number", 3),
        None,
        id="using-enter-exit",
    ),
    pytest.param(
        dedent(
            """\
            flag := { value: 0 }
            resource := {
              enter: fn(): flag.value = flag.value + 1
            }
            using resource:
              flag.value = flag.value + 1
            flag.value
        """
        ),
        ("number", 2),
        None,
        id="using-enter-only",
    ),
    pytest.param(
        dedent(
            """\
            resource := {
              enter: fn(): { x: 1 }
            }
            using resource bind r:
              r.x
        """
        ),
        ("number", 1),
        None,
        id="using-binder",
    ),
    pytest.param(
        dedent(
            """\
            resource := { enter: fn(): 7 }
            using[r] resource:
              r + 1
        """
        ),
        ("number", 8),
        None,
        id="using-handle-default-binder",
    ),
    pytest.param(
        dedent(
            """\
            resource := {
              using_enter: fn(): 1
              using_exit: fn(err): true
            }
            using resource:
              throw "boom"
            1
        """
        ),
        ("number", 1),
        None,
        id="using-using_exit-suppresses",
    ),
    pytest.param(
        dedent(
            """\
            resource := {
              exit: fn(err): err
            }
            using resource:
              throw "boom" 
        """
        ),
        ("null", None),
        None,
        id="using-exit-err-propagates",
    ),
    pytest.param(
        dedent(
            """\
            resource := { enter: fn(): 5 }
            using resource: resource + 1
        """
        ),
        ("number", 6),
        None,
        id="using-inlinebody",
    ),
    pytest.param(
        dedent(
            """\
            value := (missingVar catch err: err.type)
            value
        """
        ),
        ("string", "ShakarRuntimeError"),
        None,
        id="catch-expr-binder",
    ),
    pytest.param(
        dedent(
            """\
            fallback := missingVar @@: .message
            fallback
        """
        ),
        ("string", "Name 'missingVar' not found"),
        None,
        id="catch-expr-dot",
    ),
    pytest.param(
        dedent(
            """\
            msg := ""
            missingVar catch err: { msg = err.message }
            msg
        """
        ),
        ("string", "Name 'missingVar' not found"),
        None,
        id="catch-stmt-binder",
    ),
    pytest.param(
        dedent(
            """\
            typ := ""
            missingVar catch: { typ = .type }
            typ
        """
        ),
        ("string", "ShakarRuntimeError"),
        None,
        id="catch-stmt-dot",
    ),
    pytest.param(
        dedent(
            """\
            value := (missingVar catch (ShakarRuntimeError) bind err: err.type)
            value
        """
        ),
        ("string", "ShakarRuntimeError"),
        None,
        id="catch-type-match",
    ),
    pytest.param(
        dedent(
            """\
            fn risky(tag): { throw error(tag, "bad") }
            value := (risky("ValidationError") catch (ValidationError, Missing) bind err: err.type)
            value
        """
        ),
        ("string", "ValidationError"),
        None,
        id="catch-typed-bind",
    ),
    pytest.param(
        dedent(
            """\
            flag := 0
            fn setFlag(): { flag = 2 }
            fn run(): { defer cleanup: setFlag() ; flag = 1 }
            run()
            flag
        """
        ),
        ("number", 2),
        None,
        id="defer-runs",
    ),
    pytest.param(
        dedent(
            """\
            log := ""
            fn run(): { defer second after first: { log = log + "2" }; defer first: { log = log + "1" } }
            run()
            log
        """
        ),
        ("string", "12"),
        None,
        id="defer-after-order",
    ),
    pytest.param(
        dedent(
            """\
            log := ""
            fn push(ch): { log = log + ch }
            fn run(): { defer cleanup: { push("1") }; defer push("2") after cleanup }
            run()
            log
        """
        ),
        ("string", "12"),
        None,
        id="defer-simplecall-after",
    ),
    pytest.param(
        "defer cleanup: pass",
        None,
        ShakarRuntimeError,
        id="defer-unknown-handle",
    ),
    pytest.param(
        dedent(
            """\
            log := []
            fn run(): { defer second after first: pass; defer first after second: pass }
            run()
        """
        ),
        None,
        ShakarRuntimeError,
        id="defer-cycle-detected",
    ),
    pytest.param(
        dedent(
            """\
            fn run(): { defer tag: pass; defer tag: pass }
            run()
        """
        ),
        None,
        ShakarRuntimeError,
        id="defer-duplicate-handle",
    ),
    pytest.param(
        "assert 1 == 1",
        ("null", None),
        None,
        id="assert-pass",
    ),
    pytest.param(
        'assert false, "boom"',
        None,
        ShakarAssertionError,
        id="assert-fail",
    ),
    pytest.param(
        'throw error("boom")',
        None,
        ShakarRuntimeError,
        id="throw-new-error",
    ),
    pytest.param(
        dedent(
            """\
            fn risky(): { throw error("ValidationError", "bad", 123) }
            value := (risky() catch err: err.message)
            value
        """
        ),
        ("string", "bad"),
        None,
        id="throw-custom-catch",
    ),
    pytest.param(
        dedent(
            """\
            fn risky(): { throw error("TypeError", "bad") }
            risky() catch (ValidationError): 1
        """
        ),
        None,
        ShakarRuntimeError,
        id="throw-custom-guard-miss",
    ),
    pytest.param(
        dedent(
            """\
            fn risky(): { throw error("TypeError", "bad") }
            risky() catch err: throw
        """
        ),
        None,
        ShakarRuntimeError,
        id="throw-rethrow",
    ),
    pytest.param(
        dedent(
            """\
            fn risky(): { throw "original_error" }
            fn wrapper(): {
                risky() catch:
                    throw
            }
            wrapper() catch err: err.message
        """
        ),
        ("string", "original_error"),
        None,
        id="throw-rethrow-verify",
    ),
    pytest.param(
        dedent(
            """\
            fn risky(): { throw "boom" }
            fn wrapper():
                risky() catch: if true: throw else: 0
            wrapper() catch err: err.message
        """
        ),
        ("string", "boom"),
        None,
        id="throw-bare-inline-else",
    ),
    pytest.param(
        dedent(
            """\
            fn risky(): { throw "boom" }
            risky() catch err: throw if err.message == "boom"
            
        """
        ),
        None,
        ShakarRuntimeError,
        id="throw-bare-postfix-if",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_error_handling(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)
