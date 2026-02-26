from __future__ import annotations

from textwrap import dedent

import pytest

from shakar_ref.runner import run
from shakar_ref.runtime import ShakarRuntimeError


def test_runtime_error_string_includes_source_location() -> None:
    with pytest.raises(ShakarRuntimeError) as excinfo:
        run("missing_name")

    error = excinfo.value
    assert error.context.span is not None
    assert error.context.span.line == 1
    assert "line 1, col" in str(error)


def test_call_stack_propagates_across_function_calls() -> None:
    source = dedent(
        """\
        fn boom():
          missing_name
        fn wrapper():
          boom()
        wrapper()
        """
    )

    with pytest.raises(ShakarRuntimeError) as excinfo:
        run(source)

    stack = excinfo.value.context.call_stack
    assert stack is not None
    names = [site.name for site in stack]
    assert "boom()" in names
    assert "wrapper()" in names


def test_rethrow_preserves_original_error_context() -> None:
    source = dedent(
        """\
        fn boom():
          missing_name
        fn wrapper():
          boom() catch err:
            throw
        wrapper()
        """
    )

    with pytest.raises(ShakarRuntimeError) as excinfo:
        run(source)

    span = excinfo.value.context.span
    assert span is not None
    assert span.line == 2
    stack = excinfo.value.context.call_stack
    assert stack is not None
    assert any(site.name == "boom()" for site in stack)


def test_debug_mode_attaches_python_trace(monkeypatch) -> None:
    monkeypatch.setenv("SHAKAR_DEBUG_PY_TRACE", "1")

    with pytest.raises(ShakarRuntimeError) as excinfo:
        run("missing_name")

    assert excinfo.value.context.py_trace is not None


def test_debug_mode_enriches_pre_marked_modifier_error(monkeypatch) -> None:
    monkeypatch.setenv("SHAKAR_DEBUG_PY_TRACE", "1")
    source = dedent(
        """\
        fn run():
          wait[foo] 1
        run()
        """
    )

    with pytest.raises(ShakarRuntimeError) as excinfo:
        run(source)

    error = excinfo.value
    assert error.context.span is not None
    assert error.context.py_trace is not None
    assert error.context.call_stack is not None
    assert any(site.name == "run()" for site in error.context.call_stack)
