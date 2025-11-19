from __future__ import annotations

from typing import Any, Callable

from ..runtime import (
    Frame,
    ShkBool,
    ShkNull,
    ShkObject,
    ShkString,
    ShakarAssertionError,
    ShakarReturnSignal,
    ShakarRuntimeError,
    ShakarTypeError,
)
from .common import node_source_span as _node_source_span, render_expr as _render_expr, stringify as _stringify
from .helpers import current_function_frame as _current_function_frame, is_truthy as _is_truthy


EvalFunc = Callable[[Any, Frame], Any]


def eval_return_stmt(children: list[Any], frame: Frame, eval_func: EvalFunc) -> Any:
    if _current_function_frame(frame) is None:
        raise ShakarRuntimeError("return outside of a function")

    value = eval_func(children[0], frame) if children else ShkNull()

    raise ShakarReturnSignal(value)


def eval_return_if(children: list[Any], frame: Frame, eval_func: EvalFunc) -> Any:
    if _current_function_frame(frame) is None:
        raise ShakarRuntimeError("?ret outside of a function")

    if not children:
        raise ShakarRuntimeError("?ret requires an expression")

    value = eval_func(children[0], frame)
    if _is_truthy(value):
        raise ShakarReturnSignal(value)

    return ShkNull()


def eval_throw_stmt(children: list[Any], frame: Frame, eval_func: EvalFunc) -> Any:
    if children:
        value = eval_func(children[0], frame)
        raise coerce_throw_value(value)
    current = getattr(frame, '_active_error', None)
    if current is None:
        raise ShakarRuntimeError("throw outside of catch")
    raise current


def eval_break_stmt(frame: Frame) -> Any:
    raise ShakarReturnSignal  # pragma: no cover - placeholder to satisfy type checkers


def eval_continue_stmt(frame: Frame) -> Any:
    raise ShakarReturnSignal  # pragma: no cover - placeholder to satisfy type checkers


def coerce_throw_value(value: Any) -> ShakarRuntimeError:
    if isinstance(value, ShakarRuntimeError):
        return value

    if isinstance(value, ShkObject):
        slots = getattr(value, 'slots', {})
        marker = slots.get('__error__')

        if isinstance(marker, ShkBool) and marker.value:
            type_slot = slots.get('type')
            msg_slot = slots.get('message')

            if not isinstance(type_slot, ShkString) or not isinstance(msg_slot, ShkString):
                raise ShakarTypeError("error() objects must have string type and message")

            err = ShakarRuntimeError(msg_slot.value)
            err.shk_type = type_slot.value
            err.shk_data = slots.get('data', ShkNull())
            err.shk_payload = value
            return err

    message = _stringify(value)

    err = ShakarRuntimeError(message)
    err.shk_type = err.__class__.__name__
    err.shk_payload = value
    return err


def eval_assert(children: list[Any], frame: Frame, eval_func: EvalFunc) -> Any:
    if not children:
        raise ShakarRuntimeError("Malformed assert statement")

    cond_val = eval_func(children[0], frame)

    if _is_truthy(cond_val):
        return ShkNull()

    message = f"Assertion failed: {_assert_source_snippet(children[0], frame)}"

    if len(children) > 1:
        msg_val = eval_func(children[1], frame)
        message = _stringify(msg_val)

    raise ShakarAssertionError(message)


def _assert_source_snippet(node: Any, frame: Frame) -> str:
    src = getattr(frame, 'source', None)

    if src is not None:
        start, end = _node_source_span(node)

        if start is not None and end is not None and 0 <= start < end <= len(src):
            snippet = src[start:end].strip()

            if snippet:
                return snippet

    rendered = _render_expr(node)

    return rendered if rendered else "<expr>"
