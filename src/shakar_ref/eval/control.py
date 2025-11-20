from __future__ import annotations

from typing import Any, Callable

from ..runtime import (
    Frame,
    ShkBool,
    ShkNull,
    ShkObject,
    ShkString,
    ShakarAssertionError,
    ShakarKeyError,
    ShakarMethodNotFound,
    ShakarReturnSignal,
    ShakarRuntimeError,
    ShakarTypeError,
)
from ..tree import is_token_node, is_tree_node, tree_children, tree_label
from .blocks import eval_inline_body, eval_indent_block, temporary_bindings, temporary_subject
from .common import expect_ident_token, node_source_span as _node_source_span, render_expr as _render_expr, stringify as _stringify
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

def _build_error_payload(exc: ShakarRuntimeError) -> ShkObject:
    """Expose exception metadata to catch handlers as a lightweight object."""
    payload = getattr(exc, 'shk_payload', None)

    if isinstance(payload, ShkObject):
        return payload

    slots = {
        "message": ShkString(str(exc)),
        "type": ShkString(getattr(exc, 'shk_type', type(exc).__name__)),
    }

    if isinstance(exc, ShakarKeyError):
        slots["key"] = ShkString(str(exc.key))

    if isinstance(exc, ShakarMethodNotFound):
        slots["method"] = ShkString(exc.name)

    data = getattr(exc, 'shk_data', None)
    if data is not None:
        slots["data"] = data

    return ShkObject(slots)

def _parse_catch_components(children: list[Any]) -> tuple[Any, Any | None, list[str], Any]:
    """Split canonical catch nodes into try expression, binder token, type list, and handler."""
    if not children:
        raise ShakarRuntimeError("Malformed catch node")

    idx = 0
    try_node = children[idx]
    idx += 1
    binder = None
    type_names: list[str] = []

    if idx < len(children) and is_token_node(children[idx]):
        binder = children[idx]
        idx += 1

    if idx < len(children) and is_tree_node(children[idx]) and tree_label(children[idx]) == 'catchtypes':
        type_names = [
            expect_ident_token(tok, "Catch type")
            for tok in tree_children(children[idx]) if is_token_node(tok)
        ]
        idx += 1

    if idx >= len(children):
        raise ShakarRuntimeError("Missing catch handler")

    handler = children[idx]

    return try_node, binder, type_names, handler

def _run_catch_handler(
    handler: Any,
    frame: Frame,
    binder: Any | None,
    payload: ShkObject,
    original_exc: ShakarRuntimeError,
    allowed_types: list[str],
    eval_func: EvalFunc,
) -> Any:
    # type matching happens before we evaluate the handler body so unmatched errors bubble up
    payload_type = None

    if isinstance(payload, ShkObject):
        slot = payload.slots.get('type') if hasattr(payload, 'slots') else None

        if isinstance(slot, ShkString):
            payload_type = slot.value
        elif isinstance(slot, str):
            payload_type = slot

    if allowed_types:
        if not isinstance(payload_type, str) or payload_type not in allowed_types:
            raise original_exc

    binder_name = None
    if binder is not None:
        binder_name = expect_ident_token(binder, "Catch binder")

    def _exec_handler() -> Any:
        label = tree_label(handler)

        if label == 'inlinebody':
            return eval_inline_body(handler, frame, eval_func)

        if label == 'indentblock':
            return eval_indent_block(handler, frame, eval_func)

        return eval_func(handler, frame)

    # track the currently handled exception so a bare `throw` can rethrow it later
    prev_error = getattr(frame, '_active_error', None)
    frame._active_error = original_exc

    try:
        with temporary_subject(frame, payload):
            if binder_name:
                with temporary_bindings(frame, {binder_name: payload}):
                    return _exec_handler()
            return _exec_handler()
    finally:
        frame._active_error = prev_error

def eval_catch_expr(children: list[Any], frame: Frame, eval_func: EvalFunc) -> Any:
    try_node, binder, type_names, handler = _parse_catch_components(children)

    try:
        return eval_func(try_node, frame)
    except ShakarRuntimeError as exc:
        payload = _build_error_payload(exc)
        return _run_catch_handler(handler, frame, binder, payload, exc, type_names, eval_func)

def eval_catch_stmt(children: list[Any], frame: Frame, eval_func: EvalFunc) -> Any:
    try_node, binder, type_names, body = _parse_catch_components(children)

    try:
        eval_func(try_node, frame)
        return ShkNull()
    except ShakarRuntimeError as exc:
        payload = _build_error_payload(exc)
        _run_catch_handler(body, frame, binder, payload, exc, type_names, eval_func)
        return ShkNull()

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
