from __future__ import annotations

from typing import Callable, Optional
from ..tree import Tok

from ..runtime import (
    Frame,
    ShkBool,
    ShkNil,
    ShkObject,
    ShkSelector,
    ShkString,
    ShkValue,
    ShakarBreakSignal,
    ShakarContinueSignal,
    ShakarAssertionError,
    ShakarKeyError,
    ShakarMethodNotFound,
    ShakarMatchError,
    ShakarReturnSignal,
    ShakarRuntimeError,
    ShakarTypeError,
)
from ..tree import Node, Tree, is_token, is_tree, tree_children, tree_label
from .blocks import (
    eval_body_node,
    eval_inline_body,
    eval_indent_block,
    temporary_bindings,
    temporary_subject,
)
from .common import (
    expect_ident_token,
    node_source_span as _node_source_span,
    render_expr as _render_expr,
    stringify as _stringify,
)
from .common import token_kind as _token_kind
from .helpers import (
    current_function_frame as _current_function_frame,
    eval_anchor_scoped,
    is_truthy as _is_truthy,
)
from .expr import _compare_values
from .selector import selector_contains
from ..utils import debug_py_trace_enabled, shk_equals

EvalFunc = Callable[[Node, Frame], ShkValue]


def eval_return_stmt(
    children: list[Node], frame: Frame, eval_func: EvalFunc
) -> ShkValue:
    if _current_function_frame(frame) is None:
        raise ShakarRuntimeError("return outside of a function")

    value = eval_func(children[0], frame) if children else ShkNil()

    raise ShakarReturnSignal(value)


def eval_return_if(children: list[Node], frame: Frame, eval_func: EvalFunc) -> ShkValue:
    if _current_function_frame(frame) is None:
        raise ShakarRuntimeError("?ret outside of a function")

    if not children:
        raise ShakarRuntimeError("?ret requires an expression")

    value = eval_func(children[0], frame)
    if _is_truthy(value):
        raise ShakarReturnSignal(value)

    return ShkNil()


def eval_throw_stmt(
    children: list[Node], frame: Frame, eval_func: EvalFunc
) -> ShkValue:
    if children:
        value = eval_func(children[0], frame)
        raise coerce_throw_value(value)
    current = getattr(frame, "_active_error", None)
    if current is None:
        raise ShakarRuntimeError("throw outside of catch")
    raise current


def eval_break_stmt(frame: Frame) -> ShkValue:
    raise ShakarBreakSignal()


def eval_continue_stmt(frame: Frame) -> ShkValue:
    raise ShakarContinueSignal()


def coerce_throw_value(value: ShkValue) -> ShakarRuntimeError:
    if isinstance(value, ShakarRuntimeError):
        return value

    if isinstance(value, ShkObject):
        slots = getattr(value, "slots", {})
        marker = slots.get("__error__")

        if isinstance(marker, ShkBool) and marker.value:
            type_slot = slots.get("type")
            msg_slot = slots.get("message")

            if not isinstance(type_slot, ShkString) or not isinstance(
                msg_slot, ShkString
            ):
                raise ShakarTypeError(
                    "error() objects must have string type and message"
                )

            err = ShakarRuntimeError(msg_slot.value)
            err.shk_type = type_slot.value
            err.shk_data = slots.get("data", ShkNil())
            err.shk_payload = value
            return err

    message = _stringify(value)

    err = ShakarRuntimeError(message)
    err.shk_type = err.__class__.__name__
    err.shk_payload = value
    return err


def eval_assert(children: list[Node], frame: Frame, eval_func: EvalFunc) -> ShkValue:
    if not children:
        raise ShakarRuntimeError("Malformed assert statement")

    cond_val = eval_func(children[0], frame)

    if _is_truthy(cond_val):
        return ShkNil()

    message = f"Assertion failed: {_assert_source_snippet(children[0], frame)}"

    if len(children) > 1:
        msg_val = eval_func(children[1], frame)
        message = _stringify(msg_val)

    raise ShakarAssertionError(message)


def _build_error_payload(exc: ShakarRuntimeError) -> ShkObject:
    """Expose exception metadata to catch handlers as a lightweight object."""
    payload = getattr(exc, "shk_payload", None)

    if isinstance(payload, ShkObject):
        return payload

    type_hint = getattr(exc, "shk_type", None) or type(exc).__name__

    slots: dict[str, ShkValue] = {
        "message": ShkString(str(exc)),
        "type": ShkString(str(type_hint)),
    }

    if isinstance(exc, ShakarKeyError):
        slots["key"] = ShkString(str(exc.key))

    if isinstance(exc, ShakarMethodNotFound):
        slots["method"] = ShkString(exc.name)

    data = getattr(exc, "shk_data", None)
    if data is not None:
        slots["data"] = data

    if debug_py_trace_enabled():
        import traceback

        tb = getattr(exc, "shk_py_trace", None)
        if tb is not None:
            slots["py_trace"] = ShkString("".join(traceback.format_tb(tb)))

    return ShkObject(slots)


def _parse_catch_components(
    children: list[Node],
) -> tuple[Node, Optional[Tok], list[str], Tree]:
    """Split canonical catch nodes into try expression, binder token, type list, and handler."""
    if not children:
        raise ShakarRuntimeError("Malformed catch node")

    idx = 0
    try_node = children[idx]
    idx += 1
    binder = None
    type_names: list[str] = []

    if idx < len(children) and is_token(children[idx]):
        binder = children[idx]
        idx += 1

    if (
        idx < len(children)
        and is_tree(children[idx])
        and tree_label(children[idx]) == "catchtypes"
    ):
        type_names = [
            expect_ident_token(tok, "Catch type")
            for tok in tree_children(children[idx])
            if is_token(tok)
        ]
        idx += 1

    if idx >= len(children):
        raise ShakarRuntimeError("Missing catch handler")

    handler = children[idx]

    return try_node, binder, type_names, handler


def _run_catch_handler(
    handler: Tree,
    frame: Frame,
    binder: Optional[Tok],
    payload: ShkObject,
    original_exc: ShakarRuntimeError,
    allowed_types: list[str],
    eval_func: EvalFunc,
) -> ShkValue:
    # type matching happens before we evaluate the handler body so unmatched errors bubble up
    payload_type = None

    if isinstance(payload, ShkObject):
        slot = payload.slots.get("type") if hasattr(payload, "slots") else None

        if isinstance(slot, ShkString):
            payload_type = slot.value

    if allowed_types:
        if not isinstance(payload_type, str) or payload_type not in allowed_types:
            raise original_exc

    binder_name = None
    if binder is not None:
        binder_name = expect_ident_token(binder, "Catch binder")

    def _exec_handler() -> ShkValue:
        label = tree_label(handler)

        if label == "inlinebody":
            return eval_inline_body(handler, frame, eval_func)

        if label == "indentblock":
            return eval_indent_block(handler, frame, eval_func)

        return eval_func(handler, frame)

    # track the currently handled exception so a bare `throw` can rethrow it later
    prev_error = getattr(frame, "_active_error", None)
    frame._active_error = original_exc

    try:
        with temporary_subject(frame, payload):
            if binder_name:
                with temporary_bindings(frame, {binder_name: payload}):
                    return _exec_handler()
            return _exec_handler()
    finally:
        frame._active_error = prev_error


def eval_catch_expr(
    children: list[Node], frame: Frame, eval_func: EvalFunc
) -> ShkValue:
    try_node, binder, type_names, handler = _parse_catch_components(children)

    try:
        return eval_func(try_node, frame)
    except ShakarRuntimeError as exc:
        payload = _build_error_payload(exc)
        return _run_catch_handler(
            handler, frame, binder, payload, exc, type_names, eval_func
        )


def eval_catch_stmt(
    children: list[Node], frame: Frame, eval_func: EvalFunc
) -> ShkValue:
    try_node, binder, type_names, body = _parse_catch_components(children)

    try:
        eval_func(try_node, frame)
        return ShkNil()
    except ShakarRuntimeError as exc:
        payload = _build_error_payload(exc)
        _run_catch_handler(body, frame, binder, payload, exc, type_names, eval_func)
        return ShkNil()


def _assert_source_snippet(node: Node, frame: Frame) -> str:
    src: Optional[str] = getattr(frame, "source", None)

    if src is not None:
        start, end = _node_source_span(node)

        if start is not None and end is not None and 0 <= start < end <= len(src):
            snippet = src[start:end].strip()

            if snippet:
                return snippet

    rendered: str = _render_expr(node)

    return rendered if rendered else "<expr>"


def _extract_clause(node: Tree, label: str) -> tuple[Optional[Node], Node]:
    nodes = [
        child
        for child in tree_children(node)
        if not is_token(child) or _token_kind(child) not in {"ELIF", "ELSE", "COLON"}
    ]

    if label == "else":
        if not nodes:
            raise ShakarRuntimeError("Malformed else clause")
        return None, nodes[0]

    cond_node = nodes[0] if nodes else None
    body_node = nodes[1] if len(nodes) > 1 else None

    if cond_node is None:
        raise ShakarRuntimeError("Malformed elif clause")

    if body_node is None:
        raise ShakarRuntimeError("Malformed clause body")

    return cond_node, body_node


def eval_if_stmt(n: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    children = tree_children(n)
    cond_node = None
    body_node = None
    elif_clauses: list[tuple[Node | ShkValue, Node | ShkValue]] = []
    else_body = None

    for child in children:
        if is_token(child):
            if cond_node is None and _token_kind(child) not in {"IF", "COLON"}:
                cond_node = child
            continue

        label = tree_label(child)
        if label == "elifclause":
            clause_cond, clause_body = _extract_clause(child, label="elif")
            elif_clauses.append((clause_cond, clause_body))
            continue
        if label == "elseclause":
            _, else_body = _extract_clause(child, label="else")
            continue

        if cond_node is None:
            cond_node = child
        elif body_node is None:
            body_node = child

    if cond_node is None or body_node is None:
        raise ShakarRuntimeError("Malformed if statement")

    if _is_truthy(eval_func(cond_node, frame)):
        return eval_body_node(body_node, frame, eval_func, allow_loop_control=True)

    for clause_cond, clause_body in elif_clauses:
        if _is_truthy(eval_func(clause_cond, frame)):
            return eval_body_node(
                clause_body, frame, eval_func, allow_loop_control=True
            )

    if else_body is not None:
        return eval_body_node(else_body, frame, eval_func, allow_loop_control=True)

    return ShkNil()


def eval_match_expr(n: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    children = tree_children(n)
    if not children:
        raise ShakarRuntimeError("Malformed match expression")

    idx = 0
    cmp_op = "=="
    if is_tree(children[0]) and tree_label(children[0]) == "matchcmp":
        # matchcmp is an optional first child; default comparator is ==.
        cmp_op = _normalize_match_cmp(children[0])
        idx = 1

    if idx >= len(children):
        raise ShakarRuntimeError("Malformed match expression")

    subject_node = children[idx]
    subject = eval_func(subject_node, frame)
    else_body: Optional[Node] = None

    for child in children[idx + 1 :]:
        label = tree_label(child)
        if label == "matcharm":
            if not is_tree(child) or len(child.children) != 2:
                raise ShakarRuntimeError("Malformed match arm")
            patterns_node, body_node = child.children
            patterns = (
                tree_children(patterns_node)
                if is_tree(patterns_node)
                and tree_label(patterns_node) == "matchpatterns"
                else [patterns_node]
            )
            for pattern in patterns:
                if _match_pattern(pattern, subject, cmp_op, frame, eval_func):
                    with temporary_subject(frame, subject):
                        return eval_body_node(
                            body_node,
                            frame,
                            eval_func,
                            allow_loop_control=True,
                        )
            continue
        if label == "matchelse":
            if not is_tree(child) or not child.children:
                raise ShakarRuntimeError("Malformed match else")
            else_body = child.children[0]
            continue
        if is_token(child):
            continue

    if else_body is not None:
        with temporary_subject(frame, subject):
            return eval_body_node(else_body, frame, eval_func, allow_loop_control=True)

    raise ShakarMatchError("No match found")


def _match_pattern(
    pattern: Node,
    subject: ShkValue,
    cmp_op: str,
    frame: Frame,
    eval_func: EvalFunc,
) -> bool:
    value = eval_anchor_scoped(pattern, frame, eval_func)

    match cmp_op:
        case "==":
            if isinstance(value, ShkSelector):
                return _selector_matches(subject, value)
            return shk_equals(subject, value)
        case "!=":
            if isinstance(value, ShkSelector):
                return not _selector_matches(subject, value)
            return not shk_equals(subject, value)
        case "<" | "<=" | ">" | ">=":
            # Ordered comparators: pattern is LHS, subject is RHS.
            return _compare_values(cmp_op, value, subject)
        case "in":
            if isinstance(value, ShkSelector):
                return _selector_matches(subject, value)
            return _compare_values("in", subject, value)
        case "!in" | "not in":
            if isinstance(value, ShkSelector):
                return not _selector_matches(subject, value)
            return _compare_values(cmp_op, subject, value)
        case "~~":
            return _compare_values("~~", subject, value)
        case _:
            raise ShakarRuntimeError(f"Unknown match comparator '{cmp_op}'")


def _normalize_match_cmp(node: Tree) -> str:
    if not node.children:
        raise ShakarRuntimeError("Malformed match comparator")
    if len(node.children) == 2:
        first, second = node.children
        # Two-token comparator forms: "! in" or "not in".
        if not (is_token(first) and is_token(second)):
            raise ShakarRuntimeError("Malformed match comparator")
        if _token_kind(second) != "IN":
            raise ShakarRuntimeError("Malformed match comparator")
        if _token_kind(first) == "NEG":
            return "!in"
        if _token_kind(first) == "NOT":
            return "not in"
        raise ShakarRuntimeError(f"Unknown match comparator '{first.value}'")

    tok = node.children[0]
    if not is_token(tok):
        raise ShakarRuntimeError("Malformed match comparator")

    kind = _token_kind(tok)
    if kind == "IDENT":
        value = str(tok.value).lower()
        ident_map = {
            "eq": "==",
            "ne": "!=",
            "lt": "<",
            "le": "<=",
            "gt": ">",
            "ge": ">=",
        }
        if value in ident_map:
            return ident_map[value]
        raise ShakarRuntimeError(f"Unknown match comparator '{tok.value}'")

    token_map = {
        "EQ": "==",
        "NEQ": "!=",
        "LT": "<",
        "LTE": "<=",
        "GT": ">",
        "GTE": ">=",
        "IN": "in",
        "REGEXMATCH": "~~",
    }
    if kind in token_map:
        return token_map[kind]
    raise ShakarRuntimeError(f"Unknown match comparator '{tok.value}'")


def _selector_matches(subject: ShkValue, selector: ShkSelector) -> bool:
    return selector_contains(selector, subject)
