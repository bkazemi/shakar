from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, List, Optional

from lark import Tree, Token

from ..runtime import Frame, ShkFn, ShkNull, ShkValue, ShakarRuntimeError, ShakarTypeError
from ..tree import Node, is_token, is_tree, tree_label
from .blocks import eval_indent_block, temporary_bindings
from .chains import call_value
from .common import expect_ident_token as _expect_ident_token
from .control import _build_error_payload
from .helpers import is_truthy
from .mutation import get_field_value

EvalFunc = Callable[[Node, Frame], ShkValue]

def _lookup_method(resource: ShkValue, names: List[str], frame: Frame) -> Optional[ShkValue]:
    for name in names:
        try:
            return get_field_value(resource, name, frame)
        except (ShakarRuntimeError, ShakarTypeError):
            continue
    return None

def _call_method(method: ShkValue, args: List[ShkValue], frame: Frame, eval_func: EvalFunc) -> ShkValue:
    return call_value(method, args, frame, eval_func)

def eval_using_stmt(n: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    handle_tok: Optional[Token] = None
    binder_tok: Optional[Token] = None
    expr_node: Optional[Node] = None
    body_node: Optional[Tree] = None

    for child in n.children:
        label = tree_label(child)

        if is_tree(child) and label == "using_handle":
            handle_tok = child.children[0] if child.children else None
            continue

        if is_tree(child) and label == "using_bind":
            binder_tok = child.children[0] if child.children else None
            continue

        if is_tree(child) and str(label) in {"indentblock", "inlinebody"}:
            body_node = child  # type: ignore[assignment]
            continue

        if expr_node is None:
            expr_node = child
            continue

    if expr_node is None or body_node is None or not is_tree(body_node):
        raise ShakarRuntimeError("Malformed using statement")

    bind_name = None

    if binder_tok is not None:
        bind_name = _expect_ident_token(binder_tok, "Using binder")
    elif handle_tok is not None:
        bind_name = _expect_ident_token(handle_tok, "Using handle")

    resource = eval_func(expr_node, frame)

    enter_method = _lookup_method(resource, ["using_enter", "enter"], frame)
    value = resource

    if enter_method is not None:
        value = _call_method(enter_method, [], frame, eval_func)

    exc: Optional[BaseException] = None
    err_value: Optional[ShkValue] = None
    result: ShkValue = ShkNull()

    ctx = temporary_bindings(frame, {bind_name: value}) if bind_name is not None else nullcontext()

    with ctx:
        try:
            result = eval_indent_block(body_node, frame, eval_func)
        except BaseException as e:  # noqa: BLE001
            exc = e

            if isinstance(e, ShakarRuntimeError):
                err_value = _build_error_payload(e)
        finally:
            exit_method = _lookup_method(resource, ["using_exit", "exit", "close"], frame)

            if exit_method is not None:
                def _exit_args(method: ShkValue) -> List[ShkValue]:
                    if err_value is not None:
                        return [err_value]

                    target_fn = None

                    if isinstance(method, ShkFn):
                        target_fn = method
                    if hasattr(method, "fn") and isinstance(getattr(method, "fn"), ShkFn):
                        target_fn = getattr(method, "fn")

                    if target_fn is not None and target_fn.params is not None and len(target_fn.params) == 1:
                        return [ShkNull()]
                    return []

                args = _exit_args(exit_method)

                try:
                    exit_result = _call_method(exit_method, args, frame, eval_func)
                except BaseException as exit_exc:  # noqa: BLE001
                    if exc is not None:
                        exit_exc.__context__ = exc
                    raise
                else:
                    if exc is not None and is_truthy(exit_result):
                        exc = None

    if exc is not None:
        raise exc

    return result
