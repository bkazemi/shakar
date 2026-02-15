from __future__ import annotations

from typing import Callable, List

from ..runtime import Frame, ShkNil, ShkValue, ShakarRuntimeError
from ..tree import Node, Tree, is_tree, tree_label
from .bind import assign_ident, eval_assign_stmt
from .common import expect_ident_token
from .destructure import assign_pattern as destructure_assign_pattern
from .destructure import prepare_destructure_bindings

EvalFn = Callable[[Node, Frame], ShkValue]
ApplyOpFunc = Callable[[ShkValue, Tree, Frame, EvalFn], ShkValue]
IndexEvalFn = Callable[[Tree, Frame, EvalFn], ShkValue]


def define_let_ident(name: str, value: ShkValue, frame: Frame) -> ShkValue:
    if frame.name_exists(name):
        raise ShakarRuntimeError(
            f"Name '{name}' already defined in an outer scope or this block"
        )
    frame.define_let(name, value)
    return value


def eval_let_walrus(children: List[Node], frame: Frame, eval_fn: EvalFn) -> ShkValue:
    if len(children) != 2:
        raise ShakarRuntimeError("Malformed let walrus")

    name_node, value_node = children
    name = expect_ident_token(name_node, "Let walrus target")
    value = eval_fn(value_node, frame)
    return define_let_ident(name, value, frame)


def _assign_pattern_let(
    eval_fn: EvalFn,
    pattern: Tree,
    value: ShkValue,
    frame: Frame,
    *,
    create: bool,
    allow_broadcast: bool,
) -> None:
    def _assign_ident_wrapper(
        name: str, val: ShkValue, target_frame: Frame, create_flag: bool
    ) -> None:
        if create_flag:
            define_let_ident(name, val, target_frame)
            return
        assign_ident(name, val, target_frame, create=False)

    destructure_assign_pattern(
        eval_fn, _assign_ident_wrapper, pattern, value, frame, create, allow_broadcast
    )


def eval_let_destructure(
    node: Tree,
    frame: Frame,
    eval_fn: EvalFn,
    *,
    create: bool,
    allow_broadcast: bool,
) -> ShkValue:
    patterns, values, result = prepare_destructure_bindings(
        node,
        frame,
        eval_fn,
        allow_broadcast=allow_broadcast,
        malformed_message="Malformed let destructure",
        empty_message="Empty let destructure pattern",
    )

    for pat, val in zip(patterns, values):
        _assign_pattern_let(
            eval_fn,
            pat,
            val,
            frame,
            create=create,
            allow_broadcast=allow_broadcast,
        )

    return result if allow_broadcast else ShkNil()


def eval_let_stmt(
    node: Tree,
    frame: Frame,
    eval_fn: EvalFn,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFn,
) -> ShkValue:
    if not node.children:
        raise ShakarRuntimeError("Malformed let statement")

    inner = node.children[0]
    if is_tree(inner) and tree_label(inner) == "expr" and inner.children:
        inner = inner.children[0]

    label = tree_label(inner) if is_tree(inner) else None

    match label:
        case "walrus":
            return eval_let_walrus(inner.children, frame, eval_fn)
        case "destructure_walrus":
            return eval_let_destructure(
                inner, frame, eval_fn, create=True, allow_broadcast=True
            )
        case "destructure":
            return eval_let_destructure(
                inner, frame, eval_fn, create=False, allow_broadcast=False
            )
        case "assignstmt":
            return eval_assign_stmt(
                inner.children, frame, eval_fn, apply_op, evaluate_index_operand
            )
        case _:
            raise ShakarRuntimeError("let must prefix an assignment statement")
