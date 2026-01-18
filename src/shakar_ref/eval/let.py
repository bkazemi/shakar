from __future__ import annotations

from typing import Callable, List

from ..runtime import Frame, ShkNull, ShkValue, ShakarRuntimeError
from ..tree import Node, Tree, is_tree, tree_children, tree_label
from .bind import assign_ident, eval_assign_stmt
from .common import expect_ident_token
from .destructure import assign_pattern as destructure_assign_pattern
from .destructure import evaluate_destructure_rhs

EvalFunc = Callable[[Node, Frame], ShkValue]
ApplyOpFunc = Callable[[ShkValue, Tree, Frame, EvalFunc], ShkValue]
IndexEvalFunc = Callable[[Tree, Frame, EvalFunc], ShkValue]


def define_let_ident(name: str, value: ShkValue, frame: Frame) -> ShkValue:
    if frame.name_exists(name):
        raise ShakarRuntimeError(
            f"Name '{name}' already defined in an outer scope or this block"
        )
    frame.define_let(name, value)
    return value


def eval_let_walrus(
    children: List[Node], frame: Frame, eval_func: EvalFunc
) -> ShkValue:
    if len(children) != 2:
        raise ShakarRuntimeError("Malformed let walrus")

    name_node, value_node = children
    name = expect_ident_token(name_node, "Let walrus target")
    value = eval_func(value_node, frame)
    return define_let_ident(name, value, frame)


def _assign_pattern_let(
    eval_func: EvalFunc,
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
        eval_func, _assign_ident_wrapper, pattern, value, frame, create, allow_broadcast
    )


def eval_let_destructure(
    node: Tree,
    frame: Frame,
    eval_func: EvalFunc,
    *,
    create: bool,
    allow_broadcast: bool,
) -> ShkValue:
    if len(node.children) != 2:
        raise ShakarRuntimeError("Malformed let destructure")

    pattern_list, rhs_node = node.children
    patterns = [c for c in tree_children(pattern_list) if tree_label(c) == "pattern"]

    if not patterns:
        raise ShakarRuntimeError("Empty let destructure pattern")

    values, result = evaluate_destructure_rhs(
        eval_func, rhs_node, frame, len(patterns), allow_broadcast
    )

    for pat, val in zip(patterns, values):
        _assign_pattern_let(
            eval_func,
            pat,
            val,
            frame,
            create=create,
            allow_broadcast=allow_broadcast,
        )

    return result if allow_broadcast else ShkNull()


def eval_let_stmt(
    node: Tree,
    frame: Frame,
    eval_func: EvalFunc,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFunc,
) -> ShkValue:
    if not node.children:
        raise ShakarRuntimeError("Malformed let statement")

    inner = node.children[0]
    if is_tree(inner) and tree_label(inner) == "expr" and inner.children:
        inner = inner.children[0]

    label = tree_label(inner) if is_tree(inner) else None

    match label:
        case "walrus":
            return eval_let_walrus(inner.children, frame, eval_func)
        case "destructure_walrus":
            return eval_let_destructure(
                inner, frame, eval_func, create=True, allow_broadcast=True
            )
        case "destructure":
            return eval_let_destructure(
                inner, frame, eval_func, create=False, allow_broadcast=False
            )
        case "assignstmt":
            return eval_assign_stmt(
                inner.children, frame, eval_func, apply_op, evaluate_index_operand
            )
        case _:
            raise ShakarRuntimeError("let must prefix an assignment statement")
