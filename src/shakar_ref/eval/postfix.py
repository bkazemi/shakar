from __future__ import annotations

from typing import Callable, Iterable, List, Sequence

from ..runtime import Frame, ShkNull, ShkValue, ShakarRuntimeError
from ..tree import Node, Tree, Token, find_tree_by_label, is_token, tree_children

EvalFunc = Callable[[Node, Frame], ShkValue]
TruthyFunc = Callable[[ShkValue], bool]

def define_new_ident(name: str, value: ShkValue, frame: Frame) -> ShkValue:
    """Introduce a new binding in the current scope; error if it already exists."""
    vars_dict = getattr(frame, "vars", None)

    if vars_dict is not None and name in vars_dict:
        raise ShakarRuntimeError(f"Name '{name}' already defined in this scope")

    frame.define(name, value)

    return value

def _walrus_target_name(node: Node) -> str:
    children = tree_children(node)
    if not children:
        raise ShakarRuntimeError("Malformed walrus expression")

    target = children[0]

    if is_token(target):
        return str(target.value)

    if isinstance(target, str):
        return target

    raise ShakarRuntimeError("Unsupported walrus target")

def _split_postfix_children(children: Sequence[Node], keyword_tokens: Iterable[str]) -> tuple[Node, Node]:
    keywords = set(keyword_tokens)
    semantic: List[Node] = []

    for ch in children:
        if is_token(ch) and ch.type in keywords:
            continue

        semantic.append(ch)

    if len(semantic) != 2:
        raise ShakarRuntimeError("Malformed postfix statement")

    return semantic[0], semantic[1]

def eval_postfix_if(
    children: Sequence[Node],
    frame: Frame,
    *,
    eval_func: EvalFunc,
    truthy_fn: TruthyFunc,
) -> ShkValue:
    stmt_node, cond_node = _split_postfix_children(children, {"IF"})
    return _eval_postfix_guard(stmt_node, cond_node, frame, eval_func, truthy_fn, run_on_truthy=True)

def eval_postfix_unless(
    children: Sequence[Node],
    frame: Frame,
    *,
    eval_func: EvalFunc,
    truthy_fn: TruthyFunc,
) -> ShkValue:
    stmt_node, cond_node = _split_postfix_children(children, {"UNLESS"})

    return _eval_postfix_guard(stmt_node, cond_node, frame, eval_func, truthy_fn, run_on_truthy=False)

def _eval_postfix_guard(
    stmt_node: Node,
    cond_node: Node,
    frame: Frame,
    eval_func: EvalFunc,
    truthy_fn: TruthyFunc,
    *,
    run_on_truthy: bool,
) -> ShkValue:
    walrus_name = None
    walrus_node = find_tree_by_label(stmt_node, {"walrus", "walrus_nc"})
    if walrus_node is not None:
        walrus_name = _walrus_target_name(walrus_node)

    cond_val = eval_func(cond_node, frame)
    cond_truthy = truthy_fn(cond_val)
    should_run = cond_truthy if run_on_truthy else not cond_truthy
    if should_run:
        return eval_func(stmt_node, frame)

    if walrus_name is not None:
        define_new_ident(walrus_name, ShkNull(), frame)

    return ShkNull()
