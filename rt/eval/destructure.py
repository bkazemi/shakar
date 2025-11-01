from __future__ import annotations

from typing import Any, Iterable, List, Optional, Callable

from lark import Tree, Token

from shakar_runtime import Env, ShkArray, ShakarRuntimeError
from shakar_utils import (
    is_sequence_value,
    sequence_items,
    coerce_sequence,
    replicate_empty_sequence,
)

def _ident_token_value(node: Any) -> Optional[str]:
    if isinstance(node, Token) and node.type == "IDENT":
        return node.value
    return None

def evaluate_destructure_rhs(
    eval_fn: Callable[[Any, Env], Any],
    rhs_node: Any,
    env: Env,
    target_count: int,
    allow_broadcast: bool
) -> tuple[list[Any], Any]:
    if _tree_label(rhs_node) == "pack":
        vals = [eval_fn(child, env) for child in rhs_node.children]
        result = ShkArray(vals)
    else:
        single = eval_fn(rhs_node, env)
        vals = [single]
        result = single
    if len(vals) == 1 and target_count > 1:
        single = vals[0]
        if is_sequence_value(single):
            items = sequence_items(single)
            if len(items) == target_count:
                vals = list(items)
                result = ShkArray(vals)
            elif len(items) == 0 and allow_broadcast:
                replicated = replicate_empty_sequence(single, target_count)
                vals = replicated
                result = ShkArray(vals)
            else:
                raise ShakarRuntimeError("Destructure arity mismatch")
        elif allow_broadcast:
            vals = [single] * target_count
        else:
            raise ShakarRuntimeError("Destructure arity mismatch")
    elif len(vals) != target_count:
        raise ShakarRuntimeError("Destructure arity mismatch")
    return vals, result

def assign_pattern(
    eval_fn: Callable[[Any, Env], Any],
    assign_ident: Callable[[str, Any, Env, bool], Any],
    pattern: Tree,
    value: Any,
    env: Env,
    create: bool,
    allow_broadcast: bool
) -> None:
    if _tree_label(pattern) != "pattern" or not getattr(pattern, "children", None):
        raise ShakarRuntimeError("Malformed pattern")
    target = pattern.children[0]
    ident = _ident_token_value(target)
    if ident is not None:
        if create:
            env.define(ident, value)
        else:
            assign_ident(ident, value, env, create=False)
        return
    if _tree_label(target) == "pattern_list":
        subpatterns = [c for c in getattr(target, "children", []) if _tree_label(c) == "pattern"]
        if not subpatterns:
            raise ShakarRuntimeError("Empty nested pattern")
        seq = coerce_sequence(value, len(subpatterns))
        if seq is None:
            if allow_broadcast and len(subpatterns) > 1:
                seq = [value] * len(subpatterns)
            else:
                raise ShakarRuntimeError("Destructure expects a sequence")
        for sub_pat, sub_val in zip(subpatterns, seq):
            assign_pattern(eval_fn, assign_ident, sub_pat, sub_val, env, create, allow_broadcast)
        return
    raise ShakarRuntimeError("Unsupported pattern element")

def infer_implicit_binders(
    exprs: Iterable[Any],
    ifclause: Optional[Tree],
    env: Env,
    collect_fn: Callable[[Any, Callable[[str], None]], None]
) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    def consider(name: str) -> None:
        if name in seen:
            return
        if _name_exists(env, name):
            return
        seen.add(name)
        names.append(name)

    for expr in exprs:
        collect_fn(expr, consider)
    if ifclause is not None and ifclause.children:
        guard_expr = ifclause.children[-1]
        collect_fn(guard_expr, consider)
    return names

def apply_comp_binders(
    assign_fn: Callable[[Tree, Any, Env, bool, bool], None],
    binders: list[dict[str, Any]],
    mode: str,
    element: Any,
    iter_env: Env,
    outer_env: Env
) -> None:
    if not binders:
        return
    if len(binders) == 1:
        values = [element]
    else:
        seq = coerce_sequence(element, len(binders))
        if seq is None:
            raise ShakarRuntimeError("Comprehension element arity mismatch")
        values = seq
    for binder, val in zip(binders, values):
        target_env = outer_env if binder.get("hoist") else iter_env
        assign_fn(binder["pattern"], val, target_env, create=True, allow_broadcast=False)

def _tree_label(node: Any) -> Optional[str]:
    return node.data if isinstance(node, Tree) else None

def _name_exists(env: Env, name: str) -> bool:
    try:
        env.get(name)
        return True
    except ShakarRuntimeError:
        return False

def _child_by_label(node: Any, label: str):
    for child in getattr(node, "children", []):
        if _tree_label(child) == label:
            return child
    return None
