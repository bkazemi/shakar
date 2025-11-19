from __future__ import annotations

"""Helper routines for destructuring assignments and comprehensions."""

from typing import Any, Iterable, Optional, Callable

from lark import Tree, Token

from ..runtime import Frame, ShkArray, ShakarRuntimeError
from ..utils import (
    is_sequence_value,
    sequence_items,
    coerce_sequence,
    replicate_empty_sequence,
)
from ..tree import tree_label, tree_children

def _ident_token_value(node: Any) -> Optional[str]:
    if isinstance(node, Token) and node.type == "IDENT":
        return node.value

    return None

def evaluate_destructure_rhs(
    eval_fn: Callable[[Any, Frame], Any],
    rhs_node: Any,
    frame: Frame,
    target_count: int,
    allow_broadcast: bool
) -> tuple[list[Any], Any]:
    """Evaluate RHS once and expand/broadcast values to match the target count."""
    if tree_label(rhs_node) == "pack":
        # multiple RHS expressions separated by commas: evaluate each once.
        vals = [eval_fn(child, frame) for child in rhs_node.children]
        result = ShkArray(vals)
    else:
        # single RHS expression; treat as candidate for broadcast below.
        single = eval_fn(rhs_node, frame)
        vals = [single]
        result = single

    if len(vals) == 1 and target_count > 1:
        single = vals[0]

        if is_sequence_value(single):
            items = sequence_items(single)

            if len(items) == target_count:
                # exact match: destructure list/tuple straight into targets.
                vals = list(items)
                result = ShkArray(vals)
            elif len(items) == 0 and allow_broadcast:
                # special-case broadcasting empty iterables (e.g., [] or {})
                replicated = replicate_empty_sequence(single, target_count)
                vals = replicated
                result = ShkArray(vals)
            else:
                raise ShakarRuntimeError("Destructure arity mismatch")
        elif allow_broadcast:
            # non-sequence RHS but broadcast allowed (`:=` path): copy value.
            vals = [single] * target_count
        else:
            raise ShakarRuntimeError("Destructure arity mismatch")
    elif len(vals) != target_count:
        raise ShakarRuntimeError("Destructure arity mismatch")

    return vals, result

def assign_pattern(
    eval_fn: Callable[[Any, Frame], Any],
    assign_ident: Callable[[str, Any, Frame, bool], Any],
    pattern: Tree,
    value: Any,
    frame: Frame,
    create: bool,
    allow_broadcast: bool
) -> None:
    """Bind a destructuring pattern to a value, recursing into nested tuples."""
    if tree_label(pattern) != "pattern" or not tree_children(pattern):
        raise ShakarRuntimeError("Malformed pattern")

    target = pattern.children[0]
    ident = _ident_token_value(target)

    if ident is not None:
        if create:
            frame.define(ident, value)
        else:
            assign_ident(ident, value, frame, create=False)
        return

    if tree_label(target) == "pattern_list":
        subpatterns = [c for c in tree_children(target) if tree_label(c) == "pattern"]
        if not subpatterns:
            raise ShakarRuntimeError("Empty nested pattern")

        # nested list pattern: coerce RHS to a sequence of matching arity.
        seq = coerce_sequence(value, len(subpatterns))
        if seq is None:
            if allow_broadcast and len(subpatterns) > 1:
                seq = [value] * len(subpatterns)
            else:
                raise ShakarRuntimeError("Destructure expects a sequence")

        for sub_pat, sub_val in zip(subpatterns, seq):
            assign_pattern(eval_fn, assign_ident, sub_pat, sub_val, frame, create, allow_broadcast)
        return

    raise ShakarRuntimeError("Unsupported pattern element")

def infer_implicit_binders(
    exprs: Iterable[Any],
    ifclause: Optional[Tree],
    frame: Frame,
    collect_fn: Callable[[Any, Callable[[str], None]], None]
) -> list[str]:
    """Collect implicit binder names used inside comprehensions, skipping clashes."""
    names: list[str] = []
    seen: set[str] = set()

    def consider(name: str) -> None:
        if name in seen or _name_exists(frame, name):
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
    assign_fn: Callable[[Tree, Any, Frame, bool, bool], None],
    binders: list[dict[str, Any]],
    element: Any,
    iter_frame: Frame,
    outer_frame: Frame
) -> None:
    """Assign comprehension binder patterns for each element, honoring hoisting."""

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
        # hoisted binders write into the outer scope so closures can reuse them.
        target_frame = outer_frame if binder.get("hoist") else iter_frame
        assign_fn(binder["pattern"], val, target_frame, create=True, allow_broadcast=False)

def _name_exists(frame: Frame, name: str) -> bool:
    try:
        frame.get(name)
        return True
    except ShakarRuntimeError:
        return False
