"""Helper routines for destructuring assignments and comprehensions."""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Any

from ..tree import Tok
from ..token_types import TT

from ..runtime import (
    Frame,
    ShkArray,
    ShkBool,
    ShkChannel,
    ShkNil,
    ShkObject,
    ShkValue,
    ShakarKeyError,
    ShakarRuntimeError,
    ShakarAssertionError,
)
from ..utils import (
    is_sequence_value,
    sequence_items,
    coerce_sequence,
    replicate_empty_sequence,
)
from ..tree import Node, Tree, tree_label, tree_children
from .helpers import collect_scope_names, find_frozen_scope_frame

EvalFn = Callable[[Node, Frame], ShkValue]


def _ident_token_value(node: Node) -> Optional[str]:
    if isinstance(node, Tok) and node.type == TT.IDENT:
        return str(node.value)

    return None


def _extract_pattern_idents(patterns: list[Tree]) -> Optional[list[str]]:
    """Extract identifier names from patterns.

    Returns None if any pattern is not a simple IDENT (e.g. nested tuple
    patterns), signalling the caller to fall through to positional logic.
    """
    names: list[str] = []

    for pat in patterns:
        if tree_label(pat) != "pattern" or not tree_children(pat):
            return None
        ident = _ident_token_value(tree_children(pat)[0])
        if ident is None:
            return None
        names.append(ident)

    return names


def evaluate_destructure_rhs(
    eval_fn: EvalFn,
    rhs_node: Node,
    frame: Frame,
    target_count: int,
    allow_broadcast: bool,
    ident_names: Optional[list[str]] = None,
) -> tuple[list[ShkValue], ShkValue]:
    """Evaluate RHS once and expand/broadcast values to match the target count."""
    if tree_label(rhs_node) == "recv" and target_count == 2:
        children = tree_children(rhs_node)
        if not children:
            raise ShakarRuntimeError("Malformed receive expression")
        chan_val = eval_fn(children[0], frame)
        if isinstance(chan_val, ShkNil):
            raise ShakarRuntimeError("Receive on nil channel")
        if not isinstance(chan_val, ShkChannel):
            raise ShakarRuntimeError("Expected channel on receive")
        value, ok = chan_val.recv_with_ok(cancel_token=frame.cancel_token)
        ok_val = ShkBool(ok)
        vals = [value, ok_val]
        return vals, ShkArray(vals)

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

        # Object keyed extraction: if RHS is an object and all LHS
        # patterns are simple identifiers, extract fields by name.
        if isinstance(single, ShkObject) and ident_names:
            extracted: list[ShkValue] = []
            for name in ident_names:
                if name not in single.slots:
                    raise ShakarKeyError(name)
                extracted.append(single.slots[name])
            vals = extracted
            result = ShkArray(vals)
        elif is_sequence_value(single):
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
    eval_fn: EvalFn,
    assign_ident: Callable[[str, ShkValue, Frame, bool], ShkValue],
    pattern: Tree,
    value: ShkValue,
    frame: Frame,
    create: bool,
    allow_broadcast: bool,
) -> None:
    """Bind a destructuring pattern to a value, recursing into nested tuples."""
    if tree_label(pattern) != "pattern" or not tree_children(pattern):
        raise ShakarRuntimeError("Malformed pattern")

    target = pattern.children[0]
    ident = _ident_token_value(target)

    if ident:
        # Check for contract: pattern(IDENT, contract(expr))
        contract_node = None
        if len(pattern.children) > 1 and tree_label(pattern.children[1]) == "contract":
            contract_node = pattern.children[1]

        # Validate contract before binding
        if contract_node:
            from .match import match_structure

            contract_children = tree_children(contract_node)
            if contract_children:
                contract_expr = contract_children[0]
                contract_value = eval_fn(contract_expr, frame)
                if not match_structure(value, contract_value):
                    raise ShakarAssertionError(
                        f"Destructure contract failed: {ident} ~ {contract_value}, got {value}"
                    )

        # Bind the value
        assign_ident(ident, value, frame, create)
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
            assign_pattern(
                eval_fn, assign_ident, sub_pat, sub_val, frame, create, allow_broadcast
            )
        return

    raise ShakarRuntimeError("Unsupported pattern element")


def infer_implicit_binders(
    exprs: Iterable[Tree],
    ifclause: Optional[Tree],
    frame: Frame,
    collect_fn: Callable[[Tree, Callable[[str], None]], None],
) -> list[str]:
    """Collect implicit binder names used inside comprehensions, skipping clashes."""
    names: list[str] = []
    seen: set[str] = set()
    # Stop at the nearest frozen lexical boundary to avoid outer scope drift.
    boundary = find_frozen_scope_frame(frame)
    local_names = collect_scope_names(frame, boundary)
    frozen_names = boundary.frozen_scope_names if boundary else None

    def consider(name: str) -> None:
        if name in seen:
            return
        if name in local_names:
            return
        if frozen_names and name in frozen_names:
            return

        seen.add(name)
        names.append(name)

    for expr in exprs:
        collect_fn(expr, consider)

    if ifclause and ifclause.children:
        guard_expr = ifclause.children[-1]
        collect_fn(guard_expr, consider)

    return names


def apply_comp_binders(
    assign_fn: Callable[[Node, ShkValue, Frame], None],
    binders: list[dict[str, Any]],
    element: ShkValue,
    iter_frame: Frame,
    outer_frame: Frame,
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
        assign_fn(binder["pattern"], val, target_frame)


def eval_destructure(
    node: Tree, frame: Frame, eval_fn: EvalFn, create: bool, allow_broadcast: bool
) -> ShkValue:
    """Evaluate destructuring assignment/expression."""
    if len(node.children) != 2:
        raise ShakarRuntimeError("Malformed destructure")

    pattern_list, rhs_node = node.children
    patterns = [c for c in tree_children(pattern_list) if tree_label(c) == "pattern"]

    if not patterns:
        raise ShakarRuntimeError("Empty destructure pattern")

    ident_names = _extract_pattern_idents(patterns) if len(patterns) > 1 else None
    values, result = evaluate_destructure_rhs(
        eval_fn,
        rhs_node,
        frame,
        len(patterns),
        allow_broadcast,
        ident_names=ident_names,
    )

    # local import avoids circular reference with bind module.
    from .bind import assign_pattern_value

    for pat, val in zip(patterns, values):
        assign_pattern_value(
            pat,
            val,
            frame,
            create=create,
            allow_broadcast=allow_broadcast,
            eval_fn=eval_fn,
        )

    return result if allow_broadcast else ShkNil()
