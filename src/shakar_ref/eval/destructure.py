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
from ..tree import Node, Tree, child_by_label, tree_label, tree_children
from .helpers import collect_scope_names, find_frozen_scope_frame

EvalFn = Callable[[Node, Frame], ShkValue]

# Sentinel for "no value available" — distinct from ShkNil which is a valid user value.
_MISSING: object = object()


def _ident_token_value(node: Node) -> Optional[str]:
    if isinstance(node, Tok) and node.type == TT.IDENT:
        return str(node.value)

    return None


def _extract_pattern_info(
    patterns: list[Tree],
) -> Optional[list[tuple[str, bool]]]:
    """Extract identifier names and default flags from patterns.

    Returns a list of (name, has_default) tuples, or None if any pattern is
    non-simple (nested tuple, rest, etc.), signalling the caller to fall
    through to positional logic.
    """
    info: list[tuple[str, bool]] = []

    for pat in patterns:
        if tree_label(pat) == "pattern_rest":
            return None
        if tree_label(pat) != "pattern" or not tree_children(pat):
            return None
        ident = _ident_token_value(tree_children(pat)[0])
        if ident is None:
            return None
        has_default = child_by_label(pat, "default") is not None
        info.append((ident, has_default))

    return info


def evaluate_destructure_rhs(
    eval_fn: EvalFn,
    rhs_node: Node,
    frame: Frame,
    target_count: int,
    allow_broadcast: bool,
    ident_names: Optional[list[str]] = None,
    has_defaults: Optional[list[bool]] = None,
    rest_index: Optional[int] = None,
) -> tuple[list[ShkValue], ShkValue]:
    """Evaluate RHS once and expand/broadcast values to match the target count.

    When rest_index is set, the rest element collects surplus values.
    When has_defaults is set, missing positions with defaults get _MISSING.
    """
    # Normalize: empty defaults list => all-False list matching target_count
    if not has_defaults:
        has_defaults = [False] * target_count

    if tree_label(rhs_node) == "recv" and target_count == 2 and rest_index is None:
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

    # --- Rest pattern handling ---
    if rest_index is not None:
        # min_count is the number of non-rest patterns
        min_count = target_count - 1
        return _expand_with_rest(
            vals,
            result,
            min_count,
            rest_index,
            target_count,
            allow_broadcast,
            ident_names,
            has_defaults,
        )

    if len(vals) == 1 and target_count > 1:
        single = vals[0]

        # Object keyed extraction: if RHS is an object and all LHS
        # patterns are simple identifiers, extract fields by name.
        if isinstance(single, ShkObject) and ident_names:
            extracted: list[ShkValue] = []
            for i, name in enumerate(ident_names):
                if name not in single.slots:
                    if has_defaults[i]:
                        extracted.append(_MISSING)  # type: ignore[arg-type]
                    else:
                        raise ShakarKeyError(name)
                else:
                    extracted.append(single.slots[name])
            vals = extracted
            result = ShkArray(vals)
        elif is_sequence_value(single):
            items = sequence_items(single)

            if len(items) == target_count:
                # exact match: destructure list/tuple straight into targets.
                vals = list(items)
                result = ShkArray(vals)
            elif any(has_defaults) and len(items) < target_count:
                # Pad with _MISSING for positions that have defaults
                padded: list[ShkValue] = list(items)
                for i in range(len(items), target_count):
                    if has_defaults[i]:
                        padded.append(_MISSING)  # type: ignore[arg-type]
                    else:
                        raise ShakarRuntimeError("Destructure arity mismatch")
                vals = padded
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
        # Multi-value pack with defaults: pad if short
        if any(has_defaults) and len(vals) < target_count:
            for i in range(len(vals), target_count):
                if has_defaults[i]:
                    vals.append(_MISSING)  # type: ignore[arg-type]
                else:
                    raise ShakarRuntimeError("Destructure arity mismatch")
        elif len(vals) != target_count:
            raise ShakarRuntimeError("Destructure arity mismatch")

    return vals, result


def _expand_with_rest(
    vals: list[ShkValue],
    result: ShkValue,
    min_count: int,
    rest_index: int,
    target_count: int,
    allow_broadcast: bool,
    ident_names: Optional[list[str]],
    has_defaults: list[bool],
) -> tuple[list[ShkValue], ShkValue]:
    """Expand RHS values when a rest pattern is present."""
    # If single value, unpack it first
    if len(vals) == 1:
        single = vals[0]

        # Object rest: extract named keys, remaining => rest object
        if isinstance(single, ShkObject) and ident_names:
            out: list[ShkValue] = []
            remaining = dict(single.slots)

            for i, name in enumerate(ident_names):
                if name is None:
                    # rest position — skip for now
                    out.append(ShkNil())
                    continue
                if name not in remaining:
                    if has_defaults[i]:
                        out.append(_MISSING)  # type: ignore[arg-type]
                    else:
                        raise ShakarKeyError(name)
                else:
                    out.append(remaining.pop(name))

            # Build rest object from remaining keys
            rest_obj = ShkObject(remaining)
            out[rest_index] = rest_obj
            return out, result

        # Sequence rest: items before rest => 1:1, remaining => rest array
        if is_sequence_value(single):
            items = list(sequence_items(single))
        elif allow_broadcast:
            items = [single]
        else:
            raise ShakarRuntimeError("Destructure expects a sequence")

        return (
            _split_seq_with_rest(
                items,
                rest_index,
                min_count,
                target_count,
                has_defaults,
            ),
            result,
        )

    # Multiple RHS values (pack): split with rest
    return _split_seq_with_rest(
        vals,
        rest_index,
        min_count,
        target_count,
        has_defaults,
    ), ShkArray(vals)


def _split_seq_with_rest(
    items: list[ShkValue],
    rest_index: int,
    min_count: int,
    target_count: int,
    has_defaults: list[bool],
) -> list[ShkValue]:
    """Split a flat sequence into non-rest slots + rest array."""
    # Number of patterns after the rest
    after_rest = min_count - rest_index

    if len(items) < rest_index:
        # Not enough items even for slots before rest
        out: list[ShkValue] = list(items)
        for i in range(len(items), rest_index):
            if has_defaults[i]:
                out.append(_MISSING)  # type: ignore[arg-type]
            else:
                raise ShakarRuntimeError("Destructure arity mismatch")

        # Rest gets empty array
        out.append(ShkArray([]))

        # Slots after rest get defaults or error
        for i in range(rest_index + 1, target_count):
            if has_defaults[i]:
                out.append(_MISSING)  # type: ignore[arg-type]
            else:
                raise ShakarRuntimeError("Destructure arity mismatch")

        return out

    # Items before rest
    before = items[:rest_index]
    remaining = items[rest_index:]

    # Items that must go to slots after rest
    if len(remaining) >= after_rest:
        if after_rest > 0:
            after_items = remaining[len(remaining) - after_rest :]
            rest_items = remaining[: len(remaining) - after_rest]
        else:
            after_items = []
            rest_items = remaining
    else:
        # Not enough values for trailing slots after the rest.
        # Keep suffix alignment and require defaults for missing leading slots.
        missing_after = after_rest - len(remaining)
        after_items = []
        for i in range(rest_index + 1, rest_index + 1 + missing_after):
            if has_defaults[i]:
                after_items.append(_MISSING)  # type: ignore[arg-type]
            else:
                raise ShakarRuntimeError("Destructure arity mismatch")
        after_items.extend(remaining)
        rest_items = []

    out = list(before)
    out.append(ShkArray(rest_items))
    out.extend(after_items)

    return out


def _rest_index(patterns: list[Tree]) -> Optional[int]:
    for i, pat in enumerate(patterns):
        if tree_label(pat) == "pattern_rest":
            return i
    return None


def _pattern_binding_metadata(
    patterns: list[Tree], rest_index: Optional[int]
) -> tuple[Optional[list[str]], Optional[list[bool]]]:
    """Build keyed-extraction names and default flags for destructure expansion."""
    info = _extract_pattern_info(patterns) if len(patterns) > 1 else None
    ident_names: Optional[list[str]] = None
    has_defaults: Optional[list[bool]] = None

    if info:
        ident_names = [name for name, _ in info]
        has_defaults = [hd for _, hd in info]
        if not any(has_defaults):
            has_defaults = None
        return ident_names, has_defaults

    if rest_index is None:
        return None, None

    names: list[Optional[str]] = []
    defaults: list[bool] = []
    for pat in patterns:
        if tree_label(pat) == "pattern_rest":
            names.append(None)
            defaults.append(False)
        elif tree_label(pat) == "pattern" and tree_children(pat):
            ident = _ident_token_value(tree_children(pat)[0])
            names.append(ident)
            defaults.append(child_by_label(pat, "default") is not None)
        else:
            names.append(None)
            defaults.append(False)

    if all(n is not None for i, n in enumerate(names) if i != rest_index):
        ident_names = names  # type: ignore[assignment]
    has_defaults = defaults if any(defaults) else None
    return ident_names, has_defaults


def _materialize_destructure_defaults(
    patterns: list[Tree],
    values: list[ShkValue],
    frame: Frame,
    eval_fn: EvalFn,
) -> bool:
    """Resolve _MISSING sentinels using per-pattern default expressions."""
    had_missing = False
    for i, (pat, val) in enumerate(zip(patterns, values)):
        if val is _MISSING:
            had_missing = True
            default_node = child_by_label(pat, "default")
            if default_node and tree_children(default_node):
                values[i] = eval_fn(tree_children(default_node)[0], frame)
            else:
                raise ShakarRuntimeError("Missing value in destructure")
    return had_missing


def prepare_destructure_bindings(
    node: Tree,
    frame: Frame,
    eval_fn: EvalFn,
    *,
    allow_broadcast: bool,
    malformed_message: str,
    empty_message: str,
) -> tuple[list[Tree], list[ShkValue], ShkValue]:
    """Evaluate destructure RHS and resolve defaults before binding."""
    if len(node.children) != 2:
        raise ShakarRuntimeError(malformed_message)

    pattern_list, rhs_node = node.children
    patterns = [
        c
        for c in tree_children(pattern_list)
        if tree_label(c) in ("pattern", "pattern_rest")
    ]
    if not patterns:
        raise ShakarRuntimeError(empty_message)

    rest_index = _rest_index(patterns)
    ident_names, has_defaults = _pattern_binding_metadata(patterns, rest_index)

    values, result = evaluate_destructure_rhs(
        eval_fn,
        rhs_node,
        frame,
        len(patterns),
        allow_broadcast,
        ident_names=ident_names,
        has_defaults=has_defaults,
        rest_index=rest_index,
    )

    had_missing = _materialize_destructure_defaults(patterns, values, frame, eval_fn)
    if allow_broadcast and had_missing:
        result = ShkArray(values)

    return patterns, values, result


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
    label = tree_label(pattern)

    # Rest pattern: ...ident — just bind the collected rest value directly
    if label == "pattern_rest":
        children = tree_children(pattern)
        if not children:
            raise ShakarRuntimeError("Malformed rest pattern")
        ident = _ident_token_value(children[0])
        if not ident:
            raise ShakarRuntimeError("Malformed rest pattern")
        assign_ident(ident, value, frame, create)
        return

    if label != "pattern" or not tree_children(pattern):
        raise ShakarRuntimeError("Malformed pattern")

    target = pattern.children[0]
    ident = _ident_token_value(target)

    if ident:
        # Check for contract
        contract_node = child_by_label(pattern, "contract")

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

        # nested list pattern: preserve sequence arity errors and only broadcast
        # non-sequence values in walrus mode.
        seq: Optional[list[ShkValue]] = None

        if is_sequence_value(value):
            items = list(sequence_items(value))

            if len(items) > len(subpatterns):
                raise ShakarRuntimeError("Destructure arity mismatch")

            if len(items) == len(subpatterns):
                seq = items
            else:
                # Short sequence: pad with _MISSING where defaults exist.
                seq = list(items)
                for i in range(len(items), len(subpatterns)):
                    if child_by_label(subpatterns[i], "default"):
                        seq.append(_MISSING)  # type: ignore[arg-type]
                    else:
                        raise ShakarRuntimeError("Destructure arity mismatch")
        elif allow_broadcast and len(subpatterns) > 1:
            seq = [value] * len(subpatterns)
        else:
            raise ShakarRuntimeError("Destructure expects a sequence")

        # Materialize defaults for _MISSING values
        for i, (sub_pat, sub_val) in enumerate(zip(subpatterns, seq)):
            if sub_val is _MISSING:
                default_node = child_by_label(sub_pat, "default")
                if default_node and tree_children(default_node):
                    seq[i] = eval_fn(tree_children(default_node)[0], frame)
                else:
                    raise ShakarRuntimeError("Missing value in destructure")

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
    patterns, values, result = prepare_destructure_bindings(
        node,
        frame,
        eval_fn,
        allow_broadcast=allow_broadcast,
        malformed_message="Malformed destructure",
        empty_message="Empty destructure pattern",
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
