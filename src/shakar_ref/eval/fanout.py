from __future__ import annotations

from typing import List, Optional, Tuple, Union
from ..tree import Tok

from ..runtime import Frame, ShkValue, EvalResult, ShakarRuntimeError
from ..tree import Node, Tree, child_by_label, is_tree, tree_children, tree_label
from .bind import FanContext, RebindContext, _read_segment, _write_segment
from .common import unwrap_noanchor
from .selector import (
    evaluate_selectorlist,
    SelectorIndex,
    SelectorSlice,
    _selector_slice_to_slice,
    _normalize_index_position,
)
from ..runtime import ShkArray

from .expr import apply_binary_operator


# Operators we allow inside fanout blocks map directly to binary symbols.
_COMPOUND_MAP = {
    "fanop_pluseq": "+",
    "fanop_minuseq": "-",
    "fanop_stareq": "*",
    "fanop_slasheq": "/",
    "fanop_floordiveq": "//",
    "fanop_modeq": "%",
    "fanop_poweq": "**",
}


def eval_fanout_block(
    node: Tree,
    frame: Frame,
    eval_fn,
    apply_op,
    evaluate_index_operand,
) -> ShkValue:
    """Evaluate `Base{ .a = 1; .b += 2 }` fanout statements."""
    if len(node.children) != 2:
        raise ShakarRuntimeError("Malformed fanout block")

    base_node, block = node.children
    base_val = eval_fn(base_node, frame)

    saved_dot = frame.dot
    frame.dot = base_val

    seen_keys: set[tuple[str, ...]] = set()

    try:
        for clause in tree_children(block):
            if _name(tree_label(clause)) == "fanclause_sep":
                continue
            key = _fan_key(clause)
            if key:
                if key in seen_keys:
                    raise ShakarRuntimeError(
                        "Fanout block cannot target the same path twice"
                    )
                seen_keys.add(key)
            _eval_clause(
                base_val, clause, frame, eval_fn, apply_op, evaluate_index_operand
            )
    finally:
        frame.dot = saved_dot

    return base_val


def _fan_key(clause: Tree) -> Optional[Tuple[str, ...]]:
    """Return a hashable key for field/index paths to dedup targets."""
    parts: list[str] = []
    for child in tree_children(clause):
        if is_tree(child) and _name(tree_label(child)) == "fanpath":
            for raw_seg in tree_children(child):
                seg, label = unwrap_noanchor(raw_seg)
                label = _name(label)
                if label in {"field", "fieldsel"}:
                    parts.append(f".{seg.children[0].value}")
                elif label == "lv_index":
                    # Encode index expression textually to avoid evaluating twice.
                    parts.append(f"[{_seg_fingerprint(seg)}]")
                else:
                    return None
    return tuple(parts) if parts else None


def _eval_clause(
    base_val: ShkValue,
    clause: Tree,
    frame: Frame,
    eval_fn,
    apply_op,
    evaluate_index_operand,
) -> None:
    children = tree_children(clause)
    if len(children) < 3:
        raise ShakarRuntimeError("Malformed fanout clause")

    fanpath_node = next(
        (ch for ch in children if is_tree(ch) and _name(tree_label(ch)) == "fanpath"),
        None,
    )
    op_node = next(
        (
            ch
            for ch in children
            if is_tree(ch) and _name(tree_label(ch)).startswith("fanop_")
        ),
        None,
    )
    rhs_node = children[-1]

    if fanpath_node is None or op_node is None:
        raise ShakarRuntimeError("Malformed fanout clause")

    segments = _fan_segments(fanpath_node)
    if not segments:
        raise ShakarRuntimeError("Empty fanout path")

    target_obj, final_seg = _walk_to_parent(
        base_val, segments, frame, eval_fn, apply_op
    )
    op_label = tree_label(op_node)
    targets = _iter_targets(target_obj)

    if op_label == "fanop_assign":
        new_val = eval_fn(rhs_node, frame)
        for tgt in targets:
            _store_target(
                tgt, final_seg, new_val, frame, evaluate_index_operand, eval_fn
            )
        return

    if op_label == "fanop_apply":
        for tgt in targets:
            base = tgt.value if isinstance(tgt, RebindContext) else tgt
            old_val = _read(base, final_seg, frame, evaluate_index_operand, eval_fn)
            rhs_frame = Frame(parent=frame, dot=old_val)
            new_val = eval_fn(rhs_node, rhs_frame)
            _store_target(
                tgt, final_seg, new_val, frame, evaluate_index_operand, eval_fn
            )
        return

    op_symbol = _COMPOUND_MAP.get(op_label)
    if op_symbol is None:
        raise ShakarRuntimeError(f"Unsupported fanout operator {op_label}")

    rhs_val = eval_fn(rhs_node, frame)
    for tgt in targets:
        base = tgt.value if isinstance(tgt, RebindContext) else tgt
        old_val = _read(base, final_seg, frame, evaluate_index_operand, eval_fn)
        new_val = apply_binary_operator(op_symbol, old_val, rhs_val)
        _store_target(tgt, final_seg, new_val, frame, evaluate_index_operand, eval_fn)


def _iter_targets(target_obj: EvalResult) -> List[ShkValue | RebindContext]:
    if isinstance(target_obj, FanContext):
        return list(target_obj.contexts)
    return [target_obj]


def _fan_segments(path_node: Node) -> List[Tree]:
    return [seg for seg in tree_children(path_node) if is_tree(seg)]


def _walk_to_parent(
    base_val: ShkValue, segments: List[Tree], frame: Frame, eval_fn, apply_op
) -> tuple[ShkValue, Tree]:
    current = base_val

    for idx, seg in enumerate(segments[:-1]):
        prev = current
        current = apply_op(current, seg, frame, eval_fn)

        # When an intermediate *multi* selector returns an array, broadcast subsequent ops across its elements.
        if (
            isinstance(current, ShkArray)
            and idx < len(segments) - 1
            and _segment_is_multi_selector(seg)
            and isinstance(prev, ShkArray)
        ):
            current = _fan_context_from_selector(prev, seg, frame, eval_fn)

        if isinstance(current, RebindContext):
            current = current.value

    return current, segments[-1]


def _segment_is_multi_selector(seg: Tree) -> bool:
    """Return True if selector list contains a slice or multiple selectors."""
    selectorlist = child_by_label(seg, "selectorlist")
    if selectorlist is None:
        return False
    selectors = [ch for ch in tree_children(selectorlist) if is_tree(ch)]
    if len(selectors) > 1:
        return True
    if not selectors:
        return False
    # single selector: check if it is a slice selector
    sel = selectors[0]
    return any(
        is_tree(grand) and tree_label(grand) == "slicesel"
        for grand in tree_children(sel)
    )


def _fan_context_from_selector(
    arr: ShkArray, seg: Tree, frame: Frame, eval_fn
) -> FanContext:
    """Build FanContext targeting the original array elements selected by seg."""
    selectorlist = child_by_label(seg, "selectorlist")
    if selectorlist is None:
        raise ShakarRuntimeError("Malformed selector in fanout path")

    parts = evaluate_selectorlist(selectorlist, frame, eval_fn, clamp=True)
    indices: list[int] = []
    length = len(arr.items)

    for part in parts:
        if isinstance(part, SelectorIndex):
            idx_raw = part.value
            if not hasattr(idx_raw, "value"):
                raise ShakarRuntimeError("Selector index must be numeric")
            idx = int(idx_raw.value)
            idx = _normalize_index_position(idx, length)
            if idx < 0 or idx >= length:
                raise ShakarRuntimeError("Array index out of bounds")
            indices.append(idx)
        elif isinstance(part, SelectorSlice):
            py_slice = _selector_slice_to_slice(part, length)
            indices.extend(range(*py_slice.indices(length)))
        else:
            raise ShakarRuntimeError("Unsupported selector part in fanout")

    contexts: list[RebindContext] = []

    for i in indices:

        def _setter_factory(pos: int):
            def setter(new_val: ShkValue) -> None:
                if pos >= len(arr.items):
                    raise ShakarRuntimeError("Fanout slice target vanished")
                arr.items[pos] = new_val

            return setter

        contexts.append(RebindContext(arr.items[i], _setter_factory(i)))

    return FanContext(contexts)


def _store_target(
    target: ShkValue | RebindContext,
    final_seg: Tree,
    value: ShkValue,
    frame: Frame,
    evaluate_index_operand,
    eval_fn,
) -> None:
    if isinstance(target, RebindContext):
        # Apply store to the underlying value, then write back via setter to keep container in sync.
        _store(target.value, final_seg, value, frame, evaluate_index_operand, eval_fn)
        target.setter(target.value)
        return
    _store(target, final_seg, value, frame, evaluate_index_operand, eval_fn)


def _read(
    target: ShkValue,
    raw_final_seg: Tree,
    frame: Frame,
    evaluate_index_operand,
    eval_fn,
) -> ShkValue:
    seg, label = unwrap_noanchor(raw_final_seg)

    return _read_segment(target, seg, label, frame, evaluate_index_operand, eval_fn)


def _store(
    target: ShkValue,
    raw_final_seg: Tree,
    value: ShkValue,
    frame: Frame,
    evaluate_index_operand,
    eval_fn,
) -> None:
    seg, label = unwrap_noanchor(raw_final_seg)
    _write_segment(target, seg, label, value, frame, evaluate_index_operand, eval_fn)


def _name(label: Optional[Union[str, Tok]]) -> str:
    if isinstance(label, Tok):
        return str(label.value)
    return str(label) if label else ""


def _seg_fingerprint(seg: Tree) -> str:
    # Rough, stable-enough textual fingerprint of an lv_index child tree.
    buf: list[str] = []

    def walk(node: Node) -> None:
        if is_tree(node):
            buf.append(_name(tree_label(node)))
            for ch in tree_children(node):
                walk(ch)
        elif isinstance(node, Tok):
            buf.append(str(node.value))

    walk(seg)
    return "|".join(buf)
