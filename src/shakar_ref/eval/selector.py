from __future__ import annotations

from typing import Callable, Iterable, List, Optional

from ..runtime import (
    Frame,
    ShkArray,
    ShkString,
    ShkNumber,
    ShkNull,
    ShkSelector,
    SelectorIndex,
    SelectorSlice,
    SelectorPart,
    ShkValue,
    ShakarRuntimeError,
    ShakarTypeError,
    ShakarIndexError,
)

from ..utils import sequence_items
from ..tree import (
    Node,
    Token,
    tree_children,
    tree_label,
    child_by_label,
    child_by_labels,
    first_child,
    node_meta,
    is_tree,
    is_token,
)

EvalFunc = Callable[[Node, Frame], ShkValue]

def eval_selectorliteral(node: Tree, frame: Frame, eval_fn: EvalFunc) -> ShkSelector:
    """Build a Selector value from a literal `` `...` `` expression."""
    sellist = child_by_label(node, "sellist")
    if sellist is None:
        return ShkSelector([])

    parts: List[SelectorPart] = []

    for item in tree_children(sellist):
        label = tree_label(item)

        if label == "selitem":
            parts.extend(_selector_parts_from_selitem(item, frame, eval_fn))
        else:
            # grammar may wrap selitem inside helper nodes; unwrap when present.
            wrapped = child_by_label(item, "selitem")
            target = wrapped if wrapped is not None else item
            parts.extend(_selector_parts_from_selitem(target, frame, eval_fn))

    return ShkSelector(parts)

def evaluate_selectorlist(node: Tree, frame: Frame, eval_fn: EvalFunc, clamp: bool = True) -> List[SelectorPart]:
    """Evaluate runtime selector expressions like `xs[sel1, sel2]` into parts."""
    selectors: List[SelectorPart] = []

    for raw_selector in tree_children(node):
        inner = child_by_labels(raw_selector, {"slicesel", "indexsel"})
        target = inner if inner is not None else raw_selector
        label = tree_label(target)

        if label == "slicesel":
            # slicesel already encodes explicit start/stop/step nodes.
            selectors.append(_selector_slice_from_slicesel(target, frame, eval_fn, clamp))
            continue

        if label == "indexsel":
            expr_node = first_child(target, lambda child: not is_token(child))

            if expr_node is None:
                children = tree_children(target)

                if children:
                    expr_node = children[0]
            idx_value: ShkValue = eval_fn(expr_node, frame)
            # indexsel can evaluate to either a single index or another selector literal.
            selectors.extend(_expand_selector_value(idx_value, clamp))
            continue

        tail_value: ShkValue = eval_fn(target, frame)
        selectors.extend(_expand_selector_value(tail_value, clamp))

    return selectors

def clone_selector_parts(parts: Iterable[SelectorPart], clamp: bool) -> List[SelectorPart]:
    """Copy selector parts so mutations (e.g., clamping) do not affect originals."""
    cloned: List[SelectorPart] = []

    for part in parts:
        if isinstance(part, SelectorSlice):
            cloned.append(
                SelectorSlice(
                    start=part.start,
                    stop=part.stop,
                    step=part.step,
                    clamp=clamp,
                    exclusive_stop=part.exclusive_stop,
                )
            )
        elif isinstance(part, SelectorIndex):
            cloned.append(SelectorIndex(part.value))

    return cloned

def apply_selectors_to_value(recv: ShkValue, selectors: List[SelectorPart]) -> ShkValue:
    """Apply a list of selector parts to an array/string receiver."""
    if isinstance(recv, ShkArray):
        return _apply_selectors_to_array(recv, selectors)

    if isinstance(recv, ShkString):
        return _apply_selectors_to_string(recv, selectors)

    raise ShakarTypeError("Complex selectors only supported on arrays or strings")

def selector_iter_values(selector: ShkSelector) -> List[ShkValue]:
    """Expand a selector literal into the sequence of indices it would visit."""
    values: List[ShkValue] = []

    for part in selector.parts:
        if isinstance(part, SelectorIndex):
            idx = _selector_index_to_int(part.value)
            values.append(ShkNumber(float(idx)))
            continue

        values.extend(ShkNumber(float(i)) for i in _iterate_selector_slice(part))

    return values

def _selector_parts_from_selitem(node: Tree, frame: Frame, eval_fn: EvalFunc) -> List[SelectorPart]:
    """Turn a literal selitem node into concrete slice/index parts."""
    inner = child_by_labels(node, {"sliceitem", "indexitem"})
    target = inner if inner is not None else node
    label = tree_label(target)

    if label == "sliceitem":
        return [_selector_slice_from_sliceitem(target, frame, eval_fn)]

    if label == "indexitem":
        selatom = child_by_label(target, "selatom")
        value = _eval_selector_atom(selatom, frame, eval_fn)
        return [SelectorIndex(value)]

    return []

def _selector_slice_from_sliceitem(node: Tree, frame: Frame, eval_fn: EvalFunc) -> SelectorSlice:
    children = list(tree_children(node))
    index = 0
    start_node = None

    if index < len(children) and tree_label(children[index]) == "selatom":
        start_node = children[index]
        index += 1

    stop_node = None
    if index < len(children) and tree_label(children[index]) == "seloptstop":
        stop_node = children[index]
        index += 1

    step_node = None
    if index < len(children) and tree_label(children[index]) == "selatom":
        step_node = children[index]

    # sliceitem uses selatom nodes for start/step and seloptstop for stop.
    start_val = _coerce_selector_number(_eval_selector_atom(start_node, frame, eval_fn), allow_none=True)
    stop_value, exclusive = _eval_seloptstop(stop_node, frame, eval_fn)
    stop_val = _coerce_selector_number(stop_value, allow_none=True)
    step_val = _coerce_selector_number(_eval_selector_atom(step_node, frame, eval_fn), allow_none=True)

    _validate_slice_signs(start_val, stop_val)

    return SelectorSlice(start=start_val, stop=stop_val, step=step_val, clamp=False, exclusive_stop=exclusive)

def _selector_slice_from_slicesel(node: Tree, frame: Frame, eval_fn: EvalFunc, clamp: bool) -> SelectorSlice:
    children = list(tree_children(node))

    start_node = children[0] if len(children) > 0 else None
    stop_node = children[1] if len(children) > 1 else None
    step_node = children[2] if len(children) > 2 else None

    start_val = _coerce_selector_number(_eval_optional_expr(start_node, frame, eval_fn), allow_none=True)
    stop_val = _coerce_selector_number(_eval_optional_expr(stop_node, frame, eval_fn), allow_none=True)
    step_val = _coerce_selector_number(_eval_optional_expr(step_node, frame, eval_fn), allow_none=True)
    # slicesel originates from runtime selector expressions `xs[start:stop:step]`.

    _validate_slice_signs(start_val, stop_val)

    return SelectorSlice(start=start_val, stop=stop_val, step=step_val, clamp=clamp, exclusive_stop=True)

def _eval_optional_expr(node: Optional[Tree], frame: Frame, eval_fn: EvalFunc) -> Optional[ShkValue]:
    if node is None:
        return None

    label = tree_label(node)

    if label in {"emptyexpr", "slicearm_empty"}:
        return None

    if label == "slicearm_expr":
        ch = tree_children(node)
        return eval_fn(ch[0], frame) if ch else None

    return eval_fn(node, frame)

def _eval_selector_atom(node: Optional[Tree], frame: Frame, eval_fn: EvalFunc) -> Optional[ShkValue]:
    if node is None:
        return None

    if not is_tree(node):
        return eval_fn(node, frame)

    node_children = tree_children(node)
    if not node_children:
        return eval_fn(node, frame)

    child = node_children[0]

    if tree_label(child) == "interp":
        expr = child_by_label(child, "expr")

        if expr is None:
            child_children = tree_children(child)

            if child_children:
                expr = child_children[0]

        if expr is None:
            raise ShakarRuntimeError("Empty interpolation in selector literal")
        # `` `start:${expr}` `` — evaluate embedded expression on demand.
        return eval_fn(expr, frame)

    if is_tree(child):
        return eval_fn(child, frame)

    return eval_fn(child, frame)

def _eval_seloptstop(node: Optional[Tree], frame: Frame, eval_fn: EvalFunc) -> tuple[ShkValue, bool]:
    if node is None:
        return ShkNull(), False

    children = tree_children(node)
    exclusive = False

    # Check for LT token as first child (exclusive slice from RD parser)
    if children and is_token(children[0]) and children[0].type == 'LT':
        exclusive = True

    # Also check source segment for backwards compatibility with Lark parser
    if not exclusive:
        segment = _get_source_segment(node, frame)
        if segment is not None and segment.lstrip().startswith("<"):
            exclusive = True

    selatom = child_by_label(node, "selatom")
    value = _eval_selector_atom(selatom, frame, eval_fn)

    return value, exclusive

def _validate_slice_signs(start: Optional[int], stop: Optional[int]) -> None:
    """Reject slices with negative start and positive stop (ambiguous semantics)."""
    if start is None or stop is None:
        return

    if start < 0 and stop >= 0:
        raise ShakarRuntimeError("Slice cannot have negative start with positive stop")

def _coerce_selector_number(value: Optional[ShkValue], allow_none: bool = False) -> Optional[int]:
    if value is None or isinstance(value, ShkNull):
        if allow_none:
            return None
        raise ShakarTypeError("Selector expects a numeric bound")

    if isinstance(value, ShkNumber):
        num = value.value
    else:
        raise ShakarTypeError("Selector expects a numeric bound")

    if not float(num).is_integer():
        raise ShakarTypeError("Selector bounds must be integral")

    return int(num)

def _expand_selector_value(value: ShkValue, clamp: bool) -> List[SelectorPart]:
    if isinstance(value, ShkSelector):
        return clone_selector_parts(value.parts, clamp)

    return [SelectorIndex(value)]

def _apply_selectors_to_array(arr: ShkArray, selectors: List[SelectorPart]) -> ShkArray:
    result: List[ShkValue] = []
    items = sequence_items(arr)
    length = len(items)

    for part in selectors:
        if isinstance(part, SelectorIndex):
            idx = _selector_index_to_int(part.value)
            pos = _normalize_index_position(idx, length)

            if pos < 0 or pos >= length:
                raise ShakarIndexError("Array index out of bounds")

            result.append(items[pos])
            continue

        slice_obj = _selector_slice_to_slice(part, length)
        result.extend(items[slice_obj])

    return ShkArray(result)

def _apply_selectors_to_string(s: ShkString, selectors: List[SelectorPart]) -> ShkString:
    pieces: List[str] = []
    length = len(s.value)

    for part in selectors:
        if isinstance(part, SelectorIndex):
            idx = _selector_index_to_int(part.value)
            pos = _normalize_index_position(idx, length)

            if pos < 0 or pos >= length:
                raise ShakarIndexError("String index out of bounds")

            pieces.append(s.value[pos])
            continue

        slice_obj = _selector_slice_to_slice(part, length)
        pieces.append(s.value[slice_obj])

    return ShkString("".join(pieces))

def _selector_index_to_int(value: ShkValue) -> int:
    if isinstance(value, ShkNumber):
        num = value.value
    else:
        raise ShakarTypeError("Index selector expects a numeric value")

    if not float(num).is_integer():
        raise ShakarTypeError("Index selector expects an integer value")

    return int(num)

def _normalize_index_position(index: int, length: int) -> int:
    if index < 0:
        return index + length

    return index

def _selector_slice_to_slice(part: SelectorSlice, length: int) -> slice:
    step = part.step if part.step is not None else 1
    if step == 0:
        raise ShakarRuntimeError("Slice step cannot be zero")

    start = part.start
    stop = part.stop

    if part.clamp:
        stop_adj = stop

        if not part.exclusive_stop and stop_adj is not None:
            stop_adj = stop_adj + (1 if step > 0 else -1)

        slice_obj = slice(start, stop_adj, step)
        normalized = slice_obj.indices(length)
        return slice(*normalized)

    if start is None:
        start = 0 if step > 0 else length - 1

    if stop is None:
        stop = length if step > 0 else -1

    if not part.exclusive_stop and stop is not None:
        stop = stop + (1 if step > 0 else -1)

    if start < 0:
        start += length

    if stop is not None and stop < 0:
        stop += length

    return slice(start, stop, step)

def _iterate_selector_slice(part: SelectorSlice) -> Iterable[int]:
    step = part.step if part.step is not None else 1
    if step == 0:
        raise ShakarRuntimeError("Selector slice step cannot be zero")

    if part.start is None or part.stop is None:
        raise ShakarRuntimeError("Selector slice requires explicit start and stop when iterated")

    stop = part.stop if part.exclusive_stop else part.stop + (1 if step > 0 else -1)

    return range(part.start, stop, step)

def _get_source_segment(node: Node, frame: Frame) -> Optional[str]:
    source = getattr(frame, "source", None)
    if source is None:
        return None

    meta = node_meta(node)
    if meta is None:
        return None

    start = getattr(meta, "start_pos", None)
    end = getattr(meta, "end_pos", None)
    if start is None or end is None:
        return None

    return str(source[start:end])
