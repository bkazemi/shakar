from __future__ import annotations

from typing import Any, Iterable, List, Optional

from lark import Tree, Token

from shakar_runtime import (
    ShkArray,
    ShkString,
    ShkNumber,
    ShkNull,
    ShkSelector,
    SelectorIndex,
    SelectorSlice,
    SelectorPart,
    ShakarRuntimeError,
    ShakarTypeError,
)

from shakar_utils import sequence_items


def eval_selectorliteral(node, env, eval_fn) -> ShkSelector:
    sellist = _child_by_label(node, "sellist")
    if sellist is None:
        return ShkSelector([])
    parts: List[SelectorPart] = []
    for item in getattr(sellist, "children", []):
        label = _tree_label(item)
        if label == "selitem":
            parts.extend(_selector_parts_from_selitem(item, env, eval_fn))
        else:
            wrapped = _child_by_label(item, "selitem")
            target = wrapped if wrapped is not None else item
            parts.extend(_selector_parts_from_selitem(target, env, eval_fn))
    return ShkSelector(parts)


def evaluate_selectorlist(node, env, eval_fn, clamp: bool = True) -> List[SelectorPart]:
    selectors: List[SelectorPart] = []
    for raw_selector in getattr(node, "children", []):
        inner = _child_by_labels(raw_selector, {"slicesel", "indexsel"})
        target = inner if inner is not None else raw_selector
        label = _tree_label(target)
        if label == "slicesel":
            selectors.append(_selector_slice_from_slicesel(target, env, eval_fn, clamp))
            continue
        if label == "indexsel":
            expr_node = _first_child(target, lambda child: not isinstance(child, Token))
            if expr_node is None and getattr(target, "children", None):
                expr_node = target.children[0]
            value = eval_fn(expr_node, env)
            selectors.extend(_expand_selector_value(value, clamp))
            continue
        value = eval_fn(target, env)
        selectors.extend(_expand_selector_value(value, clamp))
    return selectors


def clone_selector_parts(parts: Iterable[SelectorPart], clamp: bool) -> List[SelectorPart]:
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


def apply_selectors_to_value(recv: Any, selectors: List[SelectorPart], env) -> Any:
    if isinstance(recv, ShkArray):
        return _apply_selectors_to_array(recv, selectors)
    if isinstance(recv, ShkString):
        return _apply_selectors_to_string(recv, selectors)
    raise ShakarTypeError("Complex selectors only supported on arrays or strings")


def selector_iter_values(selector: ShkSelector) -> List[Any]:
    values: List[Any] = []
    for part in selector.parts:
        if isinstance(part, SelectorIndex):
            idx = _selector_index_to_int(part.value)
            values.append(ShkNumber(float(idx)))
            continue
        values.extend(ShkNumber(float(i)) for i in _iterate_selector_slice(part))
    return values


def _selector_parts_from_selitem(node, env, eval_fn) -> List[SelectorPart]:
    inner = _child_by_labels(node, {"sliceitem", "indexitem"})
    target = inner if inner is not None else node
    label = _tree_label(target)
    if label == "sliceitem":
        return [_selector_slice_from_sliceitem(target, env, eval_fn)]
    if label == "indexitem":
        selatom = _child_by_label(target, "selatom")
        value = _eval_selector_atom(selatom, env, eval_fn)
        return [SelectorIndex(value)]
    return []


def _selector_slice_from_sliceitem(node, env, eval_fn) -> SelectorSlice:
    children = list(getattr(node, "children", []))
    index = 0
    start_node = None
    if index < len(children) and _tree_label(children[index]) == "selatom":
        start_node = children[index]
        index += 1
    stop_node = None
    if index < len(children) and _tree_label(children[index]) == "seloptstop":
        stop_node = children[index]
        index += 1
    step_node = None
    if index < len(children) and _tree_label(children[index]) == "selatom":
        step_node = children[index]
    start_val = _coerce_selector_number(_eval_selector_atom(start_node, env, eval_fn), allow_none=True)
    stop_value, exclusive = _eval_seloptstop(stop_node, env, eval_fn)
    stop_val = _coerce_selector_number(stop_value, allow_none=True)
    step_val = _coerce_selector_number(_eval_selector_atom(step_node, env, eval_fn), allow_none=True)
    return SelectorSlice(start=start_val, stop=stop_val, step=step_val, clamp=False, exclusive_stop=exclusive)


def _selector_slice_from_slicesel(node, env, eval_fn, clamp: bool) -> SelectorSlice:
    children = list(getattr(node, "children", []))
    start_node = children[0] if len(children) > 0 else None
    stop_node = children[1] if len(children) > 1 else None
    step_node = children[2] if len(children) > 2 else None
    start_val = _coerce_selector_number(_eval_optional_expr(start_node, env, eval_fn), allow_none=True)
    stop_val = _coerce_selector_number(_eval_optional_expr(stop_node, env, eval_fn), allow_none=True)
    step_val = _coerce_selector_number(_eval_optional_expr(step_node, env, eval_fn), allow_none=True)
    return SelectorSlice(start=start_val, stop=stop_val, step=step_val, clamp=clamp, exclusive_stop=True)


def _eval_optional_expr(node, env, eval_fn):
    if node is None:
        return None
    if _tree_label(node) == "emptyexpr":
        return None
    return eval_fn(node, env)


def _eval_selector_atom(node, env, eval_fn):
    if node is None:
        return None
    if not _is_tree(node):
        return eval_fn(node, env)
    if not getattr(node, "children", None):
        return eval_fn(node, env)
    child = node.children[0]
    if _tree_label(child) == "interp":
        expr = _child_by_label(child, "expr")
        if expr is None and getattr(child, "children", None):
            expr = child.children[0]
        if expr is None:
            raise ShakarRuntimeError("Empty interpolation in selector literal")
        return eval_fn(expr, env)
    if _is_tree(child):
        return eval_fn(child, env)
    return eval_fn(child, env)


def _eval_seloptstop(node, env, eval_fn) -> tuple[Any, bool]:
    if node is None:
        return None, False
    selatom = _child_by_label(node, "selatom")
    value = _eval_selector_atom(selatom, env, eval_fn)
    segment = _get_source_segment(node, env)
    exclusive = False
    if segment is not None:
        exclusive = segment.lstrip().startswith("<")
    return value, exclusive


def _coerce_selector_number(value: Any, allow_none: bool = False) -> Optional[int]:
    if value is None:
        if allow_none:
            return None
        raise ShakarTypeError("Selector expects a numeric bound")
    if isinstance(value, ShkNull):
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


def _expand_selector_value(value: Any, clamp: bool) -> List[SelectorPart]:
    if isinstance(value, ShkSelector):
        return clone_selector_parts(value.parts, clamp)
    return [SelectorIndex(value)]


def _apply_selectors_to_array(arr: ShkArray, selectors: List[SelectorPart]) -> ShkArray:
    result: List[Any] = []
    items = sequence_items(arr)
    length = len(items)
    for part in selectors:
        if isinstance(part, SelectorIndex):
            idx = _selector_index_to_int(part.value)
            pos = _normalize_index_position(idx, length)
            if pos < 0 or pos >= length:
                raise ShakarRuntimeError("Array index out of bounds")
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
                raise ShakarRuntimeError("String index out of bounds")
            pieces.append(s.value[pos])
            continue
        slice_obj = _selector_slice_to_slice(part, length)
        pieces.append(s.value[slice_obj])
    return ShkString("".join(pieces))


def _selector_index_to_int(value: Any) -> int:
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


def _get_source_segment(node, env) -> Optional[str]:
    source = getattr(env, "source", None)
    if source is None:
        return None
    meta = getattr(node, "meta", None)
    if meta is None:
        return None
    start = getattr(meta, "start_pos", None)
    end = getattr(meta, "end_pos", None)
    if start is None or end is None:
        return None
    return source[start:end]


def _tree_label(node: Any) -> Optional[str]:
    return node.data if isinstance(node, Tree) else None


def _is_tree(node: Any) -> bool:
    return isinstance(node, Tree)


def _child_by_label(node: Any, label: str):
    for child in getattr(node, "children", []):
        if _tree_label(child) == label:
            return child
    return None


def _child_by_labels(node: Any, labels: Iterable[str]):
    label_set = set(labels)
    for child in getattr(node, "children", []):
        if _tree_label(child) in label_set:
            return child
    return None


def _first_child(node: Any, predicate):
    for child in getattr(node, "children", []):
        if predicate(child):
            return child
    return None
