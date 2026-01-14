from __future__ import annotations

from typing import List
from ..tree import Tree

from ..runtime import Frame, ShkArray, ShkValue, ShakarRuntimeError
from ..tree import Node, is_tree, tree_children, tree_label
from .bind import RebindContext
from .mutation import get_field_value


def eval_valuefan(
    base: ShkValue, fan_node: Tree, frame: Frame, eval_func, apply_op
) -> ShkValue:
    """Evaluate value fanout `base.{...}` to an array; base is evaluated already."""
    items: List[ShkValue] = []
    seen: set[str] = set()

    for item in _iter_items(fan_node):
        key = _fingerprint(item)
        if key is not None:
            if key in seen:
                raise ShakarRuntimeError("Value fan cannot contain duplicate fields")
            seen.add(key)

        items.append(_eval_item(base, item, frame, eval_func, apply_op))

    return ShkArray(items)


def _eval_item(
    base: ShkValue, item: Tree, frame: Frame, eval_func, apply_op
) -> ShkValue:
    """Evaluate a single valuefan item, applying postfix ops starting from base.field."""
    label = tree_label(item)
    if label == "field":
        name = item.children[0].value
        return get_field_value(base, name, frame)

    if label in {"valuefan_chain", "identchain"}:
        children = tree_children(item)
        if not children:
            raise ShakarRuntimeError("Malformed value fan item")
        head = children[0]
        ops = children[1:]
        val = get_field_value(base, head.value, frame)

        for op in ops:
            val = apply_op(val, op, frame, eval_func)
            if isinstance(val, RebindContext):
                val = val.value
        return val

    # Fallback: evaluate normally with anchor = base
    val_frame = Frame(parent=frame, dot=base)
    return eval_func(item, val_frame)


def _iter_items(fan_node: Tree):
    for ch in tree_children(fan_node):
        label = tree_label(ch)
        if is_tree(ch) and label in {"fieldfan", "valuefan_list"}:
            yield from _iter_items(ch)
        elif is_tree(ch) and label == "valuefan_item":
            for gc in tree_children(ch):
                if is_tree(gc) and tree_label(gc) in {"valuefan_chain", "identchain"}:
                    yield gc
                elif is_tree(gc) and tree_label(gc) == "field":
                    yield gc
                elif not is_tree(gc):  # bare IDENT token
                    yield Tree("field", [gc])
        elif is_tree(ch) and label in {"valuefan_chain", "identchain"}:
            yield ch
        elif is_tree(ch) and label == "field":
            yield ch


def _wrap_chain(chain: Tree) -> Tree:
    # valuefan_chain already a Tree; just return
    return chain


def _fingerprint(item: Node) -> str | None:
    if not is_tree(item):
        return None

    label = tree_label(item)
    if label == "field":
        return item.children[0].value
    if label in {"valuefan_chain", "identchain"}:
        return "|".join(_flatten_tokens(item))
    return None


def _flatten_tokens(node: Node) -> List[str]:
    tokens: List[str] = []

    def walk(n: Node) -> None:
        if is_tree(n):
            tokens.append(tree_label(n))
            for ch in tree_children(n):
                walk(ch)
        else:
            tokens.append(str(getattr(n, "value", "")))

    walk(node)
    return tokens
