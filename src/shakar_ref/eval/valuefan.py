from __future__ import annotations

from typing import List, Optional
from ..tree import Tree

from ..runtime import Frame, ShkArray, ShkValue, ShakarRuntimeError
from ..token_types import TT
from ..tree import Node, is_tree, is_token, tree_children, tree_label
from .bind import FanContext, RebindContext
from .mutation import get_field_value
from .write_targets import WriteTarget


def eval_valuefan(
    base: ShkValue, fan_node: Tree, frame: Frame, eval_fn, apply_op
) -> ShkValue:
    """Evaluate value fanout `base.{...}` to an array; base is evaluated already."""
    items: List[ShkValue] = []
    seen: set[str] = set()

    for item in _iter_items(fan_node):
        key = _fingerprint(item)
        if key:
            if key in seen:
                raise ShakarRuntimeError("Value fan cannot contain duplicate fields")
            seen.add(key)

        items.append(_eval_item(base, item, frame, eval_fn, apply_op))

    return ShkArray(items)


def build_valuefan_context(
    base: ShkValue, fan_node: Tree, frame: Frame, eval_fn, apply_op
) -> FanContext:
    """Build a FanContext for valuefan in rebind chains, enabling writeback."""
    contexts: List[RebindContext] = []
    seen: set[str] = set()

    for item in _iter_items(fan_node):
        key = _fingerprint(item)
        if key:
            if key in seen:
                raise ShakarRuntimeError("Value fan cannot contain duplicate fields")
            seen.add(key)

        label = tree_label(item)

        if label == "field":
            # Simple field: capture a concrete write target.
            name = item.children[0].value
            value = get_field_value(base, name, frame)
            target = WriteTarget(
                kind="field",
                owner=base,
                name_or_index=name,
                frame=frame,
                create=False,
            )
            contexts.append(RebindContext(value, target))

        elif label in {"valuefan_chain", "identchain"}:
            # Chained access like {a.b}: evaluate but no writeback supported
            children = tree_children(item)
            if not children:
                raise ShakarRuntimeError("Malformed value fan item")
            head = children[0]
            ops = children[1:]
            val = get_field_value(base, head.value, frame)

            for op in ops:
                val = apply_op(val, op, frame, eval_fn)
                if isinstance(val, RebindContext):
                    val = val.value

            target = WriteTarget(
                kind="noop",
                owner=None,
                name_or_index=None,
                frame=frame,
                create=False,
            )
            contexts.append(RebindContext(val, target))

        else:
            # Fallback: evaluate with explicit no-op write target.
            val_frame = Frame(parent=frame, dot=base)
            val = eval_fn(item, val_frame)
            target = WriteTarget(
                kind="noop",
                owner=None,
                name_or_index=None,
                frame=frame,
                create=False,
            )
            contexts.append(RebindContext(val, target))

    return FanContext(contexts)


def _eval_item(base: ShkValue, item: Tree, frame: Frame, eval_fn, apply_op) -> ShkValue:
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
            val = apply_op(val, op, frame, eval_fn)
            if isinstance(val, RebindContext):
                val = val.value
        return val

    # Fallback: evaluate normally with anchor = base
    val_frame = Frame(parent=frame, dot=base)
    return eval_fn(item, val_frame)


def _iter_items(fan_node: Tree):
    for ch in tree_children(fan_node):
        label = tree_label(ch)
        if label in {"fieldfan", "valuefan_list", "fieldlist"}:
            yield from _iter_items(ch)
        elif label == "valuefan_item":
            for gc in tree_children(ch):
                if tree_label(gc) in {"valuefan_chain", "identchain"}:
                    yield gc
                elif tree_label(gc) == "field":
                    yield gc
                elif is_token(gc) and gc.type == TT.IDENT:
                    yield Tree("field", [gc])
        elif label in {"valuefan_chain", "identchain"}:
            yield ch
        elif label == "field":
            yield ch
        elif is_token(ch) and ch.type == TT.IDENT:
            yield Tree("field", [ch])


def _fingerprint(item: Node) -> Optional[str]:
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
