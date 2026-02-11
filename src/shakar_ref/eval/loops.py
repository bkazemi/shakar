from __future__ import annotations

from typing import Callable, Iterable, List, Optional, TypedDict
import glob

from ..tree import Tree, Tok
from ..token_types import TT

from ..runtime import (
    Frame,
    ShkArray,
    ShkFan,
    ShkChannel,
    ShkNil,
    ShkNumber,
    ShkObject,
    ShkModule,
    ShkPath,
    ShkSelector,
    ShkString,
    ShkValue,
    ShakarBreakSignal,
    ShakarContinueSignal,
    ShakarRuntimeError,
    ShakarTypeError,
)
from ..tree import Node, child_by_label, is_token, is_tree, tree_children, tree_label
from ..utils import normalize_object_key, value_in_list
from .bind import assign_pattern_value
from .blocks import eval_body_node
from .common import (
    collect_free_identifiers as _collect_free_identifiers,
    token_kind as _token_kind,
)
from .destructure import apply_comp_binders, infer_implicit_binders
from .helpers import is_truthy as _is_truthy
from .selector import selector_iter_values

EvalFn = Callable[[Node, Frame], ShkValue]


class BinderSpec(TypedDict):
    pattern: Tree
    hoist: bool


def _parse_comphead(node: Tree) -> tuple[Tree, list[BinderSpec]]:
    overspec = child_by_label(node, "overspec")
    if overspec is None:
        raise ShakarRuntimeError("Malformed comprehension head")

    return _parse_overspec(overspec)


def _parse_overspec(node: Tree) -> tuple[Tree, list[BinderSpec]]:
    children = list(node.children)
    binders: list[BinderSpec] = []

    if not children:
        raise ShakarRuntimeError("Malformed overspec")

    first = children[0]
    if tree_label(first) == "binderlist":
        if len(children) < 2:
            raise ShakarRuntimeError("Binder list requires a source")

        iter_expr_node = children[1]

        for bp in first.children:
            bp_label = tree_label(bp)

            if bp_label == "binderpattern" and bp.children:
                pattern_node = bp.children[0]
                pattern_label = tree_label(pattern_node)

                if pattern_label == "pattern" and pattern_node.children:
                    child = pattern_node.children[0]
                    if tree_label(child) == "pattern_list":
                        raise ShakarRuntimeError("Binder list cannot use parentheses")

                binders.append({"pattern": bp.children[0], "hoist": False})
            elif bp_label == "hoist" and bp.children:
                tok = bp.children[0]
                pattern = Tree("pattern", [tok])
                binders.append({"pattern": pattern, "hoist": True})
        return iter_expr_node, binders

    iter_expr_node = children[0]

    if len(children) > 1:
        pattern = children[1]
        binders.append({"pattern": pattern, "hoist": False})

    return iter_expr_node, binders


def _extract_loop_iter_and_body(
    children: List[Node],
) -> tuple[Optional[Node], Optional[Tree]]:
    iter_expr = None
    body_node = None

    for child in children:
        if is_tree(child):
            label = tree_label(child)
            if label in {"binderpattern", "hoist", "pattern"}:
                continue

            if label == "body":
                body_node = child
            elif iter_expr is None:
                iter_expr = child
            else:
                body_node = child
        else:
            if _token_kind(child) in {"FOR", "IN", "LSQB", "RSQB", "COMMA", "COLON"}:
                continue

            if iter_expr is None:
                iter_expr = child
            else:
                body_node = child

    return iter_expr, body_node


def _coerce_loop_binder(node: Tree) -> BinderSpec:
    target = node

    if tree_label(target) == "binderpattern" and target.children:
        target = target.children[0]

    if tree_label(target) == "hoist":
        tok = target.children[0] if target.children else None
        if tok is None or not is_token(tok):
            raise ShakarRuntimeError("Malformed hoisted binder")

        pattern = Tree("pattern", [tok])
        return {"pattern": pattern, "hoist": True}

    if tree_label(target) == "pattern":
        return {"pattern": target, "hoist": False}

    if _token_kind(target) == "IDENT":
        pattern = Tree("pattern", [target])
        return {"pattern": pattern, "hoist": False}

    raise ShakarRuntimeError("Malformed binder pattern")


def _pattern_requires_object_pair(pattern: Tree) -> bool:
    if not is_tree(pattern):
        return False

    return any(
        tree_label(child) == "pattern_list"
        and sum(1 for elem in tree_children(child) if tree_label(elem) == "pattern")
        >= 2
        for child in tree_children(pattern)
    )


def _iter_indexed_entries(
    value: ShkValue, binder_count: int
) -> list[tuple[ShkValue, list[ShkValue]]]:
    if binder_count <= 0:
        raise ShakarRuntimeError("Indexed loop requires at least one binder")

    if binder_count > 2:
        raise ShakarRuntimeError("Indexed loop supports at most two binders")

    entries: list[tuple[ShkValue, list[ShkValue]]] = []
    binders: list[ShkValue]
    match value:
        case ShkArray(items=items) | ShkFan(items=items):
            for idx, item in enumerate(items):
                binders = [ShkNumber(float(idx))]

                if binder_count > 1:
                    binders.append(item)
                entries.append((item, binders[:binder_count]))
        case ShkString(value=s):
            for idx, ch in enumerate(s):
                char = ShkString(ch)
                binders = [ShkNumber(float(idx))]

                if binder_count > 1:
                    binders.append(char)
                entries.append((char, binders[:binder_count]))
        case ShkModule(slots=slots) | ShkObject(slots=slots):
            for key, val in slots.items():
                binders = [ShkString(key)]

                if binder_count > 1:
                    binders.append(val)
                entries.append((val, binders[:binder_count]))
        case ShkSelector():
            values = selector_iter_values(value)

            for idx, sel in enumerate(values):
                binders = [ShkNumber(float(idx))]

                if binder_count > 1:
                    binders.append(sel)

                entries.append((sel, binders[:binder_count]))
        case ShkPath():
            values = _iter_path_values(value)

            for idx, item in enumerate(values):
                binders = [ShkNumber(float(idx))]

                if binder_count > 1:
                    binders.append(item)

                entries.append((item, binders[:binder_count]))
        case _:
            raise ShakarTypeError(f"Cannot use indexed loop on {type(value).__name__}")
    return entries


def _path_has_wildcards(text: str) -> bool:
    # Supports glob wildcards *, ?, and ** (recursive via glob.glob(..., recursive=True)).
    return "*" in text or "?" in text


def _iter_path_values(value: ShkPath) -> List[ShkValue]:
    raw = str(value)
    path = value.as_path()

    if _path_has_wildcards(raw):
        matches = glob.glob(raw, recursive=True)
        return [ShkPath(match) for match in sorted(matches)]

    if path.exists():
        if path.is_dir():
            entries = sorted(path.iterdir(), key=lambda p: p.name)
            return [ShkPath(str(p)) for p in entries]
        return [value]

    return []


def _iterable_values(value: ShkValue) -> list[ShkValue]:
    match value:
        case ShkNil():
            return []
        case ShkNumber(value=num):
            num_val = float(num)
            if not num_val.is_integer():
                raise ShakarTypeError("Cannot iterate over non-integer number")
            count = int(num_val)
            if count < 0:
                raise ShakarTypeError("Cannot iterate over negative number")
            return [ShkNumber(float(i)) for i in range(count)]
        case ShkArray(items=items):
            return list(items)
        case ShkFan(items=items):
            return list(items)
        case ShkString(value=s):
            return [ShkString(ch) for ch in s]
        case ShkModule(slots=slots) | ShkObject(slots=slots):
            return [ShkString(k) for k in slots.keys()]
        case ShkSelector():
            return selector_iter_values(value)
        case ShkPath():
            return _iter_path_values(value)
        case ShkChannel():
            raise ShakarTypeError(
                "Cannot iterate over channel in comprehension or spread"
            )
        case _:
            raise ShakarTypeError(f"Cannot iterate over {type(value).__name__}")


def _iter_channel_values(channel: ShkChannel, frame: Frame) -> Iterable[ShkValue]:
    while True:
        value, ok = channel.recv_with_ok(cancel_token=frame.cancel_token)
        if not ok:
            break
        yield value


def _apply_comp_binders_wrapper(
    binders: list[dict[str, ShkValue]],
    element: ShkValue,
    iter_frame: Frame,
    outer_frame: Frame,
    eval_fn: EvalFn,
) -> None:
    apply_comp_binders(
        lambda pattern, val, target_frame: assign_pattern_value(
            pattern,
            val,
            target_frame,
            create=True,
            allow_broadcast=False,
            eval_fn=eval_fn,
        ),
        binders,
        element,
        iter_frame,
        outer_frame,
    )


def _prepare_comprehension(
    n: Tree, frame: Frame, head_nodes: list[Tree], eval_fn: EvalFn
) -> tuple[Tree, list[dict[str, ShkValue]], Optional[Tree]]:
    comphead = child_by_label(n, "comphead")
    if comphead is None:
        raise ShakarRuntimeError("Malformed comprehension")

    ifclause = child_by_label(n, "ifclause")
    iter_expr_node, binders = _parse_comphead(comphead)

    if not binders:
        implicit_names = infer_implicit_binders(
            head_nodes,
            ifclause,
            frame,
            lambda expr, callback: _collect_free_identifiers(expr, callback),
        )

        for name in implicit_names:
            pattern = Tree("pattern", [Tok(TT.IDENT, name, 0, 0)])
            binders.append({"pattern": pattern, "hoist": False})

    iter_val = eval_fn(iter_expr_node, frame)

    return iter_val, binders, ifclause


def _iterate_comprehension(
    n: Tree, frame: Frame, head_nodes: list[Tree], eval_fn: EvalFn
) -> Iterable[tuple[ShkValue, Frame]]:
    iter_val, binders, ifclause = _prepare_comprehension(n, frame, head_nodes, eval_fn)
    outer_dot = frame.dot

    try:
        for element in _iterable_values(iter_val):
            iter_frame = Frame(parent=frame, dot=element)
            _apply_comp_binders_wrapper(binders, element, iter_frame, frame, eval_fn)

            if ifclause is not None:
                cond_node = ifclause.children[-1] if ifclause.children else None
                if cond_node is None:
                    raise ShakarRuntimeError("Malformed comprehension guard")

                cond_val = eval_fn(cond_node, iter_frame)

                if not _is_truthy(cond_val):
                    continue

            yield element, iter_frame
    finally:
        frame.dot = outer_dot


def eval_while_stmt(n: Tree, frame: Frame, eval_fn: EvalFn) -> ShkValue:
    cond_node = None
    body_node = None

    for child in tree_children(n):
        if is_token(child):
            if cond_node is None and _token_kind(child) not in {"WHILE", "COLON"}:
                cond_node = child
            continue
        if cond_node is None:
            cond_node = child
        elif body_node is None:
            body_node = child

    if cond_node is None or body_node is None:
        raise ShakarRuntimeError("Malformed while statement")

    outer_dot = frame.dot

    try:
        while _is_truthy(eval_fn(cond_node, frame)):
            try:
                eval_body_node(body_node, frame, eval_fn, allow_loop_control=True)
            except ShakarContinueSignal:
                continue
            except ShakarBreakSignal:
                break
    finally:
        frame.dot = outer_dot

    return ShkNil()


def eval_for_in(n: Tree, frame: Frame, eval_fn: EvalFn) -> ShkValue:
    pattern_node = None
    iter_expr = None
    body_node = None
    after_in = False

    for child in tree_children(n):
        if is_tree(child) and tree_label(child) == "pattern":
            pattern_node = child
            continue

        if is_token(child):
            tok = _token_kind(child)

            if tok == "FOR":
                continue
            if tok == "IN":
                after_in = True
                continue
            if tok == "COLON":
                continue

            if pattern_node is None and tok == "IDENT":
                pattern_node = Tree("pattern", [child])
                continue

            if after_in and iter_expr is None:
                iter_expr = child
            continue

        if iter_expr is None:
            iter_expr = child
        else:
            body_node = child

    if iter_expr is None or body_node is None:
        raise ShakarRuntimeError("Malformed for-in loop")

    if pattern_node is None:
        raise ShakarRuntimeError("For-in loop missing pattern")

    iter_source = eval_fn(iter_expr, frame)
    if isinstance(iter_source, ShkChannel):
        iterable = _iter_channel_values(iter_source, frame)
    else:
        iterable = _iterable_values(iter_source)
    outer_dot = frame.dot
    object_pairs: Optional[list[tuple[str, ShkValue]]] = None

    if isinstance(iter_source, ShkObject):
        object_pairs = list(iter_source.slots.items())

    needs_pair = object_pairs is not None and _pattern_requires_object_pair(
        pattern_node
    )

    try:
        for idx, value in enumerate(iterable):
            loop_frame = Frame(parent=frame, dot=outer_dot)
            assigned = value

            if needs_pair:
                key, val = object_pairs[idx]
                assigned = ShkArray([ShkString(key), val])
            assign_pattern_value(
                pattern_node,
                assigned,
                loop_frame,
                create=True,
                allow_broadcast=False,
                eval_fn=eval_fn,
            )

            try:
                eval_body_node(body_node, loop_frame, eval_fn, allow_loop_control=True)
            except ShakarContinueSignal:
                continue
            except ShakarBreakSignal:
                break
    finally:
        frame.dot = outer_dot
    return ShkNil()


def eval_for_subject(n: Tree, frame: Frame, eval_fn: EvalFn) -> ShkValue:
    iter_expr = None
    body_node = None

    for child in tree_children(n):
        # Skip FOR keyword token, but not IDENT (which is the iterable expr)
        if is_token(child) and getattr(child, "type", None) == TT.FOR:
            continue

        if iter_expr is None:
            iter_expr = child
        elif body_node is None and is_tree(child):
            body_node = child

    if iter_expr is None or body_node is None:
        raise ShakarRuntimeError("Malformed subjectful for loop")

    iter_source = eval_fn(iter_expr, frame)
    if isinstance(iter_source, ShkChannel):
        iterable = _iter_channel_values(iter_source, frame)
    else:
        iterable = _iterable_values(iter_source)
    outer_dot = frame.dot

    try:
        for value in iterable:
            loop_frame = Frame(parent=frame, dot=value)

            try:
                eval_body_node(body_node, loop_frame, eval_fn, allow_loop_control=True)
            except ShakarContinueSignal:
                continue
            except ShakarBreakSignal:
                break
    finally:
        frame.dot = outer_dot
    return ShkNil()


def eval_for_indexed(n: Tree, frame: Frame, eval_fn: EvalFn) -> ShkValue:
    if n is None:
        raise ShakarRuntimeError("Malformed indexed loop")

    children = tree_children(n)
    binder_nodes: list[Tree] = []

    for child in children:
        if is_tree(child) and tree_label(child) in {
            "binderpattern",
            "hoist",
            "pattern",
        }:
            binder_nodes.append(child)

    if not binder_nodes:
        raise ShakarRuntimeError("Indexed loop missing binder")

    iter_expr, body_node = _extract_loop_iter_and_body(children)

    if iter_expr is None or body_node is None:
        raise ShakarRuntimeError("Malformed indexed loop")

    binders = [_coerce_loop_binder(node) for node in binder_nodes]
    iterable = eval_fn(iter_expr, frame)
    entries = _iter_indexed_entries(iterable, len(binders))
    outer_dot = frame.dot

    try:
        for subject, binder_values in entries:
            loop_frame = Frame(parent=frame, dot=subject)

            for binder, binder_value in zip(binders, binder_values):
                target_frame = frame if binder.get("hoist") else loop_frame
                assign_pattern_value(
                    binder["pattern"],
                    binder_value,
                    target_frame,
                    create=True,
                    allow_broadcast=False,
                    eval_fn=eval_fn,
                )

            try:
                eval_body_node(body_node, loop_frame, eval_fn, allow_loop_control=True)
            except ShakarContinueSignal:
                continue
            except ShakarBreakSignal:
                break
    finally:
        frame.dot = outer_dot
    return ShkNil()


def eval_for_map2(n: Tree, frame: Frame, eval_fn: EvalFn) -> ShkValue:
    return eval_for_indexed(n, frame, eval_fn)


def eval_listcomp(n: Tree, frame: Frame, eval_fn: EvalFn) -> ShkArray:
    body = n.children[0] if n.children else None
    if body is None:
        raise ShakarRuntimeError("Malformed list comprehension")

    items = [
        eval_fn(body, iter_frame)
        for _, iter_frame in _iterate_comprehension(n, frame, [body], eval_fn)
    ]

    return ShkArray(items)


def eval_setcomp(n: Tree, frame: Frame, eval_fn: EvalFn) -> ShkArray:
    body = n.children[0] if n.children else None
    if body is None:
        raise ShakarRuntimeError("Malformed set comprehension")

    items: list[ShkValue] = []
    for _, iter_frame in _iterate_comprehension(n, frame, [body], eval_fn):
        result = eval_fn(body, iter_frame)

        if not value_in_list(items, result):
            items.append(result)

    return ShkArray(items)


def eval_setliteral(n: Tree, frame: Frame, eval_fn: EvalFn) -> ShkArray:
    items: list[ShkValue] = []
    for child in tree_children(n):
        val = eval_fn(child, frame)

        if not value_in_list(items, val):
            items.append(val)

    return ShkArray(items)


def eval_dictcomp(n: Tree, frame: Frame, eval_fn: EvalFn) -> ShkObject:
    if len(n.children) < 3:
        raise ShakarRuntimeError("Malformed dict comprehension")

    key_node = n.children[0]
    value_node = n.children[1]
    slots: dict[str, ShkValue] = {}

    for _, iter_frame in _iterate_comprehension(
        n, frame, [key_node, value_node], eval_fn
    ):
        key_val = eval_fn(key_node, iter_frame)
        value_val = eval_fn(value_node, iter_frame)
        key_str = normalize_object_key(key_val)
        slots[key_str] = value_val

    return ShkObject(slots)
