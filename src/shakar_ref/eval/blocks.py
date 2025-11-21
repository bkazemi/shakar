from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, List

from lark import Tree

from ..runtime import DeferEntry, Frame, ShkNull, ShakarBreakSignal, ShakarContinueSignal, ShakarRuntimeError
from ..tree import TreeNode, is_token_node, is_tree_node, tree_children, tree_label
from .common import expect_ident_token as _expect_ident_token, token_kind as _token_kind
from .helpers import is_truthy as _is_truthy

EvalFunc = Callable[[Any, Frame], Any]

def eval_program(children: List[Any], frame: Frame, eval_func: EvalFunc, allow_loop_control: bool=False) -> Any:
    """Run a stmt list under a fresh defer scope, returning last value."""
    result: Any = ShkNull()
    skip_tokens = {'SEMI', '_NL', 'INDENT', 'DEDENT'}
    push_defer_scope(frame)

    try:
        try:
            for child in children:
                if _token_kind(child) in skip_tokens:
                    continue
                result = eval_func(child, frame)
        except ShakarBreakSignal:
            if allow_loop_control:
                raise
            raise ShakarRuntimeError("break outside of a loop") from None
        except ShakarContinueSignal:
            if allow_loop_control:
                raise
            raise ShakarRuntimeError("continue outside of a loop") from None
    finally:
        pop_defer_scope(frame)

    return result

def get_subject(frame: Frame) -> Any:
    if frame.dot is None:
        raise ShakarRuntimeError("No subject available for '.'")

    return frame.dot

def push_defer_scope(frame: Frame) -> None:
    frame.push_defer_frame()

def pop_defer_scope(frame: Frame) -> None:
    entries = frame.pop_defer_frame()
    if entries:
        _run_defer_entries(entries)

_DEFER_UNVISITED = 0
_DEFER_VISITING = 1
_DEFER_DONE = 2

def _run_defer_entries(entries: List[DeferEntry]) -> None:
    if not entries:
        return

    label_map: dict[str, int] = {}

    for idx, entry in enumerate(entries):
        if entry.label:
            label_map[entry.label] = idx
    state = [_DEFER_UNVISITED] * len(entries)

    def run_index(idx: int) -> None:
        marker = state[idx]
        if marker == _DEFER_DONE:
            return
        if marker == _DEFER_VISITING:
            raise ShakarRuntimeError("Defer dependency cycle detected")

        state[idx] = _DEFER_VISITING
        entry = entries[idx]

        for dep in entry.deps:
            dep_idx = label_map.get(dep)

            if dep_idx is None:
                raise ShakarRuntimeError(f"Unknown defer handle '{dep}'")
            run_index(dep_idx)

        state[idx] = _DEFER_DONE
        entry.thunk()

    for idx in reversed(range(len(entries))):
        run_index(idx)

def schedule_defer(frame: Frame, thunk: Callable[[], None], label: str | None=None, deps: List[str] | None=None) -> None:
    if not frame.has_defer_frame():
        raise ShakarRuntimeError("Cannot use defer outside of a block")

    defer_frame = frame.current_defer_frame()
    entry = DeferEntry(thunk=thunk, label=label, deps=list(deps or []))

    if label:
        for existing in defer_frame:
            if existing.label == label:
                raise ShakarRuntimeError(f"Duplicate defer handle '{label}'")

    defer_frame.append(entry)

def eval_inline_body(node: Any, frame: Frame, eval_func: EvalFunc, allow_loop_control: bool=False) -> Any:
    if tree_label(node) == 'inlinebody':
        for child in tree_children(node):
            if tree_label(child) == 'stmtlist':
                return eval_program(child.children, frame, eval_func, allow_loop_control=allow_loop_control)

        if not tree_children(node):
            return ShkNull()
        return eval_func(node.children[0], frame)

    return eval_func(node, frame)

def eval_indent_block(node: TreeNode, frame: Frame, eval_func: EvalFunc, allow_loop_control: bool=False) -> Any:
    return eval_program(node.children, frame, eval_func, allow_loop_control=allow_loop_control)

def eval_oneline_guard(children: List[Any], frame: Frame, eval_func: EvalFunc) -> Any:
    branches: List[TreeNode] = []
    else_body: TreeNode | None = None

    for child in children:
        data = tree_label(child)
        if data == 'guardbranch':
            branches.append(child)
        elif data == 'inlinebody':
            else_body = child

    outer_dot = frame.dot

    for branch in branches:
        if not is_tree_node(branch) or len(branch.children) != 2:
            raise ShakarRuntimeError("Malformed guard branch")

        cond_node, body_node = branch.children

        with temporary_subject(frame, outer_dot):
            cond_val = eval_func(cond_node, frame)

        if _is_truthy(cond_val):
            with temporary_subject(frame, outer_dot):
                return eval_inline_body(body_node, frame, eval_func)

    if else_body is not None:
        with temporary_subject(frame, outer_dot):
            return eval_inline_body(else_body, frame, eval_func)

    frame.dot = outer_dot

    return ShkNull()

def eval_body_node(body_node: Any, frame: Frame, eval_func: EvalFunc) -> Any:
    label = tree_label(body_node) if is_tree_node(body_node) else None

    if label == 'inlinebody':
        return eval_inline_body(body_node, frame, eval_func)

    if label == 'indentblock':
        return eval_indent_block(body_node, frame, eval_func)

    return eval_func(body_node, frame)

def run_body_with_subject(body_node: Any, frame: Frame, subject_value: Any, eval_func: EvalFunc, extra_bindings: dict[str, Any] | None=None) -> Any:
    if extra_bindings:
        with temporary_subject(frame, subject_value), temporary_bindings(frame, extra_bindings):
            return eval_body_node(body_node, frame, eval_func)

    with temporary_subject(frame, subject_value):
        return eval_body_node(body_node, frame, eval_func)

@contextmanager
def temporary_subject(frame: Frame, dot: Any) -> Iterator[None]:
    prev = frame.dot
    frame.dot = dot

    try:
        yield
    finally:
        frame.dot = prev

@contextmanager
def temporary_bindings(frame: Frame, bindings: dict[str, Any]) -> Iterator[None]:
    records: list[tuple[Frame, str, Any | None, bool]] = []

    for name, value in bindings.items():
        target: Frame | None = frame

        while target is not None and name not in target.vars:
            target = getattr(target, "parent", None)

        if target is None:
            frame.define(name, value)
            records.append((frame, name, None, False))
        else:
            prev = target.vars[name]
            target.vars[name] = value
            records.append((target, name, prev, True))

    try:
        yield
    finally:
        for target, name, prev, existed in reversed(records):
            if existed:
                target.vars[name] = prev
            else:
                target.vars.pop(name, None)

def eval_defer_stmt(children: List[Any], frame: Frame, eval_func: EvalFunc) -> Any:
    if not children:
        raise ShakarRuntimeError("Malformed defer statement")

    idx = 0
    label = None

    if is_tree_node(children[0]) and tree_label(children[0]) == 'deferlabel':
        label = _expect_ident_token(children[0].children[0], "Defer label")
        idx += 1

    if idx >= len(children):
        raise ShakarRuntimeError("Missing deferred body")

    body_wrapper = children[idx]
    idx += 1
    deps: List[str] = []

    if idx < len(children):
        deps_node = children[idx]

        if is_tree_node(deps_node) and tree_label(deps_node) == 'deferdeps':
            deps = [
                _expect_ident_token(tok, "Defer dependency")
                for tok in tree_children(deps_node)
                if is_token_node(tok)
            ]
            idx += 1

    if idx != len(children):
        raise ShakarRuntimeError("Unexpected defer statement shape")

    body_kind = 'block' if is_tree_node(body_wrapper) and tree_label(body_wrapper) == 'deferblock' else 'call'
    payload = body_wrapper

    if body_kind == 'block':
        payload = body_wrapper.children[0] if is_tree_node(body_wrapper) and body_wrapper.children else Tree('inlinebody', [])

    saved_dot = frame.dot
    source = getattr(frame, 'source', None)

    def thunk() -> None:
        child_frame = Frame(parent=frame, dot=saved_dot, source=source)
        push_defer_scope(child_frame)

        try:
            if body_kind == 'block':
                eval_inline_body(payload, child_frame, eval_func)
            else:
                eval_func(payload, child_frame)
        finally:
            pop_defer_scope(child_frame)

    schedule_defer(frame, thunk, label=label, deps=deps)

    return ShkNull()
