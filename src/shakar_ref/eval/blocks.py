from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Callable, Iterator, List, Optional

from ..tree import Tree

from ..runtime import (
    BoundMethod,
    BoundCallable,
    BuiltinMethod,
    DeferEntry,
    DotValue,
    Frame,
    ShkDecorator,
    ShkFn,
    ShkNil,
    ShkValue,
    StdlibFunction,
    DecoratorContinuation,
    ShakarBreakSignal,
    ShakarContinueSignal,
    ShakarRuntimeError,
)
from ..tree import (
    Node,
    Tree,
    is_token,
    is_tree,
    tree_children,
    tree_label,
    is_inline_body,
)
from .common import expect_ident_token as _expect_ident_token, token_kind as _token_kind
from .helpers import is_truthy as _is_truthy
from .chains import call_value, eval_args_node_with_named

EvalFn = Callable[[Node, Frame], ShkValue]


def eval_program(
    children: List[Node],
    frame: Frame,
    eval_fn: EvalFn,
    allow_loop_control: bool = False,
) -> ShkValue:
    """Run a stmt list under a fresh defer scope, returning last value."""
    result: ShkValue = ShkNil()
    skip_tokens = {"SEMI", "_NL", "INDENT", "DEDENT"}
    push_defer_scope(frame)
    push_let_scope(frame)
    prev_hoisted = frame.hoisted_names
    hoisted = _collect_hoisted_defs(children, skip_tokens)
    if hoisted:
        if prev_hoisted:
            hoisted.update(prev_hoisted)
        frame.hoisted_names = hoisted

    try:
        try:
            # Identify the last non-skip child for discard tagging.
            real_children = [c for c in children if _token_kind(c) not in skip_tokens]
            last_real = real_children[-1] if real_children else None

            for child in children:
                if _token_kind(child) in skip_tokens:
                    continue

                # Tag non-final once expressions so lazy registration
                # knows this is a true statement-discard position.
                is_final = child is last_real
                if not is_final and is_tree(child) and tree_label(child) == "once_expr":
                    if child.attrs is None:
                        child.attrs = {}
                    child.attrs["discard"] = True

                result = eval_fn(child, frame)
                frame.pending_anchor_override = None  # clear stale override
        except ShakarBreakSignal:
            if allow_loop_control:
                raise
            raise ShakarRuntimeError("break outside of a loop") from None
        except ShakarContinueSignal:
            if allow_loop_control:
                raise
            raise ShakarRuntimeError("continue outside of a loop") from None
    finally:
        try:
            pop_defer_scope(frame)
        finally:
            pop_let_scope(frame)
        frame.hoisted_names = prev_hoisted

    return result


def _collect_hoisted_defs(children: List[Node], skip_tokens: set[str]) -> set[str]:
    """Collect function-like names declared in the current stmt list."""
    names: set[str] = set()
    for child in children:
        if is_token(child) and _token_kind(child) in skip_tokens:
            continue
        node = child
        if is_tree(node) and tree_label(node) == "stmt" and node.children:
            node = node.children[0]
        if not is_tree(node):
            continue
        label = tree_label(node)
        if label not in {"fndef", "decorator_def"}:
            continue
        first = node.children[0] if node.children else None
        if first and is_token(first) and _token_kind(first) == "IDENT":
            names.add(str(first.value))
    return names


def get_subject(frame: Frame) -> ShkValue:
    if frame.dot is None:
        raise ShakarRuntimeError("No subject available for '.'")

    return frame.dot


def push_defer_scope(frame: Frame) -> None:
    frame.push_defer_frame()


def pop_defer_scope(frame: Frame) -> None:
    entries = frame.pop_defer_frame()
    if entries:
        _run_defer_entries(entries)


def push_let_scope(frame: Frame) -> None:
    frame.push_let_scope()


def pop_let_scope(frame: Frame) -> None:
    frame.pop_let_scope()


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


def schedule_defer(
    frame: Frame,
    thunk: Callable[[], None],
    label: Optional[str] = None,
    deps: Optional[List[str]] = None,
) -> None:
    if not frame.has_defer_frame():
        raise ShakarRuntimeError("Cannot use defer outside of a block")

    defer_frame = frame.current_defer_frame()
    entry = DeferEntry(thunk=thunk, label=label, deps=list(deps or []))

    if label:
        for existing in defer_frame:
            if existing.label == label:
                raise ShakarRuntimeError(f"Duplicate defer handle '{label}'")

    defer_frame.append(entry)


def _eval_inline_body(
    node: Tree, frame: Frame, eval_fn: EvalFn, allow_loop_control: bool = False
) -> ShkValue:
    """Evaluate an inline body node (single statement or brace-enclosed block)."""
    for child in tree_children(node):
        if tree_label(child) == "stmtlist":
            return eval_program(
                child.children,
                frame,
                eval_fn,
                allow_loop_control=allow_loop_control,
            )

    if not tree_children(node):
        return ShkNil()

    push_let_scope(frame)
    try:
        return eval_fn(node.children[0], frame)
    finally:
        pop_let_scope(frame)


def _eval_indent_block(
    node: Tree, frame: Frame, eval_fn: EvalFn, allow_loop_control: bool = False
) -> ShkValue:
    """Evaluate an indented block body node."""
    return eval_program(
        node.children, frame, eval_fn, allow_loop_control=allow_loop_control
    )


def eval_guard(children: List[Node], frame: Frame, eval_fn: EvalFn) -> ShkValue:
    """Execute guard chains (`cond1, cond2: body`) in inline or indented form."""
    branches: List[Tree] = []
    else_body: Optional[Tree] = None

    for child in children:
        data = tree_label(child)
        if data == "guardbranch":
            branches.append(child)
        elif data == "body":
            else_body = child

    outer_dot = frame.dot

    for branch in branches:
        if not is_tree(branch) or len(branch.children) != 2:
            raise ShakarRuntimeError("Malformed guard branch")

        cond_node, body_node = branch.children

        with temporary_subject(frame, outer_dot):
            cond_val = eval_fn(cond_node, frame)

        if _is_truthy(cond_val):
            with temporary_subject(frame, outer_dot):
                return eval_body_node(body_node, frame, eval_fn)

    if else_body:
        with temporary_subject(frame, outer_dot):
            return eval_body_node(else_body, frame, eval_fn)

    frame.dot = outer_dot

    return ShkNil()


def eval_body_node(
    body_node: Node,
    frame: Frame,
    eval_fn: EvalFn,
    allow_loop_control: bool = False,
) -> ShkValue:
    label = tree_label(body_node) if is_tree(body_node) else None

    if label == "body":
        if is_inline_body(body_node):
            return _eval_inline_body(
                body_node, frame, eval_fn, allow_loop_control=allow_loop_control
            )
        return _eval_indent_block(
            body_node, frame, eval_fn, allow_loop_control=allow_loop_control
        )

    return eval_fn(body_node, frame)


def run_body_with_subject(
    body_node: Node,
    frame: Frame,
    subject_value: DotValue,
    eval_fn: EvalFn,
    extra_bindings: Optional[dict[str, ShkValue]] = None,
) -> ShkValue:
    if extra_bindings:
        with (
            temporary_subject(frame, subject_value),
            temporary_bindings(frame, extra_bindings),
        ):
            return eval_body_node(body_node, frame, eval_fn)

    with temporary_subject(frame, subject_value):
        return eval_body_node(body_node, frame, eval_fn)


@contextmanager
def temporary_subject(frame: Frame, dot: DotValue) -> Iterator[None]:
    prev = frame.dot
    frame.dot = dot

    try:
        yield
    finally:
        frame.dot = prev


@contextmanager
def temporary_bindings(frame: Frame, bindings: dict[str, ShkValue]) -> Iterator[None]:
    records: list[tuple[dict[str, ShkValue], str, ShkValue, bool]] = []
    target_scope = frame._let_scopes[-1] if frame._let_scopes else frame.vars

    for name, value in bindings.items():
        if name in target_scope:
            records.append((target_scope, name, target_scope[name], True))
        else:
            records.append((target_scope, name, ShkNil(), False))

        target_scope[name] = value

    try:
        yield
    finally:
        for scope, name, prev, existed in reversed(records):
            if existed:
                scope[name] = prev
            else:
                scope.pop(name, None)


@contextmanager
def temporary_emit_target(frame: Frame, target: ShkValue) -> Iterator[None]:
    prev = frame.emit_target
    frame.emit_target = target

    try:
        yield
    finally:
        frame.emit_target = prev


def _is_callable_emit_target(value: ShkValue) -> bool:
    return isinstance(
        value,
        (
            ShkFn,
            BoundMethod,
            BoundCallable,
            BuiltinMethod,
            StdlibFunction,
            DecoratorContinuation,
            ShkDecorator,
        ),
    )


def eval_call_stmt(n: Tree, frame: Frame, eval_fn: EvalFn) -> ShkValue:
    bind_tok: Optional[Node] = None
    target_node: Optional[Node] = None
    body_node: Optional[Tree] = None

    for child in tree_children(n):
        if is_tree(child) and tree_label(child) == "call_bind":
            bind_tok = child.children[0] if child.children else None
            continue

        if is_tree(child) and tree_label(child) == "body":
            body_node = child
            continue

        if target_node is None:
            target_node = child

    if target_node is None or body_node is None:
        raise ShakarRuntimeError("Malformed call statement")

    emit_target = eval_fn(target_node, frame)

    if not _is_callable_emit_target(emit_target):
        raise ShakarRuntimeError("call expects a callable emit target")

    bind_name = _expect_ident_token(bind_tok, "Call binder") if bind_tok else None
    ctx = (
        temporary_bindings(frame, {bind_name: emit_target})
        if bind_name
        else nullcontext()
    )

    with ctx:
        with temporary_emit_target(frame, emit_target):
            return eval_body_node(body_node, frame, eval_fn)


def eval_emit_expr(n: Tree, frame: Frame, eval_fn: EvalFn) -> ShkValue:
    emit_target = frame.get_emit_target()
    args_node = n.children[0] if n.children else None
    positional, named, interleaved = eval_args_node_with_named(
        args_node, frame, eval_fn
    )
    return call_value(
        emit_target,
        positional,
        frame,
        eval_fn,
        named=named,
        interleaved=interleaved,
        call_node=n,
    )


def eval_defer_stmt(children: List[Node], frame: Frame, eval_fn: EvalFn) -> ShkValue:
    if not children:
        raise ShakarRuntimeError("Malformed defer statement")

    idx = 0
    label = None

    if is_tree(children[0]) and tree_label(children[0]) == "deferlabel":
        label = _expect_ident_token(children[0].children[0], "Defer label")
        idx += 1

    if idx >= len(children):
        raise ShakarRuntimeError("Missing deferred body")

    body_wrapper = children[idx]
    idx += 1
    deps: List[str] = []

    if idx < len(children):
        deps_node = children[idx]

        if is_tree(deps_node) and tree_label(deps_node) == "deferdeps":
            deps = [
                _expect_ident_token(tok, "Defer dependency")
                for tok in tree_children(deps_node)
                if is_token(tok)
            ]
            idx += 1

    if idx != len(children):
        raise ShakarRuntimeError("Unexpected defer statement shape")

    body_kind = (
        "block"
        if is_tree(body_wrapper) and tree_label(body_wrapper) == "deferblock"
        else "call"
    )
    payload = body_wrapper

    if body_kind == "block":
        payload = (
            body_wrapper.children[0]
            if is_tree(body_wrapper) and body_wrapper.children
            else Tree("body", [], attrs={"inline": True})
        )

    saved_dot = frame.dot
    source = getattr(frame, "source", None)

    def thunk() -> None:
        child_frame = Frame(parent=frame, dot=saved_dot, source=source)
        push_defer_scope(child_frame)

        try:
            if body_kind == "block":
                eval_body_node(payload, child_frame, eval_fn)
            else:
                eval_fn(payload, child_frame)
        finally:
            pop_defer_scope(child_frame)

    schedule_defer(frame, thunk, label=label, deps=deps)

    return ShkNil()
