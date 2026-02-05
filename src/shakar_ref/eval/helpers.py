from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, FrozenSet, Iterator, Optional

from ..runtime import (
    Frame,
    ShkArray,
    ShkFan,
    ShkBool,
    ShkCommand,
    ShkDecorator,
    DecoratorConfigured,
    DecoratorContinuation,
    Descriptor,
    ShkDuration,
    ShkEnvVar,
    ShkChannel,
    ShkFn,
    ShkNil,
    ShkNumber,
    ShkObject,
    ShkOptional,
    ShkPath,
    ShkRegex,
    ShkSelector,
    ShkSize,
    ShkString,
    ShkType,
    ShkUnion,
    ShkValue,
    BoundMethod,
    BoundCallable,
    BuiltinMethod,
    StdlibFunction,
    ShakarTypeError,
    ShakarCancelledError,
    ShkModule,
)
from ..tree import Node, is_token, is_tree, tree_label
from ..utils import envvar_value_by_name
from .common import token_kind as _token_kind


def is_truthy(val: ShkValue) -> bool:
    match val:
        case ShkBool(value=b):
            return b
        case ShkNil():
            return False
        case ShkNumber(value=num):
            return num != 0
        case ShkDuration(nanos=nanos):
            return nanos != 0
        case ShkSize(byte_count=byte_count):
            return byte_count != 0
        case ShkString(value=s):
            return bool(s)
        case ShkArray(items=items) | ShkFan(items=items):
            return bool(items)
        case ShkModule(slots=slots) | ShkObject(slots=slots):
            return bool(slots)
        case ShkPath(value=path):
            return bool(path)
        case ShkCommand() as cmd:
            return bool(cmd.render())
        case ShkEnvVar(name=name):
            return bool(envvar_value_by_name(name))
        case (
            ShkRegex()
            | ShkSelector()
            | ShkChannel()
            | ShkFn()
            | ShkDecorator()
            | DecoratorConfigured()
            | DecoratorContinuation()
            | Descriptor()
            | BoundMethod()
            | BoundCallable()
            | BuiltinMethod()
            | StdlibFunction()
            | ShkType()
            | ShkOptional()
            | ShkUnion()
        ):
            raise ShakarTypeError("Non-boolean value used in condition")
        case _:
            raise ShakarTypeError("Unknown value type in condition")


def retargets_anchor(node: Node) -> bool:
    if is_token(node):
        return _token_kind(node) == "IDENT"

    if is_tree(node):
        label = tree_label(node)

        if label == "expr" and node.children:
            return retargets_anchor(node.children[0])

        return label not in {
            "implicit_chain",
            "subject",
            "group",
            "no_anchor",
            "literal",
            "bind",
        }
    return True


@contextmanager
def isolate_anchor_override(frame: Frame) -> Iterator[None]:
    """Prevent pending anchor overrides from leaking across nested evaluations."""
    saved = frame.pending_anchor_override
    frame.pending_anchor_override = None
    try:
        yield
    finally:
        frame.pending_anchor_override = saved


def eval_anchor_scoped(
    node: Node, frame: Frame, eval_func: Callable[[Node, Frame], ShkValue]
) -> ShkValue:
    """Evaluate a node without leaking anchor overrides into surrounding context."""
    with isolate_anchor_override(frame):
        return eval_func(node, frame)


def current_function_frame(frame: Frame) -> Optional[Frame]:
    """Walk parents to find the nearest function-call frame marker."""
    cur: Optional[Frame] = frame

    while cur is not None:
        if cur.is_function_frame():
            return cur

        cur = getattr(cur, "parent", None)

    return None


def name_in_current_frame(frame: Frame, name: str) -> bool:
    """Check only the current frame (vars + let scopes) for a binding."""
    return frame.has_let_name(name) or name in frame.vars


def find_emit_target(frame: Frame) -> Optional[ShkValue]:
    cur: Optional[Frame] = frame
    while cur is not None:
        if cur.emit_target is not None:
            return cur.emit_target
        cur = getattr(cur, "parent", None)
    return None


def collect_scope_names(frame: Frame, stop_at: Optional[Frame]) -> set[str]:
    """Collect names visible from frame up to an optional boundary frame."""
    names: set[str] = set()
    cur: Optional[Frame] = frame
    while cur is not None:
        if cur.hoisted_names:
            names.update(cur.hoisted_names)
        for scope in cur.all_let_scopes():
            names.update(scope.keys())
        names.update(cur.vars.keys())
        if stop_at is not None and cur is stop_at:
            break
        cur = cur.parent
    return names


def snapshot_visible_names(frame: Frame) -> FrozenSet[str]:
    """Freeze the visible name set at definition time for lexical scoping."""
    return frozenset(collect_scope_names(frame, stop_at=None))


def find_frozen_scope_frame(frame: Frame) -> Optional[Frame]:
    """Find the nearest frame that marks a frozen lexical boundary."""
    cur: Optional[Frame] = frame
    while cur is not None:
        if cur.frozen_scope_names is not None:
            return cur
        cur = cur.parent
    return None


def closure_frame(frame: Frame) -> Frame:
    """Create a closure frame that captures the current emit target."""
    emit_target = find_emit_target(frame)
    closure = Frame(parent=frame, dot=None, emit_target=emit_target)
    closure.capture_let_scopes(frame.all_let_scopes())
    closure.frozen_scope_names = snapshot_visible_names(frame)
    return closure


def check_cancel(frame: Frame) -> None:
    token = getattr(frame, "cancel_token", None)
    if token is not None and token.cancelled():
        raise ShakarCancelledError("Spawn task cancelled")
