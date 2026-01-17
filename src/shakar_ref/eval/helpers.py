from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Optional

from ..runtime import (
    Frame,
    ShkArray,
    ShkBool,
    ShkCommand,
    ShkDecorator,
    DecoratorConfigured,
    DecoratorContinuation,
    Descriptor,
    ShkDuration,
    ShkEnvVar,
    ShkFn,
    ShkNull,
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
    BuiltinMethod,
    StdlibFunction,
    ShakarTypeError,
)
from ..tree import Node, is_token, is_tree, tree_label
from ..utils import envvar_value_by_name
from .common import token_kind as _token_kind


def is_truthy(val: ShkValue) -> bool:
    match val:
        case ShkBool(value=b):
            return b
        case ShkNull():
            return False
        case ShkNumber(value=num):
            return num != 0
        case ShkDuration(nanos=nanos):
            return nanos != 0
        case ShkSize(byte_count=byte_count):
            return byte_count != 0
        case ShkString(value=s):
            return bool(s)
        case ShkArray(items=items):
            return bool(items)
        case ShkObject(slots=slots):
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
            | ShkFn()
            | ShkDecorator()
            | DecoratorConfigured()
            | DecoratorContinuation()
            | Descriptor()
            | BoundMethod()
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


def find_emit_target(frame: Frame) -> Optional[ShkValue]:
    cur: Optional[Frame] = frame
    while cur is not None:
        if cur.emit_target is not None:
            return cur.emit_target
        cur = getattr(cur, "parent", None)
    return None


def closure_frame(frame: Frame) -> Frame:
    """Create a closure frame that captures the current emit target."""
    emit_target = find_emit_target(frame)
    return Frame(parent=frame, dot=None, emit_target=emit_target)
