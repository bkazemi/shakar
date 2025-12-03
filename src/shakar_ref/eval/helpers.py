from __future__ import annotations

from typing import Optional

from ..runtime import Frame, ShkArray, ShkBool, ShkNull, ShkNumber, ShkObject, ShkString, ShkValue
from ..tree import Node, is_token, is_tree, tree_label
from .common import token_kind as _token_kind

def is_truthy(val: ShkValue) -> bool:
    match val:
        case ShkBool(value=b):
            return b
        case ShkNull():
            return False
        case ShkNumber(value=num):
            return num != 0
        case ShkString(value=s):
            return bool(s)
        case ShkArray(items=items):
            return bool(items)
        case ShkObject(slots=slots):
            return bool(slots)
        case _:
            return True

def retargets_anchor(node: Node) -> bool:
    if is_token(node):
        return _token_kind(node) == 'IDENT'

    if is_tree(node):
        label = tree_label(node)

        if label == 'expr' and node.children:
            return retargets_anchor(node.children[0])

        return label not in {
            'implicit_chain',
            'subject',
            'group',
            'no_anchor',
            'literal',
            'bind',
        }
    return True

def current_function_frame(frame: Frame) -> Optional[Frame]:
    """Walk parents to find the nearest function-call frame marker."""
    cur: Optional[Frame] = frame

    while cur is not None:
        if cur.is_function_frame():
            return cur

        cur = getattr(cur, "parent", None)

    return None
