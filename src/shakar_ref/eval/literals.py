from __future__ import annotations

from typing import Any

from lark import Tree

from ..runtime import ShkBool, ShkNull, ShkString, ShakarRuntimeError
from ..tree import node_meta, tree_children, tree_label, is_token_node
from .common import stringify

def eval_keyword_literal(node: Tree) -> Any:
    meta = node_meta(node)
    if meta is None:
        raise ShakarRuntimeError("Missing metadata for literal")

    start = getattr(meta, "start_pos", None)
    end = getattr(meta, "end_pos", None)
    if start is None or end is None:
        raise ShakarRuntimeError("Missing source span for literal")

    width = end - start
    match width:
        case 3:
            return ShkNull()
        case 4:
            return ShkBool(True)
        case 5:
            return ShkBool(False)

    raise ShakarRuntimeError("Unknown literal")

def eval_string_interp(node: Tree, frame, eval_func) -> ShkString:
    parts: list[str] = []

    for part in tree_children(node):
        if is_token_node(part):
            parts.append(part.value)
            continue

        if tree_label(part) == 'string_interp_expr':
            expr_node = part.children[0] if tree_children(part) else None
            if expr_node is None:
                raise ShakarRuntimeError("Empty interpolation expression")

            value = eval_func(expr_node, frame)
            parts.append(stringify(value))
            continue

        raise ShakarRuntimeError("Unexpected node in string interpolation literal")

    return ShkString("".join(parts))
