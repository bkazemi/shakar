from __future__ import annotations

import shlex
from typing import Any

from ..tree import TreeNode

from ..runtime import ShkArray, ShkBool, ShkCommand, ShkNull, ShkString, ShakarRuntimeError
from ..tree import node_meta, tree_children, tree_label, is_token_node
from .common import stringify

def eval_keyword_literal(node: TreeNode) -> Any:
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

def eval_string_interp(node: TreeNode, frame, eval_func) -> ShkString:
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

def eval_shell_string(node: TreeNode, frame, eval_func) -> ShkCommand:
    parts: list[str] = []

    for part in tree_children(node):
        if is_token_node(part):
            parts.append(part.value)
            continue

        label = tree_label(part)
        expr_node = part.children[0] if tree_children(part) else None

        if expr_node is None:
            raise ShakarRuntimeError("Empty interpolation expression")

        value = eval_func(expr_node, frame)

        if label == 'shell_interp_expr':
            rendered = _render_shell_safe(value)
        elif label == 'shell_raw_expr':
            rendered = _render_shell_raw(value)
        else:
            raise ShakarRuntimeError("Unexpected node in shell string literal")

        parts.append(rendered)
    return ShkCommand(parts)

def _render_shell_safe(value: Any) -> str:
    if isinstance(value, ShkArray):
        return " ".join(_quote_shell_value(item) for item in value.items)
    return _quote_shell_value(value)

def _render_shell_raw(value: Any) -> str:
    if isinstance(value, ShkArray):
        return " ".join(stringify(item) for item in value.items)
    return stringify(value)

def _quote_shell_value(value: Any) -> str:
    rendered = stringify(value)
    return shlex.quote(rendered)
