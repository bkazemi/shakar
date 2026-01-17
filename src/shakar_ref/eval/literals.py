from __future__ import annotations

import os
import shlex
from typing import Callable, List

from ..tree import Tree

from ..runtime import (
    Frame,
    ShkArray,
    ShkBool,
    ShkCommand,
    ShkEnvVar,
    ShkNull,
    ShkObject,
    ShkPath,
    ShkString,
    ShkValue,
    ShakarRuntimeError,
    ShakarTypeError,
)
from ..tree import node_meta, tree_children, tree_label, is_token, is_tree
from .helpers import eval_anchor_scoped
from .loops import _iterable_values
from .common import stringify

EvalFunc = Callable[[Tree, Frame], ShkValue]


def eval_keyword_literal(node: Tree) -> ShkValue:
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


def eval_array_literal(node: Tree, frame: Frame, eval_func: EvalFunc) -> ShkArray:
    items: List[ShkValue] = []

    for child in tree_children(node):
        if is_tree(child) and tree_label(child) == "spread":
            spread_expr = child.children[0] if child.children else None
            if spread_expr is None:
                raise ShakarRuntimeError("Malformed spread element")
            spread_val = eval_anchor_scoped(spread_expr, frame, eval_func)
            if isinstance(spread_val, ShkObject):
                raise ShakarTypeError("Cannot spread object into array literal")
            items.extend(_iterable_values(spread_val))
            continue

        items.append(eval_anchor_scoped(child, frame, eval_func))

    return ShkArray(items)


def eval_string_interp(node: Tree, frame: Frame, eval_func: EvalFunc) -> ShkString:
    parts: List[str] = []

    for part in tree_children(node):
        if is_token(part):
            parts.append(part.value)
            continue

        if tree_label(part) == "string_interp_expr":
            expr_node = part.children[0] if tree_children(part) else None
            if expr_node is None:
                raise ShakarRuntimeError("Empty interpolation expression")

            value = eval_anchor_scoped(expr_node, frame, eval_func)
            parts.append(stringify(value))
            continue

        raise ShakarRuntimeError("Unexpected node in string interpolation literal")

    return ShkString("".join(parts))


def eval_shell_string(node: Tree, frame: Frame, eval_func: EvalFunc) -> ShkCommand:
    parts: List[str] = []

    for part in tree_children(node):
        if is_token(part):
            parts.append(part.value)
            continue

        label = tree_label(part)
        expr_node = part.children[0] if tree_children(part) else None

        if expr_node is None:
            raise ShakarRuntimeError("Empty interpolation expression")

        value = eval_anchor_scoped(expr_node, frame, eval_func)

        if label == "shell_interp_expr":
            rendered = _render_shell_safe(value)
        elif label == "shell_raw_expr":
            rendered = _render_shell_raw(value)
        else:
            raise ShakarRuntimeError("Unexpected node in shell string literal")

        parts.append(rendered)
    return ShkCommand(parts)


def eval_path_interp(node: Tree, frame: Frame, eval_func: EvalFunc) -> ShkPath:
    parts: List[str] = []

    for part in tree_children(node):
        if is_token(part):
            parts.append(part.value)
            continue

        if tree_label(part) == "path_interp_expr":
            expr_node = part.children[0] if tree_children(part) else None
            if expr_node is None:
                raise ShakarRuntimeError("Empty interpolation expression")

            value = eval_anchor_scoped(expr_node, frame, eval_func)
            parts.append(stringify(value))
            continue

        raise ShakarRuntimeError("Unexpected node in path interpolation literal")

    return ShkPath("".join(parts))


def eval_env_string(node: Tree, frame: Frame, eval_func: EvalFunc) -> ShkEnvVar:
    """Evaluate simple env string (no interpolation)."""
    children = tree_children(node)
    if not children or not is_token(children[0]):
        raise ShakarRuntimeError("Malformed env_string node")

    var_name = children[0].value
    return ShkEnvVar(var_name)


def eval_env_interp(node: Tree, frame: Frame, eval_func: EvalFunc) -> ShkEnvVar:
    """Evaluate interpolated env string."""
    parts: List[str] = []

    for part in tree_children(node):
        if is_token(part):
            parts.append(part.value)
            continue

        if tree_label(part) == "env_interp_expr":
            expr_node = part.children[0] if tree_children(part) else None
            if expr_node is None:
                raise ShakarRuntimeError("Empty interpolation expression")
            value = eval_anchor_scoped(expr_node, frame, eval_func)
            parts.append(stringify(value))
            continue

        raise ShakarRuntimeError("Unexpected node in env string")

    var_name = "".join(parts)
    return ShkEnvVar(var_name)


def _render_shell_safe(value: ShkValue) -> str:
    if isinstance(value, ShkArray):
        return " ".join(_quote_shell_value(item) for item in value.items)
    return _quote_shell_value(value)


def _render_shell_raw(value: ShkValue) -> str:
    if isinstance(value, ShkArray):
        return " ".join(stringify(item) for item in value.items)
    return stringify(value)


def _quote_shell_value(value: ShkValue) -> str:
    rendered = stringify(value)
    return shlex.quote(rendered)
