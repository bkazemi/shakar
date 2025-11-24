from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional

from lark import Tree, Token

from ..runtime import Frame, ShkBool, ShkNumber, ShkString, ShakarRuntimeError, ShakarTypeError
from ..tree import (
    is_token,
    is_tree,
    node_meta,
    tree_children,
    tree_label,
)

def token_kind(node: Any) -> Optional[str]:
    if not is_token(node):
        return None
    tok: Token = node
    return str(tok.type)

def is_token_type(node: Any, kind: str) -> bool:
    return is_token(node) and token_kind(node) == kind

def expect_ident_token(node: Any, context: str) -> str:
    if is_token(node) and token_kind(node) == 'IDENT':
        return str(node.value)

    raise ShakarRuntimeError(f"{context} must be an identifier")

def ident_token_value(node: Any) -> Optional[str]:
    if is_token(node) and token_kind(node) == 'IDENT':
        return str(node.value)

    return None

def is_literal_node(node: Any) -> bool:
    return not isinstance(node, (Tree, Token))

def get_source_segment(node: Any, frame: Frame) -> Optional[str]:
    source = getattr(frame, 'source', None)
    if source is None:
        return None

    meta = node_meta(node)
    if meta is None:
        return None

    start = getattr(meta, "start_pos", None)
    end = getattr(meta, "end_pos", None)
    if start is None or end is None:
        return None

    return str(source[start:end])

def render_expr(node: Any) -> str:
    if is_token(node):
        return str(node.value)

    if not is_tree(node):
        return str(node)

    parts: List[str] = []

    for child in tree_children(node):
        rendered = render_expr(child)
        if rendered:
            parts.append(rendered)

    return " ".join(parts)

def node_source_span(node: Any) -> tuple[int | None, int | None]:
    meta = node_meta(node)
    start = getattr(meta, 'start_pos', None)
    end = getattr(meta, 'end_pos', None)

    if start is not None and end is not None:
        return start, end

    if is_tree(node):
        child_spans = [node_source_span(child) for child in tree_children(node)]
        child_starts = [s for s, _ in child_spans if s is not None]
        child_ends = [e for _, e in child_spans if e is not None]

        if child_starts and child_ends:
            return min(child_starts), max(child_ends)

    return None, None

def require_number(value: Any) -> None:
    if not isinstance(value, ShkNumber):
        raise ShakarTypeError("Expected number")

def token_number(token: Token, _: Any) -> ShkNumber:
    return ShkNumber(float(token.value))

def token_string(token: Token, _: Any) -> ShkString:
    raw = token.value
    token_type = getattr(token, "type", "")

    if token_type == "RAW_HASH_STRING":
        return ShkString(raw[5:-2])

    if token_type == "RAW_STRING":
        return ShkString(raw[4:-1])

    if len(raw) >= 2 and ((raw[0] == '"' and raw[-1] == '"') or (raw[0] == "'" and raw[-1] == "'")):
        raw = raw[1:-1]

    return ShkString(raw)

def stringify(value: Any) -> str:
    if isinstance(value, ShkString):
        return value.value

    if isinstance(value, ShkNumber):
        return str(value.value)

    if isinstance(value, ShkBool):
        return "true" if value.value else "false"

    if value is None:
        return "nil"

    return str(value)

def collect_free_identifiers(node: Any, callback: Callable[[str], None]) -> None:
    skip_nodes = {'field', 'fieldsel', 'fieldfan', 'fieldlist', 'key_ident', 'key_string'}

    def walk(n: Any) -> None:
        if is_token(n):
            if token_kind(n) == 'IDENT':
                callback(n.value)
            return

        if is_tree(n):
            if tree_label(n) == 'amp_lambda':
                return

            if tree_label(n) in skip_nodes:
                return

            for ch in tree_children(n):
                walk(ch)

    walk(node)
