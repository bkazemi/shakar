from __future__ import annotations

from typing import Callable, List, Optional, Tuple, TypeAlias
import re

from ..tree import Tree, Tok
from ..token_types import TT

from ..types import (
    Frame,
    ShkBool,
    ShkNumber,
    ShkPath,
    ShkString,
    ShkRegex,
    ShakarRuntimeError,
    ShakarTypeError,
    ShkValue,
    ShkNull,
)
from ..tree import Node, is_token, is_tree, node_meta, tree_children, tree_label
from ..tree import token_kind

SourceSpan: TypeAlias = tuple[int, int] | tuple[None, None]


def is_token_type(node: Node, kind: str) -> bool:
    return is_token(node) and token_kind(node) == kind


def expect_ident_token(node: Node, context: str) -> str:
    if is_token(node) and token_kind(node) in {"IDENT", "OVER"}:
        return str(node.value)

    kind = token_kind(node)
    val = getattr(node, "value", None)
    raise ShakarRuntimeError(f"{context} must be an identifier (got {kind}:{val})")


def ident_token_value(node: Node) -> Optional[str]:
    if is_token(node) and token_kind(node) in {"IDENT", "OVER"}:
        return str(node.value)

    return None


def extract_param_names(
    params_node: Optional[Node], context: str = "parameter list"
) -> Tuple[List[str], List[int]]:
    if params_node is None:
        return [], []

    names: List[str] = []
    varargs: List[int] = []
    param_index = 0

    for p in tree_children(params_node):
        if is_token(p) and token_kind(p) == "COMMA":
            continue

        if is_tree(p) and tree_label(p) == "param_spread":
            children = tree_children(p)
            if children:
                param_name = ident_token_value(children[0])
                if param_name is not None:
                    names.append(param_name)
                    varargs.append(param_index)
                    param_index += 1
                    continue

        name = ident_token_value(p)
        if name is not None:
            names.append(name)
            param_index += 1
            continue

        if is_tree(p) and tree_label(p) == "param":
            children = tree_children(p)
            if children:
                param_name = ident_token_value(children[0])
                if param_name is not None:
                    names.append(param_name)
                    param_index += 1
                    continue

        raise ShakarRuntimeError(f"Unsupported parameter node in {context}: {p}")

    return names, varargs


def is_literal_node(node: Node) -> bool:
    return not isinstance(node, (Tree, Tok))


def get_source_segment(node: Node, frame: Frame) -> Optional[str]:
    source = getattr(frame, "source", None)
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


def render_expr(node: Node) -> str:
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


def node_source_span(node: Node) -> SourceSpan:
    meta = node_meta(node)
    start = getattr(meta, "start_pos", None)
    end = getattr(meta, "end_pos", None)

    if start is not None and end is not None:
        return start, end

    if is_tree(node):
        child_spans = [node_source_span(child) for child in tree_children(node)]
        child_starts = [s for s, _ in child_spans if s is not None]
        child_ends = [e for _, e in child_spans if e is not None]

        if child_starts and child_ends:
            return min(child_starts), max(child_ends)

    return None, None


def require_number(value: ShkValue) -> ShkNumber:
    if not isinstance(value, ShkNumber):
        raise ShakarTypeError("Expected number")
    return value


def token_number(token: Tok, _: None) -> ShkNumber:
    return ShkNumber(float(token.value))


def token_string(token: Tok, _: None) -> ShkString:
    raw = token.value
    token_type = token.type

    if token_type == TT.RAW_HASH_STRING:
        return ShkString(raw[5:-2])

    if token_type == TT.RAW_STRING:
        return ShkString(raw[4:-1])

    return ShkString(strip_prefixed_quotes(str(raw), ""))


def token_path(token: Tok, _: None) -> ShkPath:
    # Path literals mirror string literal behavior: escapes are preserved.
    return ShkPath(strip_prefixed_quotes(str(token.value), "p"))


def _regex_flags(flags: str) -> tuple[int, bool]:
    py_flags = 0
    include_full = False

    for ch in flags:
        match ch:
            case "i":
                py_flags |= re.IGNORECASE
            case "m":
                py_flags |= re.MULTILINE
            case "s":
                py_flags |= re.DOTALL
            case "x":
                py_flags |= re.VERBOSE
            case "f":
                include_full = True
            case _:
                raise ShakarTypeError(f"Unknown regex flag '{ch}'")

    return py_flags, include_full


def token_regex(token: Tok, _: None) -> ShkRegex:
    value = token.value
    if not isinstance(value, tuple) or len(value) != 2:
        raise ShakarRuntimeError("Malformed regex literal")

    pattern, flags = value
    if not isinstance(pattern, str) or not isinstance(flags, str):
        raise ShakarRuntimeError("Malformed regex literal")

    py_flags, include_full = _regex_flags(flags)
    try:
        compiled = re.compile(pattern, py_flags)
    except re.error as exc:
        raise ShakarRuntimeError(f"Invalid regex: {exc}") from exc

    return ShkRegex(
        pattern=pattern, flags=flags, include_full=include_full, compiled=compiled
    )


def stringify(value: Optional[ShkValue]) -> str:
    if isinstance(value, ShkPath):
        return str(value)

    if isinstance(value, ShkString):
        return value.value

    if isinstance(value, ShkNumber):
        return str(value)

    if isinstance(value, ShkBool):
        return "true" if value.value else "false"

    if isinstance(value, ShkNull) or value is None:
        return "nil"

    return str(value)


def strip_prefixed_quotes(raw: str, prefix: str) -> str:
    if raw.startswith(f'{prefix}"') and raw.endswith('"'):
        return raw[len(prefix) + 1 : -1]
    if raw.startswith(f"{prefix}'") and raw.endswith("'"):
        return raw[len(prefix) + 1 : -1]
    return raw


def collect_free_identifiers(node: Node, callback: Callable[[str], None]) -> None:
    skip_nodes = {
        "field",
        "fieldsel",
        "fieldfan",
        "fieldlist",
        "key_ident",
        "key_string",
    }

    def walk(n: Node) -> None:
        if is_token(n):
            if token_kind(n) == "IDENT":
                callback(n.value)
            return

        if is_tree(n):
            if tree_label(n) == "amp_lambda":
                return

            if tree_label(n) in skip_nodes:
                return

            for ch in tree_children(n):
                walk(ch)

    walk(node)
