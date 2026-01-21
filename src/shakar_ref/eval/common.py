from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, TypeAlias
import re

from ..tree import Tree, Tok
from ..token_types import TT

from ..types import (
    Frame,
    ShkBool,
    ShkEnvVar,
    ShkNumber,
    ShkPath,
    ShkString,
    ShkRegex,
    ShkDuration,
    ShkSize,
    ShakarRuntimeError,
    ShakarTypeError,
    ShkValue,
    ShkNull,
)
from ..tree import Node, is_token, is_tree, node_meta, tree_children, tree_label
from ..tree import token_kind
from ..utils import envvar_value_by_name, stringify

SourceSpan: TypeAlias = tuple[int, int] | tuple[None, None]

DURATION_UNITS: Dict[str, int] = {
    "nsec": 1,
    "usec": 1_000,
    "msec": 1_000_000,
    "sec": 1_000_000_000,
    "min": 60_000_000_000,
    "hr": 3_600_000_000_000,
    "day": 86_400_000_000_000,
    "wk": 604_800_000_000_000,
}

SIZE_UNITS: Dict[str, int] = {
    "b": 1,
    "kb": 1_000,
    "mb": 1_000_000,
    "gb": 1_000_000_000,
    "tb": 1_000_000_000_000,
    "kib": 1_024,
    "mib": 1_048_576,
    "gib": 1_073_741_824,
    "tib": 1_099_511_627_776,
}


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
    names, varargs, _defaults, _contracts, _spread_contracts = (
        extract_function_signature(params_node, context=context)
    )
    return names, varargs


def extract_param_defaults(
    params_node: Optional[Node], context: str = "parameter list"
) -> List[Optional[Node]]:
    _names, _varargs, defaults, _contracts, _spread_contracts = (
        extract_function_signature(params_node, context=context)
    )
    return defaults


def extract_function_signature(
    params_node: Optional[Node], context: str = "parameter list"
) -> Tuple[
    List[str],
    List[int],
    List[Optional[Node]],
    Dict[str, Node],
    Dict[str, Node],
]:
    if params_node is None:
        return [], [], [], {}, {}

    names: List[str] = []
    varargs: List[int] = []
    defaults: List[Optional[Node]] = []
    contracts: Dict[str, Node] = {}
    spread_contracts: Dict[str, Node] = {}
    param_index = 0

    def fail(message: str) -> None:
        raise ShakarRuntimeError(message)

    for p in tree_children(params_node):
        if is_token(p) and token_kind(p) == "COMMA":
            continue

        if is_tree(p) and tree_label(p) == "param_spread":
            children = tree_children(p)
            if not children:
                raise ShakarRuntimeError(
                    f"Unsupported parameter node in {context}: {p}"
                )
            param_name = ident_token_value(children[0])
            if param_name is None:
                raise ShakarRuntimeError(
                    f"Unsupported parameter node in {context}: {p}"
                )
            names.append(param_name)
            varargs.append(param_index)
            defaults.append(None)
            contract_expr = _param_contract_expr(p)
            if contract_expr is not None:
                spread_contracts[param_name] = contract_expr
            param_index += 1
            continue

        name = ident_token_value(p)
        if name is not None:
            names.append(name)
            defaults.append(None)
            param_index += 1
            continue

        if is_tree(p) and tree_label(p) == "param":
            children = tree_children(p)
            if not children:
                raise ShakarRuntimeError(
                    f"Unsupported parameter node in {context}: {p}"
                )
            param_name = ident_token_value(children[0])
            if param_name is None:
                raise ShakarRuntimeError(
                    f"Unsupported parameter node in {context}: {p}"
                )
            names.append(param_name)
            defaults.append(param_default_expr(p, on_error=fail))
            contract_expr = _param_contract_expr(p)
            if contract_expr is not None:
                contracts[param_name] = contract_expr
            param_index += 1
            continue

        raise ShakarRuntimeError(f"Unsupported parameter node in {context}: {p}")

    return names, varargs, defaults, contracts, spread_contracts


def _param_contract_expr(node: Tree) -> Optional[Node]:
    for child in tree_children(node):
        if is_tree(child) and tree_label(child) == "contract":
            contract_children = tree_children(child)
            if contract_children:
                return contract_children[0]
            return None
    return None


def param_default_expr(
    node: Node, *, on_error: Optional[Callable[[str], None]] = None
) -> Optional[Node]:
    if not is_tree(node) or tree_label(node) != "param":
        return None

    default: Optional[Node] = None

    for child in tree_children(node):
        if is_token(child) and child.type == TT.IDENT:
            continue
        if is_tree(child) and tree_label(child) == "contract":
            continue
        if default is None:
            default = child
            continue
        if on_error is not None:
            on_error("Parameter has multiple default values")
        break

    return default


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
    """Convert NUMBER token to ShkNumber.

    Handles decimal, base-prefixed (0b/0o/0x), and underscore-separated literals.
    """
    raw = token.value
    if isinstance(raw, (int, float)):
        return ShkNumber(float(raw))

    text = str(raw)
    clean = text.replace("_", "")  # strip underscore separators

    if text.startswith(("0b", "0o", "0x")):
        base = {"0b": 2, "0o": 8, "0x": 16}[text[:2]]
        return ShkNumber(float(int(clean[2:], base)))

    return ShkNumber(float(clean))


def token_duration(token: Tok, _: None) -> ShkDuration:
    raw, nanos = token.value
    return ShkDuration(nanos=nanos, display=raw)


def token_size(token: Tok, _: None) -> ShkSize:
    raw, byte_count = token.value
    return ShkSize(byte_count=byte_count, display=raw)


def token_string(token: Tok, _: None) -> ShkString:
    raw = token.value
    token_type = token.type

    if token_type == TT.RAW_HASH_STRING:
        return ShkString(raw[5:-2])

    if token_type == TT.RAW_STRING:
        return ShkString(raw[4:-1])

    return ShkString(unescape_string_literal(strip_prefixed_quotes(str(raw), "")))


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


def strip_prefixed_quotes(raw: str, prefix: str) -> str:
    if raw.startswith(f'{prefix}"') and raw.endswith('"'):
        return raw[len(prefix) + 1 : -1]
    if raw.startswith(f"{prefix}'") and raw.endswith("'"):
        return raw[len(prefix) + 1 : -1]
    return raw


def unescape_string_literal(text: str) -> str:
    """Interpret standard escapes in a string literal."""
    # Keep this in sync with lexer doc expectations; raw strings bypass this.
    escaped = {
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "t": "\t",
        "r": "\r",
        "0": "\0",
        '"': '"',
        "'": "'",
        "\\": "\\",
    }
    hex_digits = "0123456789abcdefABCDEF"
    out: List[str] = []
    i = 0
    length = len(text)

    while i < length:
        ch = text[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue

        if i + 1 >= length:
            raise ShakarRuntimeError("Unterminated escape in string literal")

        nxt = text[i + 1]
        if nxt in escaped:
            out.append(escaped[nxt])
            i += 2
            continue

        if nxt == "u" and i + 2 < length and text[i + 2] == "{":
            j = i + 3
            while j < length and text[j] != "}":
                j += 1

            if j >= length:
                raise ShakarRuntimeError(
                    "Unterminated unicode escape in string literal"
                )

            hex_raw = text[i + 3 : j]
            if not hex_raw:
                raise ShakarRuntimeError("Empty unicode escape in string literal")

            hex_clean = hex_raw.replace("_", "")
            if not hex_clean or any(ch not in hex_digits for ch in hex_clean):
                raise ShakarRuntimeError(
                    f"Invalid unicode escape in string literal: \\u{{{hex_raw}}}"
                )

            codepoint = int(hex_clean, 16)
            if codepoint > 0x10FFFF:
                raise ShakarRuntimeError(
                    f"Unicode escape out of range: \\u{{{hex_raw}}}"
                )

            out.append(chr(codepoint))
            i = j + 1
            continue

        if nxt == "x":
            if i + 3 >= length:
                raise ShakarRuntimeError("Unterminated hex escape in string literal")

            hex_raw = text[i + 2 : i + 4]
            if len(hex_raw) != 2 or any(ch not in hex_digits for ch in hex_raw):
                raise ShakarRuntimeError(
                    f"Invalid hex escape in string literal: \\x{hex_raw}"
                )

            out.append(chr(int(hex_raw, 16)))
            i += 4
            continue

        raise ShakarRuntimeError(f"Unknown escape in string literal: \\{nxt}")

    return "".join(out)


def unwrap_noanchor(op: Tree) -> Tuple[Tree, str]:
    """Unwrap noanchor wrapper if present, return (inner_op, label)."""
    label = tree_label(op)
    if label == "noanchor":
        inner = op.children[0]
        return inner, tree_label(inner)
    return op, label


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
