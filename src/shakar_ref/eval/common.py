from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeAlias

from ..tree import Tree, Tok
from ..token_types import TT

from ..types import (
    CallSite,
    DURATION_NANOS,
    Frame,
    SIZE_BYTES,
    ShkBool,
    ShkEnvVar,
    ShkNumber,
    ShkPath,
    ShkString,
    ShkRegex,
    ShkDuration,
    ShkSize,
    ShakarRuntimeError,
    ShakarAssertionError,
    ShakarTypeError,
    ShkValue,
    ShkNil,
)
from ..tree import Node, is_token, is_tree, node_meta, tree_children, tree_label
from ..tree import token_kind
from ..utils import envvar_value_by_name, stringify

SourceSpan: TypeAlias = tuple[int, int] | tuple[None, None]


MODIFIER_REGISTRY: Dict[str, Tuple[str, ...]] = {
    "wait": ("any", "all", "group"),
    "fan": ("par",),
}


def modifier_from_node(node: Node) -> Tuple[Optional[str], Optional[Tok]]:
    if not is_tree(node) or not node.attrs:
        return None, None

    name = node.attrs.get("modifier_name")
    tok = node.attrs.get("modifier_tok")

    if not isinstance(name, str):
        name = None
    if not isinstance(tok, Tok):
        tok = None

    return name, tok


def _modifier_suggestion(name: str, allowed: Sequence[str]) -> Optional[str]:
    matches = difflib.get_close_matches(name, allowed, n=1, cutoff=0.75)
    if matches:
        return matches[0]
    return None


def validate_modifier(construct: str, name: str, tok: Optional[Tok]) -> None:
    allowed = MODIFIER_REGISTRY.get(construct)
    if allowed is None:
        raise ShakarRuntimeError(f"unknown modifier construct '{construct}'")
    if name in allowed:
        return

    expected = ", ".join(allowed)
    message = f"unknown {construct} modifier '{name}'; " f"expected one of: {expected}"
    suggestion = _modifier_suggestion(name, allowed)
    if suggestion:
        message = f"{message}; did you mean '{suggestion}'?"

    error = ShakarRuntimeError(message)
    if tok and tok.line > 0:
        tok_text = "" if tok.value is None else str(tok.value)
        col = max(1, tok.column)
        error.shk_meta = SimpleNamespace(
            line=tok.line,
            column=col,
            end_line=tok.line,
            end_column=col + max(1, len(tok_text)),
        )
        error._augmented = True  # type: ignore[attr-defined]

    raise error


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


def assert_contract_match(
    name: str,
    value: ShkValue,
    contract_expr: Node,
    frame: Frame,
    eval_fn: Callable[[Node, Frame], ShkValue],
    *,
    message_prefix: str,
) -> None:
    from .match import match_structure

    contract_value = eval_fn(contract_expr, frame)
    if not match_structure(value, contract_value):
        raise ShakarAssertionError(
            f"{message_prefix}: {name} ~ {contract_value}, got {value}"
        )


@dataclass
class DestructField:
    name: str
    default: Optional[Node]
    contract: Optional[Node]


@dataclass
class DestructFields:
    fields: List[DestructField]


def extract_param_names(
    params_node: Optional[Node], context: str = "parameter list"
) -> Tuple[List[str], List[int]]:
    names, varargs, _defaults, _contracts, _spread_contracts, _destruct_fields = (
        extract_function_signature(params_node, context=context)
    )
    return names, varargs


def extract_param_defaults(
    params_node: Optional[Node], context: str = "parameter list"
) -> List[Optional[Node]]:
    _names, _varargs, defaults, _contracts, _spread_contracts, _destruct_fields = (
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
    List[Optional[DestructFields]],
]:
    if params_node is None:
        return [], [], [], {}, {}, []

    names: List[str] = []
    varargs: List[int] = []
    defaults: List[Optional[Node]] = []
    contracts: Dict[str, Node] = {}
    spread_contracts: Dict[str, Node] = {}
    destruct_fields: List[Optional[DestructFields]] = []
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
            destruct_fields.append(None)
            contract_expr = _param_contract_expr(p)
            if contract_expr:
                spread_contracts[param_name] = contract_expr
            param_index += 1
            continue

        name = ident_token_value(p)
        if name:
            names.append(name)
            defaults.append(None)
            destruct_fields.append(None)
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
            destruct_fields.append(None)
            contract_expr = _param_contract_expr(p)
            if contract_expr:
                contracts[param_name] = contract_expr
            param_index += 1
            continue

        if is_tree(p) and tree_label(p) == "param_destruct":
            synthetic_name = f"0__destruct{param_index}"
            names.append(synthetic_name)
            defaults.append(None)
            destruct_fields.append(_extract_param_destruct_fields(p, context=context))
            contract_expr = _param_contract_expr(p)
            if contract_expr:
                contracts[synthetic_name] = contract_expr
            param_index += 1
            continue

        raise ShakarRuntimeError(f"Unsupported parameter node in {context}: {p}")

    return names, varargs, defaults, contracts, spread_contracts, destruct_fields


def _extract_param_destruct_fields(node: Node, *, context: str) -> DestructFields:
    if not is_tree(node) or tree_label(node) != "param_destruct":
        raise ShakarRuntimeError(f"Unsupported parameter node in {context}: {node}")

    fields: List[DestructField] = []
    seen: set[str] = set()

    for child in tree_children(node):
        # Contract metadata belongs to the whole destructured object argument.
        # Field extraction metadata only comes from destruct_field children.
        if is_tree(child) and tree_label(child) == "contract":
            continue

        if not is_tree(child) or tree_label(child) != "destruct_field":
            child_desc = tree_label(child) if is_tree(child) else repr(child)
            raise ShakarRuntimeError(
                f"Unsupported child node {child_desc!r} in destructuring parameter in {context}"
            )

        field_children = tree_children(child)
        if not field_children:
            raise ShakarRuntimeError(
                f"Unsupported child node 'destruct_field' in destructuring parameter in {context}"
            )

        field_name = ident_token_value(field_children[0])
        if field_name is None:
            head_desc = (
                tree_label(field_children[0])
                if is_tree(field_children[0])
                else repr(field_children[0])
            )
            raise ShakarRuntimeError(
                f"Unsupported destructuring field name node {head_desc!r} in {context}"
            )
        if field_name in seen:
            raise ShakarRuntimeError(
                f"Duplicate destructuring field '{field_name}' in {context}"
            )
        seen.add(field_name)

        default_expr, contract_expr = _extract_destruct_field_metadata(
            field_children[1:], context=context
        )
        fields.append(
            DestructField(name=field_name, default=default_expr, contract=contract_expr)
        )

    if not fields:
        raise ShakarRuntimeError(
            f"Destructuring parameter requires at least one field in {context}"
        )

    return DestructFields(fields=fields)


def _extract_destruct_field_metadata(
    metadata_nodes: List[Node], *, context: str
) -> tuple[Optional[Node], Optional[Node]]:
    default_expr: Optional[Node] = None
    contract_expr: Optional[Node] = None

    for node in metadata_nodes:
        if is_tree(node) and tree_label(node) == "contract":
            contract_children = tree_children(node)
            expr = contract_children[0] if contract_children else None
            if contract_expr is not None:
                raise ShakarRuntimeError(
                    f"Duplicate destructuring field contract in {context}"
                )
            contract_expr = expr
            continue

        if default_expr is not None:
            raise ShakarRuntimeError(
                f"Duplicate destructuring field default in {context}"
            )
        default_expr = node

    return default_expr, contract_expr


def _param_contract_expr(node: Tree) -> Optional[Node]:
    if tree_label(node) not in {"param", "param_spread", "param_destruct"}:
        return None

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
        if on_error:
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


def node_first_token_pos(node: Node) -> tuple[Optional[int], Optional[int]]:
    if is_token(node):
        if node.line > 0:
            return node.line, max(1, node.column)
        return None, None
    if is_tree(node):
        for child in tree_children(node):
            line, col = node_first_token_pos(child)
            if line is not None and col is not None:
                return line, col
    return None, None


def _resolve_line_col(node: Node, frame: Frame) -> tuple[Optional[int], Optional[int]]:
    """Resolve line/col from a node using three fallback tiers: meta, first token, source span."""
    line = None
    col = None

    meta = node_meta(node)
    if meta:
        meta_line = getattr(meta, "line", None)
        meta_col = getattr(meta, "column", None)
        if isinstance(meta_line, int) and meta_line > 0:
            line = meta_line
        if isinstance(meta_col, int) and meta_col > 0:
            col = meta_col

    if line is None or col is None:
        tok_line, tok_col = node_first_token_pos(node)
        if line is None:
            line = tok_line
        if col is None:
            col = tok_col

    if line is None or col is None:
        start, _end = node_source_span(node)
        source = getattr(frame, "source", None)
        if start is not None and source is not None:
            line = source.count("\n", 0, start) + 1
            last_nl = source.rfind("\n", 0, start)
            col = start + 1 if last_nl == -1 else start - last_nl

    return line, col


def _line_col_from_offset(source: str, offset: int) -> tuple[int, int]:
    line = source.count("\n", 0, offset) + 1
    last_nl = source.rfind("\n", 0, offset)
    col = offset + 1 if last_nl == -1 else offset - last_nl
    return line, col


def _resolve_error_span(
    node: Node, frame: Frame
) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Resolve a best-effort source span as 1-based [line/col, end_line/end_col_exclusive]."""
    line, col = _resolve_line_col(node, frame)
    end_line = None
    end_col = None

    meta = node_meta(node)
    if meta:
        meta_end_line = getattr(meta, "end_line", None)
        meta_end_col = getattr(meta, "end_column", None)
        if isinstance(meta_end_line, int) and meta_end_line > 0:
            end_line = meta_end_line
        if isinstance(meta_end_col, int) and meta_end_col > 0:
            end_col = meta_end_col

    if end_line is not None and end_col is not None:
        return line, col, end_line, end_col

    source = getattr(frame, "source", None)
    start, end = node_source_span(node)
    if source is not None and start is not None and end is not None and end > start:
        if line is None or col is None:
            line, col = _line_col_from_offset(source, start)
        end_line, end_col = _line_col_from_offset(source, end)
        return line, col, end_line, end_col

    if is_token(node) and line is not None and col is not None:
        # DURATION/SIZE/REGEX tokens store tuples; lexeme is the first element
        val = node.value
        if isinstance(val, tuple):
            val = val[0]
        text = "" if val is None else str(val)
        width = max(1, len(text))
        return line, col, line, col + width

    if line is not None and col is not None:
        return line, col, line, col + 1

    return line, col, end_line, end_col


def callsite_from_node(name: str, node: Node, frame: Frame) -> CallSite:
    line, col = _resolve_line_col(node, frame)

    return CallSite(
        name=name, line=int(line or 0), column=int(col or 0), path=frame.source_path
    )


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


def _extract_units(raw: str, unit_map: Dict[str, int]) -> Tuple[str, ...]:
    """Extract all unit suffixes from a literal like '1hr30min', largest-first."""
    # Try longest unit names first to avoid partial matches (e.g. 'mb' before 'b')
    keys = sorted(unit_map.keys(), key=len, reverse=True)
    units: List[str] = []
    pos = 0
    while pos < len(raw):
        # Skip non-alpha (digits, underscores, dots, e/E, +/-)
        if not raw[pos].isalpha():
            pos += 1
            continue

        # Consume a matched unit atomically, even if it is a duplicate.
        matched: Optional[str] = None
        for u in keys:
            if raw[pos : pos + len(u)] == u:
                matched = u
                break

        if matched is None:
            # Skip unrecognized alpha (e.g. 'e' from scientific notation)
            pos += 1
            continue

        if matched not in units:
            units.append(matched)
        pos += len(matched)

    return tuple(sorted(units, key=lambda u: unit_map[u], reverse=True))


def token_duration(token: Tok, _: None) -> ShkDuration:
    raw, nanos = token.value
    units = _extract_units(raw, DURATION_NANOS)
    return ShkDuration(nanos=nanos, units=units or None)


def token_size(token: Tok, _: None) -> ShkSize:
    raw, byte_count = token.value
    units = _extract_units(raw, SIZE_BYTES)
    return ShkSize(byte_count=byte_count, units=units or None)


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
