"""
AST transformation utilities (Prune/ChainNormalize/ArgTidy) and helpers that
were previously defined inside parse_auto.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, TypeAlias

from .tree import Tree, Tok, Transformer, v_args
from .tree import Discard
from .token_types import TT

from .tree import (
    Node,
    is_tree,
    is_token,
    tree_label,
    tree_children,
)
from .parser_rd import parse_expr_fragment
from .eval.common import param_default_expr

InterpolationSegment: TypeAlias = tuple[str, str | Node]
_HOLE_PARAM_PREFIX = "0__hole"


def looks_like_offside(code: str) -> bool:
    # Heuristic: a colon at end-of-line followed by a newline, or leading indentation on a non-empty line
    lines = code.splitlines()

    for i, ln in enumerate(lines):
        if (
            ln.rstrip().endswith(":")
            and i < len(lines) - 1
            and lines[i + 1].startswith((" ", "\t"))
        ):
            return True

        if ln and (ln[0] == " " or ln[0] == "\t"):
            return True
    return False


def enforce_subject_scope(tree: Tree) -> None:
    """Validate that bare '.' only appears inside an anchor/binder context."""
    errors: List[str] = []

    def visit(node: Node, depth: int) -> None:
        if not is_tree(node):
            return

        label = tree_label(node)
        children = list(tree_children(node))

        if label == "subject":
            if depth == 0:
                errors.append("bare '.' outside a binder/anchor context")
            return
        if label == "bindexpr" and len(children) == 2:
            visit(children[0], depth)
            visit(children[1], depth + 1)
            return
        if label in {"hook"}:
            for ch in children:
                if is_tree(ch) and tree_label(ch) in {"body"}:
                    visit(ch, depth + 1)
                else:
                    visit(ch, depth)
            return
        if label in {"waitany_arm"}:
            for idx, ch in enumerate(children):
                if idx == 1 and is_tree(ch) and tree_label(ch) == "body":
                    visit(ch, depth + 1)
                else:
                    visit(ch, depth)
            return
        if label in {"waitany_timeout", "waitany_default"}:
            for idx, ch in enumerate(children):
                if (
                    idx == len(children) - 1
                    and is_tree(ch)
                    and tree_label(ch) == "body"
                ):
                    visit(ch, depth + 1)
                else:
                    visit(ch, depth)
            return
        if label in {"forsubject", "forindexed"} and children:
            for ch in children[:-1]:
                visit(ch, depth)
            visit(children[-1], depth + 1)
            return
        if label in {"lambdacall1", "lambdacalln", "stmtsubjectassign"} and children:
            for ch in children[:-1]:
                visit(ch, depth)
            visit(children[-1], depth + 1)
            return

        for ch in children:
            visit(ch, depth)

    visit(tree, 0)

    if errors:
        raise SyntaxError(errors[0])


class ChainNormalize(Transformer):
    @staticmethod
    def _fuse(items: List[Node]) -> List[Node]:
        out: List[Node] = []
        i = 0

        while i < len(items):
            node = items[i]

            # Check for field (or noanchor-wrapped field) followed by call
            is_field = is_tree(node) and tree_label(node) == "field"
            is_noanchor_field = (
                is_tree(node)
                and tree_label(node) == "noanchor"
                and is_tree(node.children[0])
                and tree_label(node.children[0]) == "field"
            )

            if (
                (is_field or is_noanchor_field)
                and i + 1 < len(items)
                and is_tree(items[i + 1])
            ):
                nxt = items[i + 1]

                if tree_label(nxt) == "call":
                    inner = node.children[0] if is_noanchor_field else node
                    name = inner.children[0]
                    args_node = (
                        Tree("args", [Tree("amp_lambda", nxt.children)])
                        if nxt.children
                        else Tree("args", [])
                    )
                    method = Tree("method", [name, args_node])
                    out.append(
                        Tree("noanchor", [method]) if is_noanchor_field else method
                    )
                    i += 2
                    continue

            out.append(node)
            i += 1

        return out

    def implicit_chain(self, c: List[Node]) -> Tree:
        return Tree("implicit_chain", self._fuse(c))

    def explicit_chain(self, c: List[Node]) -> Tree:
        return Tree("explicit_chain", self._fuse(c))


class Prune(Transformer):
    _fragment_parser: Optional[object] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compare_depth = 0

    # ---- unified object literal ----
    def object(self, c):
        return Tree("object", c)

    @v_args(meta=True)
    def literal(self, meta, c):
        if not c:
            node = Tree("literal", [])
            node.meta = meta

            return node

        node = c[0]

        if is_token(node):
            token_type = getattr(node, "type", None)

            if token_type in (TT.STRING, getattr(TT, "STRING", None)):
                return self._transform_string_token(node)

            if token_type == TT.PATH_STRING:
                return self._transform_path_string_token(node)

            if token_type == TT.SHELL_STRING:
                return self._transform_shell_string_token(node)

            if token_type == TT.SHELL_BANG_STRING:
                return self._transform_shell_bang_token(node)

            if token_type == TT.ENV_STRING:
                return self._transform_env_string_token(node)
        return node

    # Canonicalize fields to a single node shape: obj_field(key, value)
    def obj_field_ident(self, c):
        return Tree("obj_field", [Tree("key_ident", [c[0]]), c[1]])

    def obj_field_string(self, c):
        return Tree("obj_field", [Tree("key_string", [c[0]]), c[1]])

    def obj_field_expr(self, c):
        return Tree("obj_field", [Tree("key_expr", [c[0]]), c[1]])

    def obj_field_optional(self, c):
        """Transform optional field: key?: value into key: Optional(value)"""
        key = c[0]
        value = c[1]
        # Wrap value in Optional() call
        # The correct AST structure is: explicit_chain(IDENT, call(args(...)))
        optional_call = Tree(
            "explicit_chain",
            [Tok(TT.IDENT, "Optional"), Tree("call", [Tree("args", [value])])],
        )
        return Tree("obj_field", [key, optional_call])

    def obj_sep(self, _c):
        return Discard

    # ---- string interpolation ----
    def _transform_string_token(self, token: Tok) -> Node:
        raw = token.value
        if len(raw) < 2:
            return token

        body = raw[1:-1]
        if "{" not in body:
            return token

        segments = self._split_interpolation_segments(body, token)
        if not segments:
            return token

        nodes: List[Node] = []

        for kind, payload in segments:
            if kind == "text":
                if payload:
                    nodes.append(Tok(TT.STRING, payload))
                continue

            nodes.append(Tree("string_interp_expr", [payload]))

        result = Tree("string_interp", nodes)

        meta = getattr(token, "meta", None)
        if meta is not None:
            result.meta = meta
        return result

    def _transform_shell_string_token(self, token: Tok) -> Tree:
        body = token.value
        segments = self._split_shell_interpolation_segments(body, token)
        nodes: List[Node] = []

        for kind, payload in segments:
            if kind == "text":
                if payload:
                    nodes.append(Tok(TT.STRING, payload))
                continue

            if kind == "expr":
                nodes.append(Tree("shell_interp_expr", [payload]))
                continue

            if kind == "raw_expr":
                nodes.append(Tree("shell_raw_expr", [payload]))
                continue

            raise SyntaxError("Unknown shell string segment")

        result = Tree("shell_string", nodes)

        meta = getattr(token, "meta", None)
        if meta is not None:
            result.meta = meta
        return result

    def _transform_shell_bang_token(self, token: Tok) -> Tree:
        """Transform eager shell string token into a shell_bang node."""
        shell_tree = self._transform_shell_string_token(token)
        result = Tree("shell_bang", [shell_tree])
        meta = getattr(token, "meta", None)
        if meta is not None:
            result.meta = meta
        return result

    def _transform_path_string_token(self, token: Tok) -> Node:
        raw = token.value

        if raw.startswith('p"') and raw.endswith('"'):
            body = raw[2:-1]
        elif raw.startswith("p'") and raw.endswith("'"):
            body = raw[2:-1]
        else:
            body = raw

        if "{" not in body:
            return token

        segments = self._split_interpolation_segments(body, token)
        if not segments:
            return token

        nodes: List[Node] = []

        for kind, payload in segments:
            if kind == "text":
                if payload:
                    nodes.append(Tok(TT.STRING, payload))
                continue

            nodes.append(Tree("path_interp_expr", [payload]))

        result = Tree("path_interp", nodes)
        meta = getattr(token, "meta", None)
        if meta is not None:
            result.meta = meta
        return result

    def _transform_env_string_token(self, token: Tok) -> Node:
        """Transform env string token, handling interpolation."""
        raw = token.value

        # Strip prefix and quotes: env"..." -> body
        if raw.startswith('env"') and raw.endswith('"'):
            body = raw[4:-1]
        elif raw.startswith("env'") and raw.endswith("'"):
            body = raw[4:-1]
        else:
            body = raw

        if "{" not in body:
            # No interpolation - simple env_string
            result = Tree("env_string", [Tok(TT.STRING, body)])
            meta = getattr(token, "meta", None)
            if meta is not None:
                result.meta = meta
            return result

        segments = self._split_interpolation_segments(body, token)
        if not segments:
            result = Tree("env_string", [Tok(TT.STRING, body)])
            meta = getattr(token, "meta", None)
            if meta is not None:
                result.meta = meta
            return result

        nodes: List[Node] = []
        for kind, payload in segments:
            if kind == "text":
                if payload:
                    nodes.append(Tok(TT.STRING, payload))
                continue
            nodes.append(Tree("env_interp_expr", [payload]))

        result = Tree("env_interp", nodes)
        meta = getattr(token, "meta", None)
        if meta is not None:
            result.meta = meta
        return result

    def SHELL_STRING(self, token: Tok) -> Tree:
        return self._transform_shell_string_token(token)

    def SHELL_BANG_STRING(self, token: Tok) -> Tree:
        return self._transform_shell_bang_token(token)

    def _split_interpolation_segments(
        self, text: str, token: Tok
    ) -> List[InterpolationSegment]:
        parts: List[InterpolationSegment] = []
        literal: List[str] = []
        index = 0
        length = len(text)
        saw_expr = False

        while index < length:
            ch = text[index]

            # Treat \u{...} as a literal escape so braces don't trigger interpolation.
            escape, next_index = self._try_scan_unicode_escape(text, index)
            if escape is not None:
                literal.append(escape)
                index = next_index
                continue

            if ch == "{":
                if index + 1 < length and text[index + 1] == "{":
                    literal.append("{")
                    index += 2
                    continue

                expr_text, next_index = self._extract_interpolation_expr(
                    text, index + 1, token
                )

                if literal:
                    parts.append(("text", "".join(literal)))
                    literal.clear()

                expr_node = self._parse_interpolation_expr(expr_text)
                parts.append(("expr", expr_node))
                saw_expr = True
                index = next_index
                continue
            elif ch == "}" and index + 1 < length and text[index + 1] == "}":
                literal.append("}")
                index += 2
                continue
            elif ch == "}":
                raise SyntaxError("Unescaped '}' in string interpolation literal")
            else:
                literal.append(ch)
                index += 1

        if not saw_expr:
            return []

        if literal:
            parts.append(("text", "".join(literal)))
        return parts

    def _split_shell_interpolation_segments(
        self, text: str, token: Tok
    ) -> List[InterpolationSegment]:
        parts: List[InterpolationSegment] = []
        literal: List[str] = []
        index = 0
        length = len(text)

        while index < length:
            ch = text[index]

            # Treat \u{...} as a literal escape so braces don't trigger interpolation.
            escape, next_index = self._try_scan_unicode_escape(text, index)
            if escape is not None:
                literal.append(escape)
                index = next_index
                continue

            if ch == "{":
                if index + 1 < length and text[index + 1] == "{":
                    expr_text, next_index = self._extract_interpolation_expr(
                        text, index + 2, token, raw_close=True
                    )

                    if literal:
                        parts.append(("text", "".join(literal)))
                        literal.clear()

                    expr_node = self._parse_interpolation_expr(expr_text)
                    parts.append(("raw_expr", expr_node))
                    index = next_index
                    continue

                expr_text, next_index = self._extract_interpolation_expr(
                    text, index + 1, token
                )

                if literal:
                    parts.append(("text", "".join(literal)))
                    literal.clear()

                expr_node = self._parse_interpolation_expr(expr_text)
                parts.append(("expr", expr_node))
                index = next_index
                continue
            elif ch == "}" and index + 1 < length and text[index + 1] == "}":
                literal.append("}")
                index += 2
                continue
            elif ch == "}":
                raise SyntaxError("Unescaped '}' in shell string literal")
            literal.append(ch)
            index += 1

        if literal or not parts:
            parts.append(("text", "".join(literal)))
        return parts

    def _extract_interpolation_expr(
        self, text: str, start: int, token: Tok, raw_close: bool = False
    ) -> Tuple[str, int]:
        depth = 1
        index = start
        length = len(text)
        in_quote: Optional[str] = None
        escape = False

        while index < length:
            ch = text[index]

            if in_quote is not None:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == in_quote:
                    in_quote = None
                index += 1
                continue

            if ch in ("'", '"'):
                in_quote = ch
                index += 1
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1

                if depth == 0:
                    expr = text[start:index]

                    if not expr.strip():
                        raise SyntaxError(
                            "Empty interpolation expression in string literal"
                        )

                    if raw_close:
                        if index + 1 >= length or text[index + 1] != "}":
                            raise SyntaxError(
                                f"Unterminated interpolation expression in string literal: {token.value!r}"
                            )
                        return expr, index + 2

                    return expr, index + 1
            index += 1

        raise SyntaxError(
            f"Unterminated interpolation expression in string literal: {token.value!r}"
        )

    def _try_scan_unicode_escape(
        self, text: str, index: int
    ) -> tuple[Optional[str], int]:
        # Returns (escape_text, next_index) for \u{...} or (None, index) if not present.
        length = len(text)
        if (
            text[index] == "\\"
            and index + 2 < length
            and text[index + 1] == "u"
            and text[index + 2] == "{"
        ):
            end = index + 3
            while end < length and text[end] != "}":
                end += 1
            if end < length:
                return text[index : end + 1], end + 1
            return text[index:], length
        return None, index

    def _parse_interpolation_expr(self, expr_src: str) -> Node:
        # Use RD fragment parser to keep fragment parsing self-contained
        tree = parse_expr_fragment(expr_src)
        pruned = self.__class__().transform(tree)
        return self._unwrap_fragment_expr(pruned)

    def _unwrap_fragment_expr(self, node: Node) -> Node:
        current = node

        while is_tree(current) and tree_label(current) == "expr":
            children = tree_children(current)

            if not children:
                break
            current = children[0]
        return current

    # Keep getters/setters explicit
    def obj_get(self, c):
        name = None
        body = c[-1] if c else None

        for part in c:
            if is_token(part) and part.type == TT.IDENT:
                name = part
                break
        return Tree("obj_get", [name, body])

    def obj_set(self, c):
        name = None
        param = None
        body = c[-1] if c else None

        for part in c:
            if is_token(part) and part.type == TT.IDENT:
                if name is None:
                    name = part
                elif param is None:
                    param = part
        return Tree("obj_set", [name, param, body])

    def obj_body(self, c):
        if c:
            return c[0]

        return Tree("body", [], attrs={"inline": False})

    def obj_method(self, c):
        name = None
        params = None
        body = c[-1] if c else None

        for part in c[:-1]:
            if is_token(part) and part.type == TT.IDENT:
                if name is None:
                    name = part
            elif is_tree(part) and tree_label(part) == "paramlist":
                params = part
        return Tree("obj_method", [name, params, body])

    def pattern_list_inline(self, c):
        items = []

        for tok in c:
            if is_token(tok) and tok.type == TT.IDENT:
                items.append(Tree("pattern", [tok]))

        if not items:
            raise SyntaxError("Empty inline pattern list")
        return Tree("pattern", [Tree("pattern_list", items)])

    def forin_pattern(self, c):
        return c[0] if c else Tree("pattern", [])

    def key_ident(self, c):
        return Tree("key_ident", c)

    def key_string(self, c):
        return Tree("key_string", c)

    def key_expr(self, c):
        return Tree("key_expr", c)

    def group_expr(self, c):
        return Tree("group", c)

    def setliteral(self, c):
        items = [x for x in c if not (is_token(x) and x.type == "COMMA")]
        return Tree("setliteral", items)

    def setliteral_empty(self, _):
        return Tree("setliteral", [])

    def setcomp(self, c):
        items = [x for x in c if not (is_token(x) and x.type == TT.SET)]
        return Tree("setcomp", items)

    def array(self, c):
        arr = [item for item in c if not (is_token(item) and item.type == TT.COMMA)]
        return Tree("array", arr)

    def array_empty(self, _c):
        return Tree("array", [])

    # Guard chains (inline if-else)
    def guardbranch(self, c):
        return Tree("guardbranch", c)

    def onelineguard(self, c):
        return Tree("onelineguard", c)

    # Subject assignment =a syntax
    # NOTE: rebind_primary is handled by base Transformer to preserve attrs

    def subjectassign(self, c):
        return Tree("subjectassign", c)

    def stmtsubjectassign(self, c):
        return Tree("stmtsubjectassign", c)

    def subjectassign_rhs(self, c):
        return Tree("subjectassign_rhs", c)

    def implicit_subject_chain(self, c):
        c = [x for x in c if not (is_token(x) and x.type == TT.DOT)]
        return Tree("implicit_chain", c)

    def explicit_subject_chain(self, c):
        c = [x for x in c if not (is_token(x) and x.type == TT.DOT)]
        return Tree("explicit_chain", c)

    # Dot stripping
    def field(self, c):
        ident_only = [tok for tok in c if tok.type == TT.IDENT]
        return Tree("field", ident_only or c)

    def bind(self, c):
        kept = []

        for item in c:
            if is_token(item) and item.type == TT.APPLYASSIGN:
                continue
            kept.append(item)
        return Tree("bind", kept)

    def index(self, c):
        return Tree("index", c)

    @v_args(meta=True)
    def call(self, meta, c):
        node = Tree("call", c)
        if meta is not None:
            node.meta = meta
        return node

    @v_args(meta=True)
    def lambdacall1(self, meta, c):
        body = c[-1] if c else None
        # unify postfix &(...) into a normal call with an amp_lambda arg
        node = Tree("call", [Tree("args", [Tree("amp_lambda", [body])])])
        if meta is not None:
            node.meta = meta
        return node

    @v_args(meta=True)
    def lambdacalln(self, meta, c):
        params, body = c[0], c[-1]
        node = Tree("call", [Tree("args", [Tree("amp_lambda", [params, body])])])
        if meta is not None:
            node.meta = meta
        return node

    def amp_lambda1(self, c):
        # Standalone &(expr) -> amp_lambda
        return Tree("amp_lambda", [c[0]])

    def amp_lambdan(self, c):
        # Standalone &[params](expr) -> amp_lambda with paramlist
        params, body = c[0], c[1]
        return Tree("amp_lambda", [params, body])

    def primary(self, c):
        return c[0] if len(c) == 1 else Tree("primary", c)

    def stmt(self, c):
        return c[0] if len(c) == 1 else Tree("stmt", c)

    def expr(self, c):
        return c[0]

    # ignore comments
    def COMMENT(self, _c):
        return Discard

    def comment(self, _c):
        return Discard

    # ---- using desugar ----------------------------------------------------
    def usingstmt(self, c):
        expr = None
        body = None
        handle: Optional[str] = None
        binder: Optional[str] = None

        for node in c:
            if is_tree(node) and tree_label(node) in {"body"}:
                body = node
            elif is_tree(node) and tree_label(node) == "using_handle":
                handle = _first_ident(node)
            elif is_tree(node) and tree_label(node) == "using_bind":
                binder = _first_ident(node)
            elif is_tree(node) and tree_label(node) == "bindexpr":
                expr = node
            elif expr is None and node is not Discard:
                expr = node

        if expr is None or body is None:
            raise SyntaxError("Malformed using statement")

        children: List[Node] = []
        if handle is not None:
            children.append(Tree("using_handle", [Tok(TT.IDENT, handle)]))
        if binder is not None:
            children.append(Tree("using_bind", [Tok(TT.IDENT, binder)]))
        children.append(expr)
        children.append(body)
        return Tree("usingstmt", children)

    # --- decorators ----
    def decorator(self, c):
        return Tree("decorator", c)

    def decorator_arg(self, c):
        return Tree("decorator_arg", c)

    def decorator_args(self, c):
        return Tree("decorator_args", c)

    def decorator_list(self, c):
        items = [item for item in c if item is not Discard]
        return Tree("decorator_list", items)

    def decorator_entry(self, c):
        # Store the raw decorator expression; omit leading '@' and newlines.
        expr = next(
            (
                node
                for node in c
                if not (is_token(node) and node.type in {TT.AT, TT.NEWLINE})
            ),
            None,
        )
        if expr is None:
            return Discard
        return Tree("decorator_spec", [expr])

    def decorator_target(self, c):
        # preserve every decorator expression in order; runtime evaluates them later.
        entries = [node for node in c if node is not Discard]
        if len(entries) == 1:
            return entries[0]
        return Tree("decorator_target", entries)

    def decoratordef(self, c):
        name = None
        params = None
        body = None

        for node in c:
            if is_token(node) and node.type == TT.IDENT and name is None:
                name = node
            elif is_tree(node) and tree_label(node) == "paramlist":
                params = node
            elif is_tree(node) and tree_label(node) in {"body"}:
                body = node

        children: List[Node] = []
        if name is not None:
            children.append(name)
        if params is not None:
            children.append(params)
        if body is not None:
            children.append(body)

        return Tree("decorator_def", children)

    def fnstmt(self, c):
        """
        Normalize function statements to fndef, preserving optional decorators and return contracts.
        """
        name = None
        params = None
        body = None
        return_contract = None
        decorators = None

        for node in c:
            if is_tree(node):
                label = tree_label(node)
                if label == "decorator_list":
                    decorators = node
                    continue
                elif label == "paramlist":
                    params = node
                    continue
                elif label in {"body"}:
                    body = node
                    continue
                elif label == "return_contract":
                    return_contract = node
                    continue

            if is_token(node) and node.type == TT.IDENT and name is None:
                name = node

        children: List[Node] = []

        if name is not None:
            children.append(name)
        if params is not None:
            children.append(params)
        if body is not None:
            children.append(body)
        if return_contract is not None:
            children.append(return_contract)
        if decorators is not None:
            children.append(decorators)

        return Tree("fndef", children)

    def deferstmt(self, c: List[Node]) -> Tree:
        label = None
        deps: List[Tok] = []
        body_node = None

        for node in c:
            if is_tree(node):
                tag = tree_label(node)

                if tag == "deferlabel" and label is None and node.children:
                    ident = _first_ident(node)
                    if ident:
                        label = ident
                    continue
                if tag == "deferafter":
                    deps.extend(_collect_defer_after(node))
                    continue
                if tag == "defer_block":
                    transformed = self._transform_tree(node.children[0])
                    if isinstance(transformed, Discard):
                        continue
                    body_node = Tree("deferblock", [transformed])
                    continue
                transformed = self._transform_tree(node)
                if isinstance(transformed, Discard):
                    continue
                body_node = transformed
            else:
                body_node = node

        children: List[Node] = []
        if label is not None:
            children.append(Tree("deferlabel", [Tok(TT.IDENT, label)]))
        if body_node is not None:
            children.append(body_node)
        if deps:
            children.append(Tree("deferdeps", deps))
        return Tree("deferstmt", children)

    def returnstmt(self, c: List[Node]) -> Tree:
        exprs: List[Node] = []

        for node in c:
            if is_token(node) and node.type == TT.RETURN:
                continue
            exprs.append(node)
        return Tree("returnstmt", exprs)

    def throwstmt(self, c: List[Node]) -> Tree:
        exprs: List[Node] = []

        for node in c:
            if is_token(node) and node.type == TT.THROW:
                continue
            exprs.append(node)
        return Tree("throwstmt", exprs)

    # ---- catch normalization ----
    def catchexpr(self, c: List[Node]) -> Tree:
        children = []

        for node in c:
            if is_token(node) and node.type == TT.CATCH:
                continue
            if (
                is_tree(node)
                and tree_label(node) == "catchtypes"
                and len(node.children) == 0
            ):
                continue
            children.append(node)

        return Tree("catchexpr", children)

    def catchtypes(self, c: List[Node]) -> Tree:
        items = [item for item in c if not (is_token(item) and item.type == TT.COMMA)]
        return Tree("catchtypes", items)

    def catchassign(self, c: List[Node]) -> Tree:
        items = [item for item in c if not (is_token(item) and item.type == TT.BIND)]
        return Tree("catchassign", items)

    # tidy arg nodes for printing
    def arg(self, c):
        return c[0]

    def argitem(self, c):
        return c[0]

    def arglist(self, c):
        return Tree("args", c)

    def arglistnamedmixed(self, c):
        return Tree("args", c)

    def compare(self, c):
        """Flatten CCC leg wrappers in comparison chains"""
        c = list(self._flatten_ccc_parts(c))
        return Tree("compare", c)

    def _flatten_ccc_parts(self, items):
        for item in items:
            if is_tree(item):
                label = tree_label(item)

                if label in {
                    "ccc_trailer",
                    "ccc_chain",
                    "ccc_leg",
                    "ccc_or_leg",
                    "ccc_and_leg",
                    "ccc_and_payload",
                }:
                    yield from self._flatten_ccc_parts(tree_children(item))
                    continue
            yield item


class ArgTidy(Transformer):
    def arg(self, c):
        return c[0]

    def argitem(self, c):
        return c[0]

    def arglist(self, c):
        return Tree("args", c)

    def arglistnamedmixed(self, c):
        return Tree("args", c)


def validate_named_args(tree: Tree) -> None:
    def is_namedarg(n: Node) -> bool:
        if is_tree(n) and tree_label(n) == "argnamed":
            return True

        if not is_tree(n):
            return False

        for ch in tree_children(n):
            if is_namedarg(ch):
                return True
        return False

    def walk(n: Node) -> None:
        if not is_tree(n):
            return

        label = tree_label(n)
        children = tree_children(n)

        if label in {"call", "callnc", "emitexpr"} and children:
            arglist = children[0]

            if not is_tree(arglist) or tree_label(arglist) != "arglistnamedmixed":
                return
            seen_named = False
            for ch in tree_children(arglist):
                if not is_namedarg(ch) and seen_named:
                    raise SyntaxError(
                        "Positional arguments must appear before named arguments"
                    )
                if is_namedarg(ch):
                    seen_named = True

        for ch in children:
            walk(ch)

    walk(tree)


def validate_hoisted_binders(tree: Tree) -> None:
    def walk(n: Node) -> None:
        if not is_tree(n):
            return
        label = tree_label(n)

        if label == "binderlist":
            pairs: list[tuple[str, bool]] = []

            for ch in tree_children(n):
                if is_tree(ch) and tree_label(ch) == "binder":
                    name = _first_ident(ch)
                    if name is None:
                        continue
                    is_hoisted = any(
                        is_token(tok) and tok.type == TT.PLUS
                        for tok in tree_children(ch)
                        if is_token(tok)
                    )
                    pairs.append((name, is_hoisted))

            if pairs:
                byname: dict[str, set[str]] = {}

                for name, is_h in pairs:
                    base = name
                    byname.setdefault(base, set()).add("H" if is_h else "P")

                for base, kinds in byname.items():
                    if kinds == {"H", "P"}:
                        raise SyntaxError(
                            f"Cannot use both hoisted and local binder for '{base}' in the same binder list"
                        )

                    if sum(1 for nm, is_h in pairs if nm == base and is_h) > 1:
                        raise SyntaxError(
                            f"Duplicate hoisted binder '{base}' in binder list"
                        )

        for ch in tree_children(n):
            walk(ch)

    walk(tree)


def _strip_discard(node: Tree) -> Tree:
    kept = [child for child in tree_children(node) if child is not Discard]
    return Tree(node.data, kept)


def _collect_defer_after(node: Tree) -> List[Tok]:
    deps: List[Tok] = []

    for ch in tree_children(node):
        name = _first_ident(ch)
        if name:
            deps.append(Tok(TT.IDENT, name))

    return deps


def _desugar_call_holes(node: Node) -> Node:
    if is_token(node) or not is_tree(node):
        return node
    children = node.children

    for idx, child in enumerate(list(children)):
        lowered = _desugar_call_holes(child)

        if lowered is not child:
            children[idx] = lowered

    candidate = node
    if tree_label(candidate) == "explicit_chain":
        replacement = _chain_to_lambda_if_holes(candidate)

        if replacement is not None:
            return replacement
    return candidate


def _chain_to_lambda_if_holes(chain: Tree) -> Optional[Tree]:
    def _contains_hole(node: Node) -> bool:
        if is_token(node) or not is_tree(node):
            return False

        if tree_label(node) == "holeexpr":
            return True
        return any(_contains_hole(child) for child in tree_children(node))

    holes: List[str] = []
    children = tree_children(chain)

    if not children:
        return None
    ops = children[1:]
    hole_call_index = None

    for idx, op in enumerate(ops):
        if tree_label(op) == "call" and _contains_hole(op):
            hole_call_index = idx
            break

    if hole_call_index is not None and hole_call_index + 1 < len(ops):
        raise SyntaxError(
            "Hole partials cannot be immediately invoked; assign or pass the partial before calling it"
        )

    def clone(node: Node) -> Node:
        if is_token(node):
            return node

        if not is_tree(node):
            return node
        label = tree_label(node)
        if label is None:
            return node

        if label == "holeexpr":
            # Hole params must be impossible to spell in source so user code
            # can never collide with lowered internals.
            name = f"{_HOLE_PARAM_PREFIX}{len(holes)}"
            holes.append(name)
            return Tok(TT.IDENT, name)
        cloned_children = [clone(child) for child in tree_children(node)]
        return Tree(label, cloned_children)

    cloned_chain = clone(chain)

    if not holes:
        return None
    params = [Tok(TT.IDENT, name) for name in holes]
    paramlist = Tree("paramlist", params)
    return Tree("amp_lambda", [paramlist, cloned_chain])


@dataclass
class _ParamEntry:
    node: Node
    isolated: bool
    trailing_contract: bool


def normalize_param_contracts(node: Node) -> Node:
    """Normalize grouped/implicit parameter contracts in param lists."""
    if is_token(node) or not is_tree(node):
        return node

    if tree_label(node) == "paramlist":
        return _normalize_paramlist(node)

    node.children = [normalize_param_contracts(child) for child in tree_children(node)]
    return node


def _normalize_paramlist(node: Tree) -> Tree:
    entries: List[_ParamEntry] = []

    for child in tree_children(node):
        entries.extend(_expand_param_entry(child))

    _propagate_param_contracts(entries)
    _validate_param_entries(entries)

    node.children = [entry.node for entry in entries]
    return node


def _expand_param_entry(node: Node) -> List[_ParamEntry]:
    if is_tree(node):
        label = tree_label(node)

        if label == "param_isolated":
            inner = node.children[0] if node.children else None
            if inner is None:
                raise SyntaxError("Empty isolated parameter group")
            entries = _expand_param_entry(inner)
            for entry in entries:
                entry.isolated = True
                entry.trailing_contract = False
            return entries

        if label == "param_group":
            return _expand_param_group(node)

    return [_entry_from_node(node, isolated=False)]


def _expand_param_group(node: Tree) -> List[_ParamEntry]:
    group_params: Optional[Tree] = None
    contract_expr: Optional[Node] = None

    for child in tree_children(node):
        if is_tree(child) and tree_label(child) == "paramlist":
            group_params = child
        elif is_tree(child) and tree_label(child) == "contract":
            kids = tree_children(child)
            if kids:
                contract_expr = kids[0]

    if group_params is None or contract_expr is None:
        raise SyntaxError("Malformed parameter group")

    entries: List[_ParamEntry] = []

    for child in tree_children(group_params):
        if param_default_expr(child) is not None:
            raise SyntaxError("Grouped parameters cannot include defaults")
        if _param_contract_expr(child) is not None:
            raise SyntaxError("Grouped parameters cannot include contracts")
        if _param_is_spread(child):
            raise SyntaxError("Grouped parameters cannot include spread")

        entry = _entry_from_node(child, isolated=False)
        entry.node = _add_param_contract(entry.node, contract_expr)
        entry.trailing_contract = False
        entries.append(entry)

    return entries


def _entry_from_node(node: Node, *, isolated: bool) -> _ParamEntry:
    has_contract = _param_contract_expr(node) is not None
    trailing_contract = has_contract and not isolated
    return _ParamEntry(
        node=node, isolated=isolated, trailing_contract=trailing_contract
    )


def _propagate_param_contracts(entries: List[_ParamEntry]) -> None:
    for idx, entry in enumerate(entries):
        if not entry.trailing_contract:
            continue

        contract_expr = _param_contract_expr(entry.node)
        if contract_expr is None:
            continue

        scan = idx - 1
        while scan >= 0:
            prev = entries[scan]

            if prev.isolated:
                scan -= 1
                continue

            if _param_contract_expr(prev.node) is not None:
                break

            prev.node = _add_param_contract(prev.node, contract_expr)
            scan -= 1


def _validate_param_entries(entries: List[_ParamEntry]) -> None:
    seen: set[str] = set()
    spread_indices: List[int] = []

    for idx, entry in enumerate(entries):
        name = _param_name(entry.node)
        if name is None:
            raise SyntaxError("Parameter name missing")
        if name in seen:
            raise SyntaxError(f"Duplicate parameter name '{name}'")
        seen.add(name)

        if _param_is_spread(entry.node):
            spread_indices.append(idx)

    if len(spread_indices) > 1:
        raise SyntaxError("Only one spread parameter is allowed")

    if spread_indices and spread_indices[0] != len(entries) - 1:
        raise SyntaxError("Spread parameter must be last")


def _param_contract_expr(node: Node) -> Optional[Node]:
    if not is_tree(node):
        return None

    if tree_label(node) not in {"param", "param_spread"}:
        return None

    for child in tree_children(node):
        if is_tree(child) and tree_label(child) == "contract":
            kids = tree_children(child)
            if kids:
                return kids[0]
    return None


def _param_is_spread(node: Node) -> bool:
    return is_tree(node) and tree_label(node) == "param_spread"


def _param_name(node: Node) -> Optional[str]:
    if is_token(node) and node.type == TT.IDENT:
        return str(node.value)

    if is_tree(node) and tree_label(node) in {"param", "param_spread"}:
        children = tree_children(node)
        if children and is_token(children[0]) and children[0].type == TT.IDENT:
            return str(children[0].value)

    return None


def _add_param_contract(node: Node, contract_expr: Node) -> Node:
    if _param_contract_expr(node) is not None:
        return node

    contract_node = Tree("contract", [contract_expr])

    if is_token(node):
        return Tree("param", [node, contract_node])

    if is_tree(node) and tree_label(node) in {"param", "param_spread"}:
        node.children.append(contract_node)
        return node

    return node


def _infer_amp_lambda_params(node: Node) -> Node:
    if is_token(node) or not is_tree(node):
        return node
    label = tree_label(node)

    if label == "amp_lambda" and len(node.children) == 1:
        body = _infer_amp_lambda_params(node.children[0])
        names, uses_subject = _collect_lambda_free_names(body)

        if uses_subject and names:
            raise SyntaxError(
                "Cannot mix subject '.' with implicit parameters in amp_lambda body"
            )

        if uses_subject or not names:
            node.children = [body]
        else:
            params = [Tok(TT.IDENT, name) for name in names]
            node.children = [Tree("paramlist", params), body]
        return node
    node.children = [_infer_amp_lambda_params(child) for child in tree_children(node)]
    return node


def _collect_lambda_free_names(node: Node) -> tuple[List[str], bool]:
    names: List[str] = []
    uses_subject = False

    def append(name: str) -> None:
        if name not in names:
            names.append(name)

    def walk(n: Node, parent_label: Optional[str]) -> None:
        nonlocal uses_subject

        if is_tree(n):
            label = tree_label(n)

            if label == "amp_lambda":
                return

            if label in {"implicit_chain", "subject"}:
                uses_subject = True

            if label == "explicit_chain":
                children = tree_children(n)

                if children:
                    head = children[0]
                    ident = _get_ident_value(head)

                    if ident is not None:
                        append(ident)
                    else:
                        walk(head, label)

                    for tail in children[1:]:
                        walk(tail, label)
                return

            for idx, child in enumerate(tree_children(n)):
                walk(child, label)
            return

        if is_token(n) and n.type == TT.IDENT:
            if parent_label in {
                "field",
                "method",
                "paramlist",
                "key_ident",
                "key_string",
            }:
                return
            append(n.value)

    walk(node, None)
    return names, uses_subject


def _get_ident_value(node: Node) -> Optional[str]:
    if is_token(node) and node.type == TT.IDENT:
        return str(node.value)
    return None


_CANONICAL_RENAMES = {
    "expr_nc": "expr",
    "ternary_nc": "ternary",
    "or_nc": "or",
    "and_nc": "and",
    "bind_nc": "bind",
    "walrus_nc": "walrus",
    "nullish_nc": "nullish",
    "compare_nc": "compare",
    "add_nc": "add",
    "mul_nc": "mul",
    "pow_nc": "pow",
    "unary_nc": "unary",
}

_FLATTEN_SINGLE = {"expr"}

_IGNORED_TOKEN_TYPES = {TT.SEMI, TT.NEWLINE, TT.INDENT, TT.DEDENT}


def _canonicalize_ast(node: Node) -> Optional[Node]:
    if is_token(node):
        return None if node.type in _IGNORED_TOKEN_TYPES else node

    if not is_tree(node):
        return node
    label = tree_label(node)
    if label is None:
        return node

    renamed = _CANONICAL_RENAMES.get(label, label)
    new_children: List[Node] = []

    for child in tree_children(node):
        canon = _canonicalize_ast(child)

        if canon is None:
            continue
        new_children.append(canon)

    if renamed in _FLATTEN_SINGLE and len(new_children) == 1:
        return new_children[0]
    return Tree(renamed, new_children)


def _first_ident(node: Node) -> Optional[str]:
    queue: List[Node] = [node]

    while queue:
        cur = queue.pop(0)

        if is_token(cur) and cur.type == TT.IDENT:
            return str(cur.value)

        if is_tree(cur):
            queue.extend(tree_children(cur))
    return None


# Light-touch AST normalization used by parse_to_ast in the legacy pipeline
def canonicalize_root(tree: Tree) -> Tree:
    label = tree_label(tree)
    if label is None:
        return tree

    if label not in {"start_noindent", "start_indented"}:
        return tree
    children = [child for child in tree_children(tree) if child is not Discard]

    if (
        len(children) == 1
        and is_tree(children[0])
        and tree_label(children[0]) == "stmtlist"
    ):
        stmtlist = _strip_discard(children[0])
    else:
        stmtlist = _strip_discard(Tree("stmtlist", children))

    return Tree("module", [stmtlist])
