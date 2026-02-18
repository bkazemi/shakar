"""
Recursive Descent Parser for Shakar

This serves as:
1. A reference implementation for the C parser
2. Documentation of parsing strategy and disambiguation

Structure:
- Lexer: Tok stream from source
- Parser: Recursive descent with Pratt parsing for expressions
- AST: Tree structure expected by the evaluator
"""

from typing import Optional, List, Any
from types import SimpleNamespace
from enum import Enum
from .tree import Tree, Node, is_tree, is_token, tree_label, tree_children

from .token_types import TT, Tok
from .lexer_rd import LexError


class ParseContext(Enum):
    """Track parsing context to disambiguate comma usage"""

    NORMAL = 0  # Default: CCC allowed
    FUNCTION_ARGS = 1  # Inside f(...): commas are arg separators
    ARRAY_ELEMENTS = 2  # Inside [...]: commas are element separators
    DESTRUCTURE_PACK = 3  # Inside destructure RHS pack: commas are value separators


# ============================================================================
# Parser
# ============================================================================


class ParseError(Exception):
    """Parse error with position info"""

    def __init__(self, message: str, token: Optional[Tok] = None):
        self.message = message
        self.token = token
        self.line: Optional[int] = None
        self.column: Optional[int] = None
        self.end_line: Optional[int] = None
        self.end_column: Optional[int] = None

        if token and token.line > 0 and token.column > 0:
            self.line = token.line
            self.column = max(1, token.column)
            # DURATION/SIZE/REGEX store tuples; lexeme is always the first element
            val = token.value
            if isinstance(val, tuple):
                val = val[0]
            text = "" if val is None else str(val)
            width = max(1, len(text))
            self.end_line = self.line
            self.end_column = self.column + width

        super().__init__(
            f"{message} at line {token.line}, col {token.column}" if token else message
        )


class Parser:
    """
    Recursive descent parser for Shakar.

    Expression precedence (lowest to highest):
    1. catch expressions
    2. ternary (? :)
    3. or (||)
    4. and (&&)
    5. bind (.=)
    6. send (->)
    7. walrus (:=)
    8. nullish (??)
    9. compare (==, !=, <, >, etc.) + CCC
    10. add (+, -, +>, ^)
    11. mul (*, /, //, %)
    12. pow (**)
    13. unary (-, !, ~, ++, --, wait, <-, spawn, $)
    14. postfix (.field, [index], (call), ++, --)
    15. primary (literals, identifiers, parens)
    """

    # Tokens that terminate a bare return/throw (no value follows).
    _BARE_STMT_TERMINATORS = frozenset(
        {
            TT.NEWLINE,
            TT.EOF,
            TT.SEMI,
            TT.RBRACE,
            TT.RPAR,
            TT.COMMA,
            TT.ELSE,
            TT.ELIF,
            TT.IF,
            TT.UNLESS,
            TT.PIPE,
        }
    )
    _ASSIGNMENT_OPS = (
        TT.ASSIGN,
        TT.WALRUS,
        TT.APPLYASSIGN,
        TT.PLUSEQ,
        TT.MINUSEQ,
        TT.STAREQ,
        TT.SLASHEQ,
        TT.FLOORDIVEQ,
        TT.MODEQ,
        TT.POWEQ,
    )

    def __init__(self, tokens: List[Tok], use_indenter: bool = True):
        self.tokens = tokens
        self.pos = 0
        self.current = tokens[0] if tokens else Tok(TT.EOF, None, 0, 0)
        self.previous = self.current  # Track last consumed token for _tok positions
        self.in_inline_body = False  # Track if we're in inlinebody context
        self.use_indenter = use_indenter  # Track indentation mode
        self.paren_depth = 0  # Track parenthesis nesting depth
        self.parse_context = (
            ParseContext.NORMAL
        )  # Track parsing context for comma disambiguation
        self.call_depth = 0  # Track lexical call-block nesting for emit '>'
        # -1 means "not inside a pattern default"; >= 0 records the paren_depth
        # at which the default expression began so that ~ / ~~ are suppressed at
        # that depth but allowed inside user-written parentheses.
        self._pattern_default_paren_depth: int = -1

    def _tok(
        self,
        type_name: str,
        value: str,
        line: Optional[int] = None,
        column: Optional[int] = None,
    ) -> Tok:
        return Tok(
            TT[type_name],
            value,
            line if line is not None else self.previous.line,
            column if column is not None else self.previous.column,
        )

    def _modifier_attrs(self, modifier_tok: Tok) -> dict[str, Any]:
        return {
            "modifier_name": str(modifier_tok.value),
            "modifier_tok": modifier_tok,
        }

    def _call_tree(self, args: List[Node], call_tok: Tok) -> Tree:
        return Tree(
            "call",
            args,
            meta=SimpleNamespace(line=call_tok.line, column=call_tok.column),
        )

    def _parse_number_literal(self, literal: object) -> int | float:
        """Parse NUMBER token value, handling bases and underscores."""
        if isinstance(literal, (int, float)):
            return literal

        text = str(literal)
        clean = text.replace("_", "")

        if text.startswith(("0b", "0o", "0x")):
            base = {"0b": 2, "0o": 8, "0x": 16}[text[:2]]
            return int(clean[2:], base)

        if "." in clean or "e" in clean.lower():
            return float(clean)
        return int(clean)

    def _is_int64_min_magnitude(self, literal: object) -> bool:
        """Check if literal equals 2^63 (magnitude of i64 min)."""
        if isinstance(literal, float):
            return False
        value = self._parse_number_literal(literal)
        return isinstance(value, int) and value == 2**63

    def _try_consume_int64_min_literal(self) -> Optional[Tok]:
        """Consume optional parens around an integer literal equal to 2^63."""
        idx = self.pos
        depth = 0

        while self._lookahead_check(idx, TT.LPAR):
            depth += 1
            idx += 1

        if not self._lookahead_check(idx, TT.NUMBER):
            return None

        tok = self._lookahead_peek(idx)
        if not self._is_int64_min_magnitude(tok.value):
            return None

        idx += 1
        for _ in range(depth):
            if not self._lookahead_check(idx, TT.RPAR):
                return None
            idx += 1

        for _ in range(depth):
            self.advance()
        self.advance()
        for _ in range(depth):
            self.advance()

        return tok

    def _take_noanchor_once(self, seen: list[bool]) -> bool:
        if not self.match(TT.DOLLAR):
            return False
        if seen[0]:
            raise ParseError("Multiple '$' segments in a chain", self.current)
        seen[0] = True
        return True

    def _maybe_noanchor(self, node: Tree, wrap: bool) -> Tree:
        return Tree("noanchor", [node]) if wrap else node

    def _check_field_name(self) -> bool:
        return self.check(TT.IDENT)

    def _field_node(self, tok: Tok) -> Tree:
        return Tree("field", [self._tok("IDENT", tok.value, tok.line, tok.column)])

    def _parse_field_or_method(self, noanchor: bool) -> Tree:
        """Parse .field or .method(args) after _check_field_name() passed."""
        tok = self.advance()
        field_tok = self._tok("IDENT", tok.value, tok.line, tok.column)

        # Check for immediate call -> method node
        if self.check(TT.LPAR):
            self.advance()
            args = self.parse_arg_list()
            self.expect(TT.RPAR)
            args_node = Tree("arglistnamedmixed", args) if args else None
            children = [field_tok] + ([args_node] if args_node else [])
            return self._maybe_noanchor(Tree("method", children), noanchor)

        return self._maybe_noanchor(self._field_node(tok), noanchor)

    def _parse_index_node(self, noanchor: bool) -> Tree:
        """Parse [selectors, default: expr] — assumes current token is LSQB."""
        lsqb_tok = self.advance()
        selectors = self.parse_selector_list()

        default = None
        if self.match(TT.COMMA):
            default_tok = self.expect(TT.IDENT)
            if default_tok.value != "default":
                raise ParseError(
                    f"Expected 'default' keyword, got '{default_tok.value}'",
                    default_tok,
                )
            self.expect(TT.COLON)
            default = self.parse_expr()

        rsqb_tok = self.expect(TT.RSQB)
        children = [lsqb_tok, Tree("selectorlist", selectors), rsqb_tok]
        if default:
            children.append(default)

        return self._maybe_noanchor(Tree("index", children), noanchor)

    def _parse_plain_index_node(self) -> Tree:
        lsqb_tok = self.expect(TT.LSQB)
        selectors = self.parse_selector_list()
        rsqb_tok = self.expect(TT.RSQB)
        return Tree("index", [lsqb_tok, Tree("selectorlist", selectors), rsqb_tok])

    def _parse_valuefan(self) -> Tree:
        """Parse fan items after '{' consumed — expects '}', builds valuefan node."""
        items = self.parse_fan_items()
        self.expect(TT.RBRACE)
        valuefan_items = [Tree("valuefan_item", [item]) for item in items]

        return Tree("valuefan", [Tree("valuefan_list", valuefan_items)])

    def _check_catch_start(self) -> bool:
        """Check if current token starts a catch clause (catch or @@)."""
        return self.check(TT.CATCH) or (
            self.check(TT.AT) and self.peek(1).type == TT.AT
        )

    def _check_catch_at(self, offset: int) -> bool:
        """Check if token at lookahead offset starts a catch clause."""
        tok = self.peek(offset)
        return tok.type == TT.CATCH or (
            tok.type == TT.AT and self.peek(offset + 1).type == TT.AT
        )

    def _consume_catch_keyword(self) -> None:
        """Consume catch or @@ token(s)."""
        if self.check(TT.CATCH):
            self.advance()
        else:
            self.advance(2)  # consume '@@'

    def _parse_catch_filter(self) -> tuple[List[Tok], Optional[Tok]]:
        """Parse optional (Type1, Type2) filter and binder after catch keyword."""
        types: List[Tok] = []
        if self.match(TT.LPAR):
            types.append(self.expect(TT.IDENT))
            while self.match(TT.COMMA):
                types.append(self.expect(TT.IDENT))
            self.expect(TT.RPAR)

        binder: Optional[Tok] = None
        if self.match(TT.BIND):
            binder = self.expect(TT.IDENT)
        elif self.check(TT.IDENT) and not types:
            binder = self.advance()

        return types, binder

    def _build_catch_children(
        self, expr: Node, types: List[Tok], binder: Optional[Tok], handler: Node
    ) -> list[Node]:
        """Assemble catch node children from parsed components."""
        children: list[Node] = [expr]
        if binder:
            children.append(binder)
        if types:
            children.append(Tree("catchtypes", types))
        children.append(handler)
        return children

    def _parse_flat_binop(self, label: str, op_type: TT, next_parser: Any) -> Node:
        """Parse flat left-associative binary op with interleaved op tokens.
        Returns unwrapped next-level node when no operator is present."""
        left = next_parser()
        if not self.check(op_type):
            return left
        children: list[Node] = [left]
        while self.check(op_type):
            op = self.advance()
            children.append(self._tok(op.type.name, op.value))
            children.append(next_parser())
        return Tree(label, children)

    def _parse_left_assoc(
        self, label: str, op_label: str, op_types: tuple[TT, ...], next_parser: Any
    ) -> Node:
        """Parse left-associative binary ops that wrap each operator in an op node."""
        left = next_parser()
        while self.check(*op_types):
            op = self.advance()
            right = next_parser()
            left = Tree(label, [left, Tree(op_label, [op]), right])
        return left

    def _parse_obj_item_body(self) -> Tree:
        """Parse object item body: indented block or inline expression."""
        if self.check(TT.NEWLINE):
            return self.parse_object_body()
        expr = self.parse_expr()
        return Tree("body", [expr], attrs={"inline": True})

    def _parse_call_op(self) -> Tree:
        """Parse call (args) — assumes current is LPAR."""
        lpar_tok = self.advance()
        args = self.parse_arg_list()
        self.expect(TT.RPAR)
        if args:
            args = [Tree("arglistnamedmixed", args)]
        else:
            args = []

        return self._call_tree(args, lpar_tok)

    def _parse_implicit_chain_head(self, seen_noanchor: list[bool]) -> Optional[Tree]:
        """Parse first operation after DOT in implicit chain."""
        noanchor = self._take_noanchor_once(seen_noanchor)

        if self._check_field_name():
            return self._parse_field_or_method(noanchor)

        if self.check(TT.LPAR):
            if noanchor:
                raise ParseError(
                    "No-anchor '$' is not valid on call segments", self.current
                )
            return self._parse_call_op()

        if self.check(TT.LSQB):
            return self._parse_index_node(noanchor)

        if self.check(TT.LBRACE):
            if noanchor:
                raise ParseError(
                    "No-anchor '$' is not valid on fan segments", self.current
                )
            self.advance()  # consume {
            return self._parse_valuefan()

        if noanchor:
            raise ParseError("Expected field or index after '$' in chain", self.current)

        return None

    def _collect_postfix_ops(
        self,
        ops: List[Node],
        seen_noanchor: list[bool],
        allow_amp: bool = False,
    ) -> None:
        """Shared postfix loop — collects DOT-members, index, call, incr/decr,
        amp-lambda (when allow_amp), and chain continuation into ops."""
        continuation_active = False

        while True:
            # Field access or method call
            if self.match(TT.DOT):
                noanchor = self._take_noanchor_once(seen_noanchor)
                if self._check_field_name():
                    ops.append(self._parse_field_or_method(noanchor))
                elif self.match(TT.LBRACE):
                    if noanchor:
                        raise ParseError(
                            "No-anchor '$' is not valid on fan segments",
                            self.current,
                        )
                    ops.append(self._parse_valuefan())
                else:
                    raise ParseError("Expected field name after '.'", self.current)

            # Indexing
            elif self.check(TT.DOLLAR) or self.check(TT.LSQB):
                noanchor = self._take_noanchor_once(seen_noanchor)
                if not self.check(TT.LSQB):
                    raise ParseError(
                        "Expected field or index after '$' in chain", self.current
                    )
                ops.append(self._parse_index_node(noanchor))

            # Call
            elif self.check(TT.LPAR):
                ops.append(self._parse_call_op())

            # Postfix increment/decrement
            elif self.check(TT.INCR, TT.DECR):
                op = self.advance()
                ops.append(Tree(op.type.name.lower(), []))

            # Postfix amp-lambda: expr&(body) or expr&[params](body)
            elif allow_amp and self.match(TT.AMP):
                lam = self.parse_anonymous_fn()
                if len(lam.children) == 2:  # Has paramlist
                    ops.append(Tree("lambdacalln", lam.children))
                else:  # Subject-based lambda
                    ops.append(Tree("lambdacall1", lam.children))

            # Chain continuation handling
            elif continuation_active and self.check(TT.DEDENT):
                self.advance()
                break
            elif self.check(TT.NEWLINE):
                if not continuation_active:
                    if self._start_chain_continuation():
                        continuation_active = True
                        continue
                    break
                cont = self._continue_chain_continuation()
                if cont is True:
                    continue
                if cont is False:
                    break
                k = 0
                while self.peek(k).type == TT.NEWLINE:
                    k += 1
                raise ParseError(
                    "Expected '.' to continue indented chain or end indentation",
                    self.peek(k),
                )
            else:
                break

    def _start_chain_continuation(self) -> bool:
        """
        Detect and consume NEWLINE+INDENT that starts a dot-chain continuation.
        Leaves the DOT as the next token.
        """
        if not self.check(TT.NEWLINE):
            return False
        k = 0
        while self.peek(k).type == TT.NEWLINE:
            k += 1
        if self.peek(k).type != TT.INDENT or self.peek(k + 1).type != TT.DOT:
            return False
        while self.match(TT.NEWLINE):
            pass
        self.expect(TT.INDENT)
        return True

    def _continue_chain_continuation(self) -> Optional[bool]:
        """
        Continue or close a dot-chain continuation block.
        Returns True to continue (DOT next), False to end (DEDENT consumed),
        or None if not a continuation line.
        """
        if not self.check(TT.NEWLINE):
            return None
        k = 0
        while self.peek(k).type == TT.NEWLINE:
            k += 1
        next_tok = self.peek(k)
        if next_tok.type == TT.DOT:
            while self.match(TT.NEWLINE):
                pass
            return True
        if next_tok.type == TT.DEDENT:
            while self.match(TT.NEWLINE):
                pass
            self.expect(TT.DEDENT)
            return False
        return None

    # ========================================================================
    # Tok Navigation
    # ========================================================================

    def peek(self, offset: int = 0) -> Tok:
        """Look ahead at token"""
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return Tok(TT.EOF, None, 0, 0)

    def _lookahead_advance(self, idx: int, paren_depth: int) -> tuple[Tok, int, int]:
        """
        Lookahead-safe token advance that doesn't mutate parser state.
        Returns (current_token, new_idx, new_paren_depth).
        Does NOT skip layout tokens - lookahead needs to see all tokens.
        """
        if idx >= len(self.tokens):
            return (Tok(TT.EOF, None, 0, 0), idx, paren_depth)

        tok = self.tokens[idx]
        new_idx = idx + 1
        new_paren_depth = paren_depth

        # Update paren depth tracking
        if tok.type == TT.LPAR:
            new_paren_depth += 1
        elif tok.type == TT.RPAR:
            new_paren_depth -= 1

        return (tok, new_idx, new_paren_depth)

    def _lookahead_peek(self, idx: int) -> Tok:
        """Peek at token at idx without advancing (stateless)"""
        if idx >= len(self.tokens):
            return Tok(TT.EOF, None, 0, 0)
        return self.tokens[idx]

    def _lookahead_check(self, idx: int, *types: TT) -> bool:
        """Check if token at idx matches any of the given types (stateless)"""
        return self._lookahead_peek(idx).type in types

    def advance(self, count: int = 1) -> Tok:
        """Consume `count` tokens (default 1) and return the last one."""
        if count < 1:
            raise ValueError("advance() requires count >= 1")

        def _advance_once() -> Tok:
            prev = self.current
            self.previous = prev  # Track for _tok position defaults

            # Track paren depth for layout token skipping
            if prev.type == TT.LPAR:
                self.paren_depth += 1
            elif prev.type == TT.RPAR:
                self.paren_depth -= 1

            self.pos += 1
            if self.pos < len(self.tokens):
                self.current = self.tokens[self.pos]
            else:
                self.current = Tok(TT.EOF, None, 0, 0)

            # Skip newlines and layout tokens when inside parentheses
            if self.paren_depth > 0:
                skip_types = (
                    (TT.NEWLINE, TT.INDENT, TT.DEDENT)
                    if self.use_indenter
                    else (TT.NEWLINE,)
                )
                while self.current.type in skip_types:
                    self.pos += 1
                    if self.pos < len(self.tokens):
                        self.current = self.tokens[self.pos]
                    else:
                        self.current = Tok(TT.EOF, None, 0, 0)
                        break

            return prev

        last = _advance_once()
        for _ in range(count - 1):
            last = _advance_once()
        return last

    def _rewind(self, pos: int) -> None:
        self.pos = pos
        self.current = (
            self.tokens[self.pos]
            if self.pos < len(self.tokens)
            else Tok(TT.EOF, None, 0, 0)
        )

    def check(self, *types: TT) -> bool:
        """Check if current token matches any of the given types"""
        return self.current.type in types

    def match(self, *types: TT) -> bool:
        """Check and consume if current token matches"""
        if self.check(*types):
            self.advance()
            return True
        return False

    def expect(self, token_type: TT, message: Optional[str] = None) -> Tok:
        """Consume token of expected type or raise error"""
        if not self.check(token_type):
            msg = message or f"Expected {token_type.name}, got {self.current.type.name}"
            raise ParseError(msg, self.current)
        return self.advance()

    def expect_seq(self, *types: TT) -> tuple[Tok, ...]:
        """Consume a sequence of expected types and return their tokens"""
        tokens: list[Tok] = []
        for token_type in types:
            tokens.append(self.expect(token_type))
        return tuple(tokens)

    def _push_parse_context(self, context: ParseContext) -> ParseContext:
        previous = self.parse_context
        self.parse_context = context
        return previous

    def _pop_parse_context(self, previous: ParseContext) -> None:
        self.parse_context = previous

    def _consume_newlines(self) -> None:
        while self.match(TT.NEWLINE):
            pass

    def _expect_indented_block(self, context: str, require_arm: bool = False) -> None:
        if not self.match(TT.NEWLINE):
            raise ParseError(f"{context} expects a newline after ':'", self.current)
        if not self.check(TT.INDENT):
            raise ParseError(f"{context} expects an indented block", self.current)
        self.advance()
        if require_arm and self.check(TT.DEDENT):
            raise ParseError(f"{context} requires at least one arm", self.current)

    def _snapshot(self) -> tuple[int, Tok, Tok, int]:
        return self.pos, self.current, self.previous, self.paren_depth

    def _restore(self, snapshot: tuple[int, Tok, Tok, int]) -> None:
        self.pos, self.current, self.previous, self.paren_depth = snapshot

    def skip_layout_tokens(self) -> None:
        """Skip NEWLINE and optionally INDENT/DEDENT tokens (when using indenter)"""
        if self.use_indenter:
            while self.match(TT.NEWLINE, TT.INDENT, TT.DEDENT):
                pass
        else:
            while self.match(TT.NEWLINE):
                pass

    # ========================================================================
    # Top-Level Parsing
    # ========================================================================

    def parse(self) -> Tree:
        """Parse entire program"""
        stmtlists: list[Tree] = []

        # Skip leading newlines
        while self.match(TT.NEWLINE):
            pass

        # Collect statements into stmtlist
        stmts: list[Node] = []
        while not self.check(TT.EOF):
            # Skip empty lines
            if self.match(TT.NEWLINE):
                # If we have accumulated statements, wrap them
                if stmts:
                    stmtlists.append(Tree("stmtlist", stmts))
                    stmts = []
                continue

            stmt = self.parse_statement()
            stmts.append(Tree("stmt", [stmt]))

            # Check for semicolon separator
            if self.match(TT.SEMI):
                stmts.append(self._tok("SEMI", ";"))
                # Continue parsing more statements on same line
                # Skip whitespace but not newlines
                continue

            # Consume trailing newlines
            while self.match(TT.NEWLINE):
                pass

        # Add any remaining statements
        if stmts:
            stmtlists.append(Tree("stmtlist", stmts))

        # Return appropriate start node based on indentation mode
        start_node = "start_indented" if self.use_indenter else "start_noindent"
        return Tree(start_node, list(stmtlists))

    # ========================================================================
    # Statements
    # ========================================================================

    def parse_statement(self) -> Tree:
        """
        Parse a single statement.

        Statements include:
        - Control flow (if, while, for)
        - Declarations (fn)
        - Assignments (=, :=, .=)
        - Expressions
        - Return, break, continue
        - etc.
        """
        stmt = self._parse_statement_prefix()
        if stmt:
            return self._wrap_postfix(stmt) or stmt

        expr_start = self.pos

        stmt = self._parse_destructure_with_contracts()
        if stmt:
            return stmt

        expr = self.parse_expr()
        return self._parse_statement_from_expr(expr_start, expr)

    def _parse_statement_prefix(self) -> Optional[Tree]:
        # Dispatch table for simple statement starters
        if not hasattr(self, "_stmt_dispatch"):
            self._stmt_dispatch = {
                TT.IF: self.parse_if_stmt,
                TT.WHILE: self.parse_while_stmt,
                TT.FOR: self.parse_for_stmt,
                TT.USING: self.parse_using_stmt,
                TT.CALL: self.parse_call_stmt,
                TT.DEFER: self.parse_defer_stmt,
                TT.HOOK: self.parse_hook_stmt,
                TT.DECORATOR: self.parse_decorator_stmt,
                TT.AT: self.parse_fn_stmt,
                TT.FN: self.parse_fn_stmt,
                TT.LET: self.parse_let_stmt,
                TT.IMPORT: self.parse_import_stmt,
                TT.RETURN: self.parse_return_stmt,
                TT.BREAK: self.parse_break_stmt,
                TT.CONTINUE: self.parse_continue_stmt,
                TT.THROW: self.parse_throw_stmt,
                TT.TRY: self.parse_try_stmt,
                TT.ASSERT: self.parse_assert_stmt,
                TT.DBG: self.parse_dbg_stmt,
            }

        if self.check(TT.FN) and self.peek(1).type != TT.IDENT:
            # Allow anonymous fn literals as expression statements.
            return None

        handler = self._stmt_dispatch.get(self.current.type)
        if handler:
            return handler()

        if (
            self.check(TT.QMARK)
            and self.peek(1).type == TT.IDENT
            and self.peek(1).value == "ret"
        ):
            # ?ret expr - returnif statement
            self.advance(2)  # consume "? ret"
            value = self.parse_expr()
            return Tree("returnif", [value])

        return None

    def parse_let_stmt(self) -> Tree:
        """
        Parse let-scoped assignment:
        let <pattern> = expr
        let <pattern> := expr
        let <lvalue> = expr
        let <lvalue> := expr
        """
        self.expect(TT.LET)

        pattern_snapshot = self._snapshot()
        if self.check(TT.IDENT, TT.LPAR, TT.SPREAD):
            try:
                pattern = self.parse_pattern()
                patterns = [pattern]

                if self.match(TT.COMMA):
                    patterns.append(self.parse_pattern())
                    while self.match(TT.COMMA):
                        patterns.append(self.parse_pattern())

                # Validate: rest pattern must be last
                for i, pat in enumerate(patterns):
                    if tree_label(pat) == "pattern_rest" and i != len(patterns) - 1:
                        tok = pat.children[0] if pat.children else self.current
                        raise ParseError("Rest pattern must be last", tok)

                if self.check(TT.ASSIGN, TT.WALRUS):
                    op = self.advance()
                    rhs = self.parse_destructure_rhs()
                    label = (
                        "destructure_walrus" if op.type == TT.WALRUS else "destructure"
                    )
                    return Tree(
                        "let",
                        [
                            Tree(
                                label,
                                [Tree("pattern_list", patterns), rhs],
                            )
                        ],
                    )
            except ParseError:
                pass

            self._restore(pattern_snapshot)

        expr = self.parse_expr()

        if is_tree(expr) and tree_label(expr) == "expr" and expr.children:
            inner = expr.children[0]
            if is_tree(inner) and tree_label(inner) == "walrus":
                return Tree("let", [inner])

        if self.check(TT.ASSIGN, TT.WALRUS):
            assign_tok = self.advance()  # Capture BEFORE parsing RHS
            lvalue = self._expr_to_lvalue(expr)
            rhs = self.parse_nullish_expr()
            return Tree(
                "let",
                [Tree("assignstmt", [lvalue, assign_tok, rhs])],
            )

        raise ParseError("Expected assignment after let", self.current)

    def _parse_import_string_literal(self) -> Tree:
        if self.check(TT.STRING, TT.RAW_STRING, TT.RAW_HASH_STRING):
            tok = self.advance()
            return Tree("literal", [tok])
        raise ParseError("Expected string literal after import", self.current)

    def parse_import_stmt(self) -> Tree:
        self.expect(TT.IMPORT)

        if self.match(TT.LSQB):
            if self.match(TT.STAR):
                self.expect(TT.RSQB)
                module_node = self._parse_import_string_literal()
                return Tree("import_mixin", [module_node])

            if self.check(TT.RSQB):
                raise ParseError("Expected import names or '*'", self.current)

            names = [self.expect(TT.IDENT)]
            while self.match(TT.COMMA):
                if self.check(TT.RSQB):
                    break
                names.append(self.expect(TT.IDENT))

            self.expect(TT.RSQB)
            module_node = self._parse_import_string_literal()
            return Tree(
                "import_destructure", [Tree("import_names", names), module_node]
            )

        module_node = self._parse_import_string_literal()
        bind_tok = None
        if self.match(TT.BIND):
            bind_tok = self.expect(TT.IDENT)

        children: list[Node] = [module_node]
        if bind_tok:
            children.append(bind_tok)

        return Tree("import_stmt", children)

    def _wrap_postfix(self, stmt: Tree | Tok) -> Optional[Tree]:
        """Wrap statement with postfix if/unless guard if present."""
        wrapped = stmt if isinstance(stmt, Tree) else Tree("expr", [stmt])

        if self.check(TT.IF):
            if_tok = self.advance()
            cond = self.parse_expr()
            return Tree("postfixif", [wrapped, if_tok, Tree("expr", [cond])])

        if self.check(TT.UNLESS):
            unless_tok = self.advance()
            cond = self.parse_expr()
            return Tree("postfixunless", [wrapped, unless_tok, Tree("expr", [cond])])

        return None

    def _parse_destructure_with_contracts(self) -> Optional[Tree]:
        # Lookahead for destructuring with contracts, defaults, or rest patterns
        # Detects: ident [= default] [~ contract] (, pattern)* (:=|=)
        if not (
            self.check(TT.IDENT, TT.SPREAD) and self._is_destructure_with_contracts()
        ):
            return None

        # Parse pattern(s) using general parse_pattern() to support nested patterns
        patterns = [self.parse_pattern()]

        while self.match(TT.COMMA):
            patterns.append(self.parse_pattern())

        # Validate: rest pattern must be last
        for i, pat in enumerate(patterns):
            if tree_label(pat) == "pattern_rest" and i != len(patterns) - 1:
                tok = pat.children[0] if pat.children else self.current
                raise ParseError("Rest pattern must be last", tok)

        # Always wrap in pattern_list for evaluator consistency
        pattern_node = Tree("pattern_list", patterns)

        if self.match(TT.ASSIGN):
            rhs = self.parse_destructure_rhs()
            return Tree("destructure", [pattern_node, rhs])
        if self.match(TT.WALRUS):
            rhs = self.parse_destructure_rhs()
            return Tree("destructure_walrus", [pattern_node, rhs])
        raise ParseError("Expected = or := after pattern", self.current)

    def _parse_statement_from_expr(self, expr_start: int, expr: Node) -> Tree:
        stmt = self._parse_destructure_after_expr(expr_start)
        if stmt:
            return stmt

        stmt = self._parse_guard_chain_after_expr(expr_start)
        if stmt:
            return stmt

        stmt = self._parse_catch_stmt_after_expr(expr_start)
        if stmt:
            return stmt

        # Fanout block form: expr { .field = value; ... }
        if self.check(TT.LBRACE):
            fanblock = self.parse_fanblock()
            return Tree("fanoutblock", [expr, fanblock])

        # Base statement starts as expression
        base_stmt: Tree | Tok = expr

        # Assignment handling before postfix-if to match grammar precedence
        if self.check(*self._ASSIGNMENT_OPS):
            lvalue = self._expr_to_lvalue(expr)
            op = self.advance()
            rhs = self._parse_assignment_rhs(lvalue)
            label = "assignstmt" if op.type == TT.ASSIGN else "compound_assign"
            base_stmt = Tree(label, [lvalue, op, rhs])

        # Postfix if/unless wraps the base statement
        postfix = self._wrap_postfix(base_stmt)
        if postfix:
            return postfix

        # Just a statement/expression
        if isinstance(base_stmt, Tree):
            return base_stmt
        return Tree("expr", [base_stmt])

    def _parse_destructure_after_expr(self, expr_start: int) -> Optional[Tree]:
        # Check for destructuring: a, b, c = ... or a, b, c := ...
        if not self.check(TT.COMMA):
            return None

        first_tok = self.tokens[expr_start] if expr_start < len(self.tokens) else None
        if first_tok is None or first_tok.type not in {TT.IDENT, TT.LPAR, TT.SPREAD}:
            # Not a valid pattern start; comma belongs to surrounding expression (e.g., anon fn body)
            return None

        # Backtrack and parse as pattern_list.
        # Disable defaults: `=` here means the assignment operator, not a default.
        self._rewind(expr_start)
        pattern_list = self.parse_pattern_list(allow_default=False)

        if self.match(TT.ASSIGN):
            rhs = self.parse_destructure_rhs()
            return Tree("destructure", [pattern_list, rhs])
        if self.match(TT.WALRUS):
            rhs = self.parse_destructure_rhs()
            return Tree("destructure_walrus", [pattern_list, rhs])
        raise ParseError("Expected = or := after pattern list", self.current)

    def _parse_guard_chain_after_expr(self, expr_start: int) -> Optional[Tree]:
        # Guard chain inline form
        if not self.check(TT.COLON):
            return None
        self._rewind(expr_start)
        return self.parse_guard_chain()

    def _parse_catch_stmt_after_expr(self, expr_start: int) -> Optional[Tree]:
        # Catch statement form: expr catch ... : body
        if self.check(TT.NEWLINE):
            # Allow blank lines before a trailing catch so multiline expressions can attach.
            k = 0
            while self.peek(k).type == TT.NEWLINE:
                k += 1
            if not self._check_catch_at(k):
                return None
        elif not self._check_catch_start():
            return None
        self._rewind(expr_start)
        return self.parse_catch_stmt()

    def parse_if_stmt(self) -> Tree:
        """
        Parse if statement:
        if expr: body [elif expr: body]* [else: body]
        """
        if_tok = self.expect(TT.IF)
        cond = self.parse_expr()
        self.expect(TT.COLON)
        then_body = self.parse_body()

        elifs = []
        else_clause = None

        while self.check(TT.ELIF):
            elif_tok = self.advance()
            elif_cond = self.parse_expr()
            self.expect(TT.COLON)
            elif_body = self.parse_body()
            elifs.append(Tree("elifclause", [elif_tok, elif_cond, elif_body]))

        if self.check(TT.ELSE):
            else_tok = self.advance()
            self.expect(TT.COLON)
            else_body = self.parse_body()
            else_clause = Tree("elseclause", [else_tok, else_body])

        children = [if_tok, cond, then_body] + elifs
        if else_clause:
            children.append(else_clause)

        return Tree("ifstmt", children)

    def parse_match_expr(self) -> Tree:
        """
        Parse match expression:
        match expr:
          pattern | pattern: body
          else: body
        """
        self.expect(TT.MATCH)
        cmp_node: Optional[Tree] = None
        # Optional comparator binder: match[cmp] subject: ...
        if self._should_parse_match_cmp():
            cmp_node = self.parse_match_cmp()
        subject = self.parse_expr()
        self.expect(TT.COLON)
        self._expect_indented_block("match", require_arm=True)

        arms: list[Tree] = []
        else_body: Optional[Tree] = None

        while not self.check(TT.DEDENT, TT.EOF):
            if self.match(TT.NEWLINE):
                continue

            if self.match(TT.ELSE):
                if else_body:
                    raise ParseError("match has multiple else arms", self.current)
                self.expect(TT.COLON)
                else_body = self.parse_body()
                self._consume_newlines()
                if not self.check(TT.DEDENT):
                    raise ParseError("match else must be last", self.current)
                break

            patterns = [self.parse_match_pattern()]
            while self.match(TT.PIPE):
                patterns.append(self.parse_match_pattern())

            self.expect(TT.COLON)
            body = self.parse_body()
            arms.append(Tree("matcharm", [Tree("matchpatterns", patterns), body]))

        self.expect(TT.DEDENT)

        children: list[Node] = []
        if cmp_node:
            children.append(cmp_node)
        children.append(subject)
        children.extend(arms)
        if else_body:
            children.append(Tree("matchelse", [else_body]))
        return Tree("matchexpr", children)

    def _should_parse_match_cmp(self) -> bool:
        if not self.check(TT.LSQB):
            return False
        # Ensure this isn't a list-literal subject right before ':'.
        # peek() returns EOF past the end, so the lookahead is safe.
        if self._match_cmp_is_single():
            return self.peek(3).type != TT.COLON
        if self._match_cmp_is_neg_in():
            return self.peek(4).type != TT.COLON
        if self._match_cmp_is_not_in():
            return self.peek(4).type != TT.COLON
        return False

    def _is_match_cmp_token(self, tok: Tok) -> bool:
        if tok.type in {
            TT.EQ,
            TT.NEQ,
            TT.LT,
            TT.LTE,
            TT.GT,
            TT.GTE,
            TT.IN,
            TT.REGEXMATCH,
        }:
            return True
        return tok.type == TT.IDENT and tok.value in {
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
        }

    def _match_cmp_is_single(self) -> bool:
        cmp_tok = self.peek(1)
        end_tok = self.peek(2)
        if end_tok.type != TT.RSQB:
            return False
        return self._is_match_cmp_token(cmp_tok)

    def _match_cmp_is_neg_in(self) -> bool:
        return (
            self.peek(1).type == TT.NEG
            and self.peek(2).type == TT.IN
            and self.peek(3).type == TT.RSQB
        )

    def _match_cmp_is_not_in(self) -> bool:
        return (
            self.peek(1).type == TT.NOT
            and self.peek(2).type == TT.IN
            and self.peek(3).type == TT.RSQB
        )

    def parse_match_cmp(self) -> Tree:
        self.expect(TT.LSQB)
        # Support two-token comparators: !in / not in.
        if self.check(TT.NEG, TT.NOT) and self.peek(1).type == TT.IN:
            first = self.advance()
            second = self.advance()
            self.expect(TT.RSQB)
            return Tree("matchcmp", [first, second])
        tok = self.advance()
        if not self._is_match_cmp_token(tok):
            raise ParseError("Invalid match comparator", tok)
        self.expect(TT.RSQB)
        return Tree("matchcmp", [tok])

    def parse_match_pattern(self) -> Tree:
        pattern = self.parse_expr()
        self._validate_match_pattern(pattern)
        return pattern

    def _validate_match_pattern(self, pattern: Node) -> None:
        current: Node = pattern
        while True:
            if is_tree(current):
                label = tree_label(current)
                if label == "expr" and current.children:
                    current = current.children[0]
                    continue
                if label == "group_expr" and current.children:
                    current = current.children[0]
                    continue
            break

        if is_tree(current) and tree_label(current) in {"object", "array"}:
            raise ParseError(
                "Object/array literals are reserved for v0.2 match patterns",
                self.current,
            )

        def _first_token(node: Node) -> Optional[Tok]:
            if is_token(node):
                return node
            if not is_tree(node):
                return None
            for child in tree_children(node):
                tok = _first_token(child)
                if tok:
                    return tok
            return None

        def visit(node: Node) -> None:
            if not is_tree(node):
                return
            label = tree_label(node)
            if label in {"subject", "implicit_chain"}:
                tok = _first_token(node)
                if tok and label == "implicit_chain":
                    dot_col = tok.column - 1 if tok.column > 1 else tok.column
                    tok = Tok(TT.DOT, ".", tok.line, dot_col)
                raise ParseError(
                    "Match patterns cannot use implicit subject '.'",
                    tok,
                )
            for child in tree_children(node):
                visit(child)

        visit(pattern)

    def parse_while_stmt(self) -> Tree:
        """Parse while loop: while expr: body"""
        while_tok = self.expect(TT.WHILE)
        cond = self.parse_expr()
        self.expect(TT.COLON)
        body = self.parse_body()
        return Tree("whilestmt", [while_tok, cond, body])

    def parse_for_stmt(self) -> Tree:
        """
        Parse for loop:
        for x in expr: body (forin)
        for k, v in expr: body (forin with destructuring)
        for[i] expr: body (forindexed)
        for[k, v] expr: body (formap2)
        for expr: body (forsubject - subjectful loop)
        """
        for_tok = self.expect(TT.FOR)

        # Check for indexed syntax: for[...] (pattern bindings).
        # Disambiguation is syntax-based, not whitespace-based:
        # treat bracket content as binders only when an iterable expression
        # follows the closing bracket (i.e., `for[pat] expr: ...`).
        if self.check(TT.LSQB) and self.peek(1).type in {TT.IDENT, TT.CARET}:
            binder_snapshot = self._snapshot()

            try:
                self.advance()  # consume [
                binder1 = self.parse_binderpattern()
                binder2 = None
                if self.match(TT.COMMA):
                    binder2 = self.parse_binderpattern()
                self.expect(TT.RSQB)
            except ParseError:
                self._restore(binder_snapshot)
            else:
                # `for[binders]: ...` has no iterable, so parse it as
                # subjectful `for [ ... ]: ...` instead.
                if self.check(TT.COLON):
                    self._restore(binder_snapshot)
                else:
                    iterable = self.parse_expr()
                    self.expect(TT.COLON)
                    body = self.parse_body()
                    if binder2:
                        return Tree(
                            "formap2", [for_tok, binder1, binder2, iterable, body]
                        )
                    return Tree("forindexed", [for_tok, binder1, iterable, body])

        # Check if it's "for x in expr" or "for x, y in expr" (with destructuring)
        if self.check(TT.IDENT):
            # Lookahead to check if this is for-in
            idx = self.pos
            paren_depth = 0
            ident_values = []

            # Collect identifiers using lookahead
            tok, idx, paren_depth = self._lookahead_advance(idx, paren_depth)
            if tok.type == TT.IDENT:
                ident_values.append(tok.value)

                # Check for more identifiers
                while idx < len(self.tokens):
                    tok, new_idx, new_paren = self._lookahead_advance(idx, paren_depth)
                    if tok.type == TT.COMMA:
                        idx, paren_depth = new_idx, new_paren
                        tok, new_idx, new_paren = self._lookahead_advance(
                            idx, paren_depth
                        )
                        if tok.type == TT.IDENT:
                            ident_values.append(tok.value)
                            idx, paren_depth = new_idx, new_paren
                        else:
                            break
                    else:
                        break

                # Check if IN follows
                tok, _, _ = self._lookahead_advance(idx, paren_depth)
                if tok.type == TT.IN:
                    # Confirmed for-in, now consume tokens for real
                    idents = []
                    for _ in ident_values:
                        idents.append(self.advance())
                        if self.check(TT.COMMA):
                            self.advance()

                    in_tok = self.expect(TT.IN)
                    iterable = self.parse_expr()
                    self.expect(TT.COLON)
                    body = self.parse_body()

                    # Build pattern (use original tokens directly)
                    if len(idents) == 1:
                        # Simple pattern: for x in expr
                        pattern = Tree("pattern", [idents[0]])
                    else:
                        # Destructuring pattern: for k, v in expr
                        pattern = Tree("pattern_list_inline", idents)

                    return Tree("forin", [for_tok, pattern, in_tok, iterable, body])

        # Subjectful for: for expr: body
        iterable = self.parse_expr()
        self.expect(TT.COLON)
        body = self.parse_body()
        return Tree("forsubject", [for_tok, iterable, body])

    def parse_using_stmt(self) -> Tree:
        """
        Parse using statement:
        using [handle] expr [bind ident]: body
        """
        self.expect(TT.USING)

        handle = None
        if self.match(TT.LSQB):
            ident_tok, _ = self.expect_seq(TT.IDENT, TT.RSQB)
            handle = ident_tok

        resource = self.parse_expr()

        binder = None
        if self.match(TT.BIND):
            binder = self.expect(TT.IDENT)

        self.expect(TT.COLON)
        body = self.parse_body()

        children: List[Any] = []
        if handle:
            children.append(
                Tree(
                    "using_handle",
                    [handle],
                    attrs=self._modifier_attrs(handle),
                )
            )
        children.append(resource)
        if binder:
            children.append(Tree("using_bind", [binder]))
        children.append(body)
        return Tree("usingstmt", children)

    def parse_call_stmt(self) -> Tree:
        """
        Parse call statement:
        call [name] expr:
          body
        call expr bind name:
          body
        """
        self.expect(TT.CALL)

        handle = None
        if self.match(TT.LSQB):
            ident_tok, _ = self.expect_seq(TT.IDENT, TT.RSQB)
            handle = ident_tok

        target = self.parse_expr()

        binder = None
        if self.match(TT.BIND):
            binder = self.expect(TT.IDENT)

        if handle and binder:
            raise ParseError(
                "call supports either [name] or bind name, not both", self.current
            )

        self.expect(TT.COLON)

        self.call_depth += 1
        try:
            body = self.parse_body()
        finally:
            self.call_depth -= 1

        children: List[Any] = []
        bind_tok = handle or binder
        if bind_tok:
            children.append(
                Tree(
                    "call_bind",
                    [bind_tok],
                    attrs=self._modifier_attrs(bind_tok),
                )
            )
        children.append(target)
        children.append(body)
        return Tree("callstmt", children)

    def parse_defer_stmt(self) -> Tree:
        """
        Parse defer statement:
        defer simplecall [after ...]
        defer [label] [after ...] : body
        """
        self.expect(TT.DEFER)

        # Lookahead to see if a colon appears before a left paren; colon => block form
        colon_ahead = False
        lpar_ahead = False
        depth = 0
        scan_pos = self.pos
        while scan_pos < len(self.tokens):
            tok = self.tokens[scan_pos]
            if tok.type in {TT.LPAR, TT.LSQB, TT.LBRACE}:
                depth += 1
            elif tok.type in {TT.RPAR, TT.RSQB, TT.RBRACE} and depth > 0:
                depth -= 1
            if depth == 0:
                if tok.type == TT.COLON:
                    colon_ahead = True
                    break
                if tok.type == TT.LPAR and not lpar_ahead:
                    lpar_ahead = True
            if tok.type in {TT.NEWLINE, TT.INDENT, TT.DEDENT, TT.EOF} and depth == 0:
                break
            scan_pos += 1

        defer_children: List[Tree | Tok] = []

        if colon_ahead:
            # Optional label
            if self.check(TT.IDENT):
                label_tok = self.advance()
                defer_children.append(Tree("deferlabel", [label_tok]))

            # Optional deferafter
            if self.match(TT.AFTER):
                defer_children.append(self._parse_defer_after())

            self.expect(TT.COLON)
            body = self.parse_body()
            defer_children.append(Tree("defer_block", [body]))
            return Tree("deferstmt", defer_children)

        # simplecall branch
        call_expr = self.parse_postfix_expr()
        if not self._expr_has_call(call_expr):
            raise ParseError("defer expects a call expression or block", self.current)

        # Optional deferafter
        if self.match(TT.AFTER):
            defer_children.append(self._parse_defer_after())
        defer_children.append(call_expr)

        return Tree("deferstmt", defer_children)

    def _parse_defer_after(self) -> Tree:
        deps: List[Tok] = []

        if self.match(TT.LPAR):
            if not self.check(TT.RPAR):
                deps.append(self.expect(TT.IDENT))
                while self.match(TT.COMMA):
                    deps.append(self.expect(TT.IDENT))
            self.expect(TT.RPAR)
        else:
            deps.append(self.expect(TT.IDENT))

        return Tree("deferafter", deps)

    def _expr_has_call(self, expr: Tree | Tok) -> bool:
        if isinstance(expr, Tree):
            if expr.data in {"call", "method"}:
                return True
            return any(self._expr_has_call(ch) for ch in expr.children)
        return False

    def _scan_colon_after_parens(self, start_pos: int) -> bool:
        """Look ahead from a '(' token to find a matching ')' followed by a colon at depth 0."""
        depth = 0
        idx = start_pos

        while idx < len(self.tokens):
            tok = self.tokens[idx]

            if tok.type in {TT.LPAR, TT.LSQB, TT.LBRACE}:
                depth += 1
            elif tok.type in {TT.RPAR, TT.RSQB, TT.RBRACE}:
                depth -= 1
                if depth < 0:
                    return False

            if depth == 0:
                if tok.type == TT.COLON:
                    return True
                if tok.type in {TT.COMMA, TT.RBRACE, TT.SEMI, TT.NEWLINE, TT.EOF}:
                    return False

            idx += 1

        return False

    def _looks_like_object_item_start(self) -> bool:
        """Heuristic to detect start of the next object item without consuming tokens."""
        if self.check(TT.RBRACE):
            return True
        if self.check(TT.GET, TT.SET):
            return True
        if self.check(TT.IDENT):
            nxt = self.peek(1)
            if nxt.type == TT.COLON:
                return True
            if nxt.type == TT.LPAR and self._scan_colon_after_parens(self.pos + 1):
                return True
            # Pun: bare IDENT followed by separator or closing brace
            if nxt.type in {TT.COMMA, TT.RBRACE}:
                return True
        # Expression key: (expr): value - need to scan beyond the closing paren
        if self.check(TT.LPAR):
            # Scan for matching closing paren and check if colon follows
            depth = 0
            idx = self.pos
            while idx < len(self.tokens):
                tok = self.tokens[idx]
                if tok.type in {TT.LPAR, TT.LSQB, TT.LBRACE}:
                    depth += 1
                elif tok.type in {TT.RPAR, TT.RSQB, TT.RBRACE}:
                    depth -= 1
                    if depth == 0 and tok.type == TT.RPAR:
                        # Found closing paren, check if next is colon
                        if (
                            idx + 1 < len(self.tokens)
                            and self.tokens[idx + 1].type == TT.COLON
                        ):
                            return True
                        break
                idx += 1
        # String key
        if self.check(TT.STRING, TT.RAW_STRING, TT.RAW_HASH_STRING):
            return True
        return False

    def parse_hook_stmt(self) -> Tree:
        """
        Parse hook statement:
        hook "name": body

        Grammar: hook: HOOK (STRING | RAW_STRING | RAW_HASH_STRING) ":" (inlinebody | indentblock)
        """
        self.expect(TT.HOOK)

        if self.check(TT.STRING, TT.RAW_STRING, TT.RAW_HASH_STRING):
            name = self.advance()
        else:
            raise ParseError("Expected string name for hook", self.current)

        self.expect(TT.COLON)
        body = self.parse_body()

        # Convert inlinebody to amp_lambda if it contains implicit chain
        # For now, just wrap the body directly
        return Tree("hook", [name, Tree("amp_lambda", [body])])

    def parse_decorator_stmt(self) -> Tree:
        """
        Parse decorator declaration:
        decorator name(params): body
        """
        _, name, _ = self.expect_seq(TT.DECORATOR, TT.IDENT, TT.LPAR)
        params = self.parse_param_list()
        self.expect(TT.RPAR)

        self.expect(TT.COLON)
        body = self.parse_body()

        return Tree("decorator_def", [name, params, body])

    # Fanout assignment operator token => tree label mapping.
    _FANOP_MAP = {
        TT.ASSIGN: "fanop_assign",
        TT.APPLYASSIGN: "fanop_apply",
        TT.PLUSEQ: "fanop_pluseq",
        TT.MINUSEQ: "fanop_minuseq",
        TT.STAREQ: "fanop_stareq",
        TT.SLASHEQ: "fanop_slasheq",
        TT.FLOORDIVEQ: "fanop_floordiveq",
        TT.MODEQ: "fanop_modeq",
        TT.POWEQ: "fanop_poweq",
    }

    def parse_fanblock(self) -> Tree:
        """
        Parse fanout block: { .field = value; .field2 += value2; ... }
        Grammar: fanblock: "{" fanclause (fanclause_sep fanclause)* fanclause_sep? "}"
                 fanclause: DOT fanpath fanassignop expr
        """
        self.expect(TT.LBRACE)
        clauses = []
        clause_count = 0

        while not self.check(TT.RBRACE, TT.EOF):
            # Skip newlines and separators
            if self.match(TT.NEWLINE, TT.INDENT, TT.DEDENT):
                continue
            if self.match(TT.SEMI, TT.COMMA):
                continue

            dot_tok = self.expect(TT.DOT)

            # Parse fanpath: IDENT or [selectorlist] segments, dot-separated
            segs: List[Tree] = []
            segs.append(self._parse_fanpath_segment())

            while True:
                if self.match(TT.DOT):
                    segs.append(self._parse_fanpath_segment())
                elif self.check(TT.LSQB):
                    # Adjacent selector without dot (e.g., .rows[1].v)
                    segs.append(self._parse_lv_index_segment())
                else:
                    break

            fanpath = Tree("fanpath", segs)

            # Parse assignment operator
            fanop_label = self._FANOP_MAP.get(self.current.type)
            if fanop_label is None:
                raise ParseError("Expected fanout assignment operator", self.current)
            self.advance()
            fanop = Tree(fanop_label, [])

            value = self.parse_expr()

            clauses.append(Tree("fanclause", [dot_tok, fanpath, fanop, value]))
            clause_count += 1

        self.expect(TT.RBRACE)

        # Single-clause fanout: allow if it either fans to multiple targets *or*
        # the RHS uses the implicit subject (e.g., state{ .cur = .next }).
        if clause_count == 1:
            fanpath = clauses[0].children[1]
            value_expr = clauses[0].children[3]
            if not (
                self._fanpath_has_multiselector(fanpath)
                or self._expr_uses_subject(value_expr)
            ):
                raise ParseError(
                    "Single-clause fanout requires a multi-selector or implicit-subject RHS",
                    self.current,
                )

        return Tree("fanblock", clauses)

    def _parse_fanpath_segment(self) -> Tree:
        """Parse a single fanpath segment: IDENT field or [selector] index."""
        if self.check(TT.IDENT):
            tok = self.advance()
            return Tree("field", [self._tok("IDENT", tok.value, tok.line, tok.column)])

        if self.check(TT.LSQB):
            return self._parse_lv_index_segment()

        raise ParseError(
            "Expected identifier or [selector] in fanout path", self.current
        )

    def _parse_lv_index_segment(self) -> Tree:
        """Parse a bracketed selector segment: '[' selectorlist ']'."""
        lsqb_tok = self.advance()
        selectors = self.parse_selector_list()
        rsqb_tok = self.expect(TT.RSQB)

        return Tree("lv_index", [lsqb_tok, Tree("selectorlist", selectors), rsqb_tok])

    def _fanpath_has_multiselector(self, fanpath: Tree) -> bool:
        """Detect slice or multi-index selector in fanpath segments."""
        for seg in fanpath.children:
            if isinstance(seg, Tree) and seg.data == "lv_index":
                selectorlist = next(
                    (
                        ch
                        for ch in seg.children
                        if isinstance(ch, Tree) and ch.data == "selectorlist"
                    ),
                    None,
                )
                if selectorlist is None:
                    continue
                selectors = [ch for ch in selectorlist.children if isinstance(ch, Tree)]
                if len(selectors) > 1:
                    return True
                if selectors:
                    sel = selectors[0]
                    if any(
                        isinstance(grand, Tree) and grand.data == "slicesel"
                        for grand in sel.children
                    ):
                        return True
        return False

    def _expr_uses_subject(self, expr: Tree | Tok) -> bool:
        """Heuristic: detect use of implicit subject (`.`) inside an expression."""
        stack = [expr]
        while stack:
            node = stack.pop()
            if isinstance(node, Tree):
                if node.data in {"implicit_chain", "subject"}:
                    return True
                stack.extend(node.children)
        return False

    def parse_fn_stmt(self) -> Tree:
        """
        Parse function declaration (possibly with decorators):
        [@decorator_call]*
        fn name(params): body
        """
        # Check for decorator list before fn
        decorators = []
        while self.check(TT.AT):
            self.advance()  # consume @
            # Parse decorator: identifier with optional field access and call
            # @name, @name(), @obj.field, @obj.method()
            decorator_expr = self.parse_postfix_expr()
            # Skip newlines after decorator
            self._consume_newlines()
            # decorator_entry: just the expression, no @ token
            decorators.append(Tree("decorator_entry", [decorator_expr]))

        _, name, _ = self.expect_seq(TT.FN, TT.IDENT, TT.LPAR)
        params = self.parse_param_list()
        self.expect(TT.RPAR)

        # Optional return contract: ~ Schema
        return_contract = None
        if self.match(TT.TILDE):
            return_contract = self.parse_expr()

        self.expect(TT.COLON)
        body = self.parse_body()

        # fnstmt structure: [name, params, body, return_contract?, decorator_list?]
        children = [name, params, body]

        if return_contract:
            children.append(Tree("return_contract", [return_contract]))

        if decorators:
            decorator_list = Tree("decorator_list", decorators)
            children.append(decorator_list)

        return Tree("fnstmt", children)

    def parse_wait_expr(self) -> Tree:
        """
        Parse wait forms:
        - wait expr / wait(expr)
        - wait[IDENT]: ... (block forms)
        - wait[IDENT] expr (single-expr forms)
        """
        self.expect(TT.WAIT)

        if self.match(TT.LSQB):
            kind_tok = self.expect(TT.IDENT)
            kind = str(kind_tok.value)
            self.expect(TT.RSQB)

            if self.match(TT.COLON):
                if kind == "any":
                    return self._parse_wait_any_block(kind_tok)
                if kind == "all":
                    return self._parse_wait_named_block(
                        "waitallblock",
                        "waitallarm",
                        "wait[all]",
                        kind_tok,
                    )
                if kind == "group":
                    return self._parse_wait_group_block(
                        kind_tok,
                        context="wait[group]",
                    )
                return self._parse_wait_unknown_block(
                    kind_tok,
                    context=f"wait[{kind}]",
                )

            if kind == "any":
                # Intentional asymmetry: `wait[any]` only supports block form.
                # Unknown non-colon forms (e.g. wait[xyz] expr) are parsed into
                # waitmodifiercall so semantic validation can report unknown
                # modifiers uniformly at runtime.
                raise ParseError("wait[any] requires ':' block form", kind_tok)

            # Single-expr form: treat wait[all]/wait[group] as unary; use parens for
            # broader expressions or CCC if needed.
            expr = self.parse_unary_expr()
            if kind == "all":
                return Tree("waitallcall", [expr], attrs=self._modifier_attrs(kind_tok))
            if kind == "group":
                return Tree(
                    "waitgroupcall",
                    [expr],
                    attrs=self._modifier_attrs(kind_tok),
                )
            return Tree(
                "waitmodifiercall",
                [expr],
                attrs=self._modifier_attrs(kind_tok),
            )

        # wait expr / wait(expr)
        if self.match(TT.LPAR):
            expr = self.parse_expr()
            self.expect(TT.RPAR)
        else:
            expr = self.parse_unary_expr()
        return Tree("recv", [expr])

    def parse_spawn_expr(self) -> Tree:
        """Parse spawn call or block."""
        self.expect(TT.SPAWN)

        if self.match(TT.COLON):
            body = self.parse_body()
            return Tree("spawn", [body])

        if self.match(TT.LPAR):
            expr = self.parse_expr()
            self.expect(TT.RPAR)
        else:
            expr = self.parse_unary_expr()
        return Tree("spawn", [expr])

    def _parse_wait_any_block(self, modifier_tok: Tok) -> Tree:
        self._expect_indented_block("wait[any]", require_arm=True)

        arms: list[Tree] = []
        saw_default = False

        while not self.check(TT.DEDENT, TT.EOF):
            if self.match(TT.NEWLINE):
                continue

            if self.check(TT.IDENT) and self.current.value == "default":
                if saw_default:
                    raise ParseError(
                        "wait[any] has multiple default arms", self.current
                    )
                saw_default = True
                self.advance()
                self.expect(TT.COLON)
                body = self.parse_body()
                arms.append(Tree("waitany_default", [body]))
                self._consume_newlines()
                continue

            if self.check(TT.IDENT) and self.current.value == "timeout":
                self.advance()
                timeout_expr = self.parse_expr()
                self.expect(TT.COLON)
                body = self.parse_body()
                arms.append(Tree("waitany_timeout", [timeout_expr, body]))
                self._consume_newlines()
                continue

            head = self.parse_expr()
            self.expect(TT.COLON)
            body = self.parse_body()
            arms.append(Tree("waitany_arm", [head, body]))

            self._consume_newlines()

        self.expect(TT.DEDENT)
        return Tree("waitanyblock", arms, attrs=self._modifier_attrs(modifier_tok))

    def _parse_wait_named_block(
        self,
        root_label: str,
        arm_label: str,
        context: str,
        modifier_tok: Tok,
    ) -> Tree:
        self._expect_indented_block(context, require_arm=True)

        arms: list[Tree] = []

        while not self.check(TT.DEDENT, TT.EOF):
            if self.match(TT.NEWLINE):
                continue

            name_tok = self.expect(TT.IDENT)
            self.expect(TT.COLON)
            expr = self.parse_expr()
            arms.append(Tree(arm_label, [name_tok, expr]))

            self._consume_newlines()

        self.expect(TT.DEDENT)
        return Tree(root_label, arms, attrs=self._modifier_attrs(modifier_tok))

    def _parse_wait_group_block(self, modifier_tok: Tok, context: str) -> Tree:
        self._expect_indented_block(context, require_arm=True)

        arms: list[Tree] = []

        while not self.check(TT.DEDENT, TT.EOF):
            if self.match(TT.NEWLINE):
                continue

            expr = self.parse_expr()
            arms.append(Tree("waitgrouparm", [expr]))

            self._consume_newlines()

        self.expect(TT.DEDENT)
        return Tree("waitgroupblock", arms, attrs=self._modifier_attrs(modifier_tok))

    def _parse_wait_unknown_block(self, modifier_tok: Tok, context: str) -> Tree:
        # Unknown wait modifiers are still parsed by shape so semantic validation
        # can produce a precise construct-specific diagnostic later.
        self._expect_indented_block(context, require_arm=True)

        arms: list[Tree] = []

        while not self.check(TT.DEDENT, TT.EOF):
            if self.match(TT.NEWLINE):
                continue

            expr = self.parse_expr()
            arms.append(Tree("waitmodifierarm", [expr]))

            self._consume_newlines()

        self.expect(TT.DEDENT)
        return Tree("waitmodifierblock", arms, attrs=self._modifier_attrs(modifier_tok))

    def parse_return_stmt(self) -> Tree:
        """Parse return statement: return [expr | pack]"""
        return_tok = self.expect(TT.RETURN)

        # Check if there's a value — clause delimiters also end bare return
        if self.check(
            TT.NEWLINE,
            TT.EOF,
            TT.SEMI,
            TT.RBRACE,
            TT.ELSE,
            TT.ELIF,
            TT.IF,
            TT.UNLESS,
            TT.PIPE,
        ):
            return Tree("returnstmt", [return_tok])

        # Parse first expression
        first_expr = self.parse_expr()

        # Check for pack (comma-separated expressions)
        if self.check(TT.COMMA):
            exprs = [first_expr]
            while self.match(TT.COMMA):
                exprs.append(self.parse_expr())
            pack = Tree("pack", exprs)
            return Tree("returnstmt", [return_tok, pack])

        return Tree("returnstmt", [return_tok, first_expr])

    def parse_break_stmt(self) -> Tree:
        """Parse break statement"""
        return Tree("breakstmt", [self.expect(TT.BREAK)])

    def parse_continue_stmt(self) -> Tree:
        """Parse continue statement"""
        return Tree("continuestmt", [self.expect(TT.CONTINUE)])

    def parse_throw_stmt(self) -> Tree:
        """Parse throw statement: throw [expr]"""
        self.expect(TT.THROW)
        # Optional expression — treat clause delimiters as bare-throw boundaries
        if self.current.type in self._BARE_STMT_TERMINATORS:
            return Tree("throwstmt", [])
        value = self.parse_expr()
        return Tree("throwstmt", [value])

    def parse_assert_stmt(self) -> Tree:
        """Parse assert: assert expr [, message]"""
        self.expect(TT.ASSERT)
        value = self.parse_expr()
        # Optional comma and message expression
        if self.match(TT.COMMA):
            message = self.parse_expr()
            return Tree("assert", [value, message])
        return Tree("assert", [value])

    def parse_dbg_stmt(self) -> Tree:
        """Parse dbg: DBG (expr ("," expr)?)"""
        dbg_tok = self.expect(TT.DBG)
        first_expr = self.parse_expr()
        if self.match(TT.COMMA):
            second_expr = self.parse_expr()
            return Tree("dbg", [dbg_tok, first_expr, second_expr])
        return Tree("dbg", [dbg_tok, first_expr])

    def parse_guard_chain(self) -> Tree:
        """
        Parse guard chain (inline if-else):
        expr : body | expr : body |: else
        Supports multi-line continuation with | or |:
        """
        branches = []

        # Parse first branch
        cond = self.parse_expr()
        self.expect(TT.COLON)
        body = self.parse_body()
        branches.append(Tree("guardbranch", [cond, body]))

        # Parse additional branches (allow newlines before | or |:)
        while True:
            # Lookahead: skip newlines and check for continuation
            idx = self.pos
            paren_depth = 0

            # Skip newlines
            tok, idx, paren_depth = self._lookahead_advance(idx, paren_depth)
            while tok.type == TT.NEWLINE:
                tok, idx, paren_depth = self._lookahead_advance(idx, paren_depth)

            # Check for |
            if tok.type == TT.PIPE:
                # Consume the newlines and pipe for real
                while self.match(TT.NEWLINE):
                    pass

                if self.match(TT.PIPE):
                    if self.check(TT.COLON):  # |: for else
                        self.advance()
                        else_body = self.parse_body()
                        branches.append(else_body)
                        break

                    # Another conditional branch
                    cond = self.parse_expr()
                    self.expect(TT.COLON)
                    body = self.parse_body()
                    branches.append(Tree("guardbranch", [cond, body]))
            else:
                # No continuation, exit without consuming tokens
                break

        return Tree("onelineguard", branches)

    def parse_body(self) -> Tree:
        """
        Parse body after colon - implements "colon chooses" rule.

        Accepts:
        - Inline brace block: { stmts }
        - Indented block: INDENT stmts DEDENT
        - Single statement: stmt
        """
        # Skip newlines before block
        while self.match(TT.NEWLINE):
            pass

        if self.match(TT.LBRACE):
            # Inline block: { stmts }
            # Build a stmtlist and wrap in inlinebody to match the expected AST
            stmts: list[Node] = []
            while not self.check(TT.RBRACE, TT.EOF):
                if self.match(TT.NEWLINE):
                    continue
                if self.match(TT.SEMI):
                    stmts.append(self._tok("SEMI", ";"))
                    continue
                stmt = self.parse_statement()
                stmts.append(Tree("stmt", [stmt]))
            self.expect(TT.RBRACE)
            # Wrap stmtlist in body (inline)
            if stmts:
                return Tree("body", [Tree("stmtlist", stmts)], attrs={"inline": True})
            return Tree("body", [], attrs={"inline": True})

        if self.check(TT.INDENT):
            # Indented block
            indent_tok = self.advance()
            # Use actual indent value from lexer
            children: list[Node] = [
                self._tok(
                    "INDENT",
                    indent_tok.value if indent_tok.value is not None else "    ",
                )
            ]

            # Parse statements in the block
            while not self.check(TT.DEDENT, TT.EOF):
                if self.match(TT.NEWLINE):
                    continue
                if self.match(TT.SEMI):
                    children.append(self._tok("SEMI", ";"))
                    continue
                stmt = self.parse_statement()
                children.append(Tree("stmt", [stmt]))
                # Skip newlines between statements
                while self.check(TT.NEWLINE) and not self.check(TT.DEDENT):
                    self.match(TT.NEWLINE)

            dedent_tok = self.expect(TT.DEDENT)
            children.append(
                self._tok(
                    "DEDENT", dedent_tok.value if dedent_tok.value is not None else ""
                )
            )
            return Tree("body", children, attrs={"inline": False})

        # Single statement inline
        old_inline = self.in_inline_body
        self.in_inline_body = True
        stmt = self.parse_statement()
        self.in_inline_body = old_inline
        return Tree("body", [stmt], attrs={"inline": True})

    def _parse_required_indented_body(self, context: str) -> Tree:
        """Parse body and require the next non-newline token to be INDENT."""
        lookahead = 0
        while self.peek(lookahead).type == TT.NEWLINE:
            lookahead += 1

        next_tok = self.peek(lookahead)
        if next_tok.type != TT.INDENT:
            raise ParseError(f"{context} requires indented block form", next_tok)

        return self.parse_body()

    # ========================================================================
    # Expressions - Precedence Climbing
    # ========================================================================

    def parse_expr(self) -> Tree:
        """
        Parse expression (top level).
        Handles catch, ternary, binary ops, etc.
        """
        # Wrap in expr node to match evaluator expectations
        return Tree("expr", [self.parse_catch_expr()])

    def parse_catch_expr(self) -> Node:
        """Parse catch expression: expr catch [types] [bind x]: handler"""
        expr = self.parse_ternary_expr()

        if not self._check_catch_start():
            return expr

        self._consume_catch_keyword()
        types, binder = self._parse_catch_filter()
        self.expect(TT.COLON)

        # Peek past newlines to decide catchstmt vs catchexpr
        is_stmt = self.check(TT.LBRACE, TT.INDENT)
        if not is_stmt and self.check(TT.NEWLINE):
            k = 1
            while self.peek(k).type == TT.NEWLINE:
                k += 1
            is_stmt = self.peek(k).type == TT.INDENT
        handler = self.parse_body()

        children = self._build_catch_children(expr, types, binder, handler)
        return Tree("catchstmt" if is_stmt else "catchexpr", children)

    def parse_catch_stmt(self) -> Tree:
        """Parse catch statement starting at current position."""
        try_expr = self.parse_expr()

        if self.check(TT.NEWLINE):
            # Permit catch after blank lines when it immediately follows the expression.
            k = 0
            while self.peek(k).type == TT.NEWLINE:
                k += 1
            if not self._check_catch_at(k):
                return try_expr
            while self.match(TT.NEWLINE):
                pass

        if not self._check_catch_start():
            return try_expr

        self._consume_catch_keyword()
        types, binder = self._parse_catch_filter()
        self.expect(TT.COLON)
        body = self.parse_body()

        return Tree(
            "catchstmt", self._build_catch_children(try_expr, types, binder, body)
        )

    def parse_try_stmt(self) -> Tree:
        """Parse try statement: try: INDENT...DEDENT catch ... : INDENT...DEDENT"""
        self.expect(TT.TRY)
        self.expect(TT.COLON)
        try_body = self._parse_required_indented_body("try")

        if not self.check(TT.CATCH):
            raise ParseError("try requires a catch clause", self.current)

        self.advance()  # consume 'catch'
        types, binder = self._parse_catch_filter()
        self.expect(TT.COLON)
        handler = self._parse_required_indented_body("try catch")

        return Tree(
            "trystmt", self._build_catch_children(try_body, types, binder, handler)
        )

    def parse_ternary_expr(self) -> Node:
        """Parse ternary: expr ? then : else"""
        expr = self.parse_or_expr()

        if self.match(TT.QMARK):
            then_expr = self.parse_expr()
            self.expect(TT.COLON)
            else_expr = self.parse_ternary_expr()  # Right associative
            return Tree("ternary", [expr, then_expr, else_expr])

        # No ternary, return unwrapped
        return expr

    def parse_or_expr(self) -> Node:
        """Parse logical OR: expr || expr"""
        return self._parse_flat_binop("or", TT.OR, self.parse_and_expr)

    def parse_and_expr(self) -> Node:
        """Parse logical AND: expr && expr"""
        return self._parse_flat_binop("and", TT.AND, self.parse_bind_expr)

    def parse_bind_expr(self) -> Node:
        """Parse apply bind: lvalue .= expr"""
        left = self.parse_send_expr()

        if self.match(TT.APPLYASSIGN):
            lvalue = self._expr_to_lvalue(left)

            right = self.parse_bind_expr()  # Right associative
            return Tree("bind", [lvalue, right])

        # No bind, just return walrus level
        return left

    def parse_send_expr(self) -> Node:
        """Parse channel send: expr -> expr"""
        left = self.parse_walrus_expr()

        while self.match(TT.SEND):
            right = self.parse_walrus_expr()
            left = Tree("send", [left, right])

        return left

    def _expr_to_lvalue(self, expr: Tree | Tok) -> Tree:
        """
        Convert an expression node into an lvalue tree, preserving chain
        structure and normalizing index ops to lv_index for assignment
        handling.
        """
        core_expr: Tree | Tok = expr

        # Unwrap single-child precedence wrappers (expr, ternaryexpr, etc.)
        while isinstance(core_expr, Tree) and len(core_expr.children) == 1:
            core_expr = core_expr.children[0]

        if isinstance(core_expr, Tok) and core_expr.type == TT.IDENT:
            return Tree("lvalue", [core_expr])

        if isinstance(core_expr, Tree) and core_expr.data in {
            "explicit_chain",
            "implicit_chain",
        }:
            if not core_expr.children:
                return Tree("lvalue", [core_expr])

            head, *ops = core_expr.children

            def _copy_meta(src: Tree, dst: Tree) -> Tree:
                if hasattr(src, "meta"):
                    dst.meta = src.meta
                return dst

            def _convert_valuefan_to_fieldlist(valuefan: Tree) -> Tree:
                """Convert valuefan(valuefan_list(...)) to fieldlist(IDENT, ...)."""
                idents: List[Tok] = []
                for child in valuefan.children:
                    if isinstance(child, Tree) and child.data == "valuefan_list":
                        for item in child.children:
                            if isinstance(item, Tree) and item.data == "valuefan_item":
                                for inner in item.children:
                                    if (
                                        isinstance(inner, Tok)
                                        and inner.type == TT.IDENT
                                    ):
                                        idents.append(inner)
                                    else:
                                        raise ParseError(
                                            "Field fan in assignment only supports identifiers",
                                            (
                                                inner
                                                if isinstance(inner, Tok)
                                                else self.current
                                            ),
                                        )
                            elif isinstance(item, Tok) and item.type == TT.IDENT:
                                idents.append(item)
                            # Skip commas
                return Tree("fieldlist", idents)

            norm_ops: List[Tree | Tok] = []
            for op in ops:
                if isinstance(op, Tree) and op.data == "index":
                    norm_ops.append(_copy_meta(op, Tree("lv_index", list(op.children))))
                elif isinstance(op, Tree) and op.data == "noanchor":
                    inner = op.children[0]
                    if isinstance(inner, Tree) and inner.data == "index":
                        lv_idx = _copy_meta(
                            inner, Tree("lv_index", list(inner.children))
                        )
                        norm_ops.append(_copy_meta(op, Tree("noanchor", [lv_idx])))
                    else:
                        norm_ops.append(op)
                elif isinstance(op, Tree) and op.data == "valuefan":
                    # valuefan is expression fan-out; for assignment convert to fieldfan
                    # with proper fieldlist structure
                    fieldlist = _convert_valuefan_to_fieldlist(op)
                    norm_ops.append(_copy_meta(op, Tree("fieldfan", [fieldlist])))
                else:
                    norm_ops.append(op)

            return Tree("lvalue", [head] + norm_ops)

        return Tree("lvalue", [expr])

    def _lvalue_allows_rhs_pack(self, lvalue: Tree) -> bool:
        """Whether assignment RHS may parse comma-separated pack values."""
        if not lvalue.children:
            return False

        head, *ops = lvalue.children
        core_head: Tree | Tok = head

        while (
            isinstance(core_head, Tree)
            and core_head.data in {"group", "group_expr", "expr", "primary"}
            and len(core_head.children) == 1
        ):
            core_head = core_head.children[0]

        if isinstance(core_head, Tree) and core_head.data == "fan_literal":
            return True

        for raw_op in ops:
            op = raw_op
            if isinstance(op, Tree) and op.data == "noanchor" and op.children:
                op = op.children[0]
            if isinstance(op, Tree) and op.data in {"fieldfan", "valuefan"}:
                return True

        return False

    def _parse_assignment_rhs(self, lvalue: Tree) -> Node:
        """Parse assignment RHS; allow `expr, expr` packs for fan-shaped lvalues."""
        allow_pack = self._lvalue_allows_rhs_pack(lvalue)
        if not allow_pack:
            return self.parse_nullish_expr()

        saved_context = self._push_parse_context(ParseContext.DESTRUCTURE_PACK)
        try:
            rhs = self.parse_nullish_expr()
            if not self.check(TT.COMMA):
                return rhs

            exprs: List[Node] = [rhs]
            while self.match(TT.COMMA):
                exprs.append(self.parse_nullish_expr())

            return Tree("pack", exprs)
        finally:
            self._pop_parse_context(saved_context)

    def parse_walrus_expr(self) -> Node:
        """Parse walrus: x := expr"""
        # Check for walrus pattern: IDENT :=
        if self.check(TT.IDENT) and self.peek(1).type == TT.WALRUS:
            name = self.current
            self.advance(2)  # consume IDENT and :=
            # Parse the RHS
            value = self.parse_catch_expr()  # allow catch expressions in walrus RHS
            return Tree("walrus", [name, value])

        # No walrus, just return nullish level
        return self.parse_nullish_expr()

    def parse_nullish_expr(self) -> Node:
        """Parse nullish coalescing: expr ?? expr"""
        left = self.parse_compare_expr()

        if not self.check(TT.NULLISH):
            # No nullish operator, just return compare level
            return left

        # Collect all operands and operators into flat list
        children: list[Node] = [left]
        while self.check(TT.NULLISH):
            op = self.advance()
            children.append(op)
            right = self.parse_compare_expr()
            children.append(right)

        return Tree("nullish", children)

    def parse_compare_expr(self) -> Node:
        """
        Parse comparison with chained comparison chains (CCC):
        x == 5, and < 10, or > 100
        """
        left = self.parse_add_expr()

        # Check for comparison operator
        if not self.is_compare_op():
            # No comparison, return unwrapped
            return left

        # Parse CCC
        op_tokens = self.parse_compare_op()
        right = self.parse_add_expr()

        # Wrap operator tokens in cmpop tree
        op_tree = Tree("cmpop", op_tokens)
        children = [left, op_tree, right]

        # Check for comma-chained comparisons (only if comma is CCC)
        if self.comma_is_ccc():
            # CCC chain - consume the comma
            self.match(TT.COMMA)
            while True:
                # Check for and/or and capture as token
                or_token = None
                and_token = None
                if self.check(TT.OR):
                    or_token = self.advance()
                elif self.check(TT.AND):
                    and_token = self.advance()

                # Parse next leg
                leg_op_tokens = (
                    self.parse_compare_op() if self.is_compare_op() else None
                )
                leg_value = self.parse_add_expr()

                leg_parts: list[Node] = []
                if or_token:
                    leg_parts.append(or_token)
                    if leg_op_tokens:
                        leg_op = Tree("cmpop", leg_op_tokens)
                        leg_parts.extend([leg_op, leg_value])
                    else:
                        leg_parts.append(leg_value)
                    children.append(Tree("ccc_or_leg", leg_parts))
                else:
                    if and_token:
                        leg_parts.append(and_token)
                    if leg_op_tokens:
                        leg_op = Tree("cmpop", leg_op_tokens)
                        leg_parts.extend([leg_op, leg_value])
                    else:
                        leg_parts.append(leg_value)
                    children.append(Tree("ccc_and_leg", leg_parts))

                # Check if next comma is also CCC
                if not self.comma_is_ccc():
                    break
                self.match(TT.COMMA)  # consume it

            return Tree("compare", children)

        return Tree("compare", [left, op_tree, right])

    def comma_is_ccc(self) -> bool:
        """
        Check if comma at current position starts a CCC leg vs being a separator.

        CCC is allowed when:
        - Context is NORMAL (not in function args, array elements, or destructure pack)
        - Followed by explicit CCC markers: `, or` / `, and` / `, <cmpop>`
        - OR followed by bare addexpr in NORMAL context (implicit AND leg)
        """
        if not self.check(TT.COMMA):
            return False

        # Peek at token after comma
        next_tok = self.peek(1)

        # `, or` => definitely CCC OR leg
        if next_tok.type == TT.OR:
            return True

        # `, and` => definitely CCC AND leg (explicit)
        if next_tok.type == TT.AND:
            return True

        # Check if followed by comparison operator (`, <cmpop>` => implicit AND)
        # Lookahead past the comma
        _, idx, _ = self._lookahead_advance(self.pos, 0)  # skip comma
        is_cmp = idx < len(self.tokens) and self._is_compare_op_at(idx)

        if is_cmp:
            return True  # `, <cmpop>` => CCC implicit AND

        # At this point: `, <addexpr>` (bare value, no cmpop)
        # This is ambiguous - use context to decide:
        # - In separator contexts (args/array/pack): NOT CCC
        # - In normal context: IS CCC (implicit AND with bare addexpr)
        if self.parse_context in (
            ParseContext.FUNCTION_ARGS,
            ParseContext.ARRAY_ELEMENTS,
            ParseContext.DESTRUCTURE_PACK,
        ):
            return False  # Comma is a separator

        return True  # Default: allow CCC in normal context

    def _is_compare_op_at(self, idx: int) -> bool:
        """Check if token at idx is a comparison operator (stateless helper)"""
        if idx >= len(self.tokens):
            return False
        tok = self.tokens[idx]
        if tok.type in {
            TT.EQ,
            TT.NEQ,
            TT.LT,
            TT.LTE,
            TT.GT,
            TT.GTE,
            TT.IS,
            TT.IN,
            TT.TILDE,
            TT.REGEXMATCH,
        }:
            return True
        if (
            tok.type == TT.NOT
            and idx + 1 < len(self.tokens)
            and self.tokens[idx + 1].type == TT.IN
        ):
            return True
        if (
            tok.type == TT.NEG
            and idx + 1 < len(self.tokens)
            and self.tokens[idx + 1].type in {TT.IS, TT.IN}
        ):
            return True
        return False

    def is_compare_op(self) -> bool:
        """Check if current token is a comparison operator"""
        if (
            self._pattern_default_paren_depth >= 0
            and self.paren_depth <= self._pattern_default_paren_depth
            and self.check(TT.TILDE, TT.REGEXMATCH)
        ):
            return False
        return self._is_compare_op_at(self.pos)

    def parse_compare_op(self) -> List[Tok]:
        """Parse comparison operator - returns list of tokens for compound ops"""
        if self.check(
            TT.EQ, TT.NEQ, TT.LT, TT.LTE, TT.GT, TT.GTE, TT.TILDE, TT.REGEXMATCH
        ):
            return [self.advance()]

        if self.check(TT.IS):
            is_tok = self.advance()
            if self.check(TT.NOT):
                not_tok = self.advance()
                return [is_tok, not_tok]
            return [is_tok]

        if self.check(TT.IN):
            return [self.advance()]

        if self.check(TT.NOT):
            not_tok = self.advance()
            in_tok = self.expect(TT.IN)
            return [not_tok, in_tok]

        # Check for !is and !in
        if self.check(TT.NEG):  # ! token
            neg_tok = self.advance()
            if self.check(TT.IS):
                is_tok = self.advance()
                return [neg_tok, is_tok]
            elif self.check(TT.IN):
                in_tok = self.advance()
                return [neg_tok, in_tok]
            else:
                raise ParseError("Expected 'is' or 'in' after '!'", self.current)

        raise ParseError("Expected comparison operator", self.current)

    def parse_add_expr(self) -> Node:
        """Parse addition/subtraction: expr + expr"""
        return self._parse_left_assoc(
            "add",
            "addop",
            (TT.PLUS, TT.MINUS, TT.DEEPMERGE, TT.CARET),
            self.parse_mul_expr,
        )

    def parse_mul_expr(self) -> Node:
        """Parse multiplication/division: expr * expr"""
        return self._parse_left_assoc(
            "mul",
            "mulop",
            (TT.STAR, TT.SLASH, TT.FLOORDIV, TT.MOD),
            self.parse_pow_expr,
        )

    def parse_pow_expr(self) -> Node:
        """Parse exponentiation: expr ** expr (right associative)"""
        base = self.parse_unary_expr()

        if self.check(TT.POW):
            pow_tok = self.advance()
            exp = self.parse_pow_expr()  # Right associative
            return Tree("pow", [base, pow_tok, exp])

        # No power operator, return unwrapped
        return base

    def parse_unary_expr(self) -> Node:
        """Parse unary operators: -expr, !expr, ++expr, wait expr, <-expr, spawn expr"""
        # Throw as expression-form
        if self.check(TT.THROW):
            self.advance()
            if self.current.type in self._BARE_STMT_TERMINATORS:
                return Tree("throwstmt", [])
            value = self.parse_unary_expr()
            return Tree("throwstmt", [value])

        # Wait
        if self.check(TT.WAIT):
            return self.parse_wait_expr()

        # Receive
        if self.match(TT.RECV):
            expr = self.parse_unary_expr()
            return Tree("recv", [expr])

        # Spawn
        if self.check(TT.SPAWN):
            return self.parse_spawn_expr()

        # $ (no anchor)
        if self.match(TT.DOLLAR):
            expr = self.parse_unary_expr()
            return Tree("noanchor_expr", [expr])

        # Spread prefix
        if self.match(TT.SPREAD):
            expr = self.parse_unary_expr()
            return Tree("spread", [expr])

        # Unary prefix operators
        if self.check(TT.MINUS, TT.NOT, TT.NEG, TT.INCR, TT.DECR):
            op = self.advance()
            if op.type == TT.MINUS:
                # Allow -2^63 (i64 min) by folding into a single signed literal.
                tok = self._try_consume_int64_min_literal()
                if tok:
                    return self._tok(
                        "NUMBER",
                        -(2**63),
                        line=tok.line,
                        column=tok.column,
                    )
            expr = self.parse_unary_expr()
            op_tree = Tree("unaryprefixop", [op])
            return Tree("unary", [op_tree, expr])

        # No unary operator, return unwrapped
        return self.parse_postfix_expr()

    def parse_postfix_expr(self) -> Node:
        """
        Parse postfix expressions:
        - field access: expr.field
        - indexing: expr[index]
        - calls: expr(args)
        - postfix incr/decr: expr++
        - implicit chains: .field, .(args), .[index]
        - standalone subject: .

        Output as explicit_chain, implicit_chain, or subject
        """
        # Check for implicit chain or standalone subject
        if self.check(TT.DOT):
            next_tok = self.peek(1)

            # Standalone subject: just .
            if next_tok.type not in (
                TT.IDENT,
                TT.LPAR,
                TT.LSQB,
                TT.LBRACE,
                TT.OVER,
                TT.FAN,
                TT.DOLLAR,
            ):
                dot = self.advance()
                return Tree(
                    "subject", [self._tok("DOT", dot.value, dot.line, dot.column)]
                )

            # Implicit chain: .field, .(args), .[index], .over
            self.advance()  # consume .
            seen_noanchor = [False]
            imphead = self._parse_implicit_chain_head(seen_noanchor)
            postfix_ops = [imphead]
            self._collect_postfix_ops(postfix_ops, seen_noanchor)

            return Tree("implicit_chain", postfix_ops)

        # Explicit chain: primary + postfix ops
        primary = self.parse_primary_expr()
        postfix_ops: List[Node] = []
        seen_noanchor = [False]
        self._collect_postfix_ops(postfix_ops, seen_noanchor, allow_amp=True)

        if not postfix_ops:
            return primary

        return Tree("explicit_chain", [primary] + postfix_ops)

    # ------------------------------------------------------------------
    # Primary expression helpers
    # ------------------------------------------------------------------

    def _parse_numeric_literal(self) -> Node:
        """Parse NUMBER/DURATION/SIZE token with overflow checks."""
        tok = self.advance()

        if tok.type == TT.NUMBER:
            value = self._parse_number_literal(tok.value)
            if isinstance(value, int) and value > 2**63 - 1:
                raise LexError(
                    "Integer literal overflows int64",
                    line=tok.line,
                    column=tok.column,
                )
            return self._tok("NUMBER", value, line=tok.line, column=tok.column)

        # DURATION or SIZE — check pre-computed overflow
        raw, total = tok.value
        if total < -(2**63) or total > 2**63 - 1:
            kind = "Duration" if tok.type == TT.DURATION else "Size"
            raise LexError(
                f"{kind} literal overflows int64",
                line=tok.line,
                column=tok.column,
            )

        return self._tok(tok.type.name, tok.value, line=tok.line, column=tok.column)

    def _parse_set_primary(self) -> Tree:
        """Parse set literal or set comprehension — SET already consumed."""
        self.expect(TT.LBRACE)

        if self.check(TT.RBRACE):
            self.advance()
            return Tree("setliteral", [])

        first_expr = self.parse_expr()

        if self.check(TT.FOR, TT.OVER):
            comphead = self.parse_comphead()
            ifclause = self.parse_ifclause_opt()
            self.expect(TT.RBRACE)
            children = [first_expr, comphead]
            if ifclause:
                children.append(ifclause)
            return Tree("setcomp", children)

        items = [first_expr]
        while self.match(TT.COMMA):
            if self.check(TT.RBRACE):
                break
            items.append(self.parse_expr())
        self.expect(TT.RBRACE)

        return Tree("setliteral", items)

    def _parse_paren_primary(self) -> Tree:
        """Parse parenthesized expression — LPAR already consumed."""
        saved_context = self._push_parse_context(ParseContext.NORMAL)
        try:
            expr = self.parse_expr()
            self.expect(TT.RPAR)
            return Tree("group_expr", [expr])
        finally:
            self._pop_parse_context(saved_context)

    def _parse_array_primary(self) -> Tree:
        """Parse array literal or list comprehension — LSQB already consumed."""
        self.skip_layout_tokens()

        if self.check(TT.RSQB):
            self.advance()
            return Tree("array", [])

        saved_context = self._push_parse_context(ParseContext.ARRAY_ELEMENTS)
        try:
            first_elem = self.parse_expr()

            if self.check(TT.FOR, TT.OVER):
                comphead = self.parse_comphead()
                ifclause = self.parse_ifclause_opt()
                self.expect(TT.RSQB)
                children = [first_elem, comphead]
                if ifclause:
                    children.append(ifclause)
                return Tree("listcomp", children)

            elements = [first_elem]
            self.skip_layout_tokens()
            while True:
                if not self.match(TT.COMMA):
                    break
                self.skip_layout_tokens()
                if self.check(TT.RSQB, TT.EOF):
                    break
                elements.append(self.parse_expr())
                self.skip_layout_tokens()

            self.expect(TT.RSQB)
            return Tree("array", elements)
        finally:
            self._pop_parse_context(saved_context)

    def _parse_object_primary(self) -> Tree:
        """Parse object literal or dict comprehension — LBRACE already consumed."""
        # Try dict comprehension detection
        snapshot = self._snapshot()

        try:
            first_key = self.parse_expr()
            if self.check(TT.COLON):
                self.advance()
                first_val = self.parse_expr()
                if self.check(TT.FOR, TT.OVER):
                    comphead = self.parse_comphead()
                    ifclause = self.parse_ifclause_opt()
                    self.expect(TT.RBRACE)
                    children = [first_key, first_val, comphead]
                    if ifclause:
                        children.append(ifclause)
                    return Tree("dictcomp", children)
        except ParseError:
            pass

        # Reset cursor if not dictcomp
        self._restore(snapshot)

        items: List[Node] = []
        while not self.check(TT.RBRACE, TT.EOF):
            self.skip_layout_tokens()
            if self.check(TT.RBRACE):
                break

            items.append(self.parse_object_item())

            self.skip_layout_tokens()
            if not self.match(TT.COMMA):
                self.skip_layout_tokens()
                if self.check(TT.RBRACE):
                    break
                continue

        self.expect(TT.RBRACE)

        return Tree("object", items)

    # ------------------------------------------------------------------

    def parse_primary_expr(self) -> Node:
        """
        Parse primary expressions:
        - Literals (numbers, strings, true, false, nil)
        - Identifiers
        - Parenthesized expressions
        - Arrays, objects
        - Comprehensions
        - Rebind primary (=ident or =(lvalue))
        - etc.
        """
        if self.check(TT.MATCH):
            return self.parse_match_expr()

        # Rebind primary: =ident or =(lvalue)
        if self.match(TT.ASSIGN):
            if self.match(TT.LPAR):
                lvalue = self.parse_rebind_lvalue()
                self.expect(TT.RPAR)
                return Tree("rebind_primary", [lvalue], attrs={"grouped": True})
            ident = self.expect(TT.IDENT)
            return Tree("rebind_primary", [ident], attrs={"grouped": False})

        # Null-safe chain: ??(expr)
        if self.match(TT.NULLISH):
            self.expect(TT.LPAR)
            inner = self.parse_expr()
            self.expect(TT.RPAR)
            return Tree("nullsafe", [inner])

        # Emit expression: > arglist
        if self.check(TT.GT):
            if self.call_depth == 0:
                raise ParseError(
                    "Emit '>' is only valid inside a call block", self.current
                )
            self.advance()
            args = self.parse_emit_arg_list()
            if args:
                args = [Tree("arglistnamedmixed", args)]
            else:
                args = []
            return Tree("emitexpr", args)

        # Import expression
        if self.match(TT.IMPORT):
            return Tree("import_expr", [self._parse_import_string_literal()])

        # Numeric / duration / size literals
        if self.check(TT.NUMBER, TT.DURATION, TT.SIZE):
            return self._parse_numeric_literal()

        # Hole placeholder
        if self.match(TT.QMARK):
            return Tree("holeexpr", [])

        # String-like literals
        if self.check(
            TT.STRING,
            TT.RAW_STRING,
            TT.RAW_HASH_STRING,
            TT.SHELL_STRING,
            TT.SHELL_BANG_STRING,
            TT.PATH_STRING,
            TT.ENV_STRING,
            TT.REGEX,
        ):
            return Tree("literal", [self.advance()])

        # Boolean / nil
        if self.match(TT.TRUE):
            return self._tok("TRUE", "true")
        if self.match(TT.FALSE):
            return self._tok("FALSE", "false")
        if self.match(TT.NIL):
            return self._tok("NIL", "nil")

        # Set literal / comprehension
        if self.match(TT.SET):
            return self._parse_set_primary()

        # Fan literal
        if self.match(TT.FAN):
            return self.parse_fan_literal()

        # Identifiers
        if self.check(TT.IDENT):
            return self.advance()

        # Special identifiers
        if self.match(TT.OVER):
            return self._tok("OVER", "_")

        # Parenthesized expression
        if self.match(TT.LPAR):
            return self._parse_paren_primary()

        # Array literal / list comprehension
        if self.match(TT.LSQB):
            return self._parse_array_primary()

        # Object literal / dict comprehension
        if self.match(TT.LBRACE):
            return self._parse_object_primary()

        # Anonymous fn literal (expression form)
        if self.match(TT.FN):
            return self.parse_anonymous_fn_decl()

        # Anonymous functions - & for lambda
        if self.match(TT.AMP):
            return self.parse_anonymous_fn()

        # Selector literal: `0:10` or `{start}:{stop}`
        if self.match(TT.BACKQUOTE):
            selectors = self.parse_selector_literal_content()
            self.expect(TT.BACKQUOTE)
            return Tree("selectorliteral", [Tree("sellist", selectors)])

        raise ParseError(
            f"Unexpected token in expression: {self.current.type.name}", self.current
        )

    def parse_fan_literal(self) -> Tree:
        """Parse fan literal: fan [modifier] { expr, ... }"""
        modifiers = None

        if self.match(TT.LSQB):
            mod_tok = self.expect(TT.IDENT)
            self.expect(TT.RSQB)
            modifiers = Tree(
                "fan_modifiers",
                [self._tok("IDENT", mod_tok.value, mod_tok.line, mod_tok.column)],
                attrs=self._modifier_attrs(mod_tok),
            )

        self.expect(TT.LBRACE)
        self.skip_layout_tokens()

        items: List[Node] = []

        if not self.check(TT.RBRACE):
            saved_context = self._push_parse_context(ParseContext.ARRAY_ELEMENTS)
            try:
                items.append(self.parse_expr())
                self.skip_layout_tokens()

                while self.match(TT.COMMA):
                    self.skip_layout_tokens()
                    if self.check(TT.RBRACE, TT.EOF):
                        break
                    items.append(self.parse_expr())
                    self.skip_layout_tokens()
            finally:
                self._pop_parse_context(saved_context)

        self.expect(TT.RBRACE)

        children: List[Node] = []
        if modifiers:
            children.append(modifiers)
        children.append(Tree("fan_items", items))
        return Tree("fan_literal", children)

    # ========================================================================
    # Helper Parsers
    # ========================================================================

    def parse_rebind_lvalue(self) -> Tree:
        """
        Parse rebind lvalue: IDENT (postfix_field | postfix_index)* (fieldfan)?
        Used in =(lvalue) expressions
        """
        ident = self.expect(TT.IDENT)
        children: list[Node] = [ident]
        seen_noanchor = [False]

        # Parse postfix operations (field, index)
        while True:
            if self.match(TT.DOT):
                noanchor = self._take_noanchor_once(seen_noanchor)
                children.append(
                    self._maybe_noanchor(
                        self._field_node(self.expect(TT.IDENT)),
                        noanchor,
                    )
                )
            elif self.check(TT.DOLLAR) or self.check(TT.LSQB):
                noanchor = self._take_noanchor_once(seen_noanchor)
                if not self.check(TT.LSQB):
                    raise ParseError(
                        "Expected field or index after '$' in chain", self.current
                    )
                children.append(
                    self._maybe_noanchor(
                        self._parse_plain_index_node(),
                        noanchor,
                    )
                )
            else:
                break

        # Optional fieldfan: .{field1, field2}
        if self.check(TT.DOT):
            # Peek ahead to see if it's followed by {
            if self.peek(1).type == TT.LBRACE:
                fieldfan = self.parse_fieldfan()
                children.append(fieldfan)

        return Tree("rebind_lvalue", children)

    def parse_fieldfan(self) -> Tree:
        """Parse fieldfan: .{fieldlist} where fieldlist is IDENT ("," IDENT)*"""
        dot_tok = self.expect(TT.DOT)  # Capture DOT token
        self.expect(TT.LBRACE)

        # Parse fieldlist
        fields = []
        fields.append(self.expect(TT.IDENT))

        while self.match(TT.COMMA):
            fields.append(self.expect(TT.IDENT))

        self.expect(TT.RBRACE)

        # Build fieldlist tree - use captured tokens directly
        fieldlist = Tree("fieldlist", fields)
        return Tree("fieldfan", [dot_tok, fieldlist])

    def parse_param_list(self, end_token: TT = TT.RPAR) -> Tree:
        """Parse function parameter list with optional contracts"""
        params: List[Node] = []

        while not self.check(end_token, TT.EOF):
            params.append(self._parse_param_entry())

            if not self.match(TT.COMMA):
                break

        return Tree("paramlist", params)

    def _parse_param_entry(self) -> Node:
        if self.check(TT.LBRACE):
            return self._parse_param_destruct()

        if self.match(TT.LPAR):
            # Parenthesized parameter entry has two modes:
            # 1) isolated single param: (x), (x ~ T), ({a})
            # 2) grouped contract sugar: (a, b) ~ T, ({a}, b) ~ T
            if self.check(TT.LBRACE):
                inner_node = self._parse_param_destruct()
                has_default = False
                has_contract = self._param_destruct_has_contract(inner_node)
                is_spread = False
            else:
                inner_node, has_default, has_contract, is_spread = (
                    self._parse_param_atom()
                )

            if self.match(TT.COMMA):
                if has_default or has_contract or is_spread:
                    raise ParseError(
                        "Grouped parameters cannot include defaults, contracts, or spread",
                        self.current,
                    )
                if is_tree(inner_node) and tree_label(inner_node) == "param_destruct":
                    if not self._param_destruct_is_bare(inner_node):
                        raise ParseError(
                            "Grouped parameters cannot include defaults or contracts",
                            self.current,
                        )

                group_items: List[Node] = [inner_node]

                while True:
                    if self.check(TT.RPAR):
                        raise ParseError(
                            "Grouped parameters require at least two names",
                            self.current,
                        )

                    if self.check(TT.SPREAD):
                        raise ParseError(
                            "Grouped parameters cannot include spread",
                            self.current,
                        )

                    if self.check(TT.LBRACE):
                        # Grouped entries may include destruct items, but those
                        # items must be "bare" so the trailing group contract is
                        # the only contract source for the group.
                        group_item = self._parse_param_destruct()
                        if not self._param_destruct_is_bare(group_item):
                            raise ParseError(
                                "Grouped parameters cannot include defaults or contracts",
                                self.current,
                            )
                        group_items.append(group_item)
                    else:
                        ident = self.expect(TT.IDENT)
                        group_items.append(ident)

                    if self.check(TT.ASSIGN, TT.TILDE):
                        raise ParseError(
                            "Grouped parameters cannot include defaults or contracts",
                            self.current,
                        )

                    if not self.match(TT.COMMA):
                        break

                self.expect(TT.RPAR)

                if self.match(TT.ASSIGN):
                    raise ParseError(
                        "Grouped parameters cannot have a default", self.current
                    )

                if not self.match(TT.TILDE):
                    raise ParseError(
                        "Grouped parameters require a trailing contract", self.current
                    )

                contract = self.parse_expr()

                if self.check(TT.ASSIGN):
                    raise ParseError(
                        "Default must precede contract in parameter metadata",
                        self.current,
                    )

                return Tree(
                    "param_group",
                    [Tree("paramlist", group_items), Tree("contract", [contract])],
                )

            self.expect(TT.RPAR)

            outer_default = None
            outer_contract = None

            if self.match(TT.ASSIGN):
                outer_default = self.parse_expr()

            if self.match(TT.TILDE):
                outer_contract = self.parse_expr()
                if self.check(TT.ASSIGN):
                    raise ParseError(
                        "Default must precede contract in parameter metadata",
                        self.current,
                    )

            if is_tree(inner_node) and tree_label(inner_node) == "param_destruct":
                if outer_default is not None:
                    raise ParseError(
                        "Destructuring parameter cannot have an outer default",
                        self.current,
                    )
                if outer_contract is not None:
                    # Keep destruct params in native shape by attaching outer
                    # contract directly onto param_destruct rather than forcing
                    # them through _merge_param_metadata (which is IDENT-centric).
                    if self._param_destruct_has_contract(inner_node):
                        raise ParseError(
                            "Parameter cannot have multiple contracts", self.current
                        )
                    inner_node.children.append(Tree("contract", [outer_contract]))
                return Tree("param_isolated", [inner_node])

            merged = self._merge_param_metadata(
                inner_node, outer_default, outer_contract
            )
            return Tree("param_isolated", [merged])

        node, _, _, _ = self._parse_param_atom()
        return node

    def _parse_param_destruct(self) -> Tree:
        self.expect(TT.LBRACE)

        fields: List[Node] = []
        seen: set[str] = set()

        if self.check(TT.RBRACE):
            raise ParseError(
                "Destructuring parameter requires at least one field", self.current
            )

        while True:
            ident = self.expect(TT.IDENT)
            field_name = str(ident.value)

            if field_name in seen:
                raise ParseError(f"Duplicate destructuring field '{field_name}'", ident)
            seen.add(field_name)

            field_children: List[Node] = [ident]
            if self.match(TT.ASSIGN):
                # Preserve top-level '~' as field-contract metadata rather than
                # consuming it as a compare operator inside the default expr.
                field_children.append(self._parse_pattern_default_expr())

            if self.match(TT.TILDE):
                contract = self.parse_expr()
                field_children.append(Tree("contract", [contract]))
                if self.check(TT.ASSIGN):
                    raise ParseError(
                        "Default must precede contract in destructuring field metadata",
                        self.current,
                    )

            fields.append(Tree("destruct_field", field_children))

            if not self.match(TT.COMMA):
                break
            if self.check(TT.RBRACE):
                break

        self.expect(TT.RBRACE)

        if self.match(TT.ASSIGN):
            raise ParseError(
                "Destructuring parameter cannot have an outer default", self.current
            )

        if self.match(TT.TILDE):
            contract = self.parse_expr()
            fields.append(Tree("contract", [contract]))
            if self.check(TT.ASSIGN):
                raise ParseError(
                    "Destructuring parameter cannot have an outer default",
                    self.current,
                )

        return Tree("param_destruct", fields)

    def _param_destruct_has_contract(self, node: Node) -> bool:
        if not is_tree(node) or tree_label(node) != "param_destruct":
            return False

        for child in tree_children(node):
            if is_tree(child) and tree_label(child) == "contract":
                return True

        return False

    def _param_destruct_is_bare(self, node: Node) -> bool:
        if not is_tree(node) or tree_label(node) != "param_destruct":
            return False

        if self._param_destruct_has_contract(node):
            return False

        for child in tree_children(node):
            if not is_tree(child) or tree_label(child) != "destruct_field":
                continue
            field_children = tree_children(child)
            if len(field_children) > 1:
                return False

        return True

    def _parse_param_atom(self) -> tuple[Node, bool, bool, bool]:
        is_spread = self.match(TT.SPREAD)
        ident = self.expect(TT.IDENT)

        default = None
        contract = None

        if self.match(TT.ASSIGN):
            default = self.parse_expr()

        if self.match(TT.TILDE):
            contract = self.parse_expr()
            if self.check(TT.ASSIGN):
                raise ParseError(
                    "Default must precede contract in parameter metadata",
                    self.current,
                )

        if is_spread and default:
            raise ParseError("Spread parameter cannot have a default", self.current)

        node = self._build_param_node(ident, default, contract, is_spread)
        return node, default is not None, contract is not None, is_spread

    def _build_param_node(
        self,
        ident: Tok,
        default: Optional[Node],
        contract: Optional[Node],
        is_spread: bool,
    ) -> Node:
        if is_spread:
            children: List[Node] = [ident]
            if contract:
                children.append(Tree("contract", [contract]))
            return Tree("param_spread", children)

        if default is None and contract is None:
            return ident

        children = [ident]
        if default:
            children.append(default)
        if contract:
            children.append(Tree("contract", [contract]))
        return Tree("param", children)

    def _merge_param_metadata(
        self,
        node: Node,
        default: Optional[Node],
        contract: Optional[Node],
    ) -> Node:
        ident, existing_default, existing_contract, is_spread = (
            self._unpack_param_metadata(node)
        )

        if default:
            if existing_contract:
                raise ParseError(
                    "Default must precede contract in parameter metadata",
                    self.current,
                )
            if existing_default:
                raise ParseError(
                    "Parameter cannot have multiple defaults", self.current
                )
            if is_spread:
                raise ParseError("Spread parameter cannot have a default", self.current)

        if contract and existing_contract:
            raise ParseError("Parameter cannot have multiple contracts", self.current)

        merged_default = existing_default or default
        merged_contract = existing_contract or contract
        return self._build_param_node(ident, merged_default, merged_contract, is_spread)

    def _unpack_param_metadata(
        self, node: Node
    ) -> tuple[Tok, Optional[Node], Optional[Node], bool]:
        if isinstance(node, Tok):
            return node, None, None, False

        if not isinstance(node, Tree):
            raise ParseError("Invalid parameter node", self.current)

        label = node.data
        children = list(node.children)

        if not children or not isinstance(children[0], Tok):
            raise ParseError("Parameter missing name", self.current)

        ident = children[0]
        default = None
        contract = None
        is_spread = label == "param_spread"

        for child in children[1:]:
            if isinstance(child, Tree) and child.data == "contract":
                contract_children = child.children
                if contract_children:
                    contract = contract_children[0]
                continue
            if default is None:
                default = child

        return ident, default, contract, is_spread

    def _parse_pattern_default_expr(self) -> Node:
        """Parse a default value expression for destructure patterns.
        Parse a full expression while preserving pattern delimiters.
        Defaults need full expression power, but top-level `~` / `~~` remain
        reserved for contract metadata in pattern context.
        """
        saved_context = self._push_parse_context(ParseContext.DESTRUCTURE_PACK)
        prev_depth = self._pattern_default_paren_depth
        self._pattern_default_paren_depth = self.paren_depth
        try:
            return self.parse_catch_expr()
        finally:
            self._pattern_default_paren_depth = prev_depth
            self._pop_parse_context(saved_context)

    def _is_pattern_start_token(self, tok: Tok) -> bool:
        return tok.type in {TT.IDENT, TT.LPAR, TT.SPREAD}

    def _assign_starts_pattern_default(self) -> bool:
        """Whether the current ASSIGN token belongs to `ident = <default>`."""
        if not self.check(TT.ASSIGN):
            return False

        idx = self.pos + 1
        depth = 0
        saw_expr_token = False

        while idx < len(self.tokens):
            tok = self.tokens[idx]

            if tok.type in {TT.LPAR, TT.LBRACE, TT.LSQB}:
                depth += 1
                saw_expr_token = True
                idx += 1
                continue

            if tok.type in {TT.RPAR, TT.RBRACE, TT.RSQB}:
                if depth == 0:
                    return tok.type == TT.RPAR and saw_expr_token
                depth -= 1
                saw_expr_token = True
                idx += 1
                continue

            if depth == 0:
                if tok.type == TT.COMMA:
                    next_tok = self._lookahead_peek(idx + 1)
                    return saw_expr_token and self._is_pattern_start_token(next_tok)

                if tok.type in {TT.TILDE, TT.REGEXMATCH, TT.WALRUS}:
                    return saw_expr_token

                if tok.type == TT.ASSIGN:
                    return saw_expr_token

                if tok.type in {TT.NEWLINE, TT.SEMI, TT.PIPE, TT.EOF}:
                    return False

            saw_expr_token = True
            idx += 1

        return False

    def _lookahead_assign_starts_pattern_default(self, assign_idx: int) -> bool:
        """Lookahead variant of `_assign_starts_pattern_default`."""
        if not self._lookahead_check(assign_idx, TT.ASSIGN):
            return False

        idx = assign_idx + 1
        depth = 0
        saw_expr_token = False

        while idx < len(self.tokens):
            tok = self._lookahead_peek(idx)

            if tok.type in {TT.LPAR, TT.LBRACE, TT.LSQB}:
                depth += 1
                saw_expr_token = True
                idx += 1
                continue

            if tok.type in {TT.RPAR, TT.RBRACE, TT.RSQB}:
                if depth == 0:
                    return tok.type == TT.RPAR and saw_expr_token
                depth -= 1
                saw_expr_token = True
                idx += 1
                continue

            if depth == 0:
                if tok.type == TT.COMMA:
                    next_tok = self._lookahead_peek(idx + 1)
                    return saw_expr_token and self._is_pattern_start_token(next_tok)

                if tok.type in {TT.TILDE, TT.REGEXMATCH, TT.WALRUS}:
                    return saw_expr_token

                if tok.type == TT.ASSIGN:
                    return saw_expr_token

                if tok.type in {TT.NEWLINE, TT.SEMI, TT.PIPE, TT.EOF}:
                    return False

            saw_expr_token = True
            idx += 1

        return False

    def _parse_pattern_from_ident(
        self, ident: Tok, *, allow_default: bool = True
    ) -> Tree:
        default = None
        contract = None

        # Default before contract, matching param convention.
        # Parse full default expressions, but only when `=` is truly a default
        # marker rather than the destructure assignment delimiter.
        if (
            allow_default
            and self.check(TT.ASSIGN)
            and self._assign_starts_pattern_default()
        ):
            self.advance()
            default = self._parse_pattern_default_expr()
        if self.match(TT.TILDE):
            # Parse contract at comparison level to avoid consuming walrus/comma.
            contract = self.parse_compare_expr()

        children: List[Node] = [ident]
        if default:
            children.append(Tree("default", [default]))
        if contract:
            children.append(Tree("contract", [contract]))
        return Tree("pattern", children)

    def parse_pattern(self, *, allow_default: bool = True) -> Tree:
        # Rest/spread pattern: ...ident
        if self.match(TT.SPREAD):
            ident = self.expect(TT.IDENT)
            return Tree("pattern_rest", [ident])

        if self.check(TT.IDENT):
            return self._parse_pattern_from_ident(
                self.advance(), allow_default=allow_default
            )

        if self.match(TT.LPAR):
            items = [self.parse_pattern()]
            if not self.match(TT.COMMA):
                raise ParseError("Pattern list requires comma", self.current)
            items.append(self.parse_pattern())
            while self.match(TT.COMMA):
                items.append(self.parse_pattern())
            self.expect(TT.RPAR)

            # Rest patterns are not supported inside nested tuple patterns
            for item in items:
                if tree_label(item) == "pattern_rest":
                    tok = item.children[0] if item.children else self.current
                    raise ParseError("Rest pattern not allowed in nested pattern", tok)

            return Tree("pattern", [Tree("pattern_list", items)])

        raise ParseError("Expected pattern", self.current)

    def parse_binderpattern(self) -> Tree:
        if self.match(TT.CARET):
            ident = self.expect(TT.IDENT)
            return Tree("hoist", [ident])
        return Tree("binderpattern", [self.parse_pattern()])

    def parse_binderlist(self) -> Tree:
        items = [self.parse_binderpattern()]
        while self.match(TT.COMMA):
            items.append(self.parse_binderpattern())
        return Tree("binderlist", items)

    def parse_overspec(self) -> Tree:
        """Parse comprehension spec: iterable expression and optional binders"""
        children: List[Any] = []

        # Reset context: comprehension iterable allows CCC
        saved_context = self._push_parse_context(ParseContext.NORMAL)
        try:
            # Check for binder list syntax: [x, y] iterable
            # Only if '[' is followed by a pattern start (IDENT, ^, or '(')
            if self.check(TT.LSQB):
                next_tok = self.peek(1)
                if next_tok.type in (TT.IDENT, TT.CARET, TT.LPAR):
                    self.advance()  # consume '['
                    binder_list = self.parse_binderlist()
                    self.expect(TT.RSQB)
                    iter_expr = self.parse_expr()
                    children.append(binder_list)
                    children.append(iter_expr)
                    return Tree("overspec", children)

            # Check for common pattern: IDENT in expr (e.g., for x in data)
            if self.check(TT.IDENT) and self.peek(1).type == TT.IN:
                pattern = self.parse_pattern()
                self.expect(TT.IN)
                iter_expr = self.parse_expr()
                children.append(iter_expr)
                children.append(pattern)
                return Tree("overspec", children)

            # Otherwise: expr [bind pattern]
            iter_expr = self.parse_expr()
            children.append(iter_expr)
            if self.match(TT.BIND):
                pattern = self.parse_pattern()
                children.append(pattern)
            return Tree("overspec", children)
        finally:
            self._pop_parse_context(saved_context)

    def parse_comphead(self) -> Tree:
        if self.check(TT.FOR):
            for_tok = self.advance()
            spec = self.parse_overspec()
            return Tree("comphead", [for_tok, spec])
        if self.check(TT.OVER):
            over_tok = self.advance()
            spec = self.parse_overspec()
            return Tree("comphead", [over_tok, spec])
        raise ParseError("Expected comprehension head", self.current)

    def parse_ifclause_opt(self) -> Optional[Tree]:
        """Parse optional if-clause in comprehensions"""
        if not self.check(TT.IF):
            return None
        if_tok = self.advance()

        # Reset context: if-clause condition allows CCC
        saved_context = self._push_parse_context(ParseContext.NORMAL)
        try:
            cond = self.parse_expr()
            return Tree("ifclause", [if_tok, cond])
        finally:
            self._pop_parse_context(saved_context)

    def _parse_arg_item(self) -> Tree:
        if self.check(TT.IDENT) and self.peek(1).type == TT.COLON:
            name = self.current
            self.advance(2)
            value = self.parse_expr()
            return Tree("argitem", [Tree("namedarg", [name, value])])
        return Tree("argitem", [Tree("arg", [self.parse_expr()])])

    def parse_arg_list(self) -> Optional[List[Tree]]:
        """
        Parse function call arguments
        Returns list of argitems (unwrapped) or None if empty
        """
        if self.check(TT.RPAR):
            return None  # Empty arg list

        # Set context: commas in function args are separators, not CCC
        saved_context = self._push_parse_context(ParseContext.FUNCTION_ARGS)
        try:
            argitems: List[Tree] = []
            while not self.check(TT.RPAR, TT.EOF):
                argitems.append(self._parse_arg_item())
                if not self.match(TT.COMMA):
                    break

            # Return argitems directly (let caller wrap if needed)
            return argitems
        finally:
            self._pop_parse_context(saved_context)

    def _emit_args_terminator(self) -> bool:
        if self.check(
            TT.NEWLINE,
            TT.SEMI,
            TT.DEDENT,
            TT.EOF,
            TT.RPAR,
            TT.RSQB,
            TT.RBRACE,
            TT.COMMA,
            TT.COLON,
            TT.PIPE,
            TT.IF,
        ):
            return True
        if self.check(TT.UNLESS):
            return True
        return False

    def parse_emit_arg_list(self) -> Optional[List[Tree]]:
        """
        Parse emit arguments after '>' (no surrounding parentheses).
        Returns list of argitems (unwrapped) or None if empty.
        """
        if self.check(TT.LPAR) and self.peek(1).type == TT.RPAR:
            self.advance(2)
            return []

        if self._emit_args_terminator():
            return None

        saved_context = self._push_parse_context(ParseContext.FUNCTION_ARGS)
        try:
            argitems: List[Tree] = []

            while True:
                argitems.append(self._parse_arg_item())

                if not self.match(TT.COMMA):
                    break
                if self._emit_args_terminator():
                    break  # allow trailing comma

            return argitems
        finally:
            self._pop_parse_context(saved_context)

    def parse_selector_list(self) -> List[Tree]:
        """
        Parse selector list for indexing: [0, 1:10, :, 1:]
        Returns list of Tree('selector', [Tree('indexsel', [expr])]) nodes
        Stops before ', default:' sequence
        """
        selectors = []

        while not self.check(TT.RSQB, TT.EOF):
            # Check for slice syntax (has :)
            if self.is_slice_selector():
                slice_tree = self.parse_slice_selector()
                selectors.append(Tree("selector", [slice_tree]))
            else:
                expr = self.parse_expr()
                selectors.append(Tree("selector", [Tree("indexsel", [expr])]))

            # Check if next is ', default:' sequence - if so, stop before consuming comma
            if self.check(TT.COMMA):
                # Peek ahead to see if it's followed by 'default' ':'
                if (
                    self.peek(1).type == TT.IDENT
                    and self.peek(1).value == "default"
                    and self.peek(2).type == TT.COLON
                ):
                    break
                # Not a default clause, consume the comma and continue
                self.match(TT.COMMA)
            else:
                break

        return selectors

    def parse_selector_literal_content(self) -> List[Tree]:
        """
        Parse selector literal content between backticks: `0, 2` or `{start}:{stop}`
        Handles interpolation {expr} within selectors
        """
        selectors = []

        while not self.check(TT.BACKQUOTE, TT.EOF):
            # Check for slice syntax (has :)
            if self.is_slice_selector_literal():
                slice_tree = self.parse_slice_selector_literal()
                # Wrap in selitem for evaluator
                selectors.append(Tree("selitem", [slice_tree]))
            else:
                atom = self.parse_selector_atom()
                # Wrap atom in selatom, then in indexitem, then in selitem
                selatom = Tree("selatom", [atom])
                indexitem = Tree("indexitem", [selatom])
                selectors.append(Tree("selitem", [indexitem]))

            if not self.match(TT.COMMA):
                break

        return selectors

    def parse_selector_atom(self) -> Tree | Tok:
        """Parse selector atom: {expr} (interp), IDENT, or NUMBER"""
        # Interpolation: {expr}
        if self.match(TT.LBRACE):
            expr = self.parse_expr()
            self.expect(TT.RBRACE)
            return Tree("interp", [expr])

        # Allow signed numeric atoms in selector literals (minus is not part of NUMBER tokens).
        if self.check(TT.MINUS) and self.peek(1).type == TT.NUMBER:
            self.advance()
            tok = self.expect(TT.NUMBER)
            return self._parse_selector_number(tok, negate=True)

        # IDENT - convert to AST token with string type
        if self.check(TT.IDENT):
            return self.advance()

        # NUMBER - convert to AST token with string type
        if self.check(TT.NUMBER):
            tok = self.advance()
            return self._parse_selector_number(tok, negate=False)

        raise ParseError(
            "Expected selector atom (number, identifier, or {expr})", self.current
        )

    def _parse_selector_number(self, tok: Tok, negate: bool) -> Tok:
        value = self._parse_number_literal(tok.value)

        if isinstance(value, int):
            if negate:
                # Allow -2**63 but reject larger magnitude.
                if value > 2**63:
                    raise LexError(
                        "Integer literal overflows int64",
                        line=tok.line,
                        column=tok.column,
                    )
                return self._tok("NUMBER", -value, line=tok.line, column=tok.column)

            if value > 2**63 - 1:
                raise LexError(
                    "Integer literal overflows int64",
                    line=tok.line,
                    column=tok.column,
                )
            return self._tok("NUMBER", tok.value, line=tok.line, column=tok.column)

        if negate:
            return self._tok("NUMBER", -value, line=tok.line, column=tok.column)

        return self._tok("NUMBER", tok.value, line=tok.line, column=tok.column)

    def is_slice_selector_literal(self) -> bool:
        """Check if next tokens form a slice selector in literal context"""
        idx = self.pos
        paren_depth = 0

        # Skip initial atom if present
        if self._lookahead_check(idx, TT.LBRACE):
            # Skip to matching }
            _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)
            depth = 1
            while depth > 0 and idx < len(self.tokens):
                if self._lookahead_check(idx, TT.LBRACE):
                    depth += 1
                elif self._lookahead_check(idx, TT.RBRACE):
                    depth -= 1
                _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)
        elif self._lookahead_check(idx, TT.IDENT, TT.NUMBER):
            _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)
        # Treat a leading "-NUMBER" as an atom for slice lookahead.
        elif self._lookahead_check(idx, TT.MINUS) and self._lookahead_check(
            idx + 1, TT.NUMBER
        ):
            _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)
            _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)

        # Check if colon follows
        return self._lookahead_check(idx, TT.COLON)

    def parse_slice_selector_literal(self) -> Tree:
        """Parse slice in selector literal: {start}:{stop} or 1:10:2"""
        start = None
        if not self.check(TT.COLON):
            atom = self.parse_selector_atom()
            start = Tree("selatom", [atom])

        self.expect(TT.COLON)

        # Check for < prefix for open-ended slice
        lt_tok = None
        if self.check(TT.LT):
            lt_tok = self.advance()  # Capture LT token

        stop = None
        if not self.check(TT.COLON, TT.COMMA, TT.BACKQUOTE):
            atom = self.parse_selector_atom()
            stop = Tree("selatom", [atom])

        step = None
        if self.match(TT.COLON):
            if not self.check(TT.COMMA, TT.BACKQUOTE):
                atom = self.parse_selector_atom()
                step = Tree("selatom", [atom])

        # Build sliceitem tree according to grammar:
        # sliceitem: selatom? ":" seloptstop (":" selatom)?
        children = []
        if start:
            children.append(start)

        # seloptstop: "<" selatom | selatom?
        if lt_tok and stop:
            children.append(Tree("seloptstop", [lt_tok, stop]))
        elif stop:
            children.append(Tree("seloptstop", [stop]))
        else:
            children.append(Tree("seloptstop", []))

        if step:
            children.append(step)

        return Tree("sliceitem", children)

    def is_slice_selector(self) -> bool:
        """Check if next tokens form a slice selector"""
        # Look for : that's not part of ternary or other constructs
        # Simple heuristic: if we see : before any binary operator
        idx = self.pos
        paren_depth = 0

        # Try to parse an expression, see if : follows
        depth = 0
        while idx < len(self.tokens):
            if self._lookahead_check(idx, TT.BACKQUOTE):
                # Skip selector literal contents so internal ':' doesn't count as slice delimiter
                _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)
                while idx < len(self.tokens) and not self._lookahead_check(
                    idx, TT.BACKQUOTE
                ):
                    _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)
                # Consume closing backtick if present
                if idx < len(self.tokens):
                    _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)
                continue
            if self._lookahead_check(idx, TT.LPAR, TT.LSQB, TT.LBRACE):
                depth += 1
                _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)
            elif self._lookahead_check(idx, TT.RPAR, TT.RSQB, TT.RBRACE):
                if depth == 0:
                    break
                depth -= 1
                _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)
            elif depth == 0 and self._lookahead_check(idx, TT.COLON):
                # Found a colon - this is a slice
                return True
            elif depth == 0 and self._lookahead_check(idx, TT.COMMA):
                break
            else:
                _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)

        # Not a slice
        return False

    def parse_slice_selector(self) -> Tree:
        """Parse slice selector: start:stop:step"""
        start = None
        if not self.check(TT.COLON):
            start = self.parse_expr()

        self.expect(TT.COLON)

        stop = None
        if not self.check(TT.COLON, TT.COMMA, TT.RSQB):
            stop = self.parse_expr()

        step = None
        if self.match(TT.COLON):
            if not self.check(TT.COMMA, TT.RSQB):
                step = self.parse_expr()

        children = []
        if start:
            children.append(Tree("slicearm_expr", [start]))
        else:
            children.append(Tree("slicearm_empty", []))

        if stop:
            children.append(Tree("slicearm_expr", [stop]))
        else:
            children.append(Tree("slicearm_empty", []))

        if step:
            children.append(Tree("slicearm_expr", [step]))

        return Tree("slicesel", children)

    def parse_object_item(self) -> Tree:
        """Parse object item: key: value or method(params): body"""
        if self.match(TT.SPREAD):
            expr = self.parse_expr()
            return Tree("obj_spread", [expr])

        # Field, method, getter, setter
        # Grammar allows: (IDENT | OVER) for field names
        if self.check(TT.IDENT, TT.OVER):
            name = self.advance()

            # Method: name(params): body
            if self.match(TT.LPAR):
                params = self.parse_param_list()
                self.expect(TT.RPAR)
                self.expect(TT.COLON)
                return Tree("obj_method", [name, params, self._parse_obj_item_body()])

            # Pun: bare IDENT (not followed by : or ?) => {x} desugars to {x: x}
            if name.type == TT.IDENT and not self.check(TT.COLON, TT.QMARK):
                return Tree("obj_field", [Tree("key_ident", [name]), name])

            # Field: name: value or name?: value (optional)
            is_optional = self.match(TT.QMARK)
            self.expect(TT.COLON)
            while self.match(TT.NEWLINE, TT.INDENT):
                pass
            value = self.parse_expr()
            key = Tree("key_ident", [name])
            if is_optional:
                return Tree("obj_field_optional", [key, value])
            return Tree("obj_field", [key, value])

        # String key
        if self.check(TT.STRING, TT.RAW_STRING, TT.RAW_HASH_STRING):
            key_tok = self.advance()
            self.expect(TT.COLON)
            while self.match(TT.NEWLINE, TT.INDENT):
                pass
            value = self.parse_expr()
            key = Tree("key_string", [key_tok])
            return Tree("obj_field", [key, value])

        # Expression key: (expr): value
        if self.check(TT.LPAR):
            self.advance()  # consume LPAR
            key_expr = self.parse_expr()
            self.expect(TT.RPAR)
            self.expect(TT.COLON)
            while self.match(TT.NEWLINE, TT.INDENT):
                pass
            value = self.parse_expr()
            key = Tree("key_expr", [key_expr])
            return Tree("obj_field", [key, value])

        # Getter/setter
        if self.match(TT.GET):
            name = self.expect(TT.IDENT)
            if self.match(TT.LPAR):
                self.expect(TT.RPAR)
            self.expect(TT.COLON)
            return Tree("obj_get", [name, self._parse_obj_item_body()])

        if self.match(TT.SET):
            name = self.expect(TT.IDENT)
            _, param, _ = self.expect_seq(TT.LPAR, TT.IDENT, TT.RPAR)
            self.expect(TT.COLON)
            return Tree("obj_set", [name, param, self._parse_obj_item_body()])

        raise ParseError("Expected object item", self.current)

    def parse_object_body(self) -> Tree:
        """
        Parse a method/get/set body inside an object literal.
        No INDENT tokens inside braces; collect statements until next item or RBRACE.
        """
        stmts: List[Tree] = []

        while True:
            if self._looks_like_object_item_start():
                break

            if self.match(TT.NEWLINE):
                continue

            stmt = self.parse_statement()
            stmts.append(Tree("stmt", [stmt]))

            if self.match(TT.SEMI):
                continue

            self._consume_newlines()

            # If next token starts a new object item, stop collecting
            if self._looks_like_object_item_start():
                break

        return Tree("body", stmts, attrs={"inline": False})

    def parse_pattern_list(self, *, allow_default: bool = True) -> Tree:
        """Parse destructuring pattern list: a, b, c"""
        patterns = [self.parse_pattern(allow_default=allow_default)]

        if not self.match(TT.COMMA):
            raise ParseError(
                "Pattern list requires at least two patterns", self.current
            )

        patterns.append(self.parse_pattern(allow_default=allow_default))

        while self.match(TT.COMMA):
            patterns.append(self.parse_pattern(allow_default=allow_default))

        # Validate: rest pattern must be last
        for i, pat in enumerate(patterns):
            if tree_label(pat) == "pattern_rest" and i != len(patterns) - 1:
                tok = pat.children[0] if pat.children else self.current
                raise ParseError("Rest pattern must be last", tok)

        return Tree("pattern_list", patterns)

    def parse_destructure_rhs(self) -> Tree:
        """Parse destructuring RHS: expr or expr, expr, ..."""
        # Set context: commas in destructure pack are separators, not CCC
        saved_context = self._push_parse_context(ParseContext.DESTRUCTURE_PACK)
        try:
            first_expr = self.parse_expr()

            # Check if there are more expressions (pack)
            if self.check(TT.COMMA):
                exprs = [first_expr]
                while self.match(TT.COMMA):
                    exprs.append(self.parse_expr())
                return Tree("pack", exprs)

            # Single expression
            return first_expr
        finally:
            self._pop_parse_context(saved_context)

    def parse_destructure_pattern(self) -> Tree:
        """Parse destructuring pattern for assignments: ident [~ contract] (, ident [~ contract])*"""
        items: list[Node] = []
        items.append(self._parse_pattern_item())

        while self.match(TT.COMMA):
            items.append(self._parse_pattern_item())

        return Tree("patternlist", items)

    def _parse_pattern_item(self) -> Tree:
        """Parse a single pattern item: IDENT [= default] [~ contract] or ...IDENT"""
        if self.match(TT.SPREAD):
            ident = self.expect(TT.IDENT)
            return Tree("pattern_rest", [ident])
        return self._parse_pattern_from_ident(self.expect(TT.IDENT))

    def _is_destructure_with_contracts(self) -> bool:
        """
        Lookahead to detect destructure patterns with contracts, defaults, or rest.
        Scans all comma-separated patterns looking for any that has `= default`,
        `~ contract`, or `...rest`. Returns True if at least one special marker
        is found and the sequence ends with `:=` or `=`.
        Uses stateless lookahead helper - does not mutate parser state.
        """
        idx = self.pos
        paren_depth = 0
        has_special = False

        while True:
            # Spread pattern: ...ident
            if self._lookahead_check(idx, TT.SPREAD):
                has_special = True
                _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)
                if not self._lookahead_check(idx, TT.IDENT):
                    return False
                _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)

            # Nested tuple pattern: (a, b, ...)
            elif self._lookahead_check(idx, TT.LPAR):
                depth = 1
                _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)
                while idx < len(self.tokens) and depth > 0:
                    tok = self._lookahead_peek(idx)
                    if tok.type == TT.LPAR:
                        depth += 1
                    elif tok.type == TT.RPAR:
                        depth -= 1
                    elif tok.type in {TT.NEWLINE, TT.SEMI, TT.EOF}:
                        return False
                    _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)

            else:
                # Must be IDENT
                if not self._lookahead_check(idx, TT.IDENT):
                    return False
                _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)

                # Optional default: = <expr until comma/tilde/walrus>
                if self._lookahead_check(
                    idx, TT.ASSIGN
                ) and self._lookahead_assign_starts_pattern_default(idx):
                    has_special = True
                    _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)

                    depth = 0
                    while idx < len(self.tokens):
                        tok = self._lookahead_peek(idx)
                        if depth == 0 and tok.type in {
                            TT.COMMA,
                            TT.TILDE,
                            TT.REGEXMATCH,
                            TT.WALRUS,
                            TT.ASSIGN,
                        }:
                            break
                        if tok.type in {TT.LPAR, TT.LBRACE, TT.LSQB}:
                            depth += 1
                        elif tok.type in {TT.RPAR, TT.RBRACE, TT.RSQB}:
                            depth -= 1
                            if depth < 0:
                                return False
                        elif tok.type in {TT.NEWLINE, TT.SEMI, TT.EOF}:
                            return False
                        _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)

                # Optional contract: ~ <expr until comma/assign/walrus>
                if self._lookahead_check(idx, TT.TILDE):
                    has_special = True
                    _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)

                    depth = 0
                    while idx < len(self.tokens):
                        tok = self._lookahead_peek(idx)
                        if depth == 0 and tok.type in {TT.COMMA, TT.WALRUS, TT.ASSIGN}:
                            break
                        if tok.type in {TT.LPAR, TT.LBRACE, TT.LSQB}:
                            depth += 1
                        elif tok.type in {TT.RPAR, TT.RBRACE, TT.RSQB}:
                            depth -= 1
                            if depth < 0:
                                return False
                        elif tok.type in {TT.NEWLINE, TT.SEMI, TT.EOF}:
                            return False
                        _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)

            # After one pattern: check for comma (more patterns) or assignment
            if self._lookahead_check(idx, TT.COMMA):
                _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)
                continue

            # End of pattern list: check for assignment operator
            if has_special and self._lookahead_check(idx, TT.WALRUS, TT.ASSIGN):
                return True

            return False

    def parse_fan_items(self) -> List[Node]:
        """Parse fan items: {field1, field2} or {chain1, chain2}"""
        items: list[Node] = []

        while not self.check(TT.RBRACE, TT.EOF):
            if self.check(TT.IDENT):
                name = self.advance()
                # Check for postfix operations (calls, field access, etc.)
                postfix_ops: list[Node] = []
                while True:
                    if self.check(TT.LPAR):
                        postfix_ops.append(self._parse_call_op())
                    elif self.match(TT.DOT):
                        if not self.check(TT.IDENT):
                            raise ParseError(
                                "Expected field name after '.'", self.current
                            )
                        postfix_ops.append(self._field_node(self.advance()))
                    elif self.check(TT.LSQB):
                        postfix_ops.append(self._parse_plain_index_node())
                    else:
                        break

                # Build identchain if we have postfix ops
                if postfix_ops:
                    items.append(
                        Tree(
                            "identchain",
                            [name] + postfix_ops,
                        )
                    )
                else:
                    items.append(name)
            else:
                raise ParseError("Expected identifier in fan list", self.current)

            if not self.match(TT.COMMA):
                break

        return items

    def parse_anonymous_fn_decl(self) -> Tree:
        """
        Anonymous function literal: fn(params): body
        Thunk sugar: fn: body
        Auto-invoked form: fn(()): body
        """
        auto_invoke = False
        auto_call_tok: Optional[Tok] = None
        params: Tree = Tree("paramlist", [])

        if self.check(TT.LPAR):
            outer_lpar = self.advance()
            if self.check(TT.LPAR):
                self.advance()
                # fn ( ( ) ) : body  -> auto invoke
                self.expect(TT.RPAR)
                auto_invoke = True
                auto_call_tok = outer_lpar
            else:
                params = (
                    self.parse_param_list()
                    if not self.check(TT.RPAR)
                    else Tree("paramlist", [])
                )
            self.expect(TT.RPAR)

        # Optional return contract: ~ Schema
        return_contract = None
        if self.match(TT.TILDE):
            return_contract = self.parse_expr()

        self.expect(TT.COLON)
        if self.check(TT.LBRACE):
            lookahead = self.peek(1)
            looks_like_object = lookahead.type in {
                TT.IDENT,
                TT.STRING,
                TT.RAW_STRING,
                TT.RAW_HASH_STRING,
                TT.GET,
                TT.SET,
                TT.OVER,
            } and (self.peek(2).type == TT.COLON or lookahead.type in {TT.GET, TT.SET})

            if looks_like_object:
                expr_body = self.parse_expr()
                body = Tree("body", [Tree("stmt", [expr_body])], attrs={"inline": True})
            else:
                body = self.parse_body()
        else:
            body = self.parse_body()

        paramlist_node = None if auto_invoke else params
        anon_children: List[Any] = []
        if paramlist_node:
            anon_children.append(paramlist_node)
        anon_children.append(body)

        if return_contract:
            anon_children.append(Tree("return_contract", [return_contract]))

        anon = Tree("anonfn", anon_children)

        if auto_invoke:
            assert auto_call_tok
            return Tree("explicit_chain", [anon, self._call_tree([], auto_call_tok)])
        return anon

    def parse_anonymous_fn(self) -> Tree:
        """
        Parse amp-lambda (not decorators - those use @):
        &(expr) - implicit subject lambda
        &[params](expr) - explicit params lambda

        INVARIANT: The '&' token must be consumed by caller before calling this method.
        Expected tokens: '[' for explicit params or '(' for implicit subject.
        """
        # Verify invariant: & should have been consumed by caller
        assert self.check(
            TT.LSQB, TT.LPAR
        ), f"parse_anonymous_fn expects '[' or '(', got {self.current.type.name}. Caller must consume '&' first."

        # Check for explicit params: &[params]
        if self.match(TT.LSQB):
            # &[a, b](a + b)
            params = (
                self.parse_param_list(end_token=TT.RSQB)
                if not self.check(TT.RSQB)
                else Tree("paramlist", [])
            )
            self.expect(TT.RSQB)

            # Now expect (expr)
            self.expect(TT.LPAR)
            body = self.parse_expr()
            self.expect(TT.RPAR)

            # Return Tree('amp_lambda', [Tree('paramlist', params), body])
            return Tree("amp_lambda", [params, body])

        # Implicit subject: &(expr)
        self.expect(TT.LPAR)
        body = self.parse_expr()
        self.expect(TT.RPAR)

        # Return Tree('amp_lambda', [body])
        return Tree("amp_lambda", [body])


# ============================================================================
# Usage Example
# ============================================================================


def parse_source(source: str, use_indenter: bool = True) -> Tree:
    """
    Parse Shakar source code to AST.

    Returns Tree structure expected by the evaluator.

    Args:
        source: Source code to parse
        use_indenter: Whether to track indentation (default True)
    """
    from .lexer_rd import tokenize

    # Tokize
    tokens = tokenize(source, track_indentation=use_indenter)

    # Parse
    parser = Parser(tokens, use_indenter=use_indenter)
    return parser.parse()


def parse_expr_fragment(source: str) -> Tree:
    """
    Parse a standalone expression fragment.
    Used by string interpolation and other expression-only contexts.
    """
    from .lexer_rd import tokenize

    tokens = tokenize(source, track_indentation=False)
    parser = Parser(tokens, use_indenter=False)
    expr = parser.parse_expr()

    # Ensure we've consumed the entire fragment
    if not parser.check(TT.EOF):
        raise ParseError("Unexpected tokens after expression fragment", parser.current)
    return expr


# ============================================================================
# Main - Differential Testing
# ============================================================================

if __name__ == "__main__":
    import sys

    # Parse arguments
    use_indenter = "--no-indent" not in sys.argv
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

    # Read source from file or stdin
    if len(args) > 0 and args[0] != "-":
        with open(args[0], "r") as f:
            source = f.read()
    else:
        source = sys.stdin.read()

    try:
        tree = parse_source(source, use_indenter=use_indenter)
        print(tree.pretty())
    except ParseError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
