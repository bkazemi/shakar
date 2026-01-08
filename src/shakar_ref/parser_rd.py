"""
Recursive Descent Parser for Shakar

This serves as:
1. A reference implementation for the C parser
2. A faster alternative to Lark's Earley parser
3. Documentation of parsing strategy and disambiguation

Structure:
- Lexer: Tok stream from source
- Parser: Recursive descent with Pratt parsing for expressions
- AST: Same tree structure as Lark output for compatibility
"""

from typing import Optional, List, Any
from enum import Enum
from .tree import Tree, Node

from .token_types import TT, Tok

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
    6. walrus (:=)
    7. nullish (??)
    8. compare (==, !=, <, >, etc.) + CCC
    9. add (+, -, +>, ^)
    10. mul (*, /, //, %)
    11. pow (**)
    12. unary (-, !, ~, ++, --, await, $)
    13. postfix (.field, [index], (call), ++, --)
    14. primary (literals, identifiers, parens)
    """

    def __init__(self, tokens: List[Tok], use_indenter: bool = True):
        self.tokens = tokens
        self.pos = 0
        self.current = tokens[0] if tokens else Tok(TT.EOF, None, 0, 0)
        self.in_inline_body = False  # Track if we're in inlinebody context
        self.use_indenter = use_indenter  # Track indentation mode
        self.paren_depth = 0  # Track parenthesis nesting depth
        self.parse_context = ParseContext.NORMAL  # Track parsing context for comma disambiguation

    def _tok(self, type_name: str, value: str, line: int = 0, column: int = 0) -> Tok:
        return Tok(TT[type_name], value, line, column)

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
                skip_types = (TT.NEWLINE, TT.INDENT, TT.DEDENT) if self.use_indenter else (TT.NEWLINE,)
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
                    stmtlists.append(Tree('stmtlist', stmts))
                    stmts = []
                continue

            stmt = self.parse_statement()
            stmts.append(Tree('stmt', [stmt]))

            # Check for semicolon separator
            if self.match(TT.SEMI):
                stmts.append(self._tok('SEMI', ';'))
                # Continue parsing more statements on same line
                # Skip whitespace but not newlines
                continue

            # Consume trailing newlines
            while self.match(TT.NEWLINE):
                pass

        # Add any remaining statements
        if stmts:
            stmtlists.append(Tree('stmtlist', stmts))

        # Return appropriate start node based on indentation mode
        start_node = 'start_indented' if self.use_indenter else 'start_noindent'
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

        # Control flow
        if self.check(TT.IF):
            return self.parse_if_stmt()
        if self.check(TT.WHILE):
            return self.parse_while_stmt()
        if self.check(TT.FOR):
            return self.parse_for_stmt()
        if self.check(TT.USING):
            return self.parse_using_stmt()
        if self.check(TT.DEFER):
            return self.parse_defer_stmt()
        if self.check(TT.AWAIT):
            # Statement-form await with optional bodies/arms
            return self.parse_await_stmt()

        # Declarations
        if self.check(TT.HOOK):
            return self.parse_hook_stmt()
        if self.check(TT.DECORATOR):
            return self.parse_decorator_stmt()
        if self.check(TT.AT):
            # Decorator list followed by fn statement
            return self.parse_fn_stmt()
        if self.check(TT.FN):
            return self.parse_fn_stmt()

        # Simple statements
        if self.check(TT.RETURN):
            return self.parse_return_stmt()
        if self.check(TT.QMARK) and self.peek(1).type == TT.IDENT and self.peek(1).value == 'ret':
            # ?ret expr - returnif statement
            self.advance(2)  # consume "? ret"
            value = self.parse_expr()
            return Tree('returnif', [value])
        if self.check(TT.BREAK):
            return self.parse_break_stmt()
        if self.check(TT.CONTINUE):
            return self.parse_continue_stmt()
        if self.check(TT.THROW):
            return self.parse_throw_stmt()
        if self.check(TT.ASSERT):
            return self.parse_assert_stmt()
        if self.check(TT.DBG):
            return self.parse_dbg_stmt()
        if self.check(TT.AWAIT):
            return self.parse_await_stmt()

        # Guard chains (inline if-else)
        # expr : body | expr : body |: else
        expr_start = self.pos

        # Lookahead for destructuring with contracts
        # Scan ahead to detect pattern: "ident [~ contract] (, ident [~ contract])* (:=|=)"
        if self.check(TT.IDENT) and self._is_destructure_with_contracts():
            # Parse pattern(s) using general parse_pattern() to support nested patterns
            patterns = [self.parse_pattern()]

            while self.match(TT.COMMA):
                patterns.append(self.parse_pattern())

            # Always wrap in pattern_list for evaluator consistency
            pattern_node = Tree('pattern_list', patterns)

            if self.match(TT.ASSIGN):
                rhs = self.parse_destructure_rhs()
                return Tree('destructure', [pattern_node, rhs])
            elif self.match(TT.WALRUS):
                rhs = self.parse_destructure_rhs()
                return Tree('destructure_walrus', [pattern_node, rhs])
            else:
                raise ParseError("Expected = or := after pattern", self.current)

        expr = self.parse_expr()

        # Check for destructuring: a, b, c = ... or a, b, c := ...
        if self.check(TT.COMMA):
            first_tok = self.tokens[expr_start] if expr_start < len(self.tokens) else None
            if first_tok is None or first_tok.type not in {TT.IDENT, TT.LPAR}:
                # Not a valid pattern start; comma belongs to surrounding expression (e.g., anon fn body)
                pass
            else:
                # Backtrack and parse as pattern_list
                self.pos = expr_start
                self.current = self.tokens[self.pos] if self.pos < len(self.tokens) else Tok(TT.EOF, None, 0, 0)
                pattern_list = self.parse_pattern_list()

                if self.match(TT.ASSIGN):
                    rhs = self.parse_destructure_rhs()
                    return Tree('destructure', [pattern_list, rhs])
                elif self.match(TT.WALRUS):
                    rhs = self.parse_destructure_rhs()
                    return Tree('destructure_walrus', [pattern_list, rhs])
                else:
                    raise ParseError("Expected = or := after pattern list", self.current)

        # Guard chain inline form
        if self.check(TT.COLON):
            self.pos = expr_start
            self.current = self.tokens[self.pos] if self.pos < len(self.tokens) else Tok(TT.EOF, None, 0, 0)
            return self.parse_guard_chain()

        # Catch statement form: expr catch ... : body
        if self.check(TT.CATCH) or (self.check(TT.AT) and self.peek(1).type == TT.AT):
            self.pos = expr_start
            self.current = self.tokens[self.pos] if self.pos < len(self.tokens) else Tok(TT.EOF, None, 0, 0)
            return self.parse_catch_stmt()

        # Fanout block form: expr { .field = value; ... }
        if self.check(TT.LBRACE):
            fanblock = self.parse_fanblock()
            return Tree('fanoutblock', [expr, fanblock])

        # Base statement starts as expression
        base_stmt: Tree | Tok = expr

        # Assignment handling before postfix-if to match grammar precedence
        if self.check(
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
        ):
            if self.check(TT.ASSIGN):
                lvalue = self._expr_to_lvalue(expr)
                self.advance()  # =
                rhs = self.parse_nullish_expr()
                base_stmt = Tree('assignstmt', [lvalue, self._tok('ASSIGN', '='), rhs])
            else:
                lvalue = self._expr_to_lvalue(expr)
                op = self.advance()
                rhs = self.parse_nullish_expr()
                base_stmt = Tree('compound_assign', [lvalue, self._tok(op.type.name, op.value), rhs])

        # Postfix if/unless wraps the base statement
        if self.check(TT.IF):
            self.advance()
            cond = self.parse_expr()
            wrapped_base = base_stmt if isinstance(base_stmt, Tree) else Tree('expr', [base_stmt])
            return Tree('postfixif', [wrapped_base, self._tok('IF', 'if'), Tree('expr', [cond])])
        if self.check(TT.IDENT) and getattr(self.current, "value", "") == "unless":
            self.advance()
            cond = self.parse_expr()
            wrapped_base = base_stmt if isinstance(base_stmt, Tree) else Tree('expr', [base_stmt])
            return Tree('postfixunless', [wrapped_base, Tree('expr', [cond])])

        # Just a statement/expression
        if isinstance(base_stmt, Tree):
            return base_stmt
        return Tree('expr', [base_stmt])

    def parse_if_stmt(self) -> Tree:
        """
        Parse if statement:
        if expr: body [elif expr: body]* [else: body]
        """
        self.expect(TT.IF)
        cond = self.parse_expr()
        self.expect(TT.COLON)
        then_body = self.parse_body()

        elifs = []
        else_clause = None

        while self.check(TT.ELIF):
            self.advance()
            elif_cond = self.parse_expr()
            self.expect(TT.COLON)
            elif_body = self.parse_body()
            elifs.append(Tree('elifclause', [self._tok('ELIF', 'elif'), elif_cond, elif_body]))

        if self.check(TT.ELSE):
            self.advance()
            self.expect(TT.COLON)
            else_body = self.parse_body()
            else_clause = Tree('elseclause', [self._tok('ELSE', 'else'), else_body])

        children = [self._tok('IF', 'if'), cond, then_body] + elifs
        if else_clause:
            children.append(else_clause)

        return Tree('ifstmt', children)

    def parse_while_stmt(self) -> Tree:
        """Parse while loop: while expr: body"""
        self.expect(TT.WHILE)
        cond = self.parse_expr()
        self.expect(TT.COLON)
        body = self.parse_body()
        return Tree('whilestmt', [self._tok('WHILE', 'while'), cond, body])

    def parse_for_stmt(self) -> Tree:
        """
        Parse for loop:
        for x in expr: body (forin)
        for k, v in expr: body (forin with destructuring)
        for[i] expr: body (forindexed)
        for[k, v] expr: body (formap2)
        for expr: body (forsubject - subjectful loop)
        """
        self.expect(TT.FOR)

        # Check for indexed syntax: for[...] (pattern bindings)
        # Only consume [ if next token is IDENT or CARET (hoist marker)
        if self.check(TT.LSQB) and self.peek(1).type in {TT.IDENT, TT.CARET}:
            self.advance()  # consume [
            # Parse first binderpattern (handles ^ident hoist)
            binder1 = self.parse_binderpattern()

            # Check for comma - indicates formap2 (two patterns)
            if self.match(TT.COMMA):
                binder2 = self.parse_binderpattern()
                self.expect(TT.RSQB)
                # for[pattern, pattern] expr: body
                iterable = self.parse_expr()
                self.expect(TT.COLON)
                body = self.parse_body()
                return Tree('formap2', [self._tok('FOR', 'for'), binder1, binder2, iterable, body])
            else:
                self.expect(TT.RSQB)
                # for[pattern] expr: body
                iterable = self.parse_expr()
                self.expect(TT.COLON)
                body = self.parse_body()
                return Tree('forindexed', [self._tok('FOR', 'for'), binder1, iterable, body])

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
                        tok, new_idx, new_paren = self._lookahead_advance(idx, paren_depth)
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

                    self.expect(TT.IN)
                    iterable = self.parse_expr()
                    self.expect(TT.COLON)
                    body = self.parse_body()

                    # Build pattern
                    if len(idents) == 1:
                        # Simple pattern: for x in expr
                        pattern = Tree('pattern', [self._tok('IDENT', idents[0].value)])
                    else:
                        # Destructuring pattern: for k, v in expr
                        pattern_items = [self._tok('IDENT', id.value) for id in idents]
                        pattern = Tree('pattern_list_inline', pattern_items)

                    return Tree('forin', [self._tok('FOR', 'for'), pattern, self._tok('IN', 'in'), iterable, body])

        # Subjectful for: for expr: body
        iterable = self.parse_expr()
        self.expect(TT.COLON)
        body = self.parse_body()
        return Tree('forsubject', [self._tok('FOR', 'for'), iterable, body])

    def parse_using_stmt(self) -> Tree:
        """
        Parse using statement:
        using [handle] expr [bind ident]: body
        """
        self.expect(TT.USING)

        handle = None
        if self.match(TT.LSQB):
            ident_tok = self.expect(TT.IDENT)
            self.expect(TT.RSQB)
            handle = ident_tok

        resource = self.parse_expr()

        binder = None
        if self.check(TT.IDENT) and getattr(self.current, "value", "") == "bind":
            self.advance()
            binder = self.expect(TT.IDENT)

        self.expect(TT.COLON)
        body = self.parse_body()

        children: List[Any] = []
        if handle is not None:
            children.append(Tree('using_handle', [self._tok('IDENT', handle.value)]))
        children.append(resource)
        if binder is not None:
            children.append(Tree('using_bind', [self._tok('IDENT', binder.value)]))
        children.append(body)
        return Tree('usingstmt', children)

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
                defer_children.append(Tree('deferlabel', [self._tok('IDENT', label_tok.value)]))

            # Optional deferafter
            if self.check(TT.IDENT) and getattr(self.current, "value", "") == "after":
                self.advance()
                defer_children.append(self._parse_defer_after())

            self.expect(TT.COLON)
            body = self.parse_body()
            defer_children.append(Tree('defer_block', [body]))
            return Tree('deferstmt', defer_children)

        # simplecall branch
        call_expr = self.parse_postfix_expr()
        if not self._expr_has_call(call_expr):
            raise ParseError("defer expects a call expression or block", self.current)

        # Optional deferafter
        if self.check(TT.IDENT) and getattr(self.current, "value", "") == "after":
            self.advance()
            defer_children.append(self._parse_defer_after())
        defer_children.append(call_expr)

        return Tree('deferstmt', defer_children)

    def _parse_defer_after(self) -> Tree:
        deps: List[Tok] = []

        if self.match(TT.LPAR):
            if not self.check(TT.RPAR):
                deps.append(self._tok('IDENT', self.expect(TT.IDENT).value))
                while self.match(TT.COMMA):
                    deps.append(self._tok('IDENT', self.expect(TT.IDENT).value))
            self.expect(TT.RPAR)
        else:
            deps.append(self._tok('IDENT', self.expect(TT.IDENT).value))

        return Tree('deferafter', deps)

    def _expr_has_call(self, expr: Tree | Tok) -> bool:
        if isinstance(expr, Tree):
            if expr.data == 'call':
                return True
            return any(self._expr_has_call(ch) for ch in expr.children)
        return False

    def _scan_for_colon(self) -> bool:
        """Look ahead on the current line for a colon before newline/semicolon/dedent."""
        depth = 0
        idx = self.pos + 1
        while idx < len(self.tokens):
            tok = self.tokens[idx]
            if tok.type in {TT.NEWLINE, TT.SEMI, TT.DEDENT, TT.EOF} and depth == 0:
                return False
            if tok.type in {TT.LPAR, TT.LSQB, TT.LBRACE}:
                depth += 1
            elif tok.type in {TT.RPAR, TT.RSQB, TT.RBRACE} and depth > 0:
                depth -= 1
            if depth == 0 and tok.type == TT.COLON:
                return True
            idx += 1
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
                        if idx + 1 < len(self.tokens) and self.tokens[idx + 1].type == TT.COLON:
                            return True
                        break
                idx += 1
        # String key
        if self.check(TT.STRING, TT.RAW_STRING, TT.RAW_HASH_STRING):
            return True
        return False

    def parse_hook_stmt(self) -> Tree:
        """
        Parse hook statement (stub):
        hook "name": body

        Grammar: hook: HOOK (STRING | RAW_STRING | RAW_HASH_STRING) ":" (inlinebody | indentblock)
        """
        self.expect(TT.HOOK)
        name = self.expect(TT.STRING)  # TODO: support RAW_STRING, RAW_HASH_STRING
        self.expect(TT.COLON)
        body = self.parse_body()

        # Convert inlinebody to amp_lambda if it contains implicit chain
        # For now, just wrap the body directly
        return Tree('hook', [name, Tree('amp_lambda', [body])])

    def parse_decorator_stmt(self) -> Tree:
        """
        Parse decorator declaration:
        decorator name(params): body
        """
        self.expect(TT.DECORATOR)
        name = self.expect(TT.IDENT)

        self.expect(TT.LPAR)
        params = self.parse_param_list()
        self.expect(TT.RPAR)

        self.expect(TT.COLON)
        body = self.parse_body()

        return Tree('decorator_def', [self._tok('IDENT', name.value), params, body])

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
            if self.match(TT.NEWLINE):
                continue
            if self.match(TT.SEMI, TT.COMMA):
                continue

            # Parse fanclause: .field = expr
            self.expect(TT.DOT)

            # Parse fanpath: IDENT or [selectorlist] segments, dot-separated
            segs: List[Tree] = []

            # First segment (required)
            if self.check(TT.IDENT):
                tok = self.advance()
                segs.append(Tree('field', [self._tok('IDENT', tok.value, tok.line, tok.column)]))
            elif self.match(TT.LSQB):
                selectors = self.parse_selector_list()
                self.expect(TT.RSQB)
                segs.append(Tree('lv_index', [
                    self._tok('LSQB', '['),
                    Tree('selectorlist', selectors),
                    self._tok('RSQB', ']'),
                ]))
            else:
                raise ParseError("Expected identifier or [selector] in fanout path", self.current)

            # Subsequent segments: allow either a dot+segment or a bare [selector] segment
            while True:
                if self.match(TT.DOT):
                    if self.check(TT.IDENT):
                        tok = self.advance()
                        segs.append(Tree('field', [self._tok('IDENT', tok.value, tok.line, tok.column)]))
                    elif self.match(TT.LSQB):
                        selectors = self.parse_selector_list()
                        self.expect(TT.RSQB)
                        segs.append(Tree('lv_index', [
                            self._tok('LSQB', '['),
                            Tree('selectorlist', selectors),
                            self._tok('RSQB', ']'),
                        ]))
                    else:
                        raise ParseError("Expected identifier or [selector] after '.' in fanout path", self.current)
                elif self.check(TT.LSQB):
                    # Adjacent selector segment without a dot (e.g., .rows[1].v)
                    self.advance()  # consume '['
                    selectors = self.parse_selector_list()
                    self.expect(TT.RSQB)
                    segs.append(Tree('lv_index', [
                        self._tok('LSQB', '['),
                        Tree('selectorlist', selectors),
                        self._tok('RSQB', ']'),
                    ]))
                else:
                    break

            fanpath = Tree('fanpath', segs)

            # Parse assignment operator
            if self.check(TT.ASSIGN):
                self.advance()
                fanop = Tree('fanop_assign', [])
            elif self.check(TT.APPLYASSIGN):
                self.advance()
                fanop = Tree('fanop_apply', [])
            elif self.check(TT.PLUSEQ):
                self.advance()
                fanop = Tree('fanop_pluseq', [])
            elif self.check(TT.MINUSEQ):
                self.advance()
                fanop = Tree('fanop_minuseq', [])
            elif self.check(TT.STAREQ):
                self.advance()
                fanop = Tree('fanop_stareq', [])
            elif self.check(TT.SLASHEQ):
                self.advance()
                fanop = Tree('fanop_slasheq', [])
            elif self.check(TT.FLOORDIVEQ):
                self.advance()
                fanop = Tree('fanop_floordiveq', [])
            elif self.check(TT.MODEQ):
                self.advance()
                fanop = Tree('fanop_modeq', [])
            elif self.check(TT.POWEQ):
                self.advance()
                fanop = Tree('fanop_poweq', [])
            else:
                raise ParseError("Expected fanout assignment operator", self.current)

            # Parse value expression
            value = self.parse_expr()

            clauses.append(Tree('fanclause', [self._tok('DOT', '.'), fanpath, fanop, value]))
            clause_count += 1

        self.expect(TT.RBRACE)
        # Single-clause fanout: allow if it either fans to multiple targets *or*
        # the RHS uses the implicit subject (e.g., state{ .cur = .next }).
        if clause_count == 1:
            fanpath = clauses[0].children[1]
            value_expr = clauses[0].children[3]
            if not (self._fanpath_has_multiselector(fanpath) or self._expr_uses_subject(value_expr)):
                raise ParseError("Single-clause fanout requires a multi-selector or implicit-subject RHS", self.current)
        return Tree('fanblock', clauses)

    def _fanpath_has_multiselector(self, fanpath: Tree) -> bool:
        """Detect slice or multi-index selector in fanpath segments."""
        for seg in fanpath.children:
            if isinstance(seg, Tree) and seg.data == 'lv_index':
                selectorlist = next((ch for ch in seg.children if isinstance(ch, Tree) and ch.data == 'selectorlist'), None)
                if selectorlist is None:
                    continue
                selectors = [ch for ch in selectorlist.children if isinstance(ch, Tree)]
                if len(selectors) > 1:
                    return True
                if selectors:
                    sel = selectors[0]
                    if any(isinstance(grand, Tree) and grand.data == 'slicesel' for grand in sel.children):
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
            while self.match(TT.NEWLINE):
                pass
            # decorator_entry: just the expression, no @ token
            decorators.append(Tree('decorator_entry', [decorator_expr]))

        self.expect(TT.FN)
        name = self.expect(TT.IDENT)

        self.expect(TT.LPAR)
        params = self.parse_param_list()
        self.expect(TT.RPAR)

        # Optional return contract: ~ Schema
        return_contract = None
        if self.match(TT.TILDE):
            return_contract = self.parse_expr()

        self.expect(TT.COLON)
        body = self.parse_body()

        # fnstmt structure: [name, params, body, return_contract?, decorator_list?]
        children = [self._tok('IDENT', name.value), params, body]

        if return_contract is not None:
            children.append(Tree('return_contract', [return_contract]))

        if decorators:
            decorator_list = Tree('decorator_list', decorators)
            children.append(decorator_list)

        return Tree('fnstmt', children)

    def parse_await_stmt(self) -> Tree:
        """
        Parse await statement forms:
        - await [any](arms) [: body]
        - await [all](arms) [: body]
        - await expr : body
        """
        await_tok = self.expect(TT.AWAIT)

        # await [any|all](...) ...
        if self.match(TT.LSQB):
            kind_tok = self.advance()  # expect ANY or ALL identifiers
            kind = getattr(kind_tok, "type", None)
            self.expect(TT.RSQB)
            self.expect(TT.LPAR)
            arms = self.parse_await_arm_list(kind_tok)
            self.expect(TT.RPAR)

            trailing_body = None
            if self.match(TT.COLON):
                trailing_body = self.parse_body()

            root_label = 'awaitanycall' if kind == TT.ANY else 'awaitallcall'
            children: List[Any] = [Tree('anyarmlist' if root_label == 'awaitanycall' else 'allarmlist', arms)]
            if trailing_body is not None:
                children.append(self._tok('RPAR', ')'))
                children.append(trailing_body)
            return Tree(root_label, children)

        # await expr [: body]?
        expr = self.parse_expr()
        if self.match(TT.COLON):
            body = self.parse_body()
            return Tree('awaitstmt', [self._tok('AWAIT', await_tok.value), expr, body])
        return Tree('await_value', [self._tok('AWAIT', await_tok.value), expr])

    def parse_return_stmt(self) -> Tree:
        """Parse return statement: return [expr | pack]"""
        self.expect(TT.RETURN)

        # Check if there's a value
        if self.check(TT.NEWLINE, TT.EOF, TT.SEMI, TT.RBRACE):
            return Tree('returnstmt', [self._tok('RETURN', 'return')])

        # Parse first expression
        first_expr = self.parse_expr()

        # Check for pack (comma-separated expressions)
        if self.check(TT.COMMA):
            exprs = [first_expr]
            while self.match(TT.COMMA):
                exprs.append(self.parse_expr())
            pack = Tree('pack', exprs)
            return Tree('returnstmt', [self._tok('RETURN', 'return'), pack])

        return Tree('returnstmt', [self._tok('RETURN', 'return'), first_expr])

    def parse_break_stmt(self) -> Tree:
        """Parse break statement"""
        self.expect(TT.BREAK)
        return Tree('breakstmt', [self._tok('BREAK', 'break')])

    def parse_continue_stmt(self) -> Tree:
        """Parse continue statement"""
        self.expect(TT.CONTINUE)
        return Tree('continuestmt', [self._tok('CONTINUE', 'continue')])

    def parse_throw_stmt(self) -> Tree:
        """Parse throw statement: throw expr"""
        self.expect(TT.THROW)
        value = self.parse_expr()
        return Tree('throwstmt', [value])

    def parse_assert_stmt(self) -> Tree:
        """Parse assert: assert expr [, message]"""
        self.expect(TT.ASSERT)
        value = self.parse_expr()
        # Optional comma and message expression
        if self.match(TT.COMMA):
            message = self.parse_expr()
            return Tree('assert', [value, message])
        return Tree('assert', [value])

    def parse_dbg_stmt(self) -> Tree:
        """Parse dbg: DBG (expr ("," expr)?)"""
        self.expect(TT.DBG)
        first_expr = self.parse_expr()
        if self.match(TT.COMMA):
            second_expr = self.parse_expr()
            return Tree('dbg', [self._tok('DBG', 'dbg'), first_expr, second_expr])
        return Tree('dbg', [self._tok('DBG', 'dbg'), first_expr])

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
        branches.append(Tree('guardbranch', [cond, body]))

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
                    branches.append(Tree('guardbranch', [cond, body]))
            else:
                # No continuation, exit without consuming tokens
                break

        return Tree('onelineguard', branches)

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
            # Build a stmtlist and wrap in inlinebody to match Lark grammar
            stmts: list[Node] = []
            while not self.check(TT.RBRACE, TT.EOF):
                if self.match(TT.NEWLINE):
                    continue
                if self.match(TT.SEMI):
                    stmts.append(self._tok('SEMI', ';'))
                    continue
                stmt = self.parse_statement()
                stmts.append(Tree('stmt', [stmt]))
            self.expect(TT.RBRACE)
            # Wrap stmtlist in inlinebody
            if stmts:
                return Tree('inlinebody', [Tree('stmtlist', stmts)])
            return Tree('inlinebody', [])

        if self.check(TT.INDENT):
            # Indented block
            indent_tok = self.advance()
            # Use actual indent value from lexer
            children: list[Node] = [self._tok('INDENT', indent_tok.value if indent_tok.value is not None else '    ')]

            # Parse statements in the block
            while not self.check(TT.DEDENT, TT.EOF):
                if self.match(TT.NEWLINE):
                    continue
                if self.match(TT.SEMI):
                    children.append(self._tok('SEMI', ';'))
                    continue
                stmt = self.parse_statement()
                children.append(Tree('stmt', [stmt]))
                # Skip newlines between statements
                while self.check(TT.NEWLINE) and not self.check(TT.DEDENT):
                    self.match(TT.NEWLINE)

            dedent_tok = self.expect(TT.DEDENT)
            children.append(self._tok('DEDENT', dedent_tok.value if dedent_tok.value is not None else ''))
            return Tree('indentblock', children)

        # Single statement inline
        old_inline = self.in_inline_body
        self.in_inline_body = True
        stmt = self.parse_statement()
        self.in_inline_body = old_inline
        return Tree('inlinebody', [stmt])

    # ========================================================================
    # Expressions - Precedence Climbing
    # ========================================================================

    def parse_expr(self) -> Tree:
        """
        Parse expression (top level).
        Handles catch, ternary, binary ops, etc.
        """
        # Wrap in expr node to match Lark
        return Tree('expr', [self.parse_catch_expr()])

    def parse_catch_expr(self) -> Node:
        """Parse catch expression: expr catch [types] [bind x]: handler"""
        expr = self.parse_ternary_expr()

        if self.check(TT.CATCH) or (self.check(TT.AT) and self.peek(1).type == TT.AT):
            # catch or @@ syntax
            if self.check(TT.CATCH):
                self.advance()
            else:
                self.advance(2)  # consume '@@'

            # Optional type filter: (Type1, Type2)
            types: List[Tok] = []
            if self.match(TT.LPAR):
                ident_tok = self.expect(TT.IDENT)
                types.append(self._tok('IDENT', ident_tok.value))
                while self.match(TT.COMMA):
                    ident_tok = self.expect(TT.IDENT)
                    types.append(self._tok('IDENT', ident_tok.value))
                self.expect(TT.RPAR)

            # Optional binder: either bare IDENT or 'bind' IDENT
            binder: Optional[Tok] = None
            if self.check(TT.IDENT) and getattr(self.current, "value", "") == "bind":
                self.advance()
                binder = self.expect(TT.IDENT)
            elif self.check(TT.IDENT) and not types:
                # binder_simple form when no typed list
                binder = self.advance()

            self.expect(TT.COLON)
            # Choose expression vs block handler
            if self.check(TT.LBRACE, TT.INDENT) or (self.check(TT.NEWLINE) and self.peek(1).type == TT.INDENT):
                handler = self.parse_body()
                is_stmt = True
            else:
                handler = self.parse_expr()
                is_stmt = False

            # Build catch node
            children: list[Node] = [expr]
            if binder:
                children.append(self._tok('IDENT', binder.value))
            if types:
                children.append(Tree('catchtypes', types))
            children.append(handler)

            return Tree('catchstmt' if is_stmt else 'catchexpr', children)

        return expr

    def parse_catch_stmt(self) -> Tree:
        """Parse catch statement starting at current position."""
        try_expr = self.parse_expr()

        if not (self.check(TT.CATCH) or (self.check(TT.AT) and self.peek(1).type == TT.AT)):
            return try_expr

        if self.check(TT.CATCH):
            self.advance()
        else:
            self.advance(2)  # consume '@@'

        types: List[Tok] = []
        if self.match(TT.LPAR):
            ident_tok = self.expect(TT.IDENT)
            types.append(self._tok('IDENT', ident_tok.value))
            while self.match(TT.COMMA):
                ident_tok = self.expect(TT.IDENT)
                types.append(self._tok('IDENT', ident_tok.value))
            self.expect(TT.RPAR)

        binder_tok = None
        if self.check(TT.IDENT) and getattr(self.current, "value", "") == "bind":
            self.advance()
            binder_tok = self.expect(TT.IDENT)
        elif self.check(TT.IDENT) and not types:
            binder_tok = self.advance()

        self.expect(TT.COLON)
        body = self.parse_body()

        children: List[Any] = [try_expr]
        if binder_tok is not None:
            children.append(self._tok('IDENT', binder_tok.value))
        if types:
            children.append(Tree('catchtypes', types))
        children.append(body)
        return Tree('catchstmt', children)

    def parse_ternary_expr(self) -> Node:
        """Parse ternary: expr ? then : else"""
        expr = self.parse_or_expr()

        if self.match(TT.QMARK):
            then_expr = self.parse_expr()
            self.expect(TT.COLON)
            else_expr = self.parse_ternary_expr()  # Right associative
            return Tree('ternary', [expr, then_expr, else_expr])

        # No ternary, return unwrapped
        return expr

    def parse_or_expr(self) -> Node:
        """Parse logical OR: expr || expr"""
        left = self.parse_and_expr()

        if not self.check(TT.OR):
            # No OR, just return and level
            return left

        # Collect operands and operators (interleaved)
        children: list[Node] = [left]
        while self.check(TT.OR):
            op = self.advance()
            children.append(self._tok(op.type.name, op.value))
            right = self.parse_and_expr()
            children.append(right)

        return Tree('or', children)

    def parse_and_expr(self) -> Node:
        """Parse logical AND: expr && expr"""
        left = self.parse_bind_expr()

        if not self.check(TT.AND):
            # No AND, just return bind level
            return left

        # Collect operands and operators (interleaved)
        children: list[Node] = [left]
        while self.check(TT.AND):
            op = self.advance()
            children.append(self._tok(op.type.name, op.value))
            right = self.parse_bind_expr()
            children.append(right)

        return Tree('and', children)

    def parse_bind_expr(self) -> Node:
        """Parse apply bind: lvalue .= expr"""
        left = self.parse_walrus_expr()

        if self.match(TT.APPLYASSIGN):
            lvalue = self._expr_to_lvalue(left)

            right = self.parse_bind_expr()  # Right associative
            return Tree('bind', [lvalue, right])

        # No bind, just return walrus level
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
            return Tree('lvalue', [core_expr])

        if isinstance(core_expr, Tree) and core_expr.data in {'explicit_chain', 'implicit_chain'}:
            if not core_expr.children:
                return Tree('lvalue', [core_expr])

            head, *ops = core_expr.children

            norm_ops: List[Tree | Tok] = []
            for op in ops:
                if isinstance(op, Tree) and op.data == 'index':
                    norm_ops.append(Tree('lv_index', list(op.children)))
                elif isinstance(op, Tree) and op.data == 'valuefan':
                    # valuefan is expression fan-out; for assignment treat as fieldfan
                    norm_ops.append(Tree('fieldfan', op.children))
                else:
                    norm_ops.append(op)

            return Tree('lvalue', [head] + norm_ops)

        return Tree('lvalue', [expr])

    def parse_walrus_expr(self) -> Node:
        """Parse walrus: x := expr"""
        # Check for walrus pattern: IDENT :=
        if self.check(TT.IDENT) and self.peek(1).type == TT.WALRUS:
            name = self.current
            self.advance(2)  # consume IDENT and :=
            # Parse the RHS
            value = self.parse_catch_expr()  # allow catch expressions in walrus RHS
            return Tree('walrus', [self._tok('IDENT', name.value), value])

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
            children.append(self._tok('NULLISH', op.value))
            right = self.parse_compare_expr()
            children.append(right)

        return Tree('nullish', children)

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
        op_tree = Tree('cmpop', op_tokens)
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
                    self.advance()
                    or_token = self._tok('OR', 'or')
                elif self.check(TT.AND):
                    self.advance()
                    and_token = self._tok('AND', 'and')

                # Parse next leg
                leg_op_tokens = self.parse_compare_op() if self.is_compare_op() else None
                leg_value = self.parse_add_expr()

                leg_parts: list[Node] = []
                if or_token:
                    leg_parts.append(or_token)
                    if leg_op_tokens:
                        leg_op = Tree('cmpop', leg_op_tokens)
                        leg_parts.extend([leg_op, leg_value])
                    else:
                        leg_parts.append(leg_value)
                    children.append(Tree('ccc_or_leg', leg_parts))
                else:
                    if and_token:
                        leg_parts.append(and_token)
                    if leg_op_tokens:
                        leg_op = Tree('cmpop', leg_op_tokens)
                        leg_parts.extend([leg_op, leg_value])
                    else:
                        leg_parts.append(leg_value)
                    children.append(Tree('ccc_and_leg', leg_parts))

                # Check if next comma is also CCC
                if not self.comma_is_ccc():
                    break
                self.match(TT.COMMA)  # consume it

            return Tree('compare', children)

        return Tree('compare', [left, op_tree, right])

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

        # `, or` → definitely CCC OR leg
        if next_tok.type == TT.OR:
            return True

        # `, and` → definitely CCC AND leg (explicit)
        if next_tok.type == TT.AND:
            return True

        # Check if followed by comparison operator (`, <cmpop>` → implicit AND)
        # Lookahead past the comma
        _, idx, _ = self._lookahead_advance(self.pos, 0)  # skip comma
        is_cmp = idx < len(self.tokens) and self._is_compare_op_at(idx)

        if is_cmp:
            return True  # `, <cmpop>` → CCC implicit AND

        # At this point: `, <addexpr>` (bare value, no cmpop)
        # This is ambiguous - use context to decide:
        # - In separator contexts (args/array/pack): NOT CCC
        # - In normal context: IS CCC (implicit AND with bare addexpr)
        if self.parse_context in (ParseContext.FUNCTION_ARGS,
                                   ParseContext.ARRAY_ELEMENTS,
                                   ParseContext.DESTRUCTURE_PACK):
            return False  # Comma is a separator

        return True  # Default: allow CCC in normal context

    def _is_compare_op_at(self, idx: int) -> bool:
        """Check if token at idx is a comparison operator (stateless helper)"""
        if idx >= len(self.tokens):
            return False
        tok = self.tokens[idx]
        if tok.type in {TT.EQ, TT.NEQ, TT.LT, TT.LTE, TT.GT, TT.GTE, TT.IS, TT.IN, TT.TILDE, TT.REGEXMATCH}:
            return True
        if tok.type == TT.NOT and idx + 1 < len(self.tokens) and self.tokens[idx + 1].type == TT.IN:
            return True
        if tok.type == TT.NEG and idx + 1 < len(self.tokens) and self.tokens[idx + 1].type in {TT.IS, TT.IN}:
            return True
        return False

    def is_compare_op(self) -> bool:
        """Check if current token is a comparison operator"""
        return self._is_compare_op_at(self.pos)

    def parse_compare_op(self) -> List[Tok]:
        """Parse comparison operator - returns list of tokens for compound ops"""
        if self.check(TT.EQ, TT.NEQ, TT.LT, TT.LTE, TT.GT, TT.GTE, TT.TILDE, TT.REGEXMATCH):
            op = self.advance()
            return [self._tok(op.type.name, op.value)]

        if self.check(TT.IS):
            self.advance()
            if self.match(TT.NOT):
                return [self._tok('IS', 'is'), self._tok('NOT', 'not')]
            return [self._tok('IS', 'is')]

        if self.check(TT.IN):
            self.advance()
            return [self._tok('IN', 'in')]

        if self.match(TT.NOT):
            self.expect(TT.IN)
            return [self._tok('NOT', 'not'), self._tok('IN', 'in')]

        # Check for !is and !in
        if self.match(TT.NEG):  # ! token
            if self.match(TT.IS):
                return [self._tok('NEG', '!'), self._tok('IS', 'is')]
            elif self.match(TT.IN):
                return [self._tok('NEG', '!'), self._tok('IN', 'in')]
            else:
                raise ParseError("Expected 'is' or 'in' after '!'", self.current)

        raise ParseError("Expected comparison operator", self.current)

    def parse_add_expr(self) -> Node:
        """Parse addition/subtraction: expr + expr"""
        left = self.parse_mul_expr()

        ops_and_operands = []
        while self.check(TT.PLUS, TT.MINUS, TT.DEEPMERGE, TT.CARET):
            op = self.advance()
            right = self.parse_mul_expr()
            ops_and_operands.append((op, right))

        # If no operators, just return wrapped mulexpr
        if not ops_and_operands:
            return left

        # Build add with left-associative structure
        for op, right in ops_and_operands:
            # Wrap operator in addop tree
            op_tree = Tree('addop', [self._tok(op.type.name, op.value)])
            left = Tree('add', [left, op_tree, right])

        return left

    def parse_mul_expr(self) -> Node:
        """Parse multiplication/division: expr * expr"""
        left = self.parse_pow_expr()

        ops_and_operands = []
        while self.check(TT.STAR, TT.SLASH, TT.FLOORDIV, TT.MOD):
            op = self.advance()
            right = self.parse_pow_expr()
            ops_and_operands.append((op, right))

        if not ops_and_operands:
            return left

        for op, right in ops_and_operands:
            # Wrap operator in mulop tree
            op_tree = Tree('mulop', [self._tok(op.type.name, op.value)])
            left = Tree('mul', [left, op_tree, right])

        return left

    def parse_pow_expr(self) -> Node:
        """Parse exponentiation: expr ** expr (right associative)"""
        base = self.parse_unary_expr()

        if self.match(TT.POW):
            exp = self.parse_pow_expr()  # Right associative
            return Tree('pow', [base, self._tok('POW', '**'), exp])

        # No power operator, return unwrapped
        return base

    def parse_unary_expr(self) -> Node:
        """Parse unary operators: -expr, !expr, ++expr, await expr"""
        # Throw as expression-form
        if self.check(TT.THROW):
            self.advance()
            if self.check(TT.NEWLINE, TT.EOF, TT.SEMI, TT.RBRACE, TT.RPAR, TT.COMMA):
                return Tree('throwstmt', [])
            value = self.parse_unary_expr()
            return Tree('throwstmt', [value])

        # Await
        if self.check(TT.AWAIT):
            self.advance()
            # await expr or await (expr)
            if self.match(TT.LPAR):
                expr = self.parse_expr()
                self.expect(TT.RPAR)
            else:
                expr = self.parse_unary_expr()
            return Tree('await_value', [self._tok('AWAIT', 'await'), expr])

        # $ (no anchor)
        if self.match(TT.DOLLAR):
            expr = self.parse_unary_expr()
            return Tree('no_anchor', [expr])

        # Spread prefix
        if self.match(TT.SPREAD):
            expr = self.parse_unary_expr()
            return Tree('spread', [expr])

        # Unary prefix operators
        if self.check(TT.MINUS, TT.NOT, TT.NEG, TT.INCR, TT.DECR):
            op = self.advance()
            expr = self.parse_unary_expr()
            op_tree = Tree('unaryprefixop', [self._tok(op.type.name, op.value)])
            return Tree('unary', [op_tree, expr])

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
            # Lookahead to determine if it's subject or implicit chain
            next_tok = self.peek(1)

            # Standalone subject: just .
            if next_tok.type not in (TT.IDENT, TT.LPAR, TT.LSQB, TT.OVER):
                dot = self.advance()
                return Tree('subject', [self._tok('DOT', dot.value, dot.line, dot.column)])

            # Implicit chain: .field, .(args), .[index], .over
            self.advance()  # consume .

            # Parse the first implicit operation
            imphead = None
            if self.check(TT.IDENT, TT.OVER):
                tok = self.advance()
                imphead = Tree('field', [self._tok(tok.type.name, tok.value, tok.line, tok.column)])
            elif self.match(TT.LPAR):
                args = self.parse_arg_list()
                self.expect(TT.RPAR)
                imphead = Tree('call', args if args else [])
            elif self.check(TT.LSQB):
                self.advance()
                selectors = self.parse_selector_list()
                default = None
                if self.match(TT.COMMA):
                    # Expect 'default' keyword
                    default_tok = self.expect(TT.IDENT)
                    if default_tok.value != 'default':
                        raise ParseError(f"Expected 'default' keyword, got '{default_tok.value}'", default_tok)
                    self.expect(TT.COLON)
                    default = self.parse_expr()
                self.expect(TT.RSQB)
                children = [
                    self._tok('LSQB', '['),
                    Tree('selectorlist', selectors),
                    self._tok('RSQB', ']')
                ]
                if default:
                    children.append(default)
                imphead = Tree('index', children)

            # Now collect any additional postfix operations
            postfix_ops = [imphead]
            while True:
                if self.match(TT.DOT):
                    if self.check(TT.IDENT, TT.OVER):
                        field = self.advance()
                        postfix_ops.append(Tree('field', [self._tok(field.type.name, field.value, field.line, field.column)]))
                    elif self.match(TT.LBRACE):
                        items = self.parse_fan_items()
                        self.expect(TT.RBRACE)
                        # Build valuefan_item nodes from parsed items (IDENT or identchain)
                        valuefan_items = []
                        for item in items:
                            valuefan_items.append(Tree('valuefan_item', [item]))
                        postfix_ops.append(Tree('valuefan', [Tree('valuefan_list', valuefan_items)]))
                    else:
                        raise ParseError("Expected field name after '.'", self.current)
                elif self.check(TT.LSQB):
                    self.advance()
                    selectors = self.parse_selector_list()
                    default = None
                    if self.match(TT.COMMA):
                        self.expect(TT.IDENT)
                        self.expect(TT.COLON)
                        default = self.parse_expr()
                    self.expect(TT.RSQB)
                    children = [
                        self._tok('LSQB', '['),
                        Tree('selectorlist', selectors),
                        self._tok('RSQB', ']')
                    ]
                    if default:
                        children.append(default)
                    postfix_ops.append(Tree('index', children))
                elif self.match(TT.LPAR):
                    args = self.parse_arg_list()
                    self.expect(TT.RPAR)
                    if args:
                        args = [Tree('arglistnamedmixed', args)]
                    else:
                        args = []
                    postfix_ops.append(Tree('call', args))
                elif self.check(TT.INCR, TT.DECR):
                    op = self.advance()
                    postfix_ops.append(Tree(op.type.name.lower(), []))
                else:
                    break

            return Tree('implicit_chain', postfix_ops)

        # Otherwise parse normal primary + postfix
        primary = self.parse_primary_expr()

        # Collect postfix operations
        postfix_ops = []

        while True:
            # Field access
            if self.match(TT.DOT):
                if self.check(TT.IDENT, TT.OVER):
                    field = self.advance()
                    postfix_ops.append(Tree('field', [self._tok(field.type.name, field.value, field.line, field.column)]))
                elif self.match(TT.LBRACE):
                    # Fan syntax: .{field1, field2} or .{chain1, chain2}
                    items = self.parse_fan_items()
                    self.expect(TT.RBRACE)
                    # Build valuefan_item nodes from parsed items (IDENT or identchain)
                    valuefan_items = []
                    for item in items:
                        valuefan_items.append(Tree('valuefan_item', [item]))
                    postfix_ops.append(Tree('valuefan', [Tree('valuefan_list', valuefan_items)]))
                else:
                    raise ParseError("Expected field name after '.'", self.current)

            # Indexing
            elif self.check(TT.LSQB):
                self.advance()
                selectors = self.parse_selector_list()

                # Check for default value
                default = None
                if self.match(TT.COMMA):
                    # Expect 'default' keyword
                    default_tok = self.expect(TT.IDENT)
                    if default_tok.value != 'default':
                        raise ParseError(f"Expected 'default' keyword, got '{default_tok.value}'", default_tok)
                    self.expect(TT.COLON)
                    default = self.parse_expr()

                self.expect(TT.RSQB)

                # Build index tree: [ selectorlist ]
                children = [
                    self._tok('LSQB', '['),
                    Tree('selectorlist', selectors),
                    self._tok('RSQB', ']')
                ]
                if default:
                    children.append(default)
                postfix_ops.append(Tree('index', children))

            # Call
            elif self.match(TT.LPAR):
                args = self.parse_arg_list()
                self.expect(TT.RPAR)
                if args:
                    args = [Tree('arglistnamedmixed', args)]
                else:
                    args = []
                postfix_ops.append(Tree('call', args))

            # Postfix increment/decrement
            elif self.check(TT.INCR, TT.DECR):
                op = self.advance()
                postfix_ops.append(Tree(op.type.name.lower(), []))

            # Postfix amp-lambda: expr&(body) or expr&[params](body)
            elif self.match(TT.AMP):
                lam = self.parse_anonymous_fn()
                # Wrap as lambdacall1 or lambdacalln depending on whether it has params
                if len(lam.children) == 2:  # Has paramlist
                    postfix_ops.append(Tree('lambdacalln', lam.children))
                else:  # Subject-based lambda
                    postfix_ops.append(Tree('lambdacall1', lam.children))

            else:
                break

        # Only wrap in explicit_chain if there are postfix operations
        if not postfix_ops:
            return primary

        return Tree('explicit_chain', [primary] + postfix_ops)

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

        # Rebind primary: =ident or =(lvalue)
        if self.match(TT.ASSIGN):
            if self.match(TT.LPAR):
                # =(lvalue) - grouped rebind
                lvalue = self.parse_rebind_lvalue()
                self.expect(TT.RPAR)
                return Tree('rebind_primary_grouped', [lvalue])
            else:
                # =ident - simple rebind
                ident = self.expect(TT.IDENT)
                return Tree('rebind_primary', [self._tok('IDENT', ident.value)])

        # Null-safe chain: ??(expr)
        if self.match(TT.NULLISH):
            self.expect(TT.LPAR)
            inner = self.parse_expr()
            self.expect(TT.RPAR)
            return Tree('nullsafe', [inner])

        # Literals - just return tokens, Prune() will wrap if needed
        if self.check(TT.NUMBER):
            tok = self.advance()
            return self._tok('NUMBER', tok.value)

        # Hole placeholder for partial application
        if self.match(TT.QMARK):
            return Tree('holeexpr', [])

        if self.check(TT.STRING, TT.RAW_STRING, TT.RAW_HASH_STRING, TT.SHELL_STRING, TT.PATH_STRING, TT.REGEX):
            tok = self.advance()
            # Wrap in 'literal' tree so Prune transformer can process string interpolation
            return Tree('literal', [self._tok(tok.type.name, tok.value)])

        if self.match(TT.TRUE):
            return self._tok('TRUE', 'true')
        if self.match(TT.FALSE):
            return self._tok('FALSE', 'false')
        if self.match(TT.NIL):
            return self._tok('NIL', 'nil')

        # Set literal / comprehension
        if self.match(TT.SET):
            self.expect(TT.LBRACE)
            if self.check(TT.RBRACE):
                self.advance()
                return Tree('setliteral', [])

            first_expr = self.parse_expr()
            if self.check(TT.FOR, TT.OVER):
                comphead = self.parse_comphead()
                ifclause = self.parse_ifclause_opt()
                self.expect(TT.RBRACE)
                children = [first_expr, comphead]
                if ifclause:
                    children.append(ifclause)
                return Tree('setcomp', children)

            items = [first_expr]
            while self.match(TT.COMMA):
                if self.check(TT.RBRACE):
                    break
                items.append(self.parse_expr())
            self.expect(TT.RBRACE)
            return Tree('setliteral', items)

        # Identifiers
        if self.check(TT.IDENT):
            tok = self.advance()
            return self._tok('IDENT', tok.value)

        # Special identifiers
        if self.match(TT.OVER):
            return self._tok('OVER', '_')
        if self.match(TT.ANY):
            return self._tok('ANY', 'any')
        if self.match(TT.ALL):
            return self._tok('ALL', 'all')

        # Parenthesized expression
        if self.match(TT.LPAR):
            # Reset context: parens allow CCC (grouping indicates single value)
            saved_context = self.parse_context
            self.parse_context = ParseContext.NORMAL
            try:
                expr = self.parse_expr()
                self.expect(TT.RPAR)
                return Tree('group_expr', [expr])
            finally:
                self.parse_context = saved_context

        # Array literal
        if self.match(TT.LSQB):
            # Skip layout tokens in array
            self.skip_layout_tokens()

            if self.check(TT.RSQB):
                self.advance()
                return Tree('array', [])

            # Set context: commas in arrays are element separators, not CCC
            saved_context = self.parse_context
            self.parse_context = ParseContext.ARRAY_ELEMENTS
            try:
                first_elem = self.parse_expr()

                if self.check(TT.FOR, TT.OVER):
                    comphead = self.parse_comphead()
                    ifclause = self.parse_ifclause_opt()
                    self.expect(TT.RSQB)
                    children = [first_elem, comphead]
                    if ifclause:
                        children.append(ifclause)
                    return Tree('listcomp', children)

                elements = [first_elem]

                # Skip layout tokens
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
                return Tree('array', elements)
            finally:
                self.parse_context = saved_context

        # Object literal
        if self.match(TT.LBRACE):
            # Try dict comprehension detection
            saved_pos = self.pos
            saved_current = self.current
            saved_paren_depth = self.paren_depth

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
                        return Tree('dictcomp', children)
            except ParseError:
                pass

            # reset cursor if not dictcomp
            self.pos = saved_pos
            self.paren_depth = saved_paren_depth
            self.current = saved_current

            items = []

            while not self.check(TT.RBRACE, TT.EOF):
                # Skip newlines
                while self.match(TT.NEWLINE) or self.match(TT.INDENT) or self.match(TT.DEDENT):
                    pass

                if self.check(TT.RBRACE):
                    break

                items.append(self.parse_object_item())

                # Skip newlines/indent markers
                while self.match(TT.NEWLINE) or self.match(TT.INDENT) or self.match(TT.DEDENT):
                    pass

                if not self.match(TT.COMMA):
                    # Allow optional trailing comma but permit newline-separated entries
                    while self.match(TT.NEWLINE) or self.match(TT.INDENT) or self.match(TT.DEDENT):
                        pass
                    if self.check(TT.RBRACE):
                        break
                    continue

            self.expect(TT.RBRACE)
            return Tree('object', items)

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
            # Wrap in sellist node as expected by evaluator
            sellist = Tree('sellist', selectors)
            return Tree('selectorliteral', [sellist])

        raise ParseError(f"Unexpected token in expression: {self.current.type.name}", self.current)

    # ========================================================================
    # Helper Parsers
    # ========================================================================

    def parse_rebind_lvalue(self) -> Tree:
        """
        Parse rebind lvalue: IDENT (postfix_field | postfix_index)* (fieldfan)?
        Used in =(lvalue) expressions
        """
        ident = self.expect(TT.IDENT)
        children: list[Node] = [self._tok('IDENT', ident.value)]

        # Parse postfix operations (field, index)
        while True:
            if self.match(TT.DOT):
                field = self.expect(TT.IDENT)
                children.append(Tree('field', [self._tok('IDENT', field.value, field.line, field.column)]))
            elif self.match(TT.LSQB):
                selectors = self.parse_selector_list()
                self.expect(TT.RSQB)
                children.append(Tree('index', [
                    self._tok('LSQB', '['),
                    Tree('selectorlist', selectors),
                    self._tok('RSQB', ']')
                ]))
            else:
                break

        # Optional fieldfan: .{field1, field2}
        if self.check(TT.DOT):
            # Peek ahead to see if it's followed by {
            if self.peek(1).type == TT.LBRACE:
                fieldfan = self.parse_fieldfan()
                children.append(fieldfan)

        return Tree('rebind_lvalue', children)

    def parse_fieldfan(self) -> Tree:
        """Parse fieldfan: .{fieldlist} where fieldlist is IDENT ("," IDENT)*"""
        self.expect(TT.DOT)
        self.expect(TT.LBRACE)

        # Parse fieldlist
        fields = []
        fields.append(self.expect(TT.IDENT))

        while self.match(TT.COMMA):
            fields.append(self.expect(TT.IDENT))

        self.expect(TT.RBRACE)

        # Build fieldlist tree
        fieldlist = Tree('fieldlist', [self._tok('IDENT', f.value) for f in fields])
        return Tree('fieldfan', [self._tok('DOT', '.'), fieldlist])

    def parse_param_list(self) -> Tree:
        """Parse function parameter list with optional contracts"""
        params: List[Node] = []

        while not self.check(TT.RPAR, TT.EOF):
            is_spread = self.match(TT.SPREAD)

            param = self.expect(TT.IDENT)

            # Optional contract: ~ expr
            contract = None
            if self.match(TT.TILDE):
                contract = self.parse_expr()

            # Optional default value
            default = None
            if self.match(TT.ASSIGN):
                default = self.parse_expr()

            if is_spread and (contract or default):
                raise ParseError("Spread parameter cannot have a contract or default", self.current)

            if contract and default:
                params.append(Tree('param', [self._tok('IDENT', param.value), Tree('contract', [contract]), default]))
            elif contract:
                params.append(Tree('param', [self._tok('IDENT', param.value), Tree('contract', [contract])]))
            elif default:
                params.append(Tree('param', [self._tok('IDENT', param.value), default]))
            else:
                if is_spread:
                    params.append(Tree('param_spread', [self._tok('IDENT', param.value)]))
                else:
                    params.append(self._tok('IDENT', param.value))

            if not self.match(TT.COMMA):
                break

        return Tree('paramlist', params)

    def parse_pattern(self) -> Tree:
        if self.check(TT.IDENT):
            tok = self.advance()
            # Check for contract: ident ~ expr
            contract = None
            if self.match(TT.TILDE):
                # Parse contract at comparison level to avoid consuming walrus/comma
                contract = self.parse_compare_expr()

            if contract is not None:
                return Tree('pattern', [self._tok('IDENT', tok.value), Tree('contract', [contract])])
            else:
                return Tree('pattern', [self._tok('IDENT', tok.value)])

        if self.match(TT.LPAR):
            items = [self.parse_pattern()]
            if not self.match(TT.COMMA):
                raise ParseError("Pattern list requires comma", self.current)
            items.append(self.parse_pattern())
            while self.match(TT.COMMA):
                items.append(self.parse_pattern())
            self.expect(TT.RPAR)
            return Tree('pattern', [Tree('pattern_list', items)])

        raise ParseError("Expected pattern", self.current)

    def parse_binderpattern(self) -> Tree:
        if self.match(TT.CARET):
            ident = self.expect(TT.IDENT)
            return Tree('hoist', [self._tok('IDENT', ident.value)])
        return Tree('binderpattern', [self.parse_pattern()])

    def parse_binderlist(self) -> Tree:
        items = [self.parse_binderpattern()]
        while self.match(TT.COMMA):
            items.append(self.parse_binderpattern())
        return Tree('binderlist', items)

    def parse_overspec(self) -> Tree:
        """Parse comprehension spec: iterable expression and optional binders"""
        children: List[Any] = []

        # Reset context: comprehension iterable allows CCC
        saved_context = self.parse_context
        self.parse_context = ParseContext.NORMAL
        try:
            if self.match(TT.LSQB):
                # for[x, y] iterable
                binder_list = self.parse_binderlist()
                self.expect(TT.RSQB)
                iter_expr = self.parse_expr()
                children.append(binder_list)
                children.append(iter_expr)
                return Tree('overspec', children)

            # Check for common pattern: IDENT in expr (e.g., for x in data)
            if self.check(TT.IDENT) and self.peek(1).type == TT.IN:
                pattern = self.parse_pattern()
                self.expect(TT.IN)
                iter_expr = self.parse_expr()
                children.append(iter_expr)
                children.append(pattern)
                return Tree('overspec', children)

            # Otherwise: expr [bind pattern]
            iter_expr = self.parse_expr()
            children.append(iter_expr)
            if self.check(TT.IDENT) and getattr(self.current, "value", "") == "bind":
                self.advance()
                pattern = self.parse_pattern()
                children.append(pattern)
            return Tree('overspec', children)
        finally:
            self.parse_context = saved_context

    def parse_comphead(self) -> Tree:
        if self.check(TT.FOR):
            self.advance()
            spec = self.parse_overspec()
            return Tree('comphead', [self._tok('FOR', 'for'), spec])
        if self.check(TT.OVER):
            self.advance()
            spec = self.parse_overspec()
            return Tree('comphead', [self._tok('OVER', 'over'), spec])
        raise ParseError("Expected comprehension head", self.current)

    def parse_ifclause_opt(self) -> Optional[Tree]:
        """Parse optional if-clause in comprehensions"""
        if not self.check(TT.IF):
            return None
        self.advance()

        # Reset context: if-clause condition allows CCC
        saved_context = self.parse_context
        self.parse_context = ParseContext.NORMAL
        try:
            cond = self.parse_expr()
            return Tree('ifclause', [self._tok('IF', 'if'), cond])
        finally:
            self.parse_context = saved_context

    def parse_arg_list(self) -> Optional[List[Tree]]:
        """
        Parse function call arguments
        Returns list of argitems (unwrapped) or None if empty
        """
        if self.check(TT.RPAR):
            return None  # Empty arg list

        # Set context: commas in function args are separators, not CCC
        saved_context = self.parse_context
        self.parse_context = ParseContext.FUNCTION_ARGS
        try:
            argitems = []

            while not self.check(TT.RPAR, TT.EOF):
                # Named argument: name: value
                if self.check(TT.IDENT) and self.peek(1).type == TT.COLON:
                    name = self.current
                    self.advance(2)  # consume IDENT and ':'
                    value = self.parse_expr()
                    arg_tree = Tree('namedarg', [self._tok('IDENT', name.value), value])
                else:
                    expr = self.parse_expr()
                    arg_tree = Tree('arg', [expr])

                argitems.append(Tree('argitem', [arg_tree]))

                if not self.match(TT.COMMA):
                    break

            # Return argitems directly (let caller wrap if needed)
            return argitems
        finally:
            self.parse_context = saved_context

    def parse_await_arm_list(self, kind_tok: Tok) -> List[Tree]:
        if getattr(kind_tok, "type", None) not in {TT.ANY, TT.ALL}:
            raise ParseError("await expects [any] or [all]", kind_tok)

        arms: List[Tree] = []

        while not self.check(TT.RPAR, TT.EOF):
            label_tok = None
            expr = None
            body = None

            # Optional label:
            if self.check(TT.IDENT) and self.peek(1).type == TT.COLON:
                label_tok = self.advance()
                self.expect(TT.COLON)

            expr = self.parse_expr()

            if self.match(TT.COLON):
                body = self.parse_body()

            arm_children: List[Any] = []
            if label_tok is not None:
                arm_children.append(self._tok('IDENT', label_tok.value))
            arm_children.append(expr)
            if body is not None:
                arm_children.append(body)
            arms.append(Tree('anyarm' if kind_tok.type == TT.ANY else 'allarm', arm_children))

            if not self.match(TT.COMMA):
                break

        return arms

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
                selectors.append(Tree('selector', [slice_tree]))
            else:
                expr = self.parse_expr()
                selectors.append(Tree('selector', [Tree('indexsel', [expr])]))

            # Check if next is ', default:' sequence - if so, stop before consuming comma
            if self.check(TT.COMMA):
                # Peek ahead to see if it's followed by 'default' ':'
                if self.peek(1).type == TT.IDENT and self.peek(1).value == 'default' and self.peek(2).type == TT.COLON:
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
                selectors.append(Tree('selitem', [slice_tree]))
            else:
                atom = self.parse_selector_atom()
                # Wrap atom in selatom, then in indexitem, then in selitem
                selatom = Tree('selatom', [atom])
                indexitem = Tree('indexitem', [selatom])
                selectors.append(Tree('selitem', [indexitem]))

            if not self.match(TT.COMMA):
                break

        return selectors

    def parse_selector_atom(self) -> Tree | Tok:
        """Parse selector atom: {expr} (interp), IDENT, or NUMBER"""
        # Interpolation: {expr}
        if self.match(TT.LBRACE):
            expr = self.parse_expr()
            self.expect(TT.RBRACE)
            return Tree('interp', [expr])

        # IDENT - convert to AST token with string type
        if self.check(TT.IDENT):
            tok = self.advance()
            return self._tok('IDENT', tok.value)

        # NUMBER - convert to AST token with string type
        if self.check(TT.NUMBER):
            tok = self.advance()
            return self._tok('NUMBER', tok.value)

        raise ParseError("Expected selector atom (number, identifier, or {expr})", self.current)

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

        # Check if colon follows
        return self._lookahead_check(idx, TT.COLON)

    def parse_slice_selector_literal(self) -> Tree:
        """Parse slice in selector literal: {start}:{stop} or 1:10:2"""
        start = None
        if not self.check(TT.COLON):
            atom = self.parse_selector_atom()
            start = Tree('selatom', [atom])

        self.expect(TT.COLON)

        # Check for < prefix for open-ended slice
        is_exclusive = self.match(TT.LT)

        stop = None
        if not self.check(TT.COLON, TT.COMMA, TT.BACKQUOTE):
            atom = self.parse_selector_atom()
            stop = Tree('selatom', [atom])

        step = None
        if self.match(TT.COLON):
            if not self.check(TT.COMMA, TT.BACKQUOTE):
                atom = self.parse_selector_atom()
                step = Tree('selatom', [atom])

        # Build sliceitem tree according to grammar:
        # sliceitem: selatom? ":" seloptstop (":" selatom)?
        children = []
        if start:
            children.append(start)

        # seloptstop: "<" selatom | selatom?
        if is_exclusive and stop:
            children.append(Tree('seloptstop', [self._tok('LT', '<'), stop]))
        elif stop:
            children.append(Tree('seloptstop', [stop]))
        else:
            children.append(Tree('seloptstop', []))

        if step:
            children.append(step)

        return Tree('sliceitem', children)

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
                while idx < len(self.tokens) and not self._lookahead_check(idx, TT.BACKQUOTE):
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
            children.append(Tree('slicearm_expr', [start]))
        else:
            children.append(Tree('slicearm_empty', []))

        if stop:
            children.append(Tree('slicearm_expr', [stop]))
        else:
            children.append(Tree('slicearm_empty', []))

        if step:
            children.append(Tree('slicearm_expr', [step]))

        return Tree('slicesel', children)

    def parse_object_item(self) -> Tree:
        """Parse object item: key: value or method(params): body"""
        if self.match(TT.SPREAD):
            expr = self.parse_expr()
            return Tree('obj_spread', [expr])

        # Field, method, getter, setter
        # Grammar allows: (IDENT | OVER) for field names
        if self.check(TT.IDENT, TT.OVER):
            name = self.advance()

            # Method: name(params): body
            if self.match(TT.LPAR):
                params = self.parse_param_list()
                self.expect(TT.RPAR)
                self.expect(TT.COLON)
                # Check if inline expression or block body
                if self.check(TT.NEWLINE):
                    body = self.parse_object_body()
                else:
                    # Inline expression
                    expr = self.parse_expr()
                    body = Tree('inlinebody', [expr])
                return Tree('obj_method', [self._tok('IDENT', name.value), params, body])

            # Field: name: value or name?: value (optional)
            is_optional = self.match(TT.QMARK)
            self.expect(TT.COLON)
            value = self.parse_expr()
            key = Tree('key_ident', [self._tok('IDENT', name.value)])
            if is_optional:
                return Tree('obj_field_optional', [key, value])
            return Tree('obj_field', [key, value])

        # String key
        if self.check(TT.STRING, TT.RAW_STRING, TT.RAW_HASH_STRING):
            key_tok = self.advance()
            self.expect(TT.COLON)
            value = self.parse_expr()
            key = Tree('key_string', [self._tok(key_tok.type.name, key_tok.value)])
            return Tree('obj_field', [key, value])

        # Expression key: (expr): value
        if self.check(TT.LPAR):
            self.advance()  # consume LPAR
            key_expr = self.parse_expr()
            self.expect(TT.RPAR)
            self.expect(TT.COLON)
            value = self.parse_expr()
            key = Tree('key_expr', [key_expr])
            return Tree('obj_field', [key, value])

        # Getter/setter
        if self.match(TT.GET):
            name = self.expect(TT.IDENT)
            # Optional empty parens
            if self.match(TT.LPAR):
                self.expect(TT.RPAR)
            self.expect(TT.COLON)
            # Check if inline expression or block body
            if self.check(TT.NEWLINE):
                body = self.parse_object_body()
            else:
                # Inline expression
                expr = self.parse_expr()
                body = Tree('inlinebody', [expr])
            return Tree('obj_get', [self._tok('IDENT', name.value), body])

        if self.match(TT.SET):
            name = self.expect(TT.IDENT)
            self.expect(TT.LPAR)
            param = self.expect(TT.IDENT)
            self.expect(TT.RPAR)
            self.expect(TT.COLON)
            # Check if inline expression or block body
            if self.check(TT.NEWLINE):
                body = self.parse_object_body()
            else:
                # Inline expression
                expr = self.parse_expr()
                body = Tree('inlinebody', [expr])
            return Tree('obj_set', [self._tok('IDENT', name.value), self._tok('IDENT', param.value), body])

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
            stmts.append(Tree('stmt', [stmt]))

            if self.match(TT.SEMI):
                continue

            while self.match(TT.NEWLINE):
                pass

            # If next token starts a new object item, stop collecting
            if self._looks_like_object_item_start():
                break

        return Tree('indentblock', stmts)

    def parse_pattern_list(self) -> Tree:
        """Parse destructuring pattern list: a, b, c"""
        patterns = [self.parse_pattern()]

        if not self.match(TT.COMMA):
            raise ParseError("Pattern list requires at least two patterns", self.current)

        patterns.append(self.parse_pattern())

        while self.match(TT.COMMA):
            patterns.append(self.parse_pattern())

        return Tree('pattern_list', patterns)

    def parse_destructure_rhs(self) -> Tree:
        """Parse destructuring RHS: expr or expr, expr, ..."""
        # Set context: commas in destructure pack are separators, not CCC
        saved_context = self.parse_context
        self.parse_context = ParseContext.DESTRUCTURE_PACK
        try:
            first_expr = self.parse_expr()

            # Check if there are more expressions (pack)
            if self.check(TT.COMMA):
                exprs = [first_expr]
                while self.match(TT.COMMA):
                    exprs.append(self.parse_expr())
                return Tree('pack', exprs)

            # Single expression
            return first_expr
        finally:
            self.parse_context = saved_context

    def parse_destructure_pattern(self) -> Tree:
        """Parse destructuring pattern for assignments: ident [~ contract] (, ident [~ contract])*"""
        items: list[Node] = []
        items.append(self._parse_pattern_item())

        while self.match(TT.COMMA):
            items.append(self._parse_pattern_item())

        return Tree('patternlist', items)

    def _parse_pattern_item(self) -> Tree:
        """Parse a single pattern item: IDENT [~ contract]"""
        ident = self.expect(TT.IDENT)
        contract = None
        if self.match(TT.TILDE):
            # Parse contract at comparison level to avoid consuming walrus/comma
            contract = self.parse_compare_expr()

        if contract is not None:
            return Tree('pattern', [self._tok('IDENT', ident.value), Tree('contract', [contract])])
        else:
            return Tree('pattern', [self._tok('IDENT', ident.value)])

    def _is_destructure_with_contracts(self) -> bool:
        """
        Lookahead to detect destructure patterns with contracts.
        Returns True if current position looks like: ident [~ contract] (, ident [~ contract])* (:=|=)
        Must have at least one comma to distinguish from simple assignment.
        Uses stateless lookahead helper - does not mutate parser state.
        """
        idx = self.pos
        paren_depth = 0
        start_idx = idx

        # Must start with IDENT
        if not self._lookahead_check(idx, TT.IDENT):
            return False
        _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)

        # Optional contract: ~ <anything until comma/assign>
        if self._lookahead_check(idx, TT.TILDE):
            _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)

            # Skip contract expression by counting bracket depth
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
                        return False  # Unbalanced
                elif tok.type in {TT.NEWLINE, TT.SEMI, TT.EOF}:
                    return False  # Hit statement boundary
                _, idx, paren_depth = self._lookahead_advance(idx, paren_depth)

        # Only match if we actually saw a contract (consumed more than just the ident)
        has_contract = start_idx + 1 < idx

        if has_contract:
            # Contract present: match single or multi-pattern with contract
            return self._lookahead_check(idx, TT.COMMA, TT.WALRUS, TT.ASSIGN)

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
                    if self.match(TT.LPAR):
                        # Call
                        args = self.parse_arg_list()
                        self.expect(TT.RPAR)
                        if args:
                            args = [Tree('arglistnamedmixed', args)]
                        else:
                            args = []
                        postfix_ops.append(Tree('call', args))
                    elif self.match(TT.DOT):
                        # Field access
                        if self.check(TT.IDENT):
                            field = self.advance()
                            postfix_ops.append(Tree('field', [self._tok('IDENT', field.value, field.line, field.column)]))
                        else:
                            raise ParseError("Expected field name after '.'", self.current)
                    elif self.check(TT.LSQB):
                        # Index
                        self.advance()
                        selectors = self.parse_selector_list()
                        self.expect(TT.RSQB)
                        postfix_ops.append(Tree('index', [
                            self._tok('LSQB', '['),
                            Tree('selectorlist', selectors),
                            self._tok('RSQB', ']')
                        ]))
                    else:
                        break

                # Build identchain if we have postfix ops
                if postfix_ops:
                    items.append(Tree('identchain', [self._tok('IDENT', name.value)] + postfix_ops))
                else:
                    items.append(self._tok('IDENT', name.value))
            else:
                raise ParseError("Expected identifier in fan list", self.current)

            if not self.match(TT.COMMA):
                break

        return items

    def parse_anonymous_fn_decl(self) -> Tree:
        """
        Anonymous function literal: fn(params): body
        Auto-invoked form: fn(()): body
        """
        self.expect(TT.LPAR)
        auto_invoke = False
        params: Tree = Tree('paramlist', [])

        if self.match(TT.LPAR):
            # fn ( ( ) ) : body  -> auto invoke
            self.expect(TT.RPAR)
            auto_invoke = True
        else:
            params = self.parse_param_list() if not self.check(TT.RPAR) else Tree('paramlist', [])
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
                body = Tree('inlinebody', [Tree('stmt', [expr_body])])
            else:
                body = self.parse_body()
        else:
            body = self.parse_body()

        paramlist_node = None if auto_invoke else params
        anon_children: List[Any] = []
        if paramlist_node is not None:
            anon_children.append(paramlist_node)
        anon_children.append(body)

        if return_contract is not None:
            anon_children.append(Tree('return_contract', [return_contract]))

        anon = Tree('anonfn', anon_children)

        if auto_invoke:
            return Tree('explicit_chain', [anon, Tree('call', [])])
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
        assert self.check(TT.LSQB, TT.LPAR), \
            f"parse_anonymous_fn expects '[' or '(', got {self.current.type.name}. Caller must consume '&' first."

        # Check for explicit params: &[params]
        if self.match(TT.LSQB):
            # &[a, b](a + b)
            params = []
            while not self.check(TT.RSQB, TT.EOF):
                if not self.check(TT.IDENT):
                    raise ParseError("Expected parameter name in lambda", self.current)
                params.append(self._tok('IDENT', self.advance().value))
                if not self.match(TT.COMMA):
                    break
            self.expect(TT.RSQB)

            # Now expect (expr)
            self.expect(TT.LPAR)
            body = self.parse_expr()
            self.expect(TT.RPAR)

            # Return Tree('amp_lambda', [Tree('paramlist', params), body])
            return Tree('amp_lambda', [Tree('paramlist', params), body])

        # Implicit subject: &(expr)
        self.expect(TT.LPAR)
        body = self.parse_expr()
        self.expect(TT.RPAR)

        # Return Tree('amp_lambda', [body])
        return Tree('amp_lambda', [body])

# ============================================================================
# Usage Example
# ============================================================================

def parse_source(source: str, use_indenter: bool = True) -> Tree:
    """
    Parse Shakar source code to AST.

    Returns Lark-compatible Tree structure for evaluator.

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

if __name__ == '__main__':
    import sys

    # Parse arguments
    use_indenter = '--no-indent' not in sys.argv
    args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]

    # Read source from file or stdin
    if len(args) > 0 and args[0] != '-':
        with open(args[0], 'r') as f:
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
