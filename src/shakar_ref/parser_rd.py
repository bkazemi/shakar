"""
Recursive Descent Parser for Shakar

This serves as:
1. A reference implementation for the C parser
2. A faster alternative to Lark's Earley parser
3. Documentation of parsing strategy and disambiguation

Structure:
- Lexer: Token stream from source
- Parser: Recursive descent with Pratt parsing for expressions
- AST: Same tree structure as Lark output for compatibility
"""

from typing import Optional, List, Any, Callable
from lark import Tree, Token

from .token_types import TT, Tok

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

    # ========================================================================
    # Token Navigation
    # ========================================================================

    def peek(self, offset: int = 0) -> Tok:
        """Look ahead at token"""
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return Tok(TT.EOF, None, 0, 0)

    def advance(self) -> Tok:
        """Consume current token and move to next"""
        prev = self.current
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current = self.tokens[self.pos]
        else:
            self.current = Tok(TT.EOF, None, 0, 0)
        return prev

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

    # ========================================================================
    # Top-Level Parsing
    # ========================================================================

    def parse(self) -> Tree:
        """Parse entire program"""
        stmtlists = []

        # Skip leading newlines
        while self.match(TT.NEWLINE):
            pass

        # Collect statements into stmtlist
        stmts = []
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
                stmts.append(Token('SEMI', ';'))
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
        return Tree(start_node, stmtlists)

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
        expr = self.parse_expr()

        # Check for guard chain syntax
        if self.check(TT.COLON):
            # Could be guard chain or just expression statement
            # Try to parse as guard chain
            self.pos = expr_start  # Backtrack
            return self.parse_guard_chain()

        # Check for postfix if/unless
        if self.check(TT.IF):
            if_tok = self.advance()
            cond = self.parse_expr()
            # Wrap base statement in Tree('expr', [...])
            base_stmt = Tree('expr', [expr])
            return Tree('postfixif', [base_stmt, Token('IF', 'if'), Tree('expr', [cond])])

        # Check for assignment operators
        if self.check(TT.ASSIGN, TT.WALRUS, TT.APPLYASSIGN,
                      TT.PLUSEQ, TT.MINUSEQ, TT.STAREQ, TT.SLASHEQ, TT.FLOORDIVEQ, TT.MODEQ):
            # This is actually handled in parse_expr for walrus/apply
            # But = assignments need special handling
            if self.check(TT.ASSIGN):
                lvalue = self._expr_to_lvalue(expr)

                eq_tok = self.advance()
                rhs = self.parse_nullish_expr()  # Parse from nullish down (like walrus)
                rhs_nc = Tree('expr_nc', [rhs])
                return Tree('assignstmt', [lvalue, Token('EQUAL', '='), rhs_nc])

            # Compound assignments
            if self.check(TT.PLUSEQ, TT.MINUSEQ, TT.STAREQ, TT.SLASHEQ, TT.FLOORDIVEQ, TT.MODEQ):
                lvalue = self._expr_to_lvalue(expr)

                op = self.advance()
                rhs = self.parse_nullish_expr()
                rhs_nc = Tree('expr_nc', [rhs])
                return Tree('compound_assign', [lvalue, Token(op.type.name, op.value), rhs_nc])

        # Just an expression statement
        return expr

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
            elifs.append(Tree('elifclause', [Token('ELIF', 'elif'), elif_cond, elif_body]))

        if self.check(TT.ELSE):
            self.advance()
            self.expect(TT.COLON)
            else_body = self.parse_body()
            else_clause = Tree('elseclause', [Token('ELSE', 'else'), else_body])

        children = [Token('IF', 'if'), cond, then_body] + elifs
        if else_clause:
            children.append(else_clause)

        return Tree('ifstmt', children)

    def parse_while_stmt(self) -> Tree:
        """Parse while loop: while expr: body"""
        while_tok = self.expect(TT.WHILE)
        cond = self.parse_expr()
        self.expect(TT.COLON)
        body = self.parse_body()
        return Tree('whilestmt', [Token('WHILE', 'while'), cond, body])

    def parse_for_stmt(self) -> Tree:
        """
        Parse for loop:
        for x in expr: body (forin)
        for[i, x] expr: body (forindexed)
        for expr: body (forsubject - subjectful loop)
        """
        for_tok = self.expect(TT.FOR)

        # Check for indexed syntax: for[...]
        if self.match(TT.LSQB):
            # Parse first identifier
            ident1 = self.expect(TT.IDENT)
            pattern1 = Tree('pattern', [Token('IDENT', ident1.value)])
            binder1 = Tree('binderpattern', [pattern1])

            # Check for comma - indicates formap2 (two patterns)
            if self.match(TT.COMMA):
                ident2 = self.expect(TT.IDENT)
                pattern2 = Tree('pattern', [Token('IDENT', ident2.value)])
                binder2 = Tree('binderpattern', [pattern2])
                self.expect(TT.RSQB)
                # for[pattern, pattern] expr: body
                iterable = self.parse_expr()
                self.expect(TT.COLON)
                body = self.parse_body()
                return Tree('formap2', [Token('FOR', 'for'), binder1, binder2, iterable, body])
            else:
                self.expect(TT.RSQB)
                # for[pattern] expr: body
                iterable = self.parse_expr()
                self.expect(TT.COLON)
                body = self.parse_body()
                return Tree('forindexed', [Token('FOR', 'for'), binder1, iterable, body])

        # Check if it's "for x in expr" or just "for expr"
        # Lookahead to see if there's an IN keyword after an identifier
        if self.check(TT.IDENT):
            # Peek ahead to see if next token is IN
            next_tok = self.peek(1)
            if next_tok.type == TT.IN:
                # for x in expr: body
                var = self.advance()
                self.expect(TT.IN)
                iterable = self.parse_expr()
                self.expect(TT.COLON)
                body = self.parse_body()
                # Wrap variable in pattern
                pattern = Tree('pattern', [Token('IDENT', var.value)])
                return Tree('forin', [Token('FOR', 'for'), pattern, Token('IN', 'in'), iterable, body])
            # Otherwise it's forsubject with identifier as expression

        # Subjectful for: for expr: body
        iterable = self.parse_expr()
        self.expect(TT.COLON)
        body = self.parse_body()
        return Tree('forsubject', [Token('FOR', 'for'), iterable, body])

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
            children.append(Tree('using_handle', [Token('IDENT', handle.value)]))
        children.append(resource)
        if binder is not None:
            children.append(Tree('using_bind', [Token('IDENT', binder.value)]))
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

        defer_children: List[Tree | Token] = []

        if colon_ahead:
            # Optional label
            if self.check(TT.IDENT):
                label_tok = self.advance()
                defer_children.append(Tree('deferlabel', [Token('IDENT', label_tok.value)]))

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
        deps: List[Token] = []

        if self.match(TT.LPAR):
            if not self.check(TT.RPAR):
                deps.append(Token('IDENT', self.expect(TT.IDENT).value))
                while self.match(TT.COMMA):
                    deps.append(Token('IDENT', self.expect(TT.IDENT).value))
            self.expect(TT.RPAR)
        else:
            deps.append(Token('IDENT', self.expect(TT.IDENT).value))

        return Tree('deferafter', deps)

    def _expr_has_call(self, expr: Tree | Token) -> bool:
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
        return False

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

        return Tree('decorator_def', [Token('IDENT', name.value), params, body])

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

        self.expect(TT.COLON)
        body = self.parse_body()

        # fnstmt structure: [name, params, body] or [name, params, body, decorator_list]
        if decorators:
            decorator_list = Tree('decorator_list', decorators)
            return Tree('fnstmt', [Token('IDENT', name.value), params, body, decorator_list])

        return Tree('fnstmt', [Token('IDENT', name.value), params, body])

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
                children.append(Token('RPAR', ')'))
                children.append(trailing_body)
            return Tree(root_label, children)

        # await expr [: body]?
        expr = self.parse_expr()
        if self.match(TT.COLON):
            body = self.parse_body()
            return Tree('awaitstmt', [Token('AWAIT', await_tok.value), expr, body])
        return Tree('await_value', [Token('AWAIT', await_tok.value), expr])

    def parse_return_stmt(self) -> Tree:
        """Parse return statement: return [expr]"""
        ret_tok = self.expect(TT.RETURN)

        # Check if there's a value
        if self.check(TT.NEWLINE, TT.EOF, TT.SEMI, TT.RBRACE):
            return Tree('returnstmt', [Token('RETURN', 'return')])

        value = self.parse_expr()
        return Tree('returnstmt', [Token('RETURN', 'return'), value])

    def parse_break_stmt(self) -> Tree:
        """Parse break statement"""
        self.expect(TT.BREAK)
        return Tree('breakstmt', [Token('BREAK', 'break')])

    def parse_continue_stmt(self) -> Tree:
        """Parse continue statement"""
        self.expect(TT.CONTINUE)
        return Tree('continuestmt', [Token('CONTINUE', 'continue')])

    def parse_throw_stmt(self) -> Tree:
        """Parse throw statement: throw expr"""
        self.expect(TT.THROW)
        value = self.parse_expr()
        return Tree('throwstmt', [value])

    def parse_assert_stmt(self) -> Tree:
        """Parse assert: assert expr"""
        self.expect(TT.ASSERT)
        value = self.parse_expr()
        return Tree('assert', [value])

    def parse_dbg_stmt(self) -> Tree:
        """Parse dbg: dbg expr (stub - treated as expression statement)"""
        self.expect(TT.DBG)
        value = self.parse_expr()
        # Just return the expression - dbg is a no-op stub for now
        return value

    def parse_guard_chain(self) -> Tree:
        """
        Parse guard chain (inline if-else):
        expr : body | expr : body |: else
        """
        branches = []

        # Parse first branch
        cond = self.parse_expr()
        self.expect(TT.COLON)
        body = self.parse_body()
        branches.append(Tree('guardbranch', [cond, body]))

        # Parse additional branches
        while self.match(TT.OR):  # | for guard chain
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
            stmts = []
            while not self.check(TT.RBRACE, TT.EOF):
                if self.match(TT.NEWLINE):
                    continue
                if self.match(TT.SEMI):
                    stmts.append(Token('SEMI', ';'))
                    continue
                stmts.append(self.parse_statement())
            self.expect(TT.RBRACE)
            return Tree('indentblock', stmts)

        if self.check(TT.INDENT):
            # Indented block
            indent_tok = self.advance()
            # Use actual indent value from lexer
            children = [Token('INDENT', indent_tok.value if indent_tok.value is not None else '    ')]

            # Parse statements in the block
            while not self.check(TT.DEDENT, TT.EOF):
                if self.match(TT.NEWLINE):
                    continue
                if self.match(TT.SEMI):
                    children.append(Token('SEMI', ';'))
                    continue
                stmt = self.parse_statement()
                children.append(Tree('stmt', [stmt]))
                # Skip newlines between statements
                while self.check(TT.NEWLINE) and not self.check(TT.DEDENT):
                    self.match(TT.NEWLINE)

            dedent_tok = self.expect(TT.DEDENT)
            children.append(Token('DEDENT', dedent_tok.value if dedent_tok.value is not None else ''))
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

    def parse_catch_expr(self) -> Tree:
        """Parse catch expression: expr catch [types] [bind x]: handler"""
        expr = self.parse_ternary_expr()

        if self.check(TT.CATCH) or (self.check(TT.AT) and self.peek(1).type == TT.AT):
            # catch or @@ syntax
            if self.check(TT.CATCH):
                self.advance()
            else:
                self.advance()  # @
                self.advance()  # @

            # Optional type filter: (Type1, Type2)
            types = None
            if self.match(TT.LPAR):
                types = []
                types.append(self.expect(TT.IDENT))
                while self.match(TT.COMMA):
                    types.append(self.expect(TT.IDENT))
                self.expect(TT.RPAR)

            # Optional binder: bind x
            binder = None
            if self.match(TT.IDENT):
                binder = self.current

            self.expect(TT.COLON)
            handler = self.parse_expr()

            # Build catch node
            children = [expr]
            if types:
                children.append(Tree('catchtypes', types))
            if binder:
                children.append(Token('IDENT', binder.value))
            children.append(handler)

            return Tree('catchexpr', children)

        return expr

    def parse_ternary_expr(self) -> Tree:
        """Parse ternary: expr ? then : else"""
        expr = self.parse_or_expr()

        if self.match(TT.QMARK):
            then_expr = self.parse_expr()
            self.expect(TT.COLON)
            else_expr = self.parse_ternary_expr()  # Right associative
            return Tree('ternaryexpr', [expr, then_expr, else_expr])

        # No ternary, wrap and return
        return Tree('ternaryexpr', [expr])

    def parse_or_expr(self) -> Tree:
        """Parse logical OR: expr || expr"""
        left = self.parse_and_expr()

        if not self.check(TT.OR):
            # No OR, just return and level
            return left

        # Collect operands and operators (interleaved)
        children = [left]
        while self.check(TT.OR):
            op = self.advance()
            children.append(Token(op.type.name, op.value))
            right = self.parse_and_expr()
            children.append(right)

        return Tree('or', children)

    def parse_and_expr(self) -> Tree:
        """Parse logical AND: expr && expr"""
        left = self.parse_bind_expr()

        if not self.check(TT.AND):
            # No AND, just return bind level
            return left

        # Collect operands and operators (interleaved)
        children = [left]
        while self.check(TT.AND):
            op = self.advance()
            children.append(Token(op.type.name, op.value))
            right = self.parse_bind_expr()
            children.append(right)

        return Tree('and', children)

    def parse_bind_expr(self) -> Tree:
        """Parse apply bind: lvalue .= expr"""
        left = self.parse_walrus_expr()

        if self.match(TT.APPLYASSIGN):
            lvalue = self._expr_to_lvalue(left)

            right = self.parse_bind_expr()  # Right associative
            return Tree('bind', [lvalue, right])

        # No bind, just return walrus level
        return left

    def _expr_to_lvalue(self, expr: Tree | Token) -> Tree:
        """
        Convert an expression node into an lvalue tree, preserving chain
        structure and normalizing index ops to lv_index for assignment
        handling.
        """
        core_expr: Tree | Token = expr

        # Unwrap single-child precedence wrappers (expr, ternaryexpr, etc.)
        while isinstance(core_expr, Tree) and len(core_expr.children) == 1:
            core_expr = core_expr.children[0]

        if isinstance(core_expr, Token) and core_expr.type == 'IDENT':
            return Tree('lvalue', [core_expr])

        if isinstance(core_expr, Tree) and core_expr.data in {'explicit_chain', 'implicit_chain'}:
            if not core_expr.children:
                return Tree('lvalue', [core_expr])

            head, *ops = core_expr.children

            norm_ops: List[Tree | Token] = []
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

    def parse_walrus_expr(self) -> Tree:
        """Parse walrus: x := expr"""
        # Check for walrus pattern: IDENT :=
        if self.check(TT.IDENT) and self.peek(1).type == TT.WALRUS:
            name = self.advance()
            self.advance()  # :=
            # Parse the RHS - wrapping in expr_nc to match Lark
            value = self.parse_nullish_expr()  # Parse from nullish down
            value_nc = Tree('expr_nc', [value])
            return Tree('walrus', [Token('IDENT', name.value), value_nc])

        # No walrus, just return nullish level
        return self.parse_nullish_expr()

    def parse_nullish_expr(self) -> Tree:
        """Parse nullish coalescing: expr ?? expr"""
        left = self.parse_compare_expr()

        if not self.check(TT.NULLISH):
            # No nullish operator, just return compare level
            return left

        # Collect all operands and operators into flat list
        children = [left]
        while self.check(TT.NULLISH):
            op = self.advance()
            children.append(Token('NULLISH', op.value))
            right = self.parse_compare_expr()
            children.append(right)

        return Tree('nullish', children)

    def parse_compare_expr(self) -> Tree:
        """
        Parse comparison with chained comparison chains (CCC):
        x == 5, and < 10, or > 100
        """
        left = self.parse_add_expr()

        # Check for comparison operator
        if not self.is_compare_op():
            # No comparison, just wrap and return
            return Tree('compareexpr', [left])

        # Parse CCC
        op_tokens = self.parse_compare_op()
        right = self.parse_add_expr()

        # Wrap operator tokens in cmpop tree
        op_tree = Tree('cmpop', op_tokens)
        children = [left, op_tree, right]

        # Check for comma-chained comparisons
        if self.match(TT.COMMA):
            # CCC chain
            while True:
                # Check for and/or
                is_or = self.match(TT.OR)
                is_and = self.match(TT.AND)

                # Parse next leg
                leg_op_tokens = self.parse_compare_op() if self.is_compare_op() else None
                leg_value = self.parse_add_expr()

                if is_or:
                    if leg_op_tokens:
                        leg_op = Tree('cmpop', leg_op_tokens)
                        children.append(Tree('ccc_or_leg', [leg_op, leg_value]))
                    else:
                        children.append(Tree('ccc_or_leg', [leg_value]))
                else:
                    if leg_op_tokens:
                        leg_op = Tree('cmpop', leg_op_tokens)
                        children.append(Tree('ccc_and_leg', [leg_op, leg_value]))
                    else:
                        children.append(Tree('ccc_and_leg', [leg_value]))

                if not self.match(TT.COMMA):
                    break

            return Tree('compareexpr', children)

        return Tree('compareexpr', [left, op_tree, right])

    def is_compare_op(self) -> bool:
        """Check if current token is a comparison operator"""
        if self.check(TT.EQ, TT.NEQ, TT.LT, TT.LTE, TT.GT, TT.GTE, TT.IS, TT.IN):
            return True
        if self.check(TT.NOT) and self.peek(1).type == TT.IN:
            return True
        if self.check(TT.NEG) and self.peek(1).type in (TT.IS, TT.IN):
            return True
        return False

    def parse_compare_op(self) -> List[Token]:
        """Parse comparison operator - returns list of tokens for compound ops"""
        if self.check(TT.EQ, TT.NEQ, TT.LT, TT.LTE, TT.GT, TT.GTE):
            op = self.advance()
            return [Token(op.type.name, op.value)]

        if self.check(TT.IS):
            self.advance()
            if self.match(TT.NOT):
                return [Token('IS', 'is'), Token('NOT', 'not')]
            return [Token('IS', 'is')]

        if self.check(TT.IN):
            self.advance()
            return [Token('IN', 'in')]

        if self.match(TT.NOT):
            self.expect(TT.IN)
            return [Token('NOT', 'not'), Token('IN', 'in')]

        # Check for !is and !in
        if self.match(TT.NEG):  # ! token
            if self.match(TT.IS):
                return [Token('NEG', '!'), Token('IS', 'is')]
            elif self.match(TT.IN):
                return [Token('NEG', '!'), Token('IN', 'in')]
            else:
                raise ParseError("Expected 'is' or 'in' after '!'", self.current)

        raise ParseError("Expected comparison operator", self.current)

    def parse_add_expr(self) -> Tree:
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

        # Build addexpr with left-associative structure
        for op, right in ops_and_operands:
            # Wrap operator in addop tree
            op_tree = Tree('addop', [Token(op.type.name, op.value)])
            left = Tree('addexpr', [left, op_tree, right])

        return left

    def parse_mul_expr(self) -> Tree:
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
            op_tree = Tree('mulop', [Token(op.type.name, op.value)])
            left = Tree('mulexpr', [left, op_tree, right])

        return left

    def parse_pow_expr(self) -> Tree:
        """Parse exponentiation: expr ** expr (right associative)"""
        base = self.parse_unary_expr()

        if self.match(TT.POW):
            exp = self.parse_pow_expr()  # Right associative
            return Tree('powexpr', [base, exp])

        # Just wrap in powexpr even if no operator
        return base

    def parse_unary_expr(self) -> Tree:
        """Parse unary operators: -expr, !expr, ++expr, await expr"""
        # Await
        if self.check(TT.AWAIT):
            await_tok = self.advance()
            # await expr or await (expr)
            if self.match(TT.LPAR):
                expr = self.parse_expr()
                self.expect(TT.RPAR)
            else:
                expr = self.parse_unary_expr()
            return Tree('await_value', [Token('AWAIT', 'await'), expr])

        # $ (no anchor)
        if self.match(TT.DOLLAR):
            expr = self.parse_unary_expr()
            return Tree('unaryexpr', [Tree('no_anchor', [expr])])

        # Unary prefix operators
        if self.check(TT.MINUS, TT.NOT, TT.NEG, TT.INCR, TT.DECR):
            op = self.advance()
            expr = self.parse_unary_expr()
            op_tree = Tree('unaryprefixop', [Token(op.type.name, op.value)])
            return Tree('unaryexpr', [op_tree, expr])

        # No unary operator, wrap postfix in unaryexpr
        return Tree('unaryexpr', [self.parse_postfix_expr()])

    def parse_postfix_expr(self) -> Tree:
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
                return Tree('subject', [Token('DOT', dot.value)])

            # Implicit chain: .field, .(args), .[index], .over
            self.advance()  # consume .

            # Parse the first implicit operation
            imphead = None
            if self.check(TT.IDENT, TT.OVER):
                tok = self.advance()
                imphead = Tree('field', [Token(tok.type.name, tok.value)])
            elif self.match(TT.LPAR):
                args = self.parse_arg_list()
                self.expect(TT.RPAR)
                imphead = Tree('call', args if args else [])
            elif self.check(TT.LSQB):
                lsqb = self.advance()
                selectors = self.parse_selector_list()
                default = None
                if self.match(TT.COMMA):
                    self.expect(TT.IDENT)  # 'default' keyword
                    self.expect(TT.COLON)
                    default = self.parse_expr()
                rsqb = self.expect(TT.RSQB)
                children = [
                    Token('LSQB', '['),
                    Tree('selectorlist', selectors),
                    Token('RSQB', ']')
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
                        postfix_ops.append(Tree('field', [Token(field.type.name, field.value)]))
                    elif self.match(TT.LBRACE):
                        items = self.parse_fan_items()
                        self.expect(TT.RBRACE)
                        postfix_ops.append(Tree('valuefan', [Tree('valuefan_list', [Tree('valuefan_item', [Token('IDENT', t.value)]) for t in items])]))
                    else:
                        raise ParseError("Expected field name after '.'", self.current)
                elif self.check(TT.LSQB):
                    lsqb = self.advance()
                    selectors = self.parse_selector_list()
                    default = None
                    if self.match(TT.COMMA):
                        self.expect(TT.IDENT)
                        self.expect(TT.COLON)
                        default = self.parse_expr()
                    rsqb = self.expect(TT.RSQB)
                    children = [
                        Token('LSQB', '['),
                        Tree('selectorlist', selectors),
                        Token('RSQB', ']')
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
                    postfix_ops.append(Tree('field', [Token(field.type.name, field.value)]))
                elif self.match(TT.LBRACE):
                    # Fan syntax: .{field1, field2} or .{chain1, chain2}
                    items = self.parse_fan_items()
                    self.expect(TT.RBRACE)
                    postfix_ops.append(Tree('valuefan', [Tree('valuefan_list', [Tree('valuefan_item', [Token('IDENT', t.value)]) for t in items])]))
                else:
                    raise ParseError("Expected field name after '.'", self.current)

            # Indexing
            elif self.check(TT.LSQB):
                lsqb = self.advance()
                selectors = self.parse_selector_list()

                # Check for default value
                default = None
                if self.match(TT.COMMA):
                    self.expect(TT.IDENT)  # 'default' keyword
                    self.expect(TT.COLON)
                    default = self.parse_expr()

                rsqb = self.expect(TT.RSQB)

                # Build index tree: [ selectorlist ]
                children = [
                    Token('LSQB', '['),
                    Tree('selectorlist', selectors),
                    Token('RSQB', ']')
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

            else:
                break

        # Only wrap in explicit_chain if there are postfix operations
        if not postfix_ops:
            return primary

        return Tree('explicit_chain', [primary] + postfix_ops)

    def parse_primary_expr(self) -> Tree:
        """
        Parse primary expressions:
        - Literals (numbers, strings, true, false, nil)
        - Identifiers
        - Parenthesized expressions
        - Arrays, objects
        - Comprehensions
        - etc.
        """

        # Literals - just return tokens, Prune() will wrap if needed
        if self.check(TT.NUMBER):
            tok = self.advance()
            return Token('NUMBER', tok.value)

        # Hole placeholder for partial application
        if self.match(TT.QMARK):
            return Tree('holeexpr', [])

        if self.check(TT.STRING, TT.RAW_STRING, TT.RAW_HASH_STRING, TT.SHELL_STRING):
            tok = self.advance()
            return Token(tok.type.name, tok.value)

        if self.match(TT.TRUE):
            return Token('TRUE', 'true')
        if self.match(TT.FALSE):
            return Token('FALSE', 'false')
        if self.match(TT.NIL):
            return Token('NIL', 'nil')

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
            return Token('IDENT', tok.value)

        # Special identifiers
        if self.match(TT.OVER):
            return Token('OVER', '_')
        if self.match(TT.ANY):
            return Token('ANY', 'any')
        if self.match(TT.ALL):
            return Token('ALL', 'all')

        # Parenthesized expression
        if self.match(TT.LPAR):
            expr = self.parse_expr()
            self.expect(TT.RPAR)
            return Tree('group_expr', [expr])

        # Array literal
        if self.match(TT.LSQB):
            # Skip newlines in array
            while self.match(TT.NEWLINE):
                pass

            if self.check(TT.RSQB):
                self.advance()
                return Tree('array', [])

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

            # Skip newlines
            while self.match(TT.NEWLINE):
                pass
            while True:
                if not self.match(TT.COMMA):
                    break
                while self.match(TT.NEWLINE):
                    pass
                if self.check(TT.RSQB, TT.EOF):
                    break
                elements.append(self.parse_expr())
                while self.match(TT.NEWLINE):
                    pass

            self.expect(TT.RSQB)
            return Tree('array', elements)

        # Object literal
        if self.match(TT.LBRACE):
            # Try dict comprehension detection
            saved_pos = self.pos
            saved_current = self.current

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

        # Selector literal: `0:10`
        if self.match(TT.BACKQUOTE):
            selectors = self.parse_selector_list()
            self.expect(TT.BACKQUOTE)
            return Tree('selectorliteral', selectors)

        raise ParseError(f"Unexpected token in expression: {self.current.type.name}", self.current)

    # ========================================================================
    # Helper Parsers
    # ========================================================================

    def parse_param_list(self) -> Tree:
        """Parse function parameter list"""
        params = []

        while not self.check(TT.RPAR, TT.EOF):
            param = self.expect(TT.IDENT)

            # Optional default value
            default = None
            if self.match(TT.ASSIGN):
                default = self.parse_expr()

            if default:
                params.append(Tree('param', [Token('IDENT', param.value), default]))
            else:
                params.append(Token('IDENT', param.value))

            if not self.match(TT.COMMA):
                break

        return Tree('paramlist', params)

    def parse_pattern(self) -> Tree:
        if self.check(TT.IDENT):
            tok = self.advance()
            return Tree('pattern', [Token('IDENT', tok.value)])

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
            return Tree('hoist', [Token('IDENT', ident.value)])
        return Tree('binderpattern', [self.parse_pattern()])

    def parse_binderlist(self) -> Tree:
        items = [self.parse_binderpattern()]
        while self.match(TT.COMMA):
            items.append(self.parse_binderpattern())
        return Tree('binderlist', items)

    def parse_overspec(self) -> Tree:
        children: List[Any] = []

        if self.match(TT.LSQB):
            binder_list = self.parse_binderlist()
            self.expect(TT.RSQB)
            iter_expr = self.parse_expr()
            children.append(binder_list)
            children.append(iter_expr)
            return Tree('overspec', children)

        iter_expr = self.parse_expr()
        children.append(iter_expr)
        if self.check(TT.IDENT) and getattr(self.current, "value", "") == "bind":
            self.advance()
            pattern = self.parse_pattern()
            children.append(pattern)
        return Tree('overspec', children)

    def parse_comphead(self) -> Tree:
        if self.check(TT.FOR):
            self.advance()
            spec = self.parse_overspec()
            return Tree('comphead', [Token('FOR', 'for'), spec])
        if self.check(TT.OVER):
            self.advance()
            spec = self.parse_overspec()
            return Tree('comphead', [Token('OVER', 'over'), spec])
        raise ParseError("Expected comprehension head", self.current)

    def parse_ifclause_opt(self) -> Optional[Tree]:
        if not self.check(TT.IF):
            return None
        self.advance()
        cond = self.parse_expr()
        return Tree('ifclause', [Token('IF', 'if'), cond])

    def parse_arg_list(self) -> Optional[List[Tree]]:
        """
        Parse function call arguments
        Returns list of argitems (unwrapped) or None if empty
        """
        if self.check(TT.RPAR):
            return None  # Empty arg list

        argitems = []

        while not self.check(TT.RPAR, TT.EOF):
            # Named argument: name=value
            if self.check(TT.IDENT) and self.peek(1).type == TT.ASSIGN:
                name = self.advance()
                self.advance()  # =
                value = self.parse_expr()
                arg_tree = Tree('namedarg', [Token('IDENT', name.value), value])
            else:
                expr = self.parse_expr()
                arg_tree = Tree('arg', [expr])

            argitems.append(Tree('argitem', [arg_tree]))

            if not self.match(TT.COMMA):
                break

        # Return argitems directly (let caller wrap if needed)
        return argitems

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
                arm_children.append(Token('IDENT', label_tok.value))
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

            if not self.match(TT.COMMA):
                break

        return selectors

    def is_slice_selector(self) -> bool:
        """Check if next tokens form a slice selector"""
        # Look for : that's not part of ternary or other constructs
        # Simple heuristic: if we see : before any binary operator
        saved_pos = self.pos

        # Try to parse an expression, see if : follows
        depth = 0
        while self.pos < len(self.tokens):
            if self.check(TT.LPAR, TT.LSQB, TT.LBRACE):
                depth += 1
                self.advance()
            elif self.check(TT.RPAR, TT.RSQB, TT.RBRACE):
                if depth == 0:
                    break
                depth -= 1
                self.advance()
            elif depth == 0 and self.check(TT.COLON):
                # Found a colon - this is a slice
                self.pos = saved_pos
                self.current = self.tokens[saved_pos]
                return True
            elif depth == 0 and self.check(TT.COMMA):
                break
            else:
                self.advance()

        # Not a slice - restore position
        self.pos = saved_pos
        self.current = self.tokens[saved_pos]
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
        # Field, method, getter, setter
        if self.check(TT.IDENT):
            name = self.advance()

            # Method: name(params): body
            if self.match(TT.LPAR):
                params = self.parse_param_list()
                self.expect(TT.RPAR)
                self.expect(TT.COLON)
                body = self.parse_object_body()
                return Tree('obj_method', [Token('IDENT', name.value), params, body])

            # Field: name: value
            self.expect(TT.COLON)
            value = self.parse_expr()
            return Tree('obj_field_ident', [Token('IDENT', name.value), value])

        # String key
        if self.check(TT.STRING, TT.RAW_STRING, TT.RAW_HASH_STRING):
            key = self.advance()
            self.expect(TT.COLON)
            value = self.parse_expr()
            return Tree('obj_field_string', [Token(key.type.name, key.value), value])

        # Getter/setter
        if self.match(TT.GET):
            name = self.expect(TT.IDENT)
            self.expect(TT.COLON)
            body = self.parse_object_body()
            return Tree('obj_get', [Token('IDENT', name.value), body])

        if self.match(TT.SET):
            name = self.expect(TT.IDENT)
            self.expect(TT.LPAR)
            param = self.expect(TT.IDENT)
            self.expect(TT.RPAR)
            self.expect(TT.COLON)
            body = self.parse_object_body()
            return Tree('obj_set', [Token('IDENT', name.value), Token('IDENT', param.value), body])

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

    def parse_destructure_pattern(self) -> Tree:
        """Parse destructuring pattern for assignments"""
        # Simple for now: x, y, z
        items = []
        items.append(self.expect(TT.IDENT))

        while self.match(TT.COMMA):
            items.append(self.expect(TT.IDENT))

        return Tree('pattern', [Token('IDENT', t.value) for t in items])

    def parse_fan_items(self) -> List[Tree]:
        """Parse fan items: {field1, field2} or {chain1, chain2}"""
        items = []

        while not self.check(TT.RBRACE, TT.EOF):
            if self.check(TT.IDENT):
                name = self.advance()
                items.append(Token('IDENT', name.value))
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
                params.append(Token('IDENT', self.advance().value))
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

    # Tokenize
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
