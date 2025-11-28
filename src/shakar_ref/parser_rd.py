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

        # Declarations
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
                      TT.PLUSEQ, TT.MINUSEQ, TT.STAREQ, TT.SLASHEQ):
            # This is actually handled in parse_expr for walrus/apply
            # But = assignments need special handling
            if self.check(TT.ASSIGN):
                # Wrap expr in lvalue
                # Need to unwrap expression wrappers to find the core
                core_expr = expr
                while isinstance(core_expr, Tree) and len(core_expr.children) == 1:
                    # Skip single-child wrapper nodes (expr, ternaryexpr, compareexpr, etc.)
                    core_expr = core_expr.children[0]

                if isinstance(core_expr, Token) and core_expr.type == 'IDENT':
                    lvalue = Tree('lvalue', [core_expr])
                elif isinstance(core_expr, Tree) and core_expr.data == 'explicit_chain':
                    # Unwrap explicit_chain - put its children directly in lvalue
                    lvalue = Tree('lvalue', list(core_expr.children))
                else:
                    # Fallback: just wrap
                    lvalue = Tree('lvalue', [expr])

                eq_tok = self.advance()
                rhs = self.parse_nullish_expr()  # Parse from nullish down (like walrus)
                rhs_nc = Tree('expr_nc', [rhs])
                return Tree('assignstmt', [lvalue, Token('EQUAL', '='), rhs_nc])

            # Compound assignments
            if self.check(TT.PLUSEQ, TT.MINUSEQ, TT.STAREQ, TT.SLASHEQ):
                # Wrap expr in lvalue
                if isinstance(expr, Token) and expr.type == 'IDENT':
                    lvalue = Tree('lvalue', [expr])
                else:
                    lvalue = Tree('lvalue', [expr])

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

    def parse_fn_stmt(self) -> Tree:
        """
        Parse function declaration:
        fn name(params): body
        """
        self.expect(TT.FN)
        name = self.expect(TT.IDENT)

        self.expect(TT.LPAR)
        params = self.parse_param_list()
        self.expect(TT.RPAR)

        self.expect(TT.COLON)
        body = self.parse_body()

        return Tree('fnstmt', [Token('IDENT', name.value), params, body])

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
            # Left should be an lvalue (identifier or chain)
            # Wrap it in Tree('lvalue', [...])
            if isinstance(left, Token) and left.type == 'IDENT':
                lvalue = Tree('lvalue', [left])
            else:
                # For now, just wrap whatever we have
                # TODO: properly parse lvalue chains
                lvalue = Tree('lvalue', [left])

            right = self.parse_bind_expr()  # Right associative
            return Tree('bind', [lvalue, right])

        # No bind, just return walrus level
        return left

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
                        postfix_ops.append(Tree('fieldfan', [Tree('fieldlist', items)]))
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
                    # Wrap args based on context:
                    # - In inline body: call(argitem, argitem, ...)
                    # - Not inline: call(arglistnamedmixed(argitem, argitem, ...))
                    if args:
                        if not self.in_inline_body:
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
                    postfix_ops.append(Tree('fieldfan', [Tree('fieldlist', items)]))
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
                # Wrap args based on context:
                # - In inline body: call(argitem, argitem, ...)
                # - Not inline: call(arglistnamedmixed(argitem, argitem, ...))
                if args:
                    if not self.in_inline_body:
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

        if self.check(TT.STRING, TT.RAW_STRING, TT.RAW_HASH_STRING, TT.SHELL_STRING):
            tok = self.advance()
            return Token(tok.type.name, tok.value)

        if self.match(TT.TRUE):
            return Token('TRUE', 'true')
        if self.match(TT.FALSE):
            return Token('FALSE', 'false')
        if self.match(TT.NIL):
            return Token('NIL', 'nil')

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
            elements = []

            # Skip newlines in array
            while self.match(TT.NEWLINE):
                pass

            while not self.check(TT.RSQB, TT.EOF):
                elements.append(self.parse_expr())

                # Skip newlines
                while self.match(TT.NEWLINE):
                    pass

                if not self.match(TT.COMMA):
                    break

                # Skip newlines after comma
                while self.match(TT.NEWLINE):
                    pass

            self.expect(TT.RSQB)
            return Tree('array', elements)

        # Object literal
        if self.match(TT.LBRACE):
            items = []

            while not self.check(TT.RBRACE, TT.EOF):
                # Skip newlines
                while self.match(TT.NEWLINE):
                    pass

                if self.check(TT.RBRACE):
                    break

                items.append(self.parse_object_item())

                # Skip newlines
                while self.match(TT.NEWLINE):
                    pass

                if not self.match(TT.COMMA):
                    # Allow optional trailing comma
                    while self.match(TT.NEWLINE):
                        pass
                    break

            self.expect(TT.RBRACE)
            return Tree('object', items)

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
                body = self.parse_expr()
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
            body = self.parse_expr()
            return Tree('obj_get', [Token('IDENT', name.value), body])

        if self.match(TT.SET):
            name = self.expect(TT.IDENT)
            self.expect(TT.LPAR)
            param = self.expect(TT.IDENT)
            self.expect(TT.RPAR)
            self.expect(TT.COLON)
            body = self.parse_expr()
            return Tree('obj_set', [Token('IDENT', name.value), Token('IDENT', param.value), body])

        raise ParseError("Expected object item", self.current)

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
