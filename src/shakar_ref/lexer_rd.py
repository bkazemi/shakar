"""
Lexer for Shakar - Recursive Descent Parser

Tokenizes Shakar source code into a stream of tokens.

Features:
- Single-pass tokenization
- Indentation-aware (emits INDENT/DEDENT)
- Position tracking (line, column)
- String literal handling (raw, shell, etc.)
"""

from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import re

from .token_types import TT, Tok

# ============================================================================
# Lexer Implementation
# ============================================================================

class Lexer:
    """
    Shakar lexer with indentation handling.

    Based on Python's indentation model:
    - Track stack of indentation levels
    - Emit INDENT when level increases
    - Emit DEDENT when level decreases
    """

    # Keyword mapping
    KEYWORDS = {
        'if': TT.IF,
        'elif': TT.ELIF,
        'else': TT.ELSE,
        'while': TT.WHILE,
        'for': TT.FOR,
        'in': TT.IN,
        'break': TT.BREAK,
        'continue': TT.CONTINUE,
        'return': TT.RETURN,
        'fn': TT.FN,
        'await': TT.AWAIT,
        'using': TT.USING,
        'defer': TT.DEFER,
        'throw': TT.THROW,
        'catch': TT.CATCH,
        'assert': TT.ASSERT,
        'dbg': TT.DBG,
        'decorator': TT.DECORATOR,
        'get': TT.GET,
        'set': TT.SET,
        'hook': TT.HOOK,
        'over': TT.OVER,
        'any': TT.ANY,
        'all': TT.ALL,
        'true': TT.TRUE,
        'false': TT.FALSE,
        'nil': TT.NIL,
        'and': TT.AND,
        'or': TT.OR,
        'not': TT.NOT,
        'is': TT.IS,
    }

    # Operator mapping: longest matches first to handle prefixes correctly
    OPERATORS = [
        # Three-character operators
        ('//=', TT.FLOORDIVEQ),
        ('**=', TT.POWEQ),

        # Two-character operators
        ('==', TT.EQ),
        ('!=', TT.NEQ),
        ('<=', TT.LTE),
        ('>=', TT.GTE),
        ('&&', TT.AND),
        ('||', TT.OR),
        (':=', TT.WALRUS),
        ('.=', TT.APPLYASSIGN),
        ('+=', TT.PLUSEQ),
        ('-=', TT.MINUSEQ),
        ('*=', TT.STAREQ),
        ('/=', TT.SLASHEQ),
        ('//', TT.FLOORDIV),
        ('**', TT.POW),
        ('%=', TT.MODEQ),
        ('++', TT.INCR),
        ('--', TT.DECR),
        ('??', TT.NULLISH),
        ('+>', TT.DEEPMERGE),

        # Single-character operators
        ('+', TT.PLUS),
        ('-', TT.MINUS),
        ('*', TT.STAR),
        ('/', TT.SLASH),
        ('%', TT.MOD),
        ('^', TT.CARET),
        ('<', TT.LT),
        ('>', TT.GT),
        ('!', TT.NEG),
        ('=', TT.ASSIGN),
        ('(', TT.LPAR),
        (')', TT.RPAR),
        ('[', TT.LSQB),
        (']', TT.RSQB),
        ('{', TT.LBRACE),
        ('}', TT.RBRACE),
        ('.', TT.DOT),
        (',', TT.COMMA),
        (':', TT.COLON),
        (';', TT.SEMI),
        ('?', TT.QMARK),
        ('@', TT.AT),
        ('$', TT.DOLLAR),
        ('`', TT.BACKQUOTE),
        ('&', TT.AMP),
        ('|', TT.PIPE),
        ('~', TT.TILDE),
    ]

    def __init__(self, source: str, track_indentation: bool = False):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Tok] = []

        # Indentation tracking
        self.track_indentation = track_indentation
        self.indent_stack = [0]  # Stack of indent levels (counts)
        self.indent_strings = ['']  # Stack of indent strings
        self.at_line_start = True
        self.pending_dedents = 0

    # ========================================================================
    # Main Tokenization
    # ========================================================================

    def tokenize(self) -> List[Tok]:
        """Tokenize entire source, return token list"""
        while self.pos < len(self.source):
            self.scan_token()

        # Emit remaining DEDENTs at EOF
        if self.track_indentation:
            while len(self.indent_stack) > 1:
                self.indent_stack.pop()
                self.emit(TT.DEDENT, '')

        self.emit(TT.EOF, None)
        return self.tokens

    def scan_token(self):
        """Scan next token"""
        # Handle indentation at line start
        if self.at_line_start and self.track_indentation:
            self.handle_indentation()
            return

        # Skip whitespace (not newlines)
        if self.skip_whitespace():
            return

        # Comments
        if self.peek() == '#':
            self.skip_comment()
            return

        # Newlines
        if self.peek() in ('\n', '\r'):
            self.scan_newline()
            return

        # String literals
        if self.peek() in ('"', "'"):
            self.scan_string()
            return

        # Raw strings
        if self.match_keyword('raw'):
            self.scan_raw_string()
            return

        # Shell strings
        if self.match_keyword('sh'):
            self.scan_shell_string()
            return

        # Numbers
        if self.peek().isdigit():
            self.scan_number()
            return

        # Identifiers and keywords
        if self.peek().isalpha() or self.peek() == '_':
            self.scan_identifier()
            return

        # Operators and punctuation
        self.scan_operator()

    # ========================================================================
    # Indentation Handling
    # ========================================================================

    def handle_indentation(self):
        """
        Handle indentation at start of line.
        Emit INDENT/DEDENT tokens as needed.
        """
        # Count leading spaces/tabs and capture the actual string
        indent = 0
        indent_str = ''
        while self.peek() in (' ', '\t'):
            ch = self.peek()
            indent_str += ch
            if ch == ' ':
                indent += 1
            else:
                indent += 8  # Tab = 8 spaces
            self.advance()

        self.at_line_start = False

        # Skip blank lines
        if self.peek() in ('\n', '\r', '#'):
            return

        current_indent = self.indent_stack[-1]

        if indent > current_indent:
            # Increase indentation
            self.indent_stack.append(indent)
            self.indent_strings.append(indent_str)
            self.emit(TT.INDENT, indent_str)

        elif indent < current_indent:
            # Decrease indentation - may emit multiple DEDENTs
            while len(self.indent_stack) > 1 and self.indent_stack[-1] > indent:
                self.indent_stack.pop()
                self.indent_strings.pop()
                # DEDENT value should be the indentation we're returning to
                dedent_str = self.indent_strings[-1] if self.indent_strings else ''
                self.emit(TT.DEDENT, dedent_str)

            if self.indent_stack[-1] != indent:
                raise LexError(f"Indentation mismatch at line {self.line}")

    # ========================================================================
    # Token Scanners
    # ========================================================================

    def scan_newline(self):
        """Scan newline character"""
        if self.peek() == '\r' and self.peek(1) == '\n':
            self.advance(2)  # consume CRLF
        else:
            self.advance()

        self.emit(TT.NEWLINE, '\n')
        self.line += 1
        self.column = 1
        self.at_line_start = True

    def scan_string(self):
        """Scan string literal: "..." or '...'"""
        quote = self.advance()
        start_line = self.line
        value = quote  # Keep opening quote

        while self.pos < len(self.source) and self.peek() != quote:
            if self.peek() == '\\':
                # Keep escape sequence as-is
                value += self.advance()
                if self.pos < len(self.source):
                    value += self.advance()
            else:
                value += self.advance()

        if self.pos >= len(self.source):
            raise LexError(f"Unterminated string at line {start_line}")

        value += self.advance()  # Closing quote
        self.emit(TT.STRING, value)

    def scan_quoted_content(self, quote: str, allow_escapes: bool = False) -> str:
        """Helper to scan content between quotes"""
        content = ''
        while self.pos < len(self.source) and self.peek() != quote:
            if allow_escapes and self.peek() == '\\':
                content += self.advance()
                if self.pos < len(self.source):
                    content += self.advance()
            else:
                content += self.advance()
        return content

    def scan_raw_string(self):
        """Scan raw string: raw"..." or raw'...' or raw#"..."#"""
        # 'raw' keyword already consumed
        if self.peek() == '#':
            # Hash-delimited raw string: raw#"..."#
            self.advance()  # consume #
            quote = self.advance()
            content = ''

            # Scan until we find quote followed by #
            while self.pos < len(self.source):
                if self.peek() == quote and self.peek(1) == '#':
                    self.advance(2)  # consume quote and #
                    full_value = f'raw#{quote}{content}{quote}#'
                    self.emit(TT.RAW_HASH_STRING, full_value)
                    return
                content += self.advance()

            raise LexError(f"Unterminated hash raw string at line {self.line}")
        else:
            # Regular raw string: raw"..." or raw'...'
            quote = self.advance()
            content = self.scan_quoted_content(quote)

            if self.pos >= len(self.source):
                raise LexError(f"Unterminated raw string at line {self.line}")

            self.advance()  # Closing quote
            full_value = f'raw{quote}{content}{quote}'
            self.emit(TT.RAW_STRING, full_value)

    def scan_shell_string(self):
        """Scan shell string: sh"..." or sh'...'"""
        # 'sh' keyword already consumed
        quote = self.advance()
        content = self.scan_quoted_content(quote)

        if self.pos >= len(self.source):
            raise LexError(f"Unterminated shell string at line {self.line}")

        self.advance()  # Closing quote
        self.emit(TT.SHELL_STRING, content)

    def scan_number(self):
        """Scan number literal"""
        value = ''

        # Integer part
        while self.peek().isdigit():
            value += self.advance()

        # Decimal part
        if self.peek() == '.' and self.peek(1).isdigit():
            value += self.advance()  # .
            while self.peek().isdigit():
                value += self.advance()

        # Scientific notation
        if self.peek() in ('e', 'E'):
            value += self.advance()
            if self.peek() in ('+', '-'):
                value += self.advance()
            while self.peek().isdigit():
                value += self.advance()

        # Keep as string to match Lark
        self.emit(TT.NUMBER, value)

    def scan_identifier(self):
        """Scan identifier or keyword"""
        value = ''

        while self.peek().isalnum() or self.peek() == '_':
            value += self.advance()

        # Check if keyword
        token_type = self.KEYWORDS.get(value, TT.IDENT)
        self.emit(token_type, value)

    def scan_operator(self):
        """Scan operators and punctuation"""
        for op_str, op_type in self.OPERATORS:
            if self.source.startswith(op_str, self.pos):
                self.advance(len(op_str))
                self.emit(op_type, op_str)
                return

        ch = self.peek()
        raise LexError(f"Unexpected character '{ch}' at line {self.line}, col {self.column}")

    # ========================================================================
    # Utilities
    # ========================================================================

    def peek(self, offset: int = 0) -> str:
        """Look ahead at character"""
        idx = self.pos + offset
        if idx < len(self.source):
            return self.source[idx]
        return '\0'

    def advance(self, n: int = 1) -> str:
        """Consume n characters and return them as a string"""
        if n < 0:
            raise ValueError(f"advance() requires n >= 0, got {n}")
        result = ''
        for _ in range(n):
            ch = self.source[self.pos] if self.pos < len(self.source) else '\0'
            result += ch
            self.pos += 1
            self.column += 1
        return result

    def match_keyword(self, keyword: str) -> bool:
        """Check if next characters match keyword"""
        for i, ch in enumerate(keyword):
            if self.peek(i) != ch:
                return False

        # Ensure it's not part of identifier
        if self.peek(len(keyword)).isalnum() or self.peek(len(keyword)) == '_':
            return False

        # Consume keyword
        for _ in keyword:
            self.advance()

        return True

    def skip_whitespace(self) -> bool:
        """Skip whitespace (not newlines), return True if any skipped"""
        skipped = False
        while self.peek() in (' ', '\t'):
            self.advance()
            skipped = True
        return skipped

    def skip_comment(self):
        """Skip comment until end of line"""
        while self.peek() not in ('\n', '\r', '\0'):
            self.advance()

    def emit(self, token_type: TT, value):
        """Emit a token"""
        tok = Tok(
            type=token_type,
            value=value,
            line=self.line,
            column=self.column
        )
        self.tokens.append(tok)

class LexError(Exception):
    """Lexical analysis error"""
    pass

# ============================================================================
# Testing
# ============================================================================

def tokenize(source: str, track_indentation: bool = False) -> List[Tok]:
    """Convenience function to tokenize source"""
    lexer = Lexer(source, track_indentation=track_indentation)
    return lexer.tokenize()


if __name__ == '__main__':
    # Simple test
    test_source = '''
fn fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
'''

    tokens = tokenize(test_source, track_indentation=True)
    for tok in tokens:
        print(tok)
