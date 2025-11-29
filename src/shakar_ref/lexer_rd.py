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
            self.advance()  # \r
            self.advance()  # \n
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
        start_col = self.column - 1
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

    def scan_raw_string(self):
        """Scan raw string: raw"..." or raw'...' or raw#"..."#"""
        # 'raw' keyword already consumed, build full token including 'raw' prefix
        # Check for hash-delimited raw string
        if self.peek() == '#':
            hash_char = self.advance()  # consume #
            quote = self.advance()  # " or '
            content = ''

            # Scan until we find quote followed by #
            while self.pos < len(self.source):
                if self.peek() == quote and self.pos + 1 < len(self.source) and self.source[self.pos + 1] == '#':
                    self.advance()  # consume quote
                    self.advance()  # consume #
                    # Emit full raw#"..."# token value
                    full_value = 'raw#' + quote + content + quote + '#'
                    self.emit(TT.RAW_HASH_STRING, full_value)
                    return
                content += self.advance()

            raise LexError(f"Unterminated hash raw string at line {self.line}")
        else:
            # Regular raw string
            quote = self.advance()
            content = ''

            while self.pos < len(self.source) and self.peek() != quote:
                content += self.advance()

            if self.pos >= len(self.source):
                raise LexError(f"Unterminated raw string at line {self.line}")

            self.advance()  # Closing quote
            # Emit full raw"..." token value
            full_value = 'raw' + quote + content + quote
            self.emit(TT.RAW_STRING, full_value)

    def scan_shell_string(self):
        """Scan shell string: sh"..." or sh'...'"""
        # 'sh' keyword already consumed
        quote = self.advance()
        value = ''

        while self.pos < len(self.source) and self.peek() != quote:
            value += self.advance()

        if self.pos >= len(self.source):
            raise LexError(f"Unterminated shell string at line {self.line}")

        self.advance()  # Closing quote
        self.emit(TT.SHELL_STRING, value)

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
        ch = self.peek()

        # Two-character operators
        two_char = ch + self.peek(1)

        if two_char == '==':
            self.advance(); self.advance()
            self.emit(TT.EQ, '==')
        elif two_char == '!=':
            self.advance(); self.advance()
            self.emit(TT.NEQ, '!=')
        elif two_char == '<=':
            self.advance(); self.advance()
            self.emit(TT.LTE, '<=')
        elif two_char == '>=':
            self.advance(); self.advance()
            self.emit(TT.GTE, '>=')
        elif two_char == '&&':
            self.advance(); self.advance()
            self.emit(TT.AND, two_char)
        elif two_char == '||':
            self.advance(); self.advance()
            self.emit(TT.OR, two_char)
        elif two_char == ':=':
            self.advance(); self.advance()
            self.emit(TT.WALRUS, ':=')
        elif two_char == '.=':
            self.advance(); self.advance()
            self.emit(TT.APPLYASSIGN, '.=')
        elif two_char == '+=':
            self.advance(); self.advance()
            self.emit(TT.PLUSEQ, '+=')
        elif two_char == '-=':
            self.advance(); self.advance()
            self.emit(TT.MINUSEQ, '-=')
        elif two_char == '*=':
            self.advance(); self.advance()
            self.emit(TT.STAREQ, '*=')
        elif two_char == '/=':
            self.advance(); self.advance()
            self.emit(TT.SLASHEQ, '/=')
        elif two_char == '//':
            self.advance(); self.advance()
            # Check for //=
            if self.peek() == '=':
                self.advance()
                self.emit(TT.FLOORDIVEQ, '//=')
            else:
                self.emit(TT.FLOORDIV, '//')
        elif two_char == '**':
            self.advance(); self.advance()
            # Check for **=
            if self.peek() == '=':
                self.advance()
                self.emit(TT.POWEQ, '**=')
            else:
                self.emit(TT.POW, '**')
        elif two_char == '%=':
            self.advance(); self.advance()
            self.emit(TT.MODEQ, '%=')
        elif two_char == '++':
            self.advance(); self.advance()
            self.emit(TT.INCR, '++')
        elif two_char == '--':
            self.advance(); self.advance()
            self.emit(TT.DECR, '--')
        elif two_char == '??':
            self.advance(); self.advance()
            self.emit(TT.NULLISH, '??')
        elif two_char == '+>':
            self.advance(); self.advance()
            self.emit(TT.DEEPMERGE, '+>')

        # Single-character operators
        elif ch == '+':
            self.advance()
            self.emit(TT.PLUS, '+')
        elif ch == '-':
            self.advance()
            self.emit(TT.MINUS, '-')
        elif ch == '*':
            self.advance()
            self.emit(TT.STAR, '*')
        elif ch == '/':
            self.advance()
            self.emit(TT.SLASH, '/')
        elif ch == '%':
            self.advance()
            self.emit(TT.MOD, '%')
        elif ch == '^':
            self.advance()
            self.emit(TT.CARET, '^')
        elif ch == '<':
            self.advance()
            self.emit(TT.LT, '<')
        elif ch == '>':
            self.advance()
            self.emit(TT.GT, '>')
        elif ch == '!':
            self.advance()
            self.emit(TT.NEG, '!')
        elif ch == '=':
            self.advance()
            self.emit(TT.ASSIGN, '=')
        elif ch == '(':
            self.advance()
            self.emit(TT.LPAR, '(')
        elif ch == ')':
            self.advance()
            self.emit(TT.RPAR, ')')
        elif ch == '[':
            self.advance()
            self.emit(TT.LSQB, '[')
        elif ch == ']':
            self.advance()
            self.emit(TT.RSQB, ']')
        elif ch == '{':
            self.advance()
            self.emit(TT.LBRACE, '{')
        elif ch == '}':
            self.advance()
            self.emit(TT.RBRACE, '}')
        elif ch == '.':
            self.advance()
            self.emit(TT.DOT, '.')
        elif ch == ',':
            self.advance()
            self.emit(TT.COMMA, ',')
        elif ch == ':':
            self.advance()
            self.emit(TT.COLON, ':')
        elif ch == ';':
            self.advance()
            self.emit(TT.SEMI, ';')
        elif ch == '?':
            self.advance()
            self.emit(TT.QMARK, '?')
        elif ch == '@':
            self.advance()
            self.emit(TT.AT, '@')
        elif ch == '$':
            self.advance()
            self.emit(TT.DOLLAR, '$')
        elif ch == '`':
            self.advance()
            self.emit(TT.BACKQUOTE, '`')
        elif ch == '&':
            self.advance()
            self.emit(TT.AMP, '&')
        elif ch == '|':
            self.advance()
            self.emit(TT.PIPE, '|')
        else:
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

    def advance(self) -> str:
        """Consume current character"""
        ch = self.source[self.pos] if self.pos < len(self.source) else '\0'
        self.pos += 1
        self.column += 1
        return ch

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

    def unescape(self, ch: str) -> str:
        """Convert escape sequence to character"""
        escape_map = {
            'n': '\n',
            'r': '\r',
            't': '\t',
            '\\': '\\',
            '"': '"',
            "'": "'",
        }
        return escape_map.get(ch, ch)

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
