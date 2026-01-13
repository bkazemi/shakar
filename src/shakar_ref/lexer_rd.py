"""
Lexer for Shakar - Recursive Descent Parser

Tokenizes Shakar source code into a stream of tokens.

Features:
- Single-pass tokenization
- Indentation-aware (emits INDENT/DEDENT)
- Position tracking (line, column)
- String literal handling (raw, shell, etc.)
"""

from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple

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
        "if": TT.IF,
        "elif": TT.ELIF,
        "else": TT.ELSE,
        "while": TT.WHILE,
        "for": TT.FOR,
        "in": TT.IN,
        "break": TT.BREAK,
        "continue": TT.CONTINUE,
        "return": TT.RETURN,
        "fn": TT.FN,
        "await": TT.AWAIT,
        "using": TT.USING,
        "call": TT.CALL,
        "defer": TT.DEFER,
        "throw": TT.THROW,
        "catch": TT.CATCH,
        "assert": TT.ASSERT,
        "dbg": TT.DBG,
        "decorator": TT.DECORATOR,
        "get": TT.GET,
        "set": TT.SET,
        "hook": TT.HOOK,
        "over": TT.OVER,
        "any": TT.ANY,
        "all": TT.ALL,
        "true": TT.TRUE,
        "false": TT.FALSE,
        "nil": TT.NIL,
        "and": TT.AND,
        "or": TT.OR,
        "not": TT.NOT,
        "is": TT.IS,
    }

    # Operator mapping: longest matches first to handle prefixes correctly
    OPERATORS = [
        # Three-character operators
        ("//=", TT.FLOORDIVEQ),
        ("**=", TT.POWEQ),
        ("...", TT.SPREAD),
        # Two-character operators
        ("~~", TT.REGEXMATCH),
        ("==", TT.EQ),
        ("!=", TT.NEQ),
        ("<=", TT.LTE),
        (">=", TT.GTE),
        ("&&", TT.AND),
        ("||", TT.OR),
        (":=", TT.WALRUS),
        (".=", TT.APPLYASSIGN),
        ("+=", TT.PLUSEQ),
        ("-=", TT.MINUSEQ),
        ("*=", TT.STAREQ),
        ("/=", TT.SLASHEQ),
        ("//", TT.FLOORDIV),
        ("**", TT.POW),
        ("%=", TT.MODEQ),
        ("++", TT.INCR),
        ("--", TT.DECR),
        ("??", TT.NULLISH),
        ("+>", TT.DEEPMERGE),
        # Single-character operators
        ("+", TT.PLUS),
        ("-", TT.MINUS),
        ("*", TT.STAR),
        ("/", TT.SLASH),
        ("%", TT.MOD),
        ("^", TT.CARET),
        ("<", TT.LT),
        (">", TT.GT),
        ("!", TT.NEG),
        ("=", TT.ASSIGN),
        ("(", TT.LPAR),
        (")", TT.RPAR),
        ("[", TT.LSQB),
        ("]", TT.RSQB),
        ("{", TT.LBRACE),
        ("}", TT.RBRACE),
        (".", TT.DOT),
        (",", TT.COMMA),
        (":", TT.COLON),
        (";", TT.SEMI),
        ("?", TT.QMARK),
        ("@", TT.AT),
        ("$", TT.DOLLAR),
        ("`", TT.BACKQUOTE),
        ("&", TT.AMP),
        ("|", TT.PIPE),
        ("~", TT.TILDE),
    ]

    DURATION_UNITS = ["nsec", "usec", "msec", "sec", "min", "hr", "day", "wk"]
    DURATION_VALUES: Dict[str, int] = {
        "nsec": 1,
        "usec": 1_000,
        "msec": 1_000_000,
        "sec": 1_000_000_000,
        "min": 60_000_000_000,
        "hr": 3_600_000_000_000,
        "day": 86_400_000_000_000,
        "wk": 604_800_000_000_000,
    }

    SIZE_UNITS = ["tib", "gib", "mib", "kib", "tb", "gb", "mb", "kb", "b"]
    SIZE_VALUES: Dict[str, int] = {
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

    def __init__(self, source: str, track_indentation: bool = False):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Tok] = []

        # Indentation tracking
        self.track_indentation = track_indentation
        self.indent_stack = [0]  # Stack of indent levels (counts)
        self.indent_strings = [""]  # Stack of indent strings
        self.at_line_start = True
        self.pending_dedents = 0

    # ========================================================================
    # Main Tokization
    # ========================================================================

    def tokenize(self) -> List[Tok]:
        """Tokenize entire source, return token list"""
        while self.pos < len(self.source):
            self.scan_token()

        # Emit remaining DEDENTs at EOF
        if self.track_indentation:
            while len(self.indent_stack) > 1:
                self.indent_stack.pop()
                self.emit(TT.DEDENT, "")

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
        if self.peek() == "#":
            self.skip_comment()
            return

        # Newlines
        if self.peek() in ("\n", "\r"):
            self.scan_newline()
            return

        # String literals
        if self.peek() in ('"', "'"):
            self.scan_string()
            return

        # Raw strings
        if self.match_keyword("raw"):
            self.scan_raw_string()
            return

        # Shell strings
        if self.match_keyword("sh"):
            self.scan_shell_string()
            return
        # Path strings
        if self.peek() == "p" and self.peek(1) in ('"', "'"):
            self.advance()  # consume 'p'
            self.scan_path_string()
            return
        # Regex literals (r"..."/flags)
        if self.peek() == "r" and self.peek(1) in ('"', "'"):
            self.scan_regex_literal()
            return

        # Numbers
        if self.peek().isdigit():
            self.scan_number()
            return

        # Identifiers and keywords
        if self.peek().isalpha() or self.peek() == "_":
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
        indent_str = ""
        while self.peek() in (" ", "\t"):
            ch = self.peek()
            indent_str += ch
            if ch == " ":
                indent += 1
            else:
                indent += 8  # Tab = 8 spaces
            self.advance()

        self.at_line_start = False

        # Skip blank lines
        if self.peek() in ("\n", "\r", "#"):
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
                dedent_str = self.indent_strings[-1] if self.indent_strings else ""
                self.emit(TT.DEDENT, dedent_str)

            if self.indent_stack[-1] != indent:
                raise LexError(f"Indentation mismatch at line {self.line}")

    # ========================================================================
    # Tok Scanners
    # ========================================================================

    def scan_newline(self):
        """Scan newline character"""
        start_line, start_col = self.line, self.column

        if self.peek() == "\r" and self.peek(1) == "\n":
            self.advance(2)  # consume CRLF
        else:
            self.advance()

        self.emit(TT.NEWLINE, "\n", start_line=start_line, start_col=start_col)
        self.line += 1
        self.column = 1
        self.at_line_start = True

    def scan_string(self):
        """Scan string literal: "..." or '...'"""
        start_line, start_col = self.line, self.column
        start_pos = self.pos
        quote = self.advance()
        self.scan_quoted_content(quote, allow_escapes=True)
        if self.pos >= len(self.source):
            raise LexError(f"Unterminated string at line {start_line}")

        self.advance()  # Closing quote
        value = self.source[start_pos : self.pos]
        self.emit(TT.STRING, value, start_line=start_line, start_col=start_col)

    def scan_quoted_content(self, quote: str, allow_escapes: bool = False) -> str:
        """Helper to scan content between quotes"""
        start_pos = self.pos
        while self.pos < len(self.source) and self.peek() != quote:
            if allow_escapes and self.peek() == "\\":
                self.advance()
                if self.pos < len(self.source):
                    self.advance()
            else:
                self.advance()
        return self.source[start_pos : self.pos]

    def scan_raw_string(self):
        """Scan raw string: raw"..." or raw'...' or raw#"..."#"""
        # 'raw' keyword already consumed - start_col is 3 chars back
        start_line, start_col = self.line, self.column - 3
        start_pos = self.pos - 3

        if self.peek() == "#":
            # Hash-delimited raw string: raw#"..."#
            self.advance()  # consume #
            quote = self.advance()

            # Scan until we find quote followed by #
            while self.pos < len(self.source):
                if self.peek() == quote and self.peek(1) == "#":
                    self.advance(2)  # consume quote and #
                    full_value = self.source[start_pos : self.pos]
                    self.emit(
                        TT.RAW_HASH_STRING,
                        full_value,
                        start_line=start_line,
                        start_col=start_col,
                    )
                    return
                self.advance()

            raise LexError(f"Unterminated hash raw string at line {self.line}")
        else:
            # Regular raw string: raw"..." or raw'...'
            quote = self.advance()
            self.scan_quoted_content(quote)
            if self.pos >= len(self.source):
                raise LexError(f"Unterminated raw string at line {self.line}")

            self.advance()  # Closing quote
            full_value = self.source[start_pos : self.pos]
            self.emit(
                TT.RAW_STRING, full_value, start_line=start_line, start_col=start_col
            )

    def scan_shell_string(self):
        """Scan shell string: sh"..." or sh'...'"""
        # 'sh' keyword already consumed - start_col is 2 chars back
        start_line, start_col = self.line, self.column - 2
        quote = self.advance()
        start_pos = self.pos
        self.scan_quoted_content(quote)

        if self.pos >= len(self.source):
            raise LexError(f"Unterminated shell string at line {self.line}")

        content = self.source[start_pos : self.pos]
        self.advance()  # Closing quote
        self.emit(TT.SHELL_STRING, content, start_line=start_line, start_col=start_col)

    def scan_path_string(self):
        """Scan path string: p"..." or p'...'"""
        # 'p' already consumed - start_col is 1 char back
        start_line, start_col = self.line, self.column - 1
        start_pos = self.pos - 1
        quote = self.advance()
        self.scan_quoted_content(quote, allow_escapes=True)
        if self.pos >= len(self.source):
            raise LexError(f"Unterminated path string at line {start_line}")

        self.advance()  # Closing quote
        full_value = self.source[start_pos : self.pos]
        self.emit(
            TT.PATH_STRING, full_value, start_line=start_line, start_col=start_col
        )

    def scan_regex_literal(self):
        """Scan regex literal: r"..." or r'...' with optional /flags."""
        start_line, start_col = self.line, self.column
        self.advance()  # consume 'r'
        quote = self.advance()
        start_pos = self.pos

        while self.pos < len(self.source) and self.peek() != quote:
            if self.peek() in ("\n", "\r"):
                raise LexError(f"Unterminated regex literal at line {start_line}")
            if self.peek() == "\\":
                self.advance()
                if self.pos < len(self.source):
                    self.advance()
            else:
                self.advance()

        if self.pos >= len(self.source):
            raise LexError(f"Unterminated regex literal at line {start_line}")

        pattern = self.source[start_pos : self.pos]
        self.advance()  # Closing quote

        flags = ""
        if self.peek() == "/":
            self.advance()  # consume /
            if self.peek() not in ("i", "m", "s", "x", "f"):
                raise LexError(f"Unknown regex flag at line {self.line}")
            while self.peek() in ("i", "m", "s", "x", "f"):
                flags += self.advance()
            if self.peek().isalnum() or self.peek() == "_":
                raise LexError(f"Unknown regex flag at line {self.line}")

        self.emit(
            TT.REGEX, (pattern, flags), start_line=start_line, start_col=start_col
        )

    def scan_number(self):
        """Scan number literal, including duration/size literals."""
        start_line, start_col = self.line, self.column
        start_pos = self.pos

        # Integer part
        while self.peek().isdigit():
            self.advance()

        # Decimal part
        if self.peek() == "." and self.peek(1).isdigit():
            self.advance()  # .
            while self.peek().isdigit():
                self.advance()

        # Scientific notation
        if self.peek() in ("e", "E"):
            self.advance()
            if self.peek() in ("+", "-"):
                self.advance()
            while self.peek().isdigit():
                self.advance()

        value = self.source[start_pos : self.pos]
        unit = self._match_unit(self.DURATION_UNITS)
        literal_type = TT.DURATION

        if unit is None:
            unit = self._match_unit(self.SIZE_UNITS)
            literal_type = TT.SIZE

        if unit is None:
            self.emit(TT.NUMBER, value, start_line=start_line, start_col=start_col)
            return

        if self.peek().isalpha() or self.peek() == "_":
            raise LexError(
                f"Invalid duration/size literal at line {start_line}, col {start_col}"
            )

        has_decimal = "." in value or "e" in value.lower()

        if has_decimal and self.peek().isdigit():
            raise LexError(
                f"Decimal component in compound literal at line {start_line}, col {start_col}"
            )

        unit_list = (
            self.DURATION_UNITS if literal_type == TT.DURATION else self.SIZE_UNITS
        )
        while self.peek().isdigit():
            while self.peek().isdigit():
                self.advance()
            unit = self._match_unit(unit_list)
            if unit is None:
                raise LexError(
                    f"Expected unit in compound literal at line {start_line}, col {start_col}"
                )

        literal_value = self.source[start_pos : self.pos]
        unit_values = (
            self.DURATION_VALUES if literal_type == TT.DURATION else self.SIZE_VALUES
        )
        kind = "duration" if literal_type == TT.DURATION else "size"
        total = self._compute_compound_value(
            literal_value, unit_values, kind, start_line
        )
        self.emit(
            literal_type,
            (literal_value, total),
            start_line=start_line,
            start_col=start_col,
        )

    def scan_identifier(self):
        """Scan identifier or keyword"""
        start_line, start_col = self.line, self.column
        start_pos = self.pos

        while self.peek().isalnum() or self.peek() == "_":
            self.advance()

        # Check if keyword
        value = self.source[start_pos : self.pos]
        token_type = self.KEYWORDS.get(value, TT.IDENT)
        self.emit(token_type, value, start_line=start_line, start_col=start_col)

    def scan_operator(self):
        """Scan operators and punctuation"""
        start_line, start_col = self.line, self.column

        for op_str, op_type in self.OPERATORS:
            if self.source.startswith(op_str, self.pos):
                self.advance(len(op_str))
                self.emit(op_type, op_str, start_line=start_line, start_col=start_col)
                return

        ch = self.peek()
        raise LexError(
            f"Unexpected character '{ch}' at line {self.line}, col {self.column}"
        )

    # ========================================================================
    # Utilities
    # ========================================================================

    def peek(self, offset: int = 0) -> str:
        """Look ahead at character"""
        idx = self.pos + offset
        if idx < len(self.source):
            return self.source[idx]
        return "\0"

    def advance(self, n: int = 1) -> str:
        """Consume n characters and return them as a string"""
        if n < 0:
            raise ValueError(f"advance() requires n >= 0, got {n}")
        result = ""
        for _ in range(n):
            ch = self.source[self.pos] if self.pos < len(self.source) else "\0"
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
        if self.peek(len(keyword)).isalnum() or self.peek(len(keyword)) == "_":
            return False

        # Consume keyword
        for _ in keyword:
            self.advance()

        return True

    def skip_whitespace(self) -> bool:
        """Skip whitespace (not newlines), return True if any skipped"""
        skipped = False
        while self.peek() in (" ", "\t"):
            self.advance()
            skipped = True
        return skipped

    def skip_comment(self):
        """Skip comment until end of line"""
        while self.peek() not in ("\n", "\r", "\0"):
            self.advance()

    def _match_unit(self, units: List[str]) -> Optional[str]:
        for unit in units:
            if self.source.startswith(unit, self.pos):
                self.advance(len(unit))
                return unit
        return None

    def _compute_compound_value(
        self, raw: str, unit_values: Dict[str, int], kind: str, line: int
    ) -> int:
        """Parse compound literal and compute total value."""
        unit_keys = sorted(unit_values.keys(), key=len, reverse=True)
        parts: List[Tuple[str, str]] = []
        pos = 0

        while pos < len(raw):
            start = pos
            while pos < len(raw) and raw[pos].isdigit():
                pos += 1

            if pos < len(raw) and raw[pos] == ".":
                pos += 1
                while pos < len(raw) and raw[pos].isdigit():
                    pos += 1

            if pos < len(raw) and raw[pos] in ("e", "E"):
                pos += 1
                if pos < len(raw) and raw[pos] in ("+", "-"):
                    pos += 1
                while pos < len(raw) and raw[pos].isdigit():
                    pos += 1

            num_str = raw[start:pos]
            unit = None
            for u in unit_keys:
                if raw.startswith(u, pos):
                    unit = u
                    pos += len(u)
                    break
            if unit is None:
                raise LexError(f"Malformed {kind} literal at line {line}")
            parts.append((num_str, unit))

        if len(parts) > 1 and any("." in n or "e" in n.lower() for n, _ in parts):
            raise LexError(
                f"Decimal component in compound {kind} literal at line {line}"
            )

        total = Decimal(0)
        for num_str, unit in parts:
            try:
                num = Decimal(num_str)
            except InvalidOperation as exc:
                raise LexError(f"Malformed {kind} literal at line {line}") from exc
            total += num * Decimal(unit_values[unit])

        if total != total.to_integral_value():
            raise LexError(
                f"{kind.capitalize()} literal not representable at line {line}"
            )

        total_int = int(total)
        if total_int < -(2**63) or total_int > 2**63 - 1:
            raise LexError(
                f"{kind.capitalize()} literal overflows int64 at line {line}"
            )

        return total_int

    def emit(
        self,
        token_type: TT,
        value,
        *,
        start_line: Optional[int] = None,
        start_col: Optional[int] = None,
    ):
        """Emit a token with position info.

        If start_line/start_col not provided, uses current position.
        """
        tok = Tok(
            type=token_type,
            value=value,
            line=start_line if start_line is not None else self.line,
            column=start_col if start_col is not None else self.column,
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


if __name__ == "__main__":
    # Simple test
    test_source = """
fn fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
"""

    tokens = tokenize(test_source, track_indentation=True)
    for tok in tokens:
        print(tok)
