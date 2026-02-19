"""
Lexer for Shakar - Recursive Descent Parser

Tokenizes Shakar source code into a stream of tokens.

Features:
- Single-pass tokenization
- Indentation-aware (emits INDENT/DEDENT)
- Position tracking (line, column)
- String literal handling (raw, shell, etc.)
"""

from typing import Dict, List, Optional, Tuple

from .token_types import TT, Tok
from .utils import parse_compound_literal
from .types import ShakarTypeError

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
        "unless": TT.UNLESS,
        "while": TT.WHILE,
        "for": TT.FOR,
        "match": TT.MATCH,
        "in": TT.IN,
        "break": TT.BREAK,
        "continue": TT.CONTINUE,
        "return": TT.RETURN,
        "fn": TT.FN,
        "let": TT.LET,
        "once": TT.ONCE,
        "wait": TT.WAIT,
        "spawn": TT.SPAWN,
        "using": TT.USING,
        "call": TT.CALL,
        "defer": TT.DEFER,
        "after": TT.AFTER,
        "throw": TT.THROW,
        "catch": TT.CATCH,
        "try": TT.TRY,
        "assert": TT.ASSERT,
        "dbg": TT.DBG,
        "decorator": TT.DECORATOR,
        "get": TT.GET,
        "set": TT.SET,
        "hook": TT.HOOK,
        "over": TT.OVER,
        "bind": TT.BIND,
        "import": TT.IMPORT,
        "fan": TT.FAN,
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
        ("->", TT.SEND),
        ("<-", TT.RECV),
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

    # Prefixed string dispatch table (longest prefix first)
    _PREFIXED_STRINGS = [
        (
            "sh_raw!",
            TT.SHELL_BANG_STRING,
            {"dedent": False, "kind": "shell string literal"},
        ),
        ("sh_raw", TT.SHELL_STRING, {"dedent": False, "kind": "shell string literal"}),
        ("sh!", TT.SHELL_BANG_STRING, {"dedent": True, "kind": "shell string literal"}),
        ("sh", TT.SHELL_STRING, {"dedent": True, "kind": "shell string literal"}),
        (
            "env",
            TT.ENV_STRING,
            {"allow_newlines": False, "full_value": True, "kind": "env string literal"},
        ),
        (
            "p",
            TT.PATH_STRING,
            {
                "allow_newlines": False,
                "full_value": True,
                "kind": "path string literal",
            },
        ),
    ]

    def __init__(
        self,
        source: str,
        track_indentation: bool = False,
        emit_comments: bool = False,
    ):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Tok] = []
        self.emit_comments = emit_comments

        # Indentation tracking
        self.track_indentation = track_indentation
        self.indent_stack = [0]  # Stack of indent levels (counts)
        self.indent_strings = [""]  # Stack of indent strings
        self.at_line_start = True
        self.pending_dedents = 0
        self.group_depth = 0
        self.line_ended_with_colon = False
        self.indent_after_colon = False
        self.prev_line_indent: Optional[int] = None
        self.prev_line_indent_str: Optional[str] = None

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
            start_line = self.line
            start_col = self.column
            start_pos = self.pos
            self.skip_comment()
            if self.emit_comments:
                value = self.source[start_pos : self.pos]
                self.emit(
                    TT.COMMENT,
                    value,
                    start_line=start_line,
                    start_col=start_col,
                )
            return

        # Newlines
        if self.peek() in {"\n", "\r"}:
            self.scan_newline()
            return

        # String literals
        if self.peek() in {'"', "'"}:
            self.scan_string()
            return

        # Raw strings (raw"..." or raw#"..."#)
        if self.peek_at("raw"):
            if self.peek(3) in {'"', "'", "#"}:
                self.advance(3)
                self.scan_raw_string()
                return

        # Prefixed strings (sh, sh!, sh_raw, sh_raw!, env, p)
        for prefix, tt, kw in self._PREFIXED_STRINGS:
            if self.match_string_prefix(prefix):
                self._scan_prefixed_string(len(prefix), tt, **kw)
                return

        # Regex literals (r"..."/flags)
        if self.match_string_prefix("r"):
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
        while self.peek() in {" ", "\t"}:
            ch = self.peek()
            indent_str += ch
            if ch == " ":
                indent += 1
            else:
                indent += 8  # Tab = 8 spaces
            self.advance()

        self.at_line_start = False

        if (
            self.indent_after_colon
            and self.group_depth > 0
            and len(self.indent_stack) == 1
            and self.prev_line_indent is not None
            and self.prev_line_indent > 0
            and self.prev_line_indent < indent
        ):
            self.indent_stack.append(self.prev_line_indent)
            self.indent_strings.append(self.prev_line_indent_str or "")

        # Skip blank lines
        if self.peek() in {"\n", "\r", "#"}:
            return

        self.prev_line_indent = indent
        self.prev_line_indent_str = indent_str

        if (
            self.group_depth > 0
            and not self.indent_after_colon
            and len(self.indent_stack) == 1
        ):
            return

        self.indent_after_colon = False

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
                # If we are inside a group and return to base indentation, ignore mismatch
                # because the base indentation was likely suppressed/ignored.
                if not (self.group_depth > 0 and len(self.indent_stack) == 1):
                    raise LexError("Indentation mismatch", line=self.line)

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
        self.scan_quoted_content(
            quote, allow_escapes=True, allow_newlines=True, kind="string literal"
        )
        if self.pos >= len(self.source):
            raise LexError("Unterminated string", line=start_line, column=start_col)

        body = self.source[start_pos + 1 : self.pos]
        self.advance()  # Closing quote

        body = self._dedent_multiline(body)
        value = f"{quote}{body}{quote}"
        self.emit(TT.STRING, value, start_line=start_line, start_col=start_col)

    def scan_quoted_content(
        self,
        quote: str,
        allow_escapes: bool = False,
        allow_newlines: bool = False,
        kind: str = "string literal",
    ) -> str:
        """Helper to scan content between quotes"""
        start_pos = self.pos

        while self.pos < len(self.source) and self.peek() != quote:
            ch = self.peek()
            if ch in {"\n", "\r"} and not allow_newlines:
                raise LexError(f"Newline in {kind}", line=self.line, column=self.column)
            if allow_escapes and ch == "\\":
                self._advance_in_literal()
                if self.pos < len(self.source):
                    if self.peek() in {"\n", "\r"} and not allow_newlines:
                        raise LexError(
                            f"Newline in {kind}", line=self.line, column=self.column
                        )
                    self._advance_in_literal()
                continue
            self._advance_in_literal()

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
                self._advance_in_literal()

            raise LexError(
                "Unterminated hash raw string", line=self.line, column=self.column
            )
        else:
            # Regular raw string: raw"..." or raw'...'
            quote = self.advance()
            self.scan_quoted_content(
                quote,
                allow_newlines=True,
                allow_escapes=False,
                kind="raw string literal",
            )
            if self.pos >= len(self.source):
                raise LexError(
                    "Unterminated raw string", line=start_line, column=start_col
                )

            self.advance()  # Closing quote
            full_value = self.source[start_pos : self.pos]
            self.emit(
                TT.RAW_STRING, full_value, start_line=start_line, start_col=start_col
            )

    def _scan_prefixed_string(
        self,
        prefix_len: int,
        token_type: TT,
        *,
        allow_escapes: bool = True,
        allow_newlines: bool = True,
        dedent: bool = False,
        full_value: bool = False,
        kind: str = "string literal",
    ):
        """Scan a prefix-delimited string (sh, sh!, sh_raw, sh_raw!, env, p)."""
        start_line, start_col = self.line, self.column - prefix_len
        start_pos = self.pos - prefix_len
        quote = self.advance()
        content_start = self.pos
        self.scan_quoted_content(
            quote, allow_escapes=allow_escapes, allow_newlines=allow_newlines, kind=kind
        )
        if self.pos >= len(self.source):
            raise LexError(f"Unterminated {kind}", line=start_line, column=start_col)
        content = self.source[content_start : self.pos]
        if dedent:
            content = self._dedent_multiline(content)
        self.advance()  # closing quote
        value = self.source[start_pos : self.pos] if full_value else content
        self.emit(token_type, value, start_line=start_line, start_col=start_col)

    def scan_regex_literal(self):
        """Scan regex literal: r"..." or r'...' with optional /flags."""
        # 'r' already consumed by match_string_prefix
        start_line, start_col = self.line, self.column - 1
        quote = self.advance()
        start_pos = self.pos

        self.scan_quoted_content(
            quote, allow_escapes=True, allow_newlines=True, kind="regex literal"
        )

        if self.pos >= len(self.source):
            raise LexError(
                "Unterminated regex literal", line=start_line, column=start_col
            )

        pattern = self.source[start_pos : self.pos]
        self.advance()  # Closing quote

        flags = ""
        if self.peek() == "/":
            self.advance()  # consume /
            if self.peek() not in {"i", "m", "s", "x", "f"}:
                raise LexError("Unknown regex flag", line=self.line, column=self.column)
            while self.peek() in {"i", "m", "s", "x", "f"}:
                flags += self.advance()
            if self.peek().isalnum() or self.peek() == "_":
                raise LexError("Unknown regex flag", line=self.line, column=self.column)

        self.emit(
            TT.REGEX, (pattern, flags), start_line=start_line, start_col=start_col
        )

    def scan_number(self):
        """Scan number literal, including duration/size literals.

        Handles: integers (decimal, 0b, 0o, 0x), floats, scientific notation,
        and duration/size literals with unit suffixes. Underscores allowed
        between digits for readability (e.g., 1_000_000, 0xdead_beef).
        """
        start_line, start_col = self.line, self.column
        start_pos = self.pos

        # Try base-prefixed integer first (0b, 0o, 0x)
        if self._scan_prefixed_integer(start_line, start_col, start_pos):
            return

        # Decimal integer part
        self._scan_digits_with_underscores(start_line, start_col)

        # Optional fractional part (.digits)
        # Reject 1.e5 (dot followed by exponent without digits).
        # Allow 1.foo (member access) by only entering if next is digit/underscore.
        if self.peek() == ".":
            next_ch = self.peek(1)
            if next_ch in {"e", "E"}:
                raise LexError(
                    "Fractional digits required after decimal point",
                    line=start_line,
                    column=start_col,
                )
            if next_ch.isdigit() or next_ch == "_":
                self.advance()  # consume '.'
                if self.peek() == "_":
                    raise LexError(
                        "Leading underscore in fractional part",
                        line=start_line,
                        column=start_col,
                    )
                self._scan_digits_with_underscores(start_line, start_col)

        # Optional exponent (e+/-digits)
        if self.peek() in {"e", "E"}:
            self.advance()
            if self.peek() in {"+", "-"}:
                self.advance()
            # Check for leading underscore in exponent
            if self.peek() == "_":
                raise LexError(
                    "Leading underscore in exponent",
                    line=start_line,
                    column=start_col,
                )
            if not self._scan_digits_with_underscores(start_line, start_col):
                raise LexError(
                    "Missing digits in exponent",
                    line=start_line,
                    column=start_col,
                )

        # Reject trailing dot without member name (1e5., 1.5.)
        # Allow 1.foo and 1e5.bar (member access on numeric literals).
        if self.peek() == "." and not self.peek(1).isalpha() and self.peek(1) != "_":
            raise LexError(
                "Trailing dot after number literal",
                line=start_line,
                column=start_col,
            )

        value = self.source[start_pos : self.pos]
        unit = self._match_unit(self.DURATION_UNITS)
        literal_type = TT.DURATION

        if unit is None:
            unit = self._match_unit(self.SIZE_UNITS)
            literal_type = TT.SIZE

        if unit is None:
            # Reject number followed by letters (invalid suffix or typo like 1scec)
            if self.peek().isalpha() or self.peek() == "_":
                raise LexError(
                    "Invalid number suffix",
                    line=start_line,
                    column=start_col,
                    end_line=start_line,
                    end_column=self._bad_token_end_col(start_col, start_pos),
                )
            self.emit(TT.NUMBER, value, start_line=start_line, start_col=start_col)
            return

        if self.peek().isalpha() or self.peek() == "_":
            raise LexError(
                "Invalid duration/size literal",
                line=start_line,
                column=start_col,
                end_line=start_line,
                end_column=self._bad_token_end_col(start_col, start_pos),
            )

        has_decimal = "." in value or "e" in value.lower()

        if has_decimal and self.peek().isdigit():
            raise LexError(
                "Decimal component in compound literal",
                line=start_line,
                column=start_col,
            )

        unit_list = (
            self.DURATION_UNITS if literal_type == TT.DURATION else self.SIZE_UNITS
        )
        while self.peek().isdigit():
            self._scan_digits_with_underscores(start_line, start_col)
            unit = self._match_unit(unit_list)
            if unit is None:
                raise LexError(
                    "Expected unit in compound literal",
                    line=start_line,
                    column=start_col,
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

    def _scan_prefixed_integer(
        self, start_line: int, start_col: int, start_pos: int
    ) -> bool:
        """Try to scan a base-prefixed integer (0b, 0o, 0x).

        Returns True if a prefixed integer was found and emitted.
        Returns False if current position is not a base prefix (caller continues).
        Raises LexError on malformed prefixed integers.
        """
        if self.peek() != "0":
            return False

        prefix = self.peek(1)
        if prefix in {"B", "O", "X"}:
            raise LexError(
                "Uppercase base prefixes are not allowed",
                line=start_line,
                column=start_col,
            )

        base = {"b": 2, "o": 8, "x": 16}.get(prefix)
        if base is None:
            return False  # not a base prefix (e.g., 0.5 or plain 0)

        self.advance(2)  # consume '0' and prefix letter

        valid = lambda ch: self._is_valid_prefixed_digit(ch, base)
        if not self._scan_digits_with_underscores(start_line, start_col, valid=valid):
            raise LexError(
                "Incomplete base-prefixed integer",
                line=start_line,
                column=start_col,
            )

        # Base-prefixed integers cannot have decimal points (0x1.5 invalid)
        if self.peek() == "." and self.peek(1).isdigit():
            raise LexError(
                "Base-prefixed integers cannot be floats",
                line=start_line,
                column=start_col,
            )

        # Base-prefixed integers cannot have duration/size suffixes (0x10sec invalid)
        if self.peek().isalnum() or self.peek() == "_":
            raise LexError(
                "Base-prefixed integers cannot have unit suffixes",
                line=start_line,
                column=start_col,
                end_line=start_line,
                end_column=self._bad_token_end_col(start_col, start_pos),
            )

        literal = self.source[start_pos : self.pos]
        self.emit(TT.NUMBER, literal, start_line=start_line, start_col=start_col)

        return True

    def _is_valid_prefixed_digit(self, ch: str, base: int) -> bool:
        """Check if character is valid digit for given base (2, 8, or 16)."""
        return (ch.isdigit() and int(ch) < base) or (
            base == 16 and ch.lower() in "abcdef"
        )

    def _scan_digits_with_underscores(
        self, line: int, col: int, valid=str.isdigit
    ) -> bool:
        """Scan digits with optional underscore separators.

        Underscores must appear between digits only (no leading/trailing/consecutive).
        Returns True if at least one digit was scanned, False otherwise.
        """
        saw_digit = False
        prev_underscore = False

        while True:
            ch = self.peek()
            if ch == "_":
                if not saw_digit or prev_underscore:
                    raise LexError(
                        "Invalid underscore in number literal",
                        line=line,
                        column=col,
                    )
                prev_underscore = True
                self.advance()
                continue

            if valid(ch):
                saw_digit = True
                prev_underscore = False
                self.advance()
                continue

            break

        if prev_underscore:
            raise LexError(
                "Trailing underscore in number literal",
                line=line,
                column=col,
            )

        return saw_digit

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
                if op_type in {TT.LPAR, TT.LSQB, TT.LBRACE}:
                    self.group_depth += 1
                elif op_type in {TT.RPAR, TT.RSQB, TT.RBRACE}:
                    self.group_depth = max(0, self.group_depth - 1)
                return

        ch = self.peek()
        raise LexError(
            f"Unexpected character '{ch}'",
            line=self.line,
            column=self.column,
        )

    def _bad_token_end_col(self, start_col: int, start_pos: int) -> int:
        """Compute exclusive end column for an invalid token spanning from
        start_pos through any remaining alphanumeric/underscore chars."""
        end = self.pos
        while end < len(self.source) and (
            self.source[end].isalnum() or self.source[end] == "_"
        ):
            end += 1

        return start_col + (end - start_pos)

    # ========================================================================
    # Utilities
    # ========================================================================

    @staticmethod
    def _strip_initial_newline(text: str) -> str:
        if text.startswith("\r\n"):
            return text[2:]
        if text.startswith("\n") or text.startswith("\r"):
            return text[1:]
        return text

    @staticmethod
    def _split_lines(text: str) -> List[Tuple[str, str]]:
        lines: List[Tuple[str, str]] = []
        start = 0
        idx = 0
        length = len(text)

        while idx < length:
            ch = text[idx]
            if ch == "\n":
                lines.append((text[start:idx], "\n"))
                idx += 1
                start = idx
                continue
            if ch == "\r":
                if idx + 1 < length and text[idx + 1] == "\n":
                    lines.append((text[start:idx], "\r\n"))
                    idx += 2
                    start = idx
                else:
                    lines.append((text[start:idx], "\r"))
                    idx += 1
                    start = idx
                continue
            idx += 1

        lines.append((text[start:], ""))
        return lines

    def _dedent_multiline(self, text: str) -> str:
        if not text:
            return text

        if not (text.startswith("\n") or text.startswith("\r")):
            return text

        text = self._strip_initial_newline(text)
        lines = self._split_lines(text)

        indents: List[int] = []
        for line, _ending in lines:
            if line.strip(" \t") == "":
                continue
            indent = 0
            while indent < len(line) and line[indent] in {" ", "\t"}:
                indent += 1
            indents.append(indent)

        if not indents:
            return "".join(line + ending for line, ending in lines)

        min_indent = min(indents)
        if min_indent == 0:
            return "".join(line + ending for line, ending in lines)

        out: List[str] = []
        for line, ending in lines:
            if line.strip(" \t") == "":
                out.append(line + ending)
            else:
                out.append(line[min_indent:] + ending)
        return "".join(out)

    def _advance_in_literal(self) -> str:
        ch = self.peek()
        if ch in {"\n", "\r"}:
            crlf = ch == "\r" and self.peek(1) == "\n"
            self.pos += 2 if crlf else 1
            self.line += 1
            self.column = 1
            return "\r\n" if crlf else ch
        self.pos += 1
        self.column += 1
        return ch

    def peek(self, offset: int = 0) -> str:
        """Look ahead at character at pos+offset."""
        idx = self.pos + offset
        return self.source[idx] if idx < len(self.source) else "\0"

    def peek_at(self, s: str, offset: int = 0) -> bool:
        """Check if characters at pos+offset match s."""
        idx = self.pos + offset
        return self.source[idx : idx + len(s)] == s

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

    def match_string_prefix(self, prefix: str) -> bool:
        """Match a prefix string keyword only if followed by a quote."""
        if not self.peek_at(prefix) or self.peek(len(prefix)) not in {'"', "'"}:
            return False

        self.advance(len(prefix))
        return True

    def skip_whitespace(self) -> bool:
        """Skip whitespace (not newlines), return True if any skipped"""
        skipped = False
        while self.peek() in {" ", "\t"}:
            self.advance()
            skipped = True
        return skipped

    def skip_comment(self):
        """Skip comment until end of line"""
        while self.peek() not in {"\n", "\r", "\0"}:
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
        """Parse compound literal and compute total value.

        Delegates to the shared parse_compound_literal utility,
        converting ShakarTypeError to LexError for lexer context.
        """
        try:
            total, _units = parse_compound_literal(raw, unit_values, kind)
        except ShakarTypeError as exc:
            raise LexError(f"Malformed {kind} literal", line=line) from exc

        return total

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
        if token_type == TT.NEWLINE:
            self.indent_after_colon = self.line_ended_with_colon
            self.line_ended_with_colon = False
        elif token_type not in {TT.INDENT, TT.DEDENT, TT.EOF}:
            self.line_ended_with_colon = token_type == TT.COLON


class LexError(Exception):
    """Lexical analysis error with optional structured location."""

    def __init__(
        self,
        message: str,
        *,
        line: Optional[int] = None,
        column: Optional[int] = None,
        end_line: Optional[int] = None,
        end_column: Optional[int] = None,
    ):
        self.line = line
        self.column = column
        self.end_line = end_line
        self.end_column = end_column

        if line is not None and column is not None:
            display = f"{message} at line {line}, col {column}"
        elif line is not None:
            display = f"{message} at line {line}"
        else:
            display = message

        super().__init__(display)


# ============================================================================
# Testing
# ============================================================================


def tokenize(
    source: str,
    track_indentation: bool = False,
    emit_comments: bool = False,
) -> List[Tok]:
    """Convenience function to tokenize source"""
    lexer = Lexer(
        source,
        track_indentation=track_indentation,
        emit_comments=emit_comments,
    )
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
