"""
Token Types for Shakar Parser

Shared between lexer and parser to avoid circular dependencies.
"""

from typing import Any
from dataclasses import dataclass
from enum import Enum, auto


class TT(Enum):
    """Token Types - mirrors grammar terminals"""

    # Literals
    NUMBER = auto()
    DURATION = auto()
    SIZE = auto()
    STRING = auto()
    RAW_STRING = auto()
    RAW_HASH_STRING = auto()
    SHELL_STRING = auto()
    SHELL_BANG_STRING = auto()
    PATH_STRING = auto()
    ENV_STRING = auto()
    REGEX = auto()
    IDENT = auto()

    # Keywords
    IF = auto()
    ELIF = auto()
    ELSE = auto()
    UNLESS = auto()
    WHILE = auto()
    FOR = auto()
    MATCH = auto()
    IN = auto()
    BREAK = auto()
    CONTINUE = auto()
    RETURN = auto()
    FN = auto()
    LET = auto()
    WAIT = auto()
    SPAWN = auto()
    USING = auto()
    CALL = auto()
    DEFER = auto()
    AFTER = auto()
    THROW = auto()
    CATCH = auto()
    ASSERT = auto()
    DBG = auto()
    DECORATOR = auto()
    GET = auto()
    SET = auto()
    HOOK = auto()
    OVER = auto()
    BIND = auto()
    IMPORT = auto()
    ANY = auto()
    ALL = auto()
    GROUP = auto()
    FAN = auto()

    # Literals
    TRUE = auto()
    FALSE = auto()
    NIL = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    FLOORDIV = auto()
    MOD = auto()
    POW = auto()
    DEEPMERGE = auto()
    CARET = auto()

    # Comparison
    EQ = auto()
    NEQ = auto()
    LT = auto()
    LTE = auto()
    GT = auto()
    GTE = auto()
    SEND = auto()  # ->
    RECV = auto()  # <-
    IS = auto()
    NOT = auto()

    # Logical
    AND = auto()
    OR = auto()
    NEG = auto()  # !

    # Structural match
    TILDE = auto()  # ~
    REGEXMATCH = auto()  # ~~

    # Assignment
    ASSIGN = auto()  # =
    WALRUS = auto()  # :=
    APPLYASSIGN = auto()  # .=
    PLUSEQ = auto()
    MINUSEQ = auto()
    STAREQ = auto()
    SLASHEQ = auto()
    FLOORDIVEQ = auto()
    MODEQ = auto()
    POWEQ = auto()

    # Punctuation
    LPAR = auto()
    RPAR = auto()
    LSQB = auto()
    RSQB = auto()
    LBRACE = auto()
    RBRACE = auto()
    DOT = auto()
    COMMA = auto()
    COLON = auto()
    SEMI = auto()
    QMARK = auto()
    NULLISH = auto()  # ??
    AT = auto()
    DOLLAR = auto()  # $
    BACKQUOTE = auto()

    # Increment/Decrement
    INCR = auto()  # ++
    DECR = auto()  # --

    # Lambda
    AMP = auto()  # &

    # Pipe (for guard chains)
    PIPE = auto()  # |

    # Spread
    SPREAD = auto()  # ...

    # Special
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()
    COMMENT = auto()


@dataclass
class Tok:
    """Token with position info"""

    type: TT
    value: Any
    line: int = 0
    column: int = 0

    def __repr__(self):
        return f"Tok({self.type.name}, {self.value!r}, {self.line}:{self.column})"
