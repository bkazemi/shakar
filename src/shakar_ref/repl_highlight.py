"""prompt_toolkit lexer for live Shakar syntax highlighting in the REPL."""

from __future__ import annotations

from typing import Callable

from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.lexers import Lexer

from .lexer_rd import Lexer as ShkLexer, LexError
from .token_types import TT, Tok

# Map highlight groups → prompt_toolkit style strings.
GROUP_STYLE = {
    "keyword": "bold ansicyan",
    "boolean": "ansicyan",
    "constant": "ansicyan",
    "number": "ansimagenta",
    "unit": "italic ansimagenta",
    "string": "ansigreen",
    "regex": "ansiyellow",
    "path": "ansiyellow",
    "identifier": "",
    "function": "bold ansiyellow",
    "decorator": "bold ansimagenta",
    "hook": "bold ansimagenta",
    "operator": "",
    "punctuation": "",
    "comment": "italic ansigray",
    "error": "bold ansired",
    "type": "bold ansiblue",
}

# Token type → highlight group (mirrors nvim-plugin/highlight_server.py TOKEN_GROUPS).
_TT_GROUP = {
    TT.IF: "keyword",
    TT.ELIF: "keyword",
    TT.ELSE: "keyword",
    TT.UNLESS: "keyword",
    TT.WHILE: "keyword",
    TT.FOR: "keyword",
    TT.IN: "keyword",
    TT.IMPORT: "keyword",
    TT.BREAK: "keyword",
    TT.CONTINUE: "keyword",
    TT.RETURN: "keyword",
    TT.FN: "keyword",
    TT.LET: "keyword",
    TT.WAIT: "keyword",
    TT.SPAWN: "keyword",
    TT.USING: "keyword",
    TT.BIND: "keyword",
    TT.CALL: "keyword",
    TT.DEFER: "keyword",
    TT.AFTER: "keyword",
    TT.THROW: "keyword",
    TT.CATCH: "keyword",
    TT.ASSERT: "keyword",
    TT.DBG: "keyword",
    TT.DECORATOR: "keyword",
    TT.GET: "keyword",
    TT.SET: "keyword",
    TT.HOOK: "keyword",
    TT.OVER: "keyword",
    TT.FAN: "keyword",
    TT.MATCH: "keyword",
    TT.NOT: "keyword",
    TT.AND: "keyword",
    TT.OR: "keyword",
    TT.IS: "keyword",
    TT.TRUE: "boolean",
    TT.FALSE: "boolean",
    TT.NIL: "constant",
    TT.NUMBER: "number",
    TT.DURATION: "number",
    TT.SIZE: "number",
    TT.STRING: "string",
    TT.RAW_STRING: "string",
    TT.RAW_HASH_STRING: "string",
    TT.SHELL_STRING: "string",
    TT.SHELL_BANG_STRING: "string",
    TT.REGEX: "regex",
    TT.PATH_STRING: "path",
    TT.ENV_STRING: "string",
    TT.IDENT: "identifier",
    TT.PLUS: "operator",
    TT.MINUS: "operator",
    TT.STAR: "operator",
    TT.SLASH: "operator",
    TT.FLOORDIV: "operator",
    TT.MOD: "operator",
    TT.POW: "operator",
    TT.DEEPMERGE: "operator",
    TT.CARET: "operator",
    TT.EQ: "operator",
    TT.NEQ: "operator",
    TT.LT: "operator",
    TT.LTE: "operator",
    TT.GT: "operator",
    TT.GTE: "operator",
    TT.SEND: "operator",
    TT.RECV: "operator",
    TT.NEG: "operator",
    TT.NULLISH: "operator",
    TT.ASSIGN: "operator",
    TT.WALRUS: "operator",
    TT.APPLYASSIGN: "operator",
    TT.PLUSEQ: "operator",
    TT.MINUSEQ: "operator",
    TT.STAREQ: "operator",
    TT.SLASHEQ: "operator",
    TT.FLOORDIVEQ: "operator",
    TT.MODEQ: "operator",
    TT.POWEQ: "operator",
    TT.INCR: "operator",
    TT.DECR: "operator",
    TT.TILDE: "operator",
    TT.REGEXMATCH: "operator",
    TT.AMP: "operator",
    TT.PIPE: "operator",
    TT.LPAR: "punctuation",
    TT.RPAR: "punctuation",
    TT.LSQB: "punctuation",
    TT.RSQB: "punctuation",
    TT.LBRACE: "punctuation",
    TT.RBRACE: "punctuation",
    TT.DOT: "punctuation",
    TT.COMMA: "punctuation",
    TT.COLON: "punctuation",
    TT.SEMI: "punctuation",
    TT.QMARK: "punctuation",
    TT.AT: "punctuation",
    TT.DOLLAR: "punctuation",
    TT.BACKQUOTE: "punctuation",
    TT.COMMENT: "comment",
    TT.SPREAD: "operator",
}

_DUR_SIZE_TT = {TT.DURATION, TT.SIZE}
_LAYOUT = {TT.NEWLINE, TT.INDENT, TT.DEDENT, TT.EOF}
_SIG_SKIP = _LAYOUT | {TT.COMMENT}
_SLOT_HEADS = {TT.WAIT, TT.USING, TT.CALL, TT.FAN}


def _prev_sig_idx(tokens: list[Tok], idx: int) -> int:
    j = idx - 1
    while j >= 0:
        tok = tokens[j]
        if tok.type not in _SIG_SKIP:
            return j
        j -= 1
    return -1


def _next_sig_idx(tokens: list[Tok], idx: int) -> int:
    j = idx + 1
    while j < len(tokens):
        tok = tokens[j]
        if tok.type not in _SIG_SKIP:
            return j
        j += 1
    return -1


def _is_bracket_slot_ident(tokens: list[Tok], idx: int) -> bool:
    prev_idx = _prev_sig_idx(tokens, idx)
    next_idx = _next_sig_idx(tokens, idx)
    if prev_idx < 0 or next_idx < 0:
        return False

    prev_tok = tokens[prev_idx]
    next_tok = tokens[next_idx]
    if prev_tok.type != TT.LSQB:
        return False
    if next_tok.type != TT.RSQB:
        return False

    head_idx = _prev_sig_idx(tokens, prev_idx)
    if head_idx < 0:
        return False
    head_tok = tokens[head_idx]
    return head_tok.type in _SLOT_HEADS


def _dur_size_spans(tok_text: str) -> StyleAndTextTuples:
    """Split a duration/size literal into number + unit sub-spans.
    Mirrors emit_dur_size_spans() in the C highlighter."""
    num_style = GROUP_STYLE["number"]
    unit_style = GROUP_STYLE["unit"]
    spans: StyleAndTextTuples = []
    pos = 0
    n = len(tok_text)

    while pos < n:
        # Number part: digits, underscores, decimal point, exponent.
        num_start = pos
        while pos < n and tok_text[pos] in "0123456789_.eE+-":
            ch = tok_text[pos]
            if ch in "+-" and pos > 0 and tok_text[pos - 1] not in "eE":
                break
            if ch in "eE" and pos > 0 and tok_text[pos - 1] not in "0123456789_":
                break
            pos += 1

        # Unit part: lowercase alpha.
        unit_start = pos
        while pos < n and tok_text[pos].islower():
            pos += 1

        if pos > unit_start and num_start < unit_start:
            spans.append((num_style, tok_text[num_start:unit_start]))
            spans.append((unit_style, tok_text[unit_start:pos]))
        else:
            # Safety: skip one char to avoid infinite loop.
            if pos == num_start:
                pos += 1

    return spans


def _highlight_line(text: str) -> StyleAndTextTuples:
    """Tokenize a single line and return styled fragments."""
    if not text:
        return [("", "")]

    try:
        lexer = ShkLexer(text, track_indentation=False, emit_comments=True)
        tokens = lexer.tokenize()
    except LexError:
        return [("", text)]

    result: StyleAndTextTuples = []
    pos = 0

    for i, tok in enumerate(tokens):
        if tok.type in _LAYOUT:
            continue
        tok_start = tok.column
        # Duration/size tokens store (literal_text, computed_value).
        raw = tok.value
        if isinstance(raw, tuple):
            raw = raw[0]
        tok_text = str(raw) if raw is not None else ""
        if not tok_text:
            continue

        # Find actual position of this token value in the line from pos onwards.
        idx = text.find(tok_text, pos)
        if idx < 0:
            continue

        # Unstyled gap before token.
        if idx > pos:
            result.append(("", text[pos:idx]))

        # Duration/size tokens: split into number + unit sub-spans.
        if tok.type in _DUR_SIZE_TT:
            result.extend(_dur_size_spans(tok_text))
        else:
            group = _TT_GROUP.get(tok.type, "")
            if tok.type == TT.IDENT and _is_bracket_slot_ident(tokens, i):
                group = "keyword"
            style = GROUP_STYLE.get(group, "")
            result.append((style, tok_text))
        pos = idx + len(tok_text)

    # Trailing unstyled text.
    if pos < len(text):
        result.append(("", text[pos:]))

    return result if result else [("", text)]


class ShakarLexer(Lexer):
    """prompt_toolkit Lexer that highlights Shakar source using the RD lexer."""

    def lex_document(self, document: Document) -> Callable[[int], StyleAndTextTuples]:
        lines = document.lines

        # Pre-compute highlights for all lines.
        cache: dict[int, StyleAndTextTuples] = {}

        def get_line(lineno: int) -> StyleAndTextTuples:
            if lineno not in cache:
                if lineno < len(lines):
                    cache[lineno] = _highlight_line(lines[lineno])
                else:
                    cache[lineno] = [("", "")]

            return cache[lineno]

        return get_line
