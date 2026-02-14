from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from shakar_ref.lexer_rd import LexError, TT, tokenize


@dataclass(frozen=True)
class Case:
    """Unified lexer case payload."""

    name: str
    source: str
    expected: Optional[Tuple[Tuple[TT, object], ...]] = None
    expected_types: Optional[Tuple[TT, ...]] = None
    expected_lines: Optional[Tuple[Tuple[str, int], ...]] = None
    exc: Optional[type[Exception]] = None
    msg: Optional[str] = None
    err_line: Optional[int] = None
    err_col: Optional[int] = None
    track_indentation: bool = False


BASIC_TOKEN_CASES: List[Case] = [
    Case("number-int", "123", expected=((TT.NUMBER, "123"),)),
    Case("number-float", "3.14", expected=((TT.NUMBER, "3.14"),)),
    Case("ident-single", "x", expected=((TT.IDENT, "x"),)),
    Case("ident-snake", "foo_bar", expected=((TT.IDENT, "foo_bar"),)),
    Case("string-double", '"hello"', expected=((TT.STRING, '"hello"'),)),
    Case("string-single", "'world'", expected=((TT.STRING, "'world'"),)),
    Case("shell-string", 'sh"echo"', expected=((TT.SHELL_STRING, "echo"),)),
    Case("shell-bang-string", 'sh!"echo"', expected=((TT.SHELL_BANG_STRING, "echo"),)),
    Case("path-string", 'p"/tmp/logs"', expected=((TT.PATH_STRING, 'p"/tmp/logs"'),)),
    Case("regex-literal", 'r"ab+"/im', expected=((TT.REGEX, ("ab+", "im")),)),
    Case("bool-true", "true", expected=((TT.TRUE, "true"),)),
    Case("bool-false", "false", expected=((TT.FALSE, "false"),)),
    Case("nil-literal", "nil", expected=((TT.NIL, "nil"),)),
]

OPERATOR_CASES: List[Case] = [
    Case("plus", "+", expected_types=(TT.PLUS,)),
    Case("minus", "-", expected_types=(TT.MINUS,)),
    Case("star", "*", expected_types=(TT.STAR,)),
    Case("slash", "/", expected_types=(TT.SLASH,)),
    Case("pow", "**", expected_types=(TT.POW,)),
    Case("floordiv", "//", expected_types=(TT.FLOORDIV,)),
    Case("eq", "==", expected_types=(TT.EQ,)),
    Case("neq", "!=", expected_types=(TT.NEQ,)),
    Case("send", "->", expected_types=(TT.SEND,)),
    Case("recv", "<-", expected_types=(TT.RECV,)),
    Case("lte", "<=", expected_types=(TT.LTE,)),
    Case("gte", ">=", expected_types=(TT.GTE,)),
    Case("lt", "<", expected_types=(TT.LT,)),
    Case("gt", ">", expected_types=(TT.GT,)),
    Case("regexmatch", "~~", expected_types=(TT.REGEXMATCH,)),
    Case("walrus", ":=", expected_types=(TT.WALRUS,)),
    Case("applyassign", ".=", expected_types=(TT.APPLYASSIGN,)),
    Case("pluseq", "+=", expected_types=(TT.PLUSEQ,)),
    Case("incr", "++", expected_types=(TT.INCR,)),
    Case("decr", "--", expected_types=(TT.DECR,)),
    Case("nullish", "??", expected_types=(TT.NULLISH,)),
    Case("deepmerge", "+>", expected_types=(TT.DEEPMERGE,)),
    Case("amp", "&", expected_types=(TT.AMP,)),
    Case("pipe", "|", expected_types=(TT.PIPE,)),
    Case("or", "||", expected_types=(TT.OR,)),
    Case("and", "&&", expected_types=(TT.AND,)),
]

KEYWORD_CASES: List[Case] = [
    Case("if", "if", expected_types=(TT.IF,)),
    Case("while", "while", expected_types=(TT.WHILE,)),
    Case("for", "for", expected_types=(TT.FOR,)),
    Case("fn", "fn", expected_types=(TT.FN,)),
    Case("return", "return", expected_types=(TT.RETURN,)),
    Case("wait", "wait", expected_types=(TT.WAIT,)),
    Case("spawn", "spawn", expected_types=(TT.SPAWN,)),
    Case("and", "and", expected_types=(TT.AND,)),
    Case("or", "or", expected_types=(TT.OR,)),
    Case("not", "not", expected_types=(TT.NOT,)),
    Case("is", "is", expected_types=(TT.IS,)),
    Case("in", "in", expected_types=(TT.IN,)),
    Case("import", "import", expected_types=(TT.IMPORT,)),
    Case("any-ident", "any", expected_types=(TT.IDENT,)),
    Case("all-ident", "all", expected_types=(TT.IDENT,)),
    Case("group-ident", "group", expected_types=(TT.IDENT,)),
    Case("over", "over", expected_types=(TT.OVER,)),
]

SHAKAR_CONSTRUCT_CASES: List[Case] = [
    Case("lambda", "&(x)", expected_types=(TT.AMP, TT.LPAR, TT.IDENT, TT.RPAR)),
    Case("walrus", "x := 5", expected_types=(TT.IDENT, TT.WALRUS, TT.NUMBER)),
    Case(
        "guard-chain", "x > 5 :", expected_types=(TT.IDENT, TT.GT, TT.NUMBER, TT.COLON)
    ),
    Case(
        "fan",
        ".{a, b}",
        expected_types=(TT.DOT, TT.LBRACE, TT.IDENT, TT.COMMA, TT.IDENT, TT.RBRACE),
    ),
    Case(
        "destructuring",
        "a, b, c",
        expected_types=(TT.IDENT, TT.COMMA, TT.IDENT, TT.COMMA, TT.IDENT),
    ),
    Case("nullish", "x ?? 0", expected_types=(TT.IDENT, TT.NULLISH, TT.NUMBER)),
    Case("deepmerge", "a +> b", expected_types=(TT.IDENT, TT.DEEPMERGE, TT.IDENT)),
]

STRING_ESCAPE_CASES: List[Case] = [
    Case("newline", r'"hello\nworld"', expected=((TT.STRING, r'"hello\nworld"'),)),
    Case("tab", r'"tab\there"', expected=((TT.STRING, r'"tab\there"'),)),
    Case("quote", r'"quote\"here"', expected=((TT.STRING, r'"quote\"here"'),)),
    Case("backslash", r'"backslash\\"', expected=((TT.STRING, r'"backslash\\"'),)),
    Case("hex", '"hex\\x1b"', expected=((TT.STRING, '"hex\\x1b"'),)),
]

POSITION_CASES: List[Case] = [
    Case("simple-lines", "x\ny\n  z", expected_lines=(("x", 1), ("y", 2), ("z", 3))),
]

LEX_ERROR_CASES: List[Case] = [
    Case(
        "unterminated-string",
        '"abc',
        exc=LexError,
        msg="Unterminated string",
        err_line=1,
        err_col=1,
    ),
    Case(
        "unterminated-shell",
        'sh"echo',
        exc=LexError,
        msg="Unterminated shell string",
        err_line=1,
        err_col=1,
    ),
    Case(
        "unterminated-path",
        'p"/tmp',
        exc=LexError,
        msg="Unterminated path string",
        err_line=1,
        err_col=1,
    ),
    Case(
        "regex-unknown-flag",
        'r"a"/z',
        exc=LexError,
        msg="Unknown regex flag",
        err_line=1,
        err_col=6,
    ),
    Case(
        "invalid-underscore",
        "1__0",
        exc=LexError,
        msg="Invalid underscore in number literal",
        err_line=1,
        err_col=1,
    ),
    Case(
        "trailing-underscore",
        "100_",
        exc=LexError,
        msg="Trailing underscore in number literal",
        err_line=1,
        err_col=1,
    ),
    Case(
        "invalid-exponent-underscore",
        "1e_5",
        exc=LexError,
        msg="Leading underscore in exponent",
        err_line=1,
        err_col=1,
    ),
    Case(
        "indentation-mismatch",
        "if true:\n  x\n y\n",
        exc=LexError,
        msg="Indentation mismatch",
        err_line=3,
        track_indentation=True,
    ),
    # Multi-line: error on line 2
    Case(
        "unterminated-string-line2",
        'x = 1\ny = "abc',
        exc=LexError,
        msg="Unterminated string",
        err_line=2,
        err_col=5,
    ),
    # Error with col offset on same line
    Case(
        "unexpected-char",
        "x = \\",
        exc=LexError,
        msg="Unexpected character",
        err_line=1,
        err_col=5,
    ),
    Case(
        "invalid-suffix",
        "123abc",
        exc=LexError,
        msg="Invalid number suffix",
        err_line=1,
        err_col=1,
    ),
]


def _non_eof_tokens(source: str) -> List[object]:
    return [token for token in tokenize(source) if token.type != TT.EOF]


@pytest.mark.parametrize("case", BASIC_TOKEN_CASES, ids=lambda case: case.name)
def test_basic_tokens(case: Case) -> None:
    tokens = _non_eof_tokens(case.source)

    assert case.expected is not None
    assert len(tokens) == len(case.expected)
    for token, (expected_type, expected_value) in zip(tokens, case.expected):
        assert token.type == expected_type
        assert token.value == expected_value


@pytest.mark.parametrize("case", OPERATOR_CASES, ids=lambda case: case.name)
def test_operators(case: Case) -> None:
    tokens = _non_eof_tokens(case.source)
    assert case.expected_types is not None
    assert [token.type for token in tokens] == list(case.expected_types)


@pytest.mark.parametrize("case", KEYWORD_CASES, ids=lambda case: case.name)
def test_keywords(case: Case) -> None:
    tokens = _non_eof_tokens(case.source)
    assert case.expected_types is not None
    assert [token.type for token in tokens] == list(case.expected_types)


@pytest.mark.parametrize("case", SHAKAR_CONSTRUCT_CASES, ids=lambda case: case.name)
def test_shakar_constructs(case: Case) -> None:
    tokens = _non_eof_tokens(case.source)
    assert case.expected_types is not None
    assert [token.type for token in tokens] == list(case.expected_types)


@pytest.mark.parametrize("case", STRING_ESCAPE_CASES, ids=lambda case: case.name)
def test_string_escapes(case: Case) -> None:
    tokens = [token for token in tokenize(case.source) if token.type == TT.STRING]
    assert case.expected is not None
    assert len(tokens) == 1
    assert len(case.expected) == 1
    assert tokens[0].value == case.expected[0][1]


def test_comments() -> None:
    source = "x = 5  # This is a comment\ny = 10"
    tokens = tokenize(source)
    filtered = [token for token in tokens if token.type not in (TT.EOF, TT.NEWLINE)]

    expected_types = [TT.IDENT, TT.ASSIGN, TT.NUMBER, TT.IDENT, TT.ASSIGN, TT.NUMBER]
    assert [token.type for token in filtered] == expected_types


@pytest.mark.parametrize("case", POSITION_CASES, ids=lambda case: case.name)
def test_position_tracking(case: Case) -> None:
    assert case.expected_lines is not None
    tokens = tokenize(case.source)
    actual_lines: Dict[str, int] = {}
    for token in tokens:
        if token.value in dict(case.expected_lines):
            actual_lines[str(token.value)] = token.line

    for value, expected_line in case.expected_lines:
        assert value in actual_lines
        assert actual_lines[value] == expected_line


@pytest.mark.parametrize("case", LEX_ERROR_CASES, ids=lambda case: case.name)
def test_lex_errors(case: Case) -> None:
    assert case.exc is not None
    assert case.msg is not None
    with pytest.raises(case.exc) as exc_info:
        tokenize(case.source, track_indentation=case.track_indentation)

    err = exc_info.value
    assert case.msg in str(err)

    if case.err_line is not None:
        assert (
            err.line == case.err_line
        ), f"expected line {case.err_line}, got {err.line}"
    if case.err_col is not None:
        assert (
            err.column == case.err_col
        ), f"expected col {case.err_col}, got {err.column}"


def test_tetris_file() -> None:
    tetris_path = Path(__file__).resolve().parent.parent / "docs" / "tetris_shakar.shk"
    if not tetris_path.exists():
        pytest.skip("tetris_shakar.shk not found")

    source = tetris_path.read_text(encoding="utf-8")
    tokens = tokenize(source, track_indentation=False)

    assert len(tokens) > 1000
    assert any(token.type == TT.FN for token in tokens)
    assert any(token.type == TT.WALRUS for token in tokens)
    assert any(token.type == TT.AMP for token in tokens)
    assert any(token.type == TT.PIPE for token in tokens)
    assert all(token.type != TT.COMMENT for token in tokens)
