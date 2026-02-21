from __future__ import annotations

import pytest

from tests.support.harness import ParseError, run_program
from shakar_ref.runner import _error_rank


def test_runner_prefers_furthest_parse_error() -> None:
    source = "try:\n    a := 1\n\n\n\n\ncatc"

    with pytest.raises(ParseError) as exc_info:
        run_program(source)

    err = exc_info.value
    assert err.line == 7
    assert err.column == 1
    assert "try requires a catch clause" in str(err)


def test_error_rank_falls_back_as_position_unit() -> None:
    class DummyExc(Exception):
        pass

    exc = DummyExc("x")
    exc.end_line = 10
    exc.end_column = None
    exc.line = 3
    exc.column = 4

    assert _error_rank(exc) == (1, 3, 4)


def test_error_rank_accepts_zero_column() -> None:
    class DummyExc(Exception):
        pass

    exc = DummyExc("x")
    exc.end_line = 8
    exc.end_column = 0
    exc.line = None
    exc.column = None

    assert _error_rank(exc) == (1, 8, 0)
