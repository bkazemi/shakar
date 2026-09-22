from __future__ import annotations

import pytest

from shakar_ref.types import ShkNumber
from tests.support.harness import ParseError, run_program


def test_runner_preserves_indented_parse_error() -> None:
    source = "try:\n    a := 1\n\n\n\n\ncatc"

    with pytest.raises(ParseError) as exc_info:
        run_program(source)

    err = exc_info.value
    assert err.line == 7
    assert err.column == 1
    assert "try requires a catch clause" in str(err)


# Indentation mode must recognize every newline form the lexer accepts;
# otherwise block bodies after the first statement fall out to top level.
@pytest.mark.parametrize(
    "newline",
    [
        pytest.param("\n", id="lf"),
        pytest.param("\r\n", id="crlf"),
        pytest.param("\r", id="cr-only"),
    ],
)
def test_runner_detects_multiline_for_all_newline_styles(newline: str) -> None:
    source = newline.join(["x := 0", "if false:", "  x = 1", "  x = 2", "x"])

    result = run_program(source)

    assert isinstance(result, ShkNumber)
    assert result.value == 0
