from __future__ import annotations

import pytest

from tests.support.harness import (
    build_keyword_cases,
    build_keyword_plan,
    parse_both_modes,
)

KEYWORD_CASES = build_keyword_cases(build_keyword_plan())


@pytest.mark.parametrize(
    "code, start",
    [pytest.param(code, start, id=name) for name, code, start in KEYWORD_CASES],
)
def test_parser_keywords(code: str, start: str) -> None:
    parse_both_modes(code, start)
