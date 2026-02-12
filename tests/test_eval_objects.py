from __future__ import annotations

from typing import cast

import pytest

from shakar_ref.eval.objects import eval_key
from shakar_ref.runtime import Frame, ShkValue
from shakar_ref.token_types import TT, Tok
from shakar_ref.tree import Node, Tree


def _unused_eval(_node: Node, _frame: Frame) -> ShkValue:
    raise AssertionError("eval_key token fallback should not call eval_fn")


@pytest.mark.parametrize(
    "token, expected",
    [
        pytest.param(Tok(TT.IDENT, "name"), "name", id="ident-token"),
        pytest.param(Tok(TT.STRING, '"alpha"'), "alpha", id="string-basic"),
        pytest.param(Tok(TT.STRING, '""quoted""'), '"quoted"', id="string-one-pair"),
        pytest.param(Tok(TT.STRING, '"unterminated'), '"unterminated', id="string-raw"),
    ],
)
def test_eval_key_token_fallback(token: Tok, expected: str) -> None:
    frame = Frame()
    result = eval_key(cast(Tree, token), frame, _unused_eval)
    assert result == expected
