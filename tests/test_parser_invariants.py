from __future__ import annotations

from typing import Dict, List, Tuple

import pytest

from shakar_ref.token_types import TT
from shakar_ref.tree import Tok, Tree
from tests.support.harness import parse_pipeline


def _strip_start_root(tree: Tree) -> Tree:
    if (
        tree.data in {"start_noindent", "start_indented"}
        and len(tree.children) == 1
        and isinstance(tree.children[0], Tree)
    ):
        return tree.children[0]
    return tree


def _parse_stmt_both_modes(source: str) -> Tree:
    noindent = parse_pipeline(source, use_indenter=False)
    indented = parse_pipeline(source, use_indenter=True)
    assert isinstance(noindent, Tree)
    assert isinstance(indented, Tree)

    noindent_root = _strip_start_root(noindent)
    indented_root = _strip_start_root(indented)
    assert noindent_root == indented_root

    assert noindent_root.data == "stmtlist"
    assert noindent_root.children

    stmt = noindent_root.children[0]
    assert isinstance(stmt, Tree)
    return stmt


def _count_nodes(tree: Tree, name: str) -> int:
    count = 0

    def walk(node: object) -> None:
        nonlocal count
        if not isinstance(node, Tree):
            return
        if node.data == name:
            count += 1
        for child in node.children:
            walk(child)

    walk(tree)
    return count


CROSS_START_CASES: List[str] = [
    "f(1,2,3)",
    "a !is b",
    "a not in b",
    "x := (a)",
    "a .= b",
    "a and b and c",
]


@pytest.mark.parametrize("source", CROSS_START_CASES)
def test_cross_start_normalized_equivalence(source: str) -> None:
    _parse_stmt_both_modes(source)


def test_bind_is_right_associative() -> None:
    stmt = _parse_stmt_both_modes("a .= b .= c")

    assert stmt.data == "bind"
    assert _count_nodes(stmt, "bind") == 2

    assert isinstance(stmt.children[0], Tree)
    assert stmt.children[0].data == "lvalue"

    nested = stmt.children[1]
    assert isinstance(nested, Tree)
    assert nested.data == "bind"

    assert isinstance(nested.children[0], Tree)
    assert nested.children[0].data == "lvalue"

    tail = nested.children[1]
    assert isinstance(tail, Tok)
    assert tail.type == TT.IDENT
    assert tail.value == "c"


def test_walrus_binds_outside_nullish_chain() -> None:
    stmt = _parse_stmt_both_modes("x := a ?? b ?? c")

    assert stmt.data == "walrus"
    assert isinstance(stmt.children[1], Tree)
    assert stmt.children[1].data == "nullish"


def test_ternary_is_right_associative() -> None:
    stmt = _parse_stmt_both_modes("1 ? 2 : 3 ? 4 : 5")

    assert stmt.data == "ternary"
    assert isinstance(stmt.children[2], Tree)
    assert stmt.children[2].data == "ternary"


def test_and_binds_tighter_than_or() -> None:
    stmt = _parse_stmt_both_modes("a and b or c")

    assert stmt.data == "or"
    assert isinstance(stmt.children[0], Tree)
    assert stmt.children[0].data == "and"


def test_postfix_if_node_shape() -> None:
    stmt = _parse_stmt_both_modes("1 if 0")

    assert stmt.data == "postfixif"
    semantic_children = [
        child
        for child in stmt.children
        if not (isinstance(child, Tok) and child.type == TT.IF)
    ]
    assert len(semantic_children) == 2


COMPARE_CASES: Dict[str, Tuple[TT, TT]] = {
    "a not in b": (TT.NOT, TT.IN),
    "a !in b": (TT.NEG, TT.IN),
    "a is not b": (TT.IS, TT.NOT),
    "a !is b": (TT.NEG, TT.IS),
}


@pytest.mark.parametrize(
    "source, expected_cmp_toks",
    [pytest.param(source, toks, id=source) for source, toks in COMPARE_CASES.items()],
)
def test_compare_variant_cmpop_tokens(
    source: str, expected_cmp_toks: Tuple[TT, TT]
) -> None:
    stmt = _parse_stmt_both_modes(source)

    assert stmt.data == "compare"
    cmpop = next(
        (
            child
            for child in stmt.children
            if isinstance(child, Tree) and child.data == "cmpop"
        ),
        None,
    )
    assert cmpop is not None

    actual = tuple(tok.type for tok in cmpop.children if isinstance(tok, Tok))
    assert actual == expected_cmp_toks
