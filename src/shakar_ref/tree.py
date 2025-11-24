"""Shared helpers for working with Lark Tree/Token nodes used across the project."""
from __future__ import annotations
from typing import Callable, Iterable, List, Optional, Set, TypeGuard, cast
from lark import Tree, Token
from typing_extensions import TypeAlias

Node: TypeAlias = Tree | Token

def is_tree(node: Node) -> TypeGuard[Tree]:
    return isinstance(node, Tree)

def is_token(node: Node) -> TypeGuard[Token]:
    return isinstance(node, Token)

def tree_label(node: Node) -> Optional[str]:
    return node.data if is_tree(node) else None

def tree_children(node: Node) -> List[Node]:
    if not is_tree(node):
        return []

    children = getattr(node, "children", None)
    if children is None:
        return []

    return list(children)

def node_meta(node: Node) -> object | None:
    return getattr(node, "meta", None)

def child_by_label(node: Node, label: str) -> Optional[Node]:
    for ch in tree_children(node):
        if tree_label(ch) == label:
            return ch

    return None

def child_by_labels(node: Node, labels: Iterable[str]) -> Optional[Node]:
    lookup: Set[str] = set(labels)

    for ch in tree_children(node):
        if tree_label(ch) in lookup:
            return ch

    return None

def first_child(node: Node, predicate: Callable[[Node], bool]) -> Optional[Node]:
    for ch in tree_children(node):
        if predicate(ch):
            return ch

    return None

def find_tree_by_label(node: Node, labels: Iterable[str]) -> Optional[Tree]:
    lookup: Set[str] = set(labels)

    if is_tree(node) and tree_label(node) in lookup:
        return node

    for child in tree_children(node):
        found = find_tree_by_label(child, lookup)
        if found is not None:
            return found

    return None
