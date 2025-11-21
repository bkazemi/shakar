"""Shared helpers for working with Lark Tree/Token nodes used across the project."""
from __future__ import annotations
from typing import Any, Callable, Iterable, List, Optional, Set, TypeGuard, cast
from lark import Tree, Token
from typing_extensions import TypeAlias

TreeNode: TypeAlias = Tree[Any]

def is_tree(node: Any) -> TypeGuard[TreeNode]:
    return isinstance(node, Tree)

def is_tree_node(node: Any) -> TypeGuard[TreeNode]:
    return is_tree(node)

def is_token(node: Any) -> TypeGuard[Token]:
    return isinstance(node, Token)

def is_token_node(node: Any) -> TypeGuard[Token]:
    return is_token(node)

def tree_label(node: Any) -> Optional[str]:
    return node.data if is_tree(node) else None

def tree_children(node: Any) -> List[Any]:
    if not is_tree(node):
        return []

    children = getattr(node, "children", None)
    if children is None:
        return []

    return list(children)

def node_meta(node: Any) -> Any:
    return getattr(node, "meta", None)

def child_by_label(node: Any, label: str) -> Optional[Any]:
    for ch in tree_children(node):
        if tree_label(ch) == label:
            return ch

    return None

def child_by_labels(node: Any, labels: Iterable[str]) -> Optional[Any]:
    lookup: Set[str] = set(labels)

    for ch in tree_children(node):
        if tree_label(ch) in lookup:
            return ch

    return None

def first_child(node: Any, predicate: Callable[[Any], bool]) -> Optional[Any]:
    for ch in tree_children(node):
        if predicate(ch):
            return ch

    return None

def find_tree_by_label(node: Any, labels: Iterable[str]) -> Optional[TreeNode]:
    lookup: Set[str] = set(labels)

    if is_tree(node) and tree_label(node) in lookup:
        return cast(TreeNode, node)

    for child in tree_children(node):
        found = find_tree_by_label(child, lookup)
        if found is not None:
            return found

    return None
