"""Shared helpers for working with Tree/Tok nodes used across the project."""

from __future__ import annotations
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    TypeGuard,
    Union,
)
from typing_extensions import TypeAlias

from .token_types import Tok, TT

# Alias for Tok
Token = Tok

__all__ = [
    "Tree",
    "Tok",
    "Token",
    "Node",
    "Transformer",
    "Discard",
    "v_args",
    "is_tree",
    "is_token",
    "tree_label",
    "tree_children",
    "node_meta",
    "child_by_label",
    "child_by_labels",
    "first_child",
    "find_tree_by_label",
    "ident_value",
    "token_kind",
]


class Tree:
    """Minimal Tree class used across the runtime."""

    __slots__ = ("data", "children", "meta", "_meta")

    def __init__(
        self, data: str, children: Sequence["Node"], meta: Optional[Any] = None
    ):
        self.data = data
        self.children: List[Node] = list(children)
        self.meta = meta
        self._meta = None  # Optional metadata slot

    def __repr__(self) -> str:
        return f"Tree({self.data!r}, {self.children!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tree):
            return False
        return self.data == other.data and self.children == other.children

    def __hash__(self) -> int:
        return hash((self.data, tuple(self.children)))

    def pretty(self, indent: str = "  ") -> str:
        """Return pretty-printed tree representation."""

        def _pretty(node: Node, level: int = 0) -> str:
            if isinstance(node, Tok):
                typ = node.type.name if hasattr(node.type, "name") else str(node.type)
                return f"{indent * level}{typ}\t{node.value!r}\n"
            lines = [f"{indent * level}{node.data}\n"]
            for child in node.children:
                lines.append(_pretty(child, level + 1))
            return "".join(lines)

        return _pretty(self)


Node: TypeAlias = Tree | Tok


class Discard:
    """Sentinel value to signal that a node should be removed from the tree."""

    pass


class Transformer:
    """Base class for tree transformers."""

    def transform(self, tree: Tree) -> Union[Node, Discard]:
        """Transform a tree by recursively visiting nodes."""
        return self._transform_tree(tree)

    def _transform_tree(self, tree: Tree) -> Union[Node, Discard]:
        """Recursively transform tree nodes."""
        # Transform children first (bottom-up)
        new_children: List[Node] = []
        for child in tree.children:
            if isinstance(child, Tree):
                result = self._transform_tree(child)
            else:
                result = child

            if not isinstance(result, Discard):
                new_children.append(result)

        # Create new tree with transformed children
        new_tree = Tree(tree.data, new_children, tree.meta)

        # Try to find and call a method for this rule
        method_name = tree.data
        if hasattr(self, method_name):
            method = getattr(self, method_name)

            # Check if method has v_args decorators
            if getattr(method, "_v_args_meta", False):
                result = method(new_tree.meta, new_children)
            elif getattr(method, "_v_args_inline", False):
                result = method(*new_children)
            else:
                result = method(new_children)

            return result if result is not None else new_tree

        return new_tree


def v_args(
    inline: bool = False, meta: bool = False, tree: bool = False
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for transformer methods."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func._v_args_inline = inline  # type: ignore[attr-defined]
        func._v_args_meta = meta  # type: ignore[attr-defined]
        func._v_args_tree = tree  # type: ignore[attr-defined]
        return func

    return decorator


def is_tree(node: Node) -> TypeGuard[Tree]:
    return isinstance(node, Tree)


def is_token(node: Node) -> TypeGuard[Tok]:
    return isinstance(node, Tok)


def tree_label(node: Node) -> Optional[str]:
    return node.data if is_tree(node) else None


def tree_children(node: Node) -> List[Node]:
    if not is_tree(node):
        return []
    return list(node.children)


def node_meta(node: Node) -> Optional[Any]:
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


def token_kind(node: Node) -> Optional[str]:
    """Return the token type name if node is a token, else None."""
    if not is_token(node):
        return None
    tok: Tok = node
    return tok.type.name if hasattr(tok.type, "name") else str(tok.type)


def ident_value(node: Node) -> Optional[str]:
    """Return identifier value if node is IDENT token, else None."""
    if is_token(node) and node.type == TT.IDENT:
        return str(node.value)
    return None
