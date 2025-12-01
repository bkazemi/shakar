"""Shared helpers for working with Tree/Token nodes used across the project.

These are minimal AST node classes compatible with Lark's interface, allowing
the codebase to work without the Lark dependency.
"""
from __future__ import annotations
from typing import Any, Callable, Iterable, List, Optional, Set, TypeGuard, Union, cast
from typing_extensions import TypeAlias


class Token(str):
    """Minimal Token class compatible with Lark's Token.

    Subclasses str so tokens can be used as strings (for backward compat).
    """
    __slots__ = ('type', 'value', 'line', 'column', 'end_line', 'end_column', 'start_pos', 'end_pos')

    def __new__(cls, type_: str, value: str, line: int = 1, column: int = 1,
                end_line: Optional[int] = None, end_column: Optional[int] = None,
                start_pos: Optional[int] = None, end_pos: Optional[int] = None):
        inst = str.__new__(cls, value)
        inst.type = type_
        inst.value = value
        inst.line = line
        inst.column = column
        inst.end_line = end_line if end_line is not None else line
        inst.end_column = end_column if end_column is not None else column + len(value)
        inst.start_pos = start_pos if start_pos is not None else 0
        inst.end_pos = end_pos if end_pos is not None else start_pos + len(value) if start_pos is not None else len(value)
        return inst

    @classmethod
    def new_borrow_pos(cls, type_: str, value: str, borrow_token: Token) -> Token:
        """Create token borrowing position info from another token."""
        return cls(type_, value,
                   borrow_token.line, borrow_token.column,
                   borrow_token.end_line, borrow_token.end_column,
                   borrow_token.start_pos, borrow_token.end_pos)

    def update(self, type_: Optional[str] = None, value: Optional[str] = None) -> Token:
        """Return new token with updated type and/or value."""
        return Token(
            type_ if type_ is not None else self.type,
            value if value is not None else self.value,
            self.line, self.column, self.end_line, self.end_column,
            self.start_pos, self.end_pos
        )

    def __repr__(self) -> str:
        return f'Token({self.type!r}, {self.value!r})'


class Tree:
    """Minimal Tree class compatible with Lark's Tree."""
    __slots__ = ('data', 'children', 'meta', '_meta')

    def __init__(self, data: str, children: List[Union[Tree, Token]], meta: Optional[Any] = None):
        self.data = data
        self.children = children
        self.meta = meta
        self._meta = None  # Lark compatibility

    def __repr__(self) -> str:
        return f'Tree({self.data!r}, {self.children!r})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tree):
            return False
        return self.data == other.data and self.children == other.children

    def __hash__(self) -> int:
        return hash((self.data, tuple(self.children)))

    def pretty(self, indent: str = '  ') -> str:
        """Return pretty-printed tree representation."""
        def _pretty(node: Union[Tree, Token], level: int = 0) -> str:
            if isinstance(node, Token):
                return f'{indent * level}{node.type}\t{node.value!r}\n'
            lines = [f'{indent * level}{node.data}\n']
            for child in node.children:
                lines.append(_pretty(child, level + 1))
            return ''.join(lines)
        return _pretty(self)


Node: TypeAlias = Tree | Token


class Discard:
    """Sentinel value to signal that a node should be removed from the tree."""
    pass


class Transformer:
    """Base class for tree transformers (compatible with Lark's Transformer)."""

    def transform(self, tree: Tree) -> Union[Tree, Token, Discard]:
        """Transform a tree by recursively visiting nodes."""
        return self._transform_tree(tree)

    def _transform_tree(self, tree: Tree) -> Union[Tree, Token, Discard]:
        """Recursively transform tree nodes."""
        # Transform children first (bottom-up)
        new_children = []
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
            if hasattr(method, '_v_args_meta') and method._v_args_meta:
                # Pass meta as first arg, then children list
                result = method(new_tree.meta, new_children)
            elif hasattr(method, '_v_args_inline') and method._v_args_inline:
                # Pass children as separate arguments (*children)
                result = method(*new_children)
            else:
                # Default: Pass children as single list argument (Lark convention)
                result = method(new_children)

            return result if result is not None else new_tree

        return new_tree


def v_args(inline: bool = False, meta: bool = False, tree: bool = False):
    """Decorator for transformer methods (Lark compatibility)."""
    def decorator(func: Callable) -> Callable:
        # Mark the function with metadata about how it should receive args
        func._v_args_inline = inline
        func._v_args_meta = meta
        func._v_args_tree = tree
        return func
    return decorator


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

def node_meta(node: Node) -> Optional[Node]:
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
