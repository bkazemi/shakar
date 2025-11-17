from __future__ import annotations
from typing import Any, List

from shakar_tree import (
    Tree,
    Token,
    is_tree,
    tree_children,
    tree_label,
)
from shakar_tree import is_token as is_token_node

def lower(ast: Any) -> Any:
    """Runtime lowering pass (currently: hole desugaring)."""
    return _desugar_call_holes(ast)

def _desugar_call_holes(node: Any) -> Any:
    if is_token_node(node) or not is_tree(node):
        return node

    children = getattr(node, "children", [])

    for idx, child in enumerate(list(children)):
        lowered = _desugar_call_holes(child)

        if lowered is not child:
            children[idx] = lowered

    candidate = node

    if tree_label(candidate) == 'explicit_chain':
        replacement = _chain_to_lambda_if_holes(candidate)

        if replacement is not None:
            return replacement

    return candidate

def _chain_to_lambda_if_holes(chain: Any) -> Any | None:
    def _contains_hole(node: Any) -> bool:
        if is_token_node(node) or not is_tree(node):
            return False

        if tree_label(node) == 'holeexpr':
            return True
        return any(_contains_hole(child) for child in tree_children(node))

    holes: List[str] = []

    children = tree_children(chain)
    if not children:
        return None

    ops = children[1:]
    hole_call_index = None

    for idx, op in enumerate(ops):
        if tree_label(op) == 'call' and _contains_hole(op):
            hole_call_index = idx
            break

    if hole_call_index is not None and hole_call_index + 1 < len(ops):
        raise SyntaxError("Hole partials cannot be immediately invoked; assign or pass the partial before calling it")

    def clone(node: Any) -> Any:
        if is_token_node(node) or not is_tree(node):
            return node

        label = tree_label(node)
        if label == 'holeexpr':
            name = f"_hole{len(holes)}"
            holes.append(name)
            return Token('IDENT', name)

        cloned_children = [clone(child) for child in tree_children(node)]
        return Tree(label, cloned_children)

    cloned_chain = clone(chain)

    if not holes:
        return None

    params = [Token('IDENT', name) for name in holes]
    paramlist = Tree('paramlist', params)

    return Tree('amp_lambda', [paramlist, cloned_chain])
