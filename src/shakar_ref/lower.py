from __future__ import annotations
from typing import List, Optional

from .tree import Tree, Tok
from .token_types import TT

from .tree import Node, is_tree, tree_children, tree_label
from .tree import is_token as is_token
from .ast_transforms import _infer_amp_lambda_params


def lower(ast: Node) -> Node:
    """Runtime lowering pass: hole desugaring + amp-lambda parameter inference."""
    ast = _desugar_call_holes(ast)
    ast = _infer_amp_lambda_params(ast)
    return ast


def _desugar_call_holes(node: Node) -> Node:
    if is_token(node) or not is_tree(node):
        return node

    children = getattr(node, "children", [])

    for idx, child in enumerate(list(children)):
        lowered = _desugar_call_holes(child)

        if lowered is not child:
            children[idx] = lowered

    candidate = node

    if tree_label(candidate) == "explicit_chain":
        replacement = _chain_to_lambda_if_holes(candidate)

        if replacement is not None:
            return replacement

    return candidate


def _chain_to_lambda_if_holes(chain: Tree) -> Optional[Tree]:
    def _contains_hole(node: Node) -> bool:
        if is_token(node) or not is_tree(node):
            return False

        if tree_label(node) == "holeexpr":
            return True
        return any(_contains_hole(child) for child in tree_children(node))

    holes: List[str] = []

    children = tree_children(chain)
    if not children:
        return None

    ops = children[1:]
    hole_call_index = None

    for idx, op in enumerate(ops):
        if tree_label(op) == "call" and _contains_hole(op):
            hole_call_index = idx
            break

    if hole_call_index is not None and hole_call_index + 1 < len(ops):
        raise SyntaxError(
            "Hole partials cannot be immediately invoked; assign or pass the partial before calling it"
        )

    def clone(node: Node) -> Node:
        if is_token(node) or not is_tree(node):
            return node

        label = tree_label(node)
        if label is None:
            return node
        if label == "holeexpr":
            name = f"_hole{len(holes)}"
            holes.append(name)
            return Tok(TT.IDENT, name, 0, 0)

        cloned_children: list[Node] = [clone(child) for child in tree_children(node)]
        cloned = Tree(label, cloned_children)
        setattr(cloned, "_meta", getattr(node, "meta", None))
        return cloned

    cloned_chain = clone(chain)

    if not holes:
        return None

    params = [Tok(TT.IDENT, name, 0, 0) for name in holes]
    param_children: list[Tok] = list(params)
    paramlist = Tree("paramlist", param_children)
    setattr(paramlist, "_meta", getattr(chain, "meta", None))

    lambda_children: list[Node] = [paramlist, cloned_chain]
    lam = Tree("amp_lambda", lambda_children)
    setattr(lam, "_meta", getattr(chain, "meta", None))
    return lam
