from __future__ import annotations
from typing import List, Optional

from .tree import Node, Tree, Tok, is_token, is_tree, tree_children, tree_label
from .token_types import TT
from .ast_transforms import (
    _HOLE_PARAM_PREFIX,
    _infer_amp_lambda_params,
    normalize_param_contracts,
)


def lower(ast: Node) -> Node:
    """Runtime lowering pass: hole desugaring + amp-lambda inference + param normalization + generator detection."""
    ast = _desugar_call_holes(ast)
    ast = _infer_amp_lambda_params(ast)
    ast = normalize_param_contracts(ast)
    _validate_noanchor_segments(ast)
    _mark_generator_fns(ast)
    return ast


def _validate_noanchor_segments(node: Node) -> None:
    def visit(child: Node, in_noanchor_expr: bool) -> None:
        if not is_tree(child):
            return

        label = tree_label(child)
        if in_noanchor_expr and label == "noanchor":
            raise SyntaxError("No-anchor segments are not allowed inside $expr")

        next_in_noanchor = in_noanchor_expr or label == "noanchor_expr"

        for grandchild in tree_children(child):
            visit(grandchild, next_in_noanchor)

    visit(node, False)


# Labels that define a new function scope boundary — yield inside these
# does not make the outer function a generator.
_FN_SCOPE_LABELS = frozenset(
    {
        "fndef",
        "anonfn",
        "amp_lambda",
        "decorator_def",
        "obj_method",
        "obj_get",
        "obj_set",
    }
)

# Subset of _FN_SCOPE_LABELS that the runtime can actually instantiate
# as generators.  decorator_def is excluded because the runtime does
# not support generator decorators — marking one would silently return
# a ShkGenerator instead of running the decorator body.
_GENERATOR_CAPABLE_LABELS = _FN_SCOPE_LABELS - {"decorator_def"}


def _mark_generator_fns(node: Node) -> None:
    """Walk AST and set attrs["generator"]=True on fndef/anonfn nodes whose
    immediate body contains yield/yield-deleg statements (lexically scoped)."""
    if not is_tree(node):
        return

    label = tree_label(node)

    if label in _GENERATOR_CAPABLE_LABELS:
        if any(_has_yield_in_scope(child) for child in tree_children(node)):
            if node.attrs is None:
                node.attrs = {}
            node.attrs["generator"] = True

    for child in tree_children(node):
        _mark_generator_fns(child)


_YIELD_BOUNDARY_LABELS = frozenset({"spawn", "deferstmt"})


def _has_yield_in_scope(node: Node) -> bool:
    """Recursively check for yieldstmt/yielddelegstmt, stopping at fn and
    yield boundaries (spawn, defer) — mirroring the runtime's semantics."""
    if not is_tree(node):
        return False

    label = tree_label(node)
    if label in ("yieldstmt", "yielddelegstmt"):
        return True

    # Do not descend into nested function definitions or yield boundaries.
    # Spawn and defer are runtime yield boundaries — a yield inside them
    # cannot reach the enclosing generator, so it must not cause the
    # enclosing function to be tagged as a generator.
    if label in _FN_SCOPE_LABELS or label in _YIELD_BOUNDARY_LABELS:
        return False

    return any(_has_yield_in_scope(child) for child in tree_children(node))


def _desugar_call_holes(node: Node) -> Node:
    if is_token(node) or not is_tree(node):
        return node

    for idx, child in enumerate(list(node.children)):
        lowered = _desugar_call_holes(child)
        if lowered is not child:
            node.children[idx] = lowered

    if tree_label(node) == "explicit_chain":
        replacement = _chain_to_lambda_if_holes(node)
        if replacement:
            return replacement

    return node


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
            # Hole params must be impossible to spell in source so user code
            # can never collide with lowered internals.
            name = f"{_HOLE_PARAM_PREFIX}{len(holes)}"
            holes.append(name)
            return Tok(TT.IDENT, name, 0, 0)

        attrs = dict(node.attrs) if node.attrs else None
        cloned = Tree(
            label,
            [clone(child) for child in tree_children(node)],
            meta=getattr(node, "meta", None),
            attrs=attrs,
        )
        cloned._meta = getattr(node, "_meta", None)
        return cloned

    cloned_chain = clone(chain)

    if not holes:
        return None

    params = [Tok(TT.IDENT, name, 0, 0) for name in holes]
    paramlist = Tree("paramlist", params)
    paramlist._meta = getattr(chain, "meta", None)

    lam = Tree("amp_lambda", [paramlist, cloned_chain])
    lam._meta = getattr(chain, "meta", None)
    return lam
