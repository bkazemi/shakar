from __future__ import annotations

import math
from typing import Callable, List, Optional, Set

from ..tree import Tok
from ..token_types import TT

from ..runtime import (
    Frame,
    ShkArray,
    ShkBool,
    ShkNull,
    ShkNumber,
    ShkObject,
    ShkPath,
    ShkValue,
    ShkSelector,
    ShkString,
    ShkRegex,
    ShakarKeyError,
    ShakarIndexError,
    ShakarRuntimeError,
    ShakarTypeError,
    regex_match_value,
)
from ..tree import Node, Tree, is_tree, node_meta, tree_children, tree_label
from ..utils import shk_equals
from .bind import FanContext, RebindContext, apply_numeric_delta
from .chains import apply_op as chain_apply_op, evaluate_index_operand
from .common import require_number, stringify, token_kind
from .helpers import (
    eval_anchor_scoped,
    is_truthy,
    isolate_anchor_override,
    retargets_anchor,
)
from .selector import selector_iter_values

EvalFunc = Callable[[Node, Frame], ShkValue]


def _consume_anchor_override(frame: Frame) -> Optional[ShkValue]:
    """Consume and return any pending anchor override set by chain evaluation."""
    override = frame.pending_anchor_override
    frame.pending_anchor_override = None
    return override


def _is_noanchor_wrapper(node: Node) -> bool:
    return is_tree(node) and tree_label(node) == "noanchor"


def _has_noanchor_segment(ops: list[Node]) -> bool:
    return any(_is_noanchor_wrapper(op) for op in ops)


def eval_explicit_chain(node: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    children = tree_children(node)
    if not children:
        raise ShakarRuntimeError("Malformed explicit chain")

    head, *ops = children

    if ops and tree_label(ops[-1]) in {"incr", "decr"}:
        tail = ops[-1]
        from .bind import resolve_chain_assignment
        from .chains import evaluate_index_operand

        context = resolve_chain_assignment(
            head,
            ops[:-1],
            frame,
            eval_func=eval_func,
            apply_op=chain_apply_op,
            evaluate_index_operand=evaluate_index_operand,
        )
        delta = 1 if tree_label(tail) == "incr" else -1
        if isinstance(context, FanContext):
            raise ShakarRuntimeError("++/-- not supported on field fan assignments")
        old_val, _ = apply_numeric_delta(context, delta)
        return old_val

    val = eval_func(head, frame)
    head_label = tree_label(head) if is_tree(head) else None
    head_is_rebind = head_label in {"rebind_primary", "rebind_primary_grouped"}
    head_is_grouped_rebind = head_label == "rebind_primary_grouped"
    tail_has_effect = False

    for op in ops:
        label = tree_label(op)
        inner_label = tree_label(op.children[0]) if label == "noanchor" else label
        if inner_label not in {"field", "fieldsel", "index"}:
            tail_has_effect = True
        val = chain_apply_op(val, op, frame, eval_func)

    if head_is_rebind:
        if not ops:
            raise ShakarRuntimeError("Prefix rebind requires a tail expression")
        if not head_is_grouped_rebind and not tail_has_effect:
            raise ShakarRuntimeError("Prefix rebind requires a tail expression")

    if isinstance(val, RebindContext):
        final = val.value
        val.setter(final)
        return final

    if isinstance(val, FanContext):
        return ShkArray(val.snapshot())
    return val


def _update_anchor(node: Node, value: ShkValue, frame: Frame) -> None:
    """Update frame.dot if node retargets anchor, using any pending override."""
    override = _consume_anchor_override(frame)  # always consume
    if not retargets_anchor(node):
        return
    frame.dot = override if override is not None else value


def normalize_unary_op(op_node: Node, frame: Frame) -> Node | str:
    if tree_label(op_node) == "unaryprefixop":
        if op_node.children:
            return normalize_unary_op(op_node.children[0], frame)
        src = getattr(frame, "source", None)
        meta = node_meta(op_node)

        if src is not None and meta is not None:
            return src[meta.start_pos : meta.end_pos]
        return ""
    return op_node


def eval_unary(
    op_node: Node,
    rhs_node: Tree,
    frame: Frame,
    eval_func: EvalFunc,
    apply_op_func=chain_apply_op,
) -> ShkValue:
    op_norm = normalize_unary_op(op_node, frame)
    op_value = op_norm.value if isinstance(op_norm, Tok) else op_norm

    if op_value in ("++", "--"):
        from .bind import resolve_assignable_node, apply_numeric_delta

        context = resolve_assignable_node(
            rhs_node,
            frame,
            eval_func=eval_func,
            apply_op=apply_op_func,
            evaluate_index_operand=evaluate_index_operand,
        )
        if isinstance(context, FanContext):
            raise ShakarRuntimeError("Increment target must end with a field or index")
        _, new_val = apply_numeric_delta(context, 1 if op_value == "++" else -1)
        return new_val

    rhs = eval_func(rhs_node, frame)

    match op_norm:
        case Tok(type=TT.PLUS) | "+":
            raise ShakarRuntimeError("unary + not supported")
        case Tok(type=TT.MINUS) | "-":
            rhs_num = _coerce_number(rhs)
            return ShkNumber(-rhs_num)
        case Tok(type=TT.TILDE) | "~":
            raise ShakarRuntimeError("bitwise ~ not supported yet")
        case Tok(type=TT.NOT) | "not":
            return ShkBool(not is_truthy(rhs))
        case Tok(type=TT.NEG) | "!":
            return ShkBool(not is_truthy(rhs))
        case _:
            raise ShakarRuntimeError("Unsupported unary op")


def eval_infix(
    children: List[Tree],
    frame: Frame,
    eval_func: EvalFunc,
    right_assoc_ops: Optional[Set[str]] = None,
) -> ShkValue:
    if not children:
        return ShkNull()

    arithmetic_ops: Set[str] = {"+", "-", "*", "/", "//", "%", "+>", "^", "**"}

    def _should_retarget(node: Node, next_op: Optional[str]) -> bool:
        if not retargets_anchor(node):
            return False
        # Avoid clobbering anchor for plain identifiers in arithmetic chains
        if token_kind(node) == "IDENT" and (
            next_op in arithmetic_ops or next_op is None
        ):
            return False
        return True

    def _eval_and_update(node: Node, next_op: Optional[str]) -> ShkValue:
        val = eval_func(node, frame)
        if _should_retarget(node, next_op):
            override = _consume_anchor_override(frame)
            frame.dot = override if override is not None else val
        else:
            _consume_anchor_override(frame)  # discard unused override
        return val

    operands = children[0::2]
    ops = [as_op(op_node) for op_node in children[1::2]]

    def evaluate_operands() -> List[ShkValue]:
        vals = []
        for i, node in enumerate(operands):
            next_op = ops[i] if i < len(ops) else None
            vals.append(_eval_and_update(node, next_op))
        return vals

    if right_assoc_ops and all(op in right_assoc_ops for op in ops):
        vals = evaluate_operands()
        acc = vals[-1]
        for i in range(len(vals) - 2, -1, -1):
            lhs = require_number(vals[i])
            rhs = require_number(acc)
            acc = ShkNumber(lhs.value**rhs.value)
        return acc

    vals = evaluate_operands()
    acc = vals[0]

    for i in range(1, len(vals)):
        op = ops[i - 1]
        acc = apply_binary_operator(op, acc, vals[i])

    return acc


def eval_compare(children: List[Tree], frame: Frame, eval_func: EvalFunc) -> ShkValue:
    if not children:
        return ShkNull()

    if len(children) == 1:
        return eval_func(children[0], frame)

    def _eval_and_update(node: Node) -> ShkValue:
        val = eval_func(node, frame)
        _update_anchor(node, val, frame)
        return val

    if (
        len(children) == 3
        and tree_label(children[1]) == "cmpop"
        and as_op(children[1]) == "~~"
    ):
        lhs_node, _, rhs_node = children
        lhs = _eval_and_update(lhs_node)
        rhs = _eval_and_update(rhs_node)
        return _regex_match(lhs, rhs)

    first_node = children[0]
    subject = _eval_and_update(first_node)
    idx = 1
    joiner = "and"
    last_comp: Optional[str] = None
    agg: Optional[bool] = None

    while idx < len(children):
        node = children[idx]
        tok = token_kind(node)

        if tok in {"AND", "and"}:
            joiner = "and"
            idx += 1
            continue
        if tok in {"OR", "or"}:
            joiner = "or"
            idx += 1
            continue

        label = tree_label(node)
        if label == "cmpop":
            comp = as_op(node)
            last_comp = comp
            idx += 1
            if idx >= len(children):
                raise ShakarRuntimeError("Missing right-hand side for comparison")
            rhs = _eval_and_update(children[idx])
            idx += 1
        else:
            if last_comp is None:
                raise ShakarRuntimeError("Comparator required in comparison chain")
            comp = last_comp
            rhs = _eval_and_update(node)
            idx += 1

        leg_val = _compare_values(comp, subject, rhs)
        agg = (
            leg_val
            if agg is None
            else ((agg and leg_val) if joiner == "and" else (agg or leg_val))
        )

    return ShkBool(bool(agg)) if agg is not None else ShkBool(True)


def eval_logical(
    kind: str, children: List[Tree], frame: Frame, eval_func: EvalFunc
) -> ShkValue:
    if not children:
        return ShkNull()

    normalized = "and" if "and" in kind else "or"
    prev_dot = frame.dot

    try:
        last_val: ShkValue = ShkBool(normalized == "and")

        for child in children:
            if token_kind(child) in {"AND", "OR"}:
                continue

            val = eval_func(child, frame)
            _update_anchor(child, val, frame)
            last_val = val

            if normalized == "and" and not is_truthy(val):
                return val
            if normalized == "or" and is_truthy(val):
                return val

        return last_val
    finally:
        frame.dot = prev_dot


def eval_nullish(children: List[Tree], frame: Frame, eval_func: EvalFunc) -> ShkValue:
    exprs = [
        child
        for child in children
        if not (isinstance(child, Tok) and child.value == "??")
    ]

    if not exprs:
        return ShkNull()

    current = eval_anchor_scoped(exprs[0], frame, eval_func)

    for expr in exprs[1:]:
        if not isinstance(current, ShkNull):
            return current
        current = eval_anchor_scoped(expr, frame, eval_func)

    return current


def eval_nullsafe(node: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    with isolate_anchor_override(frame):
        target = node
        if is_tree(node) and tree_label(node) == "nullsafe" and node.children:
            target = node.children[0]

        if not is_tree(target) or tree_label(target) != "explicit_chain":
            try:
                return eval_func(target, frame)
            except (ShakarKeyError, ShakarIndexError):
                return ShkNull()

        children = tree_children(target)
        if not children:
            return ShkNull()

        head = children[0]
        current = eval_func(head, frame)

        if isinstance(current, ShkNull):
            return ShkNull()

        for op in children[1:]:
            try:
                current = chain_apply_op(current, op, frame, eval_func)
            except (ShakarRuntimeError, ShakarTypeError) as err:
                if _nullsafe_recovers(err, current):
                    return ShkNull()
                raise

            if isinstance(current, RebindContext):
                current = current.value

            if isinstance(current, ShkNull):
                return ShkNull()
        return current


def eval_ternary(n: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    if len(n.children) != 3:
        raise ShakarRuntimeError("Malformed ternary expression")

    cond_node, true_node, false_node = n.children
    with isolate_anchor_override(frame):
        cond_val = eval_func(cond_node, frame)

    if is_truthy(cond_val):
        with isolate_anchor_override(frame):
            return eval_func(true_node, frame)

    with isolate_anchor_override(frame):
        return eval_func(false_node, frame)


def as_op(x: Node) -> str:
    if isinstance(x, Tok):
        return str(x.value)

    label = tree_label(x) if is_tree(x) else None
    if label is not None:
        if (
            label in ("addop", "mulop", "powop")
            and len(x.children) == 1
            and isinstance(x.children[0], Tok)
        ):
            return str(x.children[0].value)

        if label == "cmpop":
            tokens: list[str] = [
                str(tok.value) for tok in x.children if isinstance(tok, Tok)
            ]
            if not tokens:
                raise ShakarRuntimeError("Empty comparison operator")

            if tokens[0] == "!" and len(tokens) > 1:
                return "!" + "".join(tokens[1:])

            if len(tokens) > 1:
                return " ".join(tokens)
            return tokens[0]

    raise ShakarRuntimeError(f"Expected operator token, got {x!r}")


def apply_binary_operator(op: str, lhs: ShkValue, rhs: ShkValue) -> ShkValue:
    match op:
        case "+":
            if isinstance(lhs, ShkArray) and isinstance(rhs, ShkArray):
                return ShkArray(lhs.items + rhs.items)
            if isinstance(lhs, ShkString) or isinstance(rhs, ShkString):
                return ShkString(stringify(lhs) + stringify(rhs))
            lhs_num = require_number(lhs)
            rhs_num = require_number(rhs)
            return ShkNumber(lhs_num.value + rhs_num.value)
        case "-":
            lhs_num = require_number(lhs)
            rhs_num = require_number(rhs)
            return ShkNumber(lhs_num.value - rhs_num.value)
        case "*":
            lhs_num = require_number(lhs)
            rhs_num = require_number(rhs)
            return ShkNumber(lhs_num.value * rhs_num.value)
        case "/":
            if isinstance(lhs, ShkPath):
                if isinstance(rhs, ShkPath):
                    return ShkPath(str(lhs.as_path() / rhs.as_path()))
                if isinstance(rhs, ShkString):
                    return ShkPath(str(lhs.as_path() / rhs.value))
                raise ShakarTypeError("Path join expects a path or string on the right")
            lhs_num = require_number(lhs)
            rhs_num = require_number(rhs)
            return ShkNumber(lhs_num.value / rhs_num.value)
        case "//":
            lhs_num = require_number(lhs)
            rhs_num = require_number(rhs)
            return ShkNumber(math.floor(lhs_num.value / rhs_num.value))
        case "%":
            lhs_num = require_number(lhs)
            rhs_num = require_number(rhs)
            return ShkNumber(lhs_num.value % rhs_num.value)
        case "**":
            lhs_num = require_number(lhs)
            rhs_num = require_number(rhs)
            return ShkNumber(lhs_num.value**rhs_num.value)
    raise ShakarRuntimeError(f"Unknown operator {op}")


def _compare_values(op: str, lhs: ShkValue, rhs: ShkValue) -> bool:
    if isinstance(rhs, ShkSelector):
        return _compare_with_selector(op, lhs, rhs)

    match op:
        case "==":
            return shk_equals(lhs, rhs)
        case "!=":
            return not shk_equals(lhs, rhs)
        case "<":
            return _coerce_number(lhs) < _coerce_number(rhs)
        case "<=":
            return _coerce_number(lhs) <= _coerce_number(rhs)
        case ">":
            return _coerce_number(lhs) > _coerce_number(rhs)
        case ">=":
            return _coerce_number(lhs) >= _coerce_number(rhs)
        case "is":
            return shk_equals(lhs, rhs)
        case "!is" | "is not":
            return not shk_equals(lhs, rhs)
        case "in":
            return _contains(rhs, lhs)
        case "not in" | "!in":
            return not _contains(rhs, lhs)
        case "~":
            from .match import match_structure

            return match_structure(lhs, rhs)
        case "~~":
            return is_truthy(_regex_match(lhs, rhs))
        case _:
            raise ShakarRuntimeError(f"Unknown comparator {op}")


def _regex_match(lhs: ShkValue, rhs: ShkValue) -> ShkValue:
    if not isinstance(lhs, ShkString):
        raise ShakarTypeError("Regex match expects a string on the left")
    if not isinstance(rhs, ShkRegex):
        raise ShakarTypeError("Regex match expects a regex on the right")
    return regex_match_value(rhs, lhs.value)


def _selector_values(selector: ShkSelector) -> List[ShkNumber]:
    values = selector_iter_values(selector)
    if not values:
        raise ShakarRuntimeError("Selector literal produced no values")

    return [val for val in values if isinstance(val, ShkNumber)]


def _coerce_number(value: ShkValue) -> float:
    if isinstance(value, ShkNumber):
        return value.value

    raise ShakarTypeError("Expected number")


def _compare_with_selector(op: str, lhs: ShkValue, selector: ShkSelector) -> bool:
    values = _selector_values(selector)

    if op == "==":
        return all(shk_equals(lhs, val) for val in values)

    if op == "!=":
        return any(not shk_equals(lhs, val) for val in values)

    lhs_num = _coerce_number(lhs)
    rhs_nums = [_coerce_number(val) for val in values]

    match op:
        case "<":
            return all(lhs_num < num for num in rhs_nums)
        case "<=":
            return all(lhs_num <= num for num in rhs_nums)
        case ">":
            return all(lhs_num > num for num in rhs_nums)
        case ">=":
            return all(lhs_num >= num for num in rhs_nums)
        case _:
            raise ShakarTypeError(f"Unsupported comparator '{op}' for selector literal")


def _contains(container: ShkValue, item: ShkValue) -> bool:
    match container:
        case ShkArray(items=items):
            return any(shk_equals(element, item) for element in items)
        case ShkString(value=text):
            if isinstance(item, ShkString):
                return item.value in text
            raise ShakarTypeError("String membership requires a string value")
        case ShkObject(slots=slots):
            if isinstance(item, ShkString):
                return item.value in slots
            raise ShakarTypeError("Object membership requires a string key")
        case _:
            raise ShakarTypeError(
                f"Unsupported container type for 'in': {type(container).__name__}"
            )


def _nullsafe_recovers(err: Exception, recv: ShkValue) -> bool:
    if isinstance(recv, ShkNull):
        return True
    return isinstance(err, (ShakarKeyError, ShakarIndexError))
