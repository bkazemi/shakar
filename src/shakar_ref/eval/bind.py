from __future__ import annotations

from typing import Callable, List, Tuple

from ..runtime import (
    Frame,
    ShkArray,
    ShkFan,
    ShkNil,
    ShkNumber,
    ShkValue,
    EvalResult,
    ShakarRuntimeError,
)
from ..tree import (
    Node,
    Tree,
    child_by_label,
    is_token,
    is_tree,
    tree_children,
    tree_label,
    ident_value,
)

from .common import expect_ident_token, require_number, unwrap_noanchor
from .destructure import assign_pattern as destructure_assign_pattern
from .mutation import get_field_value, set_field_value, index_value, set_index_value
from .postfix import define_new_ident
from ..utils import fanout_values


__all__ = [
    "FanContext",
    "RebindContext",
    "assign_ident",
    "eval_walrus",
    "eval_assign_stmt",
    "eval_compound_assign",
    "eval_rebind_primary",
    "eval_apply_assign",
    "assign_lvalue",
    "assign_pattern_value",
    "apply_assign",
    "apply_numeric_delta",
    "build_fieldfan_context",
    "resolve_assignable_node",
    "resolve_chain_assignment",
    "make_ident_context",
    "read_lvalue",
    "apply_fan_op",
]


class RebindContext:
    """Tracks an assignable slot (identifier or field) so tail ops can write back."""

    __slots__ = ("value", "setter")

    def __init__(self, value: ShkValue, setter: Callable[[ShkValue], None]) -> None:
        self.value = value
        self.setter = setter


class FanContext:
    """Represents `.={a,b}` fan assignments; stores every target's context."""

    __slots__ = ("contexts", "values")

    def __init__(self, contexts: List[RebindContext]) -> None:
        self.contexts = contexts
        self.values = [ctx.value for ctx in contexts]

    def update_from_contexts(self) -> None:
        self.values = [ctx.value for ctx in self.contexts]

    def snapshot(self) -> List[ShkValue]:
        return list(self.values)


EvalFn = Callable[[Node, Frame], ShkValue]
ApplyOpFunc = Callable[[EvalResult, Tree, Frame, EvalFn], EvalResult]
IndexEvalFn = Callable[[Tree, Frame, EvalFn], ShkValue]


def assign_ident(name: str, value: ShkValue, frame: Frame, *, create: bool) -> ShkValue:
    """Assign or define an identifier in the given frame."""
    try:
        frame.set(name, value)
    except ShakarRuntimeError:
        if create:
            frame.define(name, value)
        else:
            raise
    return value


def eval_walrus(children: list[Node], frame: Frame, eval_fn: EvalFn) -> ShkValue:
    if len(children) != 2:
        raise ShakarRuntimeError("Malformed walrus expression")

    name_node, value_node = children
    name = expect_ident_token(name_node, "Walrus target")
    value = eval_fn(value_node, frame)

    return define_new_ident(name, value, frame)


def eval_assign_stmt(
    children: list[Node],
    frame: Frame,
    eval_fn: EvalFn,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFn,
) -> None:
    if len(children) < 2:
        raise ShakarRuntimeError("Malformed assignment statement")

    lvalue_node = children[0]
    value_node = children[-1]
    value = eval_fn(value_node, frame)
    assign_lvalue(
        lvalue_node,
        value,
        frame,
        eval_fn=eval_fn,
        apply_op=apply_op,
        evaluate_index_operand=evaluate_index_operand,
        create=False,
    )

    return None


_COMPOUND_ASSIGN_OPERATORS: dict[str, str] = {
    "+=": "+",
    "-=": "-",
    "*=": "*",
    "/=": "/",
    "//=": "//",
    "%=": "%",
    "**=": "**",
}


def eval_compound_assign(
    children: list[Node],
    frame: Frame,
    eval_fn: EvalFn,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFn,
) -> None:
    if len(children) < 3:
        raise ShakarRuntimeError("Malformed compound assignment")

    lvalue_node = children[0]
    rhs_node = children[-1]
    op_token = next((child for child in children[1:-1] if is_token(child)), None)

    if op_token is None:
        raise ShakarRuntimeError("Compound assignment missing operator")

    op_value = op_token.value
    op_symbol = _COMPOUND_ASSIGN_OPERATORS.get(op_value)

    if op_symbol is None:
        raise ShakarRuntimeError(f"Unsupported compound operator {op_value}")

    contexts = _resolve_lvalue_contexts(
        lvalue_node,
        frame,
        eval_fn=eval_fn,
        apply_op=apply_op,
        evaluate_index_operand=evaluate_index_operand,
    )
    rhs_value = eval_fn(rhs_node, frame)
    rhs_values = fanout_values(rhs_value, len(contexts))
    from .expr import apply_binary_operator

    for ctx, rhs_item in zip(contexts, rhs_values):
        new_value = apply_binary_operator(op_symbol, ctx.value, rhs_item)
        ctx.setter(new_value)
        ctx.value = new_value

    return None


def eval_apply_assign(
    children: list[Node],
    frame: Frame,
    eval_fn: EvalFn,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFn,
) -> ShkValue:
    lvalue_node = None
    rhs_node = None

    for child in children:
        if tree_label(child) == "lvalue":
            lvalue_node = child
        elif is_tree(child):
            rhs_node = child

    if lvalue_node is None or rhs_node is None:
        raise ShakarRuntimeError("Malformed apply-assign expression")

    return apply_assign(
        lvalue_node,
        rhs_node,
        frame,
        eval_fn=eval_fn,
        apply_op=apply_op,
        evaluate_index_operand=evaluate_index_operand,
    )


def eval_rebind_primary(
    node: Tree,
    frame: Frame,
    eval_fn: EvalFn,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFn,
) -> RebindContext:
    if not node.children:
        raise ShakarRuntimeError("Missing target for prefix rebind")

    target = node.children[0]
    ctx = resolve_rebind_lvalue(
        target,
        frame,
        eval_fn=eval_fn,
        apply_op=apply_op,
        evaluate_index_operand=evaluate_index_operand,
    )
    if isinstance(ctx, FanContext):
        raise ShakarRuntimeError("Rebind target cannot be a field fan")
    frame.dot = ctx.value

    return ctx


def make_ident_context(name: str, frame: Frame) -> RebindContext:
    """Produce a `RebindContext` for identifier assignments so tail ops can persist."""
    value = frame.get(name)

    def setter(new_value: ShkValue) -> None:
        assign_ident(name, new_value, frame, create=False)
        frame.dot = new_value

    return RebindContext(value, setter)


def resolve_assignable_node(
    node: Node,
    frame: Frame,
    *,
    eval_fn: EvalFn,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFn,
) -> RebindContext | FanContext:
    """Resolve an expression into a rebindable context, unwrapping wrappers as needed."""
    current = node

    while (
        is_tree(current)
        and tree_label(current) in {"primary", "group", "group_expr"}
        and len(current.children) == 1
    ):
        current = current.children[0]

    if name := ident_value(current):
        return make_ident_context(name, frame)

    if is_tree(current):
        label = tree_label(current)

        if label == "rebind_primary":
            ctx = eval_fn(current, frame)
            if isinstance(ctx, RebindContext):
                return ctx

            raise ShakarRuntimeError("Rebind primary did not produce a context")

        if label == "explicit_chain":
            if not current.children:
                raise ShakarRuntimeError("Malformed explicit chain")

            head = current.children[0]
            ops = list(current.children[1:])

            return resolve_chain_assignment(
                head,
                ops,
                frame,
                eval_fn=eval_fn,
                apply_op=apply_op,
                evaluate_index_operand=evaluate_index_operand,
            )

        if label == "expr" and current.children:
            return resolve_assignable_node(
                current.children[0],
                frame,
                eval_fn=eval_fn,
                apply_op=apply_op,
                evaluate_index_operand=evaluate_index_operand,
            )

    raise ShakarRuntimeError("Increment target must be assignable")


# ---------------------------------------------------------------------------
# Final-segment resolution helpers
#
# These centralise the `unwrap_noanchor => match label => field/index` dispatch
# that previously lived inline in resolve_chain_assignment,
# _complete_chain_assignment, and fanout.py's _read/_store.
# ---------------------------------------------------------------------------


def _read_segment(
    target: ShkValue,
    op: Tree,
    label: str,
    frame: Frame,
    evaluate_index_operand: IndexEvalFn,
    eval_fn: EvalFn,
) -> ShkValue:
    """Read the current value at a resolved final segment (field or index)."""
    match label:
        case "field" | "fieldsel":
            name = expect_ident_token(op.children[0], "Field access")
            return get_field_value(target, name, frame)
        case "index" | "lv_index":
            idx_val = evaluate_index_operand(op, frame, eval_fn)
            return index_value(target, idx_val, frame)

    raise ShakarRuntimeError("Unsupported read target")


def _write_segment(
    target: ShkValue,
    op: Tree,
    label: str,
    value: ShkValue,
    frame: Frame,
    evaluate_index_operand: IndexEvalFn,
    eval_fn: EvalFn,
    *,
    create: bool = False,
) -> ShkValue:
    """Write a value at a resolved final segment (field or index)."""
    match label:
        case "field" | "fieldsel":
            name = expect_ident_token(op.children[0], "Field assignment")
            return set_field_value(target, name, value, frame, create=create)
        case "index" | "lv_index":
            idx_val = evaluate_index_operand(op, frame, eval_fn)
            return set_index_value(target, idx_val, value, frame)

    raise ShakarRuntimeError("Unsupported assignment target")


def _rebind_segment(
    owner: ShkValue,
    op: Tree,
    label: str,
    frame: Frame,
    evaluate_index_operand: IndexEvalFn,
    eval_fn: EvalFn,
) -> RebindContext:
    """Build a RebindContext for a field or index final segment."""
    match label:
        case "field" | "fieldsel":
            name = expect_ident_token(op.children[0], "Field access")
            value = get_field_value(owner, name, frame)

            def field_setter(
                new_value: ShkValue,
                owner: ShkValue = owner,
                field_name: str = name,
            ) -> None:
                set_field_value(owner, field_name, new_value, frame, create=False)
                frame.dot = new_value

            return RebindContext(value, field_setter)

        case "index" | "lv_index":
            idx_val = evaluate_index_operand(op, frame, eval_fn)
            value = index_value(owner, idx_val, frame)

            def index_setter(
                new_value: ShkValue,
                owner: ShkValue = owner,
                idx_value: ShkValue = idx_val,
            ) -> None:
                set_index_value(owner, idx_value, new_value, frame)
                frame.dot = new_value

            return RebindContext(value, index_setter)

    raise ShakarRuntimeError("Target must be a field or index")


def _is_fan_literal_node(node: Node) -> bool:
    return is_tree(node) and tree_label(node) == "fan_literal"


def _unwrap_lvalue_head(node: Node) -> Node:
    current = node

    while (
        is_tree(current)
        and tree_label(current) in {"group", "group_expr", "expr", "primary"}
        and len(current.children) == 1
    ):
        current = current.children[0]

    return current


def _fan_literal_item_nodes(node: Node) -> List[Node]:
    if not _is_fan_literal_node(node):
        return []

    items_node = child_by_label(node, "fan_items")
    if items_node is None:
        return []

    return list(tree_children(items_node))


def _resolve_fan_literal_contexts(
    fan_node: Node,
    frame: Frame,
    *,
    eval_fn: EvalFn,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFn,
) -> List[RebindContext]:
    contexts: List[RebindContext] = []

    for index, item in enumerate(_fan_literal_item_nodes(fan_node)):
        try:
            ctx = resolve_assignable_node(
                item,
                frame,
                eval_fn=eval_fn,
                apply_op=apply_op,
                evaluate_index_operand=evaluate_index_operand,
            )
        except ShakarRuntimeError:
            raise ShakarRuntimeError(
                f"Fan assignment item {index + 1} must be assignable"
            ) from None

        if isinstance(ctx, FanContext):
            contexts.extend(ctx.contexts)
        else:
            contexts.append(ctx)

    return contexts


def _resolve_final_segment_contexts(
    target: ShkValue | FanContext | RebindContext,
    raw_final_op: Tree,
    frame: Frame,
    evaluate_index_operand: IndexEvalFn,
    eval_fn: EvalFn,
) -> List[RebindContext]:
    if isinstance(target, RebindContext):
        target = target.value

    if isinstance(target, FanContext):
        contexts: List[RebindContext] = []

        for ctx in target.contexts:
            contexts.extend(
                _resolve_final_segment_contexts(
                    ctx.value,
                    raw_final_op,
                    frame,
                    evaluate_index_operand,
                    eval_fn,
                )
            )

        return contexts

    final_op, label = unwrap_noanchor(raw_final_op)

    if raw_final_op is not final_op:
        frame.pending_anchor_override = target

    if label in {"field", "fieldsel", "index", "lv_index"}:
        return [
            _rebind_segment(
                target,
                final_op,
                label,
                frame,
                evaluate_index_operand,
                eval_fn,
            )
        ]

    if label == "fieldfan":
        return build_fieldfan_context(target, final_op, frame).contexts

    raise ShakarRuntimeError("Compound assignment not supported for this target")


def _resolve_fan_chain_contexts(
    fan: ShkFan,
    ops: List[Tree],
    frame: Frame,
    *,
    eval_fn: EvalFn,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFn,
) -> List[RebindContext]:
    if not ops:
        return []

    contexts: List[RebindContext] = []
    saved_dot = frame.dot
    saved_pending = frame.pending_anchor_override

    try:
        for item in fan.items:
            current: ShkValue | FanContext | RebindContext = item
            frame.dot = item

            for op in ops[:-1]:
                frame.dot = item
                frame.pending_anchor_override = None
                current = apply_op(current, op, frame, eval_fn)

                if isinstance(current, RebindContext):
                    current = current.value

            frame.pending_anchor_override = None
            contexts.extend(
                _resolve_final_segment_contexts(
                    current,
                    ops[-1],
                    frame,
                    evaluate_index_operand,
                    eval_fn,
                )
            )
    finally:
        frame.dot = saved_dot
        frame.pending_anchor_override = saved_pending

    return contexts


def _resolve_lvalue_contexts(
    node: Node,
    frame: Frame,
    *,
    eval_fn: EvalFn,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFn,
) -> List[RebindContext]:
    if not is_tree(node) or tree_label(node) != "lvalue":
        raise ShakarRuntimeError("Invalid lvalue")

    if not node.children:
        raise ShakarRuntimeError("Empty lvalue")

    head, *ops = node.children

    core_head = _unwrap_lvalue_head(head)

    if not ops and (name := ident_value(core_head)):
        return [make_ident_context(name, frame)]

    if not ops and _is_fan_literal_node(core_head):
        return _resolve_fan_literal_contexts(
            core_head,
            frame,
            eval_fn=eval_fn,
            apply_op=apply_op,
            evaluate_index_operand=evaluate_index_operand,
        )

    target = eval_fn(head, frame)

    if isinstance(target, ShkFan):
        if not ops:
            raise ShakarRuntimeError("Malformed lvalue")
        return _resolve_fan_chain_contexts(
            target,
            ops,
            frame,
            eval_fn=eval_fn,
            apply_op=apply_op,
            evaluate_index_operand=evaluate_index_operand,
        )

    if not ops:
        raise ShakarRuntimeError("Malformed lvalue")

    for op in ops[:-1]:
        target = apply_op(target, op, frame, eval_fn)

    return _resolve_final_segment_contexts(
        target,
        ops[-1],
        frame,
        evaluate_index_operand,
        eval_fn,
    )


def _lvalue_uses_fan_shape(node: Node) -> bool:
    if not is_tree(node) or tree_label(node) != "lvalue":
        return False

    if not node.children:
        return False

    head, *ops = node.children
    if _is_fan_literal_node(_unwrap_lvalue_head(head)):
        return True

    for raw_op in ops:
        op = raw_op
        if is_tree(op) and tree_label(op) == "noanchor" and op.children:
            op = op.children[0]

        if is_tree(op) and tree_label(op) in {"fieldfan", "valuefan"}:
            return True

    return False


def resolve_chain_assignment(
    head_node: Node,
    ops: List[Tree],
    frame: Frame,
    *,
    eval_fn: EvalFn,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFn,
) -> RebindContext | FanContext:
    """Walk `a.b[0]` and return the final assignable context (object slot, index, etc.)."""
    if not ops:
        return resolve_assignable_node(
            head_node,
            frame,
            eval_fn=eval_fn,
            apply_op=apply_op,
            evaluate_index_operand=evaluate_index_operand,
        )

    current = eval_fn(head_node, frame)

    if isinstance(current, RebindContext):
        current = current.value

    for idx, raw_op in enumerate(ops):
        is_last = idx == len(ops) - 1
        op, label = unwrap_noanchor(raw_op)

        # Capture anchor for final noanchor-wrapped segment (handled manually below)
        if is_last and raw_op is not op:
            frame.pending_anchor_override = current

        if is_last and label in {"field", "fieldsel", "index"}:
            return _rebind_segment(
                current,
                op,
                label,
                frame,
                evaluate_index_operand,
                eval_fn,
            )

        if is_last and label == "fieldfan":
            return build_fieldfan_context(current, op, frame)

        current = apply_op(current, raw_op, frame, eval_fn)

        if isinstance(current, RebindContext):
            current = current.value

    raise ShakarRuntimeError("Increment target must end with a field or index")


def apply_numeric_delta(ref: RebindContext, delta: int) -> tuple[ShkValue, ShkValue]:
    """Increment/decrement the referenced numeric context and return (old, new)."""
    current = ref.value
    num = require_number(current)

    new_val = ShkNumber(num.value + delta)

    ref.setter(new_val)
    ref.value = new_val

    return current, new_val


def _extract_fieldfan_names(
    fan_node: Tree,
    *,
    missing_msg: str,
    empty_msg: str,
) -> List[str]:
    """Extract field names from a normalized fieldfan node.

    After AST normalization, fieldfan always contains a fieldlist with IDENT tokens.
    """
    fieldlist_node = child_by_label(fan_node, "fieldlist")

    if fieldlist_node is None:
        raise ShakarRuntimeError(missing_msg)

    names: List[str] = []
    for tok in tree_children(fieldlist_node):
        if name := ident_value(tok):
            names.append(name)

    if not names:
        raise ShakarRuntimeError(empty_msg)

    if len(set(names)) != len(names):
        raise ShakarRuntimeError("Field fan cannot contain duplicate fields")

    return names


def build_fieldfan_context(owner: ShkValue, fan_node: Tree, frame: Frame) -> FanContext:
    """Construct a fan context for `.={a,b}` nodes so updates reach every slot."""
    names = _extract_fieldfan_names(
        fan_node,
        missing_msg="Malformed field fan",
        empty_msg="Field fan requires at least one identifier",
    )

    contexts: List[RebindContext] = []

    for name in names:
        value = get_field_value(owner, name, frame)

        def setter(
            new_value: ShkValue, owner: ShkValue = owner, field_name: str = name
        ) -> None:
            set_field_value(owner, field_name, new_value, frame, create=False)
            frame.dot = new_value

        contexts.append(RebindContext(value, setter))

    return FanContext(contexts)


def apply_fan_op(
    fan: FanContext | ShkValue,
    op: Tree,
    frame: Frame,
    *,
    apply_op: ApplyOpFunc,
    eval_fn: EvalFn,
) -> FanContext:
    """Apply chain ops across a fan context; initial fieldfan builds contexts."""
    if not isinstance(fan, FanContext):
        # First encounter: build contexts from base value and the fan node.
        return build_fieldfan_context(fan, op, frame)

    # Further ops after fan: apply to each context/value.
    label = tree_label(op)
    if label == "fieldfan":
        raise ShakarRuntimeError("Nested field fan not supported inside fanout")

    new_contexts: List[RebindContext] = []
    new_values: List[ShkValue] = []
    has_contexts = False
    has_values = False

    for ctx in fan.contexts:
        res = apply_op(ctx, op, frame, eval_fn)

        if isinstance(res, FanContext):
            if res.contexts:
                new_contexts.extend(res.contexts)
                has_contexts = True
            else:
                new_values.extend(res.values)
                has_values = True
        elif isinstance(res, RebindContext):
            new_contexts.append(res)
            has_contexts = True
        else:
            new_values.append(res)
            has_values = True

    if not fan.contexts and not new_contexts and not new_values:
        return fan

    if has_contexts and has_values:
        raise ShakarRuntimeError("Mixed field fan results not supported")

    if has_contexts:
        fan.contexts = new_contexts
        fan.update_from_contexts()
    else:
        fan.contexts = []
        fan.values = new_values

    return fan


def apply_assign(
    lvalue_node: Tree,
    rhs_node: Tree,
    frame: Frame,
    *,
    eval_fn: EvalFn,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFn,
) -> ShkValue:
    """Evaluate the `.= expr` apply-assign form (subject-aware updates)."""
    contexts = _resolve_lvalue_contexts(
        lvalue_node,
        frame,
        eval_fn=eval_fn,
        apply_op=apply_op,
        evaluate_index_operand=evaluate_index_operand,
    )
    fan_shaped = _lvalue_uses_fan_shape(lvalue_node)

    results: List[ShkValue] = []

    for ctx in contexts:
        rhs_frame = Frame(parent=frame, dot=ctx.value)
        new_val = eval_fn(rhs_node, rhs_frame)
        ctx.setter(new_val)
        ctx.value = new_val
        results.append(new_val)

    if fan_shaped or len(results) != 1:
        return ShkArray(results)

    return results[0]


def _assign_over_fancontext(
    fan: FanContext,
    final_op: Tree,
    value: ShkValue,
    frame: Frame,
    evaluate_index_operand: IndexEvalFn,
    eval_fn: EvalFn,
    create: bool,
) -> None:
    """Assign `value` to each target held in FanContext using final_op."""
    op, label = unwrap_noanchor(final_op)

    for ctx in fan.contexts:
        _write_segment(
            ctx.value,
            op,
            label,
            value,
            frame,
            evaluate_index_operand,
            eval_fn,
            create=create,
        )


def _complete_chain_assignment(
    target: ShkValue | FanContext | RebindContext,
    raw_final_op: Tree,
    value: ShkValue,
    frame: Frame,
    evaluate_index_operand: IndexEvalFn,
    eval_fn: EvalFn,
    *,
    create: bool,
) -> ShkValue:
    if isinstance(target, RebindContext):
        target = target.value

    if isinstance(target, FanContext):
        _assign_over_fancontext(
            target,
            raw_final_op,
            value,
            frame,
            evaluate_index_operand,
            eval_fn,
            create,
        )
        return value

    final_op, label = unwrap_noanchor(raw_final_op)

    if label in {"field", "fieldsel", "lv_index"}:
        return _write_segment(
            target,
            final_op,
            label,
            value,
            frame,
            evaluate_index_operand,
            eval_fn,
            create=create,
        )

    if label == "fieldfan":
        names = _extract_fieldfan_names(
            final_op,
            missing_msg="Malformed field fan-out list",
            empty_msg="Empty field fan-out list",
        )
        vals = fanout_values(value, len(names))

        for name, val in zip(names, vals):
            set_field_value(target, name, val, frame, create=create)

        return value

    raise ShakarRuntimeError("Unsupported assignment target")


def _assign_over_fan(
    fan: ShkFan,
    ops: List[Tree],
    value: ShkValue,
    frame: Frame,
    *,
    eval_fn: EvalFn,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFn,
    create: bool,
) -> ShkValue:
    if not ops:
        raise ShakarRuntimeError("Malformed fan assignment target")

    saved_dot = frame.dot
    saved_pending = frame.pending_anchor_override

    try:
        for item in fan.items:
            current: ShkValue | FanContext | RebindContext = item
            frame.dot = item
            for op in ops[:-1]:
                frame.dot = item
                frame.pending_anchor_override = None
                current = apply_op(current, op, frame, eval_fn)
                if isinstance(current, RebindContext):
                    current = current.value

            frame.pending_anchor_override = None
            _complete_chain_assignment(
                current,
                ops[-1],
                value,
                frame,
                evaluate_index_operand,
                eval_fn,
                create=create,
            )
    finally:
        frame.dot = saved_dot
        frame.pending_anchor_override = saved_pending

    return value


def assign_lvalue(
    node: Node,
    value: ShkValue,
    frame: Frame,
    *,
    eval_fn: EvalFn,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFn,
    create: bool,
) -> ShkValue:
    """Assign to an lvalue, supporting fields, indices, and field fans."""
    if not is_tree(node) or tree_label(node) != "lvalue":
        raise ShakarRuntimeError("Invalid assignment target")

    if not node.children:
        raise ShakarRuntimeError("Empty lvalue")

    head, *ops = node.children
    core_head = _unwrap_lvalue_head(head)

    if not ops and (name := ident_value(core_head)):
        return assign_ident(name, value, frame, create=create)

    if not ops and _is_fan_literal_node(core_head):
        contexts = _resolve_fan_literal_contexts(
            core_head,
            frame,
            eval_fn=eval_fn,
            apply_op=apply_op,
            evaluate_index_operand=evaluate_index_operand,
        )
        values = fanout_values(value, len(contexts))

        for ctx, item_value in zip(contexts, values):
            ctx.setter(item_value)
            ctx.value = item_value

        return value

    target = eval_fn(head, frame)

    if not ops:
        raise ShakarRuntimeError("Malformed lvalue")

    if isinstance(target, ShkFan):
        return _assign_over_fan(
            target,
            ops,
            value,
            frame,
            eval_fn=eval_fn,
            apply_op=apply_op,
            evaluate_index_operand=evaluate_index_operand,
            create=create,
        )

    for op in ops[:-1]:
        target = apply_op(target, op, frame, eval_fn)

    return _complete_chain_assignment(
        target,
        ops[-1],
        value,
        frame,
        evaluate_index_operand,
        eval_fn,
        create=create,
    )


def assign_pattern_value(
    pattern: Tree,
    value: ShkValue,
    frame: Frame,
    *,
    create: bool,
    allow_broadcast: bool,
    eval_fn: EvalFn,
) -> None:
    """Bind a destructuring pattern to a value, including walrus broadcasts."""

    def _assign_ident_wrapper(
        name: str, val: ShkValue, target_frame: Frame, create_flag: bool
    ) -> None:
        if create_flag and allow_broadcast:
            define_new_ident(name, val, target_frame)
            return
        assign_ident(name, val, target_frame, create=create_flag)

    destructure_assign_pattern(
        eval_fn, _assign_ident_wrapper, pattern, value, frame, create, allow_broadcast
    )


def read_lvalue(
    node: Node,
    frame: Frame,
    *,
    eval_fn: EvalFn,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFn,
) -> ShkValue:
    """Fetch the value behind an assignment target (e.g. for compound ops)."""
    contexts = _resolve_lvalue_contexts(
        node,
        frame,
        eval_fn=eval_fn,
        apply_op=apply_op,
        evaluate_index_operand=evaluate_index_operand,
    )
    if len(contexts) == 1 and not _lvalue_uses_fan_shape(node):
        return contexts[0].value
    return ShkFan([ctx.value for ctx in contexts])


def resolve_rebind_lvalue(
    node: Node,
    frame: Frame,
    *,
    eval_fn: EvalFn,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFn,
) -> RebindContext | FanContext:
    """Turn a `rebind_lvalue` node into a concrete context for prefix rebinds."""
    if name := ident_value(node):
        return make_ident_context(name, frame)

    if is_token(node):
        raise ShakarRuntimeError("Malformed rebind target")

    if not is_tree(node):
        raise ShakarRuntimeError("Malformed rebind target")

    label = tree_label(node)
    if label not in {"rebind_lvalue", "rebind_lvalue_grouped"}:
        raise ShakarRuntimeError("Malformed rebind target")

    children = tree_children(node)
    if not children:
        raise ShakarRuntimeError("Empty rebind target")

    head = children[0]
    ops = list(children[1:])

    return resolve_chain_assignment(
        head,
        ops,
        frame,
        eval_fn=eval_fn,
        apply_op=apply_op,
        evaluate_index_operand=evaluate_index_operand,
    )
