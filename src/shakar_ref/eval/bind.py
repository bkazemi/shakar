from __future__ import annotations

from typing import Callable, List

from ..runtime import Frame, ShkArray, ShkNumber, ShkValue, EvalResult, ShakarRuntimeError
from ..tree import Node, Tree, child_by_label, is_token, is_tree, tree_children, tree_label, ident_value

from .common import expect_ident_token, require_number
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

EvalFunc = Callable[[Node, Frame], ShkValue]
ApplyOpFunc = Callable[[EvalResult, Tree, Frame, EvalFunc], EvalResult]
IndexEvalFunc = Callable[[Tree, Frame, EvalFunc], ShkValue]

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
 
def eval_walrus(children: list[Node], frame: Frame, eval_func: EvalFunc) -> ShkValue:
    if len(children) != 2:
        raise ShakarRuntimeError("Malformed walrus expression")

    name_node, value_node = children
    name = expect_ident_token(name_node, "Walrus target")
    value = eval_func(value_node, frame)

    return define_new_ident(name, value, frame)

def eval_assign_stmt(children: list[Node], frame: Frame, eval_func: EvalFunc, apply_op: ApplyOpFunc, evaluate_index_operand: IndexEvalFunc) -> None:
    if len(children) < 2:
        raise ShakarRuntimeError("Malformed assignment statement")

    lvalue_node = children[0]
    value_node = children[-1]
    value = eval_func(value_node, frame)
    assign_lvalue(
        lvalue_node,
        value,
        frame,
        eval_func=eval_func,
        apply_op=apply_op,
        evaluate_index_operand=evaluate_index_operand,
        create=False,
    )

    return None

_COMPOUND_ASSIGN_OPERATORS: dict[str, str] = {
    '+=': '+',
    '-=': '-',
    '*=': '*',
    '/=': '/',
    '//=': '//',
    '%=': '%',
    '**=': '**',
}

def eval_compound_assign(children: list[Node], frame: Frame, eval_func: EvalFunc, apply_op: ApplyOpFunc, evaluate_index_operand: IndexEvalFunc) -> None:
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

    current_value = read_lvalue(
        lvalue_node,
        frame,
        eval_func=eval_func,
        apply_op=apply_op,
        evaluate_index_operand=evaluate_index_operand,
    )
    rhs_value = eval_func(rhs_node, frame)
    from .expr import apply_binary_operator
    new_value = apply_binary_operator(op_symbol, current_value, rhs_value)
    assign_lvalue(
        lvalue_node,
        new_value,
        frame,
        eval_func=eval_func,
        apply_op=apply_op,
        evaluate_index_operand=evaluate_index_operand,
        create=False,
    )

    return None

def eval_apply_assign(children: list[Node], frame: Frame, eval_func: EvalFunc, apply_op: ApplyOpFunc, evaluate_index_operand: IndexEvalFunc) -> ShkValue:
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
        eval_func=eval_func,
        apply_op=apply_op,
        evaluate_index_operand=evaluate_index_operand,
    )

def eval_rebind_primary(node: Tree, frame: Frame, eval_func: EvalFunc, apply_op: ApplyOpFunc, evaluate_index_operand: IndexEvalFunc) -> RebindContext:
    if not node.children:
        raise ShakarRuntimeError("Missing target for prefix rebind")

    target = node.children[0]
    ctx = resolve_rebind_lvalue(
        target,
        frame,
        eval_func=eval_func,
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
    eval_func: EvalFunc,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFunc,
) -> RebindContext | FanContext:
    """Resolve an expression into a rebindable context, unwrapping wrappers as needed."""
    current = node

    while is_tree(current) and tree_label(current) in {"primary", "group", "group_expr"} and len(current.children) == 1:
        current = current.children[0]

    if name := ident_value(current):
        return make_ident_context(name, frame)

    if is_tree(current):
        label = tree_label(current)

        if label == "rebind_primary":
            ctx = eval_func(current, frame)
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
                eval_func=eval_func,
                apply_op=apply_op,
                evaluate_index_operand=evaluate_index_operand,
            )

        if label == "expr" and current.children:
            return resolve_assignable_node(
                current.children[0],
                frame,
                eval_func=eval_func,
                apply_op=apply_op,
                evaluate_index_operand=evaluate_index_operand,
            )

    raise ShakarRuntimeError("Increment target must be assignable")

def resolve_chain_assignment(
    head_node: Node,
    ops: List[Tree],
    frame: Frame,
    *,
    eval_func: EvalFunc,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFunc,
) -> RebindContext | FanContext:
    """Walk `a.b[0]` and return the final assignable context (object slot, index, etc.)."""
    if not ops:
        return resolve_assignable_node(
            head_node,
            frame,
            eval_func=eval_func,
            apply_op=apply_op,
            evaluate_index_operand=evaluate_index_operand,
        )

    current = eval_func(head_node, frame)

    if isinstance(current, RebindContext):
        current = current.value

    for idx, op in enumerate(ops):
        is_last = idx == len(ops) - 1
        label = tree_label(op)

        if is_last and label in {"field", "fieldsel"}:
            field_name = expect_ident_token(op.children[0], "Field access")
            owner = current
            value = get_field_value(owner, field_name, frame)

            def field_setter(new_value: ShkValue, owner: ShkValue = owner, field_name: str = field_name) -> None:
                set_field_value(owner, field_name, new_value, frame, create=False)
                frame.dot = new_value

            return RebindContext(value, field_setter)

        if is_last and label == "index":
            owner = current
            idx_value = evaluate_index_operand(op, frame, eval_func)
            value = index_value(owner, idx_value, frame)

            def index_setter(new_value: ShkValue, owner: ShkValue = owner, idx_value: ShkValue = idx_value) -> None:
                set_index_value(owner, idx_value, new_value, frame)
                frame.dot = new_value

            return RebindContext(value, index_setter)

        if is_last and label == "fieldfan":
            return build_fieldfan_context(current, op, frame)

        current = apply_op(current, op, frame, eval_func)

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

def build_fieldfan_context(owner: ShkValue, fan_node: Tree, frame: Frame) -> FanContext:
    """Construct a fan context for `.={a,b}` nodes so updates reach every slot."""
    fieldlist_node = child_by_label(fan_node, "fieldlist")
    valuefan_list = child_by_label(fan_node, "valuefan_list")

    if fieldlist_node is None and valuefan_list is None:
        raise ShakarRuntimeError("Malformed field fan")

    if fieldlist_node is not None:
        names = [name for tok in tree_children(fieldlist_node) if (name := ident_value(tok))]
    else:
        # valuefan_list form (from valuefan normalization in lvalues)
        names = []
        for item in tree_children(valuefan_list):
            # valuefan_item -> child may be IDENT token
            if is_tree(item) and tree_label(item) == "valuefan_item":
                for ch in tree_children(item):
                    if name := ident_value(ch):
                        names.append(name)
                    else:
                        raise ShakarRuntimeError("Field fan only supports identifier entries")
            elif name := ident_value(item):
                names.append(name)
            else:
                raise ShakarRuntimeError("Field fan only supports identifier entries")

    if not names:
        raise ShakarRuntimeError("Field fan requires at least one identifier")

    if len(set(names)) != len(names):
        raise ShakarRuntimeError("Field fan cannot contain duplicate fields")

    contexts: List[RebindContext] = []

    for name in names:
        value = get_field_value(owner, name, frame)

        def setter(new_value: ShkValue, owner: ShkValue = owner, field_name: str = name) -> None:
            set_field_value(owner, field_name, new_value, frame, create=False)
            frame.dot = new_value

        contexts.append(RebindContext(value, setter))

    return FanContext(contexts)

def apply_fan_op(fan: FanContext | ShkValue, op: Tree, frame: Frame, *, apply_op: ApplyOpFunc, eval_func: EvalFunc) -> FanContext:
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
        res = apply_op(ctx, op, frame, eval_func)

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
    eval_func: EvalFunc,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFunc,
) -> ShkValue:
    """Evaluate the `.= expr` apply-assign form (subject-aware updates)."""
    head, *ops = lvalue_node.children

    if not ops and (name := ident_value(head)):
        target = frame.get(name)
        rhs_frame = Frame(parent=frame, dot=target)
        new_val = eval_func(rhs_node, rhs_frame)
        assign_ident(name, new_val, frame, create=False)
        return new_val

    target = eval_func(head, frame)

    if isinstance(target, RebindContext):
        target = target.value

    if not ops:
        raise ShakarRuntimeError("Malformed apply-assign target")

    for op in ops[:-1]:
        target = apply_op(target, op, frame, eval_func)

    # FanContext: apply the final op across all contexts
    if isinstance(target, FanContext):
        final_op = ops[-1]
        _apply_over_fancontext(target, final_op, rhs_node, frame, evaluate_index_operand, eval_func)
        # Apply-assign produces potentially different values per target, return array
        # (contrast with assign_lvalue which assigns same value to all, returns that value)
        return ShkArray([ctx.value for ctx in target.contexts])

    final_op = ops[-1]
    label = tree_label(final_op)
    match label:
        case "field" | "fieldsel":
            field_name = expect_ident_token(final_op.children[0], "Field assignment")
            old_val = get_field_value(target, field_name, frame)
            rhs_frame = Frame(parent=frame, dot=old_val)

            new_val = eval_func(rhs_node, rhs_frame)
            set_field_value(target, field_name, new_val, frame, create=False)
            return new_val
        case "lv_index":
            idx_val = evaluate_index_operand(final_op, frame, eval_func)
            old_val = index_value(target, idx_val, frame)
            rhs_frame = Frame(parent=frame, dot=old_val)

            new_val = eval_func(rhs_node, rhs_frame)
            set_index_value(target, idx_val, new_val, frame)
            return new_val
        case "fieldfan":
            fieldlist_node = child_by_label(final_op, "fieldlist")
            if fieldlist_node is None:
                raise ShakarRuntimeError("Malformed field fan-out list")

            names = [n for tok in fieldlist_node.children if (n := ident_value(tok))]
            if not names:
                raise ShakarRuntimeError("Empty field fan-out list")
            if len(set(names)) != len(names):
                raise ShakarRuntimeError("Field fan cannot contain duplicate fields")

            results: List[ShkValue] = []

            for name in names:
                old_val = get_field_value(target, name, frame)
                rhs_frame = Frame(parent=frame, dot=old_val)
                new_val = eval_func(rhs_node, rhs_frame)
                set_field_value(target, name, new_val, frame, create=False)
                results.append(new_val)
            return ShkArray(results)

    raise ShakarRuntimeError("Unsupported apply-assign target")


def _apply_over_fancontext(
    fan: FanContext,
    final_op: Tree,
    rhs_node: Tree,
    frame: Frame,
    evaluate_index_operand: IndexEvalFunc,
    eval_func: EvalFunc,
) -> None:
    """Apply-assign (`.=`) across all contexts in a FanContext."""
    label = tree_label(final_op)

    for ctx in fan.contexts:
        base = ctx.value
        match label:
            case "field" | "fieldsel":
                field_name = expect_ident_token(final_op.children[0], "Field assignment")
                old_val = get_field_value(base, field_name, frame)
                rhs_frame = Frame(parent=frame, dot=old_val)
                new_val = eval_func(rhs_node, rhs_frame)
                set_field_value(base, field_name, new_val, frame, create=False)
                ctx.value = new_val
            case "lv_index":
                idx_val = evaluate_index_operand(final_op, frame, eval_func)
                old_val = index_value(base, idx_val, frame)
                rhs_frame = Frame(parent=frame, dot=old_val)
                new_val = eval_func(rhs_node, rhs_frame)
                set_index_value(base, idx_val, new_val, frame)
                ctx.value = new_val
            case _:
                raise ShakarRuntimeError("Unsupported fan-out apply-assign target")


def _assign_over_fancontext(
    fan: FanContext,
    final_op: Tree,
    value: ShkValue,
    frame: Frame,
    evaluate_index_operand: IndexEvalFunc,
    eval_func: EvalFunc,
    create: bool,
) -> None:
    """Assign `value` to each target held in FanContext using final_op."""
    label = tree_label(final_op)

    for ctx in fan.contexts:
        base = ctx.value
        match label:
            case "field" | "fieldsel":
                field_name = expect_ident_token(final_op.children[0], "Field assignment")
                set_field_value(base, field_name, value, frame, create=create)
            case "lv_index":
                idx_val = evaluate_index_operand(final_op, frame, eval_func)
                set_index_value(base, idx_val, value, frame)
            case _:
                raise ShakarRuntimeError("Unsupported fan-out assignment target")

        ctx.value = base


def assign_lvalue(
    node: Node,
    value: ShkValue,
    frame: Frame,
    *,
    eval_func: EvalFunc,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFunc,
    create: bool,
) -> ShkValue:
    """Assign to an lvalue, supporting fields, indices, and field fans."""
    if not is_tree(node) or tree_label(node) != "lvalue":
        raise ShakarRuntimeError("Invalid assignment target")

    if not node.children:
        raise ShakarRuntimeError("Empty lvalue")

    head, *ops = node.children

    if not ops and (name := ident_value(head)):
        return assign_ident(name, value, frame, create=create)

    target = eval_func(head, frame)

    if not ops:
        raise ShakarRuntimeError("Malformed lvalue")

    for op in ops[:-1]:
        target = apply_op(target, op, frame, eval_func)

    # FanContext means we have fanned-out receivers; delegate assignment per receiver.
    if isinstance(target, FanContext):
        final_op = ops[-1]
        _assign_over_fancontext(target, final_op, value, frame, evaluate_index_operand, eval_func, create)
        return value

    final_op = ops[-1]

    label = tree_label(final_op)
    match label:
        case "field" | "fieldsel":
            field_name = expect_ident_token(final_op.children[0], "Field assignment")
            return set_field_value(target, field_name, value, frame, create=create)
        case "lv_index":
            idx_val = evaluate_index_operand(final_op, frame, eval_func)
            return set_index_value(target, idx_val, value, frame)
        case "fieldfan":
            fieldlist_node = child_by_label(final_op, "fieldlist")

            if fieldlist_node is None:
                raise ShakarRuntimeError("Malformed field fan-out list")

            names = [n for tok in tree_children(fieldlist_node) if (n := ident_value(tok))]
            if not names:
                raise ShakarRuntimeError("Empty field fan-out list")
            if len(set(names)) != len(names):
                raise ShakarRuntimeError("Field fan cannot contain duplicate fields")

            vals = fanout_values(value, len(names))

            for name, val in zip(names, vals):
                set_field_value(target, name, val, frame, create=create)
            return value

    raise ShakarRuntimeError("Unsupported assignment target")

def assign_pattern_value(
    pattern: Tree,
    value: ShkValue,
    frame: Frame,
    *,
    create: bool,
    allow_broadcast: bool,
    eval_func: EvalFunc,
) -> None:
    """Bind a destructuring pattern to a value, including walrus broadcasts."""
    def _assign_ident_wrapper(name: str, val: ShkValue, target_frame: Frame, create_flag: bool) -> None:
        if create_flag and allow_broadcast:
            define_new_ident(name, val, target_frame)
            return
        assign_ident(name, val, target_frame, create=create_flag)

    destructure_assign_pattern(eval_func, _assign_ident_wrapper, pattern, value, frame, create, allow_broadcast)

def read_lvalue(
    node: Node,
    frame: Frame,
    *,
    eval_func: EvalFunc,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFunc,
) -> ShkValue:
    """Fetch the value behind an assignment target (e.g. for compound ops)."""
    if not is_tree(node) or tree_label(node) != "lvalue":
        raise ShakarRuntimeError("Invalid lvalue")

    if not node.children:
        raise ShakarRuntimeError("Empty lvalue")

    head, *ops = node.children

    if not ops and (name := ident_value(head)):
        return frame.get(name)

    target = eval_func(head, frame)

    if not ops:
        raise ShakarRuntimeError("Malformed lvalue")

    for op in ops[:-1]:
        target = apply_op(target, op, frame, eval_func)
    final_op = ops[-1]

    label = tree_label(final_op)
    match label:
        case "field" | "fieldsel":
            field_name = expect_ident_token(final_op.children[0], "Field value access")
            return get_field_value(target, field_name, frame)
        case "lv_index":
            idx_val = evaluate_index_operand(final_op, frame, eval_func)
            return index_value(target, idx_val, frame)
    raise ShakarRuntimeError("Compound assignment not supported for this target")

def resolve_rebind_lvalue(
    node: Node,
    frame: Frame,
    *,
    eval_func: EvalFunc,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFunc,
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
        eval_func=eval_func,
        apply_op=apply_op,
        evaluate_index_operand=evaluate_index_operand,
    )
