from __future__ import annotations

from typing import Any, Callable, List

from lark import Tree

from shakar_runtime import Env, ShkArray, ShkNumber, ShakarRuntimeError
from shakar_tree import child_by_label, is_token_node, is_tree_node, tree_children, tree_label

from eval.common import expect_ident_token, token_kind, require_number
from eval.mutation import get_field_value, set_field_value, index_value, set_index_value
from shakar_utils import fanout_values


EvalFunc = Callable[[Any, Env], Any]
ApplyOpFunc = Callable[[Any, Tree, Env], Any]
AssignIdentFunc = Callable[[str, Any, Env, bool], Any]
IndexEvalFunc = Callable[[Tree, Env], Any]


class RebindContext:
    """Tracks an assignable slot (identifier or field) so tail ops can write back."""

    __slots__ = ("value", "setter")

    def __init__(self, value: Any, setter: Callable[[Any], None]) -> None:
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

    def snapshot(self) -> List[Any]:
        return list(self.values)


def make_ident_context(name: str, env: Env, *, assign_ident: AssignIdentFunc) -> RebindContext:
    """Produce a `RebindContext` for identifier assignments so tail ops can persist."""
    value = env.get(name)

    def setter(new_value: Any) -> None:
        assign_ident(name, new_value, env, create=False)
        env.dot = new_value

    return RebindContext(value, setter)


def resolve_assignable_node(
    node: Any,
    env: Env,
    *,
    eval_func: EvalFunc,
    apply_op: ApplyOpFunc,
    assign_ident: AssignIdentFunc,
    evaluate_index_operand: IndexEvalFunc,
) -> RebindContext:
    """Resolve an expression into a rebindable context, unwrapping wrappers as needed."""
    current = node
    while is_tree_node(current) and tree_label(current) in {"primary", "group", "group_expr"} and len(current.children) == 1:
        current = current.children[0]

    if is_token_node(current) and token_kind(current) == "IDENT":
        return make_ident_context(current.value, env, assign_ident=assign_ident)

    if is_tree_node(current):
        label = tree_label(current)
        if label == "rebind_primary":
            ctx = eval_func(current, env)
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
                env,
                eval_func=eval_func,
                apply_op=apply_op,
                assign_ident=assign_ident,
                evaluate_index_operand=evaluate_index_operand,
            )
        if label in {"expr", "expr_nc"} and current.children:
            return resolve_assignable_node(
                current.children[0],
                env,
                eval_func=eval_func,
                apply_op=apply_op,
                assign_ident=assign_ident,
                evaluate_index_operand=evaluate_index_operand,
            )

    raise ShakarRuntimeError("Increment target must be assignable")


def resolve_chain_assignment(
    head_node: Any,
    ops: List[Any],
    env: Env,
    *,
    eval_func: EvalFunc,
    apply_op: ApplyOpFunc,
    assign_ident: AssignIdentFunc,
    evaluate_index_operand: IndexEvalFunc,
) -> Any:
    """Walk `a.b[0]` and return the final assignable context (object slot, index, etc.)."""
    if not ops:
        return resolve_assignable_node(
            head_node,
            env,
            eval_func=eval_func,
            apply_op=apply_op,
            assign_ident=assign_ident,
            evaluate_index_operand=evaluate_index_operand,
        )

    current = eval_func(head_node, env)
    if isinstance(current, RebindContext):
        current = current.value

    for idx, op in enumerate(ops):
        is_last = idx == len(ops) - 1
        label = tree_label(op)

        if is_last and label in {"field", "fieldsel"}:
            field_name = expect_ident_token(op.children[0], "Field access")
            owner = current
            value = get_field_value(owner, field_name, env)

            def setter(new_value: Any, owner: Any = owner, field_name: str = field_name) -> None:
                set_field_value(owner, field_name, new_value, env, create=False)
                env.dot = new_value

            return RebindContext(value, setter)

        if is_last and label == "index":
            owner = current
            idx_value = evaluate_index_operand(op, env)
            value = index_value(owner, idx_value, env)

            def setter(new_value: Any, owner: Any = owner, idx_value: Any = idx_value) -> None:
                set_index_value(owner, idx_value, new_value, env)
                env.dot = new_value

            return RebindContext(value, setter)

        if is_last and label == "fieldfan":
            return build_fieldfan_context(current, op, env)

        current = apply_op(current, op, env)
        if isinstance(current, RebindContext):
            current = current.value

    raise ShakarRuntimeError("Increment target must end with a field or index")


def apply_numeric_delta(ref: RebindContext, delta: int) -> tuple[Any, Any]:
    """Increment/decrement the referenced numeric context and return (old, new)."""
    current = ref.value
    require_number(current)
    new_val = ShkNumber(current.value + delta)
    ref.setter(new_val)
    ref.value = new_val
    return current, new_val


def build_fieldfan_context(owner: Any, fan_node: Tree, env: Env) -> FanContext:
    """Construct a fan context for `.={a,b}` nodes so updates reach every slot."""
    fieldlist_node = child_by_label(fan_node, "fieldlist")
    if fieldlist_node is None:
        raise ShakarRuntimeError("Malformed field fan")
    names = [tok.value for tok in tree_children(fieldlist_node) if is_token_node(tok) and token_kind(tok) == "IDENT"]
    if not names:
        raise ShakarRuntimeError("Field fan requires at least one identifier")
    contexts: List[RebindContext] = []
    for name in names:
        value = get_field_value(owner, name, env)

        def setter(new_value: Any, owner: Any = owner, field_name: str = name) -> None:
            set_field_value(owner, field_name, new_value, env, create=False)
            env.dot = new_value

        contexts.append(RebindContext(value, setter))
    return FanContext(contexts)


def apply_fan_op(fan: FanContext, op: Tree, env: Env, *, apply_op: ApplyOpFunc) -> FanContext:
    """Apply an explicit-chain op to each fan target and keep contexts/values in sync."""
    new_contexts: List[RebindContext] = []
    new_values: List[Any] = []
    has_contexts = False
    has_values = False
    for ctx in fan.contexts:
        res = apply_op(ctx, op, env)
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
    env: Env,
    *,
    eval_func: EvalFunc,
    apply_op: ApplyOpFunc,
    assign_ident: AssignIdentFunc,
    evaluate_index_operand: IndexEvalFunc,
) -> Any:
    """Evaluate the `.= expr` apply-assign form (subject-aware updates)."""
    head, *ops = lvalue_node.children
    if not ops and token_kind(head) == "IDENT":
        target = env.get(head.value)
        rhs_env = Env(parent=env, dot=target)
        new_val = eval_func(rhs_node, rhs_env)
        assign_ident(head.value, new_val, env, create=False)
        return new_val
    target = eval_func(head, env)
    if isinstance(target, RebindContext):
        target = target.value
    if not ops:
        raise ShakarRuntimeError("Malformed apply-assign target")
    for op in ops[:-1]:
        target = apply_op(target, op, env)
    final_op = ops[-1]
    label = tree_label(final_op)
    match label:
        case "field" | "fieldsel":
            field_name = expect_ident_token(final_op.children[0], "Field assignment")
            old_val = get_field_value(target, field_name, env)
            rhs_env = Env(parent=env, dot=old_val)
            new_val = eval_func(rhs_node, rhs_env)
            set_field_value(target, field_name, new_val, env, create=False)
            return new_val
        case "lv_index":
            idx_val = evaluate_index_operand(final_op, env)
            old_val = index_value(target, idx_val, env)
            rhs_env = Env(parent=env, dot=old_val)
            new_val = eval_func(rhs_node, rhs_env)
            set_index_value(target, idx_val, new_val, env)
            return new_val
        case "fieldfan":
            fieldlist_node = child_by_label(final_op, "fieldlist")
            if fieldlist_node is None:
                raise ShakarRuntimeError("Malformed field fan-out list")
            names = [tok.value for tok in fieldlist_node.children if token_kind(tok) == "IDENT"]
            if not names:
                raise ShakarRuntimeError("Empty field fan-out list")
            results: List[Any] = []
            for name in names:
                old_val = get_field_value(target, name, env)
                rhs_env = Env(parent=env, dot=old_val)
                new_val = eval_func(rhs_node, rhs_env)
                set_field_value(target, name, new_val, env, create=False)
                results.append(new_val)
            return ShkArray(results)
    raise ShakarRuntimeError("Unsupported apply-assign target")


def assign_lvalue(
    node: Any,
    value: Any,
    env: Env,
    *,
    eval_func: EvalFunc,
    apply_op: ApplyOpFunc,
    assign_ident: AssignIdentFunc,
    evaluate_index_operand: IndexEvalFunc,
    create: bool,
) -> Any:
    """Assign to an lvalue, supporting fields, indices, and field fans."""
    if not is_tree_node(node) or tree_label(node) != "lvalue":
        raise ShakarRuntimeError("Invalid assignment target")
    if not node.children:
        raise ShakarRuntimeError("Empty lvalue")
    head, *ops = node.children
    if not ops and is_token_node(head) and token_kind(head) == "IDENT":
        return assign_ident(head.value, value, env, create=create)
    target = eval_func(head, env)
    if not ops:
        raise ShakarRuntimeError("Malformed lvalue")
    for op in ops[:-1]:
        target = apply_op(target, op, env)
    final_op = ops[-1]
    label = tree_label(final_op)
    match label:
        case "field" | "fieldsel":
            field_name = expect_ident_token(final_op.children[0], "Field assignment")
            return set_field_value(target, field_name, value, env, create=create)
        case "lv_index":
            idx_val = evaluate_index_operand(final_op, env)
            return set_index_value(target, idx_val, value, env)
        case "fieldfan":
            fieldlist_node = child_by_label(final_op, "fieldlist")
            if fieldlist_node is None:
                raise ShakarRuntimeError("Malformed field fan-out list")
            names = [tok.value for tok in tree_children(fieldlist_node) if token_kind(tok) == "IDENT"]
            if not names:
                raise ShakarRuntimeError("Empty field fan-out list")
            vals = fanout_values(value, len(names))
            for name, val in zip(names, vals):
                set_field_value(target, name, val, env, create=create)
            return value
    raise ShakarRuntimeError("Unsupported assignment target")


def read_lvalue(
    node: Any,
    env: Env,
    *,
    eval_func: EvalFunc,
    apply_op: ApplyOpFunc,
    evaluate_index_operand: IndexEvalFunc,
) -> Any:
    """Fetch the value behind an assignment target (e.g. for compound ops)."""
    if not is_tree_node(node) or tree_label(node) != "lvalue":
        raise ShakarRuntimeError("Invalid lvalue")
    if not node.children:
        raise ShakarRuntimeError("Empty lvalue")
    head, *ops = node.children
    if not ops and is_token_node(head) and token_kind(head) == "IDENT":
        return env.get(head.value)
    target = eval_func(head, env)
    if not ops:
        raise ShakarRuntimeError("Malformed lvalue")
    for op in ops[:-1]:
        target = apply_op(target, op, env)
    final_op = ops[-1]
    label = tree_label(final_op)
    match label:
        case "field" | "fieldsel":
            field_name = expect_ident_token(final_op.children[0], "Field value access")
            return get_field_value(target, field_name, env)
        case "lv_index":
            idx_val = evaluate_index_operand(final_op, env)
            return index_value(target, idx_val, env)
    raise ShakarRuntimeError("Compound assignment not supported for this target")


def resolve_rebind_lvalue(
    node: Any,
    env: Env,
    *,
    eval_func: EvalFunc,
    apply_op: ApplyOpFunc,
    assign_ident: AssignIdentFunc,
    evaluate_index_operand: IndexEvalFunc,
) -> RebindContext:
    """Turn a `rebind_lvalue` node into a concrete context for prefix rebinds."""
    if is_token_node(node):
        if token_kind(node) == "IDENT":
            return make_ident_context(node.value, env, assign_ident=assign_ident)
        raise ShakarRuntimeError("Malformed rebind target")
    if not is_tree_node(node):
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
        env,
        eval_func=eval_func,
        apply_op=apply_op,
        assign_ident=assign_ident,
        evaluate_index_operand=evaluate_index_operand,
    )
