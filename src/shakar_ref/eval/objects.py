from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from ..tree import Tok
from ..token_types import TT

from ..runtime import Descriptor, Frame, ShkFn, ShkObject, ShkString, ShkValue, ShakarRuntimeError, ShakarTypeError
from ..tree import Tree, is_token, is_tree, tree_children, tree_label
from .common import expect_ident_token as _expect_ident_token, extract_param_names

EvalFunc = Callable[[Tree, Frame], ShkValue]


def eval_object(n: Tree, frame: Frame, eval_func: EvalFunc) -> ShkObject:
    """Build an object literal, installing descriptors/getters as needed."""
    slots: dict[str, ShkValue] = {}

    def _install_descriptor(name: str, getter: Optional[ShkFn]=None, setter: Optional[ShkFn]=None) -> None:
        existing = slots.get(name)

        if isinstance(existing, Descriptor):
            if getter is not None:
                existing.getter = getter
            if setter is not None:
                existing.setter = setter
            slots[name] = existing
        else:
            slots[name] = Descriptor(getter=getter, setter=setter)

    def _unwrap_ident(node: Tree) -> Optional[str]:
        cur = node
        seen = set()

        while is_tree(cur) and cur.children and id(cur) not in seen:
            seen.add(id(cur))

            if len(cur.children) != 1:
                break
            cur = cur.children[0]
        return cur.value if isinstance(cur, Tok) and cur.type == TT.IDENT else None

    def _maybe_method_signature(key_node: Optional[Tree]) -> Optional[Tuple[str, List[str]]]:
        if key_node is None or tree_label(key_node) != 'key_expr':
            return None
        if not is_tree(key_node):
            return None

        target = key_node.children[0] if key_node.children else None
        chain = None

        if is_tree(target):
            if tree_label(target) == 'explicit_chain':
                chain = target
            else:
                chain = next(
                    (ch for ch in tree_children(target) if tree_label(ch) == 'explicit_chain'),
                    None,
                )

        if chain is None or len(chain.children) != 2:
            return None

        head, call_node = chain.children
        name = _expect_ident_token(head, "Object method key")

        if tree_label(call_node) != 'call':
            return None

        args_node = call_node.children[0] if call_node.children else None
        params: List[str] = []

        if is_tree(args_node):
            queue = list(args_node.children)

            while queue:
                raw = queue.pop(0)

                raw_label = tree_label(raw)
                if raw_label in {'namedarg', 'kwarg'}:
                    return None

                raw_children = tree_children(raw) if is_tree(raw) else None
                if raw_children and raw_label not in {'args', 'arglist', 'arglistnamedmixed', 'argitem', 'arg'}:
                    queue.extend(raw_children)
                    continue

                ident = _unwrap_ident(raw)
                if ident is None:
                    if is_tree(raw):
                        queue.extend(tree_children(raw))
                    else:
                        return None
                else:
                    params.append(ident)
        elif args_node is not None:
            ident = _unwrap_ident(args_node)

            if ident is None:
                return None
            params.append(ident)
        return (name, params)

    def handle_item(item: Tree) -> None:
        match item.data:
            case 'obj_field':
                key_node, val_node = item.children
                method_sig = _maybe_method_signature(key_node)

                if method_sig:
                    name, params = method_sig
                    method_fn = ShkFn(params=params, body=val_node, frame=Frame(parent=frame))
                    slots[name] = method_fn
                    return
                key = eval_key(key_node, frame, eval_func)
                val = eval_func(val_node, frame)
                slots[str(key)] = val
            case 'obj_get':
                name_tok, body = item.children

                if name_tok is None:
                    raise ShakarRuntimeError("Getter missing name")

                key = _expect_ident_token(name_tok, "Getter name")
                getter_fn = ShkFn(params=None, body=body, frame=Frame(parent=frame))
                _install_descriptor(key, getter=getter_fn)
            case 'obj_set':
                name_tok, param_tok, body = item.children

                if name_tok is None or param_tok is None:
                    raise ShakarRuntimeError("Setter missing name or parameter")

                key = _expect_ident_token(name_tok, "Setter name")
                param_name = _expect_ident_token(param_tok, "Setter parameter")
                setter_fn = ShkFn(params=[param_name], body=body, frame=Frame(parent=frame))
                _install_descriptor(key, setter=setter_fn)
            case 'obj_method':
                name_tok, params_node, body = item.children

                if name_tok is None:
                    raise ShakarRuntimeError("Method missing name")
                method_name = _expect_ident_token(name_tok, "Method name")
                param_names, varargs = extract_param_names(params_node, context="method definition")
                method_fn = ShkFn(params=param_names, body=body, frame=Frame(parent=frame), vararg_indices=varargs)
                slots[method_name] = method_fn
            case 'obj_spread':
                spread_expr = item.children[0] if item.children else None
                if spread_expr is None:
                    raise ShakarRuntimeError("Object spread missing value")
                spread_val = eval_func(spread_expr, frame)
                if not isinstance(spread_val, ShkObject):
                    raise ShakarTypeError("Object spread expects an object value")
                for key, val in spread_val.slots.items():
                    slots[key] = val
            case _:
                raise ShakarRuntimeError(f"Unknown object item {item.data}")

    for child in tree_children(n):
        if is_token(child):
            continue

        child_label = tree_label(child)
        if child_label == 'object_items':
            for item in tree_children(child):
                if not is_tree(item) or tree_label(item) == 'obj_sep':
                    continue
                handle_item(item)
        else:
            if child_label == 'obj_sep':
                continue

            if is_tree(child):
                handle_item(child)
    return ShkObject(slots)


def eval_key(k: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue | str:
    label = tree_label(k)

    if label is not None:
        match label:
            case 'key_ident':
                t = k.children[0]
                if isinstance(t, Tok) and t.type == TT.IDENT:
                    return str(t.value)
                return eval_func(k, frame)
            case 'key_string':
                t = k.children[0]
                if not isinstance(t, Tok) or t.type != TT.STRING:
                    return eval_func(k, frame)
                s = str(t.value)

                if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
                    s = s[1:-1]
                return s
            case 'key_expr':
                v = eval_func(k.children[0], frame)
                return v.value if isinstance(v, ShkString) else v

    if is_token(k) and k.type in {TT.IDENT, TT.STRING}:
        return str(k.value).strip('"').strip("'")

    return eval_func(k, frame)
