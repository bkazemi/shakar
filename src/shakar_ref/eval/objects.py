from __future__ import annotations

from typing import Any, List

from lark import Tree, Token

from ..runtime import Descriptor, Frame, ShkFn, ShkObject, ShkString, ShakarRuntimeError
from ..tree import child_by_label, is_token_node, is_tree_node, tree_children, tree_label
from .common import expect_ident_token as _expect_ident_token

EvalFunc = callable

def eval_object(n: Tree, frame: Frame, eval_func) -> ShkObject:
    """Build an object literal, installing descriptors/getters as needed."""
    slots: dict[str, Any] = {}

    def _install_descriptor(name: str, getter: ShkFn | None=None, setter: ShkFn | None=None) -> None:
        existing = slots.get(name)

        if isinstance(existing, Descriptor):
            if getter is not None:
                existing.getter = getter
            if setter is not None:
                existing.setter = setter
            slots[name] = existing
        else:
            slots[name] = Descriptor(getter=getter, setter=setter)

    def _extract_params(params_node: Tree | None) -> List[str]:
        if params_node is None:
            return []

        names: List[str] = []
        queue = list(tree_children(params_node))
        while queue:
            node = queue.pop(0)
            ident = _unwrap_ident(node)

            if ident is not None:
                names.append(ident)
                continue

            if is_tree_node(node):
                queue.extend(tree_children(node))
        return names

    def _unwrap_ident(node: Any) -> str | None:
        cur = node
        seen = set()

        while is_tree_node(cur) and cur.children and id(cur) not in seen:
            seen.add(id(cur))

            if len(cur.children) != 1:
                break
            cur = cur.children[0]
        return cur.value if isinstance(cur, Token) and cur.type == 'IDENT' else None

    def _maybe_method_signature(key_node: Any) -> tuple[str, List[str]] | None:
        if tree_label(key_node) != 'key_expr':
            return None
        target = key_node.children[0] if key_node.children else None
        chain = None

        if is_tree_node(target):
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

        if is_tree_node(args_node):
            queue = list(args_node.children)

            while queue:
                raw = queue.pop(0)

                raw_label = tree_label(raw)
                if raw_label in {'namedarg', 'kwarg'}:
                    return None

                raw_children = tree_children(raw) if is_tree_node(raw) else None
                if raw_children and raw_label not in {'args', 'arglist', 'arglistnamedmixed', 'argitem', 'arg'}:
                    queue.extend(raw_children)
                    continue

                ident = _unwrap_ident(raw)
                if ident is None:
                    if is_tree_node(raw):
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

                key = name_tok.value
                getter_fn = ShkFn(params=None, body=body, frame=Frame(parent=frame))
                _install_descriptor(key, getter=getter_fn)
            case 'obj_set':
                name_tok, param_tok, body = item.children

                if name_tok is None or param_tok is None:
                    raise ShakarRuntimeError("Setter missing name or parameter")

                key = name_tok.value
                setter_fn = ShkFn(params=[param_tok.value], body=body, frame=Frame(parent=frame))
                _install_descriptor(key, setter=setter_fn)
            case 'obj_method':
                name_tok, params_node, body = item.children

                if name_tok is None:
                    raise ShakarRuntimeError("Method missing name")
                param_names = _extract_params(params_node)
                method_fn = ShkFn(params=param_names, body=body, frame=Frame(parent=frame))
                slots[name_tok.value] = method_fn
            case _:
                raise ShakarRuntimeError(f"Unknown object item {item.data}")

    for child in tree_children(n):
        if is_token_node(child):
            continue

        child_label = tree_label(child)
        if child_label == 'object_items':
            for item in tree_children(child):
                if not is_tree_node(item) or tree_label(item) == 'obj_sep':
                    continue
                handle_item(item)
        else:
            if child_label == 'obj_sep':
                continue

            if is_tree_node(child):
                handle_item(child)
    return ShkObject(slots)

def eval_key(k: Any, frame: Frame, eval_func) -> Any:
    label = tree_label(k)

    if label is not None:
        match label:
            case 'key_ident':
                t = k.children[0]; return t.value
            case 'key_string':
                t = k.children[0]; s = t.value

                if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
                    s = s[1:-1]
                return s
            case 'key_expr':
                v = eval_func(k.children[0], frame)
                return v.value if isinstance(v, ShkString) else v

    if is_token_node(k) and k.type in ('IDENT', 'STRING'):
        return k.value.strip('"').strip("'")

    return eval_func(k, frame)
