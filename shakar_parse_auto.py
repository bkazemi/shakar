import sys
import argparse
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple

from lark import Lark, Transformer, Tree, UnexpectedInput, Token
from lark.visitors import Discard, v_args
from lark.indenter import Indenter

try:  # prefer package import when available
    from rt.shakar_tree import (
        is_tree,
        is_token,
        tree_label,
        tree_children,
        node_meta,
        child_by_label,
        child_by_labels,
        first_child,
    )
except ImportError:  # fallback for direct execution
    from shakar_tree import (
        is_tree,
        is_token,
        tree_label,
        tree_children,
        node_meta,
        child_by_label,
        child_by_labels,
        first_child,
    )

def pretty_inline(t, indent=""):
    lines = []
    if is_tree(t):
        children = tree_children(t)
        if tree_label(t) == 'method' and children and is_token(children[0]):
            # inline: method <name>
            head = f"{indent}{tree_label(t)} {children[0].value}"
            rest = children[1:]
            lines.append(head)
            for c in rest:
                lines.extend(pretty_inline(c, indent + "  "))
            return lines
        lines.append(f"{indent}{tree_label(t)}")
        for c in children:
            lines.extend(pretty_inline(c, indent + "  "))
        return lines
    else:
        # Token
        return [f"{indent}{t.type.lower()}  {t.value}"]

def enforce_subject_scope(tree: Tree) -> None:
    """Validate that bare '.' only appears inside an anchor/binder context."""
    errors: List[str] = []

    def visit(node: Any, depth: int) -> None:
        if not is_tree(node):
            return
        label = tree_label(node)
        children = list(tree_children(node))

        if label == 'subject':
            if depth == 0:
                errors.append("bare '.' outside a binder/anchor context")
            return

        if label == 'bindexpr' and len(children) == 2:
            visit(children[0], depth)
            visit(children[1], depth + 1)
            return

        if label in {'awaitstmt', 'hook'}:
            for ch in children:
                if is_tree(ch) and tree_label(ch) in {'inlinebody', 'indentblock'}:
                    visit(ch, depth + 1)
                else:
                    visit(ch, depth)
            return

        if label in {'forsubject', 'forindexed'} and children:
            for ch in children[:-1]:
                visit(ch, depth)
            visit(children[-1], depth + 1)
            return

        if label in {'lambdacall1', 'lambdacalln', 'stmtsubjectassign'} and children:
            for ch in children[:-1]:
                visit(ch, depth)
            visit(children[-1], depth + 1)
            return

        for ch in children:
            visit(ch, depth)

    visit(tree, 0)
    if errors:
        raise SyntaxError(errors[0])

class ChainNormalize(Transformer):
    @staticmethod
    def _fuse(items):
        out: List[Any] = []
        i = 0
        while i < len(items):
            node = items[i]
            if (
                is_tree(node) and tree_label(node) == 'field'
                and i + 1 < len(items) and is_tree(items[i + 1])
            ):
                nxt = items[i + 1]
                if tree_label(nxt) == 'call':
                    name = node.children[0]
                    out.append(Tree('method', [name, *nxt.children]))
                    i += 2
                    continue
                if tree_label(nxt) in {'lambdacall1', 'lambdacalln'}:
                    name = node.children[0]
                    body = nxt.children[-1] if nxt.children else None
                    params = next((ch for ch in nxt.children if is_tree(ch) and tree_label(ch) == 'paramlist'), None)
                    lam_children = ([params] if params is not None else []) + ([body] if body is not None else [])
                    args_node = Tree('args', [Tree('amp_lambda', lam_children)])
                    out.append(Tree('method', [name, args_node]))
                    i += 2
                    continue
            out.append(node)
            i += 1
        return out

    def implicit_chain(self, c):
        return Tree('implicit_chain', self._fuse(c))

    def explicit_chain(self, c):
        return Tree('explicit_chain', self._fuse(c))

class Prune(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compare_depth = 0

    def _transform_tree(self, tree):
        if is_tree(tree) and tree_label(tree) in {'compareexpr', 'compareexpr_nc'}:
            self._compare_depth += 1
            try:
                return super()._transform_tree(tree)
            finally:
                self._compare_depth -= 1
        return super()._transform_tree(tree)
    # ---- unified object literal ----
    def object(self, c):
        return Tree('object', c)

    # Canonicalize fields to a single node shape: obj_field(key, value)
    def obj_field_ident(self, c):
        # c = [IDENT, expr]
        return Tree('obj_field', [Tree('key_ident', [c[0]]), c[1]])

    def obj_field_string(self, c):
        # c = [STRING, expr]
        return Tree('obj_field', [Tree('key_string', [c[0]]), c[1]])

    def obj_field_expr(self, c):
        # c = [expr_key, expr_val]
        return Tree('obj_field', [Tree('key_expr', [c[0]]), c[1]])

    def obj_sep(self, c):
        return Discard

    # Keep getters/setters explicit; prune keeps only key + body shape later if you want
    def obj_get(self, c):
        name = None
        body = c[-1] if c else None
        for part in c:
            if is_token(part) and getattr(part, "type", "") == "IDENT":
                name = part
                break
        return Tree('obj_get', [name, body])

    def obj_set(self, c):
        name = None
        param = None
        body = c[-1] if c else None
        for part in c:
            if is_token(part) and getattr(part, "type", "") == "IDENT":
                if name is None:
                    name = part
                elif param is None:
                    param = part
        return Tree('obj_set', [name, param, body])

    def obj_method(self, c):
        name = None
        params = None
        body = c[-1] if c else None
        for part in c[:-1]:
            if is_token(part) and getattr(part, "type", "") == "IDENT":
                if name is None:
                    name = part
            elif is_tree(part) and tree_label(part) == 'paramlist':
                params = part
        return Tree('obj_method', [name, params, body])

    def key_ident(self, c):
        return Tree('key_ident', c)

    def key_string(self, c):
        return Tree('key_string', c)

    def key_expr(self, c):
        return Tree('key_expr', c)

    def group_expr(self, c):
        return Tree('group', c)

    def setliteral(self, c):
        items = [x for x in c if not (is_token(x) and x.type == 'COMMA')]
        return Tree('setliteral', items)

    def setliteral_empty(self, _):
        return Tree('setliteral', [])

    def setcomp(self, c):
        items = [x for x in c if not (is_token(x) and getattr(x, "type", "") == "SET")]
        return Tree('setcomp', items)

    def hook(self, c):
        event_name = None
        body = None
        for node in c:
            if is_token(node) and getattr(node, "type", "") == "STRING":
                event_name = node
            elif is_tree(node) and tree_label(node) in {'inlinebody', 'indentblock'}:
                body = node
        children: list[Any] = []
        if event_name is not None:
            children.append(event_name)
        if body is not None:
            children.append(Tree('amp_lambda', [body]))
        else:
            children.append(Tree('amp_lambda', [Tree('inlinebody', [])]))
        return Tree('hook', children)

    def lambdacall1(self, c):
        body = c[-1] if c else None
        # unify postfix &(...) into a normal call with an amp_lambda arg
        return Tree('call', [Tree('args', [Tree('amp_lambda', [body])])])

    def lambdacalln(self, c):
        params, body = c[0], c[-1]
        return Tree('call', [Tree('args', [Tree('amp_lambda', [params, body])])])

    def array(self, c):
        filtered = [
            node
            for node in c
            if not (is_token(node) and node.type in {"LSQB", "RSQB", "COMMA", "_NL"})
        ]
        return Tree('array', filtered)

    def start_indented(self, children):
        """Normalize neutral expressions parsed under indented start."""
        items = [x for x in children if x is not Discard]
        if not items:
            return Discard

        if len(items) == 2 and self._is_ident_token(items[0]):
            merged = self._merge_ident_head(items[0], items[1])
            if merged is not None:
                return merged

        if (
            len(items) == 2
            and is_tree(items[0]) and tree_label(items[0]) == 'explicit_chain'
            and is_tree(items[1]) and tree_label(items[1]) == 'explicit_chain'
        ):
            return Tree('explicit_chain', list(items[0].children) + list(items[1].children))

        return self._keep_or_flatten('start_indented', items)

    @staticmethod
    def _is_ident_token(node: Any) -> bool:
        return is_token(node) and getattr(node, 'type', None) == 'IDENT'

    def _merge_ident_head(self, ident: Token, node: Any) -> Optional[Tree]:
        if is_tree(node):
            label = tree_label(node)
            children = tree_children(node)
            if label in {'implicit_chain', 'explicit_chain'}:
                return Tree('explicit_chain', [ident, *children])
            if label == 'call':
                return Tree('explicit_chain', [ident, node])
            if label == 'amp_lambda':
                return Tree('explicit_chain', [ident, Tree('call', [Tree('args', [node])])])
            if label == 'postfixexpr' and children:
                first = children[0]
                if is_tree(first) and tree_label(first) in {'lambdacall1', 'lambdacalln'}:
                    callee = next((ch for ch in tree_children(first) if is_tree(ch) and tree_label(ch) == 'callee'), None)
                    if callee:
                        callee_children = tree_children(callee)
                        if callee_children:
                            chain = callee_children[0]
                            if is_tree(chain) and tree_label(chain) in {'implicit_chain', 'explicit_chain'}:
                                callee.children[0] = Tree('explicit_chain', [ident, *chain.children])
                                return node
        return None

    # helper
    def _keep_or_flatten(self, name, children, alias=None):
        return children[0] if len(children) == 1 else Tree(alias or name, children)

    def stmt(self, c):
      c = [x for x in c if x is not Discard]
      if not c:
        return Discard

      return self._keep_or_flatten('stmt', c)

    def stmtlist(self, c):
      c = [x for x in c if x is not Discard]
      if not c:
        return Discard

      return self._keep_or_flatten('stmtlist', c)

    def simplecall(self, c):
        """Normalize one-liner call statements to the same explicit_chain+call
        shape as expression-mode calls. Safe no-op if it isn't a direct call.
        """
        if not c:
            return Tree('simplecall', c)

        head = c[0]
        if is_tree(head) and tree_label(head) == 'callee' and head.children:
            head = head.children[0]
        if is_tree(head) and tree_label(head) == 'explicit_chain':
            chain, rest = head, c[1:]
        else:
            chain, rest = Tree('explicit_chain', [head]), c[1:]

        callnode = None
        for node in rest:
            if is_tree(node) and tree_label(node) == 'call':
                callnode = node
                break
            if is_tree(node) and tree_label(node) in {'args', 'arglist', 'arglistnamedmixed'}:
                # Wrap raw args into a call node
                callnode = Tree('call', [node] if tree_label(node) == 'args' else list(node.children))
                break

        if callnode is None:
            callnode = Tree('call', [])

        # Prefer the canonical chain fuse if available, otherwise build directly
        try:
          # XXX
          #return ChainNormalize._fuse(chain, callnode)  # keeps chain invariants identical to expr-mode
          raise NameError()
        except NameError:
            return Tree('explicit_chain', list(chain.children) + [callnode])

    def fnstmt(self, c):
        name = None
        params = None
        body = None
        for node in c:
            if is_token(node) and getattr(node, "type", "") == "IDENT" and name is None:
                name = node
            elif is_tree(node) and tree_label(node) == 'paramlist':
                params = node
            elif is_tree(node) and tree_label(node) in {'inlinebody', 'indentblock'}:
                body = node
        children: list[Any] = []
        if name is not None:
            children.append(name)
        if params is not None:
            children.append(params)
        if body is not None:
            children.append(body)
        return Tree('fndef', children)

    def deferstmt(self, c):
        label = None
        deps: List[Any] = []
        body_node = None
        for node in c:
            if is_tree(node):
                tag = tree_label(node)
                if tag == 'deferlabel' and label is None:
                    label = _first_ident(node)
                    continue
                if tag == 'deferafter':
                    deps.extend(_collect_defer_after(node))
                    continue
                if tag == 'defer_block':
                    block = self._transform_tree(node.children[0])
                    body_node = Tree('deferblock', [block])
                else:
                    body_node = self._transform_tree(node)
        children: List[Any] = []
        if label is not None:
            children.append(Tree('deferlabel', [Token('IDENT', label)]))
        if body_node is not None:
            children.append(body_node)
        if deps:
            children.append(Tree('deferdeps', deps))
        return Tree('deferstmt', children)

    def returnstmt(self, c):
        exprs: List[Any] = []
        for node in c:
            if is_token(node):
                continue
            exprs.append(self._transform_tree(node) if is_tree(node) else node)
        return Tree('returnstmt', exprs)

    def amp_lambda1(self, c):
        return Tree('amp_lambda', [c[0]]) # body only; unary implicit '.'

    def amp_lambdan(self, c):
        params, body = c
        return Tree('amp_lambda', [params, body]) # explicit paramlist + body

    # Prune
    def slicearm_empty(self, _):    # unify all “missing expr” to a single sentinel
        return Tree('emptyexpr', [])

    def slicearm_expr(self, c):
        return c[0]                 # unwrap the expr

    def slicesel(self, c):
        arms = list(c)
        if len(arms) == 2:          # no third colon → empty step
            arms.append(Tree('emptyexpr', []))
        return Tree('slicesel', arms)   # exactly [start, stop, step]

    # XXX: make sure this doesn't break stuff
    #def addop(self, c): return c[0]     # PLUS | MINUS
    #def mulop(self, c): return c[0]     # STAR | SLASH | PERCENT, etc
    #def powop(self, c): return c[0]     # POW or '**'
    #def literal(self, c): return c[0]   # (from the previous error)

    # precedence scaffolding: collapse when singleton; keep & rename when operative
    def ternaryexpr(self, c):   return self._keep_or_flatten('ternaryexpr', c, 'ternary')
    def orexpr(self, c):        return self._keep_or_flatten('orexpr', c, 'or')
    def andexpr(self, c):       return self._keep_or_flatten('andexpr', c, 'and')
    def bindexpr(self, c):
        filtered = [item for item in c if not (is_token(item) and getattr(item, 'type', None) == 'APPLYASSIGN')]
        return self._keep_or_flatten('bindexpr', filtered, 'bind')
    def walrusexpr(self, c):    return self._keep_or_flatten('walrusexpr', c, 'walrus')
    def nullishexpr(self, c):   return self._keep_or_flatten('nullishexpr', c, 'nullish')
    def compareexpr(self, c):
        return self._keep_or_flatten('compareexpr', c, 'compare')
    def addexpr(self, c):       return self._keep_or_flatten('addexpr', c, 'add')
    def mulexpr(self, c):       return self._keep_or_flatten('mulexpr', c, 'mul')
    def powexpr(self, c):       return self._keep_or_flatten('powexpr', c, 'pow')
    def unaryexpr(self, c):     return self._keep_or_flatten('unaryexpr', c, 'unary')

    # precedence scaffolding: collapse when singleton; keep & rename when operative
    def ternaryexpr_nc(self, c):   return self._keep_or_flatten('ternaryexpr_nc', c, 'ternary_nc')
    def orexpr_nc(self, c):        return self._keep_or_flatten('orexpr_nc', c, 'or_nc')
    def andexpr_nc(self, c):       return self._keep_or_flatten('andexpr_nc', c, 'and_nc')
    def bindexpr_nc(self, c):
        filtered = [item for item in c if not (is_token(item) and getattr(item, 'type', None) == 'APPLYASSIGN')]
        return self._keep_or_flatten('bindexpr_nc', filtered, 'bind_nc')
    def walrusexpr_nc(self, c):    return self._keep_or_flatten('walrusexpr_nc', c, 'walrus_nc')
    def nullishexpr_nc(self, c):   return self._keep_or_flatten('nullishexpr_nc', c, 'nullish_nc')
    def compareexpr_nc(self, c):
        return self._keep_or_flatten('compareexpr_nc', c, 'compare_nc')
    def addexpr_nc(self, c):       return self._keep_or_flatten('addexpr_nc', c, 'add_nc')
    def mulexpr_nc(self, c):       return self._keep_or_flatten('mulexpr_nc', c, 'mul_nc')
    def powexpr_nc(self, c):       return self._keep_or_flatten('powexpr_nc', c, 'pow_nc')
    def unaryexpr_nc(self, c):     return self._keep_or_flatten('unaryexpr_nc', c, 'unary_nc')

    # collapse explicit head when it's just a single, non-hop child
    def explicit_chain(self, c):
        if len(c) == 1:
            child = c[0]
            if not (is_tree(child) and tree_label(child) in {'field', 'index', 'call', 'incr', 'decr', 'fieldfan'}):
                return child
        return Tree('explicit_chain', c)
    def implicit_chain(self, c): return Tree('implicit_chain', c)
    def field(self, c):
        # Normalize field to hold only the IDENT token (drop DOT), so printers show the name
        ident_only = [tok for tok in c if getattr(tok, 'type', None) == 'IDENT']
        return Tree('field', ident_only or c)

    def bind(self, c):
        kept = []
        for item in c:
            if is_token(item) and getattr(item, 'type', None) == 'APPLYASSIGN':
                continue
            kept.append(item)
        return Tree('bind', kept)

    def index(self, c):  return Tree('index', c)
    def call(self, c):   return Tree('call', c)

    # light unwrapping that’s always safe
    def primary(self, c):       return c[0] if len(c) == 1 else Tree('primary', c)
    def expr(self, c):          return c[0]  # expr is always a wrapper in this grammar

    # ignore comments
    def COMMENT(self, c): return Discard
    def comment(self, c): return Discard

    def AND(self, c):
        if getattr(self, '_compare_depth', 0) > 0:
            return Token('AND', 'and')
        return Discard

    def OR(self, c):
        if getattr(self, '_compare_depth', 0) > 0:
            return Token('OR', 'or')
        return Discard

    def COLON(self, c): return Discard

# tidy arg nodes for printing
class ArgTidy(Transformer):
    def arg(self, c):        return c[0]              # unwrap single arg
    def argitem(self, c):    return c[0]
    def arglist(self, c):    return Tree('args', c)   # nicer label
    def arglistnamedmixed(self, c): return Tree('args', c)

def _enforce_toplevel_line_separators(tree, code: str, start_sym: str):
    """
    Forbid two *top-level* stmtlists on the same physical source line without a semicolon.
    Conservative and meta-spill-proof: only uses each child's *start* position/line.
    """

    if start_sym != "start_indented":
        return
    if not is_tree(tree) or tree_label(tree) != "start_indented":
        return

    src = code.replace("\r\n", "\n")

    # Collect (start_pos, line) for top-level stmtlists only
    starts = []
    for ch in tree_children(tree):
        if is_tree(ch) and tree_label(ch) == "stmtlist":
            m = node_meta(ch)
            if m and hasattr(m, "start_pos") and hasattr(m, "line"):
                starts.append((m.start_pos, m.line))
    if len(starts) < 2:
        return

    def line_start(pos: int) -> int:
        i = src.rfind("\n", 0, pos)
        return 0 if i == -1 else i + 1

    for (a_s, a_line), (b_s, b_line) in zip(starts, starts[1:]):
        # If they start on different lines, it's fine.
        if a_line != b_line:
            continue

        # Same physical line: require a semicolon somewhere before B on that line.
        b_line_start = line_start(b_s)
        slice_before_b_on_line = src[b_line_start:b_s]
        if ";" in slice_before_b_on_line:
            continue  # explicitly separated on the same line

        # Otherwise it's true same-line adjacency without ';' -> error at B
        e = UnexpectedInput(None)
        e.line = b_line
        e.column = (b_s - b_line_start) + 1
        e.token = None
        e.expected = {"SEMI"}
        raise e

class ShakarIndenter(Indenter):
    # Match your grammar
    NL_type = "_NL"
    INDENT_type = "INDENT"
    DEDENT_type = "DEDENT"
    OPEN_PAREN_types = ["LPAR", "LSQB", "LBRACE"]
    CLOSE_PAREN_types = ["RPAR", "RSQB", "RBRACE"]
    tab_len = 8

KEYWORDS = {
    "and": "AND",
    "or":  "OR",
    "is":  "IS",
    "in":  "IN",
    "not": "NOT",
    "for": "FOR",
    "if": "IF",
    "elif": "ELIF",
    "else": "ELSE",
    "await": "AWAIT",
    "any": "ANY",
    "all": "ALL",
    "using": "USING",
    "over": "OVER",
    "hook": "HOOK",
    "fn": "FN",
    "defer": "DEFER",
    "dbg": "DBG",
    "return": "RETURN",
    "assert": "ASSERT",
    "get": "GET",
    "set": "SET",
}

def _remap_ident(t: Token) -> Token:
    # Only remap exact word matches, never prefixes
    t.type = KEYWORDS.get(t.value, t.type)
    return t

def build_parser(grammar_text: str, parser_kind: str, use_indenter: bool, start_sym: str):
    if use_indenter:
        return Lark(
            grammar_text,
            parser=parser_kind,
            lexer="basic",
            postlex=ShakarIndenter(),
            start=start_sym,
            maybe_placeholders=False,
            propagate_positions=True,
            lexer_callbacks={"IDENT": _remap_ident},
        )
    else:
      return Lark(
          grammar_text,
          parser=parser_kind,
          lexer="basic",
          start=start_sym,
          maybe_placeholders=False,
          propagate_positions=True,
          lexer_callbacks={"IDENT": _remap_ident},
      )
    '''else: # dynamic ver
        return Lark(
            grammar_text,
            parser=parser_kind,
            lexer="dynamic",
            start=start_sym,
            maybe_placeholders=False,
            propagate_positions=True,
        )'''

def looks_like_offside(code: str) -> bool:
    # Heuristic: a colon at end-of-line followed by a newline, or leading indentation on a non-empty line
    lines = code.splitlines()
    for i, ln in enumerate(lines):
        if ln.rstrip().endswith(":") and i < len(lines)-1 and lines[i+1].startswith((" ", "\t")):
            return True
        if ln and (ln[0] == " " or ln[0] == "\t"):
            return True
    return False

def parse_or_first_error(parser, src: str):
    """
    Try to parse and, on failure, re-raise the *first* UnexpectedInput encountered
    so that the caller (main) can print a concise, consistent error message with
    e.get_context(...), Saw, and Expected.
    """
    try:
        return parser.parse(src)

    except UnexpectedInput as whole_err:
        acc = ""
        for i, line in enumerate(src.splitlines(keepends=True), 1):
            acc += line
            try:
                parser.parse(acc)
            except UnexpectedInput as e:
                # Re-raise the earliest failing prefix error
                raise e
        # If all prefixes were ok, re-raise the original (usually EOF/dedent problem)
        raise whole_err

def validate_named_args(tree):
    def is_namedarg(n):
        return is_tree(n) and tree_label(n) == "namedarg"

    def unwrap(n):
        if is_tree(n) and tree_label(n) in {"argitem", "arg"}:
            return tree_children(n)[0] if tree_children(n) else n
        return n

    def visit(n):
        if not is_tree(n):
            return
        if tree_label(n) == "call":
            argnode = None
            for ch in tree_children(n):
                if is_tree(ch) and tree_label(ch) in {"arglistnamedmixed", "arglist", "args"}:
                    argnode = ch; break
            if argnode is not None:
                seen_named = False
                names = set()
                for raw in tree_children(argnode):
                    a = unwrap(raw)
                    if is_namedarg(a):
                        seen_named = True
                        children = tree_children(a)
                        if not children or not is_token(children[0]):
                            continue
                        name = children[0].value
                        if name in names: raise SyntaxError(f"Duplicate named argument '{name}'")
                        names.add(name)
                    else:
                        if seen_named: raise SyntaxError("Positional argument after named argument")
        for ch in tree_children(n):
            visit(ch)
    visit(tree)



def validate_hoisted_binders(tree):
    def hoist_info_from_node(n):
        res = []
        if is_tree(n) and tree_label(n) == "forindexed":
            for ch in tree_children(n):
                if is_tree(ch) and tree_label(ch) == "hoist":
                    children = tree_children(ch)
                    if children and is_token(children[0]):
                        res.append((ch.children[0].value, True))
                    else:
                        res.append(("<anon>", True))
                    break
                if is_token(ch) and getattr(ch, "type", "") == "IDENT":
                    res.append((ch.value, False))
                    break
        if is_tree(n) and tree_label(n) == "formap2":
            seen = 0
            for ch in tree_children(n):
                if is_tree(ch) and tree_label(ch) == "hoist":
                    children = tree_children(ch)
                    if children and is_token(children[0]):
                        res.append((ch.children[0].value, True)); seen += 1
                elif is_token(ch) and getattr(ch, "type", "") == "IDENT":
                    res.append((ch.value, False)); seen += 1
                if seen >= 2: break
        if is_tree(n) and tree_label(n) == "overspec":
            for ch in tree_children(n):
                if is_tree(ch) and tree_label(ch) == "binderlist":
                    for item in tree_children(ch):
                        if is_tree(item) and tree_label(item) == "hoist":
                            children = tree_children(item)
                            if children and is_token(children[0]):
                                res.append((item.children[0].value, True))
                        elif is_token(item) and getattr(item, "type", "") == "IDENT":
                            res.append((item.value, False))
        return res

    def walk(n):
        if not is_tree(n):
            return
        pairs = hoist_info_from_node(n)
        if pairs:
            byname = {}
            for name, is_h in pairs:
                base = name
                byname.setdefault(base, set()).add("H" if is_h else "P")
            for base, kinds in byname.items():
                if kinds == {"H", "P"}:
                    raise SyntaxError(f"Cannot use both hoisted and local binder for '{base}' in the same binder list")
                if sum(1 for nm, is_h in pairs if nm == base and is_h) > 1:
                    raise SyntaxError(f"Duplicate hoisted binder '{base}' in binder list")
        for ch in tree_children(n):
            walk(ch)
    walk(tree)


def parse_source(
    code: str,
    grammar_text: str,
    *,
    parser_kind: str = "earley",
    force_indenter: Optional[bool] = None,
) -> Tuple[Tree, str]:
    """Parse source text and return the raw Lark tree and start symbol."""

    tried: List[tuple[str, str]] = []

    def try_parse(use_indenter: bool) -> Tuple[Tree, str]:
        start_sym = "start_indented" if use_indenter else "start_noindent"
        parser = build_parser(grammar_text, parser_kind, use_indenter, start_sym)
        tried.append((("indenter" if use_indenter else "no-indenter"), parser_kind))
        return parse_or_first_error(parser, code), start_sym

    def auto_mode() -> Tuple[Tree, str]:
        use_indenter = looks_like_offside(code)
        try:
            return try_parse(use_indenter)
        except UnexpectedInput:
            return try_parse(not use_indenter)

    try:
        if force_indenter is None:
            return auto_mode()
        return try_parse(force_indenter)
    except UnexpectedInput as err:
        tried_modes = ", ".join(f"{mode}/{kind}" for mode, kind in tried)
        ctx = err.get_context(code, span=80)
        token = getattr(err, "token", None)
        saw = f"{token.type} {repr(str(token))}" if token else "EOF"
        expected = ", ".join(sorted(getattr(err, "expected", [])))
        raise SyntaxError(
            f"[ParseError] modes tried: {tried_modes}\n"
            f"line {err.line}, col {err.column}\n{ctx}\nSaw: {saw}\nExpected: {expected}"
        ) from err


def parse_to_ast(
    code: str,
    grammar_text: str,
    *,
    parser_kind: str = "earley",
    force_indenter: Optional[bool] = None,
    normalize: bool = False,
) -> Tree:
    """Parse source and return the canonical AST used by the runtime."""

    tree, _ = parse_source(
        code,
        grammar_text,
        parser_kind=parser_kind,
        force_indenter=force_indenter,
    )
    enforce_subject_scope(tree)
    validate_named_args(tree)
    validate_hoisted_binders(tree)
    pruned = Prune().transform(tree)
    if normalize:
        pruned = ChainNormalize().transform(pruned)
    normalized = _normalize_root(pruned)
    canonical = _canonicalize_ast(normalized)
    canonical = _desugar_call_holes(canonical)
    canonical = _infer_amp_lambda_params(canonical)
    if not is_tree(canonical):
        canonical = Tree('module', [canonical])
    return canonical

_CANONICAL_RENAMES = {
    'expr_nc': 'expr',
    'ternary_nc': 'ternary',
    'or_nc': 'or',
    'and_nc': 'and',
    'bind_nc': 'bind',
    'walrus_nc': 'walrus',
    'nullish_nc': 'nullish',
    'compare_nc': 'compare',
    'add_nc': 'add',
    'mul_nc': 'mul',
    'pow_nc': 'pow',
    'unary_nc': 'unary',
}

_FLATTEN_SINGLE = {'expr'}

_IGNORED_TOKEN_TYPES = {'SEMI', '_NL', 'INDENT', 'DEDENT'}

# Canonical AST node expectations for downstream consumers. Every Tree returned
# by parse_to_ast adheres to these shapes:
#   module(head, ...)            - single root node grouping the program
#   stmtlist(stmt, ...)          - ordered statements (no Discard children)
#   explicit_chain(head, op...)  - head expression followed by postfix ops
#   fieldfan(fieldlist)          - fan-out fields with IDENT tokens only
#   call(args)                   - call nodes always contain a single args tree
#   args(expr, ...)              - positional arguments in evaluation order
#   walrus/assign/bind/etc.      - keep grammar names but operands flattened
# The runtime relies on these invariants to avoid runtime tree_shape probing.
def _normalize_root(tree: Tree) -> Tree:
    label = tree_label(tree)
    if label not in {"start_noindent", "start_indented"}:
        return tree
    children = [child for child in tree_children(tree) if child is not Discard]
    if len(children) == 1 and is_tree(children[0]) and tree_label(children[0]) == "stmtlist":
        stmtlist = _strip_discard(children[0])
    else:
        stmtlist = _strip_discard(Tree("stmtlist", children))
    return Tree("module", [stmtlist])

def _strip_discard(node: Tree) -> Tree:
    kept = [child for child in tree_children(node) if child is not Discard]
    return Tree(node.data, kept)

def _desugar_call_holes(node: Any) -> Any:
    if is_token(node) or not is_tree(node):
        return node
    transformed_children = [_desugar_call_holes(child) for child in tree_children(node)]
    new_node = Tree(tree_label(node), transformed_children)
    if tree_label(node) == 'explicit_chain':
        replacement = _chain_to_lambda_if_holes(new_node)
        if replacement is not None:
            return replacement
    return new_node

def _chain_to_lambda_if_holes(chain: Tree) -> Tree | None:
    holes: List[str] = []

    def clone(node: Any) -> Any:
        if is_token(node):
            return node
        if not is_tree(node):
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

def _infer_amp_lambda_params(node: Any) -> Any:
    if is_token(node) or not is_tree(node):
        return node
    label = tree_label(node)
    if label == 'amp_lambda' and len(node.children) == 1:
        body = _infer_amp_lambda_params(node.children[0])
        names, uses_subject = _collect_lambda_free_names(body)
        if uses_subject and names:
            raise SyntaxError("Cannot mix subject '.' with implicit parameters in amp_lambda body")
        if uses_subject or not names:
            node.children = [body]
        else:
            params = [Token('IDENT', name) for name in names]
            node.children = [Tree('paramlist', params), body]
        return node
    node.children = [_infer_amp_lambda_params(child) for child in tree_children(node)]
    return node

def _collect_lambda_free_names(node: Any) -> tuple[List[str], bool]:
    names: List[str] = []
    uses_subject = False

    def append(name: str) -> None:
        if name not in names:
            names.append(name)

    def walk(n: Any, parent_label: Optional[str]) -> None:
        nonlocal uses_subject
        if is_tree(n):
            label = tree_label(n)
            if label == 'amp_lambda':
                return
            if label in {'implicit_chain', 'subject'}:
                uses_subject = True
            if label == 'explicit_chain':
                children = tree_children(n)
                if children:
                    head = children[0]
                    ident = _get_ident_value(head)
                    if ident is not None:
                        append(ident)
                    else:
                        walk(head, label)
                    for tail in children[1:]:
                        walk(tail, label)
                return
            for idx, child in enumerate(tree_children(n)):
                walk(child, label)
            return
        if is_token(n) and getattr(n, 'type', None) == 'IDENT':
            if parent_label in {'field', 'paramlist', 'key_ident', 'key_string'}:
                return
            append(n.value)

    walk(node, None)
    return names, uses_subject

def _get_ident_value(node: Any) -> Optional[str]:
    if is_token(node) and getattr(node, 'type', None) == 'IDENT':
        return node.value
    return None

def _canonicalize_ast(node: Any) -> Any:
    if is_token(node):
        return None if node.type in _IGNORED_TOKEN_TYPES else node
    if not is_tree(node):
        return node
    label = tree_label(node)
    renamed = _CANONICAL_RENAMES.get(label, label)
    new_children: List[Any] = []
    for child in tree_children(node):
        canon = _canonicalize_ast(child)
        if canon is None:
            continue
        new_children.append(canon)
    if renamed in _FLATTEN_SINGLE and len(new_children) == 1:
        return new_children[0]
    return Tree(renamed, new_children)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", help="Path to a source file (defaults to stdin)")
    ap.add_argument("-g", "--grammar", default="grammar.lark", help="Path to grammar.lark")
    ap.add_argument("--earley", action="store_true", help="Use Earley (default)")
    ap.add_argument("--lalr", action="store_true", help="Use LALR")
    ap.add_argument("--indenter", action="store_true", help="Force Indenter ON")
    ap.add_argument("--no-indenter", action="store_true", help="Force Indenter OFF")
    ap.add_argument("--tree", action="store_true", help="Print parse tree")
    ap.add_argument("--prune", action="store_true", help="Prune precedence wrapper nodes")
    ap.add_argument("--normalize", action="store_true", help="Fuse [field, call] → method in chains")

    args = ap.parse_args()

    parser_kind = "earley" if (args.earley or not args.lalr) else "lalr"
    grammar_text = Path(args.grammar).read_text(encoding="utf-8")
    code = Path(args.source).read_text(encoding="utf-8") if args.source else sys.stdin.read()

    # Decide mode
    force = None
    if args.indenter:
        force = True
    elif args.no_indenter:
        force = False

    try:
        tree, start_sym = parse_source(
            code,
            grammar_text,
            parser_kind=parser_kind,
            force_indenter=force,
        )
        #_enforce_toplevel_line_separators(tree, code, start_sym)
    except SyntaxError as err:
        sys.stderr.write(str(err) + "\n")
        sys.exit(1)

    if args.tree:
        if args.prune:
            tree = Prune().transform(tree)
        enforce_subject_scope(tree)
        validate_named_args(tree)
        validate_hoisted_binders(tree)
        if args.normalize:
            tree = ChainNormalize().transform(tree)
        tree = ArgTidy().transform(tree)
        print("\n".join(pretty_inline(tree)))
    else:
        enforce_subject_scope(tree)
        print("OK")

def _prune_assert(self, c):
    filtered = [node for node in c if not (is_token(node) and getattr(node, "type", "") == "ASSERT")]
    node = Tree('assert', filtered)
    meta = getattr(c[0], 'meta', None) if c else None
    if meta is not None:
        node.meta = meta
    return node

setattr(Prune, 'assert', _prune_assert)

if __name__ == "__main__":
    main()

def _first_ident(node: Any) -> str | None:
    queue = [node]
    while queue:
        cur = queue.pop(0)
        if is_token(cur) and getattr(cur, "type", "") == "IDENT":
            return cur.value
        if is_tree(cur):
            queue.extend(tree_children(cur))
    return None

def _collect_defer_after(node: Tree) -> List[Token]:
    deps: List[Token] = []
    for ch in tree_children(node):
        name = _first_ident(ch)
        if name:
            deps.append(Token('IDENT', name))
    return deps
