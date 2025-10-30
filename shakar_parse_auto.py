import sys
import argparse
from pathlib import Path
from lark import Lark, Transformer, Tree, UnexpectedInput, Token
from lark.visitors import Discard
from lark.indenter import Indenter

def pretty_inline(t, indent=""):
    lines = []
    if isinstance(t, Tree):
        if t.data == 'method' and t.children and isinstance(t.children[0], Token):
            # inline: method <name>
            head = f"{indent}{t.data} {t.children[0].value}"
            rest = t.children[1:]
            lines.append(head)
            for c in rest:
                lines.extend(pretty_inline(c, indent + "  "))
            return lines
        lines.append(f"{indent}{t.data}")
        for c in t.children:
            lines.extend(pretty_inline(c, indent + "  "))
        return lines
    else:
        # Token
        return [f"{indent}{t.type.lower()}  {t.value}"]

def validate_subject_scope(tree):
    SUBJECT_NODES = {"subject"}

    def walk(n, depth):
        if not isinstance(n, Tree):
            return
        d = n.data

        # bare dot
        if d in SUBJECT_NODES:
            if depth == 0:
                raise SyntaxError("bare '.' outside a binder/anchor context")
            return

        # .+ = RHS   (stmt form)
        if d == 'bind' and n.children:
            *head, rhs = n.children
            for ch in head: walk(ch, depth)
            walk(rhs, depth + 1)
            return

        # LHS .= RHS (expr form)
        if d == 'bindexpr' and len(n.children) == 2:
            lhs, rhs = n.children
            walk(lhs, depth)
            walk(rhs, depth + 1)
            return

        # await … : (inlinebody|indentblock)
        if d == 'awaitstmt':
            for ch in n.children:
                if isinstance(ch, Tree) and ch.data in ('inlinebody', 'indentblock'):
                    walk(ch, depth + 1)
                else:
                    walk(ch, depth)
            return

        # subjectful loops
        if d in ('forsubject', 'forindexed'):
            # last child is the body
            for i, ch in enumerate(n.children):
                walk(ch, depth + 1 if i == len(n.children) - 1 else depth)
            return

        # lambda callee sigil bodies
        if d in ('lambdacall1', 'lambdacalln'):
            # last child is the lambda body expr
            for i, ch in enumerate(n.children):
                walk(ch, depth + 1 if i == len(n.children) - 1 else depth)
            return

        # default
        for ch in n.children:
            walk(ch, depth)

    walk(tree, 0)

class ValidateSubjectScope(Transformer):
    """
    Enforce spec rule: bare '.' is only legal inside a binder/anchor context.
    Binders:
      - bindexpr (LHS .= RHS): subject during RHS
      - awaitstmt with trailing body: subject inside the body
      - forsubject / forindexed bodies: subject inside body
      - lambdacall1 / lambdacalln: subject inside the lambda body
      - stmtsubjectassign: subject while parsing its tail
    """
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.errors = []

    # helper to validate an arbitrary subtree under a temporary subject scope
    def _with_subject(self, subtree):
        self.depth += 1
        out = self.transform(subtree)
        self.depth -= 1
        return out

    # Node emitted by grammar for bare '.'
    def subject(self, children):
        if self.depth == 0:
            self.errors.append("bare '.' outside a binder/anchor context")
        return Tree('subject', children)

    # LHS .= RHS (as per your grammar: bindexpr: walrusexpr | lvalue '.=' bindexexpr)
    def bindexpr(self, children):
        # right-assoc chain: lvalue '.=' <bindexpr>
        if len(children) == 2:
            lhs, rhs = children
            # validate RHS under subject
            rhs_valid = self._with_subject(rhs)
            return Tree('bindexpr', [lhs, rhs_valid])
        return Tree('bindexpr', children)

    # await expr ':' (inlinebody | indentblock)
    def awaitstmt(self, children):
        # Find trailing body subtrees and validate them under subject
        new_children = []
        colon_seen = False
        for ch in children:
            if isinstance(ch, Tree) and ch.data in ('inlinebody', 'indentblock'):
                new_children.append(self._with_subject(ch))
                colon_seen = False
            else:
                new_children.append(ch)
        return Tree('awaitstmt', new_children)

    # Subjectful loops (your grammar has forsubject / forindexed)
    def forsubject(self, children):
        # body is last child after ':'
        if children:
            body = children[-1]
            children[-1] = self._with_subject(body)
        return Tree('forsubject', children)

    def forindexed(self, children):
        if children:
            body = children[-1]
            children[-1] = self._with_subject(body)
        return Tree('forindexed', children)

    # Lambda callee sigil forms; enable subject in the lambda body expr
    def lambdacall1(self, children):
        # callee '&' '(' expr ')'
        if children:
            children[-1] = self._with_subject(children[-1])
        return Tree('lambdacall1', children)

    def lambdacalln(self, children):
        # callee '&' '[' paramlist ']' '(' expr ')'
        if children:
            children[-1] = self._with_subject(children[-1])
        return Tree('lambdacalln', children)

    # Statement-subject assignment (=LHS tail)
    def stmtsubjectassign(self, children):
        if children:
            # convention: tail is last child
            children[-1] = self._with_subject(children[-1])
        return Tree('stmtsubjectassign', children)

class ValidateArity(Transformer):
    def destructure(self, c):
        if not c:
            return Tree('destructure', c)
        lhs, rhs = c[0], c[1] if len(c) > 1 else None
        if isinstance(lhs, Tree) and lhs.data == 'pattern_list' and isinstance(rhs, Tree) and rhs.data == 'tuple':
            lhs_count = len(lhs.children)
            rhs_count = len(rhs.children)
            if lhs_count != rhs_count:
                raise ValueError(f"Destructure arity mismatch: LHS has {lhs_count} names, RHS has {rhs_count} values")
        return Tree('destructure', c)

    def destructure_walrus(self, c):
        if not c:
            return Tree('destructure_walrus', c)
        lhs, rhs = c[0], c[1] if len(c) > 1 else None
        if isinstance(lhs, Tree) and lhs.data == 'pattern_list' and isinstance(rhs, Tree) and rhs.data == 'tuple':
            lhs_count = len(lhs.children)
            rhs_count = len(rhs.children)
            if lhs_count != rhs_count:
                raise ValueError(f"Destructure walrus arity mismatch: LHS has {lhs_count} names, RHS has {rhs_count} values")
        return Tree('destructure_walrus', c)

class ChainNormalize(Transformer):
    def _fuse(self, items):
      out, i = [], 0
      while i < len(items):
          t = items[i]
          if (isinstance(t, Tree) and getattr(t, 'data', None) == 'field'
              and i+1 < len(items) and isinstance(items[i+1], Tree)):
              nxt = items[i+1]
              if getattr(nxt, 'data', None) == 'call':
                  name = t.children[0]
                  out.append(Tree('method', [name, *nxt.children]))  # args live on call.children
                  i += 2
                  continue
              if getattr(nxt, 'data', None) in ('lambdacall1', 'lambdacalln'):
                  name = t.children[0]
                  # Normalize ampersand-lambda -> args(amp_lambda(...))
                  body = nxt.children[-1] if getattr(nxt, 'children', None) else None
                  param = None
                  for ch in getattr(nxt, 'children', []):
                      if isinstance(ch, Tree) and getattr(ch, 'data', None) == 'paramlist':
                          param = ch; break
                  lam_children = ([param] if param is not None else []) + ([body] if body is not None else [])
                  args_node = Tree('args', [Tree('amp_lambda', lam_children)])
                  out.append(Tree('method', [name, args_node]))
                  i += 2
                  continue
          out.append(t); i += 1
      return out

    def implicit_chain(self, c): return Tree('implicit_chain', self._fuse(c))
    def explicit_chain(self, c): return Tree('explicit_chain', self._fuse(c))

class Prune(Transformer):
    # ---- unified object literal ----
    def object(self, c):
        from lark import Tree
        return Tree('object', c)

    # Canonicalize fields to a single node shape: obj_field(key, value)
    def obj_field_ident(self, c):
        from lark import Tree
        # c = [IDENT, expr]
        return Tree('obj_field', [Tree('key_ident', [c[0]]), c[1]])

    def obj_field_string(self, c):
        from lark import Tree
        # c = [STRING, expr]
        return Tree('obj_field', [Tree('key_string', [c[0]]), c[1]])

    def obj_field_expr(self, c):
        from lark import Tree
        # c = [expr_key, expr_val]
        return Tree('obj_field', [Tree('key_expr', [c[0]]), c[1]])

    # Keep getters/setters explicit; prune keeps only key + body shape later if you want
    def obj_get(self, c):
        from lark import Tree
        name = None
        body = c[-1] if c else None
        for part in c:
            if isinstance(part, Token) and getattr(part, "type", "") == "IDENT":
                name = part
                break
        return Tree('obj_get', [name, body])

    def obj_set(self, c):
        from lark import Tree
        name = None
        param = None
        body = c[-1] if c else None
        for part in c:
            if isinstance(part, Token) and getattr(part, "type", "") == "IDENT":
                if name is None:
                    name = part
                elif param is None:
                    param = part
        return Tree('obj_set', [name, param, body])

    def obj_method(self, c):
        from lark import Tree
        name = None
        params = None
        body = c[-1] if c else None
        for part in c[:-1]:
            if isinstance(part, Token) and getattr(part, "type", "") == "IDENT":
                if name is None:
                    name = part
            elif isinstance(part, Tree) and getattr(part, "data", "") == "paramlist":
                params = part
        return Tree('obj_method', [name, params, body])

    def key_ident(self, c):
        from lark import Tree
        return Tree('key_ident', c)

    def key_string(self, c):
        from lark import Tree
        return Tree('key_string', c)

    def key_expr(self, c):
        from lark import Tree
        return Tree('key_expr', c)

    def setcomp(self, c):
        from lark import Tree, Token
        items = [x for x in c if not (isinstance(x, Token) and getattr(x, "type", "") == "SET")]
        return Tree('setcomp', items)

    def lambdacall1(self, c):
        body = c[-1] if c else None
        # unify postfix &(...) into a normal call with an amp_lambda arg
        return Tree('call', [Tree('args', [Tree('amp_lambda', [body])])])

    def lambdacalln(self, c):
        params, body = c[0], c[-1]
        return Tree('call', [Tree('args', [Tree('amp_lambda', [params, body])])])

    def array(self, c):
      elems = []
      for node in c:
          # keep only real child expressions/values; drop punctuation/whitespace markers
          if isinstance(node, Token):
            # Skip tokens like [, ], , and any newline tokens you pass through
            if node.type in ("LSQB", "RSQB", "COMMA", "_NL"):
              continue
          elems.append(node)

      return Tree('array', elems)

    def start_indented(self, c):
        """Normalize neutral expressions parsed under indented start:
        - Fuse [IDENT, implicit/explicit chain] into a single explicit_chain
        - For lambdacall callee with an implicit/explicit chain, prepend the IDENT head
        - Strip the 'start_indented' wrapper by returning the inner expression
        """
        c = [x for x in c if x is not Discard]

        if len(c) == 2 and isinstance(c[0], Token) and getattr(c[0], 'type', None) == 'IDENT' and isinstance(c[1], Tree) and c[1].data == 'call':
          # fuse root [IDENT, call(...)] into explicit_chain(IDENT, call)
          return Tree('explicit_chain', [c[0], c[1]])

        if len(c) == 2 and isinstance(c[0], Token) and getattr(c[0], 'type', None) == 'IDENT' and isinstance(c[1], Tree):
            head_ident = c[0]
            t = c[1]
            # Case 1: bare implicit/explicit chain at root
            if t.data in ('implicit_chain', 'explicit_chain'):
                return Tree('explicit_chain', [head_ident, *t.children])
            # Case 2: lambdacallN/1 where callee holds an implicit/explicit chain
            if t.data == 'postfixexpr' and t.children and isinstance(t.children[0], Tree) and t.children[0].data in ('lambdacall1','lambdacalln'):
                lc = t.children[0]
                if lc.children and isinstance(lc.children[0], Tree) and lc.children[0].data == 'callee':
                    callee = lc.children[0]
                    if callee.children and isinstance(callee.children[0], Tree) and callee.children[0].data in ('implicit_chain','explicit_chain'):
                        chain = callee.children[0]
                        callee.children[0] = Tree('explicit_chain', [head_ident, *chain.children])
                        return t
        # Default: collapse wrapper if singleton
        return self._keep_or_flatten('start_indented', c)

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
        if isinstance(head, Tree) and getattr(head, 'data', None) == 'explicit_chain':
            chain, rest = head, c[1:]
        else:
            chain, rest = Tree('explicit_chain', [head]), c[1:]

        callnode = None
        for node in rest:
            if isinstance(node, Tree) and getattr(node, 'data', None) == 'call':
                callnode = node
                break
            if isinstance(node, Tree) and getattr(node, 'data', None) in ('args','arglist','arglistnamedmixed'):
                # Wrap raw args into a call node
                callnode = Tree('call', [node] if node.data == 'args' else list(node.children))
                break

        if callnode is None:
            # Not a direct one-liner call; leave shape unchanged
            return Tree('simplecall', c)

        # Prefer the canonical chain fuse if available, otherwise build directly
        try:
          # XXX
          #return ChainNormalize._fuse(chain, callnode)  # keeps chain invariants identical to expr-mode
          raise NameError()
        except NameError:
            return Tree('explicit_chain', list(chain.children) + [callnode])

    # value-level lambdas
    def amp_lambda1(self, c):
        return Tree('amp_lambda', [c[0]]) # body only; unary implicit '.'

    def amp_lambdan(self, c):
        params, body = c
        return Tree('amp_lambda', [params, body]) # explicit paramlist + body

    # Prune
    def slicearm_empty(self, _):    # unify all “missing expr” to a single sentinel
        from lark import Tree
        return Tree('emptyexpr', [])

    def slicearm_expr(self, c):
        return c[0]                 # unwrap the expr

    def slicesel(self, c):
        from lark import Tree
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
    def bindexpr(self, c):      return self._keep_or_flatten('bindexpr', c, 'bind')
    def walrusexpr(self, c):    return self._keep_or_flatten('walrusexpr', c, 'walrus')
    def nullishexpr(self, c):   return self._keep_or_flatten('nullishexpr', c, 'nullish')
    def compareexpr(self, c):   return self._keep_or_flatten('compareexpr', c, 'compare')
    def addexpr(self, c):       return self._keep_or_flatten('addexpr', c, 'add')
    def mulexpr(self, c):       return self._keep_or_flatten('mulexpr', c, 'mul')
    def powexpr(self, c):       return self._keep_or_flatten('powexpr', c, 'pow')
    def unaryexpr(self, c):     return self._keep_or_flatten('unaryexpr', c, 'unary')

    # precedence scaffolding: collapse when singleton; keep & rename when operative
    def ternaryexpr_nc(self, c):   return self._keep_or_flatten('ternaryexpr_nc', c, 'ternary_nc')
    def orexpr_nc(self, c):        return self._keep_or_flatten('orexpr_nc', c, 'or_nc')
    def andexpr_nc(self, c):       return self._keep_or_flatten('andexpr_nc', c, 'and_nc')
    def bindexpr_nc(self, c):      return self._keep_or_flatten('bindexpr_nc', c, 'bind_nc')
    def walrusexpr_nc(self, c):    return self._keep_or_flatten('walrusexpr_nc', c, 'walrus_nc')
    def nullishexpr_nc(self, c):   return self._keep_or_flatten('nullishexpr_nc', c, 'nullish_nc')
    def compareexpr_nc(self, c):   return self._keep_or_flatten('compareexpr_nc', c, 'compare_nc')
    def addexpr_nc(self, c):       return self._keep_or_flatten('addexpr_nc', c, 'add_nc')
    def mulexpr_nc(self, c):       return self._keep_or_flatten('mulexpr_nc', c, 'mul_nc')
    def powexpr_nc(self, c):       return self._keep_or_flatten('powexpr_nc', c, 'pow_nc')
    def unaryexpr_nc(self, c):     return self._keep_or_flatten('unaryexpr_nc', c, 'unary_nc')

    # collapse explicit head when it's just a single, non-hop child
    def explicit_chain(self, c):
        if len(c) == 1:
            child = c[0]
            if not (isinstance(child, Tree) and child.data in {'field', 'index', 'call', 'incr', 'decr'}):
                return child
        return Tree('explicit_chain', c)
    def implicit_chain(self, c): return Tree('implicit_chain', c)
    def field(self, c):
        # Normalize field to hold only the IDENT token (drop DOT), so printers show the name
        ident_only = [tok for tok in c if getattr(tok, 'type', None) == 'IDENT']
        return Tree('field', ident_only or c)

    def index(self, c):  return Tree('index', c)
    def call(self, c):   return Tree('call', c)

    # light unwrapping that’s always safe
    def primary(self, c):       return c[0] if len(c) == 1 else Tree('primary', c)
    def expr(self, c):          return c[0]  # expr is always a wrapper in this grammar

    # ignore comments
    def COMMENT(self, c): return Discard
    def comment(self, c): return Discard

    def AND(self, c): return Discard
    def OR(self, c): return Discard

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
    if not isinstance(tree, Tree) or getattr(tree, "data", None) != "start_indented":
        return

    src = code.replace("\r\n", "\n")

    # Collect (start_pos, line) for top-level stmtlists only
    starts = []
    for ch in tree.children:
        if isinstance(ch, Tree) and ch.data == "stmtlist":
            m = getattr(ch, "meta", None)
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
    def is_namedarg(n): return isinstance(n, Tree) and n.data == "namedarg"
    def unwrap(n):
        if isinstance(n, Tree) and n.data in ("argitem","arg"): return n.children[0] if n.children else n
        return n
    def visit(n):
        if not isinstance(n, Tree): return
        if n.data == "call":
            argnode = None
            for ch in n.children:
                if isinstance(ch, Tree) and ch.data in ("arglistnamedmixed","arglist","args"):
                    argnode = ch; break
            if argnode is not None:
                seen_named = False
                names = set()
                for raw in argnode.children:
                    a = unwrap(raw)
                    if is_namedarg(a):
                        seen_named = True
                        if not a.children or not isinstance(a.children[0], Token): continue
                        name = a.children[0].value
                        if name in names: raise SyntaxError(f"Duplicate named argument '{name}'")
                        names.add(name)
                    else:
                        if seen_named: raise SyntaxError("Positional argument after named argument")
        for ch in getattr(n, "children", ()): visit(ch)
    visit(tree)



def validate_hoisted_binders(tree):
    def hoist_info_from_node(n):
        res = []
        if isinstance(n, Tree) and n.data == "forindexed":
            for ch in n.children:
                if isinstance(ch, Tree) and ch.data == "hoist":
                    if ch.children and isinstance(ch.children[0], Token):
                        res.append((ch.children[0].value, True))
                    else:
                        res.append(("<anon>", True))
                    break
                if isinstance(ch, Token) and getattr(ch, "type", "") == "IDENT":
                    res.append((ch.value, False))
                    break
        if isinstance(n, Tree) and n.data == "formap2":
            seen = 0
            for ch in n.children:
                if isinstance(ch, Tree) and ch.data == "hoist":
                    if ch.children and isinstance(ch.children[0], Token):
                        res.append((ch.children[0].value, True)); seen += 1
                elif isinstance(ch, Token) and getattr(ch, "type", "") == "IDENT":
                    res.append((ch.value, False)); seen += 1
                if seen >= 2: break
        if isinstance(n, Tree) and n.data == "overspec":
            for ch in n.children:
                if isinstance(ch, Tree) and ch.data == "binderlist":
                    for item in ch.children:
                        if isinstance(item, Tree) and item.data == "hoist":
                            if item.children and isinstance(item.children[0], Token):
                                res.append((item.children[0].value, True))
                        elif isinstance(item, Token) and getattr(item, "type", "") == "IDENT":
                            res.append((item.value, False))
        return res

    def walk(n):
        if not isinstance(n, Tree):
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
        for ch in getattr(n, "children", ()):
            walk(ch)
    walk(tree)


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

    tried = []

    def try_parse(use_indenter: bool):
        start_sym = "start_indented" if use_indenter else "start_noindent"
        p = build_parser(grammar_text, parser_kind, use_indenter, start_sym)
        tried.append(("indenter" if use_indenter else "no-indenter", parser_kind))
        return parse_or_first_error(p, code), start_sym

    # Auto strategy: try no-indenter first (IDE-like), then indenter if needed or looks like a block
    try:
        if force is None:
            use_indenter = False
            if looks_like_offside(code):
                use_indenter = True
            try:
                tree, start_sym = try_parse(use_indenter)
            except:
                tree, start_sym = try_parse(not use_indenter)
        else:
            tree, start_sym = try_parse(force)
        #_enforce_toplevel_line_separators(tree, code, start_sym)
    except UnexpectedInput as e:
        sys.stderr.write("[ERROR] Parse failed in modes tried: %s\n" % (", ".join(f"{m}/{k}" for m,k in tried)))
        ctx = e.get_context(code, span=80)
        saw_tok = getattr(e, "token", None)
        saw = f"{saw_tok.type} {repr(str(saw_tok))}" if saw_tok else "EOF"
        exp = ", ".join(sorted(getattr(e, "expected", [])))
        sys.stderr.write(f"[ParseError] line {e.line}, col {e.column}\n{ctx}\nSaw: {saw}\nExpected: {exp}\n")
        import sys as _sys
        _sys.exit(1)

    if args.tree:
        if args.prune: tree = Prune().transform(tree)
        validate_named_args(tree)
        validate_hoisted_binders(tree)
        if args.normalize: tree = ChainNormalize().transform(tree)
        tree = ArgTidy().transform(tree)
        print("\n".join(pretty_inline(tree)))
    else:
        validate_subject_scope(tree)
        print("OK")
if __name__ == "__main__":
    main()
