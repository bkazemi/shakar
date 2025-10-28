from __future__ import annotations

from pathlib import Path
from lark import Lark
import sys
from shakar_parse_auto import build_parser  # use the project's builder (lexer remap + indenter)
from shakar_parse_auto import Prune
from shakar_lower import lower
from shakar_eval import eval_expr
from shakar_runtime import Env

def _read_grammar(grammar_path: str|None) -> str:
    if grammar_path:
        p = Path(grammar_path)
        if p.exists():
            return p.read_text(encoding="utf-8")
    # Fallback: sibling grammar.lark next to this file
    parent_dir = Path(__file__).resolve().parent.parent
    fallback = parent_dir / "grammar.lark"
    if fallback.exists():
        return fallback.read_text(encoding="utf-8")
    raise FileNotFoundError("grammar.lark not found. pass an explicit path")

def make_parser(grammar_path: str|None=None, use_indenter: bool=False, start_sym: str|None=None) -> Lark:
    g = _read_grammar(grammar_path)
    if start_sym is None:
        start_sym = "start_indented" if use_indenter else "start_noindent"
    # Earley + basic lexer when indenter is used; dynamic otherwise is inside build_parser already
    return build_parser(g, parser_kind="earley", use_indenter=use_indenter, start_sym=start_sym)

def run(src: str, grammar_path: str|None=None, use_indenter: bool=False) -> object:
    parser = make_parser(grammar_path, use_indenter=use_indenter)
    tree = parser.parse(src)
    ast = Prune().transform(tree)
    ast2 = lower(ast)
    # Unwrap start_* roots only when they contain a single child; otherwise keep
    # them so statement lists execute in order during evaluation.
    d = getattr(ast2, 'data', None)
    if d in ('start_noindent', 'start_indented'):
        children = getattr(ast2, 'children', None)
        if children and len(children) == 1:
            ast2 = children[0]

    return eval_expr(ast2, Env(source=src), source=src)

if __name__ == "__main__":
    # quick smoke
    #print(run("1 !is 2"))
    #print(run("[1,2,3].map&(x * x).to_list()"))
    #print(run("func(\"yo\")"))
    #print(run("1*1"))
    #print(run("&(.trim())"))
    #print(run("a = 1"))
    if len(sys.argv) > 1:
      print(run(sys.argv[1]))
