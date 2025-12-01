# lark.old/

This directory contains the legacy Lark-based parser (`parse_auto.py`) that has been deprecated in favor of the recursive-descent parser (`src/shakar_ref/parser_rd.py`).

## Current Status

The RD parser is now the default and handles all grammar/parser tests and runtime scenarios successfully.

## Temporary Usage

`parse_auto.py` is still used by:
- `sanity_check_basic.py`: AST scenario tests (6 tests) that require features not yet implemented in RD parser
- `sanity_treecheck.py`: Tree validation tests that specifically test Lark parsing behavior

## AST Test Gaps

The following 6 AST tests still require Lark's Earley parser:
1. `lambda-infer-zipwith` - Amp-lambda with inference
2. `lambda-hole-desugar` - Lambda hole desugaring
3. `lambda-dot-mix-error` - Lambda dot syntax error detection
4. `hook-inline-body` - Hook inline body handling
5. `decorator-ast-def` - Decorator AST definition
6. `decorated-fn-ast` - Decorated function AST

These will be fixed when the RD parser implements the corresponding features.

## Dependencies

`parse_auto.py` requires the `lark` package. All other code in the project uses local `Tree`/`Token` implementations from `src/shakar_ref/tree.py`.

## Future

Once the RD parser implements amp-lambda transforms, hooks, and decorators, this directory can be removed entirely.
