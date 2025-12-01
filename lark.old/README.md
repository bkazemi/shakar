# lark.old/

This directory contains the legacy Lark-based parser (`parse_auto.py`) that has been **deprecated** in favor of the recursive-descent parser (`src/shakar_ref/parser_rd.py`).

## Current Status

**Lark deprecation is essentially complete.** The RD parser is now the default and handles all sanity tests successfully:
- All grammar/parser tests pass
- All AST scenario tests pass
- All runtime tests pass

## Remaining Usage

`parse_auto.py` is still used by:
- `sanity_treecheck.py`: Tree validation tests that specifically validate Lark parsing behavior and invariants

This is a specialized validation tool that tests parse tree structure, not a required dependency for the main codebase.

## Dependencies

`parse_auto.py` requires the `lark` package. All other code in the project uses local `Tree`/`Token` implementations from `src/shakar_ref/tree.py`.

## Future

`sanity_treecheck.py` could be rewritten to use the RD parser, at which point this directory and the Lark dependency can be removed entirely. However, since it's just a validation tool (not part of the main runtime), this is low priority.
