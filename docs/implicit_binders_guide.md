# Implicit Binders in Comprehensions: User Guide

## Overview

When a comprehension has no explicit `bind` or `over[...]` clause, Shakar automatically creates implicit binders from undefined identifiers found in the comprehension body and guard.

## How It Works

```shakar
# Explicit binding (clear and explicit)
result = [x * 2 over[x] src]

# Implicit binding (x is undefined, so auto-bound)
result = [x * 2 over src]
```

### Collection Rules

1. **All undefined identifiers** in the body and guard are collected as binders
2. **Body-first, then guard**: Body expressions are scanned left-to-right first, then guard expressions
3. **First-appearance order**: Within body or guard, names bind in the order they first appear
4. **Outer scope names** visible at the comprehension site are captured (lexical snapshot)
5. **Earlier locals count**: names defined earlier in the same block are captured
6. **Function defs are hoisted**: function names in the same block are visible even if declared later
7. **Arity must match** the iterable elements (or runtime error occurs)

### The Body-First Rule

This is the key to understanding binding order:

```shakar
pairs := [[1, 5], [2, 3]]

# Body mentions 'x', guard mentions 'y' => order: [x, y]
result := [x over pairs if y > 3]
# x=first element, y=second element ✓ Intuitive!

# Body mentions 'y', guard mentions 'x' => order: [y, x]
result := [y over pairs if x < 4]
# y=first element, x=second element (order follows code appearance)
```

**Why body-first?** The body expression declares *what you're working with*, so it gets priority. This makes the common case intuitive.

## Examples

### ✅ Single Implicit Binder
```shakar
src := [1, 2, 3]
result := [x * 2 over src]
# x binds to each element: [2, 4, 6]
```

### ✅ Multiple Implicit Binders
```shakar
pairs := [[1, 10], [2, 20], [3, 30]]
result := [a + b over pairs]
# a, b bind to pair elements: [11, 22, 33]
```

### ✅ Guard Contributes Binders
```shakar
pairs := [[1, 5], [2, 3]]
result := [x over pairs if y > 3]
# Creates TWO binders [x, y] from body and guard
# [1, 5]: x=1, y=5, check y>3 ✓ => include 1
# [2, 3]: x=2, y=3, check y>3 ✗ => skip
# result = [1]
```

### ⚠️ Be Mindful of Variable Names
```shakar
pairs := [[1, 5], [2, 3]]

# Works correctly: body-first gives natural order
result := [x over pairs if y > 3]
# Order: [x, y] => x=first, y=second
# [1, 5]: x=1, y=5, check y>3 ✓ => [1]

# Still works, but naming is misleading
result := [second over pairs if first < 4]
# Order: [second, first] (body-first!)
# second=elem[0], first=elem[1]
# [2, 3]: second=2, first=3, check 3<4 ✓ => [2]
```

**Tip**: Name variables according to their order of appearance in code, not their semantic role in the array.

### ❌ Arity Mismatch Error
```shakar
pairs := [[1, 2]]
# Body uses 'a' and 'b', guard uses 'c' => THREE binders
# But pairs only have TWO elements
result := [a + b over pairs if c > 0]
# ERROR: Destructure arity mismatch
```

### ⚠️ Outer Scope Capture
```shakar
x := 99
src := [1, 2, 3]
result := [x over src]
# 'x' exists in outer scope, so NOT an implicit binder
# result = [99, 99, 99]  ⚠️ Captures outer x, doesn't bind to elements!
```

## Best Practices

### ✅ DO: Use implicit binders for simple cases
```shakar
numbers := [1, 2, 3, 4, 5]
doubled := [x * 2 over numbers]
positives := [x over numbers if x > 0]

# Multi-variable is fine when order is clear
pairs := [[1, 10], [2, 20]]
sums := [a + b over pairs]
```

### ✅ DO: Use implicit subject for objects
```shakar
users := [{name: "alice", age: 25}, {name: "bob", age: 17}]
# Implicit subject (dot) is cleaner than destructuring
adults := [.name over users if .age >= 18]

# Alternative: explicit binding works too
adults := [u.name over [u] users if u.age >= 18]
```

### ✅ DO: Use explicit binding when clarity matters
```shakar
pairs := [[1, 5], [2, 3]]
# Explicit binding makes the mapping clear
result := [value over [key, value] pairs if key > 1]

# Especially useful when names don't follow appearance order
entries := [["alice", 25], ["bob", 17]]
adults := [name over [name, age] entries if age >= 18]
```

### ⚠️ CAREFUL: Multiple undefined names across body and guard
```shakar
pairs := [[1, 5], [2, 3]]
# This works but requires understanding body-first rule
result := [x over pairs if y > 4]  # x=first, y=second

# Consider explicit binding for complex filtering
result := [key over [key, value] pairs if value > 4]
```

## When Implicit Binding Fails

Implicit binding will **not** be used if:
- You provide explicit `bind` or `over[...]` clauses
- All identifiers resolve to outer scope (nothing undefined)

## Summary

**Implicit binding is convenient but requires care:**
- ✅ Great for simple single-variable comprehensions
- ⚠️ Watch out for order when mixing body and guard variables
- ❌ Can cause confusion with multiple undefined names
- 💡 When in doubt, use explicit binding for clarity

**Rule of thumb**: If you need to think about which variable binds to which position, use explicit binding instead.
