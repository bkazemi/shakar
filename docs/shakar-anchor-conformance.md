# Shakar §4 — Anchor/Subject Conformance Tests (a01, 2025-08-17)

> These scenarios assert the **normative laws** in §4.6. Each block shows the **program** and the **expected observation** (`# expect:`).

## A. Anchor law
```shakar
# The first explicit subject in a grouping sets the anchor.
a and (b and .x()) and .y()
# expect: `.x()` is called with receiver `b`; `.y()` is called with receiver `a`
```

## B. Selector law (inside vs outside)
```shakar
users[ .len-1 ]
# expect: inside the selector, '.' == users (the base)

(users[0] and .name)
# expect: in this grouping, '.' == users[0]
```

## C. Leading-dot chain law
```shakar
user and (.profile[0].name.trim()) and .id
# expect: `.id` applies to `user` (the anchor), not to the result of `.trim()`
```

## D. Binder-shadowing law (loop × apply-assign)
```shakar
for[i] names:
  names[i] .= .trim()
# expect (RHS): '.' == old names[i]; (outside RHS): '.' == loop element for that iteration
```

## E. Prefix-rebind locality
```shakar
(=u .trim()).len > 0 and u.has("x")
# expect: within the expression, '.' == u; after it, '.' reverts to the enclosing anchor
# forbid: `=.` (prefix rebind requires an identifier)
```

## F. Guard head parens with named args
```shakar
# ok (standalone call)
send("bob", subject: "Hi")

# ok (guard head)
send("bob", subject: "Hi"): log("sent")
# expect: Calls already use parentheses, so guard heads remain unambiguous.
```

## G. Nil iteration
```shakar
xs := nil
for xs: do(.)
# expect: no-op (runtime nil value), but `for nil:` (literal) is a compile-time error
```

## H. Selector binds '.' to base (multi-selector)
```shakar
xs[ 1 : .len-1,  10:0:-2 ]
# expect: inside each selector, '.' == xs (the base); indices OOB clamp only for slices
```

## I. Hoisted loop binders (`^name`)
```shakar
for[^idx] 0..3: use(idx)
print(idx)
# expect: prints 2; if the loop ran 0 times, idx is introduced as nil

sum := 0
for[j, ^sum] arr:
  sum = sum + .
# expect: sum visible after loop; j is loop-local
```

## J. Illegals (must be rejected)
```shakar
. = 1             # illegal: '.' is not an lvalue
=.trim()          # illegal: prefix rebind requires an identifier
ready(user): start()             # ok
send "bob", subject: "Hi": log() # illegal: calls require parentheses
```
