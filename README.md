**Shakar** is a WIP scripting language for the Python and Lua niche: small CLI tools and embeddable runtimes that want nicer expression syntax and subjectful flows without dragging in a giant runtime.

Shakar stays deliberately small: eager evaluation, dynamic types, exceptions, and a focused set of sugars for the kinds of things you do in scripts all the time (clean a value, check a few conditions, update a map or object, maybe bail).

The **source of truth** for the language is the [design notes](./shakar-design-notes.md). It describe the grammar, implicit subject rules for `.`, and the rest of the semantics in detail.

---

## Highlights

### Subjectful statements and apply-assign

Shakar has a first-class notion of a “statement subject”. A line that starts with `=LHS<tail>` evaluates `<tail>` with `.` bound to the current value of `LHS`, then writes the final result back to `LHS`:

```shakar
# boring, fully spelled
user.name = user.name.trim().title() ?? "guest"

# statement-subject sugar
=user.name.trim().title() ?? "guest"

line := "name=  Ada "
=line.split("=").last().trim().title()  # still assigns back to `line`

=(user).profile.contact.name.trim()  # only when you absolutely mean to rewrite `user`

Statement-subjects always begin the statement. The plain form (`=name<tail>`) already lets you walk fields or selectors—`=user.profile.email.trim()` writes back to `user.profile.email`. Grouping only exists so you can keep a different identifier as the destination while visiting another branch; `=(user)` says “no matter how deep this tail goes, the write still lands on `user`.” Use that escape hatch sparingly.
```

Apply-assign `.=` does the same thing at expression level: inside the right-hand side, `.` is the old value of the left-hand side and the result is written back in place, while also yielding the updated value:

```shakar
user.name .= .trim().title() ?? user.name

# fan-out over a selector
user.{name, email} .= ??(.trim().lower()) ?? "unknown"

cfg["db", default: {}]["host"] .= .trim() ?? "localhost"
```

Both forms are built on the same implicit-subject rules, which makes “update this thing based on its old value” cheap to write without making the flow mysterious.

### Fan-out helpers and loops

- Assignment fan-out: `=user.{name, email}.trim()` and `user.{first, last} .= .title()` walk a single base and apply the tail to each listed field (identifiers only on the LHS).
- Value fan-out: `user.{fullName(), email}` collects multiple projections from one base into an array and auto-spreads in positional calls: `send(user.{fullName(), email})` ⇒ `send(user.fullName(), user.email)`. Duplicate paths error; named args never auto-spread (wrap to pass as one argument).
- Fan-out block statement: `state{ .cur = .next; .x += 1; .name .= .trim() }` anchors `.` to `state`, runs clauses top→down, and errors on duplicate targets.
- `while` loops support inline or indented bodies and honor `break`/`continue`.

---

### Implicit subject and anchor stack

The implicit subject `.` only exists inside constructs that explicitly bind it (statement-subject `=LHS<tail>`, `.=` apply-assign, subjectful `for` loops, selectors, `await`, path-lambdas, and a few others). It never leaks across statements.

```shakar
=user.profile.settings.ensureDefaults()
user.name .= .trim()

for orders:
  .status == "open" and .total > 100:
    notify(.customer)
```

Within a grouping that has an anchor, leading-dot chains hang off that anchor:

```shakar
user and (.profile.name.trim()) and .id
# .profile... runs with subject = user
# .id still sees user, not the trimmed name
```

The anchor stack rules in the design notes spell out exactly where `.` comes from and when it is restored, so even heavy dot-chains still have predictable meaning.

---

### Punctuation guards

Guards replace most `if / elif / else` boilerplate with vertical chains. This keeps the “happy path” at the top and exception or retry paths lined up beneath it.

```shakar
ready():
  start()
| retries < 3:
  retry()
|:
  log("not ready")
```

Each `| expr:` is an `elif`, and `|:` is the final `else`. Guards are layout based and use the same colon-plus-indent blocks as the rest of the language. Single-line forms (`stmt if cond` and `if cond: stmt`) are still available when you want something terse.

---

### Nil-safe chains `??(...)` and coalescing `??`

Prefix `??(...)` evaluates a chain of calls or indexing and short-circuits to `nil` on the first failure. Infix `??` is a standard null-coalescing operator.

```shakar
nickname := ??(user.profile.nick.trim()) ?? "guest"
```

You get “try this whole chain, if anything in it falls apart use this fallback” as a single expression instead of hand-rolled `if` plus temporary variables.

---

### Comparison comma-chains (CCC)

CCC locks in a subject once, then lets you stream more comparisons with commas. You do not repeat the left operand, and you can still mix `and` and `or` in a controlled way.

```shakar
temp := read_temp()
temp > 40, < 60, != 55:
  fan.on()

# Mix joins explicitly; AND legs can inherit the comparator, OR legs must restate it
alert := temp > 70, or >= limits.max
ok    := temp >= 45, and <= 65, or == preferred

# Selector literals work inside CCC legs
level := sensor.level
level == `warn, error`, or >= `critical:`:
  notify(level)
```

- A chain begins with any comparison followed by a comma; the left operand is evaluated once and becomes the chain subject.
- Each comma leg is `[, and|or] [explicit comparator] Expr`. The joiner defaults to `and` and stays in effect until another `, or` or `, and` appears.
- When the joiner is `and`, you may omit the comparator to reuse the previous one. `, or` legs must spell the comparator they want (`temp > 5, or > limit`).
- Any bare `and` or `or` that is not preceded by a comma exits chain mode and resumes normal boolean parsing.
- The chain desugars to ordinary comparisons joined by the current sticky boolean operator, so short-circuit behavior and precedence match explicit `and` and `or`.

---

### Decorators (`@wrap`)

Attach lightweight wrappers without hand-writing higher-order functions. Decorator bodies see two implicit names: `f` (the next callable in the chain) and `args` (the current positional arguments as a mutable array). If the body finishes without `return`, the runtime automatically calls `f(args)` for you.

```shakar
log := []

decorator log_calls():
  log += [args[0]]

decorator with_prefix(prefix):
  args[0] = prefix + args[0]

@log_calls
@with_prefix("[prefix] ")
fn announce(msg):
  msg

announce("ready")   # => "[prefix] ready"
log                 # => ["ready"]
```

Stacking `@decorator` lines is deterministic: the decorator closest to the `fn` header runs first. Pass arguments normally (`@with_prefix(">> ")`). You can mutate `args`, replace it entirely, or `return` early to short-circuit.

---

### Selectors and bulk updates

Selectors let you address multiple fields or shaped regions of a structure in one shot, and they compose naturally with `.=` and other subjectful operations.

```shakar
# Fan-out updates over multiple fields
user.{name, email} .= ??(.trim().lower()) ?? "unknown"

# Nested defaults with a selector and default key
cfg["db", default: {}]["host"] .= .trim() ?? "localhost"

# CCC with selector literals
assert 4 < `5:10`, != `7, 8`
```

Selector literals use backticks and have precise semantics for ranges, lists, and membership; the design notes cover the details.

---

### Tooling and introspection

Shakar ships with a small reference parser and runtime, plus a `--tree` mode so you can inspect the AST of a file when you want to see exactly what your sugars are doing.

```bash
python shakar_parse_auto.py --tree path/to/file.shk
```

There is no hidden control flow beyond what is written in the spec, which keeps it friendly to formatters, linters, and AI-assisted refactors.

## Status / Roadmap

- **Spec + reference runtime**: this repo contains both; the spec lives in `docs/`.
- **Hooks / decorators / tracing**: hooks + decorators implemented; tracing is still tracked in the design notes.
- **Tree-sitter grammar**: maintained in `tree-sitter-shakar/`.
- **Tests**: run `python sanity_check_basic.py` after semantic changes; it exercises the grammar/runtime end-to-end.

### Running the reference interpreter

The Python runtime now lives under `src/shakar_ref/`. Run programs via:

```bash
PYTHONPATH=src python -m shakar_ref.runner path/to/program.shk
```

Use `-` instead of a path to read from stdin.

This is early research. Expect breaking changes while the surface syntax settles. Always treat the [design notes](./docs/shakar-design-notes.md) as the source of truth. Contributions that improve the spec, runtime, or tests are welcome :-)
