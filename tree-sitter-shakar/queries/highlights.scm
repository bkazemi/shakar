; Tree-sitter highlight queries for Shakar

;; Keywords
(nil) @constant.builtin
(boolean) @constant.builtin

[
  "if" "elif" "else"
  "for" "in" "over" "bind"
  "await" "any" "all"
  "using" "hook" "defer"
  "return" "assert" "dbg"
  "break" "continue" "throw"
  "decorator" "fn"
  "and" "or" "not" "is"
  "catch" "set" "get"
] @keyword

(catch_expression "catch" @keyword)
(catch_statement "catch" @keyword)
(catch_expression "=>" @operator)
(catch_sugar_expression "@@" @operator)
(catch_sugar_expression "=>" @operator)

(guard_chain "|" @operator)
(guard_chain "|:" @operator)

;; Punctuation (mirror python defaults)
[
  "("
  ")"
  "["
  "]"
  "{"
  "}"
] @punctuation.bracket

[
  ","
  ":"
] @punctuation.delimiter

;; Default selector
(default_selector "default" @keyword)

;; Subject placeholder (implicit dot) — distinct capture, with a fallback delimiter color
((subject_expression ".") @shakar.subject @punctuation.delimiter)
(subject_expression) @shakar.subject
(field_expression "." @punctuation.delimiter)
(field_fan "." @punctuation.delimiter)

;; Selector literals: backtick delimiters
(selector_literal "`" @punctuation.special)

;; Identifiers inside selector literals (allow separate styling, e.g. italics)
(selector_literal (selector (primary_expression (identifier) @shakar.selector)))
(selector_literal (selector (slice_selector (primary_expression (identifier)) @shakar.selector)))

;; Literals
(string) @string
(raw_string) @string.special
(raw_hash_string) @string.special
(shell_string) @string.special
(number) @number

;; Comments
(comment) @comment

;; Identifiers
(identifier) @variable
(field_expression name: (identifier) @property)

((identifier) @type
  (#match? @type "^[A-Z].*[a-z]"))

((identifier) @constant
  (#match? @constant "^[A-Z][A-Z0-9_]*$"))

;; Assignment statements that start with '=' (rebind)
(rebind_statement "=" @operator)

; Decorators (match Python behavior: bold @ and decorator name)
((decorator_entry "@" @attribute)
  (#set! "priority" 110))

((decorator_entry decorator: (_) @attribute)
  (#set! "priority" 110))

;; Null-safe call marker (distinguish from infix ?? operator)
((nullsafe_expression "??") @punctuation.special)

; CCC separator commas
(ccc_separator) @operator

;; await[any]/await[all] keywords and timeout
(await_any_statement "await" @keyword)
(await_all_statement "await" @keyword)
(await_any_expression "await" @keyword)
(await_all_expression "await" @keyword)
(await_any_statement "[" @punctuation.bracket)
(await_all_statement "[" @punctuation.bracket)
(await_any_expression "[" @punctuation.bracket)
(await_all_expression "[" @punctuation.bracket)
(await_any_statement "any" @keyword)
(await_all_statement "all" @keyword)
(await_any_expression "any" @keyword)
(await_all_expression "all" @keyword)
(await_any_statement "]" @punctuation.bracket)
(await_all_statement "]" @punctuation.bracket)
(await_any_expression "]" @punctuation.bracket)
(await_all_expression "]" @punctuation.bracket)
(await_any_statement "(" @punctuation.bracket)
(await_any_statement ")" @punctuation.bracket)
(await_all_statement "(" @punctuation.bracket)
(await_all_statement ")" @punctuation.bracket)
(await_any_expression "(" @punctuation.bracket)
(await_any_expression ")" @punctuation.bracket)
(await_all_expression "(" @punctuation.bracket)
(await_all_expression ")" @punctuation.bracket)
(await_arm "timeout" @keyword)
(await_arm name: (identifier) @variable.parameter)

;; Ternary
(conditional_expression "?" @operator)
(conditional_expression ":" @operator)

;; Operators
[
  (assignment_expression "=" @operator)
  (assignment_expression "+=" @operator)
  (assignment_expression "-=" @operator)
  (assignment_expression "*=" @operator)
  (assignment_expression "/=" @operator)
  (assignment_expression "//=" @operator)
  (assignment_expression "%=" @operator)
  (assignment_expression "**=" @operator)
  (assignment_expression "<<=" @operator)
  (assignment_expression ">>=" @operator)
  (assignment_expression "&=" @operator)
  (assignment_expression "^=" @operator)
  (assignment_expression "|=" @operator)
  (assignment_expression "or=" @operator)
  (assignment_expression "+>=" @operator)
]
( "@" @operator)
(application_expression ".=" @operator)
(walrus_expression ":=" @operator)
[
  (binary_expression "+" @operator)
  (binary_expression "-" @operator)
  (binary_expression "+>" @operator)
  (binary_expression "^" @operator)
  (binary_expression "*" @operator)
  (binary_expression "/" @operator)
  (binary_expression "//" @operator)
  (binary_expression "%" @operator)
  (binary_expression "**" @operator)
]
[
  (unary_expression "+" @operator)
  (unary_expression "-" @operator)
  (unary_expression "not" @operator)
  (unary_expression "!" @operator)
  (unary_expression "$" @operator)
  (unary_expression "~" @operator)
  (unary_expression "++" @operator)
  (unary_expression "--" @operator)
]
(range_expression "??" @keyword.operator)

;; Comparison operators (including CCC chains)
(cmp_operator) @operator

;; Functions
(lambda_expression "&" @function.macro)

;; Objects
(object_literal "{" @punctuation.bracket)
(object_literal "}" @punctuation.bracket)
(array_literal "[" @punctuation.bracket)
(array_literal "]" @punctuation.bracket)
(set_literal "set" @keyword)
(dict_comprehension "{" @punctuation.bracket)
(dict_comprehension "}" @punctuation.bracket)
