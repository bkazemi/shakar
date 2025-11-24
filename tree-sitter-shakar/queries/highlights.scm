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

;; Subject placeholder (implicit dot) — distinct capture, with a fallback delimiter color
((subject_expression ".") @shakar.subject @punctuation.delimiter)
(subject_expression) @shakar.subject
(field_expression "." @punctuation.delimiter)
(field_fan "." @punctuation.delimiter)

;; Selector literals: backtick delimiters
(selector_literal "`" @punctuation.special)

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

;; Assignment statements that start with '=' (rebind)
(rebind_statement "=" @operator)

; Decorators (match Python behavior: bold @ and decorator name)
((decorator_entry "@" @attribute)
  (#set! "priority" 110))

((decorator_entry decorator: (primary_expression (identifier) @attribute))
  (#set! "priority" 110))

((decorator_entry decorator: (call_expression
    function: (primary_expression (identifier) @attribute)))
  (#set! "priority" 110))

;; Null-safe call marker (distinguish from infix ?? operator)
((nullsafe_expression "??") @punctuation.special)

; CCC separator commas
(ccc_separator) @operator

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
[
  (compare_expression "==" @operator)
  (compare_expression "!=" @operator)
  (compare_expression "<=" @operator)
  (compare_expression ">=" @operator)
  (compare_expression "<" @operator)
  (compare_expression ">" @operator)
  (compare_expression "is" @operator)
  (compare_expression "in" @operator)
  (compare_expression "!is" @operator)
  (compare_expression "not" @operator)
  (compare_expression "!in" @operator)
]
(range_expression "??" @operator)

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
(ccc_separator) @operator
