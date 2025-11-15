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
  "and" "or" "not" "is"
  "catch" "set" "get"
] @keyword

((identifier) @keyword
  (#match? @keyword "^(if|elif|else|for|in|over|bind|await|using|hook|defer|return|assert|dbg|and|or|not|is|catch|set|get|fn)$"))


(catch_expression "catch" @keyword)
(catch_statement "catch" @keyword)
(catch_expression "=>" @operator)
(catch_sugar_expression "@@" @operator)
(catch_sugar_expression "=>" @operator)

(guard_chain "|" @operator)
(guard_chain "|:" @operator)

;; Literals
(string) @string
(number) @number

;; Comments
(comment) @comment

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

;; Identifiers
(identifier) @variable
(field_expression name: (identifier) @property)

;; Functions
(call_expression
  function: (primary_expression (identifier) @function.call))

(lambda_expression) @function

;; Objects
(object_literal "{" @punctuation.bracket)
(object_literal "}" @punctuation.bracket)
(array_literal "[" @punctuation.bracket)
(array_literal "]" @punctuation.bracket)
(set_literal "set" @keyword)
(dict_comprehension "{" @punctuation.bracket)
(dict_comprehension "}" @punctuation.bracket)
