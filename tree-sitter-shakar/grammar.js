// Tree-sitter grammar for Shakar
// This is an initial port derived from the current Lark grammar.

const PREC = {
  assignment: 1,
  walrus: 2,
  catch: 3,
  conditional: 4,
  nullish: 5,
  logical_or: 6,
  logical_and: 7,
  compare: 8,
  range: 9,
  add: 10,
  mul: 11,
  pow: 12,
  unary: 13,
  call: 14,
  member: 15,
};

module.exports = grammar({
  name: 'shakar',

  extras: $ => [
    /[ \t]/,
    $.newline,
    $.comment,
  ],

  externals: $ => [
    $.newline,
    $.indent,
    $.dedent,
  ],

  conflicts: $ => [
    [$.assignment_statement, $.expression_statement],
    [$.postfix_expression, $.lvalue],
    [$.object_literal, $.block],
    [$.pattern, $.tuple_pattern],
    [$.primary_expression, $.lvalue],
    [$.primary_expression, $.pattern],
    [$.for_statement, $.primary_expression],
    [$.primary_expression, $.object_item],
    [$.return_statement, $.primary_expression],
    [$.return_statement, $.list_comprehension],
    [$.return_statement, $.set_comprehension],
    [$.return_statement],
    [$.using_statement, $.primary_expression],
    [$.await_guard_chain, $.primary_expression],
    [$.await_guard_chain, $.await_target],
    [$.await_target, $.primary_expression],
    [$.await_guard_chain, $.await_target, $.primary_expression],
    [$.await_any_statement, $.await_any_expression],
    [$.await_all_statement, $.await_all_expression],
    [$.inline_body, $.block],
    [$.brace_block, $.object_literal],
    [$.inline_body, $.dict_comprehension],
    [$.inline_body, $.object_item],
    [$.await_arm, $.primary_expression],
    [$.dbg_statement, $.primary_expression],
    [$.dbg_statement],
    [$.throw_statement],
    [$.slice_selector],
    [$.break_statement],
    [$.break_statement, $.terminator],
    [$.continue_statement],
    [$.continue_statement, $.terminator],
    [$.field_fan, $.value_fan],
    [$.field_fan, $.value_fan_item],
    [$.binder_pattern, $.pattern],
    [$.statement_block],
    [$.guard_statement],
    [$.guard_chain],
    [$.expression_statement, $.catch_statement, $.inline_body, $.catch_expression],
    [$.expression_statement, $.catch_statement, $.catch_expression],
    [$.destructure_statement, $.pack_expression],
    [$.destructure_statement],
    [$.assert_statement],
    [$.defer_statement],
    [$.fanout_block_statement],
    [$.hook_statement],
    [$.decorator_statement],
    [$.catch_statement],
    [$.if_statement],
    [$.pack_expression],
    [$.function_statement],
  ],

  supertypes: $ => [
    $._statement,
    $._expression,
  ],

  inline: $ => [
    $._literal,
    $._primary,
    $._pattern_atom,
  ],

  word: $ => $.identifier,

  rules: {
    source_file: $ => repeat(seq($._statement, optional($.newline))),

    _statement: $ => choice(
      $.expression_statement,
      $.assignment_statement,
      $.destructure_statement,
      $.return_statement,
      $.assert_statement,
      $.await_statement,
      $.await_any_statement,
      $.await_all_statement,
      $.break_statement,
      $.continue_statement,
      $.throw_statement,
      $.using_statement,
      $.defer_statement,
      $.hook_statement,
      $.fanout_block_statement,
      $.decorator_statement,
      $.function_statement,
      $.catch_statement,
      $.dbg_statement,
      $.if_statement,
      $.while_statement,
      $.for_statement,
      $.guard_statement,
      $.newline,
      $.comment
    ),

    expression_statement: $ => prec.right(1, seq($._expression, optional($.terminator))),

    assignment_statement: $ => prec.right(2, seq(
      field('target', $.lvalue),
      choice(
        '=',
        '+=',
        '-=',
        '*=',
        '/=',
        '//=',
        '%=',
        '**=',
        '<<=',
        '>>=',
        '&=',
        '^=',
        '|=',
        'or=',
        '+>='
      ),
      field('value', $._expression),
      optional($.terminator)
    )),

    destructure_statement: $ => seq(
      field('pattern', $.tuple_pattern),
      choice('=', ':='),
      field('value', choice($.pack_expression, $._expression)),
      optional($.terminator)
    ),

    pack_expression: $ => seq(
      field('first', $._expression),
      repeat1(seq(',', field('rest', $._expression)))
    ),

    return_statement: $ => prec.dynamic(1, seq(
      'return',
      optional(field('value', $._expression)),
      optional($.terminator)
    )),

    assert_statement: $ => seq(
      'assert',
      field('condition', $._expression),
      optional(seq(',', field('message', $._expression))),
      optional($.terminator)
    ),

    dbg_statement: $ => seq(
      'dbg',
      optional(seq(field('value', $._expression), repeat(seq(',', $._expression)))),
      optional($.terminator)
    ),

    using_statement: $ => seq(
      'using',
      optional(seq('[', $.identifier, ']')),
      field('value', $._expression),
      optional(seq('bind', $.identifier)),
      ':',
      field('body', $.block_or_inline)
    ),

    block_or_inline: $ => choice(
      $.statement_block,
      $.inline_body
    ),

    defer_statement: $ => seq(
      'defer',
      field('callable', $._expression),
      optional($.terminator)
    ),

    hook_statement: $ => seq(
      'hook',
      field('name', $.string),
      ':',
      field('body', $.block_or_inline),
      optional($.terminator)
    ),

    fanout_block_statement: $ => seq(
      field('base', $._expression),
      field('fanout', $.fan_block),
      optional($.terminator)
    ),

    fan_block: $ => seq(
      '{',
      optional(seq(
        $.fan_clause,
        repeat(seq(optional($.fan_clause_separator), $.fan_clause)),
        optional($.fan_clause_separator)
      )),
      '}'
    ),

    fan_clause_separator: _ => choice(',', ';'),

    fan_clause: $ => seq(
      '.',
      field('path', $.fan_path),
      field('operator', $.fan_assignment_operator),
      field('value', $._expression)
    ),

    fan_path: $ => repeat1($.fan_segment),

    fan_segment: $ => choice(
      field('field', $.identifier),
      field('index', $.fan_index)
    ),

    fan_index: $ => seq('[', $.selector_list, ']'),

    fan_assignment_operator: _ => choice(
      '=',
      '.=',
      '+=',
      '-=',
      '*=',
      '/=',
      '//=',
      '%=',
      '**='
    ),

    decorator_statement: $ => seq(
      'decorator',
      field('name', $.identifier),
      optional(seq('(', optional($.parameter_list), ')')),
      ':',
      field('body', $.block_or_inline),
      optional($.terminator)
    ),

    function_statement: $ => seq(
      optional($.decorator_list),
      'fn',
      field('name', $.identifier),
      '(',
      optional($.parameter_list),
      ')',
      ':',
      field('body', $.block_or_inline),
      optional($.terminator)
    ),

    catch_statement: $ => seq(
      field('subject', $._expression),
      'catch',
      optional($.catch_tail),
      ':',
      field('body', $.block_or_inline),
      optional($.terminator)
    ),

    await_statement: $ => seq(
      'await',
      field('subject', $.await_target),
      ':',
      field('body', $.block_or_inline)
    ),

    break_statement: $ => seq('break', optional($.terminator)),

    continue_statement: $ => seq('continue', optional($.terminator)),

    throw_statement: $ => seq(
      'throw',
      optional(field('value', $._expression)),
      optional($.terminator)
    ),

    await_any_statement: $ => seq(
      'await',
      '[',
      'any',
      ']',
      '(',
      optional($.await_arm_list),
      ')',
      optional(seq(':', $.block_or_inline))
    ),

    await_all_statement: $ => seq(
      'await',
      '[',
      'all',
      ']',
      '(',
      optional($.await_arm_list),
      ')',
      optional(seq(':', $.block_or_inline))
    ),

    await_any_expression: $ => seq(
      'await',
      '[',
      'any',
      ']',
      '(',
      optional($.await_arm_list),
      ')'
    ),

    await_all_expression: $ => seq(
      'await',
      '[',
      'all',
      ']',
      '(',
      optional($.await_arm_list),
      ')'
    ),

    guard_statement: $ => seq(
      choice($.guard_chain, $.await_guard_chain),
      optional($.terminator)
    ),

    guard_chain: $ => seq(
      $.guard_branch,
      repeat(seq(optional($.newline), choice('|', '||'), $.guard_branch)),
      optional(seq(
        optional($.newline),
        choice('|:', '||:'),
        field('default_body', choice($.statement_block, $.inline_body))
      ))
    ),

    await_guard_chain: $ => seq(
      'await',
      choice(seq('(', $._expression, ')'), $._expression),
      ':',
      $.inline_body
    ),

    await_arm_list: $ => commaSep1($.await_arm),

    await_arm: $ => choice(
      seq(
        'timeout',
        field('timeout', $._expression),
        optional(seq(':', field('body', $.block_or_inline)))
      ),
      seq(
        field('name', $.identifier),
        ':',
        field('value', $._expression),
        optional(seq(':', field('body', $.block_or_inline)))
      ),
      seq(
        field('value', $._expression),
        optional(seq(':', field('body', $.block_or_inline)))
      )
    ),

    guard_branch: $ => seq(
      field('condition', $._expression),
      ':',
      field('body', choice(
        $.statement_block,
        $.inline_body
      ))
    ),

    _simple_statement: $ => choice(
      $.expression_statement,
      $.assignment_statement,
      $.destructure_statement,
      $.return_statement,
      $.assert_statement,
      $.break_statement,
      $.continue_statement,
      $.throw_statement,
      $.defer_statement,
      $.dbg_statement
    ),

    inline_body: $ => choice(
      $.brace_block,
      $._simple_statement
    ),

    if_statement: $ => seq(
      'if',
      field('condition', $._expression),
      ':',
      field('consequence', $.block_or_inline),
      repeat($.elif_clause),
      optional($.else_clause)
    ),

    elif_clause: $ => seq(
      'elif',
      field('condition', $._expression),
      ':',
      field('body', $.block_or_inline)
    ),

    else_clause: $ => seq(
      'else',
      ':',
      field('body', $.block_or_inline)
    ),

    while_statement: $ => seq(
      'while',
      field('condition', $._expression),
      ':',
      field('body', $.block_or_inline)
    ),

    for_statement: $ => choice(
      seq(
        'for',
        field('binder', choice($.identifier, $.tuple_pattern)),
        'in',
        field('iterable', $._expression),
        ':',
        field('body', $.block_or_inline)
      ),
      seq(
        'for',
        field('subject', $._expression),
        ':',
        field('body', $.block_or_inline)
      ),
      seq(
        'for',
        '[',
        field('index', $.binder_pattern),
        ']',
        field('iterable', $._expression),
        ':',
        field('body', $.block_or_inline)
      ),
      seq(
        'for',
        '[',
        field('key', $.binder_pattern),
        ',',
        field('value', $.binder_pattern),
        ']',
        field('iterable', $._expression),
        ':',
        field('body', $.block_or_inline)
      )
    ),

    brace_block: $ => seq(
      '{',
      repeat($._statement),
      '}'
    ),

    statement_block: $ => seq(
      $.newline,
      $.indent,
      repeat1(seq($._statement, optional($.newline))),
      $.dedent
    ),

    block: $ => choice(
      $.brace_block,
      $.statement_block
    ),

    terminator: _ => ';',

    _expression: $ => choice(
      $.application_expression,
      $.walrus_expression,
      $.catch_expression,
      $.catch_sugar_expression,
      $.conditional_expression,
      $.range_expression,
      $.logical_expression,
      $.compare_expression,
      $.binary_expression,
      $.unary_expression,
      $.await_expression,
      $.await_any_expression,
      $.await_all_expression,
      $.function_expression,
      $.call_expression,
      $.lambda_expression,
      $.primary_expression
    ),

    application_expression: $ => prec.right(PREC.walrus, seq(
      field('target', $.lvalue),
      '.=',
      field('value', $._expression)
    )),

    walrus_expression: $ => prec.right(PREC.walrus, seq(
      field('name', $.identifier),
      ':=',
      field('value', $._expression)
    )),

    catch_expression: $ => prec.left(PREC.catch, seq(
      field('subject', $._expression),
      'catch',
      optional($.catch_tail),
      choice(
        seq('=>', field('handler', $._expression)),
        seq(':', field('handler', $.inline_body))
      )
    )),

    catch_sugar_expression: $ => prec.left(PREC.catch, seq(
      field('subject', $._expression),
      '@@',
      optional($.catch_tail),
      choice(
        seq('=>', field('handler', $._expression)),
        seq(':', field('handler', $.inline_body))
      )
    )),

    catch_tail: $ => choice(
      seq('(', $.identifier, repeat(seq(',', $.identifier)), ')', optional(seq('bind', $.identifier))),
      $.identifier
    ),

    conditional_expression: $ => prec.right(PREC.conditional, seq(
      field('condition', $._expression),
      '?',
      field('consequence', $._expression),
      ':',
      field('alternative', $._expression)
    )),

    logical_expression: $ => choice(
      prec.left(PREC.logical_or, seq($._expression, 'or', $._expression)),
      prec.left(PREC.logical_and, seq($._expression, 'and', $._expression))
    ),

    compare_expression: $ => prec.left(PREC.compare, seq(
      $._expression,
      $.cmp_operator,
      $._expression,
      optional($.ccc_chain)
    )),

    cmp_operator: _ => choice('==','!=','<=','>=','<','>','is','in','!is','not','!in'),

    ccc_chain: $ => prec.right(seq(
      alias(',', $.ccc_separator),
      $.ccc_leg,
      repeat(seq(alias(',', $.ccc_separator), $.ccc_leg))
    )),

    ccc_leg: $ => choice(
      $.ccc_or_leg,
      $.ccc_and_leg
    ),

    ccc_or_leg: $ => seq(
      'or',
      $.cmp_operator,
      $._expression
    ),

    ccc_and_leg: $ => seq(
      optional('and'),
      $.ccc_and_payload
    ),

    ccc_and_payload: $ => choice(
      seq($.cmp_operator, $._expression),
      $._expression
    ),

    range_expression: $ => prec.left(PREC.nullish, seq(
      field('left', $._expression),
      '??',
      field('right', $._expression)
    )),

    binary_expression: $ => choice(
      prec.left(PREC.add, seq($._expression, choice('+','-','+>','^'), $._expression)),
      prec.left(PREC.mul, seq($._expression, choice('*','/','//','%'), $._expression)),
      prec.right(PREC.pow, seq($._expression, '**', $._expression))
    ),

    unary_expression: $ => prec(PREC.unary, seq(
      choice('+', '-', 'not', '!', '$', '~', '++', '--'),
      $._expression
    )),

    await_expression: $ => prec(PREC.unary, seq(
      'await',
      $.await_target
    )),

    await_target: $ => choice(
      seq('(', $._expression, ')'),
      $._expression
    ),

    function_expression: $ => seq(
      'fn',
      '(',
      optional($.parameter_list),
      ')',
      ':',
      field('body', $.block_or_inline)
    ),

    postfix_expression: $ => prec.left(PREC.member, seq(
      choice($.primary_expression, $.postfix_expression),
      choice(
        $.call_arguments,
        $.field_expression,
        $.index_expression,
        $.field_fan,
        $.value_fan
      )
    )),

    call_expression: $ => $.postfix_expression,

    call_arguments: $ => seq(
      '(',
      optional(seq($._expression, repeat(seq(',', $._expression)),
        optional(seq(',', $.named_argument, repeat(seq(',', $.named_argument)))))),
      ')'
    ),

    named_argument: $ => seq(
      field('name', $.identifier),
      ':',
      field('value', $._expression)
    ),

    lambda_expression: $ => seq(
      '&',
      optional(seq('[', $.parameter_list, ']')),
      '(',
      field('body', $._expression),
      ')'
    ),

    lvalue: $ => prec.left(PREC.member, seq(
      $.primary_expression,
      repeat(choice($.field_expression, $.index_expression, $.field_fan))
    )),

    field_expression: $ => seq('.', field('name', $.identifier)),

    index_expression: $ => seq('[', $.selector_list, ']'),

    selector_list: $ => seq(
      $.selector,
      repeat(seq(',', $.selector)),
      optional(seq(',', $.default_selector))
    ),

    selector: $ => choice($.slice_selector, $._expression),

    default_selector: $ => seq(
      'default',
      ':',
      $._expression
    ),

    slice_selector: $ => seq(
      optional($._expression),
      ':',
      optional($._expression),
      optional(seq(':', optional($._expression)))
    ),

    field_fan: $ => seq(
      '.',
      '{',
      commaSep1($.identifier),
      '}'
    ),

    value_fan: $ => seq(
      '.',
      '{',
      commaSep1($.value_fan_item),
      '}'
    ),

    value_fan_item: $ => choice(
      field('field', $.identifier),
      $.value_fan_chain
    ),

    value_fan_chain: $ => seq(
      field('head', $.identifier),
      repeat1(choice(
        $.call_arguments,
        $.field_expression,
        $.index_expression,
        $.field_fan,
        $.value_fan
      ))
    ),

    primary_expression: $ => choice(
      $.object_literal,
      $.array_literal,
      $.set_literal,
      $.set_comprehension,
      $.dict_comprehension,
      $.list_comprehension,
      $.subject_expression,
      $.selector_literal,
      $.nullsafe_expression,
      $.amp_subject_reference,
      $.hole_expression,
      $.rebind_expression,
      $._literal,
      $.identifier,
      seq('(', $._expression, ')')
    ),

    _primary: $ => prec(PREC.member, choice(
      $.primary_expression,
      $.call_expression
    )),

    _literal: $ => choice($.string, $.raw_string, $.raw_hash_string, $.shell_string, $.number, $.boolean, $.nil),

    string: _ => choice(
      token(seq('"', repeat(choice(/[^"\\\n]/, /\\./)), '"')),
      token(seq("'", repeat(choice(/[^'\\\n]/, /\\./)), "'"))
    ),

    raw_string: _ => choice(
      token(seq('raw"', repeat(/[^"\n]/), '"')),
      token(seq("raw'", repeat(/[^'\n]/), "'"))
    ),

    // Simplified raw hash string: raw#" ... "# (no nested "#)
    raw_hash_string: _ => token(seq('raw#"', repeat(/[^"\n]/), '"#')),

    shell_string: _ => choice(
      token(seq('sh"', repeat(choice(/[^"\\\n]/, /\\./)), '"')),
      token(seq("sh'", repeat(choice(/[^'\\\n]/, /\\./)), "'"))
    ),

    number: _ => token(seq(
      optional('-'),
      choice(
        /\d+\.\d*/,
        /\d*\.\d+/,
        /\d+/
      ),
      optional(seq(/[eE]/, optional(/[+-]/), /\d+/))
    )),

    boolean: _ => choice('true', 'false'),

    nil: _ => 'nil',

    comment: _ => token(choice(
      seq('#', /.*/),
    )),

    identifier: _ => /[A-Za-z_]\w*/,

    object_literal: $ => seq(
      '{',
      optional($.object_items),
      '}'
    ),

    object_items: $ => commaSep1($.object_item),

    object_item: $ => choice(
      seq(field('key', $.identifier), ':', field('value', $._expression)),
      seq(field('key', $.string), ':', field('value', $._expression)),
      seq(field('key', $._expression), ':', field('value', $._expression)),
      seq('get', $.identifier, optional(seq('(', ')')), ':', field('body', $.block)),
      seq('set', $.identifier, '(', $.identifier, ')', ':', field('body', $.block)),
      seq($.identifier, '(', optional($.parameter_list), ')', ':', field('body', $.block))
    ),

    parameter_list: $ => commaSep1($.identifier),

    decorator_list: $ => repeat1($.decorator_entry),

    decorator_entry: $ => seq('@', field('decorator', $._expression)),

    array_literal: $ => seq(
      '[',
      optional(seq($._expression, repeat(seq(',', $._expression)), optional(','))),
      ']'
    ),

    set_literal: $ => seq(
      'set',
      '{',
      optional(seq($._expression, repeat(seq(',', $._expression)), optional(','))),
      '}'
    ),

    list_comprehension: $ => seq(
      '[',
      field('body', $._expression),
      $.comprehension_head,
      optional($.if_clause),
      ']'
    ),

    set_comprehension: $ => seq(
      'set',
      '{',
      field('body', $._expression),
      $.comprehension_head,
      optional($.if_clause),
      '}'
    ),

    dict_comprehension: $ => seq(
      '{',
      field('key', $._expression),
      ':',
      field('value', $._expression),
      $.comprehension_head,
      optional($.if_clause),
      '}'
    ),

    comprehension_head: $ => seq(
      choice('over', 'for'),
      $.overspec
    ),

    overspec: $ => choice(
      seq('[', $.binder_list, ']', $._expression),
      seq($._expression, optional(seq('bind', $.pattern)))
    ),

    binder_list: $ => commaSep1($.binder_pattern),

    binder_pattern: $ => choice(
      seq('^', $.identifier),
      $.pattern
    ),

    if_clause: $ => seq(
      'if',
      $._expression
    ),

    rebind_expression: $ => choice(
      seq('=', $.identifier),
      seq('=', '(', $.lvalue, ')')
    ),

    subject_expression: $ => '.',

    selector_literal: $ => seq(
      '`',
      commaSep1($.selector),
      '`'
    ),

    nullsafe_expression: $ => seq(
      '??',
      '(',
      $._expression,
      ')'
    ),

    amp_subject_reference: $ => seq(
      '&',
      '.',
      $.identifier
    ),

    hole_expression: _ => '?',

    tuple_pattern: $ => seq(
      '(',
      commaSep1($.pattern),
      ')'
    ),

    pattern: $ => choice(
      $.identifier,
      $.tuple_pattern
    ),

    _pattern_atom: $ => choice($.identifier, $.tuple_pattern),
  },
});

function commaSep1(rule) {
  return seq(rule, repeat(seq(',', rule)));
}
