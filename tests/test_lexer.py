"""Comprehensive tests for the C compiler lexer."""

import pytest

from compiler.lexer import Lexer, LexerError
from compiler.tokens import Token, TokenType


# ── Helpers ──────────────────────────────────────────────────────────────────


def tokenize(source: str) -> list[Token]:
	return Lexer(source).tokenize()


def types(source: str) -> list[TokenType]:
	return [t.type for t in tokenize(source)]


def first_token(source: str) -> Token:
	tokens = tokenize(source)
	assert len(tokens) >= 2  # at least one token + EOF
	return tokens[0]


# ── Keywords ─────────────────────────────────────────────────────────────────


class TestKeywords:
	@pytest.mark.parametrize(
		"keyword,expected",
		[
			("auto", TokenType.AUTO),
			("break", TokenType.BREAK),
			("case", TokenType.CASE),
			("char", TokenType.CHAR),
			("const", TokenType.CONST),
			("continue", TokenType.CONTINUE),
			("default", TokenType.DEFAULT),
			("do", TokenType.DO),
			("double", TokenType.DOUBLE),
			("else", TokenType.ELSE),
			("enum", TokenType.ENUM),
			("extern", TokenType.EXTERN),
			("float", TokenType.FLOAT),
			("for", TokenType.FOR),
			("goto", TokenType.GOTO),
			("if", TokenType.IF),
			("int", TokenType.INT),
			("long", TokenType.LONG),
			("register", TokenType.REGISTER),
			("return", TokenType.RETURN),
			("short", TokenType.SHORT),
			("signed", TokenType.SIGNED),
			("sizeof", TokenType.SIZEOF),
			("static", TokenType.STATIC),
			("struct", TokenType.STRUCT),
			("switch", TokenType.SWITCH),
			("typedef", TokenType.TYPEDEF),
			("union", TokenType.UNION),
			("unsigned", TokenType.UNSIGNED),
			("void", TokenType.VOID),
			("volatile", TokenType.VOLATILE),
			("while", TokenType.WHILE),
		],
	)
	def test_keyword(self, keyword: str, expected: TokenType) -> None:
		tok = first_token(keyword)
		assert tok.type == expected
		assert tok.value == keyword

	def test_keyword_prefix_is_identifier(self) -> None:
		tok = first_token("integer")
		assert tok.type == TokenType.IDENTIFIER
		assert tok.value == "integer"

	def test_keyword_with_trailing_underscore_is_identifier(self) -> None:
		tok = first_token("int_")
		assert tok.type == TokenType.IDENTIFIER


# ── Identifiers ──────────────────────────────────────────────────────────────


class TestIdentifiers:
	def test_simple_identifier(self) -> None:
		tok = first_token("foo")
		assert tok.type == TokenType.IDENTIFIER
		assert tok.value == "foo"

	def test_underscore_identifier(self) -> None:
		tok = first_token("_foo_bar")
		assert tok.type == TokenType.IDENTIFIER
		assert tok.value == "_foo_bar"

	def test_identifier_with_digits(self) -> None:
		tok = first_token("var123")
		assert tok.type == TokenType.IDENTIFIER
		assert tok.value == "var123"

	def test_single_underscore(self) -> None:
		tok = first_token("_")
		assert tok.type == TokenType.IDENTIFIER
		assert tok.value == "_"


# ── Integer Literals ─────────────────────────────────────────────────────────


class TestIntegerLiterals:
	def test_decimal(self) -> None:
		tok = first_token("42")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "42"

	def test_zero(self) -> None:
		tok = first_token("0")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "0"

	def test_hex_lower(self) -> None:
		tok = first_token("0xff")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "0xff"

	def test_hex_upper(self) -> None:
		tok = first_token("0XAB")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "0XAB"

	def test_octal(self) -> None:
		tok = first_token("077")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "077"

	def test_long_suffix(self) -> None:
		tok = first_token("42L")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "42L"

	def test_unsigned_suffix(self) -> None:
		tok = first_token("42U")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "42U"

	def test_unsigned_long_suffix(self) -> None:
		tok = first_token("42UL")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "42UL"

	def test_hex_with_suffix(self) -> None:
		tok = first_token("0xFFu")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "0xFFu"


# ── Float Literals ───────────────────────────────────────────────────────────


class TestFloatLiterals:
	def test_basic_float(self) -> None:
		tok = first_token("3.14")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == "3.14"

	def test_float_with_exponent(self) -> None:
		tok = first_token("1e10")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == "1e10"

	def test_float_with_signed_exponent(self) -> None:
		tok = first_token("2.5e-3")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == "2.5e-3"

	def test_float_starting_with_dot(self) -> None:
		tok = first_token(".5")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == ".5"

	def test_float_f_suffix(self) -> None:
		tok = first_token("1.0f")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == "1.0f"

	def test_float_L_suffix(self) -> None:
		tok = first_token("1.0L")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == "1.0L"


# ── String Literals ──────────────────────────────────────────────────────────


class TestStringLiterals:
	def test_simple_string(self) -> None:
		tok = first_token('"hello"')
		assert tok.type == TokenType.STRING_LITERAL
		assert tok.value == '"hello"'

	def test_empty_string(self) -> None:
		tok = first_token('""')
		assert tok.type == TokenType.STRING_LITERAL
		assert tok.value == '""'

	def test_escape_sequences(self) -> None:
		tok = first_token(r'"hello\nworld\t!"')
		assert tok.type == TokenType.STRING_LITERAL
		assert tok.value == r'"hello\nworld\t!"'

	def test_escaped_quote(self) -> None:
		tok = first_token(r'"say \"hi\""')
		assert tok.type == TokenType.STRING_LITERAL
		assert tok.value == r'"say \"hi\""'

	def test_escaped_backslash(self) -> None:
		tok = first_token(r'"path\\to"')
		assert tok.type == TokenType.STRING_LITERAL
		assert tok.value == r'"path\\to"'

	def test_unterminated_string_raises(self) -> None:
		with pytest.raises(LexerError, match="Unterminated string literal"):
			tokenize('"hello')

	def test_newline_in_string_raises(self) -> None:
		with pytest.raises(LexerError, match="Unterminated string literal"):
			tokenize('"hello\nworld"')


# ── Character Literals ───────────────────────────────────────────────────────


class TestCharLiterals:
	def test_simple_char(self) -> None:
		tok = first_token("'a'")
		assert tok.type == TokenType.CHAR_LITERAL
		assert tok.value == "'a'"

	def test_escape_char(self) -> None:
		tok = first_token(r"'\n'")
		assert tok.type == TokenType.CHAR_LITERAL
		assert tok.value == r"'\n'"

	def test_escaped_quote_char(self) -> None:
		tok = first_token("'\\''")
		assert tok.type == TokenType.CHAR_LITERAL
		assert tok.value == "'\\''"

	def test_escaped_backslash_char(self) -> None:
		tok = first_token(r"'\\'")
		assert tok.type == TokenType.CHAR_LITERAL
		assert tok.value == r"'\\'"

	def test_unterminated_char_raises(self) -> None:
		with pytest.raises(LexerError, match="Unterminated character literal"):
			tokenize("'a")


# ── Operators ────────────────────────────────────────────────────────────────


class TestOperators:
	@pytest.mark.parametrize(
		"source,expected_type",
		[
			("+", TokenType.PLUS),
			("-", TokenType.MINUS),
			("*", TokenType.STAR),
			("/", TokenType.SLASH),
			("%", TokenType.PERCENT),
			("&", TokenType.AMPERSAND),
			("|", TokenType.PIPE),
			("^", TokenType.CARET),
			("~", TokenType.TILDE),
			("!", TokenType.BANG),
			("=", TokenType.ASSIGN),
			("<", TokenType.LESS),
			(">", TokenType.GREATER),
			(".", TokenType.DOT),
			("?", TokenType.QUESTION),
		],
	)
	def test_single_char_operator(self, source: str, expected_type: TokenType) -> None:
		tok = first_token(source)
		assert tok.type == expected_type
		assert tok.value == source

	@pytest.mark.parametrize(
		"source,expected_type",
		[
			("+=", TokenType.PLUS_ASSIGN),
			("-=", TokenType.MINUS_ASSIGN),
			("*=", TokenType.STAR_ASSIGN),
			("/=", TokenType.SLASH_ASSIGN),
			("%=", TokenType.PERCENT_ASSIGN),
			("&=", TokenType.AMP_ASSIGN),
			("|=", TokenType.PIPE_ASSIGN),
			("^=", TokenType.CARET_ASSIGN),
			("<<=", TokenType.LSHIFT_ASSIGN),
			(">>=", TokenType.RSHIFT_ASSIGN),
			("==", TokenType.EQUAL),
			("!=", TokenType.NOT_EQUAL),
			("<=", TokenType.LESS_EQUAL),
			(">=", TokenType.GREATER_EQUAL),
			("&&", TokenType.AND),
			("||", TokenType.OR),
			("<<", TokenType.LSHIFT),
			(">>", TokenType.RSHIFT),
			("++", TokenType.INCREMENT),
			("--", TokenType.DECREMENT),
			("->", TokenType.ARROW),
		],
	)
	def test_multi_char_operator(self, source: str, expected_type: TokenType) -> None:
		tok = first_token(source)
		assert tok.type == expected_type
		assert tok.value == source

	def test_ellipsis(self) -> None:
		tok = first_token("...")
		assert tok.type == TokenType.ELLIPSIS
		assert tok.value == "..."


# ── Punctuation ──────────────────────────────────────────────────────────────


class TestPunctuation:
	@pytest.mark.parametrize(
		"source,expected_type",
		[
			("(", TokenType.LPAREN),
			(")", TokenType.RPAREN),
			("[", TokenType.LBRACKET),
			("]", TokenType.RBRACKET),
			("{", TokenType.LBRACE),
			("}", TokenType.RBRACE),
			(";", TokenType.SEMICOLON),
			(":", TokenType.COLON),
			(",", TokenType.COMMA),
			("#", TokenType.HASH),
		],
	)
	def test_punctuation(self, source: str, expected_type: TokenType) -> None:
		tok = first_token(source)
		assert tok.type == expected_type
		assert tok.value == source


# ── Comments ─────────────────────────────────────────────────────────────────


class TestComments:
	def test_line_comment_skipped(self) -> None:
		assert types("a // comment\nb") == [TokenType.IDENTIFIER, TokenType.IDENTIFIER, TokenType.EOF]

	def test_block_comment_skipped(self) -> None:
		toks = tokenize("a /* comment */ b")
		assert [t.type for t in toks] == [TokenType.IDENTIFIER, TokenType.IDENTIFIER, TokenType.EOF]

	def test_multiline_block_comment(self) -> None:
		src = "x /* line1\nline2\nline3 */ y"
		toks = tokenize(src)
		assert [t.type for t in toks] == [TokenType.IDENTIFIER, TokenType.IDENTIFIER, TokenType.EOF]
		assert toks[0].value == "x"
		assert toks[1].value == "y"

	def test_unterminated_block_comment_raises(self) -> None:
		with pytest.raises(LexerError, match="Unterminated block comment"):
			tokenize("/* never closed")

	def test_comment_with_star_inside(self) -> None:
		toks = tokenize("a /* * */ b")
		assert [t.type for t in toks] == [TokenType.IDENTIFIER, TokenType.IDENTIFIER, TokenType.EOF]

	def test_line_comment_at_end(self) -> None:
		toks = tokenize("x // end")
		assert [t.type for t in toks] == [TokenType.IDENTIFIER, TokenType.EOF]


# ── Line/Column Tracking ────────────────────────────────────────────────────


class TestPositionTracking:
	def test_first_token_position(self) -> None:
		tok = first_token("x")
		assert tok.line == 1
		assert tok.column == 1

	def test_second_line_position(self) -> None:
		toks = tokenize("a\nb")
		assert toks[1].line == 2
		assert toks[1].column == 1

	def test_column_advances(self) -> None:
		toks = tokenize("ab cd")
		assert toks[0].column == 1
		assert toks[1].column == 4

	def test_eof_position(self) -> None:
		toks = tokenize("x\n")
		eof = toks[-1]
		assert eof.type == TokenType.EOF
		assert eof.line == 2


# ── Multi-Token Sequences ───────────────────────────────────────────────────


class TestMultiTokenSequences:
	def test_simple_declaration(self) -> None:
		toks = tokenize("int x;")
		assert [t.type for t in toks] == [
			TokenType.INT, TokenType.IDENTIFIER, TokenType.SEMICOLON, TokenType.EOF,
		]

	def test_function_call(self) -> None:
		toks = tokenize("foo(1, 2)")
		assert [t.type for t in toks] == [
			TokenType.IDENTIFIER, TokenType.LPAREN,
			TokenType.INTEGER_LITERAL, TokenType.COMMA,
			TokenType.INTEGER_LITERAL, TokenType.RPAREN,
			TokenType.EOF,
		]

	def test_assignment_expression(self) -> None:
		toks = tokenize("x = y + 1;")
		assert [t.type for t in toks] == [
			TokenType.IDENTIFIER, TokenType.ASSIGN,
			TokenType.IDENTIFIER, TokenType.PLUS,
			TokenType.INTEGER_LITERAL, TokenType.SEMICOLON,
			TokenType.EOF,
		]

	def test_pointer_dereference(self) -> None:
		toks = tokenize("*ptr->field")
		assert [t.type for t in toks] == [
			TokenType.STAR, TokenType.IDENTIFIER,
			TokenType.ARROW, TokenType.IDENTIFIER,
			TokenType.EOF,
		]

	def test_compound_assignment(self) -> None:
		toks = tokenize("x += 1;")
		assert [t.type for t in toks] == [
			TokenType.IDENTIFIER, TokenType.PLUS_ASSIGN,
			TokenType.INTEGER_LITERAL, TokenType.SEMICOLON,
			TokenType.EOF,
		]

	def test_shift_expression(self) -> None:
		toks = tokenize("a << 2")
		assert [t.type for t in toks] == [
			TokenType.IDENTIFIER, TokenType.LSHIFT,
			TokenType.INTEGER_LITERAL, TokenType.EOF,
		]

	def test_comparison_chain(self) -> None:
		toks = tokenize("a <= b && c >= d")
		expected = [
			TokenType.IDENTIFIER, TokenType.LESS_EQUAL,
			TokenType.IDENTIFIER, TokenType.AND,
			TokenType.IDENTIFIER, TokenType.GREATER_EQUAL,
			TokenType.IDENTIFIER, TokenType.EOF,
		]
		assert [t.type for t in toks] == expected


# ── Error Cases ──────────────────────────────────────────────────────────────


class TestErrorCases:
	def test_invalid_character(self) -> None:
		with pytest.raises(LexerError, match="Unexpected character"):
			tokenize("@")

	def test_error_reports_line(self) -> None:
		with pytest.raises(LexerError) as exc_info:
			tokenize("\n@")
		assert exc_info.value.line == 2

	def test_error_reports_column(self) -> None:
		with pytest.raises(LexerError) as exc_info:
			tokenize("  @")
		assert exc_info.value.column == 3

	def test_invalid_hex_literal(self) -> None:
		with pytest.raises(LexerError, match="Invalid hex literal"):
			tokenize("0xG")


# ── Full Program Tokenization ────────────────────────────────────────────────


class TestFullPrograms:
	def test_hello_world(self) -> None:
		src = """\
int main() {
	return 0;
}
"""
		toks = tokenize(src)
		expected_types = [
			TokenType.INT, TokenType.IDENTIFIER,
			TokenType.LPAREN, TokenType.RPAREN,
			TokenType.LBRACE,
			TokenType.RETURN, TokenType.INTEGER_LITERAL, TokenType.SEMICOLON,
			TokenType.RBRACE,
			TokenType.EOF,
		]
		assert [t.type for t in toks] == expected_types

	def test_function_with_params(self) -> None:
		src = "int add(int a, int b) { return a + b; }"
		toks = tokenize(src)
		expected_types = [
			TokenType.INT, TokenType.IDENTIFIER,
			TokenType.LPAREN,
			TokenType.INT, TokenType.IDENTIFIER, TokenType.COMMA,
			TokenType.INT, TokenType.IDENTIFIER,
			TokenType.RPAREN, TokenType.LBRACE,
			TokenType.RETURN, TokenType.IDENTIFIER, TokenType.PLUS, TokenType.IDENTIFIER,
			TokenType.SEMICOLON,
			TokenType.RBRACE,
			TokenType.EOF,
		]
		assert [t.type for t in toks] == expected_types

	def test_if_else(self) -> None:
		src = """\
if (x > 0) {
	y = x;
} else {
	y = -x;
}
"""
		toks = tokenize(src)
		assert toks[0].type == TokenType.IF
		assert any(t.type == TokenType.ELSE for t in toks)

	def test_while_loop(self) -> None:
		src = "while (i < 10) { i++; }"
		toks = tokenize(src)
		assert toks[0].type == TokenType.WHILE
		assert any(t.type == TokenType.INCREMENT for t in toks)

	def test_for_loop(self) -> None:
		src = "for (i = 0; i < n; i++) { sum += arr[i]; }"
		toks = tokenize(src)
		assert toks[0].type == TokenType.FOR
		assert any(t.type == TokenType.PLUS_ASSIGN for t in toks)
		assert any(t.type == TokenType.LBRACKET for t in toks)

	def test_struct_definition(self) -> None:
		src = """\
struct Point {
	int x;
	int y;
};
"""
		toks = tokenize(src)
		assert toks[0].type == TokenType.STRUCT
		assert toks[1].type == TokenType.IDENTIFIER
		assert toks[1].value == "Point"

	def test_program_with_comments(self) -> None:
		src = """\
/* Main function */
int main() {
	// Return success
	return 0;
}
"""
		toks = tokenize(src)
		expected_types = [
			TokenType.INT, TokenType.IDENTIFIER,
			TokenType.LPAREN, TokenType.RPAREN,
			TokenType.LBRACE,
			TokenType.RETURN, TokenType.INTEGER_LITERAL, TokenType.SEMICOLON,
			TokenType.RBRACE,
			TokenType.EOF,
		]
		assert [t.type for t in toks] == expected_types

	def test_pointer_and_address(self) -> None:
		src = "int *p = &x;"
		toks = tokenize(src)
		assert [t.type for t in toks] == [
			TokenType.INT, TokenType.STAR, TokenType.IDENTIFIER,
			TokenType.ASSIGN, TokenType.AMPERSAND, TokenType.IDENTIFIER,
			TokenType.SEMICOLON, TokenType.EOF,
		]

	def test_ternary_operator(self) -> None:
		src = "x = a > b ? a : b;"
		toks = tokenize(src)
		assert any(t.type == TokenType.QUESTION for t in toks)
		assert any(t.type == TokenType.COLON for t in toks)

	def test_sizeof_expression(self) -> None:
		src = "sizeof(int)"
		toks = tokenize(src)
		assert [t.type for t in toks] == [
			TokenType.SIZEOF, TokenType.LPAREN, TokenType.INT, TokenType.RPAREN, TokenType.EOF,
		]

	def test_variadic_function(self) -> None:
		src = "int printf(const char *fmt, ...);"
		toks = tokenize(src)
		assert any(t.type == TokenType.ELLIPSIS for t in toks)
		assert any(t.type == TokenType.CONST for t in toks)

	def test_switch_statement(self) -> None:
		src = """\
switch (x) {
	case 1: break;
	default: break;
}
"""
		toks = tokenize(src)
		assert toks[0].type == TokenType.SWITCH
		assert any(t.type == TokenType.CASE for t in toks)
		assert any(t.type == TokenType.DEFAULT for t in toks)
		assert any(t.type == TokenType.BREAK for t in toks)

	def test_string_with_escapes_in_program(self) -> None:
		src = r'char *s = "hello\n\tworld";'
		toks = tokenize(src)
		str_tok = [t for t in toks if t.type == TokenType.STRING_LITERAL][0]
		assert str_tok.value == r'"hello\n\tworld"'


# ── Edge Cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
	def test_empty_source(self) -> None:
		toks = tokenize("")
		assert len(toks) == 1
		assert toks[0].type == TokenType.EOF

	def test_whitespace_only(self) -> None:
		toks = tokenize("   \n\n\t\t  ")
		assert len(toks) == 1
		assert toks[0].type == TokenType.EOF

	def test_adjacent_operators(self) -> None:
		toks = tokenize("a+++b")
		# Should parse as a ++ + b
		assert [t.type for t in toks] == [
			TokenType.IDENTIFIER, TokenType.INCREMENT,
			TokenType.PLUS, TokenType.IDENTIFIER,
			TokenType.EOF,
		]

	def test_adjacent_minus(self) -> None:
		toks = tokenize("a---b")
		# Should parse as a -- - b
		assert [t.type for t in toks] == [
			TokenType.IDENTIFIER, TokenType.DECREMENT,
			TokenType.MINUS, TokenType.IDENTIFIER,
			TokenType.EOF,
		]

	def test_shift_assign_vs_nested(self) -> None:
		toks = tokenize("a >>= 1")
		assert toks[1].type == TokenType.RSHIFT_ASSIGN

	def test_multiple_strings_concatenated(self) -> None:
		toks = tokenize('"a" "b"')
		str_toks = [t for t in toks if t.type == TokenType.STRING_LITERAL]
		assert len(str_toks) == 1
		assert str_toks[0].value == '"ab"'

	def test_hash_in_source(self) -> None:
		toks = tokenize("#include")
		assert toks[0].type == TokenType.HASH
		assert toks[1].type == TokenType.IDENTIFIER
		assert toks[1].value == "include"

	def test_dot_followed_by_identifier(self) -> None:
		toks = tokenize("s.field")
		assert [t.type for t in toks] == [
			TokenType.IDENTIFIER, TokenType.DOT, TokenType.IDENTIFIER, TokenType.EOF,
		]

	def test_negative_number_is_minus_then_literal(self) -> None:
		toks = tokenize("-42")
		assert [t.type for t in toks] == [
			TokenType.MINUS, TokenType.INTEGER_LITERAL, TokenType.EOF,
		]

	def test_zero_literal(self) -> None:
		tok = first_token("0")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "0"

	def test_reuse_lexer(self) -> None:
		lexer = Lexer("int x;")
		toks1 = lexer.tokenize()
		toks2 = lexer.tokenize()
		assert [t.type for t in toks1] == [t.type for t in toks2]


# ── Acceptance criteria ──────────────────────────────────────────────────────


# ── String Literal Concatenation (C standard phase 6) ────────────────────────


class TestStringConcatenation:
	def test_simple_concatenation(self) -> None:
		tok = first_token('"hello" " " "world"')
		assert tok.type == TokenType.STRING_LITERAL
		assert tok.value == '"hello world"'

	def test_two_strings(self) -> None:
		tok = first_token('"foo" "bar"')
		assert tok.value == '"foobar"'

	def test_concatenation_across_whitespace(self) -> None:
		tok = first_token('"a"   "b"')
		assert tok.value == '"ab"'

	def test_concatenation_across_newlines(self) -> None:
		tok = first_token('"a"\n"b"')
		assert tok.value == '"ab"'

	def test_concatenation_across_mixed_whitespace(self) -> None:
		tok = first_token('"a" \t\n  "b"')
		assert tok.value == '"ab"'

	def test_single_string_no_op(self) -> None:
		tok = first_token('"hello"')
		assert tok.value == '"hello"'

	def test_empty_string_concatenation(self) -> None:
		tok = first_token('"" ""')
		assert tok.value == '""'

	def test_empty_plus_nonempty(self) -> None:
		tok = first_token('"" "hello"')
		assert tok.value == '"hello"'

	def test_nonempty_plus_empty(self) -> None:
		tok = first_token('"hello" ""')
		assert tok.value == '"hello"'

	def test_three_strings(self) -> None:
		tok = first_token('"a" "b" "c"')
		assert tok.value == '"abc"'

	def test_preserves_first_token_position(self) -> None:
		toks = tokenize('x = "a" "b";')
		str_tok = [t for t in toks if t.type == TokenType.STRING_LITERAL][0]
		assert str_tok.value == '"ab"'
		assert str_tok.column == 5

	def test_concatenation_with_escape_newline(self) -> None:
		tok = first_token(r'"hello\n" "world"')
		assert tok.value == r'"hello\nworld"'

	def test_concatenation_with_escape_tab(self) -> None:
		tok = first_token(r'"a\t" "b"')
		assert tok.value == r'"a\tb"'

	def test_concatenation_with_escape_null(self) -> None:
		tok = first_token(r'"a\0" "b"')
		assert tok.value == r'"a\0b"'

	def test_concatenation_with_escape_backslash(self) -> None:
		tok = first_token(r'"a\\" "b"')
		assert tok.value == r'"a\\b"'

	def test_concatenation_with_escaped_quote(self) -> None:
		tok = first_token(r'"a\"" "b"')
		assert tok.value == r'"a\"b"'

	def test_concatenation_with_escape_single_quote(self) -> None:
		tok = first_token(r'"a\'" "b"')
		assert tok.value == r'"a\'b"'

	def test_concatenation_with_hex_escape(self) -> None:
		tok = first_token(r'"a\x41" "b"')
		assert tok.value == r'"a\x41b"'

	def test_concatenation_with_octal_escape(self) -> None:
		tok = first_token(r'"a\101" "b"')
		assert tok.value == r'"a\101b"'

	def test_concatenation_with_escape_r(self) -> None:
		tok = first_token(r'"a\r" "b"')
		assert tok.value == r'"a\rb"'

	def test_mixed_escapes_across_parts(self) -> None:
		tok = first_token(r'"hello\n" "\tworld\0"')
		assert tok.value == r'"hello\n\tworld\0"'

	def test_non_adjacent_strings_not_concatenated(self) -> None:
		toks = tokenize('"a" , "b"')
		str_toks = [t for t in toks if t.type == TokenType.STRING_LITERAL]
		assert len(str_toks) == 2
		assert str_toks[0].value == '"a"'
		assert str_toks[1].value == '"b"'

	def test_string_in_expression_context(self) -> None:
		toks = tokenize('printf("hello" " " "world");')
		str_tok = [t for t in toks if t.type == TokenType.STRING_LITERAL][0]
		assert str_tok.value == '"hello world"'

	def test_token_count_after_concatenation(self) -> None:
		toks = tokenize('"a" "b" "c"')
		assert len(toks) == 2  # one STRING_LITERAL + EOF


class TestAcceptanceCriteria:
	def test_acceptance(self) -> None:
		tokens = Lexer("int main() { return 0; }").tokenize()
		assert any(t.type == TokenType.INT for t in tokens)
