"""Edge-case tests for lexer tokenization of tricky inputs."""

import pytest

from compiler.lexer import Lexer, LexerError, interpret_c_escapes
from compiler.tokens import IntegerSuffix, TokenType


def tokenize(source: str) -> list:
	return Lexer(source).tokenize()


def types(tokens):
	return [t.type for t in tokens]


def values(tokens):
	return [t.value for t in tokens]


# --- Adjacent string literals ---


class TestAdjacentStringLiterals:
	def test_two_adjacent_strings_concatenated(self):
		tokens = tokenize('"hello" "world"')
		assert tokens[0].type == TokenType.STRING_LITERAL
		assert tokens[0].value == '"helloworld"'
		assert tokens[1].type == TokenType.EOF

	def test_three_adjacent_strings(self):
		tokens = tokenize('"a" "b" "c"')
		assert tokens[0].value == '"abc"'
		assert tokens[1].type == TokenType.EOF

	def test_adjacent_strings_with_newline_between(self):
		tokens = tokenize('"hello"\n"world"')
		assert tokens[0].value == '"helloworld"'

	def test_adjacent_strings_with_escape_sequences(self):
		tokens = tokenize(r'"hello\n" "world"')
		assert tokens[0].value == '"hello\\nworld"'

	def test_non_adjacent_strings_separated_by_other_token(self):
		tokens = tokenize('"a" , "b"')
		assert tokens[0].type == TokenType.STRING_LITERAL
		assert tokens[0].value == '"a"'
		assert tokens[1].type == TokenType.COMMA
		assert tokens[2].type == TokenType.STRING_LITERAL
		assert tokens[2].value == '"b"'

	def test_empty_adjacent_strings(self):
		tokens = tokenize('"" ""')
		assert tokens[0].value == '""'

	def test_single_string_no_concat(self):
		tokens = tokenize('"only"')
		assert tokens[0].value == '"only"'


# --- Integer literal suffixes ---


class TestIntegerSuffixes:
	def test_unsigned_suffix_lower(self):
		tokens = tokenize("42u")
		assert tokens[0].suffix == IntegerSuffix.U
		assert tokens[0].value == "42"

	def test_unsigned_suffix_upper(self):
		tokens = tokenize("42U")
		assert tokens[0].suffix == IntegerSuffix.U

	def test_long_suffix_lower(self):
		tokens = tokenize("42l")
		assert tokens[0].suffix == IntegerSuffix.L

	def test_long_suffix_upper(self):
		tokens = tokenize("42L")
		assert tokens[0].suffix == IntegerSuffix.L

	def test_unsigned_long_suffix(self):
		tokens = tokenize("42UL")
		assert tokens[0].suffix == IntegerSuffix.UL

	def test_unsigned_long_suffix_lower(self):
		tokens = tokenize("42ul")
		assert tokens[0].suffix == IntegerSuffix.UL

	def test_long_long_suffix(self):
		tokens = tokenize("42LL")
		assert tokens[0].suffix == IntegerSuffix.LL

	def test_long_long_suffix_lower(self):
		tokens = tokenize("42ll")
		assert tokens[0].suffix == IntegerSuffix.LL

	def test_unsigned_long_long_suffix(self):
		tokens = tokenize("42ULL")
		assert tokens[0].suffix == IntegerSuffix.ULL

	def test_unsigned_long_long_lower(self):
		tokens = tokenize("42ull")
		assert tokens[0].suffix == IntegerSuffix.ULL

	def test_long_unsigned_order(self):
		"""LU should also parse as UL."""
		tokens = tokenize("42LU")
		assert tokens[0].suffix == IntegerSuffix.UL

	def test_long_long_unsigned_order(self):
		"""LLU should parse as ULL."""
		tokens = tokenize("42LLU")
		assert tokens[0].suffix == IntegerSuffix.ULL

	def test_no_suffix(self):
		tokens = tokenize("42")
		assert tokens[0].suffix == IntegerSuffix.NONE

	def test_suffix_value_excludes_suffix_chars(self):
		tokens = tokenize("100ULL")
		assert tokens[0].value == "100"
		assert tokens[0].type == TokenType.INTEGER_LITERAL


# --- Hex and octal literals ---


class TestHexOctalLiterals:
	def test_hex_lower(self):
		tokens = tokenize("0xff")
		assert tokens[0].type == TokenType.INTEGER_LITERAL
		assert tokens[0].value == "0xff"

	def test_hex_upper(self):
		tokens = tokenize("0XFF")
		assert tokens[0].value == "0XFF"

	def test_hex_mixed_case(self):
		tokens = tokenize("0xAbCd")
		assert tokens[0].value == "0xAbCd"

	def test_hex_with_suffix(self):
		tokens = tokenize("0xFFu")
		assert tokens[0].value == "0xFF"
		assert tokens[0].suffix == IntegerSuffix.U

	def test_hex_with_ull_suffix(self):
		tokens = tokenize("0x1ULL")
		assert tokens[0].value == "0x1"
		assert tokens[0].suffix == IntegerSuffix.ULL

	def test_hex_zero(self):
		tokens = tokenize("0x0")
		assert tokens[0].value == "0x0"

	def test_hex_no_digits_raises(self):
		with pytest.raises(LexerError, match="Invalid hex literal"):
			tokenize("0x")

	def test_hex_no_digits_followed_by_non_hex(self):
		with pytest.raises(LexerError, match="Invalid hex literal"):
			tokenize("0xG")

	def test_octal_basic(self):
		tokens = tokenize("077")
		assert tokens[0].type == TokenType.INTEGER_LITERAL
		assert tokens[0].value == "077"

	def test_octal_with_suffix(self):
		tokens = tokenize("077L")
		assert tokens[0].value == "077"
		assert tokens[0].suffix == IntegerSuffix.L

	def test_zero_is_decimal_not_octal(self):
		tokens = tokenize("0")
		assert tokens[0].type == TokenType.INTEGER_LITERAL
		assert tokens[0].value == "0"

	def test_zero_dot_is_float(self):
		tokens = tokenize("0.5")
		assert tokens[0].type == TokenType.FLOAT_LITERAL
		assert tokens[0].value == "0.5"


# --- Escape sequences in char literals ---


class TestCharEscapeSequences:
	def test_hex_escape_in_char(self):
		tokens = tokenize("'\\x41'")
		assert tokens[0].type == TokenType.CHAR_LITERAL
		assert tokens[0].value == "'\\x41'"

	def test_null_escape(self):
		tokens = tokenize("'\\0'")
		assert tokens[0].type == TokenType.CHAR_LITERAL
		assert tokens[0].value == "'\\0'"

	def test_backslash_escape(self):
		tokens = tokenize("'\\\\'")
		assert tokens[0].type == TokenType.CHAR_LITERAL
		assert tokens[0].value == "'\\\\'"

	def test_newline_escape(self):
		tokens = tokenize("'\\n'")
		assert tokens[0].value == "'\\n'"

	def test_tab_escape(self):
		tokens = tokenize("'\\t'")
		assert tokens[0].value == "'\\t'"

	def test_single_quote_escape(self):
		tokens = tokenize("'\\''")
		assert tokens[0].value == "'\\''"

	def test_octal_escape_in_char(self):
		tokens = tokenize("'\\101'")
		assert tokens[0].value == "'\\101'"

	def test_unterminated_char_literal(self):
		with pytest.raises(LexerError, match="Unterminated character literal"):
			tokenize("'a")

	def test_char_newline_in_literal(self):
		with pytest.raises(LexerError, match="Unterminated character literal"):
			tokenize("'\n'")


# --- interpret_c_escapes function ---


class TestInterpretCEscapes:
	def test_hex_escape(self):
		assert interpret_c_escapes("\\x41") == "A"

	def test_null_escape(self):
		assert interpret_c_escapes("\\0") == "\0"

	def test_octal_escape(self):
		assert interpret_c_escapes("\\101") == "A"

	def test_backslash_escape(self):
		assert interpret_c_escapes("\\\\") == "\\"

	def test_newline_escape(self):
		assert interpret_c_escapes("\\n") == "\n"

	def test_hex_no_digits_passthrough(self):
		assert interpret_c_escapes("\\x") == "\\x"

	def test_unknown_escape_passthrough(self):
		assert interpret_c_escapes("\\q") == "\\q"

	def test_octal_012(self):
		assert interpret_c_escapes("\\012") == "\n"

	def test_hex_truncates_to_byte(self):
		assert interpret_c_escapes("\\x141") == "A"

	def test_backslash_at_end(self):
		"""Trailing backslash with no following char is passed through."""
		assert interpret_c_escapes("abc\\") == "abc\\"

	def test_multiple_escapes(self):
		assert interpret_c_escapes("\\t\\n\\0") == "\t\n\0"

	def test_mixed_text_and_escapes(self):
		assert interpret_c_escapes("hi\\nbye") == "hi\nbye"


# --- Multi-character tokens near EOF ---


class TestTokensNearEOF:
	def test_plus_at_eof(self):
		tokens = tokenize("+")
		assert tokens[0].type == TokenType.PLUS
		assert tokens[1].type == TokenType.EOF

	def test_increment_at_eof(self):
		tokens = tokenize("++")
		assert tokens[0].type == TokenType.INCREMENT

	def test_less_at_eof(self):
		tokens = tokenize("<")
		assert tokens[0].type == TokenType.LESS

	def test_lshift_at_eof(self):
		tokens = tokenize("<<")
		assert tokens[0].type == TokenType.LSHIFT

	def test_lshift_assign_at_eof(self):
		tokens = tokenize("<<=")
		assert tokens[0].type == TokenType.LSHIFT_ASSIGN

	def test_arrow_at_eof(self):
		tokens = tokenize("->")
		assert tokens[0].type == TokenType.ARROW

	def test_rshift_assign_at_eof(self):
		tokens = tokenize(">>=")
		assert tokens[0].type == TokenType.RSHIFT_ASSIGN

	def test_and_at_eof(self):
		tokens = tokenize("&&")
		assert tokens[0].type == TokenType.AND

	def test_or_at_eof(self):
		tokens = tokenize("||")
		assert tokens[0].type == TokenType.OR

	def test_equal_at_eof(self):
		tokens = tokenize("==")
		assert tokens[0].type == TokenType.EQUAL

	def test_not_equal_at_eof(self):
		tokens = tokenize("!=")
		assert tokens[0].type == TokenType.NOT_EQUAL

	def test_ampersand_vs_and(self):
		"""Single & at EOF should be AMPERSAND, not AND."""
		tokens = tokenize("&")
		assert tokens[0].type == TokenType.AMPERSAND

	def test_pipe_vs_or(self):
		tokens = tokenize("|")
		assert tokens[0].type == TokenType.PIPE


# --- Comment handling ---


class TestCommentHandling:
	def test_line_comment_skipped(self):
		tokens = tokenize("a // comment\nb")
		assert types(tokens) == [TokenType.IDENTIFIER, TokenType.IDENTIFIER, TokenType.EOF]

	def test_block_comment_skipped(self):
		tokens = tokenize("a /* comment */ b")
		assert types(tokens) == [TokenType.IDENTIFIER, TokenType.IDENTIFIER, TokenType.EOF]

	def test_nested_block_comment_star(self):
		"""/* inside /* is not nesting -- first */ closes."""
		tokens = tokenize("a /* outer /* inner */ b")
		assert types(tokens) == [TokenType.IDENTIFIER, TokenType.IDENTIFIER, TokenType.EOF]

	def test_line_comment_inside_block(self):
		"""// inside block comment is ignored."""
		tokens = tokenize("a /* // not a line comment */ b")
		assert types(tokens) == [TokenType.IDENTIFIER, TokenType.IDENTIFIER, TokenType.EOF]

	def test_unterminated_block_comment(self):
		with pytest.raises(LexerError, match="Unterminated block comment"):
			tokenize("a /* never closed")

	def test_unterminated_string(self):
		with pytest.raises(LexerError, match="Unterminated string literal"):
			tokenize('"no end')

	def test_newline_in_string_raises(self):
		with pytest.raises(LexerError, match="Unterminated string literal"):
			tokenize('"line1\nline2"')

	def test_block_comment_multiline(self):
		tokens = tokenize("a /* line1\nline2\nline3 */ b")
		assert types(tokens) == [TokenType.IDENTIFIER, TokenType.IDENTIFIER, TokenType.EOF]

	def test_line_comment_at_eof(self):
		tokens = tokenize("a // trailing")
		assert types(tokens) == [TokenType.IDENTIFIER, TokenType.EOF]

	def test_block_comment_with_stars(self):
		tokens = tokenize("a /*** fancy ***/ b")
		assert types(tokens) == [TokenType.IDENTIFIER, TokenType.IDENTIFIER, TokenType.EOF]

	def test_slash_not_comment(self):
		"""Lone / followed by non-/ non-* is division."""
		tokens = tokenize("a / b")
		assert tokens[1].type == TokenType.SLASH


# --- Ellipsis vs dot-dot sequences ---


class TestEllipsisAndDots:
	def test_ellipsis(self):
		tokens = tokenize("...")
		assert tokens[0].type == TokenType.ELLIPSIS
		assert tokens[0].value == "..."

	def test_single_dot(self):
		tokens = tokenize(".")
		assert tokens[0].type == TokenType.DOT

	def test_two_dots_are_two_tokens(self):
		tokens = tokenize("..")
		assert tokens[0].type == TokenType.DOT
		assert tokens[1].type == TokenType.DOT
		assert tokens[2].type == TokenType.EOF

	def test_four_dots_is_ellipsis_plus_dot(self):
		tokens = tokenize("....")
		assert tokens[0].type == TokenType.ELLIPSIS
		assert tokens[1].type == TokenType.DOT

	def test_ellipsis_in_params(self):
		tokens = tokenize("(int a, ...)")
		ellipsis_tokens = [t for t in tokens if t.type == TokenType.ELLIPSIS]
		assert len(ellipsis_tokens) == 1

	def test_dot_followed_by_digit_is_float(self):
		tokens = tokenize(".5")
		assert tokens[0].type == TokenType.FLOAT_LITERAL
		assert tokens[0].value == ".5"

	def test_dot_member_access(self):
		tokens = tokenize("a.b")
		assert tokens[0].type == TokenType.IDENTIFIER
		assert tokens[1].type == TokenType.DOT
		assert tokens[2].type == TokenType.IDENTIFIER


# --- Identifier-like keywords ---


class TestIdentifierLikeKeywords:
	def test_int_is_keyword(self):
		tokens = tokenize("int")
		assert tokens[0].type == TokenType.INT

	def test_integer_is_identifier(self):
		tokens = tokenize("integer")
		assert tokens[0].type == TokenType.IDENTIFIER
		assert tokens[0].value == "integer"

	def test_return_is_keyword(self):
		tokens = tokenize("return")
		assert tokens[0].type == TokenType.RETURN

	def test_returning_is_identifier(self):
		tokens = tokenize("returning")
		assert tokens[0].type == TokenType.IDENTIFIER

	def test_if_is_keyword(self):
		tokens = tokenize("if")
		assert tokens[0].type == TokenType.IF

	def test_ifdef_is_identifier(self):
		tokens = tokenize("ifdef")
		assert tokens[0].type == TokenType.IDENTIFIER

	def test_do_is_keyword(self):
		tokens = tokenize("do")
		assert tokens[0].type == TokenType.DO

	def test_done_is_identifier(self):
		tokens = tokenize("done")
		assert tokens[0].type == TokenType.IDENTIFIER

	def test_for_is_keyword(self):
		tokens = tokenize("for")
		assert tokens[0].type == TokenType.FOR

	def test_forever_is_identifier(self):
		tokens = tokenize("forever")
		assert tokens[0].type == TokenType.IDENTIFIER

	def test_underscore_identifier(self):
		tokens = tokenize("_foo")
		assert tokens[0].type == TokenType.IDENTIFIER
		assert tokens[0].value == "_foo"

	def test_all_underscores(self):
		tokens = tokenize("___")
		assert tokens[0].type == TokenType.IDENTIFIER

	def test_Bool_keyword(self):
		tokens = tokenize("_Bool")
		assert tokens[0].type == TokenType.BOOL

	def test_Bool_prefix_is_identifier(self):
		tokens = tokenize("_Boolean")
		assert tokens[0].type == TokenType.IDENTIFIER


# --- Maximum-length identifiers ---


class TestLongIdentifiers:
	def test_very_long_identifier(self):
		name = "a" * 1000
		tokens = tokenize(name)
		assert tokens[0].type == TokenType.IDENTIFIER
		assert tokens[0].value == name

	def test_long_identifier_with_digits(self):
		name = "var_" + "0123456789" * 50
		tokens = tokenize(name)
		assert tokens[0].type == TokenType.IDENTIFIER
		assert tokens[0].value == name

	def test_identifier_starts_with_underscore_and_digits(self):
		name = "_" + "1" * 500
		tokens = tokenize(name)
		assert tokens[0].type == TokenType.IDENTIFIER
		assert tokens[0].value == name


# --- Empty source input ---


class TestEmptyInput:
	def test_empty_string(self):
		tokens = tokenize("")
		assert len(tokens) == 1
		assert tokens[0].type == TokenType.EOF

	def test_only_whitespace(self):
		tokens = tokenize("   \t\n\r  ")
		assert len(tokens) == 1
		assert tokens[0].type == TokenType.EOF

	def test_only_comment(self):
		tokens = tokenize("// nothing here")
		assert len(tokens) == 1
		assert tokens[0].type == TokenType.EOF

	def test_only_block_comment(self):
		tokens = tokenize("/* nothing */")
		assert len(tokens) == 1
		assert tokens[0].type == TokenType.EOF


# --- Float edge cases ---


class TestFloatEdgeCases:
	def test_float_with_exponent(self):
		tokens = tokenize("1e10")
		assert tokens[0].type == TokenType.FLOAT_LITERAL
		assert tokens[0].value == "1e10"

	def test_float_with_negative_exponent(self):
		tokens = tokenize("1e-5")
		assert tokens[0].type == TokenType.FLOAT_LITERAL

	def test_float_with_f_suffix(self):
		tokens = tokenize("3.14f")
		assert tokens[0].type == TokenType.FLOAT_LITERAL
		assert tokens[0].value == "3.14f"

	def test_float_with_L_suffix(self):
		tokens = tokenize("3.14L")
		assert tokens[0].type == TokenType.FLOAT_LITERAL

	def test_dot_float_with_exponent(self):
		tokens = tokenize(".5e2")
		assert tokens[0].type == TokenType.FLOAT_LITERAL
		assert tokens[0].value == ".5e2"

	def test_invalid_exponent_raises(self):
		with pytest.raises(LexerError, match="Invalid float literal exponent"):
			tokenize("1e")

	def test_invalid_exponent_with_sign_raises(self):
		with pytest.raises(LexerError, match="Invalid float literal exponent"):
			tokenize("1e+")


# --- Line/column tracking ---


class TestLineColumnTracking:
	def test_first_token_position(self):
		tokens = tokenize("x")
		assert tokens[0].line == 1
		assert tokens[0].column == 1

	def test_second_line_token(self):
		tokens = tokenize("a\nb")
		assert tokens[1].line == 2
		assert tokens[1].column == 1

	def test_column_after_spaces(self):
		tokens = tokenize("   x")
		assert tokens[0].column == 4

	def test_position_after_block_comment(self):
		tokens = tokenize("/* comment */x")
		assert tokens[0].value == "x"
		assert tokens[0].line == 1


# --- Miscellaneous tricky inputs ---


class TestMiscTricky:
	def test_all_compound_assign_operators(self):
		source = "+= -= *= /= %= &= |= ^= <<= >>="
		tokens = tokenize(source)
		expected = [
			TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
			TokenType.STAR_ASSIGN, TokenType.SLASH_ASSIGN,
			TokenType.PERCENT_ASSIGN, TokenType.AMP_ASSIGN,
			TokenType.PIPE_ASSIGN, TokenType.CARET_ASSIGN,
			TokenType.LSHIFT_ASSIGN, TokenType.RSHIFT_ASSIGN,
			TokenType.EOF,
		]
		assert types(tokens) == expected

	def test_string_with_double_quote_escape(self):
		tokens = tokenize(r'"say \"hello\""')
		assert tokens[0].type == TokenType.STRING_LITERAL

	def test_consecutive_operators_no_space(self):
		tokens = tokenize("a+-b")
		assert types(tokens) == [
			TokenType.IDENTIFIER, TokenType.PLUS, TokenType.MINUS,
			TokenType.IDENTIFIER, TokenType.EOF,
		]

	def test_hash_token(self):
		tokens = tokenize("#")
		assert tokens[0].type == TokenType.HASH

	def test_question_colon_ternary(self):
		tokens = tokenize("a?b:c")
		assert types(tokens) == [
			TokenType.IDENTIFIER, TokenType.QUESTION,
			TokenType.IDENTIFIER, TokenType.COLON,
			TokenType.IDENTIFIER, TokenType.EOF,
		]

	def test_unexpected_character_raises(self):
		with pytest.raises(LexerError, match="Unexpected character"):
			tokenize("@")

	def test_slash_assign_vs_comment(self):
		"""'/=' should be SLASH_ASSIGN, not start of comment."""
		tokens = tokenize("a /= b")
		assert tokens[1].type == TokenType.SLASH_ASSIGN
