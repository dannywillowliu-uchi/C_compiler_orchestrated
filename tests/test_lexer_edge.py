"""Edge-case tests for the lexer: operators, numeric literals, strings, chars, identifiers, comments."""

import pytest

from compiler.lexer import Lexer, LexerError, interpret_c_escapes
from compiler.tokens import IntegerSuffix, TokenType


def tokenize(source: str) -> list:
	return Lexer(source).tokenize()


def token_types(source: str) -> list[TokenType]:
	return [t.type for t in tokenize(source)]


def token_values(source: str) -> list[str]:
	return [t.value for t in tokenize(source)]


# ---------------------------------------------------------------------------
# Multi-character operators
# ---------------------------------------------------------------------------

class TestMultiCharOperators:
	def test_increment_decrement(self):
		toks = tokenize("++ --")
		assert toks[0].type == TokenType.INCREMENT
		assert toks[1].type == TokenType.DECREMENT

	def test_compound_assignment_operators(self):
		ops = {
			"+=": TokenType.PLUS_ASSIGN,
			"-=": TokenType.MINUS_ASSIGN,
			"*=": TokenType.STAR_ASSIGN,
			"/=": TokenType.SLASH_ASSIGN,
			"%=": TokenType.PERCENT_ASSIGN,
			"&=": TokenType.AMP_ASSIGN,
			"|=": TokenType.PIPE_ASSIGN,
			"^=": TokenType.CARET_ASSIGN,
		}
		for op, expected in ops.items():
			toks = tokenize(op)
			assert toks[0].type == expected, f"Failed for {op}"

	def test_shift_assign(self):
		toks = tokenize("<<= >>=")
		assert toks[0].type == TokenType.LSHIFT_ASSIGN
		assert toks[1].type == TokenType.RSHIFT_ASSIGN

	def test_shift_operators(self):
		toks = tokenize("<< >>")
		assert toks[0].type == TokenType.LSHIFT
		assert toks[1].type == TokenType.RSHIFT

	def test_comparison_operators(self):
		toks = tokenize("== != <= >=")
		assert toks[0].type == TokenType.EQUAL
		assert toks[1].type == TokenType.NOT_EQUAL
		assert toks[2].type == TokenType.LESS_EQUAL
		assert toks[3].type == TokenType.GREATER_EQUAL

	def test_logical_operators(self):
		toks = tokenize("&& ||")
		assert toks[0].type == TokenType.AND
		assert toks[1].type == TokenType.OR

	def test_arrow_operator(self):
		toks = tokenize("->")
		assert toks[0].type == TokenType.ARROW
		assert toks[0].value == "->"

	def test_ellipsis(self):
		toks = tokenize("...")
		assert toks[0].type == TokenType.ELLIPSIS
		assert toks[0].value == "..."

	def test_shift_vs_shift_assign_disambiguation(self):
		"""<< followed by = should be <<=, not << then =."""
		toks = tokenize("<<=")
		assert toks[0].type == TokenType.LSHIFT_ASSIGN
		assert len(toks) == 2  # LSHIFT_ASSIGN + EOF

	def test_less_less_space_equals(self):
		"""<< = (with space) should be << then =."""
		toks = tokenize("<< =")
		assert toks[0].type == TokenType.LSHIFT
		assert toks[1].type == TokenType.ASSIGN

	def test_plus_plus_equals(self):
		"""++= should tokenize as ++ then =."""
		toks = tokenize("++=")
		assert toks[0].type == TokenType.INCREMENT
		assert toks[1].type == TokenType.ASSIGN

	def test_minus_minus_greater(self):
		"""--> should tokenize as -- then >."""
		toks = tokenize("-->")
		assert toks[0].type == TokenType.DECREMENT
		assert toks[1].type == TokenType.GREATER

	def test_dot_vs_ellipsis(self):
		""".. should be two dots, ... should be ellipsis."""
		toks = tokenize("...")
		assert toks[0].type == TokenType.ELLIPSIS
		toks2 = tokenize(". .")
		assert toks2[0].type == TokenType.DOT
		assert toks2[1].type == TokenType.DOT


# ---------------------------------------------------------------------------
# Numeric literal edge cases
# ---------------------------------------------------------------------------

class TestNumericLiterals:
	def test_hex_literal_lowercase(self):
		toks = tokenize("0xff")
		assert toks[0].type == TokenType.INTEGER_LITERAL
		assert toks[0].value == "0xff"

	def test_hex_literal_uppercase(self):
		toks = tokenize("0XFF")
		assert toks[0].type == TokenType.INTEGER_LITERAL
		assert toks[0].value == "0XFF"

	def test_hex_literal_mixed_case(self):
		toks = tokenize("0xAbCd")
		assert toks[0].value == "0xAbCd"

	def test_octal_literal(self):
		toks = tokenize("077")
		assert toks[0].type == TokenType.INTEGER_LITERAL
		assert toks[0].value == "077"

	def test_zero_literal(self):
		toks = tokenize("0")
		assert toks[0].type == TokenType.INTEGER_LITERAL
		assert toks[0].value == "0"

	def test_suffix_u(self):
		toks = tokenize("42u")
		assert toks[0].suffix == IntegerSuffix.U

	def test_suffix_U(self):
		toks = tokenize("42U")
		assert toks[0].suffix == IntegerSuffix.U

	def test_suffix_l(self):
		toks = tokenize("0l")
		assert toks[0].suffix == IntegerSuffix.L

	def test_suffix_L(self):
		toks = tokenize("0L")
		assert toks[0].suffix == IntegerSuffix.L

	def test_suffix_ll(self):
		toks = tokenize("0ll")
		assert toks[0].suffix == IntegerSuffix.LL

	def test_suffix_LL(self):
		toks = tokenize("0LL")
		assert toks[0].suffix == IntegerSuffix.LL

	def test_suffix_ul(self):
		toks = tokenize("0ul")
		assert toks[0].suffix == IntegerSuffix.UL

	def test_suffix_UL(self):
		toks = tokenize("0UL")
		assert toks[0].suffix == IntegerSuffix.UL

	def test_suffix_ull(self):
		toks = tokenize("0ull")
		assert toks[0].suffix == IntegerSuffix.ULL

	def test_suffix_ULL(self):
		toks = tokenize("0ULL")
		assert toks[0].suffix == IntegerSuffix.ULL

	def test_suffix_lu(self):
		"""l followed by u should also parse as UL."""
		toks = tokenize("0lu")
		assert toks[0].suffix == IntegerSuffix.UL

	def test_suffix_llu(self):
		"""ll followed by u should parse as ULL."""
		toks = tokenize("0llu")
		assert toks[0].suffix == IntegerSuffix.ULL

	def test_hex_with_suffix(self):
		toks = tokenize("0xFFu")
		assert toks[0].type == TokenType.INTEGER_LITERAL
		assert toks[0].value == "0xFF"
		assert toks[0].suffix == IntegerSuffix.U

	def test_octal_with_suffix(self):
		toks = tokenize("077L")
		assert toks[0].type == TokenType.INTEGER_LITERAL
		assert toks[0].suffix == IntegerSuffix.L

	def test_invalid_hex_literal(self):
		with pytest.raises(LexerError):
			tokenize("0x")

	def test_float_with_exponent(self):
		toks = tokenize("1e10")
		assert toks[0].type == TokenType.FLOAT_LITERAL
		assert toks[0].value == "1e10"

	def test_float_negative_exponent(self):
		toks = tokenize("1e-5")
		assert toks[0].type == TokenType.FLOAT_LITERAL

	def test_float_positive_exponent(self):
		toks = tokenize("1e+5")
		assert toks[0].type == TokenType.FLOAT_LITERAL

	def test_float_with_dot(self):
		toks = tokenize("3.14")
		assert toks[0].type == TokenType.FLOAT_LITERAL

	def test_float_starting_with_dot(self):
		toks = tokenize(".5")
		assert toks[0].type == TokenType.FLOAT_LITERAL
		assert toks[0].value == ".5"

	def test_float_dot_with_exponent(self):
		toks = tokenize(".5e3")
		assert toks[0].type == TokenType.FLOAT_LITERAL

	def test_float_suffix_f(self):
		toks = tokenize("3.14f")
		assert toks[0].type == TokenType.FLOAT_LITERAL
		assert toks[0].value == "3.14f"

	def test_float_suffix_L(self):
		toks = tokenize("3.14L")
		assert toks[0].type == TokenType.FLOAT_LITERAL

	def test_invalid_float_exponent(self):
		with pytest.raises(LexerError):
			tokenize("1e")

	def test_zero_dot_zero(self):
		toks = tokenize("0.0")
		assert toks[0].type == TokenType.FLOAT_LITERAL


# ---------------------------------------------------------------------------
# String literal edge cases
# ---------------------------------------------------------------------------

class TestStringLiterals:
	def test_empty_string(self):
		toks = tokenize('""')
		assert toks[0].type == TokenType.STRING_LITERAL
		assert toks[0].value == '""'

	def test_adjacent_string_concatenation(self):
		"""Adjacent string literals should be merged."""
		toks = tokenize('"hello" " " "world"')
		# Should produce a single merged string literal
		strings = [t for t in toks if t.type == TokenType.STRING_LITERAL]
		assert len(strings) == 1
		assert strings[0].value == '"hello world"'

	def test_adjacent_strings_with_newline(self):
		toks = tokenize('"a"\n"b"')
		strings = [t for t in toks if t.type == TokenType.STRING_LITERAL]
		assert len(strings) == 1
		assert strings[0].value == '"ab"'

	def test_string_with_escape_newline(self):
		toks = tokenize(r'"hello\nworld"')
		assert toks[0].type == TokenType.STRING_LITERAL
		assert r"\n" in toks[0].value

	def test_string_with_hex_escape(self):
		toks = tokenize(r'"A\x42C"')
		assert toks[0].type == TokenType.STRING_LITERAL

	def test_string_with_null(self):
		toks = tokenize(r'"null\0char"')
		assert toks[0].type == TokenType.STRING_LITERAL

	def test_unterminated_string(self):
		with pytest.raises(LexerError):
			tokenize('"unterminated')

	def test_unterminated_string_newline(self):
		with pytest.raises(LexerError):
			tokenize('"line1\nline2"')

	def test_string_with_escaped_quote(self):
		toks = tokenize(r'"say \"hi\""')
		assert toks[0].type == TokenType.STRING_LITERAL

	def test_string_with_backslash_at_end(self):
		"""Backslash right before EOF in string should error."""
		with pytest.raises(LexerError):
			tokenize('"abc\\')


# ---------------------------------------------------------------------------
# interpret_c_escapes edge cases
# ---------------------------------------------------------------------------

class TestInterpretCEscapes:
	def test_simple_escapes(self):
		assert interpret_c_escapes(r"\n") == "\n"
		assert interpret_c_escapes(r"\t") == "\t"
		assert interpret_c_escapes(r"\\") == "\\"
		assert interpret_c_escapes(r"\'") == "'"
		assert interpret_c_escapes(r'\"') == '"'
		assert interpret_c_escapes(r"\a") == "\a"
		assert interpret_c_escapes(r"\b") == "\b"
		assert interpret_c_escapes(r"\f") == "\f"
		assert interpret_c_escapes(r"\r") == "\r"
		assert interpret_c_escapes(r"\v") == "\v"

	def test_null_escape(self):
		assert interpret_c_escapes(r"\0") == "\0"

	def test_hex_escape(self):
		assert interpret_c_escapes(r"\x41") == "A"
		assert interpret_c_escapes(r"\x61") == "a"

	def test_hex_escape_truncates_to_byte(self):
		# \x100 should truncate to 0x00
		assert interpret_c_escapes(r"\x100") == "\x00"

	def test_hex_escape_no_digits(self):
		"""\\x with no hex digits should pass through."""
		assert interpret_c_escapes(r"\x") == "\\x"

	def test_octal_escape(self):
		assert interpret_c_escapes(r"\101") == "A"  # 0o101 = 65 = 'A'

	def test_octal_zero_with_more_digits(self):
		"""\\012 should be octal 012 = 10 = newline."""
		assert interpret_c_escapes(r"\012") == "\n"

	def test_octal_max_three_digits(self):
		"""Octal escapes consume at most 3 digits."""
		result = interpret_c_escapes(r"\1234")
		# \123 = 83 = 'S', then '4'
		assert result == "S4"

	def test_unknown_escape_passthrough(self):
		"""Unknown escapes like \\z pass through as-is."""
		assert interpret_c_escapes(r"\z") == "\\z"

	def test_trailing_backslash(self):
		"""A trailing backslash with nothing after it should be kept."""
		assert interpret_c_escapes("\\") == "\\"

	def test_empty_string(self):
		assert interpret_c_escapes("") == ""

	def test_no_escapes(self):
		assert interpret_c_escapes("hello") == "hello"

	def test_mixed_escapes(self):
		result = interpret_c_escapes(r"A\nB\tC\x44")
		assert result == "A\nB\tCD"


# ---------------------------------------------------------------------------
# Character literal edge cases
# ---------------------------------------------------------------------------

class TestCharLiterals:
	def test_simple_char(self):
		toks = tokenize("'a'")
		assert toks[0].type == TokenType.CHAR_LITERAL
		assert toks[0].value == "'a'"

	def test_null_char(self):
		toks = tokenize(r"'\0'")
		assert toks[0].type == TokenType.CHAR_LITERAL

	def test_escaped_newline_char(self):
		toks = tokenize(r"'\n'")
		assert toks[0].type == TokenType.CHAR_LITERAL

	def test_escaped_backslash_char(self):
		toks = tokenize(r"'\\'")
		assert toks[0].type == TokenType.CHAR_LITERAL

	def test_escaped_single_quote(self):
		toks = tokenize(r"'\''")
		assert toks[0].type == TokenType.CHAR_LITERAL

	def test_hex_char(self):
		toks = tokenize(r"'\x41'")
		assert toks[0].type == TokenType.CHAR_LITERAL

	def test_unterminated_char(self):
		with pytest.raises(LexerError):
			tokenize("'a")

	def test_unterminated_char_newline(self):
		with pytest.raises(LexerError):
			tokenize("'a\n'")

	def test_multi_char_literal(self):
		"""Multi-character char literals like 'ab' should tokenize (validity checked later)."""
		toks = tokenize("'ab'")
		assert toks[0].type == TokenType.CHAR_LITERAL
		assert toks[0].value == "'ab'"

	def test_empty_char_literal(self):
		"""Empty char literal '' should tokenize (semantic check later)."""
		toks = tokenize("''")
		assert toks[0].type == TokenType.CHAR_LITERAL


# ---------------------------------------------------------------------------
# Identifier edge cases
# ---------------------------------------------------------------------------

class TestIdentifiers:
	def test_underscore_start(self):
		toks = tokenize("_foo")
		assert toks[0].type == TokenType.IDENTIFIER
		assert toks[0].value == "_foo"

	def test_double_underscore(self):
		toks = tokenize("__bar")
		assert toks[0].type == TokenType.IDENTIFIER
		assert toks[0].value == "__bar"

	def test_underscore_only(self):
		toks = tokenize("_")
		assert toks[0].type == TokenType.IDENTIFIER
		assert toks[0].value == "_"

	def test_identifier_with_digits(self):
		toks = tokenize("var123")
		assert toks[0].type == TokenType.IDENTIFIER
		assert toks[0].value == "var123"

	def test_long_identifier(self):
		name = "a" * 200
		toks = tokenize(name)
		assert toks[0].type == TokenType.IDENTIFIER
		assert toks[0].value == name

	def test_keyword_prefix_is_identifier(self):
		"""'integer' starts with 'int' but should be an identifier."""
		toks = tokenize("integer")
		assert toks[0].type == TokenType.IDENTIFIER
		assert toks[0].value == "integer"

	def test_identifier_not_keyword(self):
		toks = tokenize("iff")
		assert toks[0].type == TokenType.IDENTIFIER

	def test_Bool_keyword(self):
		toks = tokenize("_Bool")
		assert toks[0].type == TokenType.BOOL

	def test_similar_to_Bool_is_identifier(self):
		toks = tokenize("_bool")
		assert toks[0].type == TokenType.IDENTIFIER


# ---------------------------------------------------------------------------
# Comment handling edge cases
# ---------------------------------------------------------------------------

class TestComments:
	def test_line_comment_skipped(self):
		toks = tokenize("a // comment\nb")
		types = [t.type for t in toks if t.type != TokenType.EOF]
		assert types == [TokenType.IDENTIFIER, TokenType.IDENTIFIER]

	def test_block_comment_skipped(self):
		toks = tokenize("a /* comment */ b")
		types = [t.type for t in toks if t.type != TokenType.EOF]
		assert types == [TokenType.IDENTIFIER, TokenType.IDENTIFIER]

	def test_block_comment_multiline(self):
		toks = tokenize("a /* line1\nline2\nline3 */ b")
		types = [t.type for t in toks if t.type != TokenType.EOF]
		assert types == [TokenType.IDENTIFIER, TokenType.IDENTIFIER]

	def test_nested_block_comment_start_in_line_comment(self):
		"""/* inside // should not start a block comment."""
		toks = tokenize("a // /* not a block\nb")
		types = [t.type for t in toks if t.type != TokenType.EOF]
		assert types == [TokenType.IDENTIFIER, TokenType.IDENTIFIER]

	def test_unterminated_block_comment(self):
		with pytest.raises(LexerError):
			tokenize("/* never closed")

	def test_block_comment_with_star_inside(self):
		"""Stars inside block comment should not prematurely close it."""
		toks = tokenize("a /* * ** *** */ b")
		types = [t.type for t in toks if t.type != TokenType.EOF]
		assert types == [TokenType.IDENTIFIER, TokenType.IDENTIFIER]

	def test_line_comment_at_eof(self):
		toks = tokenize("a // comment")
		types = [t.type for t in toks if t.type != TokenType.EOF]
		assert types == [TokenType.IDENTIFIER]

	def test_slash_not_comment(self):
		"""A bare / should be a SLASH token."""
		toks = tokenize("a / b")
		assert toks[1].type == TokenType.SLASH

	def test_block_comment_immediately_closed(self):
		toks = tokenize("a /**/ b")
		types = [t.type for t in toks if t.type != TokenType.EOF]
		assert types == [TokenType.IDENTIFIER, TokenType.IDENTIFIER]


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------

class TestPositionTracking:
	def test_first_token_position(self):
		toks = tokenize("x")
		assert toks[0].line == 1
		assert toks[0].column == 1

	def test_second_line_position(self):
		toks = tokenize("x\ny")
		assert toks[1].line == 2
		assert toks[1].column == 1

	def test_column_after_spaces(self):
		toks = tokenize("   x")
		assert toks[0].column == 4

	def test_multichar_operator_position(self):
		toks = tokenize("  <<=")
		assert toks[0].line == 1
		assert toks[0].column == 3


# ---------------------------------------------------------------------------
# Miscellaneous edge cases
# ---------------------------------------------------------------------------

class TestMiscEdgeCases:
	def test_empty_source(self):
		toks = tokenize("")
		assert len(toks) == 1
		assert toks[0].type == TokenType.EOF

	def test_only_whitespace(self):
		toks = tokenize("   \n\t\n  ")
		assert len(toks) == 1
		assert toks[0].type == TokenType.EOF

	def test_only_comment(self):
		toks = tokenize("// just a comment")
		assert len(toks) == 1
		assert toks[0].type == TokenType.EOF

	def test_unexpected_character(self):
		with pytest.raises(LexerError):
			tokenize("@")

	def test_hash_token(self):
		toks = tokenize("#")
		assert toks[0].type == TokenType.HASH
