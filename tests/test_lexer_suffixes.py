"""Comprehensive tests for integer suffixes, hex floats, and numeric literal edge cases."""

import pytest

from compiler.lexer import Lexer, LexerError
from compiler.tokens import IntegerSuffix, TokenType


def lex(source: str):
	return Lexer(source).tokenize()


def first(source: str):
	return lex(source)[0]


# --- Integer suffix: LL / ll ---

class TestLongLongSuffix:
	def test_ll_lower(self):
		tok = first("123ll")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "123"
		assert tok.suffix == IntegerSuffix.LL

	def test_ll_upper(self):
		tok = first("456LL")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "456"
		assert tok.suffix == IntegerSuffix.LL

	def test_ll_mixed_case(self):
		# C standard allows mixed case lL / Ll
		tok = first("789lL")
		assert tok.suffix == IntegerSuffix.LL

	def test_zero_ll(self):
		tok = first("0LL")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "0"
		assert tok.suffix == IntegerSuffix.LL

	def test_hex_ll(self):
		tok = first("0xFFll")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "0xFF"
		assert tok.suffix == IntegerSuffix.LL

	def test_octal_ll(self):
		tok = first("077LL")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "077"
		assert tok.suffix == IntegerSuffix.LL


# --- Integer suffix: ULL / ull ---

class TestUnsignedLongLongSuffix:
	def test_ull_lower(self):
		tok = first("100ull")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "100"
		assert tok.suffix == IntegerSuffix.ULL

	def test_ull_upper(self):
		tok = first("200ULL")
		assert tok.suffix == IntegerSuffix.ULL

	def test_llu_order(self):
		# u can come after ll: 100LLU
		tok = first("100LLu")
		assert tok.suffix == IntegerSuffix.ULL

	def test_llu_upper(self):
		tok = first("100LLU")
		assert tok.suffix == IntegerSuffix.ULL

	def test_hex_ull(self):
		tok = first("0xDEADull")
		assert tok.value == "0xDEAD"
		assert tok.suffix == IntegerSuffix.ULL

	def test_octal_ull(self):
		tok = first("0777ULL")
		assert tok.value == "0777"
		assert tok.suffix == IntegerSuffix.ULL


# --- Existing suffixes still work ---

class TestExistingSuffixes:
	def test_u_suffix(self):
		tok = first("42u")
		assert tok.suffix == IntegerSuffix.U

	def test_U_suffix(self):
		tok = first("42U")
		assert tok.suffix == IntegerSuffix.U

	def test_l_suffix(self):
		tok = first("42l")
		assert tok.suffix == IntegerSuffix.L

	def test_L_suffix(self):
		tok = first("42L")
		assert tok.suffix == IntegerSuffix.L

	def test_ul_suffix(self):
		tok = first("42ul")
		assert tok.suffix == IntegerSuffix.UL

	def test_UL_suffix(self):
		tok = first("42UL")
		assert tok.suffix == IntegerSuffix.UL

	def test_lu_suffix(self):
		tok = first("42lu")
		assert tok.suffix == IntegerSuffix.UL

	def test_no_suffix(self):
		tok = first("42")
		assert tok.suffix == IntegerSuffix.NONE


# --- Hex float literals ---

class TestHexFloats:
	def test_basic_hex_float(self):
		tok = first("0x1.0p10")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == "0x1.0p10"

	def test_hex_float_upper(self):
		tok = first("0X1.0P10")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == "0X1.0P10"

	def test_hex_float_negative_exponent(self):
		tok = first("0x1.8p-3")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == "0x1.8p-3"

	def test_hex_float_positive_exponent(self):
		tok = first("0x1.Fp+4")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == "0x1.Fp+4"

	def test_hex_float_no_fraction(self):
		# 0x1p10 - integer part only, exponent present
		tok = first("0x1p10")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == "0x1p10"

	def test_hex_float_no_integer_part(self):
		# 0x.8p0 - no integer digits before dot
		tok = first("0x.8p0")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == "0x.8p0"

	def test_hex_float_f_suffix(self):
		tok = first("0x1.0p10f")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == "0x1.0p10f"

	def test_hex_float_L_suffix(self):
		tok = first("0x1.0p10L")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == "0x1.0p10L"

	def test_hex_float_missing_exponent(self):
		with pytest.raises(LexerError, match="requires exponent"):
			lex("0x1.0")

	def test_hex_float_invalid_exponent(self):
		with pytest.raises(LexerError, match="exponent"):
			lex("0x1.0p")

	def test_hex_float_large(self):
		tok = first("0xABCDEF.123p+20")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == "0xABCDEF.123p+20"


# --- Edge case: zero literal ---

class TestZeroLiteral:
	def test_plain_zero(self):
		tok = first("0")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "0"
		assert tok.suffix == IntegerSuffix.NONE

	def test_zero_u(self):
		tok = first("0u")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "0"
		assert tok.suffix == IntegerSuffix.U

	def test_zero_l(self):
		tok = first("0L")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "0"
		assert tok.suffix == IntegerSuffix.L

	def test_zero_ull(self):
		tok = first("0ULL")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "0"
		assert tok.suffix == IntegerSuffix.ULL

	def test_zero_float(self):
		tok = first("0.0")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == "0.0"

	def test_zero_exponent(self):
		tok = first("0e0")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == "0e0"

	def test_hex_zero(self):
		tok = first("0x0")
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.value == "0x0"


# --- Edge case: floats without leading digit ---

class TestDotFloats:
	def test_dot_5(self):
		tok = first(".5")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == ".5"

	def test_dot_123(self):
		tok = first(".123")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == ".123"

	def test_dot_5_e3(self):
		tok = first(".5e3")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == ".5e3"

	def test_dot_5_E_neg(self):
		tok = first(".5E-2")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == ".5E-2"

	def test_dot_5_f_suffix(self):
		tok = first(".5f")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == ".5f"

	def test_dot_5_L_suffix(self):
		tok = first(".5L")
		assert tok.type == TokenType.FLOAT_LITERAL
		assert tok.value == ".5L"


# --- Multiple tokens with suffixes ---

class TestSuffixInContext:
	def test_ll_in_expression(self):
		tokens = lex("100LL + 200ULL")
		assert tokens[0].suffix == IntegerSuffix.LL
		assert tokens[0].value == "100"
		assert tokens[1].type == TokenType.PLUS
		assert tokens[2].suffix == IntegerSuffix.ULL
		assert tokens[2].value == "200"

	def test_hex_float_in_assignment(self):
		tokens = lex("x = 0x1.0p10;")
		assert tokens[0].type == TokenType.IDENTIFIER
		assert tokens[1].type == TokenType.ASSIGN
		assert tokens[2].type == TokenType.FLOAT_LITERAL
		assert tokens[2].value == "0x1.0p10"
		assert tokens[3].type == TokenType.SEMICOLON

	def test_mixed_numeric_types(self):
		tokens = lex("0 .5 0xFF 42LL 0x1p0")
		assert tokens[0].type == TokenType.INTEGER_LITERAL
		assert tokens[0].value == "0"
		assert tokens[1].type == TokenType.FLOAT_LITERAL
		assert tokens[1].value == ".5"
		assert tokens[2].type == TokenType.INTEGER_LITERAL
		assert tokens[2].value == "0xFF"
		assert tokens[3].type == TokenType.INTEGER_LITERAL
		assert tokens[3].suffix == IntegerSuffix.LL
		assert tokens[4].type == TokenType.FLOAT_LITERAL
		assert tokens[4].value == "0x1p0"

	def test_octal_zero_vs_decimal(self):
		# '0' followed by non-octal should be decimal zero
		tokens = lex("0 + 1")
		assert tokens[0].type == TokenType.INTEGER_LITERAL
		assert tokens[0].value == "0"
		assert tokens[0].suffix == IntegerSuffix.NONE
