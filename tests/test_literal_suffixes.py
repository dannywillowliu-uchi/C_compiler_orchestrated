"""Tests for integer literal suffix tracking through lexer, parser, and AST."""

import pytest

from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.tokens import IntegerSuffix, TokenType
from compiler.ast_nodes import IntLiteral, ExprStmt, Program


# --- Lexer suffix detection ---


class TestLexerSuffixDetection:
	"""Verify that the lexer correctly detects and stores integer suffixes."""

	@pytest.mark.parametrize("source,expected_suffix", [
		("42", IntegerSuffix.NONE),
		("0", IntegerSuffix.NONE),
		("100", IntegerSuffix.NONE),
	])
	def test_no_suffix(self, source: str, expected_suffix: IntegerSuffix) -> None:
		tokens = Lexer(source).tokenize()
		tok = tokens[0]
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.suffix == expected_suffix

	@pytest.mark.parametrize("source,expected_suffix", [
		("42u", IntegerSuffix.U),
		("42U", IntegerSuffix.U),
	])
	def test_unsigned_suffix(self, source: str, expected_suffix: IntegerSuffix) -> None:
		tokens = Lexer(source).tokenize()
		tok = tokens[0]
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.suffix == expected_suffix

	@pytest.mark.parametrize("source,expected_suffix", [
		("42l", IntegerSuffix.L),
		("42L", IntegerSuffix.L),
	])
	def test_long_suffix(self, source: str, expected_suffix: IntegerSuffix) -> None:
		tokens = Lexer(source).tokenize()
		tok = tokens[0]
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.suffix == expected_suffix

	@pytest.mark.parametrize("source,expected_suffix", [
		("42ul", IntegerSuffix.UL),
		("42UL", IntegerSuffix.UL),
		("42uL", IntegerSuffix.UL),
		("42Ul", IntegerSuffix.UL),
		("42lu", IntegerSuffix.UL),
		("42LU", IntegerSuffix.UL),
		("42lU", IntegerSuffix.UL),
		("42Lu", IntegerSuffix.UL),
	])
	def test_unsigned_long_suffix(self, source: str, expected_suffix: IntegerSuffix) -> None:
		tokens = Lexer(source).tokenize()
		tok = tokens[0]
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.suffix == expected_suffix

	@pytest.mark.parametrize("source,expected_suffix", [
		("42ll", IntegerSuffix.LL),
		("42LL", IntegerSuffix.LL),
		("42lL", IntegerSuffix.LL),
		("42Ll", IntegerSuffix.LL),
	])
	def test_long_long_suffix(self, source: str, expected_suffix: IntegerSuffix) -> None:
		tokens = Lexer(source).tokenize()
		tok = tokens[0]
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.suffix == expected_suffix

	@pytest.mark.parametrize("source,expected_suffix", [
		("42ull", IntegerSuffix.ULL),
		("42ULL", IntegerSuffix.ULL),
		("42uLL", IntegerSuffix.ULL),
		("42Ull", IntegerSuffix.ULL),
		("42llu", IntegerSuffix.ULL),
		("42LLU", IntegerSuffix.ULL),
		("42llU", IntegerSuffix.ULL),
		("42LLu", IntegerSuffix.ULL),
	])
	def test_unsigned_long_long_suffix(self, source: str, expected_suffix: IntegerSuffix) -> None:
		tokens = Lexer(source).tokenize()
		tok = tokens[0]
		assert tok.type == TokenType.INTEGER_LITERAL
		assert tok.suffix == expected_suffix


# --- Numeric value extraction (suffix stripped from value) ---


class TestNumericValueExtraction:
	"""Verify that token values contain only the numeric part, not the suffix."""

	@pytest.mark.parametrize("source,expected_value", [
		("42u", "42"),
		("42UL", "42"),
		("42ull", "42"),
		("42LL", "42"),
		("42l", "42"),
		("100", "100"),
	])
	def test_decimal_value_without_suffix(self, source: str, expected_value: str) -> None:
		tokens = Lexer(source).tokenize()
		assert tokens[0].value == expected_value

	@pytest.mark.parametrize("source,expected_value", [
		("0xFF", "0xFF"),
		("0xFFu", "0xFF"),
		("0xFFUL", "0xFF"),
		("0xDEADull", "0xDEAD"),
	])
	def test_hex_value_without_suffix(self, source: str, expected_value: str) -> None:
		tokens = Lexer(source).tokenize()
		assert tokens[0].value == expected_value

	@pytest.mark.parametrize("source,expected_value", [
		("077", "077"),
		("077u", "077"),
		("077ULL", "077"),
	])
	def test_octal_value_without_suffix(self, source: str, expected_value: str) -> None:
		tokens = Lexer(source).tokenize()
		assert tokens[0].value == expected_value


# --- Hex and octal with suffixes ---


class TestHexOctalSuffixes:
	"""Verify suffix detection works for hex and octal literals."""

	def test_hex_unsigned(self) -> None:
		tokens = Lexer("0xFFu").tokenize()
		assert tokens[0].suffix == IntegerSuffix.U
		assert tokens[0].value == "0xFF"

	def test_hex_unsigned_long_long(self) -> None:
		tokens = Lexer("0xDEADULL").tokenize()
		assert tokens[0].suffix == IntegerSuffix.ULL
		assert tokens[0].value == "0xDEAD"

	def test_octal_long(self) -> None:
		tokens = Lexer("077L").tokenize()
		assert tokens[0].suffix == IntegerSuffix.L
		assert tokens[0].value == "077"

	def test_octal_unsigned_long(self) -> None:
		tokens = Lexer("0777UL").tokenize()
		assert tokens[0].suffix == IntegerSuffix.UL
		assert tokens[0].value == "0777"


# --- Suffix propagation to AST ---


class TestSuffixPropagationToAST:
	"""Verify suffix information propagates from lexer through parser to AST nodes."""

	def _parse_expr(self, source: str) -> IntLiteral:
		"""Parse a single expression statement and return the IntLiteral."""
		full = f"void f() {{ {source}; }}"
		program = Parser.from_source(full).parse()
		assert isinstance(program, Program)
		func = program.declarations[0]
		stmt = func.body.statements[0]
		assert isinstance(stmt, ExprStmt)
		assert isinstance(stmt.expression, IntLiteral)
		return stmt.expression

	def test_no_suffix_propagation(self) -> None:
		lit = self._parse_expr("42")
		assert lit.value == 42
		assert lit.suffix == ""

	def test_unsigned_suffix_propagation(self) -> None:
		lit = self._parse_expr("42u")
		assert lit.value == 42
		assert lit.suffix == "u"

	def test_long_suffix_propagation(self) -> None:
		lit = self._parse_expr("42L")
		assert lit.value == 42
		assert lit.suffix == "l"

	def test_unsigned_long_suffix_propagation(self) -> None:
		lit = self._parse_expr("42UL")
		assert lit.value == 42
		assert lit.suffix == "ul"

	def test_long_long_suffix_propagation(self) -> None:
		lit = self._parse_expr("42LL")
		assert lit.value == 42
		assert lit.suffix == "ll"

	def test_unsigned_long_long_suffix_propagation(self) -> None:
		lit = self._parse_expr("42ULL")
		assert lit.value == 42
		assert lit.suffix == "ull"

	def test_hex_suffix_propagation(self) -> None:
		lit = self._parse_expr("0xFFu")
		assert lit.value == 0xFF
		assert lit.suffix == "u"

	def test_zero_suffix_propagation(self) -> None:
		lit = self._parse_expr("0L")
		assert lit.value == 0
		assert lit.suffix == "l"


# --- Existing integer parsing still works ---


class TestExistingIntegerParsing:
	"""Verify that existing integer literal parsing is not broken."""

	def test_simple_decimal(self) -> None:
		tokens = Lexer("123").tokenize()
		assert tokens[0].type == TokenType.INTEGER_LITERAL
		assert tokens[0].value == "123"
		assert tokens[0].suffix == IntegerSuffix.NONE

	def test_zero(self) -> None:
		tokens = Lexer("0").tokenize()
		assert tokens[0].type == TokenType.INTEGER_LITERAL
		assert tokens[0].value == "0"

	def test_hex_literal(self) -> None:
		tokens = Lexer("0xFF").tokenize()
		assert tokens[0].type == TokenType.INTEGER_LITERAL
		assert tokens[0].value == "0xFF"

	def test_octal_literal(self) -> None:
		tokens = Lexer("077").tokenize()
		assert tokens[0].type == TokenType.INTEGER_LITERAL
		assert tokens[0].value == "077"

	def test_float_not_affected(self) -> None:
		tokens = Lexer("3.14").tokenize()
		assert tokens[0].type == TokenType.FLOAT_LITERAL
		assert tokens[0].suffix == IntegerSuffix.NONE

	def test_multiple_tokens_in_expression(self) -> None:
		tokens = Lexer("42u + 10L").tokenize()
		assert tokens[0].type == TokenType.INTEGER_LITERAL
		assert tokens[0].value == "42"
		assert tokens[0].suffix == IntegerSuffix.U
		assert tokens[2].type == TokenType.INTEGER_LITERAL
		assert tokens[2].value == "10"
		assert tokens[2].suffix == IntegerSuffix.L

	def test_full_program_with_suffixes(self) -> None:
		source = "int main() { return 42ULL; }"
		program = Parser.from_source(source).parse()
		func = program.declarations[0]
		ret = func.body.statements[0]
		assert isinstance(ret.expression, IntLiteral)
		assert ret.expression.value == 42
		assert ret.expression.suffix == "ull"
