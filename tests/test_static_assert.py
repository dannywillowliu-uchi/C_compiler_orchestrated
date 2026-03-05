"""Tests for _Static_assert support."""

import pytest

from compiler.parser import Parser, ParseError
from compiler.semantic import SemanticAnalyzer, SemanticError


def parse(source: str):
	return Parser.from_source(source).parse()


def analyze(source: str) -> list[SemanticError]:
	program = parse(source)
	analyzer = SemanticAnalyzer()
	try:
		analyzer.analyze(program)
	except SemanticError:
		pass
	return analyzer.errors


# --- Parsing tests ---


class TestStaticAssertParsing:
	def test_global_scope(self):
		prog = parse("_Static_assert(1, \"ok\");")
		assert len(prog.declarations) == 1

	def test_inside_function(self):
		prog = parse("int main() { _Static_assert(1, \"ok\"); return 0; }")
		assert len(prog.declarations) == 1

	def test_inside_struct(self):
		prog = parse("struct S { int x; _Static_assert(1, \"size check\"); int y; };")
		from compiler.ast_nodes import StructDecl
		decl = prog.declarations[0]
		assert isinstance(decl, StructDecl)
		assert len(decl.members) == 2
		assert len(decl.static_asserts) == 1

	def test_inside_union(self):
		prog = parse("union U { int x; _Static_assert(1, \"union check\"); float f; };")
		from compiler.ast_nodes import UnionDecl
		decl = prog.declarations[0]
		assert isinstance(decl, UnionDecl)
		assert len(decl.members) == 2
		assert len(decl.static_asserts) == 1

	def test_missing_semicolon(self):
		with pytest.raises(ParseError):
			parse("_Static_assert(1, \"ok\")")

	def test_missing_message(self):
		with pytest.raises(ParseError):
			parse("_Static_assert(1);")

	def test_missing_expression(self):
		with pytest.raises(ParseError):
			parse("_Static_assert(, \"msg\");")


# --- Semantic analysis tests ---


class TestStaticAssertSemantic:
	def test_valid_assertion_no_errors(self):
		errors = analyze("_Static_assert(1, \"should pass\");")
		assert len(errors) == 0

	def test_valid_nonzero_expression(self):
		errors = analyze("_Static_assert(42, \"nonzero\");")
		assert len(errors) == 0

	def test_failing_assertion(self):
		errors = analyze("_Static_assert(0, \"this fails\");")
		assert len(errors) == 1
		assert "static assertion failed" in str(errors[0])
		assert "this fails" in str(errors[0])

	def test_arithmetic_expression(self):
		errors = analyze("_Static_assert(2 + 2 == 4, \"math works\");")
		assert len(errors) == 0

	def test_arithmetic_expression_fails(self):
		errors = analyze("_Static_assert(2 + 2 == 5, \"bad math\");")
		assert len(errors) == 1
		assert "static assertion failed" in str(errors[0])

	def test_sizeof_expression(self):
		errors = analyze("_Static_assert(sizeof(int) == 4, \"int is 4 bytes\");")
		assert len(errors) == 0

	def test_sizeof_char_expression(self):
		errors = analyze("_Static_assert(sizeof(char) == 1, \"char is 1 byte\");")
		assert len(errors) == 0

	def test_negation_expression(self):
		errors = analyze("_Static_assert(!0, \"not zero\");")
		assert len(errors) == 0

	def test_negation_fails(self):
		errors = analyze("_Static_assert(!1, \"not one\");")
		assert len(errors) == 1

	def test_bitwise_expression(self):
		errors = analyze("_Static_assert((3 & 1) == 1, \"bitwise and\");")
		assert len(errors) == 0

	def test_shift_expression(self):
		errors = analyze("_Static_assert((1 << 3) == 8, \"shift\");")
		assert len(errors) == 0

	def test_ternary_expression(self):
		errors = analyze("_Static_assert(1 ? 1 : 0, \"ternary\");")
		assert len(errors) == 0

	def test_comparison_operators(self):
		errors = analyze("_Static_assert(3 > 2, \"gt\"); _Static_assert(2 < 3, \"lt\");")
		assert len(errors) == 0

	def test_logical_operators(self):
		errors = analyze("_Static_assert(1 && 1, \"and\"); _Static_assert(0 || 1, \"or\");")
		assert len(errors) == 0

	def test_in_function_scope(self):
		errors = analyze("int main() { _Static_assert(1, \"in func\"); return 0; }")
		assert len(errors) == 0

	def test_failing_in_function_scope(self):
		errors = analyze("int main() { _Static_assert(0, \"func fail\"); return 0; }")
		assert len(errors) == 1
		assert "func fail" in str(errors[0])

	def test_in_struct_scope(self):
		errors = analyze("struct S { int x; _Static_assert(sizeof(int) == 4, \"struct check\"); };")
		assert len(errors) == 0

	def test_failing_in_struct_scope(self):
		errors = analyze("struct S { int x; _Static_assert(0, \"struct fail\"); };")
		assert len(errors) == 1
		assert "struct fail" in str(errors[0])

	def test_in_union_scope(self):
		errors = analyze("union U { int x; _Static_assert(sizeof(int) == 4, \"union check\"); float f; };")
		assert len(errors) == 0

	def test_failing_in_union_scope(self):
		errors = analyze("union U { int x; _Static_assert(0, \"union fail\"); float f; };")
		assert len(errors) == 1
		assert "union fail" in str(errors[0])

	def test_enum_constant_expression(self):
		src = "enum E { A = 1, B = 2 }; _Static_assert(A == 1, \"enum val\");"
		errors = analyze(src)
		assert len(errors) == 0

	def test_multiple_assertions(self):
		src = """
		_Static_assert(1, "first");
		_Static_assert(2, "second");
		_Static_assert(sizeof(char) == 1, "third");
		"""
		errors = analyze(src)
		assert len(errors) == 0

	def test_multiple_failures(self):
		src = """
		_Static_assert(0, "fail1");
		_Static_assert(0, "fail2");
		"""
		errors = analyze(src)
		assert len(errors) == 2

	def test_non_constant_expression(self):
		src = "int x; _Static_assert(x, \"not const\");"
		errors = analyze(src)
		assert len(errors) >= 1
		assert "not a constant expression" in str(errors[0])

	def test_char_literal_expression(self):
		errors = analyze("_Static_assert('A', \"char lit\");")
		assert len(errors) == 0

	def test_cast_expression(self):
		errors = analyze("_Static_assert((int)1, \"cast\");")
		assert len(errors) == 0

	def test_unary_minus(self):
		errors = analyze("_Static_assert(-1, \"negative\");")
		assert len(errors) == 0

	def test_complex_expression(self):
		errors = analyze("_Static_assert((sizeof(int) * 2) == 8, \"complex\");")
		assert len(errors) == 0
