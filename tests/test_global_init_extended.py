"""Tests for global initializer values: char literals, negative literals, enum constants."""

from compiler.ir import IRGlobalVar
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer


def compile_to_ir(source: str):
	tokens = Lexer(source).tokenize()
	prog = Parser(tokens).parse()
	analyzer = SemanticAnalyzer()
	analyzer.analyze(prog)
	return IRGenerator().generate(prog)


def get_global(ir_prog, name: str) -> IRGlobalVar:
	matches = [g for g in ir_prog.globals if g.name == name]
	assert len(matches) == 1, f"Expected 1 global named '{name}', found {len(matches)}"
	return matches[0]


# ---------------------------------------------------------------
# Char literal initializers in global arrays
# ---------------------------------------------------------------


class TestCharLiteralGlobalInit:
	def test_char_array_from_char_literals(self):
		"""char arr[] = {'A', 'B', 'C'} should store ordinals."""
		ir_prog = compile_to_ir("""
			char arr[3] = {'A', 'B', 'C'};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [65, 66, 67]

	def test_char_array_with_escape_chars(self):
		"""Char literals with escape sequences."""
		ir_prog = compile_to_ir("""
			char arr[3] = {'\\n', '\\t', '\\0'};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [10, 9, 0]

	def test_char_array_mixed_int_and_char(self):
		"""Mix of int and char literals in an array."""
		ir_prog = compile_to_ir("""
			int arr[4] = {'A', 1, 'Z', 0};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [65, 1, 90, 0]


# ---------------------------------------------------------------
# Negative literal initializers in global arrays
# ---------------------------------------------------------------


class TestNegativeLiteralGlobalInit:
	def test_negative_int_array(self):
		"""int arr[] = {-1, -2, -3} should store negative values."""
		ir_prog = compile_to_ir("""
			int arr[3] = {-1, -2, -3};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [-1, -2, -3]

	def test_mixed_positive_negative(self):
		"""Mix of positive and negative literals."""
		ir_prog = compile_to_ir("""
			int arr[4] = {10, -20, 30, -40};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [10, -20, 30, -40]

	def test_negative_char_literal(self):
		"""Negated char literal in global init."""
		ir_prog = compile_to_ir("""
			int arr[2] = {-'A', -'B'};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [-65, -66]

	def test_zero_and_negative(self):
		"""Zero mixed with negatives."""
		ir_prog = compile_to_ir("""
			int arr[3] = {0, -1, 0};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [0, -1, 0]


# ---------------------------------------------------------------
# Enum constant initializers in global arrays
# ---------------------------------------------------------------


class TestEnumConstantGlobalInit:
	def test_enum_array_init(self):
		"""Array initialized with enum constants."""
		ir_prog = compile_to_ir("""
			enum Color { RED, GREEN, BLUE };
			int arr[3] = {RED, GREEN, BLUE};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [0, 1, 2]

	def test_enum_with_explicit_values(self):
		"""Enum constants with explicit values."""
		ir_prog = compile_to_ir("""
			enum Status { OK = 200, NOT_FOUND = 404, ERROR = 500 };
			int arr[3] = {OK, NOT_FOUND, ERROR};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [200, 404, 500]

	def test_enum_mixed_with_literals(self):
		"""Mix of enum constants and integer literals."""
		ir_prog = compile_to_ir("""
			enum Vals { A = 10, B = 20 };
			int arr[4] = {A, 5, B, 99};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [10, 5, 20, 99]


# ---------------------------------------------------------------
# Mixed initializer types
# ---------------------------------------------------------------


class TestMixedGlobalInit:
	def test_all_types_combined(self):
		"""Char literals, negatives, enums, and plain ints together."""
		ir_prog = compile_to_ir("""
			enum Misc { VAL = 42 };
			int arr[5] = {'X', -3, VAL, 0, 100};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [88, -3, 42, 0, 100]

	def test_partial_init_zero_padded(self):
		"""Partial initializer should be zero-padded."""
		ir_prog = compile_to_ir("""
			int arr[5] = {-1, 'Z'};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [-1, 90, 0, 0, 0]


# ---------------------------------------------------------------
# Nested struct initializers
# ---------------------------------------------------------------


class TestNestedStructGlobalInit:
	def test_struct_with_char_field(self):
		"""Struct initializer with char literal value."""
		ir_prog = compile_to_ir("""
			struct Pair { int x; int y; };
			struct Pair p = {'A', -1};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "p")
		assert g.initializer_values == [65, -1]

	def test_struct_with_enum_field(self):
		"""Struct initializer with enum constant."""
		ir_prog = compile_to_ir("""
			enum Dir { UP = 1, DOWN = 2 };
			struct S { int dir; int val; };
			struct S s = {UP, -10};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "s")
		assert g.initializer_values == [1, -10]

	def test_nested_struct_array(self):
		"""Array of structs with mixed initializers."""
		ir_prog = compile_to_ir("""
			enum Consts { X = 5 };
			struct V { int a; int b; };
			struct V arr[2] = {{X, -1}, {'B', 0}};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [5, -1, 66, 0]
