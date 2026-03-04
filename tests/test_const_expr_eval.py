"""Tests for compile-time constant expression evaluation in global initializers and array sizes."""

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
# sizeof(type) in global initializers
# ---------------------------------------------------------------


class TestSizeofInGlobalInit:
	def test_sizeof_int_scalar(self):
		"""Global initialized with sizeof(int)."""
		ir_prog = compile_to_ir("""
			int sz = sizeof(int);
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "sz")
		assert g.initializer == 4

	def test_sizeof_char_scalar(self):
		"""Global initialized with sizeof(char)."""
		ir_prog = compile_to_ir("""
			int sz = sizeof(char);
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "sz")
		assert g.initializer == 1

	def test_sizeof_pointer_scalar(self):
		"""Global initialized with sizeof(int*)."""
		ir_prog = compile_to_ir("""
			int sz = sizeof(int*);
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "sz")
		assert g.initializer == 8

	def test_sizeof_struct_in_global(self):
		"""Global initialized with sizeof(struct)."""
		ir_prog = compile_to_ir("""
			struct Pair { int x; int y; };
			int sz = sizeof(struct Pair);
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "sz")
		assert g.initializer == 8

	def test_sizeof_in_global_array_init(self):
		"""sizeof used in global array initializer list."""
		ir_prog = compile_to_ir("""
			int sizes[3] = {sizeof(int), sizeof(char), sizeof(int*)};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "sizes")
		assert g.initializer_values == [4, 1, 8]

	def test_sizeof_struct_with_padding(self):
		"""sizeof struct with alignment padding."""
		ir_prog = compile_to_ir("""
			struct Mixed { char c; int i; };
			int sz = sizeof(struct Mixed);
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "sz")
		assert g.initializer == 8

	def test_sizeof_union_in_global(self):
		"""sizeof union in global initializer."""
		ir_prog = compile_to_ir("""
			union U { int i; char c; };
			int sz = sizeof(union U);
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "sz")
		assert g.initializer == 4


# ---------------------------------------------------------------
# Enum arithmetic in global initializers
# ---------------------------------------------------------------


class TestEnumArithmeticGlobalInit:
	def test_enum_add(self):
		"""Global initialized with enum constant addition."""
		ir_prog = compile_to_ir("""
			enum Vals { A = 10, B = 20 };
			int sum = A + B;
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "sum")
		assert g.initializer == 30

	def test_enum_subtract(self):
		"""Global initialized with enum subtraction."""
		ir_prog = compile_to_ir("""
			enum Vals { X = 50, Y = 15 };
			int diff = X - Y;
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "diff")
		assert g.initializer == 35

	def test_enum_multiply(self):
		"""Global initialized with enum multiplication."""
		ir_prog = compile_to_ir("""
			enum Nums { A = 3, B = 7 };
			int prod = A * B;
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "prod")
		assert g.initializer == 21

	def test_enum_mixed_with_literal(self):
		"""Enum constant combined with integer literal in global init."""
		ir_prog = compile_to_ir("""
			enum Consts { BASE = 100 };
			int val = BASE + 42;
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "val")
		assert g.initializer == 142

	def test_enum_in_array_init(self):
		"""Enum arithmetic expressions in global array initializer."""
		ir_prog = compile_to_ir("""
			enum Nums { A = 2, B = 3 };
			int arr[3] = {A + B, A * B, B - A};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [5, 6, 1]

	def test_enum_with_expression_values(self):
		"""Enum constants defined with constant expressions."""
		ir_prog = compile_to_ir("""
			enum Seq { A = 1, B = 2, C = 4 };
			int arr[3] = {A, B, C};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [1, 2, 4]


# ---------------------------------------------------------------
# Constant expressions for array sizes
# ---------------------------------------------------------------


class TestConstExprArraySizes:
	def test_binary_expr_array_size(self):
		"""Array sized by binary constant expression."""
		ir_prog = compile_to_ir("""
			int arr[2 + 3] = {1, 2, 3, 4, 5};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [1, 2, 3, 4, 5]
		assert g.total_size == 20

	def test_multiply_expr_array_size(self):
		"""Array sized by multiplication expression."""
		ir_prog = compile_to_ir("""
			int arr[2 * 3] = {10, 20, 30, 40, 50, 60};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [10, 20, 30, 40, 50, 60]
		assert g.total_size == 24

	def test_enum_const_array_size(self):
		"""Array sized by enum constant."""
		ir_prog = compile_to_ir("""
			enum Sizes { SIZE = 4 };
			int arr[SIZE] = {1, 2, 3, 4};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [1, 2, 3, 4]
		assert g.total_size == 16

	def test_sizeof_array_size(self):
		"""Array sized by sizeof expression."""
		ir_prog = compile_to_ir("""
			int arr[sizeof(int)] = {1, 2, 3, 4};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [1, 2, 3, 4]
		assert g.total_size == 16

	def test_complex_expr_array_size(self):
		"""Array sized by complex constant expression."""
		ir_prog = compile_to_ir("""
			enum Dims { N = 2 };
			int arr[N * 3 + 1] = {1, 2, 3, 4, 5, 6, 7};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [1, 2, 3, 4, 5, 6, 7]
		assert g.total_size == 28


# ---------------------------------------------------------------
# Binary arithmetic on integer constants
# ---------------------------------------------------------------


class TestBinaryArithmeticConst:
	def test_addition_in_global(self):
		ir_prog = compile_to_ir("""
			int x = 10 + 20;
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "x")
		assert g.initializer == 30

	def test_subtraction_in_global(self):
		ir_prog = compile_to_ir("""
			int x = 100 - 37;
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "x")
		assert g.initializer == 63

	def test_multiplication_in_global(self):
		ir_prog = compile_to_ir("""
			int x = 6 * 7;
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "x")
		assert g.initializer == 42

	def test_division_in_global(self):
		ir_prog = compile_to_ir("""
			int x = 100 / 4;
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "x")
		assert g.initializer == 25

	def test_modulo_in_global(self):
		ir_prog = compile_to_ir("""
			int x = 17 % 5;
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "x")
		assert g.initializer == 2

	def test_nested_arithmetic(self):
		ir_prog = compile_to_ir("""
			int x = (3 + 4) * 2 - 1;
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "x")
		assert g.initializer == 13


# ---------------------------------------------------------------
# Unary operations on constants
# ---------------------------------------------------------------


class TestUnaryConstOps:
	def test_unary_minus_global(self):
		ir_prog = compile_to_ir("""
			int x = -42;
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "x")
		assert g.initializer == -42

	def test_bitwise_not_global(self):
		ir_prog = compile_to_ir("""
			int x = ~0;
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "x")
		assert g.initializer == -1

	def test_unary_in_array_init(self):
		ir_prog = compile_to_ir("""
			int arr[3] = {-1, ~0, -(-5)};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [-1, -1, 5]


# ---------------------------------------------------------------
# Cast expressions on constants
# ---------------------------------------------------------------


class TestCastConstExpr:
	def test_cast_in_global_init(self):
		ir_prog = compile_to_ir("""
			int x = (int)65;
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "x")
		assert g.initializer == 65

	def test_cast_in_array_init(self):
		ir_prog = compile_to_ir("""
			int arr[2] = {(int)'A', (int)3};
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "arr")
		assert g.initializer_values == [65, 3]


# ---------------------------------------------------------------
# Sizeof arithmetic combinations
# ---------------------------------------------------------------


class TestSizeofArithmeticCombo:
	def test_sizeof_plus_literal(self):
		ir_prog = compile_to_ir("""
			int x = sizeof(int) + 1;
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "x")
		assert g.initializer == 5

	def test_sizeof_times_literal(self):
		ir_prog = compile_to_ir("""
			int x = sizeof(int) * 10;
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "x")
		assert g.initializer == 40

	def test_sizeof_struct_arithmetic(self):
		ir_prog = compile_to_ir("""
			struct S { int a; int b; };
			int x = sizeof(struct S) / sizeof(int);
			int main() { return 0; }
		""")
		g = get_global(ir_prog, "x")
		assert g.initializer == 2
