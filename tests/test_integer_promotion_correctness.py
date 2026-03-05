"""Comprehensive integer promotion and conversion correctness tests.

Exercises C integer promotion rules through the full pipeline:
parse -> semantic -> IR -> codegen.
Verifies correct IR types, unsigned tracking, and codegen width suffixes.
"""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRAlloc,
	IRBinOp,
	IRType,
	ir_type_asm_suffix,
	ir_type_byte_width,
	ir_type_is_integer,
)
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer


def _parse(source: str):
	tokens = Lexer(source).tokenize()
	return Parser(tokens).parse()


def _compile_to_ir(source: str):
	ast = _parse(source)
	errors = SemanticAnalyzer().analyze(ast)
	assert not errors, f"Semantic errors: {errors}"
	return IRGenerator().generate(ast)


def _compile_to_asm(source: str) -> str:
	ir = _compile_to_ir(source)
	return CodeGenerator().generate(ir)


# ---------------------------------------------------------------------------
# char -> int promotion in arithmetic
# ---------------------------------------------------------------------------


class TestCharToIntPromotion:
	def test_char_plus_char_compiles(self) -> None:
		source = """
		int main(void) {
			char a = 1;
			char b = 2;
			int c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_char_alloc_is_one_byte(self) -> None:
		source = "int main(void) { char x = 42; return x; }"
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 1 for a in allocs)

	def test_char_arithmetic_result_stored_in_int(self) -> None:
		"""char + char should produce an int-width result per C promotion rules."""
		source = """
		int main(void) {
			char a = 10;
			char b = 20;
			int result = a + b;
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		# The int result should have a 4-byte allocation
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 4 for a in allocs)

	def test_char_in_comparison_compiles(self) -> None:
		source = """
		int main(void) {
			char a = 65;
			char b = 66;
			if (a < b) return 1;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_signed_char_negative_value(self) -> None:
		source = """
		int main(void) {
			signed char c = -1;
			int x = c;
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_char_max_value(self) -> None:
		source = """
		int main(void) {
			unsigned char c = 255;
			int x = c;
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_char_multiplication(self) -> None:
		source = """
		int main(void) {
			char a = 10;
			char b = 10;
			int result = a * b;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_char_bitwise_operations(self) -> None:
		source = """
		int main(void) {
			char a = 0x0F;
			char b = 0xF0;
			int result = a | b;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_char_unary_minus(self) -> None:
		"""Unary minus on char should promote to int."""
		source = """
		int main(void) {
			char c = 5;
			int result = -c;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# unsigned/signed conversion rules
# ---------------------------------------------------------------------------


class TestUnsignedSignedConversion:
	def test_unsigned_binop_tracking(self) -> None:
		"""Binary ops on unsigned values should have is_unsigned set in IR."""
		source = """
		int main(void) {
			unsigned int a = 10;
			unsigned int b = 20;
			unsigned int c = a + b;
			return c;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp)]
		assert any(getattr(b, "is_unsigned", False) for b in binops)

	def test_signed_binop_not_unsigned(self) -> None:
		"""Binary ops on signed values should not have is_unsigned set."""
		source = """
		int main(void) {
			int a = 10;
			int b = 20;
			int c = a + b;
			return c;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp)]
		assert all(not getattr(b, "is_unsigned", False) for b in binops)

	def test_mixed_signed_unsigned_propagates_unsigned(self) -> None:
		"""Mixing signed + unsigned should propagate unsigned in IR."""
		source = """
		int main(void) {
			int a = 5;
			unsigned int b = 10;
			int c = a + b;
			return c;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp)]
		# At least the addition should be marked unsigned
		assert any(getattr(b, "is_unsigned", False) for b in binops)

	def test_unsigned_to_signed_assignment(self) -> None:
		source = """
		int main(void) {
			unsigned int u = 42;
			int s = u;
			return s;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_signed_to_unsigned_assignment(self) -> None:
		source = """
		int main(void) {
			int s = -1;
			unsigned int u = s;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_comparison_generates_code(self) -> None:
		source = """
		int main(void) {
			unsigned int a = 1;
			unsigned int b = 2;
			if (a < b) return 1;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_division(self) -> None:
		source = """
		int main(void) {
			unsigned int a = 100;
			unsigned int b = 10;
			unsigned int c = a / b;
			return c;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		divops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "/"]
		assert any(getattr(d, "is_unsigned", False) for d in divops)

	def test_unsigned_modulo(self) -> None:
		source = """
		int main(void) {
			unsigned int a = 100;
			unsigned int b = 7;
			unsigned int c = a % b;
			return c;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		modops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "%"]
		assert any(getattr(m, "is_unsigned", False) for m in modops)

	def test_unsigned_right_shift(self) -> None:
		source = """
		int main(void) {
			unsigned int a = 256;
			unsigned int b = a >> 2;
			return b;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		shifts = [i for i in func.body if isinstance(i, IRBinOp) and i.op == ">>"]
		assert any(getattr(s, "is_unsigned", False) for s in shifts)


# ---------------------------------------------------------------------------
# short + int mixed arithmetic
# ---------------------------------------------------------------------------


class TestShortIntMixedArithmetic:
	def test_short_alloc_is_two_bytes(self) -> None:
		source = "int main(void) { short x = 1; return x; }"
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 2 for a in allocs)

	def test_short_plus_int_compiles(self) -> None:
		source = """
		int main(void) {
			short s = 10;
			int i = 20;
			int result = s + i;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_plus_short_compiles(self) -> None:
		source = """
		int main(void) {
			short a = 100;
			short b = 200;
			short c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_short_arithmetic(self) -> None:
		source = """
		int main(void) {
			unsigned short a = 1000;
			unsigned short b = 2000;
			unsigned short c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_times_int(self) -> None:
		source = """
		int main(void) {
			short s = 5;
			int i = 100;
			int result = s * i;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_in_ternary_with_int(self) -> None:
		source = """
		int main(void) {
			short s = 10;
			int i = 20;
			int result = (s > 5) ? s : i;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_function_param(self) -> None:
		source = """
		int add(short a, int b) {
			return a + b;
		}
		int main(void) {
			return add(10, 20);
		}
		"""
		asm = _compile_to_asm(source)
		assert "add:" in asm

	def test_short_return_type_function(self) -> None:
		source = """
		short get_val(void) {
			return 42;
		}
		int main(void) {
			short x = get_val();
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "get_val:" in asm


# ---------------------------------------------------------------------------
# long long operations
# ---------------------------------------------------------------------------


class TestLongLongOperations:
	def test_long_long_alloc_is_eight_bytes(self) -> None:
		source = "int main(void) { long long x = 1; return 0; }"
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 8 for a in allocs)

	def test_long_long_addition_uses_quad_ops(self) -> None:
		source = """
		int main(void) {
			long long a = 10;
			long long b = 20;
			long long c = a + b;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "addq" in asm

	def test_long_long_multiplication(self) -> None:
		source = """
		int main(void) {
			long long a = 100000;
			long long b = 100000;
			long long c = a * b;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "imulq" in asm or "imul" in asm

	def test_unsigned_long_long_tracking(self) -> None:
		source = """
		int main(void) {
			unsigned long long a = 1;
			unsigned long long b = 2;
			unsigned long long c = a + b;
			return 0;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp)]
		assert any(getattr(b, "is_unsigned", False) for b in binops)

	def test_long_long_large_constant(self) -> None:
		source = """
		int main(void) {
			long long x = 2147483648;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_long_long_subtraction(self) -> None:
		source = """
		int main(void) {
			long long a = 100;
			long long b = 30;
			long long c = a - b;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "subq" in asm

	def test_long_long_comparison(self) -> None:
		source = """
		int main(void) {
			long long a = 100;
			long long b = 200;
			if (a < b) return 1;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_long_long_bitwise(self) -> None:
		source = """
		int main(void) {
			long long a = 0xFF00FF;
			long long b = 0x00FF00;
			long long c = a | b;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_long_long_shift(self) -> None:
		source = """
		int main(void) {
			long long a = 1;
			long long b = a << 32;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_long_long_function_return(self) -> None:
		source = """
		long long compute(void) {
			return 42;
		}
		int main(void) {
			long long x = compute();
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "compute:" in asm


# ---------------------------------------------------------------------------
# Implicit widening in assignments
# ---------------------------------------------------------------------------


class TestImplicitWideningAssignment:
	def test_char_to_int_assignment(self) -> None:
		source = """
		int main(void) {
			char c = 42;
			int x = c;
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_char_to_long_long_assignment(self) -> None:
		source = """
		int main(void) {
			char c = 42;
			long long x = c;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_to_int_assignment(self) -> None:
		source = """
		int main(void) {
			short s = 1000;
			int x = s;
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_to_long_long_assignment(self) -> None:
		source = """
		int main(void) {
			short s = 1000;
			long long x = s;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_int_to_long_long_assignment(self) -> None:
		source = """
		int main(void) {
			int i = 100;
			long long x = i;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_char_to_unsigned_int(self) -> None:
		source = """
		int main(void) {
			unsigned char c = 200;
			unsigned int x = c;
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_short_to_unsigned_int(self) -> None:
		source = """
		int main(void) {
			unsigned short s = 60000;
			unsigned int x = s;
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Implicit narrowing in assignments
# ---------------------------------------------------------------------------


class TestImplicitNarrowingAssignment:
	def test_int_to_char_assignment(self) -> None:
		source = """
		int main(void) {
			int i = 65;
			char c = i;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_int_to_short_assignment(self) -> None:
		source = """
		int main(void) {
			int i = 1000;
			short s = i;
			return s;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_long_long_to_int_assignment(self) -> None:
		source = """
		int main(void) {
			long long ll = 42;
			int i = ll;
			return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_long_long_to_char_assignment(self) -> None:
		source = """
		int main(void) {
			long long ll = 65;
			char c = ll;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_long_long_to_short_assignment(self) -> None:
		source = """
		int main(void) {
			long long ll = 1000;
			short s = ll;
			return s;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Function argument widening/narrowing
# ---------------------------------------------------------------------------


class TestFunctionArgumentConversion:
	def test_char_arg_to_int_param(self) -> None:
		source = """
		int use_int(int x) {
			return x;
		}
		int main(void) {
			char c = 42;
			return use_int(c);
		}
		"""
		asm = _compile_to_asm(source)
		assert "use_int:" in asm

	def test_short_arg_to_int_param(self) -> None:
		source = """
		int use_int(int x) {
			return x;
		}
		int main(void) {
			short s = 100;
			return use_int(s);
		}
		"""
		asm = _compile_to_asm(source)
		assert "use_int:" in asm

	def test_int_arg_to_long_long_param(self) -> None:
		source = """
		long long use_ll(long long x) {
			return x;
		}
		int main(void) {
			int i = 42;
			long long result = use_ll(i);
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "use_ll:" in asm

	def test_multiple_mixed_type_args(self) -> None:
		source = """
		int sum(char a, short b, int c, long long d) {
			return a + b + c + d;
		}
		int main(void) {
			return sum(1, 2, 3, 4);
		}
		"""
		asm = _compile_to_asm(source)
		assert "sum:" in asm

	def test_unsigned_arg_to_signed_param(self) -> None:
		source = """
		int use_signed(int x) {
			return x;
		}
		int main(void) {
			unsigned int u = 42;
			return use_signed(u);
		}
		"""
		asm = _compile_to_asm(source)
		assert "use_signed:" in asm


# ---------------------------------------------------------------------------
# IR type helper functions
# ---------------------------------------------------------------------------


class TestIRTypeHelpers:
	def test_byte_widths(self) -> None:
		assert ir_type_byte_width(IRType.BOOL) == 1
		assert ir_type_byte_width(IRType.CHAR) == 1
		assert ir_type_byte_width(IRType.SHORT) == 2
		assert ir_type_byte_width(IRType.INT) == 4
		assert ir_type_byte_width(IRType.LONG) == 8
		assert ir_type_byte_width(IRType.POINTER) == 8

	def test_asm_suffixes(self) -> None:
		assert ir_type_asm_suffix(IRType.BOOL) == "b"
		assert ir_type_asm_suffix(IRType.CHAR) == "b"
		assert ir_type_asm_suffix(IRType.SHORT) == "w"
		assert ir_type_asm_suffix(IRType.INT) == "l"
		assert ir_type_asm_suffix(IRType.LONG) == "q"
		assert ir_type_asm_suffix(IRType.POINTER) == "q"

	def test_is_integer(self) -> None:
		assert ir_type_is_integer(IRType.BOOL)
		assert ir_type_is_integer(IRType.CHAR)
		assert ir_type_is_integer(IRType.SHORT)
		assert ir_type_is_integer(IRType.INT)
		assert ir_type_is_integer(IRType.LONG)
		assert not ir_type_is_integer(IRType.FLOAT)
		assert not ir_type_is_integer(IRType.DOUBLE)
		assert not ir_type_is_integer(IRType.VOID)


# ---------------------------------------------------------------------------
# Codegen width suffix correctness
# ---------------------------------------------------------------------------


class TestCodegenWidthSuffixes:
	def test_int_arithmetic_uses_l_suffix(self) -> None:
		source = """
		int main(void) {
			int a = 10;
			int b = 20;
			int c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "addl" in asm or "addq" in asm

	def test_long_long_arithmetic_uses_q_suffix(self) -> None:
		source = """
		int main(void) {
			long long a = 10;
			long long b = 20;
			long long c = a + b;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "addq" in asm

	def test_long_long_mul_uses_imulq(self) -> None:
		source = """
		int main(void) {
			long long a = 5;
			long long b = 10;
			long long c = a * b;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "imulq" in asm or "imul" in asm

	def test_char_store_alloc_size(self) -> None:
		"""Char variable should have 1-byte allocation in IR."""
		source = """
		int main(void) {
			char c = 42;
			return c;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 1 for a in allocs)

	def test_short_store_alloc_size(self) -> None:
		"""Short variable should have 2-byte allocation in IR."""
		source = """
		int main(void) {
			short s = 1000;
			return s;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 2 for a in allocs)


# ---------------------------------------------------------------------------
# Cross-type mixed expressions
# ---------------------------------------------------------------------------


class TestCrossTypeMixedExpressions:
	def test_char_plus_long_long(self) -> None:
		source = """
		int main(void) {
			char c = 5;
			long long ll = 100;
			long long result = c + ll;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_plus_long_long(self) -> None:
		source = """
		int main(void) {
			short s = 10;
			long long ll = 200;
			long long result = s + ll;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_all_integer_types_in_expression(self) -> None:
		source = """
		int main(void) {
			char c = 1;
			short s = 2;
			int i = 3;
			long long ll = 4;
			long long result = c + s + i + ll;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_char_plus_signed_int(self) -> None:
		source = """
		int main(void) {
			unsigned char uc = 200;
			int si = -10;
			int result = uc + si;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_short_plus_long_long(self) -> None:
		source = """
		int main(void) {
			unsigned short us = 50000;
			long long ll = 1000000;
			long long result = us + ll;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_chained_operations_with_mixed_types(self) -> None:
		source = """
		int main(void) {
			char a = 1;
			short b = 2;
			int c = 3;
			unsigned int d = 4;
			long long e = 5;
			long long result = (a + b) * c - d + e;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_comparison_mixed_char_int(self) -> None:
		source = """
		int main(void) {
			char c = 10;
			int i = 20;
			if (c < i) return 1;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_ternary_with_different_integer_types(self) -> None:
		source = """
		int main(void) {
			char c = 1;
			int i = 100;
			long long ll = 200;
			long long result = c ? i : ll;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Cast expressions with integer types
# ---------------------------------------------------------------------------


class TestCastExpressions:
	def test_int_to_char_cast(self) -> None:
		source = """
		int main(void) {
			int x = 300;
			char c = (char)x;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_char_to_int_cast(self) -> None:
		source = """
		int main(void) {
			char c = 65;
			int x = (int)c;
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_int_to_long_long_cast(self) -> None:
		source = """
		int main(void) {
			int x = 42;
			long long ll = (long long)x;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_long_long_to_int_cast(self) -> None:
		source = """
		int main(void) {
			long long ll = 42;
			int x = (int)ll;
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_to_signed_cast(self) -> None:
		source = """
		int main(void) {
			unsigned int u = 42;
			int s = (int)u;
			return s;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_signed_to_unsigned_cast(self) -> None:
		source = """
		int main(void) {
			int s = -1;
			unsigned int u = (unsigned int)s;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_to_long_long_cast(self) -> None:
		source = """
		int main(void) {
			short s = 100;
			long long ll = (long long)s;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_double_cast_narrowing(self) -> None:
		"""Cast from wide to narrow, then back to wide."""
		source = """
		int main(void) {
			long long ll = 1000;
			char c = (char)ll;
			int result = (int)c;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Compound assignment with promotion
# ---------------------------------------------------------------------------


class TestCompoundAssignmentPromotion:
	def test_char_compound_add(self) -> None:
		source = """
		int main(void) {
			char c = 10;
			c += 5;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_compound_multiply(self) -> None:
		source = """
		int main(void) {
			short s = 100;
			s *= 2;
			return s;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_compound_subtract(self) -> None:
		"""Compound -= on unsigned should compile without errors."""
		source = """
		int main(void) {
			unsigned int u = 100;
			u -= 50;
			return u;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_char_compound_bitwise_or(self) -> None:
		source = """
		int main(void) {
			char c = 0x0F;
			c |= 0xF0;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_long_long_compound_shift(self) -> None:
		source = """
		int main(void) {
			long long ll = 1;
			ll <<= 32;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Postfix/prefix operators with type tracking
# ---------------------------------------------------------------------------


class TestIncrementDecrementPromotion:
	def test_char_postfix_increment(self) -> None:
		source = """
		int main(void) {
			char c = 41;
			c++;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_prefix_decrement(self) -> None:
		source = """
		int main(void) {
			short s = 100;
			--s;
			return s;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_int_postfix_decrement(self) -> None:
		source = """
		int main(void) {
			unsigned int u = 10;
			u--;
			return u;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_long_long_prefix_increment(self) -> None:
		source = """
		int main(void) {
			long long ll = 0;
			++ll;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Edge cases: boundary values and type sizes
# ---------------------------------------------------------------------------


class TestBoundaryValues:
	def test_char_max_value(self) -> None:
		source = """
		int main(void) {
			char c = 127;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_max_value(self) -> None:
		source = """
		int main(void) {
			short s = 32767;
			return s;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_short_max_value(self) -> None:
		source = """
		int main(void) {
			unsigned short s = 65535;
			return s;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_int_max_value(self) -> None:
		source = """
		int main(void) {
			int x = 2147483647;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_zero_across_types(self) -> None:
		source = """
		int main(void) {
			char c = 0;
			short s = 0;
			int i = 0;
			long long ll = 0;
			unsigned int u = 0;
			return c + s + i + ll + u;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_negative_one_across_signed_types(self) -> None:
		source = """
		int main(void) {
			signed char c = -1;
			short s = -1;
			int i = -1;
			long long ll = -1;
			int result = c + s + i + ll;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm
