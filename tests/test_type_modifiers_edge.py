"""Edge-case tests for type modifiers: long long, unsigned, short, signedness interactions."""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRAlloc,
	IRBinOp,
	IRType,
	ir_type_byte_width,
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
# long long
# ---------------------------------------------------------------------------


class TestLongLongArithmetic:
	def test_long_long_variable_decl(self) -> None:
		source = "int main(void) { long long x = 100; return x; }"
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 8 for a in allocs)

	def test_long_long_addition(self) -> None:
		source = """
		int main(void) {
			long long a = 10;
			long long b = 20;
			long long c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_long_long_large_constant(self) -> None:
		source = """
		int main(void) {
			long long x = 2147483648;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

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
		assert "main:" in asm
		assert "imulq" in asm or "imul" in asm

	def test_long_long_return(self) -> None:
		source = """
		long long get_val(void) {
			return 42;
		}
		int main(void) {
			long long x = get_val();
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "get_val:" in asm

	def test_long_long_ir_type_is_long(self) -> None:
		source = "int main(void) { long long x = 1; return x; }"
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		long_allocs = [a for a in allocs if a.size == 8]
		assert len(long_allocs) >= 1


# ---------------------------------------------------------------------------
# unsigned overflow semantics
# ---------------------------------------------------------------------------


class TestUnsignedOverflow:
	def test_unsigned_int_decl(self) -> None:
		source = "int main(void) { unsigned int x = 0; return x; }"
		ir = _compile_to_ir(source)
		assert ir.functions[0] is not None

	def test_unsigned_variable_generates_asm(self) -> None:
		source = """
		int main(void) {
			unsigned int x = 4294967295;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_addition(self) -> None:
		source = """
		int main(void) {
			unsigned int a = 10;
			unsigned int b = 20;
			unsigned int c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_propagates_in_ir(self) -> None:
		"""Operations on unsigned values should have is_unsigned set."""
		source = """
		int main(void) {
			unsigned int a = 5;
			unsigned int b = 3;
			unsigned int c = a + b;
			return c;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp)]
		assert any(getattr(b, "is_unsigned", False) for b in binops)

	def test_unsigned_long_long(self) -> None:
		source = """
		int main(void) {
			unsigned long long x = 1;
			return 0;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 8 for a in allocs)


# ---------------------------------------------------------------------------
# short truncation
# ---------------------------------------------------------------------------


class TestShortTruncation:
	def test_short_variable_decl(self) -> None:
		source = "int main(void) { short x = 1; return x; }"
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 2 for a in allocs)

	def test_short_addition(self) -> None:
		source = """
		int main(void) {
			short a = 10;
			short b = 20;
			short c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_ir_type(self) -> None:
		assert ir_type_byte_width(IRType.SHORT) == 2

	def test_unsigned_short(self) -> None:
		source = """
		int main(void) {
			unsigned short x = 65535;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_parameter(self) -> None:
		source = """
		int add_shorts(short a, short b) {
			return a + b;
		}
		int main(void) {
			return add_shorts(10, 20);
		}
		"""
		asm = _compile_to_asm(source)
		assert "add_shorts:" in asm

	def test_short_return_type(self) -> None:
		source = """
		short get_short(void) {
			return 42;
		}
		int main(void) {
			short x = get_short();
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "get_short:" in asm


# ---------------------------------------------------------------------------
# signed/unsigned comparison and mixed signedness
# ---------------------------------------------------------------------------


class TestSignedUnsignedComparison:
	def test_signed_int_comparison(self) -> None:
		source = """
		int main(void) {
			int a = -1;
			int b = 1;
			if (a < b) return 1;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_comparison(self) -> None:
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

	def test_mixed_signedness_expr(self) -> None:
		"""Mixing signed and unsigned in an expression should compile."""
		source = """
		int main(void) {
			int a = 5;
			unsigned int b = 10;
			int c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_subtraction(self) -> None:
		source = """
		int main(void) {
			unsigned int a = 10;
			unsigned int b = 3;
			unsigned int c = a - b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_signed_unsigned_assignment(self) -> None:
		source = """
		int main(void) {
			unsigned int u = 42;
			int s = u;
			return s;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Combined modifier tests
# ---------------------------------------------------------------------------


class TestCombinedModifiers:
	def test_signed_long_long(self) -> None:
		source = "int main(void) { signed long long x = -1; return 0; }"
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_short_arithmetic(self) -> None:
		source = """
		int main(void) {
			unsigned short a = 100;
			unsigned short b = 200;
			unsigned short c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_signed_char(self) -> None:
		source = "int main(void) { signed char c = -1; return c; }"
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_char(self) -> None:
		source = "int main(void) { unsigned char c = 255; return c; }"
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_multiple_type_vars_in_function(self) -> None:
		source = """
		int main(void) {
			short s = 1;
			long long ll = 2;
			unsigned int u = 3;
			int total = s + ll + u;
			return total;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm
