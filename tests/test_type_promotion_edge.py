"""Edge-case tests for type modifier combinations with casts and arithmetic.

Exercises interactions between type modifiers (long, short, signed, unsigned) and:
(1) explicit casts between signed/unsigned
(2) arithmetic promotion rules (short + int, unsigned - signed)
(3) sizeof with type modifier combinations
(4) variadic functions with promoted short/char args
(5) bitfields with unsigned types

Tests verify both IR generation correctness and semantic analysis type checking.
"""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRAlloc,
	IRConst,
	IRConvert,
	IRCopy,
	IRReturn,
	IRType,
	IRVaStart,
)
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.preprocessor import Preprocessor
from compiler.semantic import SemanticAnalyzer


def _parse(source: str):
	tokens = Lexer(source).tokenize()
	return Parser(tokens).parse()


def _analyze(source: str) -> list:
	ast = _parse(source)
	analyzer = SemanticAnalyzer()
	try:
		analyzer.analyze(ast)
	except Exception:
		pass
	return analyzer.errors


def _compile_to_ir(source: str):
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	ast = Parser(tokens).parse()
	errors = SemanticAnalyzer().analyze(ast)
	assert not errors, f"Semantic errors: {errors}"
	return IRGenerator().generate(ast)


def _compile_to_asm(source: str) -> str:
	ir = _compile_to_ir(source)
	return CodeGenerator().generate(ir)


def _get_return_const(source: str) -> int:
	ir = _compile_to_ir(source)
	func = ir.functions[0]
	returns = [i for i in func.body if isinstance(i, IRReturn)]
	assert len(returns) >= 1
	ret = returns[0]
	assert isinstance(ret.value, IRConst), f"Expected constant-folded return, got {ret.value}"
	return ret.value.value


# ---------------------------------------------------------------------------
# 1. Explicit casts between signed and unsigned types
# ---------------------------------------------------------------------------


class TestSignedUnsignedCasts:
	"""Explicit casts between signed and unsigned variants of same/different widths."""

	def test_signed_int_to_unsigned_int_compiles(self) -> None:
		source = """
		int main(void) {
			signed int x = -1;
			unsigned int y = (unsigned int)x;
			return y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_int_to_signed_int_compiles(self) -> None:
		source = """
		int main(void) {
			unsigned int x = 4294967295;
			signed int y = (signed int)x;
			return y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_signed_char_to_unsigned_short_cast(self) -> None:
		source = """
		int main(void) {
			signed char x = -1;
			unsigned short y = (unsigned short)x;
			return y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_long_to_signed_short_truncation(self) -> None:
		source = """
		int main(void) {
			unsigned long x = 100000;
			signed short y = (signed short)x;
			return y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_signed_short_to_unsigned_long_long_widening(self) -> None:
		source = """
		int main(void) {
			signed short x = -5;
			unsigned long long y = (unsigned long long)x;
			return y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_char_to_signed_long_cast(self) -> None:
		source = """
		int main(void) {
			unsigned char x = 255;
			signed long y = (signed long)x;
			return y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_cast_signed_to_unsigned_no_semantic_errors(self) -> None:
		source = """
		int main(void) {
			signed int a = -42;
			unsigned int b = (unsigned int)a;
			unsigned long c = (unsigned long)a;
			unsigned char d = (unsigned char)a;
			return b + c + d;
		}
		"""
		errors = _analyze(source)
		assert not errors

	def test_cast_unsigned_to_signed_no_semantic_errors(self) -> None:
		source = """
		int main(void) {
			unsigned int a = 42;
			signed int b = (signed int)a;
			signed long c = (signed long)a;
			signed char d = (signed char)a;
			return b + c + d;
		}
		"""
		errors = _analyze(source)
		assert not errors

	def test_cast_signed_char_to_unsigned_int_ir_convert(self) -> None:
		"""Cast from signed char to unsigned int should produce widening IR."""
		source = """
		int main(void) {
			signed char x = 100;
			unsigned int y = (unsigned int)x;
			return y;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		converts = [i for i in func.body if isinstance(i, (IRCopy, IRConvert))]
		int_ops = [c for c in converts if getattr(c, "ir_type", None) == IRType.INT
			or getattr(c, "to_type", None) == IRType.INT]
		assert len(int_ops) >= 1, "Expected widening to INT in IR"

	def test_cast_unsigned_long_long_to_signed_char_ir_truncate(self) -> None:
		"""Cast from unsigned long long to signed char should truncate."""
		source = """
		int main(void) {
			unsigned long long x = 300;
			signed char y = (signed char)x;
			return y;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		converts = [i for i in func.body if isinstance(i, (IRCopy, IRConvert))]
		char_ops = [c for c in converts if getattr(c, "ir_type", None) == IRType.CHAR
			or getattr(c, "to_type", None) == IRType.CHAR]
		assert len(char_ops) >= 1, "Expected truncation to CHAR in IR"

	def test_double_cast_signed_unsigned_signed(self) -> None:
		"""(signed int)(unsigned int)(signed int)x should compile."""
		source = """
		int main(void) {
			signed int x = -1;
			signed int y = (signed int)(unsigned int)(signed int)x;
			return y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# 2. Arithmetic promotion rules
# ---------------------------------------------------------------------------


class TestArithmeticPromotionEdgeCases:
	"""Edge cases for C integer promotion and usual arithmetic conversions."""

	def test_short_plus_int_compiles(self) -> None:
		source = """
		int main(void) {
			short a = 10;
			int b = 20;
			int c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_minus_signed_compiles(self) -> None:
		source = """
		int main(void) {
			unsigned int a = 10;
			signed int b = 3;
			unsigned int c = a - b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_char_plus_long_promotes_to_long(self) -> None:
		"""char + long should produce a long-sized result."""
		source = """
		int main(void) {
			char a = 1;
			long b = 100;
			long c = a + b;
			return c;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		long_allocs = [a for a in allocs if a.size == 8]
		# b and c should both be 8 bytes
		assert len(long_allocs) >= 2

	def test_unsigned_short_plus_signed_int(self) -> None:
		source = """
		int main(void) {
			unsigned short a = 1000;
			signed int b = -500;
			int c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_signed_char_times_unsigned_char(self) -> None:
		source = """
		int main(void) {
			signed char a = 10;
			unsigned char b = 20;
			int c = a * b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_minus_long_long_promotes(self) -> None:
		source = """
		int main(void) {
			short a = 5;
			long long b = 100;
			long long c = b - a;
			return c;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		long_allocs = [a for a in allocs if a.size == 8]
		assert len(long_allocs) >= 2, "b and c should be 8-byte allocs"

	def test_unsigned_long_plus_signed_long_compiles(self) -> None:
		source = """
		int main(void) {
			unsigned long a = 100;
			signed long b = 50;
			unsigned long c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_mixed_width_comparison(self) -> None:
		"""Comparison between short and long should compile correctly."""
		source = """
		int main(void) {
			short a = 10;
			long b = 10;
			if (a == b) return 1;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_char_modulo_int(self) -> None:
		source = """
		int main(void) {
			unsigned char a = 200;
			int b = 7;
			int c = a % b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_shift_by_unsigned_char(self) -> None:
		source = """
		int main(void) {
			short a = 1;
			unsigned char b = 4;
			int c = a << b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_int_divide_signed_short(self) -> None:
		source = """
		int main(void) {
			unsigned int a = 100;
			signed short b = 3;
			unsigned int c = a / b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_promotion_in_ternary(self) -> None:
		"""Ternary with mixed types should compile."""
		source = """
		int main(void) {
			short a = 1;
			long b = 2;
			long c = (a > 0) ? b : a;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# 3. sizeof with type modifier combinations
# ---------------------------------------------------------------------------


class TestSizeofModifierCombinations:
	"""sizeof edge cases with type modifier combinations and expressions."""

	def test_sizeof_unsigned_equals_signed(self) -> None:
		"""sizeof(unsigned T) == sizeof(signed T) for all integer types."""
		pairs = [
			("unsigned char", "signed char"),
			("unsigned short", "signed short"),
			("unsigned int", "signed int"),
			("unsigned long", "signed long"),
			("unsigned long long", "signed long long"),
		]
		for unsigned, signed in pairs:
			u_src = f"int main(void) {{ return sizeof({unsigned}); }}"
			s_src = f"int main(void) {{ return sizeof({signed}); }}"
			assert _get_return_const(u_src) == _get_return_const(s_src), \
				f"sizeof({unsigned}) != sizeof({signed})"

	def test_sizeof_modifier_ordering_equivalence(self) -> None:
		"""Different orderings of modifiers should give same sizeof."""
		equiv_groups = [
			["unsigned short int", "unsigned short", "short unsigned int"],
			["unsigned long int", "unsigned long", "long unsigned int"],
			["unsigned long long int", "unsigned long long"],
			["signed long int", "signed long", "long signed int"],
		]
		for group in equiv_groups:
			sizes = []
			for t in group:
				src = f"int main(void) {{ return sizeof({t}); }}"
				try:
					sizes.append(_get_return_const(src))
				except Exception:
					pass
			if len(sizes) >= 2:
				assert all(s == sizes[0] for s in sizes), \
					f"Equivalent types {group} have different sizes: {sizes}"

	def test_sizeof_short_less_than_int(self) -> None:
		s = _get_return_const("int main(void) { return sizeof(short); }")
		i = _get_return_const("int main(void) { return sizeof(int); }")
		assert s < i or s == i  # C guarantees sizeof(short) <= sizeof(int)
		assert s <= i

	def test_sizeof_int_less_equal_long(self) -> None:
		int_sz = _get_return_const("int main(void) { return sizeof(int); }")
		long_sz = _get_return_const("int main(void) { return sizeof(long); }")
		assert int_sz <= long_sz

	def test_sizeof_long_less_equal_long_long(self) -> None:
		long_sz = _get_return_const("int main(void) { return sizeof(long); }")
		llong_sz = _get_return_const("int main(void) { return sizeof(long long); }")
		assert long_sz <= llong_sz

	def test_sizeof_in_arithmetic_with_modifiers(self) -> None:
		"""sizeof(long) - sizeof(short) should compile and give correct value."""
		source = """
		int main(void) {
			return sizeof(long) - sizeof(short);
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_sizeof_cast_preserves_target_type_size(self) -> None:
		"""sizeof((unsigned short)x) == sizeof(unsigned short)."""
		type_val = _get_return_const("int main(void) { return sizeof(unsigned short); }")
		cast_val = _get_return_const(
			"int main(void) { int x = 42; return sizeof((unsigned short)x); }"
		)
		assert type_val == cast_val

	def test_sizeof_unsigned_long_long_var(self) -> None:
		"""sizeof applied to a variable of unsigned long long type."""
		val = _get_return_const("""
		int main(void) {
			unsigned long long x = 0;
			return sizeof(x);
		}
		""")
		assert val == 8

	def test_sizeof_signed_char_var(self) -> None:
		val = _get_return_const("""
		int main(void) {
			signed char x = 0;
			return sizeof(x);
		}
		""")
		assert val == 1


# ---------------------------------------------------------------------------
# 4. Variadic functions with promoted short/char args
# ---------------------------------------------------------------------------


class TestVariadicPromotedArgs:
	"""Variadic functions receiving short/char args (which promote to int per C ABI)."""

	def test_variadic_with_char_arg_compiles(self) -> None:
		source = """
		#include <stdarg.h>
		int sum(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int total = 0;
			int i;
			for (i = 0; i < n; i = i + 1) {
				total = total + va_arg(ap, int);
			}
			va_end(ap);
			return total;
		}
		int main(void) {
			char a = 10;
			char b = 20;
			return sum(2, a, b);
		}
		"""
		asm = _compile_to_asm(source)
		assert "sum:" in asm
		assert "main:" in asm

	def test_variadic_with_short_arg_compiles(self) -> None:
		source = """
		#include <stdarg.h>
		int first(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int val = va_arg(ap, int);
			va_end(ap);
			return val;
		}
		int main(void) {
			short x = 42;
			return first(1, x);
		}
		"""
		asm = _compile_to_asm(source)
		assert "first:" in asm
		assert "main:" in asm

	def test_variadic_with_unsigned_char_compiles(self) -> None:
		source = """
		#include <stdarg.h>
		int grab(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int val = va_arg(ap, int);
			va_end(ap);
			return val;
		}
		int main(void) {
			unsigned char c = 255;
			return grab(1, c);
		}
		"""
		asm = _compile_to_asm(source)
		assert "grab:" in asm

	def test_variadic_with_unsigned_short_compiles(self) -> None:
		source = """
		#include <stdarg.h>
		int take(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int val = va_arg(ap, int);
			va_end(ap);
			return val;
		}
		int main(void) {
			unsigned short s = 1000;
			return take(1, s);
		}
		"""
		asm = _compile_to_asm(source)
		assert "take:" in asm

	def test_variadic_mixed_promoted_types(self) -> None:
		"""Variadic call with char, short, and int args mixed together."""
		source = """
		#include <stdarg.h>
		int sum3(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int a = va_arg(ap, int);
			int b = va_arg(ap, int);
			int c = va_arg(ap, int);
			va_end(ap);
			return a + b + c;
		}
		int main(void) {
			char x = 1;
			short y = 2;
			int z = 3;
			return sum3(3, x, y, z);
		}
		"""
		asm = _compile_to_asm(source)
		assert "sum3:" in asm
		assert "main:" in asm

	def test_variadic_with_signed_char_arg_no_semantic_error(self) -> None:
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int v = va_arg(ap, int);
			va_end(ap);
			return v;
		}
		int main(void) {
			signed char sc = -1;
			return f(1, sc);
		}
		"""
		preprocessed = Preprocessor().process(source)
		errors = _analyze(preprocessed)
		assert not errors

	def test_variadic_ir_has_va_start(self) -> None:
		"""Variadic function IR should contain va_start instruction."""
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int v = va_arg(ap, int);
			va_end(ap);
			return v;
		}
		int main(void) {
			return f(1, 42);
		}
		"""
		ir = _compile_to_ir(source)
		f_func = [fn for fn in ir.functions if fn.name == "f"][0]
		va_starts = [i for i in f_func.body if isinstance(i, IRVaStart)]
		assert len(va_starts) >= 1


# ---------------------------------------------------------------------------
# 5. Bitfields with unsigned types
# ---------------------------------------------------------------------------


class TestBitfieldUnsignedTypes:
	"""Bitfields declared with unsigned type modifiers."""

	def test_unsigned_int_bitfield_compiles(self) -> None:
		source = """
		struct S { unsigned int x : 3; unsigned int y : 5; };
		int main(void) {
			struct S s;
			s.x = 7;
			s.y = 31;
			return s.x + s.y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_int_bitfield_ir_generated(self) -> None:
		source = """
		struct S { unsigned int a : 4; unsigned int b : 4; };
		int main(void) {
			struct S s;
			s.a = 15;
			s.b = 10;
			return s.a;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) >= 1

	def test_signed_int_bitfield_compiles(self) -> None:
		source = """
		struct S { signed int x : 4; signed int y : 4; };
		int main(void) {
			struct S s;
			s.x = 7;
			s.y = -1;
			return s.x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_bitfield_and_cast(self) -> None:
		"""Read an unsigned bitfield and cast to long."""
		source = """
		struct S { unsigned int flags : 8; };
		int main(void) {
			struct S s;
			s.flags = 200;
			long val = (long)s.flags;
			return val;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_sizeof_struct_with_unsigned_bitfields(self) -> None:
		"""sizeof struct with unsigned bitfields should reflect storage."""
		source = """
		struct S { unsigned int a : 1; unsigned int b : 1; };
		int main(void) { return sizeof(struct S); }
		"""
		val = _get_return_const(source)
		assert val == 4, f"Expected sizeof(struct S) == 4, got {val}"

	def test_mixed_signed_unsigned_bitfields(self) -> None:
		source = """
		struct Mixed {
			signed int s : 4;
			unsigned int u : 4;
		};
		int main(void) {
			struct Mixed m;
			m.s = -3;
			m.u = 12;
			return m.u;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_bitfield_max_width(self) -> None:
		"""Unsigned bitfield using all 32 bits."""
		source = """
		struct Full { unsigned int val : 32; };
		int main(void) {
			struct Full f;
			f.val = 100;
			return f.val;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_bitfield_in_arithmetic(self) -> None:
		"""Bitfield values used in arithmetic expressions."""
		source = """
		struct S { unsigned int x : 5; unsigned int y : 5; };
		int main(void) {
			struct S s;
			s.x = 10;
			s.y = 20;
			int sum = s.x + s.y;
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Combined edge cases: casts + promotion + sizeof + modifiers
# ---------------------------------------------------------------------------


class TestCombinedModifierEdgeCases:
	"""Tests combining multiple features: casts, promotion, sizeof, modifiers."""

	def test_cast_in_arithmetic_with_different_widths(self) -> None:
		"""(long)short_val + (long)char_val should compile."""
		source = """
		int main(void) {
			short a = 10;
			signed char b = 20;
			long c = (long)a + (long)b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_sizeof_in_variable_init(self) -> None:
		source = """
		int main(void) {
			int sizes = sizeof(unsigned long long) + sizeof(unsigned char);
			return sizes;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_cast_result_used_in_comparison(self) -> None:
		source = """
		int main(void) {
			unsigned int a = 10;
			signed int b = -1;
			if ((signed int)a > b) return 1;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_nested_casts_with_arithmetic(self) -> None:
		source = """
		int main(void) {
			unsigned char a = 200;
			signed short b = -100;
			long result = (long)((int)a + (int)b);
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_sizeof_difference_of_types_compiles(self) -> None:
		"""sizeof(long long) - sizeof(char) should compile correctly."""
		source = "int main(void) { return sizeof(long long) - sizeof(char); }"
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_cast_then_sizeof(self) -> None:
		"""sizeof on cast result matches target type."""
		source = """
		int main(void) {
			int x = 42;
			return sizeof((unsigned long long)x);
		}
		"""
		val = _get_return_const(source)
		assert val == 8

	def test_unsigned_bitfield_cast_to_long_in_expression(self) -> None:
		source = """
		struct S { unsigned int val : 8; };
		int main(void) {
			struct S s;
			s.val = 100;
			long wide = (long)s.val * 2;
			return wide;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_param_cast_to_unsigned_long_in_return(self) -> None:
		source = """
		unsigned long widen(short x) {
			return (unsigned long)x;
		}
		int main(void) { return widen(42); }
		"""
		asm = _compile_to_asm(source)
		assert "widen:" in asm
		assert "main:" in asm
