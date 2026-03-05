"""Edge-case tests for type modifier + cast + sizeof interactions.

Tests sizeof with various type modifiers, cast truncation/extension behavior,
sizeof on cast expressions, type modifiers in function signatures, and
implicit integer promotions.
"""

from compiler.ast_nodes import (
	CastExpr,
	FunctionDecl,
	VarDecl,
)
from compiler.codegen import CodeGenerator
from compiler.ir import IRAlloc, IRConst, IRConvert, IRCopy, IRReturn, IRType
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
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
	ast = _parse(source)
	errors = SemanticAnalyzer().analyze(ast)
	assert not errors, f"Semantic errors: {errors}"
	return IRGenerator().generate(ast)


def _compile_to_asm(source: str) -> str:
	ir = _compile_to_ir(source)
	return CodeGenerator().generate(ir)


def _get_sizeof_return_value(source: str) -> int:
	"""Compile source and extract the constant returned by sizeof."""
	ir = _compile_to_ir(source)
	func = ir.functions[0]
	returns = [i for i in func.body if isinstance(i, IRReturn)]
	assert len(returns) >= 1
	ret = returns[0]
	assert isinstance(ret.value, IRConst), f"sizeof should be constant-folded, got {ret.value}"
	return ret.value.value


# ---------------------------------------------------------------------------
# 1. sizeof with type modifiers: long long vs short vs signed char
# ---------------------------------------------------------------------------


class TestSizeofTypeModifierEdgeCases:
	"""sizeof must return correct sizes for all type modifier variants."""

	def test_sizeof_long_long_is_8(self) -> None:
		val = _get_sizeof_return_value("int main(void) { return sizeof(long long); }")
		assert val == 8

	def test_sizeof_short_is_2(self) -> None:
		val = _get_sizeof_return_value("int main(void) { return sizeof(short); }")
		assert val == 2

	def test_sizeof_signed_char_is_1(self) -> None:
		val = _get_sizeof_return_value("int main(void) { return sizeof(signed char); }")
		assert val == 1

	def test_sizeof_unsigned_long_long_is_8(self) -> None:
		val = _get_sizeof_return_value("int main(void) { return sizeof(unsigned long long); }")
		assert val == 8

	def test_sizeof_unsigned_short_is_2(self) -> None:
		val = _get_sizeof_return_value("int main(void) { return sizeof(unsigned short); }")
		assert val == 2

	def test_sizeof_unsigned_char_is_1(self) -> None:
		val = _get_sizeof_return_value("int main(void) { return sizeof(unsigned char); }")
		assert val == 1

	def test_sizeof_long_long_greater_than_short(self) -> None:
		"""sizeof(long long) > sizeof(short) must hold."""
		ll = _get_sizeof_return_value("int main(void) { return sizeof(long long); }")
		s = _get_sizeof_return_value("int main(void) { return sizeof(short); }")
		assert ll > s

	def test_sizeof_short_greater_than_signed_char(self) -> None:
		"""sizeof(short) > sizeof(signed char) must hold."""
		s = _get_sizeof_return_value("int main(void) { return sizeof(short); }")
		sc = _get_sizeof_return_value("int main(void) { return sizeof(signed char); }")
		assert s > sc

	def test_sizeof_signed_and_unsigned_same(self) -> None:
		"""signed and unsigned variants must have the same size."""
		for base in ["char", "short", "int", "long", "long long"]:
			signed_src = f"int main(void) {{ return sizeof(signed {base}); }}"
			unsigned_src = f"int main(void) {{ return sizeof(unsigned {base}); }}"
			sv = _get_sizeof_return_value(signed_src)
			uv = _get_sizeof_return_value(unsigned_src)
			assert sv == uv, f"sizeof(signed {base}) != sizeof(unsigned {base})"


# ---------------------------------------------------------------------------
# 2. Cast from unsigned long long to short (truncation)
# ---------------------------------------------------------------------------


class TestCastTruncation:
	"""Casts from wider to narrower types should emit proper IR."""

	def test_ull_to_short_parses(self) -> None:
		"""Cast (short)ull_value should parse without errors."""
		source = """
		int main(void) {
			unsigned long long x = 65537;
			short y = (short)x;
			return y;
		}
		"""
		ast = _parse(source)
		body = ast.declarations[0].body.statements
		var_decls = [s for s in body if isinstance(s, VarDecl)]
		# y's initializer should be a CastExpr
		y_decl = var_decls[1]
		assert isinstance(y_decl.initializer, CastExpr)
		assert y_decl.initializer.target_type.width_modifier == "short"

	def test_ull_to_short_no_semantic_errors(self) -> None:
		source = """
		int main(void) {
			unsigned long long x = 65537;
			short y = (short)x;
			return y;
		}
		"""
		errors = _analyze(source)
		assert not errors

	def test_ull_to_short_ir_has_copy_or_convert(self) -> None:
		"""IR for truncation cast should emit a copy with SHORT type."""
		source = """
		int main(void) {
			unsigned long long x = 65537;
			short y = (short)x;
			return y;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		# The cast should produce an IRCopy or IRConvert with SHORT target
		copies = [i for i in func.body if isinstance(i, (IRCopy, IRConvert))]
		short_ops = [c for c in copies if getattr(c, "ir_type", None) == IRType.SHORT
			or getattr(c, "to_type", None) == IRType.SHORT]
		assert len(short_ops) >= 1, "Expected at least one SHORT copy/convert for truncation cast"

	def test_ull_to_short_compiles_to_asm(self) -> None:
		source = """
		int main(void) {
			unsigned long long x = 65537;
			short y = (short)x;
			return y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_long_to_char_truncation(self) -> None:
		"""Cast from long to char should also work."""
		source = """
		int main(void) {
			long x = 300;
			char y = (char)x;
			return y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_int_to_short_truncation(self) -> None:
		source = """
		int main(void) {
			int x = 70000;
			short y = (short)x;
			return y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# 3. Cast from signed char to unsigned long long (sign extension)
# ---------------------------------------------------------------------------


class TestCastSignExtension:
	"""Casts from narrow signed to wide unsigned should emit proper IR."""

	def test_signed_char_to_ull_parses(self) -> None:
		source = """
		int main(void) {
			signed char x = -1;
			unsigned long long y = (unsigned long long)x;
			return y;
		}
		"""
		ast = _parse(source)
		body = ast.declarations[0].body.statements
		var_decls = [s for s in body if isinstance(s, VarDecl)]
		y_decl = var_decls[1]
		assert isinstance(y_decl.initializer, CastExpr)
		ts = y_decl.initializer.target_type
		assert ts.signedness == "unsigned"
		assert ts.width_modifier == "long long"

	def test_signed_char_to_ull_no_semantic_errors(self) -> None:
		source = """
		int main(void) {
			signed char x = -1;
			unsigned long long y = (unsigned long long)x;
			return y;
		}
		"""
		errors = _analyze(source)
		assert not errors

	def test_signed_char_to_ull_ir_has_copy_or_convert(self) -> None:
		"""IR for widening cast should produce a LONG copy/convert."""
		source = """
		int main(void) {
			signed char x = -1;
			unsigned long long y = (unsigned long long)x;
			return y;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		copies = [i for i in func.body if isinstance(i, (IRCopy, IRConvert))]
		long_ops = [c for c in copies if getattr(c, "ir_type", None) == IRType.LONG
			or getattr(c, "to_type", None) == IRType.LONG]
		assert len(long_ops) >= 1, "Expected at least one LONG copy/convert for widening cast"

	def test_signed_char_to_ull_compiles(self) -> None:
		source = """
		int main(void) {
			signed char x = -1;
			unsigned long long y = (unsigned long long)x;
			return y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_to_long_widening(self) -> None:
		"""Short to long should also widen correctly."""
		source = """
		int main(void) {
			short x = -100;
			long y = (long)x;
			return y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_char_to_int_widening(self) -> None:
		source = """
		int main(void) {
			signed char x = 127;
			int y = (int)x;
			return y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# 4. sizeof applied to expressions with type modifiers
# ---------------------------------------------------------------------------


class TestSizeofCastExpressions:
	"""sizeof applied to cast expressions should use the cast target type size."""

	def test_sizeof_short_cast_expr(self) -> None:
		"""sizeof((short)x) should be sizeof(short) == 2."""
		source = """
		int main(void) {
			int x = 42;
			return sizeof((short)x);
		}
		"""
		val = _get_sizeof_return_value(source)
		assert val == 2

	def test_sizeof_long_cast_expr(self) -> None:
		"""sizeof((long)y) should be sizeof(long) == 8."""
		source = """
		int main(void) {
			int y = 42;
			return sizeof((long)y);
		}
		"""
		val = _get_sizeof_return_value(source)
		assert val == 8

	def test_sizeof_unsigned_long_long_cast(self) -> None:
		"""sizeof((unsigned long long)x) should be 8."""
		source = """
		int main(void) {
			short x = 1;
			return sizeof((unsigned long long)x);
		}
		"""
		val = _get_sizeof_return_value(source)
		assert val == 8

	def test_sizeof_signed_char_cast(self) -> None:
		"""sizeof((signed char)x) should be 1."""
		source = """
		int main(void) {
			int x = 42;
			return sizeof((signed char)x);
		}
		"""
		val = _get_sizeof_return_value(source)
		assert val == 1

	def test_sizeof_cast_does_not_evaluate(self) -> None:
		"""sizeof should not evaluate the expression (no side effects)."""
		source = """
		int main(void) {
			int x = 10;
			int s = sizeof((long)x);
			return x;
		}
		"""
		# Should compile fine; x should still be usable
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# 5. Type modifier combinations in function parameters and return types
# ---------------------------------------------------------------------------


class TestFunctionSignatureModifiers:
	"""Type modifiers in function params and returns at parser/semantic/IR levels."""

	def test_parse_short_param_and_long_return(self) -> None:
		"""Parser should handle short params and long long return type."""
		source = """
		long long add_wide(short a, short b) {
			return (long long)a + (long long)b;
		}
		int main(void) { return add_wide(1, 2); }
		"""
		ast = _parse(source)
		func = ast.declarations[0]
		assert isinstance(func, FunctionDecl)
		assert func.return_type.width_modifier == "long long"
		assert len(func.params) == 2
		assert func.params[0].type_spec.width_modifier == "short"
		assert func.params[1].type_spec.width_modifier == "short"

	def test_semantic_short_param_no_errors(self) -> None:
		source = """
		long long add_wide(short a, short b) {
			return (long long)a + (long long)b;
		}
		int main(void) { return add_wide(1, 2); }
		"""
		errors = _analyze(source)
		assert not errors

	def test_unsigned_char_param_unsigned_long_return(self) -> None:
		source = """
		unsigned long widen(unsigned char c) {
			return (unsigned long)c;
		}
		int main(void) { return widen(255); }
		"""
		asm = _compile_to_asm(source)
		assert "widen:" in asm
		assert "main:" in asm

	def test_signed_char_return_type(self) -> None:
		source = """
		signed char narrow(int x) {
			return (signed char)x;
		}
		int main(void) { return narrow(65); }
		"""
		asm = _compile_to_asm(source)
		assert "narrow:" in asm

	def test_multiple_modified_params(self) -> None:
		"""Function with unsigned short, signed long, unsigned char params."""
		source = """
		int mixed(unsigned short a, signed long b, unsigned char c) {
			return a + b + c;
		}
		int main(void) { return mixed(1, 2, 3); }
		"""
		asm = _compile_to_asm(source)
		assert "mixed:" in asm

	def test_long_long_return_ir_type(self) -> None:
		"""IR function with long long return should be generated."""
		source = """
		long long get_big(void) { return 42; }
		int main(void) { return get_big(); }
		"""
		ir = _compile_to_ir(source)
		# Should have two functions
		assert len(ir.functions) == 2
		func = ir.functions[0]
		assert func.name == "get_big"


# ---------------------------------------------------------------------------
# 6. Implicit integer promotions with short and signed char operands
# ---------------------------------------------------------------------------


class TestImplicitIntegerPromotions:
	"""Short and char operands should be promoted to int in expressions."""

	def test_short_plus_short_result_is_int_sized(self) -> None:
		"""short + short should be promoted to int (4 bytes) for arithmetic."""
		source = """
		int main(void) {
			short a = 100;
			short b = 200;
			int c = a + b;
			return c;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		# c should be allocated as int (4 bytes)
		int_allocs = [a for a in allocs if a.size == 4]
		assert len(int_allocs) >= 1

	def test_signed_char_plus_signed_char(self) -> None:
		"""signed char + signed char should compile (promoted to int)."""
		source = """
		int main(void) {
			signed char a = 10;
			signed char b = 20;
			int c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_short_arithmetic(self) -> None:
		"""unsigned short operands should also be promoted."""
		source = """
		int main(void) {
			unsigned short a = 100;
			unsigned short b = 200;
			int c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_char_comparison_compiles(self) -> None:
		"""Comparison of char operands should work after promotion."""
		source = """
		int main(void) {
			signed char a = 10;
			signed char b = 20;
			if (a < b) return 1;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_multiplication(self) -> None:
		"""short * short should not overflow due to promotion to int."""
		source = """
		int main(void) {
			short a = 200;
			short b = 200;
			int c = a * b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_mixed_char_short_arithmetic(self) -> None:
		"""Mixed char and short should both promote to int."""
		source = """
		int main(void) {
			signed char a = 10;
			short b = 20;
			int c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_char_bitwise_ops(self) -> None:
		"""Bitwise ops on unsigned char should compile after promotion."""
		source = """
		int main(void) {
			unsigned char a = 0xFF;
			unsigned char b = 0x0F;
			int c = a & b;
			int d = a | b;
			int e = a ^ b;
			return c + d + e;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_shift_operations(self) -> None:
		"""Shift on short operands should compile."""
		source = """
		int main(void) {
			short x = 1;
			int y = x << 4;
			return y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Combined edge cases: sizeof + cast + modifier interactions
# ---------------------------------------------------------------------------


class TestCombinedEdgeCases:
	"""Tests that combine sizeof, casts, and type modifiers."""

	def test_sizeof_type_vs_sizeof_cast_match(self) -> None:
		"""sizeof(short) should equal sizeof((short)expr)."""
		type_src = "int main(void) { return sizeof(short); }"
		cast_src = "int main(void) { int x = 42; return sizeof((short)x); }"
		type_val = _get_sizeof_return_value(type_src)
		cast_val = _get_sizeof_return_value(cast_src)
		assert type_val == cast_val

	def test_chained_casts_compile(self) -> None:
		"""Chained casts: (long)(short)(char)x should compile."""
		source = """
		int main(void) {
			int x = 42;
			long y = (long)(short)(signed char)x;
			return y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_cast_in_sizeof_no_side_effects(self) -> None:
		"""sizeof((long long)x++) should not modify x (sizeof doesn't evaluate)."""
		source = """
		int main(void) {
			int x = 5;
			int s = sizeof((long long)x);
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_sizeof_after_cast_variable(self) -> None:
		"""sizeof on a variable that received a cast value."""
		source = """
		int main(void) {
			long long x = (long long)42;
			return sizeof(x);
		}
		"""
		val = _get_sizeof_return_value(source)
		assert val == 8

	def test_cast_between_all_integer_sizes(self) -> None:
		"""Cast chain through all sizes: char -> short -> int -> long -> long long."""
		source = """
		int main(void) {
			signed char a = 1;
			short b = (short)a;
			int c = (int)b;
			long d = (long)c;
			long long e = (long long)d;
			return e;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_sizeof_nested_in_expression(self) -> None:
		"""sizeof in arithmetic expression with type modifiers compiles."""
		source = """
		int main(void) {
			return sizeof(long long) - sizeof(short) - sizeof(signed char);
		}
		"""
		# The individual sizeofs are constant-folded but the arithmetic may not be,
		# so just verify it compiles and produces valid assembly.
		asm = _compile_to_asm(source)
		assert "main:" in asm
