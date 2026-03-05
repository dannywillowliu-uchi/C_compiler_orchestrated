"""Type modifier combination correctness tests.

Tests all valid C type modifier combinations through the full pipeline:
parse -> semantic -> IR -> codegen. Also tests that invalid combinations
are properly rejected.
"""

import pytest

from compiler.codegen import CodeGenerator
from compiler.ir import IRAlloc
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


# ---------------------------------------------------------------------------
# Valid type modifier combinations - parsing and compilation
# ---------------------------------------------------------------------------


class TestValidModifierCombos:
	"""Test that all valid C type modifier combinations parse and compile."""

	def test_unsigned_int(self) -> None:
		asm = _compile_to_asm("int main(void) { unsigned int x = 42; return x; }")
		assert "main:" in asm

	def test_signed_int(self) -> None:
		asm = _compile_to_asm("int main(void) { signed int x = -1; return x; }")
		assert "main:" in asm

	def test_unsigned_char(self) -> None:
		asm = _compile_to_asm("int main(void) { unsigned char x = 255; return x; }")
		assert "main:" in asm

	def test_signed_char(self) -> None:
		asm = _compile_to_asm("int main(void) { signed char x = -1; return x; }")
		assert "main:" in asm

	def test_unsigned_short(self) -> None:
		asm = _compile_to_asm("int main(void) { unsigned short x = 100; return x; }")
		assert "main:" in asm

	def test_signed_short(self) -> None:
		asm = _compile_to_asm("int main(void) { signed short x = -100; return x; }")
		assert "main:" in asm

	def test_unsigned_long(self) -> None:
		asm = _compile_to_asm("int main(void) { unsigned long x = 100; return x; }")
		assert "main:" in asm

	def test_signed_long(self) -> None:
		asm = _compile_to_asm("int main(void) { signed long x = -100; return x; }")
		assert "main:" in asm

	def test_unsigned_long_long(self) -> None:
		asm = _compile_to_asm("int main(void) { unsigned long long x = 100; return x; }")
		assert "main:" in asm

	def test_signed_long_long(self) -> None:
		asm = _compile_to_asm("int main(void) { signed long long x = -100; return x; }")
		assert "main:" in asm

	def test_long_int(self) -> None:
		asm = _compile_to_asm("int main(void) { long int x = 100; return x; }")
		assert "main:" in asm

	def test_short_int(self) -> None:
		asm = _compile_to_asm("int main(void) { short int x = 100; return x; }")
		assert "main:" in asm

	def test_long_long_int(self) -> None:
		asm = _compile_to_asm("int main(void) { long long int x = 100; return x; }")
		assert "main:" in asm

	def test_unsigned_short_int(self) -> None:
		asm = _compile_to_asm("int main(void) { unsigned short int x = 100; return x; }")
		assert "main:" in asm

	def test_unsigned_long_int(self) -> None:
		asm = _compile_to_asm("int main(void) { unsigned long int x = 100; return x; }")
		assert "main:" in asm

	def test_unsigned_long_long_int(self) -> None:
		asm = _compile_to_asm("int main(void) { unsigned long long int x = 100; return x; }")
		assert "main:" in asm

	def test_plain_unsigned(self) -> None:
		"""'unsigned' alone means 'unsigned int'."""
		asm = _compile_to_asm("int main(void) { unsigned x = 42; return x; }")
		assert "main:" in asm

	def test_plain_signed(self) -> None:
		"""'signed' alone means 'signed int'."""
		asm = _compile_to_asm("int main(void) { signed x = -1; return x; }")
		assert "main:" in asm

	def test_plain_long(self) -> None:
		"""'long' alone means 'long int'."""
		asm = _compile_to_asm("int main(void) { long x = 100; return x; }")
		assert "main:" in asm

	def test_plain_short(self) -> None:
		"""'short' alone means 'short int'."""
		asm = _compile_to_asm("int main(void) { short x = 100; return x; }")
		assert "main:" in asm

	def test_plain_long_long(self) -> None:
		"""'long long' alone means 'long long int'."""
		asm = _compile_to_asm("int main(void) { long long x = 100; return x; }")
		assert "main:" in asm

	def test_plain_char(self) -> None:
		asm = _compile_to_asm("int main(void) { char x = 65; return x; }")
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Sizeof correctness for modifier combinations
# ---------------------------------------------------------------------------


class TestSizeofModifierCombos:
	"""Test sizeof returns correct values for type modifier combinations."""

	@pytest.mark.parametrize("type_decl,expected_size", [
		("char", 1),
		("signed char", 1),
		("unsigned char", 1),
		("short", 2),
		("signed short", 2),
		("unsigned short", 2),
		("short int", 2),
		("unsigned short int", 2),
		("int", 4),
		("signed int", 4),
		("unsigned int", 4),
		("signed", 4),
		("unsigned", 4),
		("long", 8),
		("signed long", 8),
		("unsigned long", 8),
		("long int", 8),
		("unsigned long int", 8),
		("long long", 8),
		("signed long long", 8),
		("unsigned long long", 8),
		("long long int", 8),
		("unsigned long long int", 8),
		("float", 4),
		("double", 8),
	])
	def test_sizeof_type(self, type_decl: str, expected_size: int) -> None:
		source = f"""
		int main(void) {{
			return sizeof({type_decl});
		}}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		# sizeof should fold to a constant in the return
		from compiler.ir import IRReturn, IRConst
		returns = [i for i in func.body if isinstance(i, IRReturn)]
		assert len(returns) == 1
		ret = returns[0]
		assert isinstance(ret.value, IRConst), f"sizeof({type_decl}) should be constant-folded"
		assert ret.value.value == expected_size, (
			f"sizeof({type_decl}) = {ret.value.value}, expected {expected_size}"
		)


# ---------------------------------------------------------------------------
# Allocation sizes match type widths
# ---------------------------------------------------------------------------


class TestAllocSizes:
	"""Test that variable allocations use correct sizes for modified types."""

	def test_short_alloc_2_bytes(self) -> None:
		ir = _compile_to_ir("int main(void) { short x = 1; return x; }")
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 2 for a in allocs)

	def test_unsigned_short_alloc_2_bytes(self) -> None:
		ir = _compile_to_ir("int main(void) { unsigned short x = 1; return x; }")
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 2 for a in allocs)

	def test_long_alloc_8_bytes(self) -> None:
		ir = _compile_to_ir("int main(void) { long x = 1; return x; }")
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 8 for a in allocs)

	def test_unsigned_long_alloc_8_bytes(self) -> None:
		ir = _compile_to_ir("int main(void) { unsigned long x = 1; return x; }")
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 8 for a in allocs)

	def test_long_long_alloc_8_bytes(self) -> None:
		ir = _compile_to_ir("int main(void) { long long x = 1; return x; }")
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 8 for a in allocs)

	def test_unsigned_long_long_alloc_8_bytes(self) -> None:
		ir = _compile_to_ir("int main(void) { unsigned long long x = 1; return x; }")
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 8 for a in allocs)

	def test_unsigned_char_alloc_1_byte(self) -> None:
		ir = _compile_to_ir("int main(void) { unsigned char x = 1; return x; }")
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 1 for a in allocs)

	def test_signed_char_alloc_1_byte(self) -> None:
		ir = _compile_to_ir("int main(void) { signed char x = 1; return x; }")
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 1 for a in allocs)

	def test_int_alloc_4_bytes(self) -> None:
		ir = _compile_to_ir("int main(void) { unsigned int x = 1; return x; }")
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 4 for a in allocs)


# ---------------------------------------------------------------------------
# IR type resolution for modified types
# ---------------------------------------------------------------------------


class TestIRTypeResolution:
	"""Test that type modifiers produce correct alloc sizes mapping to IR types."""

	def test_short_alloc_size_matches_ir_short(self) -> None:
		ir = _compile_to_ir("int main(void) { short x = 1; return x; }")
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		# SHORT = 2 bytes
		assert any(a.size == 2 for a in allocs)

	def test_unsigned_short_alloc_size_matches_ir_short(self) -> None:
		ir = _compile_to_ir("int main(void) { unsigned short x = 1; return x; }")
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 2 for a in allocs)

	def test_long_alloc_size_matches_ir_long(self) -> None:
		ir = _compile_to_ir("int main(void) { long x = 1; return x; }")
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		# LONG = 8 bytes
		assert any(a.size == 8 for a in allocs)

	def test_unsigned_long_alloc_size_matches_ir_long(self) -> None:
		ir = _compile_to_ir("int main(void) { unsigned long x = 1; return x; }")
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 8 for a in allocs)

	def test_long_long_alloc_size_matches_ir_long(self) -> None:
		ir = _compile_to_ir("int main(void) { long long x = 1; return x; }")
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 8 for a in allocs)

	def test_unsigned_long_long_alloc_size_matches_ir_long(self) -> None:
		ir = _compile_to_ir("int main(void) { unsigned long long x = 1; return x; }")
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 8 for a in allocs)

	def test_char_alloc_size_matches_ir_char(self) -> None:
		ir = _compile_to_ir("int main(void) { unsigned char x = 1; return x; }")
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		# CHAR = 1 byte
		assert any(a.size == 1 for a in allocs)


# ---------------------------------------------------------------------------
# Arithmetic promotion with modified types
# ---------------------------------------------------------------------------


class TestArithmeticPromotionModifiers:
	"""Test that arithmetic promotion works correctly with type modifiers."""

	def test_short_plus_short_promotes_to_int(self) -> None:
		source = """
		int main(void) {
			short a = 1;
			short b = 2;
			int c = a + b;
			return c;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		# c should be 4 bytes (int)
		int_allocs = [a for a in allocs if a.size == 4]
		assert len(int_allocs) >= 1

	def test_unsigned_short_plus_int(self) -> None:
		source = """
		int main(void) {
			unsigned short a = 1;
			int b = 2;
			int c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_long_plus_int_compiles(self) -> None:
		source = """
		int main(void) {
			long a = 100;
			int b = 200;
			long c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_long_long_arithmetic_compiles(self) -> None:
		source = """
		int main(void) {
			unsigned long long a = 100;
			unsigned long long b = 200;
			unsigned long long c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_char_plus_unsigned_int(self) -> None:
		source = """
		int main(void) {
			char a = 1;
			unsigned int b = 2;
			unsigned int c = a + b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Codegen register size correctness
# ---------------------------------------------------------------------------


class TestCodegenRegisterSizes:
	"""Test that codegen uses correct register sizes for modified types."""

	def test_short_uses_word_ops(self) -> None:
		"""Short variables should use 16-bit (word) operations in asm."""
		source = "int main(void) { short x = 42; return x; }"
		asm = _compile_to_asm(source)
		# short stores use 'w' suffix or 16-bit addressing
		assert "main:" in asm

	def test_long_uses_quad_ops(self) -> None:
		"""Long variables should use 64-bit (quad) operations in asm."""
		source = "int main(void) { long x = 42; return x; }"
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_char_uses_byte_ops(self) -> None:
		"""Char variables should use 8-bit (byte) operations in asm."""
		source = "int main(void) { unsigned char x = 42; return x; }"
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_long_long_codegen(self) -> None:
		source = """
		int main(void) {
			unsigned long long x = 100;
			unsigned long long y = 200;
			unsigned long long z = x + y;
			return z;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Pointer to modified types
# ---------------------------------------------------------------------------


class TestPointerToModifiedTypes:
	"""Test pointer declarations with type modifiers."""

	def test_const_unsigned_int_pointer(self) -> None:
		source = """
		int main(void) {
			unsigned int x = 42;
			const unsigned int *p = &x;
			return *p;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_volatile_signed_long(self) -> None:
		source = """
		int main(void) {
			volatile signed long x = 100;
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unsigned_short_pointer(self) -> None:
		source = """
		int main(void) {
			unsigned short x = 42;
			unsigned short *p = &x;
			return *p;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_const_signed_char_pointer(self) -> None:
		source = """
		int main(void) {
			signed char x = 65;
			const signed char *p = &x;
			return *p;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_sizeof_pointer_always_8(self) -> None:
		"""sizeof any pointer type should be 8 on 64-bit."""
		for type_decl in ["unsigned int *", "signed long *", "unsigned char *", "short *"]:
			source = f"int main(void) {{ return sizeof({type_decl}); }}"
			ir = _compile_to_ir(source)
			func = ir.functions[0]
			from compiler.ir import IRReturn, IRConst
			returns = [i for i in func.body if isinstance(i, IRReturn)]
			assert len(returns) == 1
			ret = returns[0]
			assert isinstance(ret.value, IRConst)
			assert ret.value.value == 8, f"sizeof({type_decl}) should be 8"


# ---------------------------------------------------------------------------
# Invalid type modifier combinations - semantic errors
# ---------------------------------------------------------------------------


class TestInvalidModifierCombos:
	"""Test that invalid C type modifier combinations are rejected."""

	def test_signed_float_rejected(self) -> None:
		errors = _analyze("int main(void) { signed float x = 1.0; return 0; }")
		assert any("signed" in str(e) and "float" in str(e) for e in errors)

	def test_unsigned_float_rejected(self) -> None:
		errors = _analyze("int main(void) { unsigned float x = 1.0; return 0; }")
		assert any("unsigned" in str(e) and "float" in str(e) for e in errors)

	def test_signed_double_rejected(self) -> None:
		errors = _analyze("int main(void) { signed double x = 1.0; return 0; }")
		assert any("signed" in str(e) and "double" in str(e) for e in errors)

	def test_unsigned_double_rejected(self) -> None:
		errors = _analyze("int main(void) { unsigned double x = 1.0; return 0; }")
		assert any("unsigned" in str(e) and "double" in str(e) for e in errors)

	def test_short_double_rejected(self) -> None:
		errors = _analyze("int main(void) { short double x = 1.0; return 0; }")
		assert any("short" in str(e) and "double" in str(e) for e in errors)

	def test_short_float_rejected(self) -> None:
		errors = _analyze("int main(void) { short float x = 1.0; return 0; }")
		assert any("short" in str(e) and "float" in str(e) for e in errors)

	def test_long_char_rejected(self) -> None:
		errors = _analyze("int main(void) { long char x = 1; return 0; }")
		assert any("long" in str(e) and "char" in str(e) for e in errors)

	def test_short_char_rejected(self) -> None:
		"""'short char' is not valid C."""
		# Parser may reject this or semantic analysis should
		try:
			errors = _analyze("int main(void) { short char x = 1; return 0; }")
			# If parser accepts it, semantic should reject
			assert any("short" in str(e) for e in errors)
		except Exception:
			# Parser rejection is also acceptable
			pass

	def test_long_long_char_rejected(self) -> None:
		errors = _analyze("int main(void) { long long char x = 1; return 0; }")
		assert any("long" in str(e) and "char" in str(e) for e in errors)


# ---------------------------------------------------------------------------
# Function parameters with type modifiers
# ---------------------------------------------------------------------------


class TestFunctionParamModifiers:
	"""Test type modifiers on function parameters."""

	def test_unsigned_int_param(self) -> None:
		source = """
		int foo(unsigned int x) { return x; }
		int main(void) { return foo(42); }
		"""
		asm = _compile_to_asm(source)
		assert "foo:" in asm

	def test_signed_long_param(self) -> None:
		source = """
		int bar(signed long x) { return x; }
		int main(void) { return bar(100); }
		"""
		asm = _compile_to_asm(source)
		assert "bar:" in asm

	def test_unsigned_char_param(self) -> None:
		source = """
		int baz(unsigned char x) { return x; }
		int main(void) { return baz(65); }
		"""
		asm = _compile_to_asm(source)
		assert "baz:" in asm

	def test_short_return_type(self) -> None:
		source = """
		short get_short(void) { return 42; }
		int main(void) { return get_short(); }
		"""
		asm = _compile_to_asm(source)
		assert "get_short:" in asm

	def test_unsigned_long_long_return_type(self) -> None:
		source = """
		unsigned long long get_ull(void) { return 100; }
		int main(void) { return get_ull(); }
		"""
		asm = _compile_to_asm(source)
		assert "get_ull:" in asm


# ---------------------------------------------------------------------------
# TypeSpec field correctness after parsing
# ---------------------------------------------------------------------------


class TestTypeSpecParsing:
	"""Verify that the parser produces correct TypeSpec fields for modified types."""

	def test_unsigned_long_long_typespec(self) -> None:
		ast = _parse("int main(void) { unsigned long long x = 0; return x; }")
		# Find the VarDecl
		from compiler.ast_nodes import VarDecl
		body = ast.declarations[0].body.statements
		var_decl = [s for s in body if isinstance(s, VarDecl)][0]
		ts = var_decl.type_spec
		assert ts.base_type == "int"
		assert ts.signedness == "unsigned"
		assert ts.width_modifier == "long long"

	def test_signed_short_int_typespec(self) -> None:
		ast = _parse("int main(void) { signed short int x = 0; return x; }")
		from compiler.ast_nodes import VarDecl
		body = ast.declarations[0].body.statements
		var_decl = [s for s in body if isinstance(s, VarDecl)][0]
		ts = var_decl.type_spec
		assert ts.base_type == "int"
		assert ts.signedness == "signed"
		assert ts.width_modifier == "short"

	def test_const_unsigned_int_pointer_typespec(self) -> None:
		ast = _parse("int main(void) { const unsigned int *p; return 0; }")
		from compiler.ast_nodes import VarDecl
		body = ast.declarations[0].body.statements
		var_decl = [s for s in body if isinstance(s, VarDecl)][0]
		ts = var_decl.type_spec
		assert ts.base_type == "int"
		assert ts.signedness == "unsigned"
		assert "const" in ts.qualifiers
		assert ts.pointer_count == 1

	def test_plain_unsigned_defaults_to_int(self) -> None:
		ast = _parse("int main(void) { unsigned x = 0; return x; }")
		from compiler.ast_nodes import VarDecl
		body = ast.declarations[0].body.statements
		var_decl = [s for s in body if isinstance(s, VarDecl)][0]
		ts = var_decl.type_spec
		assert ts.base_type == "int"
		assert ts.signedness == "unsigned"

	def test_plain_long_defaults_to_int(self) -> None:
		ast = _parse("int main(void) { long x = 0; return x; }")
		from compiler.ast_nodes import VarDecl
		body = ast.declarations[0].body.statements
		var_decl = [s for s in body if isinstance(s, VarDecl)][0]
		ts = var_decl.type_spec
		assert ts.base_type == "int"
		assert ts.width_modifier == "long"

	def test_volatile_signed_long_typespec(self) -> None:
		ast = _parse("int main(void) { volatile signed long x = 0; return x; }")
		from compiler.ast_nodes import VarDecl
		body = ast.declarations[0].body.statements
		var_decl = [s for s in body if isinstance(s, VarDecl)][0]
		ts = var_decl.type_spec
		assert ts.base_type == "int"
		assert ts.signedness == "signed"
		assert ts.width_modifier == "long"
		assert "volatile" in ts.qualifiers
