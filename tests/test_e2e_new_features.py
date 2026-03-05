"""End-to-end tests for newly added features: bitfields, goto, type modifiers, _Bool.

Tests exercise the full pipeline: parse -> semantic -> IR -> (optional optimize) -> codegen.
"""

from compiler.__main__ import compile_source
from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRAlloc,
	IRBinOp,
	IRJump,
	IRLabelInstr,
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


def _full_pipeline(source: str, optimize: bool = False) -> str:
	return compile_source(source, optimize=optimize)


# ---------------------------------------------------------------------------
# Bitfields
# ---------------------------------------------------------------------------

class TestBitfieldsE2E:
	"""End-to-end tests for struct bitfield declarations."""

	def test_bitfield_parse_single_member(self) -> None:
		prog = _parse("struct S { int flag : 1; };")
		decl = prog.declarations[0]
		assert decl.members[0].name == "flag"
		assert decl.members[0].bit_width == 1

	def test_bitfield_parse_multiple_members(self) -> None:
		prog = _parse("struct Packed { int a : 3; int b : 5; int c : 8; };")
		decl = prog.declarations[0]
		assert len(decl.members) == 3
		assert decl.members[0].bit_width == 3
		assert decl.members[1].bit_width == 5
		assert decl.members[2].bit_width == 8

	def test_bitfield_mixed_with_regular_member(self) -> None:
		prog = _parse("struct Mix { int x; int flag : 1; int y; };")
		decl = prog.declarations[0]
		assert decl.members[0].bit_width is None
		assert decl.members[1].bit_width == 1
		assert decl.members[2].bit_width is None

	def test_bitfield_unsigned_type(self) -> None:
		prog = _parse("struct U { unsigned int a : 4; };")
		decl = prog.declarations[0]
		assert decl.members[0].bit_width == 4
		assert decl.members[0].type_spec.signedness == "unsigned"

	def test_bitfield_signed_type(self) -> None:
		prog = _parse("struct S { signed int val : 7; };")
		decl = prog.declarations[0]
		assert decl.members[0].bit_width == 7
		assert decl.members[0].type_spec.signedness == "signed"

	def test_bitfield_in_function_compiles_to_asm(self) -> None:
		source = """
		struct Flags { int a : 3; int b : 5; };
		int main(void) {
			struct Flags f;
			return 0;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_bitfield_with_type_modifier(self) -> None:
		"""Bitfield with unsigned short type modifier."""
		prog = _parse("struct S { unsigned short bits : 4; };")
		decl = prog.declarations[0]
		member = decl.members[0]
		assert member.bit_width == 4
		assert member.type_spec.signedness == "unsigned"
		assert member.type_spec.width_modifier == "short"

	def test_bitfield_max_width_int(self) -> None:
		prog = _parse("struct S { int full : 32; };")
		decl = prog.declarations[0]
		assert decl.members[0].bit_width == 32

	def test_bitfield_width_one(self) -> None:
		prog = _parse("struct S { unsigned int flag : 1; };")
		decl = prog.declarations[0]
		assert decl.members[0].bit_width == 1


# ---------------------------------------------------------------------------
# Goto / Labels
# ---------------------------------------------------------------------------

class TestGotoE2E:
	"""End-to-end tests for goto statements and labels."""

	def test_forward_goto_generates_jmp(self) -> None:
		source = """
		int main(void) {
			goto end;
			int x = 99;
			end: return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_forward_goto_ir_has_jump_to_label(self) -> None:
		source = """
		int main(void) {
			goto target;
			int x = 5;
			target: return 0;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		jumps = [i for i in func.body if isinstance(i, IRJump)]
		labels = [i for i in func.body if isinstance(i, IRLabelInstr)]
		label_names = {lbl.name for lbl in labels}
		assert any(j.target in label_names for j in jumps)

	def test_backward_goto_creates_loop(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			start:
			i = i + 1;
			if (i < 10) goto start;
			return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm or "jl" in asm or "jg" in asm or "je" in asm

	def test_goto_with_multiple_labels(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			goto second;
			first: x = 1;
			goto done;
			second: x = 2;
			goto first;
			done: return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		jumps = [i for i in func.body if isinstance(i, IRJump)]
		assert len(jumps) >= 3

	def test_goto_interacting_with_while_loop(self) -> None:
		"""Goto that breaks out of a loop via label after the loop."""
		source = """
		int main(void) {
			int sum = 0;
			int i = 0;
			while (i < 100) {
				sum = sum + i;
				i = i + 1;
				if (sum > 50) goto bail;
			}
			return sum;
			bail: return -1;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "jmp" in asm

	def test_goto_interacting_with_for_loop(self) -> None:
		source = """
		int main(void) {
			int result = 0;
			for (int i = 0; i < 10; i = i + 1) {
				if (i == 5) goto found;
				result = result + 1;
			}
			return result;
			found: return 42;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm
		assert "ret" in asm

	def test_goto_full_pipeline_optimized(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			goto skip;
			x = 100;
			skip: return x;
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		assert "main:" in asm
		assert "ret" in asm

	def test_label_at_end_of_function(self) -> None:
		source = """
		int main(void) {
			goto end;
			end: return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "ret" in asm


# ---------------------------------------------------------------------------
# Type Modifiers (long, short, signed, unsigned)
# ---------------------------------------------------------------------------

class TestTypeModifiersE2E:
	"""End-to-end tests for long/short/signed/unsigned type modifiers."""

	def test_long_variable_allocates_8_bytes(self) -> None:
		source = "int main(void) { long x = 100; return x; }"
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 8 for a in allocs)

	def test_long_long_allocates_8_bytes(self) -> None:
		source = "int main(void) { long long x = 200; return x; }"
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 8 for a in allocs)

	def test_short_variable_allocates_2_bytes(self) -> None:
		source = "int main(void) { short x = 10; return x; }"
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 2 for a in allocs)

	def test_unsigned_int_propagates_to_ir(self) -> None:
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

	def test_signed_int_declaration(self) -> None:
		source = "int main(void) { signed int x = -42; return x; }"
		asm = _compile_to_asm(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_unsigned_long_allocates_8_bytes(self) -> None:
		source = "int main(void) { unsigned long x = 999; return x; }"
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 8 for a in allocs)

	def test_short_arithmetic_generates_asm(self) -> None:
		source = """
		int main(void) {
			short a = 10;
			short b = 20;
			short c = a + b;
			return c;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_long_arithmetic_generates_asm(self) -> None:
		source = """
		int main(void) {
			long a = 1000;
			long b = 2000;
			long c = a + b;
			return c;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm

	def test_unsigned_short_type_modifier(self) -> None:
		source = "int main(void) { unsigned short x = 50; return x; }"
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 2 for a in allocs)

	def test_type_modifiers_in_function_params(self) -> None:
		source = """
		int add(long a, short b) { return a + b; }
		int main(void) { return add(100, 20); }
		"""
		asm = _compile_to_asm(source)
		assert "add:" in asm
		assert "main:" in asm

	def test_mixed_type_arithmetic_compiles(self) -> None:
		"""Mixing short and long in expressions should compile."""
		source = """
		int main(void) {
			short s = 5;
			long l = 100;
			long result = s + l;
			return result;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_type_modifiers_optimized_pipeline(self) -> None:
		source = """
		int main(void) {
			unsigned int x = 10;
			unsigned int y = 20;
			unsigned int z = x + y;
			return z;
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		assert "main:" in asm
		assert "ret" in asm


# ---------------------------------------------------------------------------
# _Bool Type
# ---------------------------------------------------------------------------

class TestBoolE2E:
	"""End-to-end tests for _Bool type usage."""

	def test_bool_declaration_compiles(self) -> None:
		source = "int main(void) { _Bool b = 1; return b; }"
		asm = _compile_to_asm(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_bool_allocates_1_byte(self) -> None:
		source = "int main(void) { _Bool b = 1; return b; }"
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 1 for a in allocs)

	def test_bool_truncation_nonzero_normalizes(self) -> None:
		"""Assigning 42 to _Bool should generate != 0 normalization."""
		source = "int main(void) { _Bool b = 42; return b; }"
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "!="]
		assert len(binops) >= 1

	def test_bool_zero_assignment(self) -> None:
		source = "int main(void) { _Bool b = 0; return b; }"
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_in_if_condition(self) -> None:
		source = """
		int main(void) {
			_Bool flag = 1;
			if (flag) return 10;
			return 20;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm
		# Should have a conditional jump for the if
		has_cond = any(
			kw in asm for kw in ["je", "jne", "jz", "jnz", "cmpl", "cmpb", "cmpq", "testb", "testl"]
		)
		assert has_cond

	def test_bool_in_while_condition(self) -> None:
		source = """
		int main(void) {
			_Bool running = 1;
			int count = 0;
			while (running) {
				count = count + 1;
				if (count == 5) running = 0;
			}
			return count;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_bool_from_comparison(self) -> None:
		source = """
		int main(void) {
			int x = 10;
			_Bool is_positive = x > 0;
			return is_positive;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_logical_operations(self) -> None:
		source = """
		int main(void) {
			_Bool a = 1;
			_Bool b = 0;
			_Bool c = a && b;
			return c;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm

	def test_bool_negation(self) -> None:
		source = """
		int main(void) {
			_Bool b = 1;
			_Bool nb = !b;
			return nb;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_full_pipeline_optimized(self) -> None:
		source = """
		int main(void) {
			_Bool x = 1;
			_Bool y = 0;
			if (x) return 1;
			return 0;
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		assert "main:" in asm
		assert "ret" in asm


# ---------------------------------------------------------------------------
# Feature Combinations
# ---------------------------------------------------------------------------

class TestFeatureCombinations:
	"""Tests combining multiple new features together."""

	def test_bitfield_with_unsigned_type(self) -> None:
		"""Bitfields using unsigned int type modifier."""
		prog = _parse("""
		struct Flags {
			unsigned int read : 1;
			unsigned int write : 1;
			unsigned int exec : 1;
			unsigned int reserved : 5;
		};
		""")
		decl = prog.declarations[0]
		assert len(decl.members) == 4
		for m in decl.members:
			assert m.type_spec.signedness == "unsigned"
			assert m.bit_width is not None

	def test_goto_with_bool_condition(self) -> None:
		"""Using _Bool in a conditional that leads to a goto."""
		source = """
		int main(void) {
			_Bool done = 0;
			int val = 0;
			loop:
			val = val + 1;
			if (val >= 10) done = 1;
			if (!done) goto loop;
			return val;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "jmp" in asm

	def test_goto_in_nested_control_flow(self) -> None:
		"""Goto jumping out of nested if/while."""
		source = """
		int main(void) {
			int result = 0;
			int i = 0;
			while (i < 20) {
				if (i > 10) {
					if (i == 15) goto found;
				}
				result = result + 1;
				i = i + 1;
			}
			return result;
			found: return 42;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_long_variable_with_goto(self) -> None:
		"""Using long type with goto control flow."""
		source = """
		int main(void) {
			long sum = 0;
			long i = 0;
			start:
			sum = sum + i;
			i = i + 1;
			if (i < 5) goto start;
			return sum;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm

	def test_bool_with_type_modifier_arithmetic(self) -> None:
		"""_Bool result from unsigned comparison."""
		source = """
		int main(void) {
			unsigned int x = 100;
			unsigned int y = 200;
			_Bool less = x < y;
			if (less) return 1;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_struct_with_bitfield_and_regular_in_function(self) -> None:
		"""Struct with both bitfield and regular members used in a function."""
		source = """
		struct Config {
			int enabled : 1;
			int mode : 3;
			int value;
		};
		int main(void) {
			struct Config c;
			return 0;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_short_variable_with_bool_cast(self) -> None:
		"""Using short value to set a _Bool."""
		source = """
		int main(void) {
			short s = 42;
			_Bool b = s;
			return b;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_multiple_features_complex(self) -> None:
		"""Complex test combining goto, _Bool, long, and unsigned."""
		source = """
		int main(void) {
			unsigned long counter = 0;
			_Bool found = 0;
			search:
			counter = counter + 1;
			if (counter == 42) {
				found = 1;
				goto done;
			}
			if (counter < 100) goto search;
			done:
			if (found) return 1;
			return 0;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "jmp" in asm
		assert "ret" in asm

	def test_multiple_features_optimized(self) -> None:
		"""Full pipeline with optimization combining multiple features."""
		source = """
		int main(void) {
			short a = 10;
			long b = 20;
			_Bool bigger = b > a;
			if (bigger) goto yes;
			return 0;
			yes: return 1;
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		assert "main:" in asm
		assert "ret" in asm

	def test_unsigned_loop_counter_with_goto_exit(self) -> None:
		"""Unsigned loop counter with goto-based early exit."""
		source = """
		int main(void) {
			unsigned int i = 0;
			unsigned int sum = 0;
			loop:
			sum = sum + i;
			i = i + 1;
			if (i > 10) goto end;
			goto loop;
			end: return sum;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		jumps = [i for i in func.body if isinstance(i, IRJump)]
		assert len(jumps) >= 2
