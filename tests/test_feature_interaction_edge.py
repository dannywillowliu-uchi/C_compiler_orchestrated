"""Edge-case tests for feature interactions: bitfields + type modifiers + goto.

Covers parsing, semantic analysis, and IR generation for combinations of
recently added features that might interact in unexpected ways.
"""

import pytest

from compiler.__main__ import compile_source
from compiler.ast_nodes import StructMember, TypeSpec
from compiler.codegen import CodeGenerator
from compiler.ir import IRJump
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer, SemanticError


def _parse(source: str):
	tokens = Lexer(source).tokenize()
	return Parser(tokens).parse()


def _analyze(source: str):
	ast = _parse(source)
	return SemanticAnalyzer().analyze(ast), ast


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
# Bitfields with signed/unsigned type modifiers (parsing)
# ---------------------------------------------------------------------------


class TestBitfieldTypedModifiersParsing:
	def test_unsigned_int_bitfield(self) -> None:
		prog = _parse("struct S { unsigned int a : 4; };")
		m = prog.declarations[0].members[0]
		assert m.bit_width == 4
		assert m.type_spec.signedness == "unsigned"
		assert m.type_spec.base_type == "int"

	def test_signed_int_bitfield(self) -> None:
		prog = _parse("struct S { signed int a : 7; };")
		m = prog.declarations[0].members[0]
		assert m.bit_width == 7
		assert m.type_spec.signedness == "signed"

	def test_unsigned_char_bitfield(self) -> None:
		prog = _parse("struct S { unsigned char flags : 6; };")
		m = prog.declarations[0].members[0]
		assert m.bit_width == 6
		assert m.type_spec.signedness == "unsigned"
		assert m.type_spec.base_type == "char"

	def test_signed_char_bitfield(self) -> None:
		prog = _parse("struct S { signed char val : 3; };")
		m = prog.declarations[0].members[0]
		assert m.bit_width == 3
		assert m.type_spec.signedness == "signed"
		assert m.type_spec.base_type == "char"

	def test_unsigned_short_bitfield(self) -> None:
		prog = _parse("struct S { unsigned short x : 10; };")
		m = prog.declarations[0].members[0]
		assert m.bit_width == 10
		assert m.type_spec.signedness == "unsigned"
		assert m.type_spec.width_modifier == "short"

	def test_short_bitfield(self) -> None:
		prog = _parse("struct S { short bits : 8; };")
		m = prog.declarations[0].members[0]
		assert m.bit_width == 8
		assert m.type_spec.width_modifier == "short"

	def test_long_bitfield(self) -> None:
		"""long int bitfield -- width up to 64 on 64-bit, just parse check."""
		prog = _parse("struct S { long val : 16; };")
		m = prog.declarations[0].members[0]
		assert m.bit_width == 16
		assert m.type_spec.width_modifier == "long"

	def test_mixed_signed_unsigned_bitfields(self) -> None:
		prog = _parse("""
		struct Mixed {
			signed int a : 4;
			unsigned int b : 4;
			int c : 8;
			unsigned char d : 3;
		};
		""")
		members = prog.declarations[0].members
		assert members[0].type_spec.signedness == "signed"
		assert members[1].type_spec.signedness == "unsigned"
		assert members[2].type_spec.signedness is None
		assert members[3].type_spec.signedness == "unsigned"
		assert members[3].type_spec.base_type == "char"

	def test_all_bitfield_widths_one(self) -> None:
		"""Single-bit fields with various type modifiers."""
		prog = _parse("""
		struct Bits {
			unsigned int a : 1;
			signed int b : 1;
			int c : 1;
			unsigned char d : 1;
		};
		""")
		for m in prog.declarations[0].members:
			assert m.bit_width == 1


# ---------------------------------------------------------------------------
# Bitfields with type modifiers (semantic analysis)
# ---------------------------------------------------------------------------


class TestBitfieldTypedModifiersSemantic:
	def test_unsigned_char_bitfield_width_exceeds(self) -> None:
		"""unsigned char bitfield width > 8 should be an error."""
		prog = _parse("struct S { unsigned char a : 9; };")
		with pytest.raises(SemanticError, match="exceeds type width"):
			SemanticAnalyzer().analyze(prog)

	def test_signed_int_bitfield_max_width(self) -> None:
		prog = _parse("struct S { signed int a : 32; };")
		errors = SemanticAnalyzer().analyze(prog)
		assert len(errors) == 0

	def test_signed_int_bitfield_exceeds(self) -> None:
		prog = _parse("struct S { signed int a : 33; };")
		with pytest.raises(SemanticError, match="exceeds type width"):
			SemanticAnalyzer().analyze(prog)

	def test_unsigned_short_bitfield_max_width(self) -> None:
		prog = _parse("struct S { unsigned short a : 16; };")
		errors = SemanticAnalyzer().analyze(prog)
		assert len(errors) == 0

	def test_unsigned_short_bitfield_exceeds(self) -> None:
		prog = _parse("struct S { unsigned short a : 17; };")
		with pytest.raises(SemanticError, match="exceeds type width"):
			SemanticAnalyzer().analyze(prog)

	def test_mixed_modifier_bitfield_struct_valid(self) -> None:
		prog = _parse("""
		struct S {
			unsigned int a : 3;
			signed int b : 5;
			unsigned char c : 7;
		};
		""")
		errors = SemanticAnalyzer().analyze(prog)
		assert len(errors) == 0


# ---------------------------------------------------------------------------
# Bitfields with type modifiers (IR generation / layout)
# ---------------------------------------------------------------------------


class TestBitfieldTypedModifiersIR:
	def test_unsigned_bitfield_read_write_pipeline(self) -> None:
		source = """
		struct S { unsigned int a : 4; unsigned int b : 4; };
		int main(void) {
			struct S s;
			s.a = 10;
			s.b = 5;
			return s.a;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_signed_bitfield_read_write_pipeline(self) -> None:
		source = """
		struct S { signed int x : 8; };
		int main(void) {
			struct S s;
			s.x = -5;
			return s.x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_bitfield_layout(self) -> None:
		"""short bitfields should pack into 2-byte storage units."""
		members = [
			StructMember(type_spec=TypeSpec(base_type="int", width_modifier="short"), name="a", bit_width=4),
			StructMember(type_spec=TypeSpec(base_type="int", width_modifier="short"), name="b", bit_width=4),
		]
		gen = IRGenerator()
		gen._structs["S"] = members
		gen._compute_bitfield_layout("S")
		layout = gen._bitfield_layouts["S"]
		assert layout["a"][3] == 2  # storage_size = 2 bytes
		assert layout["b"][3] == 2

	def test_unsigned_char_bitfield_layout(self) -> None:
		"""unsigned char bitfields should pack into 1-byte storage units."""
		members = [
			StructMember(
				type_spec=TypeSpec(base_type="char", signedness="unsigned"),
				name="a",
				bit_width=3,
			),
			StructMember(
				type_spec=TypeSpec(base_type="char", signedness="unsigned"),
				name="b",
				bit_width=5,
			),
		]
		gen = IRGenerator()
		gen._structs["S"] = members
		gen._compute_bitfield_layout("S")
		size = gen._compute_struct_size("S")
		assert size == 1

	def test_mixed_signed_unsigned_bitfield_ir(self) -> None:
		source = """
		struct S {
			unsigned int flags : 4;
			signed int offset : 12;
			unsigned int tag : 16;
		};
		int main(void) {
			struct S s;
			s.flags = 0;
			s.offset = -1;
			s.tag = 100;
			return s.flags;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) >= 1

	def test_full_pipeline_bitfield_with_modifiers(self) -> None:
		source = """
		struct Packet {
			unsigned int version : 4;
			unsigned int ihl : 4;
			unsigned int dscp : 6;
			unsigned int ecn : 2;
			unsigned int total_length : 16;
		};
		int main(void) {
			struct Packet p;
			p.version = 4;
			p.ihl = 5;
			return p.version;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm


# ---------------------------------------------------------------------------
# Goto jumping over variable declarations
# ---------------------------------------------------------------------------


class TestGotoOverDeclarations:
	def test_goto_skips_int_declaration(self) -> None:
		source = """
		int main(void) {
			goto skip;
			int x = 42;
			skip: return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_goto_skips_multiple_declarations(self) -> None:
		source = """
		int main(void) {
			int a = 1;
			goto end;
			int b = 2;
			int c = 3;
			long d = 4;
			end: return a;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_goto_skips_short_declaration(self) -> None:
		source = """
		int main(void) {
			goto skip;
			short s = 100;
			skip: return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_goto_skips_unsigned_long_declaration(self) -> None:
		source = """
		int main(void) {
			goto skip;
			unsigned long big = 999999;
			skip: return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_goto_skips_struct_declaration(self) -> None:
		source = """
		struct S { int a : 3; int b : 5; };
		int main(void) {
			goto skip;
			struct S s;
			skip: return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_goto_skips_bool_declaration(self) -> None:
		source = """
		int main(void) {
			goto skip;
			_Bool flag = 1;
			skip: return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_forward_goto_skips_decl_and_assignment(self) -> None:
		"""Variable declared before goto should still be accessible after label."""
		source = """
		int main(void) {
			int x = 10;
			goto end;
			x = 99;
			end: return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		jumps = [i for i in func.body if isinstance(i, IRJump)]
		assert len(jumps) >= 1


# ---------------------------------------------------------------------------
# Goto into/out of loops
# ---------------------------------------------------------------------------


class TestGotoLoopInteractions:
	def test_goto_out_of_while_loop(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			while (i < 100) {
				if (i == 5) goto done;
				i = i + 1;
			}
			done: return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_goto_out_of_for_loop(self) -> None:
		source = """
		int main(void) {
			int i;
			for (i = 0; i < 50; i = i + 1) {
				if (i == 10) goto bail;
			}
			return i;
			bail: return -1;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_goto_out_of_do_while_loop(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			do {
				if (i == 3) goto exit;
				i = i + 1;
			} while (i < 10);
			exit: return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_goto_out_of_nested_loops(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			int j = 0;
			while (i < 10) {
				j = 0;
				while (j < 10) {
					if (i == 3 && j == 7) goto found;
					j = j + 1;
				}
				i = i + 1;
			}
			return -1;
			found: return i * 10 + j;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_goto_into_loop_body(self) -> None:
		"""Goto from outside into the body of a while loop (valid in C)."""
		source = """
		int main(void) {
			int x = 0;
			goto inside;
			while (x < 10) {
				inside:
				x = x + 1;
			}
			return x;
		}
		"""
		# Should at least parse and compile; behavior is implementation-defined
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_goto_between_loop_iterations(self) -> None:
		"""Goto-based loop that mimics continue behavior."""
		source = """
		int main(void) {
			int i = 0;
			int sum = 0;
			loop:
			if (i >= 10) goto done;
			i = i + 1;
			if (i % 2 == 0) goto loop;
			sum = sum + i;
			goto loop;
			done: return sum;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		jumps = [i for i in func.body if isinstance(i, IRJump)]
		assert len(jumps) >= 3

	def test_goto_replaces_break_in_for(self) -> None:
		source = """
		int main(void) {
			int result = 0;
			int i;
			for (i = 0; i < 100; i = i + 1) {
				result = result + i;
				if (result > 50) goto stop;
			}
			stop: return result;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm


# ---------------------------------------------------------------------------
# Combined: bitfields + goto
# ---------------------------------------------------------------------------


class TestBitfieldGotoInteraction:
	def test_goto_skips_bitfield_assignment(self) -> None:
		source = """
		struct S { int a : 4; int b : 4; };
		int main(void) {
			struct S s;
			s.a = 3;
			goto skip;
			s.b = 7;
			skip: return s.a;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_bitfield_access_after_backward_goto(self) -> None:
		source = """
		struct Counter { unsigned int val : 8; };
		int main(void) {
			struct Counter c;
			c.val = 0;
			loop:
			c.val = c.val + 1;
			if (c.val < 10) goto loop;
			return c.val;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bitfield_write_after_conditional_goto(self) -> None:
		source = """
		struct Flags { unsigned int a : 1; unsigned int b : 1; };
		int main(void) {
			struct Flags f;
			f.a = 0;
			f.b = 0;
			if (f.a) goto done;
			f.b = 1;
			done: return f.b;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_goto_out_of_loop_with_bitfield_counter(self) -> None:
		source = """
		struct S { unsigned int count : 16; };
		int main(void) {
			struct S s;
			s.count = 0;
			while (s.count < 100) {
				s.count = s.count + 1;
				if (s.count == 50) goto done;
			}
			done: return s.count;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Combined: type modifiers + goto
# ---------------------------------------------------------------------------


class TestTypeModifierGotoInteraction:
	def test_unsigned_counter_goto_loop(self) -> None:
		source = """
		int main(void) {
			unsigned int i = 0;
			unsigned int sum = 0;
			top:
			sum = sum + i;
			i = i + 1;
			if (i <= 10) goto top;
			return sum;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		jumps = [i for i in func.body if isinstance(i, IRJump)]
		assert len(jumps) >= 1

	def test_long_long_with_goto(self) -> None:
		source = """
		int main(void) {
			long long x = 0;
			goto skip;
			x = 999999999;
			skip: return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_short_variable_with_goto_loop(self) -> None:
		source = """
		int main(void) {
			short i = 0;
			short sum = 0;
			again:
			sum = sum + i;
			i = i + 1;
			if (i < 5) goto again;
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_mixed_types_with_goto(self) -> None:
		source = """
		int main(void) {
			short s = 1;
			long l = 100;
			unsigned int u = 0;
			goto calc;
			s = 999;
			calc:
			u = s + l;
			return u;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_signed_char_with_goto(self) -> None:
		source = """
		int main(void) {
			signed char c = 0;
			top:
			c = c + 1;
			if (c < 10) goto top;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Combined: bitfields + type modifiers + goto (triple interaction)
# ---------------------------------------------------------------------------


class TestTripleInteraction:
	def test_unsigned_bitfield_with_goto_loop(self) -> None:
		source = """
		struct S { unsigned int count : 8; unsigned int done : 1; };
		int main(void) {
			struct S s;
			s.count = 0;
			s.done = 0;
			loop:
			s.count = s.count + 1;
			if (s.count >= 10) {
				s.done = 1;
				goto exit;
			}
			goto loop;
			exit: return s.count;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_signed_bitfield_goto_skip(self) -> None:
		source = """
		struct S { signed int val : 8; };
		int main(void) {
			struct S s;
			s.val = -5;
			goto skip;
			s.val = 100;
			skip: return s.val;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_short_bitfield_goto_backward(self) -> None:
		source = """
		struct S { short x : 8; };
		int main(void) {
			struct S s;
			s.x = 0;
			again:
			s.x = s.x + 1;
			if (s.x < 5) goto again;
			return s.x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_long_with_bitfield_and_goto(self) -> None:
		source = """
		struct S { unsigned int flags : 4; };
		int main(void) {
			long accumulator = 0;
			struct S s;
			s.flags = 0;
			loop:
			accumulator = accumulator + 1;
			s.flags = s.flags + 1;
			if (s.flags < 10) goto loop;
			return accumulator;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_multiple_structs_goto_between(self) -> None:
		source = """
		struct A { unsigned int x : 4; };
		struct B { signed int y : 8; };
		int main(void) {
			struct A a;
			struct B b;
			a.x = 0;
			b.y = 0;
			goto set_b;
			set_a:
			a.x = 15;
			goto done;
			set_b:
			b.y = -10;
			goto set_a;
			done: return a.x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		jumps = [i for i in func.body if isinstance(i, IRJump)]
		assert len(jumps) >= 3

	def test_unsigned_short_bitfield_in_goto_loop(self) -> None:
		source = """
		struct Timer { unsigned short ticks : 12; };
		int main(void) {
			struct Timer t;
			t.ticks = 0;
			tick:
			t.ticks = t.ticks + 1;
			if (t.ticks < 100) goto tick;
			return t.ticks;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_complex_triple_interaction(self) -> None:
		"""Complex scenario: multiple typed bitfields, multiple gotos, mixed types."""
		source = """
		struct Packet {
			unsigned int version : 4;
			unsigned int type : 4;
			unsigned int length : 8;
		};
		int main(void) {
			struct Packet pkt;
			unsigned long total = 0;
			short iter = 0;
			pkt.version = 1;
			pkt.type = 0;
			pkt.length = 0;
			process:
			pkt.type = pkt.type + 1;
			pkt.length = pkt.length + 10;
			total = total + pkt.length;
			iter = iter + 1;
			if (iter < 3) goto process;
			return total;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_triple_interaction_optimized(self) -> None:
		"""Full pipeline with optimization combining all three features."""
		source = """
		struct S { unsigned int val : 16; };
		int main(void) {
			struct S s;
			long result = 0;
			s.val = 1;
			loop:
			result = result + s.val;
			s.val = s.val + 1;
			if (s.val <= 5) goto loop;
			return result;
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		assert "main:" in asm
		assert "ret" in asm


# ---------------------------------------------------------------------------
# Edge cases: goto with struct declarations containing bitfields
# ---------------------------------------------------------------------------


class TestGotoWithBitfieldStructDecl:
	def test_goto_skips_struct_with_bitfields_decl(self) -> None:
		"""Goto jumping over a struct variable declaration that has bitfields."""
		source = """
		struct S { unsigned int a : 4; unsigned int b : 4; };
		int main(void) {
			goto end;
			struct S s;
			s.a = 1;
			end: return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_goto_around_bitfield_struct_definition(self) -> None:
		"""Struct definition before goto, used after label."""
		source = """
		struct S { unsigned int x : 8; };
		int main(void) {
			struct S s;
			s.x = 42;
			goto done;
			s.x = 0;
			done: return s.x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_label_before_bitfield_write(self) -> None:
		source = """
		struct S { unsigned int a : 4; unsigned int b : 4; };
		int main(void) {
			struct S s;
			s.a = 0;
			s.b = 0;
			goto target;
			return -1;
			target:
			s.a = 15;
			s.b = 15;
			return s.a;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm
		assert "main:" in asm
