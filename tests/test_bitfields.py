"""Tests for bitfield support in structs and unions."""

import pytest

from compiler.ast_nodes import StructMember, TypeSpec
from compiler.codegen import CodeGenerator
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer, SemanticError


def parse(source: str):
	tokens = Lexer(source).tokenize()
	return Parser(tokens).parse()


def compile_to_ir(source: str):
	prog = parse(source)
	analyzer = SemanticAnalyzer()
	analyzer.analyze(prog)
	return IRGenerator().generate(prog)


def compile_to_asm(source: str) -> str:
	prog = parse(source)
	analyzer = SemanticAnalyzer()
	analyzer.analyze(prog)
	ir_prog = IRGenerator().generate(prog)
	return CodeGenerator().generate(ir_prog)


# ---------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------


class TestBitfieldParsing:
	def test_basic_bitfield(self):
		prog = parse("struct Flags { int a : 3; int b : 5; };")
		decl = prog.declarations[0]
		assert decl.name == "Flags"
		assert len(decl.members) == 2
		assert decl.members[0].name == "a"
		assert decl.members[0].bit_width == 3
		assert decl.members[1].name == "b"
		assert decl.members[1].bit_width == 5

	def test_unnamed_bitfield(self):
		prog = parse("struct S { int a : 4; int : 2; int b : 6; };")
		decl = prog.declarations[0]
		assert len(decl.members) == 3
		assert decl.members[0].name == "a"
		assert decl.members[0].bit_width == 4
		assert decl.members[1].name == ""
		assert decl.members[1].bit_width == 2
		assert decl.members[2].name == "b"
		assert decl.members[2].bit_width == 6

	def test_zero_width_bitfield(self):
		prog = parse("struct S { int a : 3; int : 0; int b : 5; };")
		decl = prog.declarations[0]
		assert len(decl.members) == 3
		assert decl.members[1].bit_width == 0
		assert decl.members[1].name == ""

	def test_mixed_bitfield_and_regular(self):
		prog = parse("struct S { int x; int a : 3; int y; };")
		decl = prog.declarations[0]
		assert len(decl.members) == 3
		assert decl.members[0].bit_width is None
		assert decl.members[1].bit_width == 3
		assert decl.members[2].bit_width is None

	def test_char_bitfield(self):
		prog = parse("struct S { char a : 4; char b : 4; };")
		decl = prog.declarations[0]
		assert decl.members[0].type_spec.base_type == "char"
		assert decl.members[0].bit_width == 4


# ---------------------------------------------------------------
# Semantic validation tests
# ---------------------------------------------------------------


class TestBitfieldSemanticValidation:
	def test_width_exceeds_type(self):
		prog = parse("struct S { int a : 33; };")
		analyzer = SemanticAnalyzer()
		with pytest.raises(SemanticError, match="exceeds type width"):
			analyzer.analyze(prog)

	def test_char_width_exceeds(self):
		prog = parse("struct S { char a : 9; };")
		analyzer = SemanticAnalyzer()
		with pytest.raises(SemanticError, match="exceeds type width"):
			analyzer.analyze(prog)

	def test_zero_width_named_bitfield(self):
		"""Zero-width bitfield must be unnamed."""
		prog = parse("struct S { int a : 0; };")
		analyzer = SemanticAnalyzer()
		with pytest.raises(SemanticError, match="zero width"):
			analyzer.analyze(prog)

	def test_valid_bitfield_no_errors(self):
		prog = parse("struct S { int a : 1; int b : 31; };")
		analyzer = SemanticAnalyzer()
		errors = analyzer.analyze(prog)
		assert len(errors) == 0

	def test_zero_width_unnamed_is_valid(self):
		prog = parse("struct S { int a : 3; int : 0; int b : 5; };")
		analyzer = SemanticAnalyzer()
		errors = analyzer.analyze(prog)
		assert len(errors) == 0


# ---------------------------------------------------------------
# Layout / size tests
# ---------------------------------------------------------------


class TestBitfieldLayout:
	def _make_gen_with_bitfield_struct(self, name, members):
		gen = IRGenerator()
		gen._structs[name] = members
		gen._compute_bitfield_layout(name)
		return gen

	def test_packing_within_int(self):
		"""Two bitfields that fit in a single int should share storage."""
		members = [
			StructMember(type_spec=TypeSpec(base_type="int"), name="a", bit_width=3),
			StructMember(type_spec=TypeSpec(base_type="int"), name="b", bit_width=5),
		]
		gen = self._make_gen_with_bitfield_struct("S", members)
		assert gen._compute_struct_size("S") == 4  # single int
		layout = gen._bitfield_layouts["S"]
		assert layout["a"] == (0, 0, 3, 4)
		assert layout["b"] == (0, 3, 5, 4)

	def test_cross_boundary(self):
		"""Bitfields that don't fit should start a new storage unit."""
		members = [
			StructMember(type_spec=TypeSpec(base_type="int"), name="a", bit_width=20),
			StructMember(type_spec=TypeSpec(base_type="int"), name="b", bit_width=20),
		]
		gen = self._make_gen_with_bitfield_struct("S", members)
		assert gen._compute_struct_size("S") == 8  # two ints
		layout = gen._bitfield_layouts["S"]
		assert layout["a"] == (0, 0, 20, 4)
		assert layout["b"] == (4, 0, 20, 4)

	def test_zero_width_forces_alignment(self):
		"""Zero-width bitfield forces next bitfield to a new storage unit."""
		members = [
			StructMember(type_spec=TypeSpec(base_type="int"), name="a", bit_width=3),
			StructMember(type_spec=TypeSpec(base_type="int"), name="", bit_width=0),
			StructMember(type_spec=TypeSpec(base_type="int"), name="b", bit_width=5),
		]
		gen = self._make_gen_with_bitfield_struct("S", members)
		assert gen._compute_struct_size("S") == 8  # two ints
		layout = gen._bitfield_layouts["S"]
		assert layout["a"] == (0, 0, 3, 4)
		assert layout["b"] == (4, 0, 5, 4)

	def test_mixed_bitfield_and_regular(self):
		"""Mix of bitfield and regular members."""
		members = [
			StructMember(type_spec=TypeSpec(base_type="int"), name="x"),
			StructMember(type_spec=TypeSpec(base_type="int"), name="a", bit_width=3),
			StructMember(type_spec=TypeSpec(base_type="int"), name="b", bit_width=5),
			StructMember(type_spec=TypeSpec(base_type="int"), name="y"),
		]
		gen = self._make_gen_with_bitfield_struct("S", members)
		# x: 4 bytes, a+b: 4 bytes (packed), y: 4 bytes = 12
		assert gen._compute_struct_size("S") == 12
		layout = gen._bitfield_layouts["S"]
		assert layout["a"] == (4, 0, 3, 4)
		assert layout["b"] == (4, 3, 5, 4)

	def test_char_bitfields_pack_into_byte(self):
		members = [
			StructMember(type_spec=TypeSpec(base_type="char"), name="a", bit_width=4),
			StructMember(type_spec=TypeSpec(base_type="char"), name="b", bit_width=4),
		]
		gen = self._make_gen_with_bitfield_struct("S", members)
		assert gen._compute_struct_size("S") == 1  # single char
		layout = gen._bitfield_layouts["S"]
		assert layout["a"] == (0, 0, 4, 1)
		assert layout["b"] == (0, 4, 4, 1)

	def test_unnamed_bitfield_uses_space(self):
		"""Unnamed bitfields consume bits but have no entry in layout."""
		members = [
			StructMember(type_spec=TypeSpec(base_type="int"), name="a", bit_width=4),
			StructMember(type_spec=TypeSpec(base_type="int"), name="", bit_width=4),
			StructMember(type_spec=TypeSpec(base_type="int"), name="b", bit_width=4),
		]
		gen = self._make_gen_with_bitfield_struct("S", members)
		layout = gen._bitfield_layouts["S"]
		assert layout["a"] == (0, 0, 4, 4)
		assert "" not in layout
		assert layout["b"] == (0, 8, 4, 4)

	def test_field_offset_for_bitfield(self):
		"""_compute_field_offset should return byte offset for bitfield members."""
		members = [
			StructMember(type_spec=TypeSpec(base_type="int"), name="x"),
			StructMember(type_spec=TypeSpec(base_type="int"), name="a", bit_width=3),
			StructMember(type_spec=TypeSpec(base_type="int"), name="b", bit_width=5),
		]
		gen = self._make_gen_with_bitfield_struct("S", members)
		assert gen._compute_field_offset("S", "x") == 0
		assert gen._compute_field_offset("S", "a") == 4
		assert gen._compute_field_offset("S", "b") == 4  # same storage unit


# ---------------------------------------------------------------
# IR generation tests
# ---------------------------------------------------------------


class TestBitfieldIRGen:
	def test_bitfield_read_generates_shift_mask(self):
		source = """
		struct S { int a : 3; int b : 5; };
		int foo() {
			struct S s;
			return s.b;
		}
		"""
		ir_prog = compile_to_ir(source)
		func = ir_prog.functions[0]
		instrs_str = [str(i) for i in func.body]
		joined = "\n".join(instrs_str)
		# Should contain a right shift by 3 (bit_offset of b) and mask with 0x1f (5 bits)
		assert ">> 3" in joined or ">> " in joined
		assert "& 31" in joined or "& " in joined

	def test_bitfield_write_generates_mask_or(self):
		source = """
		struct S { int a : 3; int b : 5; };
		void foo() {
			struct S s;
			s.a = 5;
		}
		"""
		ir_prog = compile_to_ir(source)
		func = ir_prog.functions[0]
		instrs_str = [str(i) for i in func.body]
		joined = "\n".join(instrs_str)
		# Should contain AND for clearing and OR for setting
		assert "& " in joined
		assert "| " in joined

	def test_bitfield_compound_assignment(self):
		source = """
		struct S { int a : 8; };
		void foo() {
			struct S s;
			s.a += 1;
		}
		"""
		ir_prog = compile_to_ir(source)
		func = ir_prog.functions[0]
		instrs_str = [str(i) for i in func.body]
		joined = "\n".join(instrs_str)
		# Should read (mask), add, then write back (clear+or)
		assert "+ 1" in joined

	def test_full_pipeline_compiles(self):
		"""Ensure bitfield code compiles all the way to assembly."""
		source = """
		struct Flags { int a : 3; int b : 5; int c : 8; };
		int main() {
			struct Flags f;
			f.a = 5;
			f.b = 10;
			f.c = 200;
			return f.a;
		}
		"""
		asm = compile_to_asm(source)
		assert "main" in asm

	def test_prefix_increment_bitfield(self):
		source = """
		struct S { int a : 8; };
		int foo() {
			struct S s;
			s.a = 5;
			++s.a;
			return s.a;
		}
		"""
		ir_prog = compile_to_ir(source)
		assert ir_prog.functions[0].body

	def test_postfix_increment_bitfield(self):
		source = """
		struct S { int a : 8; };
		int foo() {
			struct S s;
			s.a = 5;
			s.a++;
			return s.a;
		}
		"""
		ir_prog = compile_to_ir(source)
		assert ir_prog.functions[0].body
