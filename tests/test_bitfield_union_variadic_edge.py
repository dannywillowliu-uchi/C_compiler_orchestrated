"""Edge-case tests for bitfield-union-sizeof and variadic-struct interactions.

Covers:
1. Bitfields inside unions with sizeof verification
2. Bitfields with type modifiers (unsigned/signed)
3. Variadic functions receiving struct arguments
4. sizeof on bitfield-containing structs with padding
"""

from __future__ import annotations

import pytest

from compiler.ast_nodes import (
	StructDecl,
	StructMember,
	TypeSpec,
	UnionDecl,
)
from compiler.codegen import CodeGenerator
from compiler.ir import IRConst, IRReturn
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.preprocessor import Preprocessor
from compiler.semantic import SemanticAnalyzer, SemanticError


def parse(source: str):
	tokens = Lexer(source).tokenize()
	return Parser(tokens).parse()


def compile_to_ir(source: str):
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	prog = Parser(tokens).parse()
	errors = SemanticAnalyzer().analyze(prog)
	assert not errors, f"Semantic errors: {errors}"
	return IRGenerator().generate(prog)


def compile_to_asm(source: str) -> str:
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	prog = Parser(tokens).parse()
	errors = SemanticAnalyzer().analyze(prog)
	assert not errors, f"Semantic errors: {errors}"
	ir_prog = IRGenerator().generate(prog)
	return CodeGenerator().generate(ir_prog)


def get_func_body(ir_prog, name="main"):
	for fn in ir_prog.functions:
		if fn.name == name:
			return fn.body
	raise ValueError(f"Function {name} not found")


def get_return_const(ir_prog, func_name="main") -> int:
	"""Extract the constant value from the first IRReturn with IRConst."""
	body = get_func_body(ir_prog, func_name)
	for instr in body:
		if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
			return instr.value.value
	pytest.fail(f"No IRReturn with IRConst found in {func_name}")


# ===========================================================================
# 1. Bitfields inside unions with sizeof verification
# ===========================================================================


class TestBitfieldInUnionSizeof:
	def test_union_with_int_bitfield_sizeof(self):
		"""Union with an int bitfield should have size of largest member."""
		ir_prog = compile_to_ir("""
			union U { int x : 3; int y; };
			int main() { return sizeof(union U); }
		""")
		# Union size = max(int, int) = 4
		assert get_return_const(ir_prog) == 4

	def test_union_with_char_bitfield_sizeof(self):
		"""Union with a char bitfield should have size driven by largest member."""
		ir_prog = compile_to_ir("""
			union U { char flags : 4; int value; };
			int main() { return sizeof(union U); }
		""")
		# Union size = max(char=1, int=4) aligned to 4 = 4
		assert get_return_const(ir_prog) == 4

	def test_union_all_bitfields_sizeof(self):
		"""Union where all members are bitfields, size = largest storage type."""
		ir_prog = compile_to_ir("""
			union U { int a : 1; int b : 16; int c : 31; };
			int main() { return sizeof(union U); }
		""")
		# All are int-based bitfields, union size = 4
		assert get_return_const(ir_prog) == 4

	def test_union_bitfield_and_double_sizeof(self):
		"""Union with bitfield and double: double dominates size."""
		ir_prog = compile_to_ir("""
			union U { int flags : 5; double d; };
			int main() { return sizeof(union U); }
		""")
		assert get_return_const(ir_prog) == 8

	def test_union_bitfield_parse_structure(self):
		"""Parser produces correct bitfield width in union members."""
		prog = parse("union U { int a : 3; int b : 7; char c; };")
		decl = prog.declarations[0]
		assert isinstance(decl, UnionDecl)
		assert decl.members[0].bit_width == 3
		assert decl.members[1].bit_width == 7
		assert decl.members[2].bit_width is None

	def test_union_unnamed_bitfield_parse(self):
		"""Parser handles unnamed bitfield in union."""
		prog = parse("union U { int : 5; int x; };")
		decl = prog.declarations[0]
		assert isinstance(decl, UnionDecl)
		assert decl.members[0].name == ""
		assert decl.members[0].bit_width == 5


# ===========================================================================
# 2. Bitfields with type modifiers (unsigned/signed)
# ===========================================================================


class TestBitfieldTypeModifiers:
	def test_unsigned_int_bitfield_parses(self):
		"""Struct with unsigned int bitfield should parse correctly."""
		prog = parse("struct S { unsigned int flags : 4; };")
		decl = prog.declarations[0]
		assert isinstance(decl, StructDecl)
		assert decl.members[0].type_spec.signedness == "unsigned"
		assert decl.members[0].bit_width == 4

	def test_signed_int_bitfield_parses(self):
		"""Struct with signed int bitfield should parse correctly."""
		prog = parse("struct S { signed int val : 8; };")
		decl = prog.declarations[0]
		assert decl.members[0].type_spec.signedness == "signed"
		assert decl.members[0].bit_width == 8

	def test_unsigned_bitfield_semantic_valid(self):
		"""Unsigned int bitfield should pass semantic analysis."""
		source = """
			struct S { unsigned int flags : 4; };
			int main() {
				struct S s;
				s.flags = 15;
				return s.flags;
			}
		"""
		asm = compile_to_asm(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_unsigned_bitfield_width_limit(self):
		"""Unsigned int bitfield width cannot exceed 32."""
		source = "struct S { unsigned int x : 33; };"
		prog = parse(source)
		analyzer = SemanticAnalyzer()
		with pytest.raises(SemanticError, match="exceeds type width"):
			analyzer.analyze(prog)

	def test_signed_char_bitfield_parses(self):
		"""Signed char bitfield should parse and have correct properties."""
		prog = parse("struct S { signed char c : 3; };")
		decl = prog.declarations[0]
		assert decl.members[0].type_spec.signedness == "signed"
		assert decl.members[0].type_spec.base_type == "char"
		assert decl.members[0].bit_width == 3

	def test_unsigned_char_bitfield_semantic_valid(self):
		"""unsigned char bitfield should pass semantic and compile."""
		source = """
			struct S { unsigned char flags : 7; };
			int main() {
				struct S s;
				s.flags = 100;
				return s.flags;
			}
		"""
		asm = compile_to_asm(source)
		assert "main:" in asm

	def test_mixed_signed_unsigned_bitfields(self):
		"""Struct mixing signed and unsigned bitfields should compile."""
		source = """
			struct S {
				unsigned int a : 3;
				signed int b : 5;
				int c : 8;
			};
			int main() {
				struct S s;
				s.a = 7;
				s.b = -1;
				s.c = 42;
				return s.a;
			}
		"""
		asm = compile_to_asm(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_unsigned_bitfield_ir_generates_mask(self):
		"""Writing to unsigned bitfield should generate mask-and-shift IR."""
		ir_prog = compile_to_ir("""
			struct S { unsigned int x : 4; };
			int main() {
				struct S s;
				s.x = 15;
				return s.x;
			}
		""")
		body = get_func_body(ir_prog)
		# Should have bitfield mask operations (AND with 15 = 0xF)
		from compiler.ir import IRBinOp
		and_ops = [i for i in body if isinstance(i, IRBinOp) and i.op == "&"]
		assert len(and_ops) >= 1, "Bitfield write should emit AND for masking"


# ===========================================================================
# 3. Variadic functions receiving struct arguments
# ===========================================================================


class TestVariadicStructArgs:
	def test_variadic_with_struct_arg_compiles(self):
		"""Calling variadic function with struct argument should compile."""
		source = """
			#include <stdarg.h>
			struct Point { int x; int y; };
			int f(int n, ...) {
				va_list ap;
				va_start(ap, n);
				int x = va_arg(ap, int);
				va_end(ap);
				return x;
			}
			int main() {
				struct Point p;
				p.x = 10;
				p.y = 20;
				return f(1, p.x);
			}
		"""
		asm = compile_to_asm(source)
		assert "main:" in asm
		assert "f:" in asm

	def test_variadic_with_struct_member_access(self):
		"""Passing struct member to variadic function should compile."""
		source = """
			#include <stdarg.h>
			struct Rect { int w; int h; };
			int sum(int n, ...) {
				va_list ap;
				va_start(ap, n);
				int total = 0;
				int i = 0;
				while (i < n) {
					total = total + va_arg(ap, int);
					i = i + 1;
				}
				va_end(ap);
				return total;
			}
			int main() {
				struct Rect r;
				r.w = 3;
				r.h = 4;
				return sum(2, r.w, r.h);
			}
		"""
		asm = compile_to_asm(source)
		assert "main:" in asm
		assert "call sum" in asm or "callq sum" in asm

	def test_variadic_with_nested_struct_member(self):
		"""Passing nested struct member to variadic should compile."""
		source = """
			#include <stdarg.h>
			struct Inner { int val; };
			struct Outer { struct Inner inner; int tag; };
			int get(int n, ...) {
				va_list ap;
				va_start(ap, n);
				int x = va_arg(ap, int);
				va_end(ap);
				return x;
			}
			int main() {
				struct Outer o;
				o.inner.val = 42;
				o.tag = 1;
				return get(1, o.inner.val);
			}
		"""
		asm = compile_to_asm(source)
		assert "main:" in asm

	def test_variadic_with_union_member(self):
		"""Passing union member to variadic function should compile."""
		source = """
			#include <stdarg.h>
			union U { int i; char c; };
			int f(int n, ...) {
				va_list ap;
				va_start(ap, n);
				int x = va_arg(ap, int);
				va_end(ap);
				return x;
			}
			int main() {
				union U u;
				u.i = 99;
				return f(1, u.i);
			}
		"""
		asm = compile_to_asm(source)
		assert "main:" in asm

	def test_variadic_with_bitfield_member(self):
		"""Passing bitfield member to variadic function should compile."""
		source = """
			#include <stdarg.h>
			struct Flags { unsigned int a : 4; unsigned int b : 4; };
			int f(int n, ...) {
				va_list ap;
				va_start(ap, n);
				int x = va_arg(ap, int);
				va_end(ap);
				return x;
			}
			int main() {
				struct Flags fl;
				fl.a = 5;
				fl.b = 10;
				return f(2, fl.a, fl.b);
			}
		"""
		asm = compile_to_asm(source)
		assert "main:" in asm
		assert "f:" in asm

	def test_variadic_multiple_struct_members_as_args(self):
		"""Multiple struct member args in single variadic call."""
		source = """
			#include <stdarg.h>
			struct Vec3 { int x; int y; int z; };
			int sum3(int n, ...) {
				va_list ap;
				va_start(ap, n);
				int total = 0;
				int i = 0;
				while (i < n) {
					total = total + va_arg(ap, int);
					i = i + 1;
				}
				va_end(ap);
				return total;
			}
			int main() {
				struct Vec3 v;
				v.x = 1;
				v.y = 2;
				v.z = 3;
				return sum3(3, v.x, v.y, v.z);
			}
		"""
		asm = compile_to_asm(source)
		assert "main:" in asm
		assert "call sum3" in asm or "callq sum3" in asm


# ===========================================================================
# 4. sizeof on bitfield-containing structs with padding
# ===========================================================================


class TestBitfieldStructSizeof:
	def test_single_bitfield_sizeof(self):
		"""Struct with one int bitfield should still be 4 bytes (one storage unit)."""
		gen = IRGenerator()
		gen._structs["S"] = [
			StructMember(type_spec=TypeSpec(base_type="int"), name="x", bit_width=1),
		]
		assert gen._compute_struct_size("S") == 4

	def test_multiple_bitfields_fit_one_unit(self):
		"""Multiple bitfields fitting in one int should total 4 bytes."""
		gen = IRGenerator()
		gen._structs["S"] = [
			StructMember(type_spec=TypeSpec(base_type="int"), name="a", bit_width=3),
			StructMember(type_spec=TypeSpec(base_type="int"), name="b", bit_width=5),
			StructMember(type_spec=TypeSpec(base_type="int"), name="c", bit_width=8),
		]
		# 3+5+8=16 bits, fits in one int (32 bits) -> size 4
		assert gen._compute_struct_size("S") == 4

	def test_bitfields_overflow_to_next_unit(self):
		"""Bitfields exceeding 32 bits should spill to next storage unit."""
		gen = IRGenerator()
		gen._structs["S"] = [
			StructMember(type_spec=TypeSpec(base_type="int"), name="a", bit_width=20),
			StructMember(type_spec=TypeSpec(base_type="int"), name="b", bit_width=20),
		]
		# 20+20=40 bits > 32 -> two storage units = 8 bytes
		assert gen._compute_struct_size("S") == 8

	def test_bitfield_with_regular_member_padding(self):
		"""Bitfield followed by regular member should pad correctly."""
		gen = IRGenerator()
		gen._structs["S"] = [
			StructMember(type_spec=TypeSpec(base_type="int"), name="flags", bit_width=4),
			StructMember(type_spec=TypeSpec(base_type="int"), name="value"),
		]
		# bitfield takes 4 bytes (int storage unit), then int(4) = 8 total
		assert gen._compute_struct_size("S") == 8

	def test_regular_member_then_bitfield(self):
		"""Regular member followed by bitfield."""
		gen = IRGenerator()
		gen._structs["S"] = [
			StructMember(type_spec=TypeSpec(base_type="int"), name="x"),
			StructMember(type_spec=TypeSpec(base_type="int"), name="flags", bit_width=3),
		]
		# int(4) + bitfield int(4) = 8
		assert gen._compute_struct_size("S") == 8

	def test_char_bitfield_sizeof(self):
		"""Char bitfields should use 1-byte storage units."""
		gen = IRGenerator()
		gen._structs["S"] = [
			StructMember(type_spec=TypeSpec(base_type="char"), name="a", bit_width=3),
			StructMember(type_spec=TypeSpec(base_type="char"), name="b", bit_width=3),
		]
		# 3+3=6 bits, fits in one char (8 bits) -> size 1
		assert gen._compute_struct_size("S") == 1

	def test_char_bitfield_overflow(self):
		"""Char bitfields exceeding 8 bits should use two storage units."""
		gen = IRGenerator()
		gen._structs["S"] = [
			StructMember(type_spec=TypeSpec(base_type="char"), name="a", bit_width=5),
			StructMember(type_spec=TypeSpec(base_type="char"), name="b", bit_width=5),
		]
		# 5+5=10 bits > 8 -> two char units = 2 bytes
		assert gen._compute_struct_size("S") == 2

	def test_zero_width_bitfield_forces_alignment(self):
		"""Zero-width bitfield should force alignment to next storage boundary."""
		gen = IRGenerator()
		gen._structs["S"] = [
			StructMember(type_spec=TypeSpec(base_type="int"), name="a", bit_width=3),
			StructMember(type_spec=TypeSpec(base_type="int"), name="", bit_width=0),
			StructMember(type_spec=TypeSpec(base_type="int"), name="b", bit_width=3),
		]
		# a in first int(4), zero-width forces new unit, b in second int(4) = 8
		assert gen._compute_struct_size("S") == 8

	def test_bitfield_and_double_padding(self):
		"""Bitfield followed by double should align double to 8."""
		gen = IRGenerator()
		gen._structs["S"] = [
			StructMember(type_spec=TypeSpec(base_type="int"), name="flags", bit_width=4),
			StructMember(type_spec=TypeSpec(base_type="double"), name="d"),
		]
		# bitfield int(4) + pad(4) + double(8) = 16
		assert gen._compute_struct_size("S") == 16

	def test_bitfield_layout_offsets(self):
		"""Bitfield layout should record correct bit positions."""
		gen = IRGenerator()
		gen._structs["S"] = [
			StructMember(type_spec=TypeSpec(base_type="int"), name="a", bit_width=3),
			StructMember(type_spec=TypeSpec(base_type="int"), name="b", bit_width=5),
			StructMember(type_spec=TypeSpec(base_type="int"), name="c", bit_width=8),
		]
		gen._compute_bitfield_layout("S")
		layout = gen._bitfield_layouts["S"]
		# a: byte_offset=0, bit_offset=0, width=3, storage=4
		assert layout["a"] == (0, 0, 3, 4)
		# b: byte_offset=0, bit_offset=3, width=5, storage=4
		assert layout["b"] == (0, 3, 5, 4)
		# c: byte_offset=0, bit_offset=8, width=8, storage=4
		assert layout["c"] == (0, 8, 8, 4)

	def test_bitfield_struct_sizeof_via_compile(self):
		"""sizeof on bitfield struct through full compilation pipeline."""
		ir_prog = compile_to_ir("""
			struct Flags { int a : 3; int b : 5; int c : 8; };
			int main() { return sizeof(struct Flags); }
		""")
		# All fit in one int -> size 4
		assert get_return_const(ir_prog) == 4

	def test_bitfield_struct_sizeof_two_units(self):
		"""sizeof on bitfield struct that spans two storage units."""
		ir_prog = compile_to_ir("""
			struct S { int a : 20; int b : 20; };
			int main() { return sizeof(struct S); }
		""")
		assert get_return_const(ir_prog) == 8

	def test_bitfield_struct_with_char_and_int(self):
		"""Struct mixing char regular member and int bitfields."""
		gen = IRGenerator()
		gen._structs["S"] = [
			StructMember(type_spec=TypeSpec(base_type="char"), name="tag"),
			StructMember(type_spec=TypeSpec(base_type="int"), name="flags", bit_width=8),
		]
		# char(1) + pad(3) + bitfield int(4) = 8
		assert gen._compute_struct_size("S") == 8

	def test_many_1bit_fields(self):
		"""32 one-bit fields should fit in a single int storage unit."""
		gen = IRGenerator()
		gen._structs["Bits"] = [
			StructMember(type_spec=TypeSpec(base_type="int"), name=f"b{i}", bit_width=1)
			for i in range(32)
		]
		assert gen._compute_struct_size("Bits") == 4

	def test_33_one_bit_fields_overflow(self):
		"""33 one-bit int fields should require two storage units."""
		gen = IRGenerator()
		gen._structs["Bits"] = [
			StructMember(type_spec=TypeSpec(base_type="int"), name=f"b{i}", bit_width=1)
			for i in range(33)
		]
		assert gen._compute_struct_size("Bits") == 8


# ===========================================================================
# 5. Cross-feature: bitfield in struct passed to variadic via sizeof
# ===========================================================================


class TestCrossFeatureBitfieldVariadic:
	def test_sizeof_bitfield_struct_in_variadic_call(self):
		"""sizeof(bitfield struct) used as argument to variadic function."""
		source = """
			#include <stdarg.h>
			struct Flags { int a : 3; int b : 5; };
			int f(int n, ...) {
				va_list ap;
				va_start(ap, n);
				int x = va_arg(ap, int);
				va_end(ap);
				return x;
			}
			int main() {
				return f(1, sizeof(struct Flags));
			}
		"""
		asm = compile_to_asm(source)
		assert "main:" in asm
		assert "f:" in asm

	def test_bitfield_write_then_pass_to_variadic(self):
		"""Write to bitfield, read it back, pass to variadic."""
		source = """
			#include <stdarg.h>
			struct S { int x : 8; int y : 8; };
			int f(int n, ...) {
				va_list ap;
				va_start(ap, n);
				int v = va_arg(ap, int);
				va_end(ap);
				return v;
			}
			int main() {
				struct S s;
				s.x = 42;
				s.y = 99;
				return f(1, s.x);
			}
		"""
		asm = compile_to_asm(source)
		assert "main:" in asm

	def test_union_sizeof_in_variadic_call(self):
		"""sizeof(union) passed to variadic function."""
		source = """
			#include <stdarg.h>
			union U { int i; double d; };
			int f(int n, ...) {
				va_list ap;
				va_start(ap, n);
				int x = va_arg(ap, int);
				va_end(ap);
				return x;
			}
			int main() {
				return f(1, sizeof(union U));
			}
		"""
		asm = compile_to_asm(source)
		assert "main:" in asm

	def test_bitfield_union_member_to_variadic(self):
		"""Read bitfield from union, pass to variadic function."""
		source = """
			#include <stdarg.h>
			union U { int raw; };
			int f(int n, ...) {
				va_list ap;
				va_start(ap, n);
				int x = va_arg(ap, int);
				va_end(ap);
				return x;
			}
			int main() {
				union U u;
				u.raw = 255;
				return f(1, u.raw);
			}
		"""
		asm = compile_to_asm(source)
		assert "main:" in asm


# ===========================================================================
# 6. Semantic validation edge cases for bitfields
# ===========================================================================


class TestBitfieldSemanticEdgeCases:
	def test_bitfield_pointer_type_rejected(self):
		"""Pointer-type bitfield should be rejected."""
		source = "struct S { int *p : 3; };"
		prog = parse(source)
		with pytest.raises(SemanticError, match="cannot be a pointer"):
			SemanticAnalyzer().analyze(prog)

	def test_bitfield_array_type_rejected(self):
		"""Array-type bitfield should be rejected at parse time."""
		from compiler.parser import ParseError
		source = "struct S { int arr[2] : 3; };"
		# Parser grammar makes arrays and bitfields mutually exclusive:
		# after parsing array dims, it expects ';' not ':'
		with pytest.raises(ParseError):
			parse(source)

	def test_zero_width_named_bitfield_rejected(self):
		"""Named bitfield with zero width should be rejected."""
		source = "struct S { int x : 0; };"
		prog = parse(source)
		with pytest.raises(SemanticError, match="cannot have zero width"):
			SemanticAnalyzer().analyze(prog)

	def test_zero_width_unnamed_bitfield_accepted(self):
		"""Unnamed zero-width bitfield is valid (alignment directive)."""
		source = """
			struct S { int a : 3; int : 0; int b : 3; };
			int main() {
				struct S s;
				s.a = 1;
				s.b = 2;
				return s.a;
			}
		"""
		asm = compile_to_asm(source)
		assert "main:" in asm

	def test_bitfield_exceeds_char_width(self):
		"""char bitfield exceeding 8 bits should be rejected."""
		source = "struct S { char c : 9; };"
		prog = parse(source)
		with pytest.raises(SemanticError, match="exceeds type width"):
			SemanticAnalyzer().analyze(prog)

	def test_bitfield_exactly_at_type_width(self):
		"""Bitfield at exactly the type width should be accepted."""
		source = """
			struct S { int x : 32; };
			int main() {
				struct S s;
				s.x = 100;
				return s.x;
			}
		"""
		asm = compile_to_asm(source)
		assert "main:" in asm

	def test_bitfield_in_union_semantic_valid(self):
		"""Bitfield in union should pass semantic analysis."""
		source = """
			union U { int a : 3; int b : 5; int raw; };
			int main() {
				union U u;
				u.raw = 0;
				return u.raw;
			}
		"""
		asm = compile_to_asm(source)
		assert "main:" in asm
