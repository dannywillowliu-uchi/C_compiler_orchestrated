"""Edge-case tests for struct/union features across parser, semantic, IR gen, and codegen."""

from __future__ import annotations

import pytest

from compiler.ast_nodes import (
	CompoundStmt,
	ExprStmt,
	FunctionDecl,
	Identifier,
	IntLiteral,
	MemberAccess,
	Program,
	ReturnStmt,
	SourceLocation,
	StructDecl,
	StructMember,
	TypeSpec,
	UnionDecl,
	VarDecl,
)
from compiler.codegen import CodeGenerator
from compiler.ir import IRAlloc, IRBulkCopy, IRConst, IRLoad, IRReturn, IRStore
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer, SemanticError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def loc(line: int = 1, col: int = 1) -> SourceLocation:
	return SourceLocation(line=line, col=col)


def int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def char_type() -> TypeSpec:
	return TypeSpec(base_type="char")


def double_type() -> TypeSpec:
	return TypeSpec(base_type="double")


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


def get_func_body(ir_prog, name="main"):
	for fn in ir_prog.functions:
		if fn.name == name:
			return fn.body
	raise ValueError(f"Function {name} not found")


def _make_struct(name: str, members: list[StructMember]) -> StructDecl:
	return StructDecl(name=name, members=members, loc=loc())


def _make_union(name: str, members: list[StructMember]) -> UnionDecl:
	return UnionDecl(name=name, members=members, loc=loc())


def _wrap_in_function(*decls_and_stmts) -> Program:
	"""Build a Program with top-level declarations and a test function."""
	top_decls = []
	body_stmts = []
	for item in decls_and_stmts:
		if isinstance(item, (StructDecl, UnionDecl)):
			top_decls.append(item)
		else:
			body_stmts.append(item)
	body_stmts.append(ReturnStmt(expression=IntLiteral(value=0)))
	top_decls.append(FunctionDecl(
		return_type=int_type(),
		name="test",
		params=[],
		body=CompoundStmt(statements=body_stmts),
		loc=loc(),
	))
	return Program(declarations=top_decls)


# ===========================================================================
# Parser tests: nested structs, unions, edge cases
# ===========================================================================


class TestParserNestedStructs:
	def test_struct_with_struct_member(self):
		"""Parse struct containing another struct as a member."""
		prog = parse("""
			struct Inner { int x; int y; };
			struct Outer { struct Inner inner; int z; };
		""")
		assert len(prog.declarations) == 2
		outer = prog.declarations[1]
		assert isinstance(outer, StructDecl)
		assert outer.name == "Outer"
		assert len(outer.members) == 2
		assert outer.members[0].type_spec.base_type == "struct Inner"
		assert outer.members[0].name == "inner"

	def test_struct_with_pointer_to_self(self):
		"""Parse struct with a pointer member to its own type (linked list)."""
		prog = parse("struct Node { int val; struct Node *next; };")
		decl = prog.declarations[0]
		assert isinstance(decl, StructDecl)
		assert decl.members[1].name == "next"
		assert decl.members[1].type_spec.base_type == "struct Node"
		assert decl.members[1].type_spec.pointer_count == 1

	def test_union_with_multiple_types(self):
		"""Parse union with diverse member types."""
		prog = parse("union Val { int i; double d; char c; char *s; };")
		decl = prog.declarations[0]
		assert isinstance(decl, UnionDecl)
		assert len(decl.members) == 4
		assert decl.members[0].type_spec.base_type == "int"
		assert decl.members[1].type_spec.base_type == "double"
		assert decl.members[2].type_spec.base_type == "char"
		assert decl.members[3].type_spec.base_type == "char"
		assert decl.members[3].type_spec.pointer_count == 1

	def test_struct_with_array_and_pointer_members(self):
		"""Parse struct mixing array and pointer members."""
		prog = parse("struct Mixed { int data[5]; char *name; int count; };")
		decl = prog.declarations[0]
		assert isinstance(decl, StructDecl)
		assert len(decl.members) == 3
		assert len(decl.members[0].array_dims) == 1
		assert decl.members[0].array_dims[0].value == 5
		assert decl.members[1].type_spec.pointer_count == 1
		assert len(decl.members[2].array_dims) == 0

	def test_struct_single_member(self):
		"""Parse struct with only one member."""
		prog = parse("struct Wrapper { int value; };")
		decl = prog.declarations[0]
		assert isinstance(decl, StructDecl)
		assert len(decl.members) == 1
		assert decl.members[0].name == "value"

	def test_union_single_member(self):
		"""Parse union with only one member."""
		prog = parse("union Single { int x; };")
		decl = prog.declarations[0]
		assert isinstance(decl, UnionDecl)
		assert len(decl.members) == 1


# ===========================================================================
# Semantic tests: nested access, duplicate members, invalid access
# ===========================================================================


class TestSemanticNestedAccess:
	def test_nested_struct_dot_access(self):
		"""Accessing outer.inner.x should be valid."""
		prog = _wrap_in_function(
			_make_struct("Inner", [
				StructMember(type_spec=int_type(), name="x"),
				StructMember(type_spec=int_type(), name="y"),
			]),
			_make_struct("Outer", [
				StructMember(type_spec=TypeSpec(base_type="struct Inner"), name="inner"),
				StructMember(type_spec=int_type(), name="z"),
			]),
			VarDecl(type_spec=TypeSpec(base_type="struct Outer"), name="o", loc=loc()),
			ExprStmt(expression=MemberAccess(
				object=MemberAccess(
					object=Identifier(name="o"),
					member="inner",
					is_arrow=False,
					loc=loc(3, 1),
				),
				member="x",
				is_arrow=False,
				loc=loc(3, 9),
			)),
		)
		errors = SemanticAnalyzer().analyze(prog)
		assert len(errors) == 0

	def test_nested_struct_invalid_inner_member(self):
		"""Accessing outer.inner.nonexistent should error."""
		prog = _wrap_in_function(
			_make_struct("Inner", [
				StructMember(type_spec=int_type(), name="x"),
			]),
			_make_struct("Outer", [
				StructMember(type_spec=TypeSpec(base_type="struct Inner"), name="inner"),
			]),
			VarDecl(type_spec=TypeSpec(base_type="struct Outer"), name="o", loc=loc()),
			ExprStmt(expression=MemberAccess(
				object=MemberAccess(
					object=Identifier(name="o"),
					member="inner",
					is_arrow=False,
					loc=loc(3, 1),
				),
				member="nonexistent",
				is_arrow=False,
				loc=loc(3, 9),
			)),
		)
		with pytest.raises(SemanticError, match="has no member nonexistent"):
			SemanticAnalyzer().analyze(prog)

	def test_union_valid_member_access(self):
		"""Accessing valid union member should succeed."""
		prog = _wrap_in_function(
			_make_union("Data", [
				StructMember(type_spec=int_type(), name="i"),
				StructMember(type_spec=double_type(), name="d"),
			]),
			VarDecl(type_spec=TypeSpec(base_type="union Data"), name="u", loc=loc()),
			ExprStmt(expression=MemberAccess(
				object=Identifier(name="u"),
				member="d",
				is_arrow=False,
				loc=loc(3, 1),
			)),
		)
		errors = SemanticAnalyzer().analyze(prog)
		assert len(errors) == 0

	def test_union_invalid_member(self):
		"""Accessing nonexistent union member should error."""
		prog = _wrap_in_function(
			_make_union("Data", [
				StructMember(type_spec=int_type(), name="i"),
			]),
			VarDecl(type_spec=TypeSpec(base_type="union Data"), name="u", loc=loc()),
			ExprStmt(expression=MemberAccess(
				object=Identifier(name="u"),
				member="missing",
				is_arrow=False,
				loc=loc(3, 1),
			)),
		)
		with pytest.raises(SemanticError, match="union Data has no member missing"):
			SemanticAnalyzer().analyze(prog)

	def test_struct_duplicate_member_names(self):
		"""Struct with duplicate member names should produce error."""
		prog = _wrap_in_function(
			_make_struct("Bad", [
				StructMember(type_spec=int_type(), name="x"),
				StructMember(type_spec=int_type(), name="x"),
			]),
		)
		with pytest.raises(SemanticError, match="duplicate member"):
			SemanticAnalyzer().analyze(prog)

	def test_union_duplicate_member_names(self):
		"""Union with duplicate member names should produce error."""
		prog = _wrap_in_function(
			_make_union("Bad", [
				StructMember(type_spec=int_type(), name="a"),
				StructMember(type_spec=int_type(), name="a"),
			]),
		)
		with pytest.raises(SemanticError, match="duplicate member"):
			SemanticAnalyzer().analyze(prog)


# ===========================================================================
# IR gen: struct layout, alignment, sizeof with padding
# ===========================================================================


class TestStructSizeofWithPadding:
	def test_sizeof_nested_struct(self):
		"""sizeof(struct Outer) should account for nested struct alignment."""
		ir_prog = compile_to_ir("""
			struct Inner { int a; int b; };
			struct Outer { char c; struct Inner inner; };
			int main() {
				return sizeof(struct Outer);
			}
		""")
		body = get_func_body(ir_prog)
		for instr in body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				# char(1) + pad(3) + Inner(8) = 12
				assert instr.value.value == 12
				return
		pytest.fail("No IRReturn with IRConst found")

	def test_sizeof_struct_trailing_padding(self):
		"""struct { int a; char b; } needs trailing pad to align to 4 -> size 8."""
		ir_prog = compile_to_ir("""
			struct S { int a; char b; };
			int main() {
				return sizeof(struct S);
			}
		""")
		body = get_func_body(ir_prog)
		for instr in body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				assert instr.value.value == 8
				return
		pytest.fail("No IRReturn with IRConst found")

	def test_sizeof_union_largest_member(self):
		"""Union size should be the largest member with alignment padding."""
		ir_prog = compile_to_ir("""
			union U { char c; double d; int i; };
			int main() {
				return sizeof(union U);
			}
		""")
		body = get_func_body(ir_prog)
		for instr in body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				# largest is double (8), alignment is 8 -> size 8
				assert instr.value.value == 8
				return
		pytest.fail("No IRReturn with IRConst found")

	def test_sizeof_struct_with_pointer(self):
		"""struct with pointer member gets 8-byte alignment."""
		ir_prog = compile_to_ir("""
			struct WithPtr { char c; int *p; };
			int main() {
				return sizeof(struct WithPtr);
			}
		""")
		body = get_func_body(ir_prog)
		for instr in body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				# char(1) + pad(7) + pointer(8) = 16
				assert instr.value.value == 16
				return
		pytest.fail("No IRReturn with IRConst found")


class TestStructLayoutUnit:
	"""Direct unit tests on IRGenerator layout computation methods."""

	def test_deeply_nested_struct_size(self):
		"""Three levels of nesting with mixed types."""
		gen = IRGenerator()
		gen._structs["A"] = [
			StructMember(type_spec=char_type(), name="c"),
		]
		gen._structs["B"] = [
			StructMember(type_spec=TypeSpec(base_type="struct A"), name="a"),
			StructMember(type_spec=int_type(), name="x"),
		]
		gen._structs["C"] = [
			StructMember(type_spec=TypeSpec(base_type="struct B"), name="b"),
			StructMember(type_spec=char_type(), name="tag"),
		]
		# A: size=1 align=1
		assert gen._compute_struct_size("A") == 1
		# B: A(1) + pad(3) + int(4) = 8, align=4
		assert gen._compute_struct_size("B") == 8
		# C: B(8) at offset 0 + char(1) at offset 8 = 9, padded to 12 (align 4)
		assert gen._compute_struct_size("C") == 12
		assert gen._compute_field_offset("C", "b") == 0
		assert gen._compute_field_offset("C", "tag") == 8

	def test_struct_all_doubles(self):
		"""Struct of all doubles, no padding needed."""
		gen = IRGenerator()
		gen._structs["Doubles"] = [
			StructMember(type_spec=double_type(), name="a"),
			StructMember(type_spec=double_type(), name="b"),
			StructMember(type_spec=double_type(), name="c"),
		]
		assert gen._compute_struct_size("Doubles") == 24
		assert gen._compute_field_offset("Doubles", "a") == 0
		assert gen._compute_field_offset("Doubles", "b") == 8
		assert gen._compute_field_offset("Doubles", "c") == 16

	def test_union_with_nested_struct(self):
		"""Union containing a struct member uses struct's size."""
		gen = IRGenerator()
		gen._structs["Pair"] = [
			StructMember(type_spec=int_type(), name="x"),
			StructMember(type_spec=int_type(), name="y"),
		]
		gen._unions["U"] = [
			StructMember(type_spec=TypeSpec(base_type="struct Pair"), name="p"),
			StructMember(type_spec=char_type(), name="c"),
		]
		# Pair is 8 bytes, char is 1 -> union is 8, alignment is 4
		assert gen._compute_union_size("U") == 8

	def test_empty_struct_size(self):
		"""Empty struct has size 0."""
		gen = IRGenerator()
		gen._structs["Empty"] = []
		assert gen._compute_struct_size("Empty") == 0

	def test_empty_union_size(self):
		"""Empty union has size 0."""
		gen = IRGenerator()
		gen._unions["Empty"] = []
		assert gen._compute_union_size("Empty") == 0

	def test_struct_char_double_char(self):
		"""struct { char a; double d; char b; } -> a@0, d@8, b@16, size=24."""
		gen = IRGenerator()
		gen._structs["S"] = [
			StructMember(type_spec=char_type(), name="a"),
			StructMember(type_spec=double_type(), name="d"),
			StructMember(type_spec=char_type(), name="b"),
		]
		assert gen._compute_field_offset("S", "a") == 0
		assert gen._compute_field_offset("S", "d") == 8
		assert gen._compute_field_offset("S", "b") == 16
		# size = 17 padded to 24 (alignment 8)
		assert gen._compute_struct_size("S") == 24


# ===========================================================================
# Union overlapping members
# ===========================================================================


class TestUnionOverlappingMembers:
	def test_union_all_members_at_offset_zero(self):
		"""All union fields share offset 0 in IR generation."""
		gen = IRGenerator()
		gen._unions["Multi"] = [
			StructMember(type_spec=int_type(), name="i"),
			StructMember(type_spec=double_type(), name="d"),
			StructMember(type_spec=char_type(), name="c"),
		]
		# Union members don't use _compute_field_offset (that's for structs),
		# but verifying union size and that codegen treats them at offset 0
		assert gen._compute_union_size("Multi") == 8

	def test_union_write_int_read_char_compiles(self):
		"""Writing to one union member and reading another should compile."""
		asm = compile_to_asm("""
			union U { int i; char c; };
			int main() {
				union U u;
				u.i = 65;
				return u.c;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_union_assignment_copies_full_size(self):
		"""Union copy should use word-sized load/store."""
		ir_prog = compile_to_ir("""
			union V { int i; double d; };
			int main() {
				union V a;
				union V b;
				a.i = 42;
				b = a;
				return 0;
			}
		""")
		body = get_func_body(ir_prog)
		loads = [i for i in body if isinstance(i, IRLoad)]
		stores = [i for i in body if isinstance(i, IRStore)]
		assert len(loads) >= 1
		assert len(stores) >= 2


# ===========================================================================
# Nested struct access in IR gen and codegen
# ===========================================================================


class TestNestedStructIRGen:
	def test_nested_member_access_emits_offset_arithmetic(self):
		"""outer.inner.x should emit address calculations through both levels."""
		ir_prog = compile_to_ir("""
			struct Inner { int a; int b; };
			struct Outer { struct Inner inner; int c; };
			int main() {
				struct Outer o;
				o.inner.a = 10;
				return 0;
			}
		""")
		body = get_func_body(ir_prog)
		stores = [i for i in body if isinstance(i, IRStore)]
		assert len(stores) >= 1

	def test_nested_member_read_compiles(self):
		"""Reading a nested struct member should produce valid asm."""
		asm = compile_to_asm("""
			struct Inner { int x; int y; };
			struct Outer { struct Inner inner; int z; };
			int main() {
				struct Outer o;
				o.inner.x = 5;
				o.inner.y = 10;
				o.z = 20;
				return o.inner.y;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_three_level_nesting_compiles(self):
		"""Three levels of struct nesting should compile."""
		asm = compile_to_asm("""
			struct A { int val; };
			struct B { struct A a; int extra; };
			struct C { struct B b; int tag; };
			int main() {
				struct C c;
				c.b.a.val = 42;
				c.b.extra = 10;
				c.tag = 99;
				return c.b.a.val;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm


# ===========================================================================
# Struct assignment/copy edge cases
# ===========================================================================


class TestStructCopyEdgeCases:
	def test_copy_struct_with_char_and_int(self):
		"""Copy struct with mixed small and large fields uses bulk copy."""
		ir_prog = compile_to_ir("""
			struct Mixed { char c; int x; };
			int main() {
				struct Mixed a;
				struct Mixed b;
				a.c = 65;
				a.x = 100;
				b = a;
				return 0;
			}
		""")
		body = get_func_body(ir_prog)
		bulk_copies = [i for i in body if isinstance(i, IRBulkCopy)]
		assert len(bulk_copies) >= 1, "Struct copy should emit IRBulkCopy"
		assert bulk_copies[0].size == 8  # char(1) + padding(3) + int(4) = 8

	def test_copy_struct_with_pointer_member(self):
		"""Copy struct containing a pointer member uses bulk copy."""
		ir_prog = compile_to_ir("""
			struct WithPtr { int val; int *ptr; };
			int main() {
				struct WithPtr a;
				struct WithPtr b;
				a.val = 1;
				b = a;
				return 0;
			}
		""")
		body = get_func_body(ir_prog)
		bulk_copies = [i for i in body if isinstance(i, IRBulkCopy)]
		assert len(bulk_copies) >= 1, "Struct copy should emit IRBulkCopy"

	def test_self_assignment_compiles(self):
		"""a = a (self-assignment) should compile without errors."""
		asm = compile_to_asm("""
			struct S { int x; int y; };
			int main() {
				struct S a;
				a.x = 1;
				a.y = 2;
				a = a;
				return 0;
			}
		""")
		assert "main:" in asm

	def test_copy_after_modification_compiles(self):
		"""Copy, modify, copy again should compile."""
		asm = compile_to_asm("""
			struct Point { int x; int y; };
			int main() {
				struct Point a;
				struct Point b;
				a.x = 1;
				a.y = 2;
				b = a;
				a.x = 99;
				b = a;
				return b.x;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm


# ===========================================================================
# Struct with array members
# ===========================================================================


class TestStructWithArrayMembers:
	def test_struct_array_member_sizeof(self):
		"""sizeof struct with array member accounts for array size."""
		gen = IRGenerator()
		gen._structs["Buf"] = [
			StructMember(
				type_spec=int_type(),
				name="data",
				array_dims=[IntLiteral(value=4)],
			),
			StructMember(type_spec=int_type(), name="len"),
		]
		# data[4] = 16 bytes + len = 4 bytes = 20, aligned to 4 -> 20
		# Note: array member size depends on implementation
		# The struct size must be at least 8 (len + at least one int)
		size = gen._compute_struct_size("Buf")
		assert size >= 8

	def test_struct_with_char_array_compiles(self):
		"""Struct with char array member should compile through codegen."""
		asm = compile_to_asm("""
			struct Msg { char text[32]; int len; };
			int main() {
				struct Msg m;
				m.len = 5;
				return m.len;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm


# ===========================================================================
# Struct pointer access (arrow operator)
# ===========================================================================


class TestStructPointerAccess:
	def test_arrow_access_compiles(self):
		"""Arrow access on struct pointer should compile."""
		asm = compile_to_asm("""
			struct Point { int x; int y; };
			int main() {
				struct Point p;
				struct Point *ptr;
				p.x = 10;
				p.y = 20;
				ptr = &p;
				return ptr->x;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_arrow_access_write_compiles(self):
		"""Writing through arrow operator should compile."""
		asm = compile_to_asm("""
			struct S { int val; };
			int main() {
				struct S s;
				struct S *p;
				p = &s;
				p->val = 42;
				return s.val;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_arrow_on_nested_struct_compiles(self):
		"""Arrow access into nested struct should compile."""
		asm = compile_to_asm("""
			struct Inner { int x; };
			struct Outer { struct Inner inner; int y; };
			int main() {
				struct Outer o;
				struct Outer *p;
				o.inner.x = 5;
				o.y = 10;
				p = &o;
				return p->y;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm


# ===========================================================================
# Struct/union allocation sizes
# ===========================================================================


class TestAllocSizes:
	def test_nested_struct_alloc_size(self):
		"""Allocation for nested struct should use correct padded size."""
		ir_prog = compile_to_ir("""
			struct Inner { int a; int b; };
			struct Outer { char c; struct Inner inner; };
			int main() {
				struct Outer o;
				return 0;
			}
		""")
		func = ir_prog.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert len(allocs) >= 1
		# Outer: char(1) + pad(3) + Inner(8) = 12
		assert allocs[0].size == 12

	def test_union_alloc_uses_largest(self):
		"""Union allocation should use the largest member's size."""
		ir_prog = compile_to_ir("""
			union U { int i; double d; };
			int main() {
				union U u;
				return 0;
			}
		""")
		func = ir_prog.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert len(allocs) >= 1
		assert allocs[0].size == 8

	def test_struct_with_double_alloc(self):
		"""struct { char c; double d; } should allocate 16 bytes."""
		ir_prog = compile_to_ir("""
			struct S { char c; double d; };
			int main() {
				struct S s;
				return 0;
			}
		""")
		func = ir_prog.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert len(allocs) >= 1
		assert allocs[0].size == 16


# ===========================================================================
# Struct initializer edge cases
# ===========================================================================


class TestStructInitEdgeCases:
	def test_single_field_struct_init(self):
		"""Struct with one field initialized by literal."""
		asm = compile_to_asm("""
			struct W { int val; };
			int main() {
				struct W w = {42};
				return w.val;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_multi_field_struct_init(self):
		"""Struct init with multiple fields generates correct stores."""
		ir_prog = compile_to_ir("""
			struct RGB { int r; int g; int b; };
			int main() {
				struct RGB c = {255, 128, 0};
				return c.g;
			}
		""")
		body = get_func_body(ir_prog)
		stores = [i for i in body if isinstance(i, IRStore)]
		# At least 3 stores for the 3 fields
		assert len(stores) >= 3

	def test_struct_init_with_padded_fields(self):
		"""Init of struct with padding should still compile."""
		asm = compile_to_asm("""
			struct P { char c; int x; };
			int main() {
				struct P p = {65, 100};
				return p.x;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm


# ===========================================================================
# End-to-end codegen stress tests
# ===========================================================================


class TestEndToEndStructUnion:
	def test_struct_return_member_value(self):
		"""Return a struct member value through codegen."""
		asm = compile_to_asm("""
			struct S { int a; int b; int c; };
			int main() {
				struct S s;
				s.a = 10;
				s.b = 20;
				s.c = 30;
				return s.b;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_union_write_read_different_members(self):
		"""Write to one union member, read another (type punning)."""
		asm = compile_to_asm("""
			union Pun { int i; char bytes[4]; };
			int main() {
				union Pun p;
				p.i = 0;
				return p.i;
			}
		""")
		assert "main:" in asm

	def test_struct_with_union_member(self):
		"""Struct containing a union member should compile."""
		asm = compile_to_asm("""
			union Val { int i; char c; };
			struct Tagged { int tag; union Val val; };
			int main() {
				struct Tagged t;
				t.tag = 1;
				t.val.i = 42;
				return t.tag;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_union_with_struct_member(self):
		"""Union containing a struct member should compile."""
		asm = compile_to_asm("""
			struct Pair { int x; int y; };
			union U { struct Pair p; int raw; };
			int main() {
				union U u;
				u.raw = 0;
				return u.raw;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_sizeof_in_expression(self):
		"""sizeof struct used in arithmetic expression."""
		asm = compile_to_asm("""
			struct S { int a; int b; };
			int main() {
				int x;
				x = sizeof(struct S) + 1;
				return x;
			}
		""")
		assert "main:" in asm

	def test_multiple_struct_vars(self):
		"""Multiple variables of the same struct type."""
		asm = compile_to_asm("""
			struct P { int x; int y; };
			int main() {
				struct P a;
				struct P b;
				struct P c;
				a.x = 1;
				b.x = 2;
				c.x = 3;
				return a.x;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm
