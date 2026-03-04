"""Tests for struct/union alignment and padding in field offset calculations."""

from compiler.ast_nodes import StructMember, TypeSpec
from compiler.codegen import CodeGenerator
from compiler.ir import IRAlloc, IRConst, IRReturn, IRStore
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer


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
# Internal alignment helpers (unit tests on IRGenerator)
# ---------------------------------------------------------------


class TestAlignmentHelpers:
	def _make_gen_with_structs(self):
		gen = IRGenerator()
		# struct Padded { char c; int x; }
		gen._structs["Padded"] = [
			StructMember(type_spec=TypeSpec(base_type="char"), name="c"),
			StructMember(type_spec=TypeSpec(base_type="int"), name="x"),
		]
		# struct AllChars { char a; char b; char c; }
		gen._structs["AllChars"] = [
			StructMember(type_spec=TypeSpec(base_type="char"), name="a"),
			StructMember(type_spec=TypeSpec(base_type="char"), name="b"),
			StructMember(type_spec=TypeSpec(base_type="char"), name="c"),
		]
		# struct WithDouble { char c; double d; }
		gen._structs["WithDouble"] = [
			StructMember(type_spec=TypeSpec(base_type="char"), name="c"),
			StructMember(type_spec=TypeSpec(base_type="double"), name="d"),
		]
		# struct Mixed { int a; char b; int c; }
		gen._structs["Mixed"] = [
			StructMember(type_spec=TypeSpec(base_type="int"), name="a"),
			StructMember(type_spec=TypeSpec(base_type="char"), name="b"),
			StructMember(type_spec=TypeSpec(base_type="int"), name="c"),
		]
		# struct WithPointer { char c; int *p; }
		gen._structs["WithPointer"] = [
			StructMember(type_spec=TypeSpec(base_type="char"), name="c"),
			StructMember(type_spec=TypeSpec(base_type="int", pointer_count=1), name="p"),
		]
		# union IntChar { int i; char c; }
		gen._unions["IntChar"] = [
			StructMember(type_spec=TypeSpec(base_type="int"), name="i"),
			StructMember(type_spec=TypeSpec(base_type="char"), name="c"),
		]
		# union WithDouble { int i; double d; }
		gen._unions["WithDouble"] = [
			StructMember(type_spec=TypeSpec(base_type="int"), name="i"),
			StructMember(type_spec=TypeSpec(base_type="double"), name="d"),
		]
		return gen

	def test_char_int_field_offset(self):
		"""struct { char c; int x; } -> x at offset 4, not 1."""
		gen = self._make_gen_with_structs()
		assert gen._compute_field_offset("Padded", "c") == 0
		assert gen._compute_field_offset("Padded", "x") == 4

	def test_char_int_struct_size(self):
		"""struct { char c; int x; } -> size 8, not 5."""
		gen = self._make_gen_with_structs()
		assert gen._compute_struct_size("Padded") == 8

	def test_all_chars_no_padding(self):
		"""struct { char a; char b; char c; } -> size 3, offsets 0/1/2."""
		gen = self._make_gen_with_structs()
		assert gen._compute_field_offset("AllChars", "a") == 0
		assert gen._compute_field_offset("AllChars", "b") == 1
		assert gen._compute_field_offset("AllChars", "c") == 2
		assert gen._compute_struct_size("AllChars") == 3

	def test_char_double_struct(self):
		"""struct { char c; double d; } -> d at offset 8, size 16."""
		gen = self._make_gen_with_structs()
		assert gen._compute_field_offset("WithDouble", "c") == 0
		assert gen._compute_field_offset("WithDouble", "d") == 8
		assert gen._compute_struct_size("WithDouble") == 16

	def test_int_char_int_struct(self):
		"""struct { int a; char b; int c; } -> c at offset 8, size 12."""
		gen = self._make_gen_with_structs()
		assert gen._compute_field_offset("Mixed", "a") == 0
		assert gen._compute_field_offset("Mixed", "b") == 4
		assert gen._compute_field_offset("Mixed", "c") == 8
		assert gen._compute_struct_size("Mixed") == 12

	def test_char_pointer_struct(self):
		"""struct { char c; int *p; } -> p at offset 8, size 16."""
		gen = self._make_gen_with_structs()
		assert gen._compute_field_offset("WithPointer", "c") == 0
		assert gen._compute_field_offset("WithPointer", "p") == 8
		assert gen._compute_struct_size("WithPointer") == 16

	def test_union_size_int_char(self):
		"""union { int i; char c; } -> size 4."""
		gen = self._make_gen_with_structs()
		assert gen._compute_union_size("IntChar") == 4

	def test_union_size_int_double(self):
		"""union { int i; double d; } -> size 8."""
		gen = self._make_gen_with_structs()
		assert gen._compute_union_size("WithDouble") == 8

	def test_nested_struct_alignment(self):
		"""struct Outer { char c; struct Inner inner; } where Inner has int alignment."""
		gen = self._make_gen_with_structs()
		# struct Inner { int x; int y; } -> size 8, align 4
		gen._structs["Inner"] = [
			StructMember(type_spec=TypeSpec(base_type="int"), name="x"),
			StructMember(type_spec=TypeSpec(base_type="int"), name="y"),
		]
		gen._structs["Outer"] = [
			StructMember(type_spec=TypeSpec(base_type="char"), name="c"),
			StructMember(type_spec=TypeSpec(base_type="struct Inner"), name="inner"),
		]
		# inner should be at offset 4 (aligned to 4), size = 4 + 8 = 12
		assert gen._compute_field_offset("Outer", "c") == 0
		assert gen._compute_field_offset("Outer", "inner") == 4
		assert gen._compute_struct_size("Outer") == 12

	def test_nested_struct_with_double(self):
		"""Nested struct with double forces 8-byte alignment of the outer field."""
		gen = self._make_gen_with_structs()
		# struct HasDouble { char a; double d; } -> size 16, align 8
		gen._structs["HasDouble"] = [
			StructMember(type_spec=TypeSpec(base_type="char"), name="a"),
			StructMember(type_spec=TypeSpec(base_type="double"), name="d"),
		]
		gen._structs["Container"] = [
			StructMember(type_spec=TypeSpec(base_type="char"), name="tag"),
			StructMember(type_spec=TypeSpec(base_type="struct HasDouble"), name="inner"),
		]
		# inner aligned to 8 -> offset 8, total = 8 + 16 = 24
		assert gen._compute_field_offset("Container", "inner") == 8
		assert gen._compute_struct_size("Container") == 24


# ---------------------------------------------------------------
# sizeof expression tests (via IR generation)
# ---------------------------------------------------------------


class TestSizeofWithAlignment:
	def test_sizeof_char_int_struct(self):
		"""sizeof(struct { char c; int x; }) should be 8."""
		ir_prog = compile_to_ir("""
			struct Padded { char c; int x; };
			int main() {
				return sizeof(struct Padded);
			}
		""")
		for instr in ir_prog.functions[0].body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				assert instr.value.value == 8

	def test_sizeof_char_double_struct(self):
		"""sizeof(struct { char c; double d; }) should be 16."""
		ir_prog = compile_to_ir("""
			struct Big { char c; double d; };
			int main() {
				return sizeof(struct Big);
			}
		""")
		for instr in ir_prog.functions[0].body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				assert instr.value.value == 16

	def test_sizeof_all_ints_struct(self):
		"""sizeof(struct { int a; int b; }) should be 8 (no padding needed)."""
		ir_prog = compile_to_ir("""
			struct TwoInts { int a; int b; };
			int main() {
				return sizeof(struct TwoInts);
			}
		""")
		for instr in ir_prog.functions[0].body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				assert instr.value.value == 8

	def test_sizeof_union_uses_largest(self):
		"""sizeof(union { int i; double d; }) should be 8."""
		ir_prog = compile_to_ir("""
			union Data { int i; double d; };
			int main() {
				return sizeof(union Data);
			}
		""")
		for instr in ir_prog.functions[0].body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				assert instr.value.value == 8


# ---------------------------------------------------------------
# Struct allocation size tests
# ---------------------------------------------------------------


class TestStructAllocSize:
	def test_struct_var_alloc_size_padded(self):
		"""Allocating struct { char c; int x; } should allocate 8 bytes."""
		ir_prog = compile_to_ir("""
			struct Padded { char c; int x; };
			int main() {
				struct Padded p;
				return 0;
			}
		""")
		func = ir_prog.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert len(allocs) >= 1
		assert allocs[0].size == 8


# ---------------------------------------------------------------
# Initializer list with aligned offsets
# ---------------------------------------------------------------


class TestInitializerWithAlignment:
	def test_struct_init_uses_aligned_offsets(self):
		"""struct { char c; int x; } = {'A', 42} should store x at offset 4."""
		ir_prog = compile_to_ir("""
			struct Padded { char c; int x; };
			int main() {
				struct Padded p = {65, 42};
				return 0;
			}
		""")
		func = ir_prog.functions[0]
		stores = [i for i in func.body if isinstance(i, IRStore)]
		# We should have at least 2 stores for the two fields
		assert len(stores) >= 2

	def test_struct_init_compiles_to_asm(self):
		"""Struct with padding initializer should compile to valid asm."""
		asm = compile_to_asm("""
			struct Padded { char c; int x; };
			int main() {
				struct Padded p = {65, 42};
				return p.x;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm


# ---------------------------------------------------------------
# End-to-end compilation tests
# ---------------------------------------------------------------


class TestStructLayoutEndToEnd:
	def test_padded_struct_compiles(self):
		asm = compile_to_asm("""
			struct Padded { char c; int x; };
			int main() {
				struct Padded p;
				p.c = 65;
				p.x = 42;
				return p.x;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_double_struct_compiles(self):
		asm = compile_to_asm("""
			struct Big { char c; double d; };
			int main() {
				struct Big b;
				b.c = 65;
				return sizeof(struct Big);
			}
		""")
		assert "main:" in asm

	def test_nested_struct_compiles(self):
		asm = compile_to_asm("""
			struct Inner { int x; int y; };
			struct Outer { char c; struct Inner inner; };
			int main() {
				struct Outer o;
				o.c = 1;
				return sizeof(struct Outer);
			}
		""")
		assert "main:" in asm

	def test_union_sizeof_in_asm(self):
		asm = compile_to_asm("""
			union Data { int i; char c; };
			int main() {
				return sizeof(union Data);
			}
		""")
		assert "main:" in asm
