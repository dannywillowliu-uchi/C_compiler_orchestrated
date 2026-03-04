"""Tests for struct/union copy assignment via field-by-field IR generation."""

from compiler.ir import IRLoad, IRStore, IRBinOp, IRType
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


def get_func_body(ir_prog, name="main"):
	for fn in ir_prog.functions:
		if fn.name == name:
			return fn.body
	raise ValueError(f"Function {name} not found")


class TestSimpleStructCopy:
	"""Test direct struct assignment: a = b."""

	def test_simple_struct_copy_emits_loads_and_stores(self):
		source = """
		struct Point { int x; int y; };
		int main() {
			struct Point a;
			struct Point b;
			a.x = 1;
			a.y = 2;
			b = a;
			return 0;
		}
		"""
		ir_prog = compile_to_ir(source)
		body = get_func_body(ir_prog)
		# After the assignment 'b = a', we should see field-by-field
		# IRLoad/IRStore pairs (not a single IRCopy for the whole struct)
		loads = [i for i in body if isinstance(i, IRLoad)]
		stores = [i for i in body if isinstance(i, IRStore)]
		# At minimum: 2 stores for a.x=1 and a.y=2, plus 2 load/store pairs for the copy
		assert len(loads) >= 2, f"Expected at least 2 loads for field copy, got {len(loads)}"
		assert len(stores) >= 4, f"Expected at least 4 stores, got {len(stores)}"

	def test_struct_copy_preserves_field_values(self):
		"""Verify the copy reads from source fields and writes to dest fields."""
		source = """
		struct Pair { int a; int b; };
		int main() {
			struct Pair x;
			struct Pair y;
			x.a = 10;
			x.b = 20;
			y = x;
			return 0;
		}
		"""
		ir_prog = compile_to_ir(source)
		body = get_func_body(ir_prog)
		# Should have IRLoad instructions reading from source struct
		loads_after_stores = []
		store_count = 0
		for i in body:
			if isinstance(i, IRStore):
				store_count += 1
			if isinstance(i, IRLoad) and store_count >= 2:
				loads_after_stores.append(i)
		assert len(loads_after_stores) >= 2, "Expected loads for field-by-field copy"


class TestStructInitFromVariable:
	"""Test struct initialization from another struct variable."""

	def test_init_from_variable(self):
		source = """
		struct Vec2 { int x; int y; };
		int main() {
			struct Vec2 a;
			a.x = 5;
			a.y = 10;
			struct Vec2 b = a;
			return 0;
		}
		"""
		ir_prog = compile_to_ir(source)
		body = get_func_body(ir_prog)
		loads = [i for i in body if isinstance(i, IRLoad)]
		stores = [i for i in body if isinstance(i, IRStore)]
		# 2 stores for init + 2 load/store pairs for copy = at least 2 loads, 4 stores
		assert len(loads) >= 2
		assert len(stores) >= 4


class TestNestedStructCopy:
	"""Test nested struct copy."""

	def test_nested_struct_copy(self):
		source = """
		struct Inner { int a; int b; };
		struct Outer { struct Inner inner; int c; };
		int main() {
			struct Outer x;
			struct Outer y;
			x.inner.a = 1;
			x.inner.b = 2;
			x.c = 3;
			y = x;
			return 0;
		}
		"""
		ir_prog = compile_to_ir(source)
		body = get_func_body(ir_prog)
		loads = [i for i in body if isinstance(i, IRLoad)]
		stores = [i for i in body if isinstance(i, IRStore)]
		# Copy of Outer has 3 fields (inner.a, inner.b, c) -> 3 load/store pairs
		# Plus 3 stores for initialization
		assert len(loads) >= 3, f"Expected at least 3 loads for nested copy, got {len(loads)}"
		assert len(stores) >= 6, f"Expected at least 6 stores, got {len(stores)}"

	def test_nested_struct_member_copy(self):
		"""Test copying a nested struct member: y.inner = x.inner."""
		source = """
		struct Inner { int a; int b; };
		struct Outer { struct Inner inner; int c; };
		int main() {
			struct Outer x;
			struct Outer y;
			x.inner.a = 10;
			x.inner.b = 20;
			y.inner = x.inner;
			return 0;
		}
		"""
		ir_prog = compile_to_ir(source)
		body = get_func_body(ir_prog)
		loads = [i for i in body if isinstance(i, IRLoad)]
		stores = [i for i in body if isinstance(i, IRStore)]
		# 2 stores for init + 2 load/store pairs for inner copy
		assert len(loads) >= 2
		assert len(stores) >= 4


class TestStructWithArrayMembers:
	"""Test copying structs that contain array members."""

	def test_struct_with_int_array(self):
		source = """
		struct Data { int arr[3]; int val; };
		int main() {
			struct Data a;
			struct Data b;
			a.val = 42;
			b = a;
			return 0;
		}
		"""
		ir_prog = compile_to_ir(source)
		body = get_func_body(ir_prog)
		loads = [i for i in body if isinstance(i, IRLoad)]
		stores = [i for i in body if isinstance(i, IRStore)]
		# arr[3] requires 3 load/store pairs + val requires 1 = 4 total
		assert len(loads) >= 4, f"Expected at least 4 loads for array+scalar copy, got {len(loads)}"
		assert len(stores) >= 5, f"Expected at least 5 stores (1 init + 4 copy), got {len(stores)}"


class TestUnionCopy:
	"""Test union copy via word-sized memory copy."""

	def test_simple_union_copy(self):
		source = """
		union Value { int i; char c; };
		int main() {
			union Value a;
			union Value b;
			a.i = 42;
			b = a;
			return 0;
		}
		"""
		ir_prog = compile_to_ir(source)
		body = get_func_body(ir_prog)
		loads = [i for i in body if isinstance(i, IRLoad)]
		stores = [i for i in body if isinstance(i, IRStore)]
		# Union of size 4: one 4-byte load/store
		assert len(loads) >= 1
		assert len(stores) >= 2  # 1 for init + 1 for copy

	def test_union_init_from_variable(self):
		source = """
		union Data { int x; float f; };
		int main() {
			union Data a;
			a.x = 100;
			union Data b = a;
			return 0;
		}
		"""
		ir_prog = compile_to_ir(source)
		body = get_func_body(ir_prog)
		loads = [i for i in body if isinstance(i, IRLoad)]
		stores = [i for i in body if isinstance(i, IRStore)]
		assert len(loads) >= 1
		assert len(stores) >= 2


class TestCopyAfterModification:
	"""Test that copy after modification captures the modified values."""

	def test_copy_after_field_modification(self):
		source = """
		struct Point { int x; int y; };
		int main() {
			struct Point a;
			struct Point b;
			a.x = 1;
			a.y = 2;
			b = a;
			a.x = 100;
			b = a;
			return 0;
		}
		"""
		ir_prog = compile_to_ir(source)
		body = get_func_body(ir_prog)
		loads = [i for i in body if isinstance(i, IRLoad)]
		stores = [i for i in body if isinstance(i, IRStore)]
		# Two struct copies (2 fields each) = 4 load/store pairs
		# Plus 3 field stores (a.x=1, a.y=2, a.x=100)
		assert len(loads) >= 4
		assert len(stores) >= 7


class TestStructCopyIRStructure:
	"""Verify the IR structure of struct copy uses correct patterns."""

	def test_copy_uses_offset_arithmetic(self):
		"""Verify that struct copy emits address arithmetic with field offsets."""
		source = """
		struct S { int a; int b; };
		int main() {
			struct S x;
			struct S y;
			y = x;
			return 0;
		}
		"""
		ir_prog = compile_to_ir(source)
		body = get_func_body(ir_prog)
		# Should have IRBinOp with "+" for computing field addresses
		add_ops = [
			i for i in body
			if isinstance(i, IRBinOp) and i.op == "+"
		]
		# At least 4 adds: src+0, dst+0, src+4, dst+4 for a two-field struct
		assert len(add_ops) >= 4

	def test_copy_load_store_types_match_fields(self):
		"""Verify loads/stores in copy use correct IR types for each field."""
		source = """
		struct Mixed { char c; int i; };
		int main() {
			struct Mixed a;
			struct Mixed b;
			b = a;
			return 0;
		}
		"""
		ir_prog = compile_to_ir(source)
		body = get_func_body(ir_prog)
		# Find load/store pairs from the copy
		copy_loads = [i for i in body if isinstance(i, IRLoad)]
		copy_stores = [i for i in body if isinstance(i, IRStore)]
		# Should have one CHAR load/store and one INT load/store
		load_types = {ld.ir_type for ld in copy_loads}
		store_types = {s.ir_type for s in copy_stores}
		assert IRType.CHAR in load_types, "Expected CHAR load for char field"
		assert IRType.INT in load_types or IRType.INT in store_types, "Expected INT for int field"


class TestEdgeCases:
	"""Edge cases for struct copy."""

	def test_single_field_struct_copy(self):
		source = """
		struct Single { int val; };
		int main() {
			struct Single a;
			struct Single b;
			a.val = 42;
			b = a;
			return 0;
		}
		"""
		ir_prog = compile_to_ir(source)
		body = get_func_body(ir_prog)
		loads = [i for i in body if isinstance(i, IRLoad)]
		assert len(loads) >= 1

	def test_three_field_struct_copy(self):
		source = """
		struct Vec3 { int x; int y; int z; };
		int main() {
			struct Vec3 a;
			struct Vec3 b;
			a.x = 1;
			a.y = 2;
			a.z = 3;
			b = a;
			return 0;
		}
		"""
		ir_prog = compile_to_ir(source)
		body = get_func_body(ir_prog)
		loads = [i for i in body if isinstance(i, IRLoad)]
		stores = [i for i in body if isinstance(i, IRStore)]
		assert len(loads) >= 3
		assert len(stores) >= 6

	def test_multiple_assignments(self):
		"""Test chained struct assignments: c = b; b = a;"""
		source = """
		struct Point { int x; int y; };
		int main() {
			struct Point a;
			struct Point b;
			struct Point c;
			a.x = 1;
			a.y = 2;
			b = a;
			c = b;
			return 0;
		}
		"""
		ir_prog = compile_to_ir(source)
		body = get_func_body(ir_prog)
		loads = [i for i in body if isinstance(i, IRLoad)]
		# Two copies, 2 fields each = 4 loads minimum
		assert len(loads) >= 4
