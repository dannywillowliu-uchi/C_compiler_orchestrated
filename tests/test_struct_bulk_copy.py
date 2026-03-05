"""Tests for struct bulk copy codegen using rep movsb / unrolled moves."""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRAlloc,
	IRBulkCopy,
	IRConst,
	IRFunction,
	IRProgram,
	IRReturn,
	IRTemp,
	IRType,
)
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.preprocessor import Preprocessor
from compiler.semantic import SemanticAnalyzer


def compile_source(source: str) -> str:
	"""Run C source through the full compiler pipeline, returning assembly."""
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	ast = Parser(tokens).parse()
	errors = SemanticAnalyzer().analyze(ast)
	assert not errors, f"Semantic errors: {errors}"
	ir = IRGenerator().generate(ast)
	return CodeGenerator().generate(ir)


def compile_to_ir(source: str) -> IRProgram:
	"""Run C source through frontend, returning IR."""
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	ast = Parser(tokens).parse()
	errors = SemanticAnalyzer().analyze(ast)
	assert not errors, f"Semantic errors: {errors}"
	return IRGenerator().generate(ast)


class TestIRBulkCopyEmission:
	"""Verify the IR generator emits IRBulkCopy for struct operations."""

	def test_struct_assignment_emits_bulk_copy(self) -> None:
		source = """
		struct Point { int x; int y; };
		int main() {
			struct Point a;
			a.x = 1;
			a.y = 2;
			struct Point b;
			b = a;
			return b.x;
		}
		"""
		ir = compile_to_ir(source)
		func = [f for f in ir.functions if f.name == "main"][0]
		bulk_copies = [i for i in func.body if isinstance(i, IRBulkCopy)]
		assert len(bulk_copies) >= 1, "Expected at least one IRBulkCopy for struct assignment"
		assert bulk_copies[0].size == 8, f"Expected size 8 for Point (2 ints), got {bulk_copies[0].size}"

	def test_struct_init_from_var_emits_bulk_copy(self) -> None:
		source = """
		struct Vec3 { int x; int y; int z; };
		int main() {
			struct Vec3 a;
			a.x = 10;
			a.y = 20;
			a.z = 30;
			struct Vec3 b = a;
			return b.y;
		}
		"""
		ir = compile_to_ir(source)
		func = [f for f in ir.functions if f.name == "main"][0]
		bulk_copies = [i for i in func.body if isinstance(i, IRBulkCopy)]
		assert len(bulk_copies) >= 1, "Expected IRBulkCopy for struct initialization"
		assert bulk_copies[0].size == 12, f"Expected size 12 for Vec3, got {bulk_copies[0].size}"

	def test_nested_struct_emits_bulk_copy(self) -> None:
		source = """
		struct Inner { int a; int b; };
		struct Outer { struct Inner inner; int c; };
		int main() {
			struct Outer x;
			x.inner.a = 1;
			x.inner.b = 2;
			x.c = 3;
			struct Outer y = x;
			return y.c;
		}
		"""
		ir = compile_to_ir(source)
		func = [f for f in ir.functions if f.name == "main"][0]
		bulk_copies = [i for i in func.body if isinstance(i, IRBulkCopy)]
		assert len(bulk_copies) >= 1, "Expected IRBulkCopy for nested struct copy"


class TestBulkCopyCodegenRepMovsb:
	"""Verify rep movsb is used for larger struct copies (>= 16 bytes)."""

	def test_large_struct_uses_rep_movsb(self) -> None:
		source = """
		struct Big { int a; int b; int c; int d; int e; };
		int main() {
			struct Big x;
			x.a = 1;
			struct Big y = x;
			return y.a;
		}
		"""
		asm = compile_source(source)
		assert "rep movsb" in asm, f"Expected rep movsb for 20-byte struct copy, got:\n{asm}"

	def test_16_byte_struct_uses_rep_movsb(self) -> None:
		source = """
		struct S16 { long a; long b; };
		int main() {
			struct S16 x;
			x.a = 100;
			x.b = 200;
			struct S16 y = x;
			return 0;
		}
		"""
		asm = compile_source(source)
		assert "rep movsb" in asm, "Expected rep movsb for 16-byte struct"


class TestBulkCopyCodegenUnrolled:
	"""Verify small struct copies use unrolled mov sequences."""

	def test_small_struct_uses_unrolled_mov(self) -> None:
		source = """
		struct Small { int x; int y; };
		int main() {
			struct Small a;
			a.x = 5;
			a.y = 10;
			struct Small b = a;
			return b.x;
		}
		"""
		asm = compile_source(source)
		# 8-byte struct should NOT use rep movsb, should use direct movq
		assert "rep movsb" not in asm, "Small struct (8 bytes) should use unrolled copy, not rep movsb"

	def test_12_byte_struct_unrolled(self) -> None:
		source = """
		struct S12 { int a; int b; int c; };
		int main() {
			struct S12 x;
			x.a = 1;
			x.b = 2;
			x.c = 3;
			struct S12 y = x;
			return y.b;
		}
		"""
		asm = compile_source(source)
		# 12 bytes: should use unrolled (movq + movl), not rep movsb
		assert "rep movsb" not in asm, "12-byte struct should use unrolled copy"


class TestBulkCopyDirectIR:
	"""Test codegen directly with hand-crafted IRBulkCopy instructions."""

	def test_codegen_bulk_copy_large(self) -> None:
		"""32-byte bulk copy should produce rep movsb."""
		src = IRTemp("src")
		dst = IRTemp("dst")
		program = IRProgram(functions=[
			IRFunction(
				name="test_bulk",
				params=[],
				body=[
					IRAlloc(dest=src, size=32),
					IRAlloc(dest=dst, size=32),
					IRBulkCopy(dest_addr=dst, src_addr=src, size=32),
					IRReturn(value=IRConst(0)),
				],
				return_type=IRType.INT,
			)
		])
		asm = CodeGenerator().generate(program)
		assert "rep movsb" in asm
		assert "$32" in asm  # size loaded into %rcx

	def test_codegen_bulk_copy_small(self) -> None:
		"""8-byte bulk copy should produce unrolled movq."""
		src = IRTemp("src")
		dst = IRTemp("dst")
		program = IRProgram(functions=[
			IRFunction(
				name="test_bulk_small",
				params=[],
				body=[
					IRAlloc(dest=src, size=8),
					IRAlloc(dest=dst, size=8),
					IRBulkCopy(dest_addr=dst, src_addr=src, size=8),
					IRReturn(value=IRConst(0)),
				],
				return_type=IRType.INT,
			)
		])
		asm = CodeGenerator().generate(program)
		assert "rep movsb" not in asm
		assert "movq" in asm

	def test_codegen_bulk_copy_zero(self) -> None:
		"""Zero-size bulk copy should produce no copy instructions."""
		src = IRTemp("src")
		dst = IRTemp("dst")
		program = IRProgram(functions=[
			IRFunction(
				name="test_zero",
				params=[],
				body=[
					IRAlloc(dest=src, size=8),
					IRAlloc(dest=dst, size=8),
					IRBulkCopy(dest_addr=dst, src_addr=src, size=0),
					IRReturn(value=IRConst(0)),
				],
				return_type=IRType.INT,
			)
		])
		asm = CodeGenerator().generate(program)
		assert "rep movsb" not in asm

	def test_codegen_bulk_copy_odd_size(self) -> None:
		"""15-byte bulk copy should produce unrolled movq + movl + movw + movb."""
		src = IRTemp("src")
		dst = IRTemp("dst")
		program = IRProgram(functions=[
			IRFunction(
				name="test_odd",
				params=[],
				body=[
					IRAlloc(dest=src, size=15),
					IRAlloc(dest=dst, size=15),
					IRBulkCopy(dest_addr=dst, src_addr=src, size=15),
					IRReturn(value=IRConst(0)),
				],
				return_type=IRType.INT,
			)
		])
		asm = CodeGenerator().generate(program)
		# 15 = 8 + 4 + 2 + 1 -> movq, movl, movw, movb
		assert "rep movsb" not in asm
		# Should have a mix of move sizes
		assert "movq" in asm
		assert "movl" in asm or "movw" in asm or "movb" in asm


class TestStructAssignmentEndToEnd:
	"""End-to-end tests for struct assignment correctness."""

	def test_struct_member_access_after_copy(self) -> None:
		source = """
		struct Point { int x; int y; };
		int main() {
			struct Point a;
			a.x = 42;
			a.y = 99;
			struct Point b = a;
			return b.x;
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_struct_assignment_operator(self) -> None:
		source = """
		struct Pair { int first; int second; };
		int main() {
			struct Pair a;
			a.first = 10;
			a.second = 20;
			struct Pair b;
			b.first = 0;
			b.second = 0;
			b = a;
			return b.second;
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm

	def test_nested_struct_assignment(self) -> None:
		source = """
		struct Inner { int val; };
		struct Outer { struct Inner in1; struct Inner in2; };
		int main() {
			struct Outer a;
			a.in1.val = 100;
			a.in2.val = 200;
			struct Outer b = a;
			return b.in2.val;
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm

	def test_large_struct_copy(self) -> None:
		source = """
		struct Big { int a; int b; int c; int d; int e; int f; };
		int main() {
			struct Big x;
			x.a = 1;
			x.b = 2;
			x.c = 3;
			x.d = 4;
			x.e = 5;
			x.f = 6;
			struct Big y = x;
			return y.f;
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm
		assert "rep movsb" in asm

	def test_member_access_nested_struct_copy(self) -> None:
		"""Assignment to a nested struct member should use bulk copy."""
		source = """
		struct V2 { int x; int y; };
		struct Container { struct V2 pos; int id; };
		int main() {
			struct V2 p;
			p.x = 5;
			p.y = 10;
			struct Container c;
			c.pos = p;
			c.id = 42;
			return c.id;
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm


class TestIRBulkCopyStr:
	"""Test IRBulkCopy string representation."""

	def test_str(self) -> None:
		instr = IRBulkCopy(
			dest_addr=IRTemp("dst"),
			src_addr=IRTemp("src"),
			size=24,
		)
		assert str(instr) == "bulkcopy dst, src, 24"
