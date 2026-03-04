"""Tests for static local variables and global float/string/char initializers."""

import struct

from compiler.codegen import CodeGenerator
from compiler.ir import IRGlobalRef, IRGlobalVar, IRLoad, IRStore, IRType
from compiler.ir_gen import IRGenerator
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer


def _ir(src: str):
	prog = Parser.from_source(src).parse()
	SemanticAnalyzer().analyze(prog)
	return IRGenerator().generate(prog)


def _asm(src: str) -> str:
	ir_prog = _ir(src)
	return CodeGenerator().generate(ir_prog)


# ---------------------------------------------------------------------------
# Static local variables - IR generation
# ---------------------------------------------------------------------------


class TestStaticLocalIR:
	def test_static_local_emits_global(self) -> None:
		src = """
		int counter() {
			static int count = 0;
			count = count + 1;
			return count;
		}
		int main() { return counter(); }
		"""
		ir_prog = _ir(src)
		# Should have a global with mangled name counter.count
		global_names = [g.name for g in ir_prog.globals]
		assert "counter.count" in global_names

	def test_static_local_has_static_storage(self) -> None:
		src = """
		int f() {
			static int x = 5;
			return x;
		}
		int main() { return f(); }
		"""
		ir_prog = _ir(src)
		g = next(g for g in ir_prog.globals if g.name == "f.x")
		assert g.storage_class == "static"
		assert g.initializer == 5
		assert g.ir_type == IRType.INT

	def test_static_local_read_uses_global_ref(self) -> None:
		src = """
		int f() {
			static int x = 10;
			return x;
		}
		int main() { return f(); }
		"""
		ir_prog = _ir(src)
		func = next(f for f in ir_prog.functions if f.name == "f")
		loads = [
			i for i in func.body
			if isinstance(i, IRLoad) and isinstance(i.address, IRGlobalRef)
		]
		mangled_refs = [ld for ld in loads if ld.address.name == "f.x"]
		assert len(mangled_refs) >= 1

	def test_static_local_write_uses_store(self) -> None:
		src = """
		int f() {
			static int x = 0;
			x = x + 1;
			return x;
		}
		int main() { return f(); }
		"""
		ir_prog = _ir(src)
		func = next(f for f in ir_prog.functions if f.name == "f")
		stores = [
			i for i in func.body
			if isinstance(i, IRStore) and isinstance(i.address, IRGlobalRef)
			and i.address.name == "f.x"
		]
		assert len(stores) >= 1

	def test_static_local_not_on_stack(self) -> None:
		"""Static locals should NOT generate IRAlloc instructions."""
		src = """
		int f() {
			static int x = 0;
			return x;
		}
		int main() { return f(); }
		"""
		ir_prog = _ir(src)
		func = next(f for f in ir_prog.functions if f.name == "f")
		from compiler.ir import IRAlloc
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert len(allocs) == 0

	def test_static_local_with_initializer_value(self) -> None:
		src = """
		int f() {
			static int x = 42;
			return x;
		}
		int main() { return f(); }
		"""
		ir_prog = _ir(src)
		g = next(g for g in ir_prog.globals if g.name == "f.x")
		assert g.initializer == 42

	def test_static_local_no_initializer(self) -> None:
		src = """
		int f() {
			static int x;
			x = x + 1;
			return x;
		}
		int main() { return f(); }
		"""
		ir_prog = _ir(src)
		g = next(g for g in ir_prog.globals if g.name == "f.x")
		assert g.initializer is None


# ---------------------------------------------------------------------------
# Static local variables - codegen
# ---------------------------------------------------------------------------


class TestStaticLocalCodegen:
	def test_static_local_counter_in_data_section(self) -> None:
		src = """
		int counter() {
			static int count = 0;
			count = count + 1;
			return count;
		}
		int main() { return counter(); }
		"""
		asm = _asm(src)
		# Static local should be in data section (initialized to 0)
		assert "counter.count:" in asm
		# Should NOT have .globl for static variables
		assert ".globl counter.count" not in asm

	def test_static_local_with_initial_value_codegen(self) -> None:
		src = """
		int f() {
			static int x = 100;
			return x;
		}
		int main() { return f(); }
		"""
		asm = _asm(src)
		assert "f.x:" in asm
		assert ".long 100" in asm

	def test_static_local_uninitialized_in_bss(self) -> None:
		src = """
		int f() {
			static int x;
			return x;
		}
		int main() { return f(); }
		"""
		asm = _asm(src)
		assert "f.x:" in asm
		assert ".section .bss" in asm


# ---------------------------------------------------------------------------
# Global float/double initializers - IR generation
# ---------------------------------------------------------------------------


class TestGlobalFloatIR:
	def test_global_double_initializer(self) -> None:
		src = """
		double pi = 3.14159;
		int main() { return 0; }
		"""
		ir_prog = _ir(src)
		g = next(g for g in ir_prog.globals if g.name == "pi")
		assert isinstance(g, IRGlobalVar)
		assert g.ir_type == IRType.DOUBLE
		assert g.float_initializer == 3.14159

	def test_global_float_initializer(self) -> None:
		src = """
		float x = 2.5f;
		int main() { return 0; }
		"""
		ir_prog = _ir(src)
		g = next(g for g in ir_prog.globals if g.name == "x")
		assert g.ir_type == IRType.FLOAT
		assert g.float_initializer == 2.5

	def test_global_negative_double(self) -> None:
		src = """
		double neg = -1.5;
		int main() { return 0; }
		"""
		ir_prog = _ir(src)
		g = next(g for g in ir_prog.globals if g.name == "neg")
		assert g.float_initializer == -1.5


# ---------------------------------------------------------------------------
# Global float/double initializers - codegen
# ---------------------------------------------------------------------------


class TestGlobalFloatCodegen:
	def test_global_double_data_section(self) -> None:
		src = """
		double pi = 3.14;
		int main() { return 0; }
		"""
		asm = _asm(src)
		assert ".section .data" in asm
		assert "pi:" in asm
		# Should emit .quad with IEEE 754 bits for 3.14
		bits = struct.unpack("<Q", struct.pack("<d", 3.14))[0]
		assert f".quad {bits}" in asm

	def test_global_float_data_section(self) -> None:
		src = """
		float x = 2.5f;
		int main() { return 0; }
		"""
		asm = _asm(src)
		assert "x:" in asm
		bits = struct.unpack("<I", struct.pack("<f", 2.5))[0]
		assert f".long {bits}" in asm

	def test_global_negative_double_data_section(self) -> None:
		src = """
		double neg = -1.5;
		int main() { return 0; }
		"""
		asm = _asm(src)
		bits = struct.unpack("<Q", struct.pack("<d", -1.5))[0]
		assert f".quad {bits}" in asm


# ---------------------------------------------------------------------------
# Global char initializer
# ---------------------------------------------------------------------------


class TestGlobalCharIR:
	def test_global_char_initializer(self) -> None:
		src = """
		char c = 'A';
		int main() { return 0; }
		"""
		ir_prog = _ir(src)
		g = next(g for g in ir_prog.globals if g.name == "c")
		assert g.ir_type == IRType.CHAR
		assert g.initializer == ord("A")


class TestGlobalCharCodegen:
	def test_global_char_data_section(self) -> None:
		src = """
		char c = 'A';
		int main() { return 0; }
		"""
		asm = _asm(src)
		assert "c:" in asm
		assert f".byte {ord('A')}" in asm


# ---------------------------------------------------------------------------
# Global negative integer initializer
# ---------------------------------------------------------------------------


class TestGlobalNegativeInt:
	def test_global_negative_int_ir(self) -> None:
		src = """
		int x = -42;
		int main() { return 0; }
		"""
		ir_prog = _ir(src)
		g = next(g for g in ir_prog.globals if g.name == "x")
		assert g.initializer == -42

	def test_global_negative_int_codegen(self) -> None:
		src = """
		int x = -42;
		int main() { return 0; }
		"""
		asm = _asm(src)
		assert ".long -42" in asm


# ---------------------------------------------------------------------------
# Global string initializer
# ---------------------------------------------------------------------------


class TestGlobalStringIR:
	def test_global_string_pointer(self) -> None:
		src = """
		char* msg = "hello";
		int main() { return 0; }
		"""
		ir_prog = _ir(src)
		g = next(g for g in ir_prog.globals if g.name == "msg")
		assert g.ir_type == IRType.POINTER
		assert g.string_label is not None
		# The string data should also be emitted
		assert len(ir_prog.string_data) >= 1
		assert any(s.value == "hello" for s in ir_prog.string_data)


class TestGlobalStringCodegen:
	def test_global_string_pointer_asm(self) -> None:
		src = """
		char* msg = "hello";
		int main() { return 0; }
		"""
		asm = _asm(src)
		assert "msg:" in asm
		assert '.asciz "hello"' in asm
		# The global should reference the string label
		assert ".quad .str" in asm
