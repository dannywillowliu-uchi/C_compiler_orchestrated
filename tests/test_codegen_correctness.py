"""Tests for codegen width correctness: global array initializers and short stores/loads."""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRConst,
	IRFunction,
	IRGlobalVar,
	IRLoad,
	IRProgram,
	IRReturn,
	IRStore,
	IRTemp,
	IRType,
)


def _gen(func: IRFunction, globals_list: list[IRGlobalVar] | None = None) -> str:
	"""Helper: generate assembly for a single-function program."""
	return CodeGenerator().generate(IRProgram([func], globals_list or []))


# ---------------------------------------------------------------------------
# Global array initializer directives
# ---------------------------------------------------------------------------


class TestGlobalArrayInitializers:
	def test_int_array_uses_long(self) -> None:
		"""Global int array initializer should emit .long for each element."""
		g = IRGlobalVar("arr_int", IRType.INT, initializer_values=[10, 20, 30])
		prog = IRProgram([], [g])
		asm = CodeGenerator().generate(prog)
		assert ".long 10" in asm
		assert ".long 20" in asm
		assert ".long 30" in asm
		# Should not use .quad for int elements
		for line in asm.split("\n"):
			stripped = line.strip()
			if stripped.startswith(".quad") and any(v in stripped for v in ["10", "20", "30"]):
				raise AssertionError(f"INT array should not use .quad: {stripped}")

	def test_char_array_uses_byte(self) -> None:
		"""Global char array initializer should emit .byte for each element."""
		g = IRGlobalVar("arr_char", IRType.CHAR, initializer_values=[65, 66, 67])
		prog = IRProgram([], [g])
		asm = CodeGenerator().generate(prog)
		assert ".byte 65" in asm
		assert ".byte 66" in asm
		assert ".byte 67" in asm

	def test_short_array_uses_word(self) -> None:
		"""Global short array initializer should emit .word for each element."""
		g = IRGlobalVar("arr_short", IRType.SHORT, initializer_values=[100, 200, 300])
		prog = IRProgram([], [g])
		asm = CodeGenerator().generate(prog)
		assert ".word 100" in asm
		assert ".word 200" in asm
		assert ".word 300" in asm

	def test_long_array_uses_quad(self) -> None:
		"""Global long array initializer should emit .quad for each element."""
		g = IRGlobalVar("arr_long", IRType.LONG, initializer_values=[1000, 2000])
		prog = IRProgram([], [g])
		asm = CodeGenerator().generate(prog)
		assert ".quad 1000" in asm
		assert ".quad 2000" in asm

	def test_pointer_array_uses_quad(self) -> None:
		"""Global pointer array initializer should emit .quad for each element."""
		g = IRGlobalVar("arr_ptr", IRType.POINTER, initializer_values=[0, 0])
		prog = IRProgram([], [g])
		asm = CodeGenerator().generate(prog)
		assert ".quad 0" in asm


# ---------------------------------------------------------------------------
# Short store/load (movw / movswq)
# ---------------------------------------------------------------------------


class TestShortStoreLoad:
	def test_store_short_uses_movw(self) -> None:
		"""SHORT store should emit movw %cx, (%rax)."""
		body = [
			IRStore(IRTemp("t0"), IRConst(42), ir_type=IRType.SHORT),
			IRReturn(None),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.VOID, [IRType.POINTER])
		asm = _gen(func)
		assert "movw %cx, (%rax)" in asm

	def test_store_short_no_movl(self) -> None:
		"""SHORT store should NOT use movl (4-byte write)."""
		body = [
			IRStore(IRTemp("t0"), IRConst(42), ir_type=IRType.SHORT),
			IRReturn(None),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.VOID, [IRType.POINTER])
		asm = _gen(func)
		assert "movl %ecx, (%rax)" not in asm

	def test_store_short_no_movq(self) -> None:
		"""SHORT store should NOT use movq (8-byte write)."""
		body = [
			IRStore(IRTemp("t0"), IRConst(42), ir_type=IRType.SHORT),
			IRReturn(None),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.VOID, [IRType.POINTER])
		asm = _gen(func)
		assert "movq %rcx, (%rax)" not in asm

	def test_load_short_uses_movswq(self) -> None:
		"""SHORT load should emit movswq for sign-extended 16-bit read."""
		body = [
			IRLoad(IRTemp("t1"), IRTemp("t0"), ir_type=IRType.SHORT),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.POINTER])
		asm = _gen(func)
		assert "movswq (%rax), %rax" in asm

	def test_load_short_no_movq_deref(self) -> None:
		"""SHORT load should NOT use movq for the dereference."""
		body = [
			IRLoad(IRTemp("t1"), IRTemp("t0"), ir_type=IRType.SHORT),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.POINTER])
		asm = _gen(func)
		assert "movq (%rax), %rax" not in asm

	def test_short_store_load_roundtrip(self) -> None:
		"""Store a short value and load it back - verify both movw and movswq present."""
		body = [
			IRStore(IRTemp("t0"), IRConst(1234), ir_type=IRType.SHORT),
			IRLoad(IRTemp("t1"), IRTemp("t0"), ir_type=IRType.SHORT),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.POINTER])
		asm = _gen(func)
		assert "movw %cx, (%rax)" in asm
		assert "movswq (%rax), %rax" in asm


# ---------------------------------------------------------------------------
# BSS section for SHORT
# ---------------------------------------------------------------------------


class TestBssShort:
	def test_bss_short_uses_zero_2(self) -> None:
		"""Uninitialized SHORT global should use .zero 2."""
		prog = IRProgram([], [IRGlobalVar("g_short", IRType.SHORT)])
		asm = CodeGenerator().generate(prog)
		assert ".zero 2" in asm
