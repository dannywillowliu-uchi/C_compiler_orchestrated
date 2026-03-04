"""Tests for correct .bss sizing of uninitialized global arrays and structs."""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRFunction,
	IRGlobalVar,
	IRProgram,
	IRReturn,
	IRConst,
	IRType,
)


def _gen_bss(globals_list: list[IRGlobalVar]) -> str:
	"""Generate assembly for a program with only global vars (no functions)."""
	prog = IRProgram(
		functions=[IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)],
		globals=globals_list,
	)
	return CodeGenerator().generate(prog)


class TestBssArraySizing:
	def test_int_array_100(self) -> None:
		"""int arr[100] should emit .zero 400."""
		g = IRGlobalVar(name="arr", ir_type=IRType.INT, total_size=400)
		asm = _gen_bss([g])
		assert ".zero 400" in asm

	def test_char_array_256(self) -> None:
		"""char buf[256] should emit .zero 256."""
		g = IRGlobalVar(name="buf", ir_type=IRType.CHAR, total_size=256)
		asm = _gen_bss([g])
		assert ".zero 256" in asm

	def test_short_array_50(self) -> None:
		"""short arr[50] should emit .zero 100."""
		g = IRGlobalVar(name="arr", ir_type=IRType.SHORT, total_size=100)
		asm = _gen_bss([g])
		assert ".zero 100" in asm

	def test_pointer_array_10(self) -> None:
		"""void* ptrs[10] should emit .zero 80."""
		g = IRGlobalVar(name="ptrs", ir_type=IRType.POINTER, total_size=80)
		asm = _gen_bss([g])
		assert ".zero 80" in asm


class TestBssStructSizing:
	def test_struct_24_bytes(self) -> None:
		"""A struct with total_size=24 should emit .zero 24."""
		g = IRGlobalVar(name="my_struct", ir_type=IRType.INT, total_size=24)
		asm = _gen_bss([g])
		assert ".zero 24" in asm

	def test_struct_128_bytes(self) -> None:
		"""A large struct should emit the full size."""
		g = IRGlobalVar(name="big_struct", ir_type=IRType.CHAR, total_size=128)
		asm = _gen_bss([g])
		assert ".zero 128" in asm


class TestBssMultiDimArray:
	def test_2d_int_array(self) -> None:
		"""int matrix[10][10] should emit .zero 400."""
		g = IRGlobalVar(name="matrix", ir_type=IRType.INT, total_size=400)
		asm = _gen_bss([g])
		assert ".zero 400" in asm

	def test_3d_char_array(self) -> None:
		"""char cube[4][4][4] should emit .zero 64."""
		g = IRGlobalVar(name="cube", ir_type=IRType.CHAR, total_size=64)
		asm = _gen_bss([g])
		assert ".zero 64" in asm


class TestBssScalarFallback:
	"""Scalar globals (total_size=0) should still use type-based sizes."""

	def test_scalar_int(self) -> None:
		g = IRGlobalVar(name="x", ir_type=IRType.INT)
		asm = _gen_bss([g])
		assert ".zero 4" in asm

	def test_scalar_char(self) -> None:
		g = IRGlobalVar(name="c", ir_type=IRType.CHAR)
		asm = _gen_bss([g])
		assert ".zero 1" in asm

	def test_scalar_short(self) -> None:
		g = IRGlobalVar(name="s", ir_type=IRType.SHORT)
		asm = _gen_bss([g])
		assert ".zero 2" in asm

	def test_scalar_long(self) -> None:
		g = IRGlobalVar(name="l", ir_type=IRType.LONG)
		asm = _gen_bss([g])
		assert ".zero 8" in asm

	def test_scalar_pointer(self) -> None:
		g = IRGlobalVar(name="p", ir_type=IRType.POINTER)
		asm = _gen_bss([g])
		assert ".zero 8" in asm

	def test_scalar_double(self) -> None:
		g = IRGlobalVar(name="d", ir_type=IRType.DOUBLE)
		asm = _gen_bss([g])
		assert ".zero 8" in asm


class TestBssMultipleGlobals:
	def test_mixed_globals(self) -> None:
		"""Multiple globals with different sizes in one program."""
		globals_list = [
			IRGlobalVar(name="arr", ir_type=IRType.INT, total_size=400),
			IRGlobalVar(name="scalar", ir_type=IRType.INT),
			IRGlobalVar(name="big", ir_type=IRType.CHAR, total_size=1024),
		]
		asm = _gen_bss(globals_list)
		assert ".zero 400" in asm
		assert ".zero 4" in asm
		assert ".zero 1024" in asm
