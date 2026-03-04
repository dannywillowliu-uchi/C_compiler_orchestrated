"""Tests for width-correct memory access and variadic ABI in codegen."""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRCall,
	IRConst,
	IRFunction,
	IRGlobalRef,
	IRGlobalVar,
	IRLoad,
	IRProgram,
	IRReturn,
	IRStore,
	IRStringData,
	IRTemp,
	IRType,
)


def _gen(func: IRFunction, globals_list: list[IRGlobalVar] | None = None, strings: list[IRStringData] | None = None) -> str:
	"""Helper: generate assembly for a single-function program."""
	return CodeGenerator().generate(IRProgram([func], globals_list or [], strings or []))


# ---------------------------------------------------------------------------
# Load widths
# ---------------------------------------------------------------------------


class TestLoadWidths:
	def test_load_char_uses_movzbl(self) -> None:
		body = [
			IRLoad(IRTemp("t1"), IRTemp("t0"), ir_type=IRType.CHAR),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.POINTER])
		asm = _gen(func)
		assert "movzbl (%rax), %eax" in asm

	def test_load_int_uses_movl_movslq(self) -> None:
		body = [
			IRLoad(IRTemp("t1"), IRTemp("t0"), ir_type=IRType.INT),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.POINTER])
		asm = _gen(func)
		assert "movl (%rax), %eax" in asm
		assert "movslq %eax, %rax" in asm

	def test_load_pointer_uses_movq(self) -> None:
		body = [
			IRLoad(IRTemp("t1"), IRTemp("t0"), ir_type=IRType.POINTER),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.POINTER])
		asm = _gen(func)
		assert "movq (%rax), %rax" in asm

	def test_load_char_no_movq_deref(self) -> None:
		"""CHAR load should NOT use movq for the dereference."""
		body = [
			IRLoad(IRTemp("t1"), IRTemp("t0"), ir_type=IRType.CHAR),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.POINTER])
		asm = _gen(func)
		lines = asm.split("\n")
		# Find the load instruction - should not be movq (%rax), %rax
		deref_lines = [ln.strip() for ln in lines if "(%rax)," in ln and "movq" in ln and "%rax" in ln.split(",")[-1]]
		assert len(deref_lines) == 0, f"Unexpected movq dereference for CHAR load: {deref_lines}"


# ---------------------------------------------------------------------------
# Store widths
# ---------------------------------------------------------------------------


class TestStoreWidths:
	def test_store_char_uses_movb(self) -> None:
		body = [
			IRStore(IRTemp("t0"), IRConst(65), ir_type=IRType.CHAR),
			IRReturn(None),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.VOID, [IRType.POINTER])
		asm = _gen(func)
		assert "movb %cl, (%rax)" in asm

	def test_store_int_uses_movl(self) -> None:
		body = [
			IRStore(IRTemp("t0"), IRConst(42), ir_type=IRType.INT),
			IRReturn(None),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.VOID, [IRType.POINTER])
		asm = _gen(func)
		assert "movl %ecx, (%rax)" in asm

	def test_store_pointer_uses_movq(self) -> None:
		body = [
			IRStore(IRTemp("t0"), IRConst(0), ir_type=IRType.POINTER),
			IRReturn(None),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.VOID, [IRType.POINTER])
		asm = _gen(func)
		assert "movq %rcx, (%rax)" in asm

	def test_store_char_no_movq(self) -> None:
		"""CHAR store should use movb, not movq for the store."""
		body = [
			IRStore(IRTemp("t0"), IRConst(65), ir_type=IRType.CHAR),
			IRReturn(None),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.VOID, [IRType.POINTER])
		asm = _gen(func)
		assert "movq %rcx, (%rax)" not in asm


# ---------------------------------------------------------------------------
# Variadic ABI (%al = SSE arg count)
# ---------------------------------------------------------------------------


class TestVariadicABI:
	def test_call_sets_al_zero_no_float_args(self) -> None:
		"""Calls with no float args should set %al to 0."""
		body = [
			IRCall(IRTemp("t0"), "printf", [IRGlobalRef(".LC0"), IRConst(42)], [IRType.POINTER, IRType.INT]),
			IRReturn(IRConst(0)),
		]
		func = IRFunction("main", [], body, IRType.INT)
		asm = _gen(func)
		assert "movb $0, %al" in asm

	def test_call_sets_al_one_float_arg(self) -> None:
		"""Calls with one float arg should set %al to 1."""
		body = [
			IRCall(
				IRTemp("t0"), "printf",
				[IRGlobalRef(".LC0"), IRTemp("t1")],
				[IRType.POINTER, IRType.DOUBLE],
			),
			IRReturn(IRConst(0)),
		]
		func = IRFunction("main", [IRTemp("t1")], body, IRType.INT, [IRType.DOUBLE])
		asm = _gen(func)
		assert "movb $1, %al" in asm

	def test_al_set_before_call(self) -> None:
		"""movb $N, %al must appear before the call instruction."""
		body = [
			IRCall(IRTemp("t0"), "puts", [IRGlobalRef(".LC0")], [IRType.POINTER]),
			IRReturn(IRConst(0)),
		]
		func = IRFunction("main", [], body, IRType.INT)
		asm = _gen(func)
		lines = asm.split("\n")
		al_idx = None
		call_idx = None
		for i, line in enumerate(lines):
			if "movb $" in line and "%al" in line:
				al_idx = i
			if "\tcall " in line and al_idx is not None:
				call_idx = i
				break
		assert al_idx is not None, "movb $N, %al not found"
		assert call_idx is not None, "call not found after movb"
		assert al_idx < call_idx, "movb %al must come before call"


# ---------------------------------------------------------------------------
# Global .data / .bss directives
# ---------------------------------------------------------------------------


class TestGlobalDirectives:
	def test_bss_char_uses_zero_1(self) -> None:
		prog = IRProgram([], [IRGlobalVar("g_char", IRType.CHAR)])
		asm = CodeGenerator().generate(prog)
		assert ".zero 1" in asm

	def test_bss_int_uses_zero_4(self) -> None:
		prog = IRProgram([], [IRGlobalVar("g_int", IRType.INT)])
		asm = CodeGenerator().generate(prog)
		assert ".zero 4" in asm

	def test_bss_pointer_uses_zero_8(self) -> None:
		prog = IRProgram([], [IRGlobalVar("g_ptr", IRType.POINTER)])
		asm = CodeGenerator().generate(prog)
		assert ".zero 8" in asm

	def test_data_char_uses_byte(self) -> None:
		prog = IRProgram([], [IRGlobalVar("g_char", IRType.CHAR, initializer=65)])
		asm = CodeGenerator().generate(prog)
		assert ".byte 65" in asm

	def test_data_int_uses_long(self) -> None:
		prog = IRProgram([], [IRGlobalVar("g_int", IRType.INT, initializer=42)])
		asm = CodeGenerator().generate(prog)
		assert ".long 42" in asm

	def test_data_pointer_uses_quad(self) -> None:
		prog = IRProgram([], [IRGlobalVar("g_ptr", IRType.POINTER, initializer=0)])
		asm = CodeGenerator().generate(prog)
		assert ".quad 0" in asm

	def test_data_int_no_quad(self) -> None:
		"""INT global should use .long, not .quad."""
		prog = IRProgram([], [IRGlobalVar("g_int", IRType.INT, initializer=42)])
		asm = CodeGenerator().generate(prog)
		# .quad should not appear for this INT global (no initializer_values)
		lines = [ln.strip() for ln in asm.split("\n")]
		# Find the section between g_int: and the next section
		in_gint = False
		for line in lines:
			if line == "g_int:":
				in_gint = True
				continue
			if in_gint:
				if line.startswith("."):
					break
				assert ".quad" not in line, f"INT global should not use .quad: {line}"


# ---------------------------------------------------------------------------
# Indirect call uses %r11 (not %rax which conflicts with %al)
# ---------------------------------------------------------------------------


class TestIndirectCall:
	def test_indirect_call_uses_r11(self) -> None:
		"""Indirect calls should use %r11 since %rax is used for %al."""
		body = [
			IRCall(
				IRTemp("t1"), "fn_ptr",
				[IRConst(1)], [IRType.INT],
				indirect=True, func_value=IRTemp("t0"),
			),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.POINTER])
		asm = _gen(func)
		assert "call *%r11" in asm
