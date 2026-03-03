"""Tests for codegen hardening: implicit epilogue and fallback returns."""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRBinOp,
	IRCall,
	IRCondJump,
	IRConst,
	IRCopy,
	IRFunction,
	IRInstruction,
	IRJump,
	IRLabelInstr,
	IRProgram,
	IRReturn,
	IRTemp,
	IRType,
)


def _gen(func: IRFunction) -> str:
	"""Helper: generate assembly for a single-function program."""
	return CodeGenerator().generate(IRProgram([func]))


def _asm_lines(asm: str) -> list[str]:
	"""Split assembly into stripped lines."""
	return [line.strip() for line in asm.split("\n") if line.strip()]


# ---------------------------------------------------------------------------
# Void function with no return
# ---------------------------------------------------------------------------


class TestVoidNoReturn:
	def test_implicit_epilogue_emitted(self) -> None:
		"""A void function with no return should get an implicit epilogue."""
		body: list[IRInstruction] = [
			IRCall(None, "do_something", []),
		]
		func = IRFunction("setup", [], body, IRType.VOID)
		asm = _gen(func)
		# Must contain epilogue instructions
		assert "movq %rbp, %rsp" in asm
		assert "popq %rbp" in asm
		assert "ret" in asm

	def test_no_rax_set_for_void(self) -> None:
		"""Implicit epilogue for void should NOT set %rax to 0."""
		body: list[IRInstruction] = [
			IRCall(None, "do_something", []),
		]
		func = IRFunction("setup", [], body, IRType.VOID)
		asm = _gen(func)
		lines = _asm_lines(asm)
		# Find the implicit epilogue at the end (after the call)
		call_idx = next(i for i, line in enumerate(lines) if "call do_something" in line)
		epilogue_lines = lines[call_idx + 1:]
		# Should not contain movq $0, %rax
		assert not any("movq $0, %rax" in line for line in epilogue_lines)

	def test_empty_void_function(self) -> None:
		"""A void function with empty body still gets an epilogue."""
		func = IRFunction("noop", [], [], IRType.VOID)
		asm = _gen(func)
		assert "movq %rbp, %rsp" in asm
		assert "popq %rbp" in asm
		assert "ret" in asm


# ---------------------------------------------------------------------------
# Void function with early return
# ---------------------------------------------------------------------------


class TestVoidEarlyReturn:
	def test_explicit_return_not_doubled(self) -> None:
		"""A void function ending with IRReturn should NOT get a second epilogue."""
		body: list[IRInstruction] = [
			IRCall(None, "do_something", []),
			IRReturn(),
		]
		func = IRFunction("setup", [], body, IRType.VOID)
		asm = _gen(func)
		# Count 'ret' instructions -- should be exactly 1
		lines = _asm_lines(asm)
		ret_count = sum(1 for line in lines if line == "ret")
		assert ret_count == 1

	def test_early_return_in_branch(self) -> None:
		"""Void function with return inside an if-branch but not at the end."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("cond"), IRConst(1)),
			IRCondJump(IRTemp("cond"), "L_early", "L_skip"),
			IRLabelInstr("L_early"),
			IRReturn(),
			IRLabelInstr("L_skip"),
			IRCall(None, "fallthrough_work", []),
			# No return here -- needs implicit epilogue
		]
		func = IRFunction("maybe_return", [], body, IRType.VOID)
		asm = _gen(func)
		lines = _asm_lines(asm)
		# Should have 2 ret instructions: one from the explicit return,
		# one from the implicit epilogue
		ret_count = sum(1 for line in lines if line == "ret")
		assert ret_count == 2


# ---------------------------------------------------------------------------
# Void function with if/else where only one branch returns
# ---------------------------------------------------------------------------


class TestVoidPartialReturn:
	def test_one_branch_returns_other_falls_through(self) -> None:
		"""
		void f(int x) {
		    if (x) return;
		    do_work();
		}
		The else-branch has no return, so an implicit epilogue is needed.
		"""
		body: list[IRInstruction] = [
			IRCondJump(IRTemp("x"), "L_ret", "L_work"),
			IRLabelInstr("L_ret"),
			IRReturn(),
			IRLabelInstr("L_work"),
			IRCall(None, "do_work", []),
			# falls through -- last instr is IRCall, not IRReturn
		]
		func = IRFunction("f", [IRTemp("x")], body, IRType.VOID)
		asm = _gen(func)
		lines = _asm_lines(asm)
		# 2 ret: explicit + implicit
		ret_count = sum(1 for line in lines if line == "ret")
		assert ret_count == 2
		# Implicit epilogue comes at the very end
		assert lines[-1] == "ret"
		assert lines[-2] == "popq %rbp"
		assert lines[-3] == "movq %rbp, %rsp"

	def test_both_branches_return_no_extra_epilogue(self) -> None:
		"""
		void f(int x) {
		    if (x) return;
		    else return;
		}
		Last instruction is IRReturn so no implicit epilogue needed.
		"""
		body: list[IRInstruction] = [
			IRCondJump(IRTemp("x"), "L_then", "L_else"),
			IRLabelInstr("L_then"),
			IRReturn(),
			IRLabelInstr("L_else"),
			IRReturn(),
		]
		func = IRFunction("f", [IRTemp("x")], body, IRType.VOID)
		asm = _gen(func)
		lines = _asm_lines(asm)
		ret_count = sum(1 for line in lines if line == "ret")
		assert ret_count == 2  # exactly the two explicit returns


# ---------------------------------------------------------------------------
# Non-void function structure verification
# ---------------------------------------------------------------------------


class TestNonVoidFallback:
	def test_nonvoid_fallback_returns_zero(self) -> None:
		"""A non-void function with no return should emit movq $0, %rax + epilogue."""
		body: list[IRInstruction] = [
			IRCall(None, "setup", []),
		]
		func = IRFunction("get_value", [], body, IRType.INT)
		asm = _gen(func)
		lines = _asm_lines(asm)
		# Should have fallback return 0
		assert "movq $0, %rax" in asm
		# Epilogue at end
		assert lines[-1] == "ret"
		assert lines[-2] == "popq %rbp"
		assert lines[-3] == "movq %rbp, %rsp"
		assert lines[-4] == "movq $0, %rax"

	def test_nonvoid_with_explicit_return_no_fallback(self) -> None:
		"""Non-void function ending with IRReturn doesn't get fallback."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("t0"), IRConst(42)),
			IRReturn(IRTemp("t0")),
		]
		func = IRFunction("get_value", [], body, IRType.INT)
		asm = _gen(func)
		lines = _asm_lines(asm)
		ret_count = sum(1 for line in lines if line == "ret")
		assert ret_count == 1

	def test_nonvoid_partial_return_gets_fallback(self) -> None:
		"""
		int f(int x) {
		    if (x) return 1;
		    // missing return -- should get return 0 fallback
		}
		"""
		body: list[IRInstruction] = [
			IRCondJump(IRTemp("x"), "L_ret", "L_end"),
			IRLabelInstr("L_ret"),
			IRReturn(IRConst(1)),
			IRLabelInstr("L_end"),
			# No return -- last instr is a label
		]
		func = IRFunction("f", [IRTemp("x")], body, IRType.INT)
		asm = _gen(func)
		lines = _asm_lines(asm)
		# Should have the explicit return and the implicit fallback
		ret_count = sum(1 for line in lines if line == "ret")
		assert ret_count == 2
		# Last 4 lines should be the fallback epilogue
		assert lines[-4] == "movq $0, %rax"
		assert lines[-3] == "movq %rbp, %rsp"
		assert lines[-2] == "popq %rbp"
		assert lines[-1] == "ret"

	def test_nonvoid_function_with_loop_no_guaranteed_return(self) -> None:
		"""
		int f() {
		    int i = 0;
		    while (i < 10) {
		        if (i == 5) return i;
		        i = i + 1;
		    }
		    // falls through -- needs implicit return 0
		}
		"""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("i"), IRConst(0)),
			IRLabelInstr("loop"),
			IRBinOp(IRTemp("cond"), IRTemp("i"), "<", IRConst(10)),
			IRCondJump(IRTemp("cond"), "body", "end"),
			IRLabelInstr("body"),
			IRBinOp(IRTemp("eq"), IRTemp("i"), "==", IRConst(5)),
			IRCondJump(IRTemp("eq"), "found", "cont"),
			IRLabelInstr("found"),
			IRReturn(IRTemp("i")),
			IRLabelInstr("cont"),
			IRBinOp(IRTemp("i"), IRTemp("i"), "+", IRConst(1)),
			IRJump("loop"),
			IRLabelInstr("end"),
			# No return after loop -- needs fallback
		]
		func = IRFunction("f", [], body, IRType.INT)
		asm = _gen(func)
		lines = _asm_lines(asm)
		# Explicit return inside loop + implicit fallback
		ret_count = sum(1 for line in lines if line == "ret")
		assert ret_count == 2
		# Last lines are the fallback
		assert lines[-4] == "movq $0, %rax"
		assert lines[-1] == "ret"
