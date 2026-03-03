"""Tests for x86-64 code generator."""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRAlloc,
	IRBinOp,
	IRCall,
	IRCondJump,
	IRConst,
	IRCopy,
	IRFunction,
	IRGlobalRef,
	IRGlobalVar,
	IRInstruction,
	IRJump,
	IRLabelInstr,
	IRLoad,
	IRParam,
	IRProgram,
	IRReturn,
	IRStringData,
	IRStore,
	IRTemp,
	IRType,
	IRUnaryOp,
)


def _gen(func: IRFunction) -> str:
	"""Helper: generate assembly for a single-function program."""
	return CodeGenerator().generate(IRProgram([func]))


# ---------------------------------------------------------------------------
# Smoke / import
# ---------------------------------------------------------------------------


class TestImport:
	def test_import(self) -> None:
		from compiler.codegen import CodeGenerator as CG  # noqa: F811

		assert CG is not None

	def test_empty_program(self) -> None:
		asm = CodeGenerator().generate(IRProgram())
		assert ".section .text" in asm


# ---------------------------------------------------------------------------
# Function prologue / epilogue
# ---------------------------------------------------------------------------


class TestPrologueEpilogue:
	def test_prologue_present(self) -> None:
		func = IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)
		asm = _gen(func)
		assert "pushq %rbp" in asm
		assert "movq %rsp, %rbp" in asm

	def test_epilogue_present(self) -> None:
		func = IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)
		asm = _gen(func)
		assert "movq %rbp, %rsp" in asm
		assert "popq %rbp" in asm
		assert "ret" in asm

	def test_globl_directive(self) -> None:
		func = IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)
		asm = _gen(func)
		assert ".globl main" in asm
		assert "\nmain:" in asm

	def test_stack_frame_allocated(self) -> None:
		"""When temps are used, subq $N, %rsp appears in prologue."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("t0"), IRConst(42)),
			IRReturn(IRTemp("t0")),
		]
		func = IRFunction("f", [], body, IRType.INT)
		asm = _gen(func)
		assert "subq $" in asm

	def test_no_frame_when_no_temps(self) -> None:
		"""A function with no temps should not allocate a frame beyond prologue."""
		func = IRFunction("noop", [], [IRReturn()], IRType.VOID)
		asm = _gen(func)
		lines = asm.split("\n")
		# No subq for frame allocation between pushq %rbp and first instruction
		prologue_subs = [line for line in lines if "subq" in line and "%rsp" in line]
		assert len(prologue_subs) == 0


# ---------------------------------------------------------------------------
# IRCopy
# ---------------------------------------------------------------------------


class TestCopy:
	def test_copy_const(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("t0"), IRConst(42)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "movq $42" in asm

	def test_copy_temp_to_temp(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("a"), IRConst(1)),
			IRCopy(IRTemp("b"), IRTemp("a")),
			IRReturn(IRTemp("b")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "movq $1" in asm
		# b should be loaded from a's stack slot
		assert "(%rbp)" in asm


# ---------------------------------------------------------------------------
# IRBinOp
# ---------------------------------------------------------------------------


class TestBinOp:
	def test_add(self) -> None:
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRConst(3), "+", IRConst(4)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "addq %rcx, %rax" in asm

	def test_sub(self) -> None:
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRConst(10), "-", IRConst(3)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "subq %rcx, %rax" in asm

	def test_mul(self) -> None:
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRConst(5), "*", IRConst(6)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "imulq %rcx, %rax" in asm

	def test_div(self) -> None:
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRConst(10), "/", IRConst(3)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "cqto" in asm
		assert "idivq %rcx" in asm

	def test_mod(self) -> None:
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRConst(10), "%", IRConst(3)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "cqto" in asm
		assert "idivq %rcx" in asm
		assert "movq %rdx, %rax" in asm

	def test_comparison_less(self) -> None:
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRConst(1), "<", IRConst(2)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "cmpq %rcx, %rax" in asm
		assert "setl %al" in asm
		assert "movzbq %al, %rax" in asm

	def test_comparison_equal(self) -> None:
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRConst(5), "==", IRConst(5)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "sete %al" in asm

	def test_comparison_not_equal(self) -> None:
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRConst(1), "!=", IRConst(2)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "setne %al" in asm

	def test_comparison_greater(self) -> None:
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRConst(5), ">", IRConst(3)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "setg %al" in asm

	def test_comparison_le(self) -> None:
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRConst(5), "<=", IRConst(5)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "setle %al" in asm

	def test_comparison_ge(self) -> None:
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRConst(5), ">=", IRConst(3)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "setge %al" in asm

	def test_bitwise_and(self) -> None:
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRConst(0xFF), "&", IRConst(0x0F)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "andq %rcx, %rax" in asm

	def test_bitwise_or(self) -> None:
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRConst(0xF0), "|", IRConst(0x0F)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "orq %rcx, %rax" in asm

	def test_bitwise_xor(self) -> None:
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRConst(0xFF), "^", IRConst(0x0F)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "xorq %rcx, %rax" in asm

	def test_shift_left(self) -> None:
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRConst(1), "<<", IRConst(4)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "salq %cl, %rax" in asm

	def test_shift_right(self) -> None:
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRConst(16), ">>", IRConst(2)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "sarq %cl, %rax" in asm

	def test_binop_with_temps(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("a"), IRConst(10)),
			IRCopy(IRTemp("b"), IRConst(20)),
			IRBinOp(IRTemp("c"), IRTemp("a"), "+", IRTemp("b")),
			IRReturn(IRTemp("c")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "addq %rcx, %rax" in asm


# ---------------------------------------------------------------------------
# IRUnaryOp
# ---------------------------------------------------------------------------


class TestUnaryOp:
	def test_negate(self) -> None:
		body: list[IRInstruction] = [
			IRUnaryOp(IRTemp("t0"), "-", IRConst(5)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "negq %rax" in asm

	def test_bitwise_not(self) -> None:
		body: list[IRInstruction] = [
			IRUnaryOp(IRTemp("t0"), "~", IRConst(0xFF)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "notq %rax" in asm

	def test_logical_not(self) -> None:
		body: list[IRInstruction] = [
			IRUnaryOp(IRTemp("t0"), "!", IRConst(1)),
			IRReturn(IRTemp("t0")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "cmpq $0, %rax" in asm
		assert "sete %al" in asm
		assert "movzbq %al, %rax" in asm


# ---------------------------------------------------------------------------
# IRLoad / IRStore
# ---------------------------------------------------------------------------


class TestLoadStore:
	def test_load(self) -> None:
		body: list[IRInstruction] = [
			IRAlloc(IRTemp("ptr"), 8),
			IRLoad(IRTemp("val"), IRTemp("ptr")),
			IRReturn(IRTemp("val")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "movq (%rax), %rax" in asm

	def test_store(self) -> None:
		body: list[IRInstruction] = [
			IRAlloc(IRTemp("ptr"), 8),
			IRStore(IRTemp("ptr"), IRConst(42)),
			IRReturn(),
		]
		asm = _gen(IRFunction("f", [], body, IRType.VOID))
		assert "movq %rcx, (%rax)" in asm

	def test_store_and_load_roundtrip(self) -> None:
		body: list[IRInstruction] = [
			IRAlloc(IRTemp("p"), 8),
			IRStore(IRTemp("p"), IRConst(99)),
			IRLoad(IRTemp("v"), IRTemp("p")),
			IRReturn(IRTemp("v")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "movq %rcx, (%rax)" in asm
		assert "movq (%rax), %rax" in asm


# ---------------------------------------------------------------------------
# IRLabelInstr / IRJump / IRCondJump
# ---------------------------------------------------------------------------


class TestControlFlow:
	def test_label(self) -> None:
		body: list[IRInstruction] = [
			IRLabelInstr("L0"),
			IRReturn(),
		]
		asm = _gen(IRFunction("f", [], body, IRType.VOID))
		assert "\nL0:" in asm

	def test_jump(self) -> None:
		body: list[IRInstruction] = [
			IRJump("L_end"),
			IRLabelInstr("L_end"),
			IRReturn(),
		]
		asm = _gen(IRFunction("f", [], body, IRType.VOID))
		assert "jmp L_end" in asm

	def test_condjump(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("cond"), IRConst(1)),
			IRCondJump(IRTemp("cond"), "L_true", "L_false"),
			IRLabelInstr("L_true"),
			IRReturn(IRConst(1)),
			IRLabelInstr("L_false"),
			IRReturn(IRConst(0)),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "cmpq $0, %rax" in asm
		assert "jne L_true" in asm
		assert "jmp L_false" in asm

	def test_loop(self) -> None:
		"""Simple while-loop IR: while (i < 10) i = i + 1; return i;"""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("i"), IRConst(0)),
			IRLabelInstr("loop"),
			IRBinOp(IRTemp("cond"), IRTemp("i"), "<", IRConst(10)),
			IRCondJump(IRTemp("cond"), "body", "end"),
			IRLabelInstr("body"),
			IRBinOp(IRTemp("i"), IRTemp("i"), "+", IRConst(1)),
			IRJump("loop"),
			IRLabelInstr("end"),
			IRReturn(IRTemp("i")),
		]
		asm = _gen(IRFunction("count", [], body, IRType.INT))
		assert "loop:" in asm
		assert "jne body" in asm
		assert "jmp loop" in asm
		assert "end:" in asm


# ---------------------------------------------------------------------------
# IRCall
# ---------------------------------------------------------------------------


class TestCall:
	def test_simple_call(self) -> None:
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "foo", []),
			IRReturn(IRTemp("r")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "call foo" in asm

	def test_call_with_args(self) -> None:
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "add", [IRConst(1), IRConst(2)]),
			IRReturn(IRTemp("r")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "movq $1, %rdi" in asm
		assert "movq $2, %rsi" in asm
		assert "call add" in asm

	def test_call_six_args(self) -> None:
		"""All 6 register args should be used."""
		args = [IRConst(i) for i in range(6)]
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "sixargs", args),
			IRReturn(IRTemp("r")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "movq $0, %rdi" in asm
		assert "movq $1, %rsi" in asm
		assert "movq $2, %rdx" in asm
		assert "movq $3, %rcx" in asm
		assert "movq $4, %r8" in asm
		assert "movq $5, %r9" in asm
		assert "call sixargs" in asm

	def test_call_seven_args_pushes_stack(self) -> None:
		"""7th arg goes on the stack."""
		args = [IRConst(i) for i in range(7)]
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "manyargs", args),
			IRReturn(IRTemp("r")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		# 7th arg pushed on stack
		assert "pushq %rax" in asm
		# 1 stack arg (odd) => alignment padding
		assert "subq $8, %rsp" in asm
		assert "call manyargs" in asm
		# Cleanup: 1 arg * 8 + 8 padding = 16
		assert "addq $16, %rsp" in asm

	def test_call_eight_args(self) -> None:
		"""8 args: 6 in regs, 2 on stack (even, no padding)."""
		args = [IRConst(i) for i in range(8)]
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "eightargs", args),
			IRReturn(IRTemp("r")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "call eightargs" in asm
		# 2 stack args (even) => cleanup 16 bytes, no padding sub
		assert "addq $16, %rsp" in asm

	def test_call_no_dest(self) -> None:
		"""void call (no return value captured)."""
		body: list[IRInstruction] = [
			IRCall(None, "exit", [IRConst(0)]),
		]
		asm = _gen(IRFunction("f", [], body, IRType.VOID))
		assert "call exit" in asm

	def test_call_result_stored(self) -> None:
		"""Return value from rax stored to dest stack slot."""
		body: list[IRInstruction] = [
			IRCall(IRTemp("x"), "getval", []),
			IRReturn(IRTemp("x")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "call getval" in asm
		# After call, rax is stored to x's stack slot
		lines = asm.split("\n")
		call_idx = next(i for i, line in enumerate(lines) if "call getval" in line)
		# The store should be after the call
		post_call = "\n".join(lines[call_idx + 1 :])
		assert "movq %rax," in post_call


# ---------------------------------------------------------------------------
# IRReturn
# ---------------------------------------------------------------------------


class TestReturn:
	def test_return_const(self) -> None:
		func = IRFunction("f", [], [IRReturn(IRConst(42))], IRType.INT)
		asm = _gen(func)
		assert "movq $42, %rax" in asm
		assert "ret" in asm

	def test_return_temp(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("r"), IRConst(7)),
			IRReturn(IRTemp("r")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "ret" in asm

	def test_return_void(self) -> None:
		func = IRFunction("f", [], [IRReturn()], IRType.VOID)
		asm = _gen(func)
		assert "ret" in asm
		# No movq $..., %rax before epilogue
		lines = asm.split("\n")
		ret_idx = next(i for i, line in enumerate(lines) if "ret" in line)
		# The line before ret should be popq %rbp (epilogue), not a movq into %rax
		assert "popq %rbp" in lines[ret_idx - 1]


# ---------------------------------------------------------------------------
# IRAlloc
# ---------------------------------------------------------------------------


class TestAlloc:
	def test_alloc_emits_sub(self) -> None:
		body: list[IRInstruction] = [
			IRAlloc(IRTemp("buf"), 32),
			IRReturn(),
		]
		asm = _gen(IRFunction("f", [], body, IRType.VOID))
		assert "subq $32, %rsp" in asm

	def test_alloc_stores_rsp(self) -> None:
		body: list[IRInstruction] = [
			IRAlloc(IRTemp("buf"), 16),
			IRReturn(),
		]
		asm = _gen(IRFunction("f", [], body, IRType.VOID))
		assert "movq %rsp, %rax" in asm

	def test_alloc_alignment(self) -> None:
		"""Alloc of 5 bytes rounds up to 16."""
		body: list[IRInstruction] = [
			IRAlloc(IRTemp("buf"), 5),
			IRReturn(),
		]
		asm = _gen(IRFunction("f", [], body, IRType.VOID))
		assert "subq $16, %rsp" in asm


# ---------------------------------------------------------------------------
# IRParam (no-op in codegen)
# ---------------------------------------------------------------------------


class TestParam:
	def test_param_is_noop(self) -> None:
		"""IRParam instructions should not crash or emit anything special."""
		body: list[IRInstruction] = [
			IRParam(IRConst(1)),
			IRParam(IRConst(2)),
			IRCall(IRTemp("x"), "add", [IRConst(1), IRConst(2)]),
			IRReturn(IRTemp("x")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "call add" in asm


# ---------------------------------------------------------------------------
# Function parameters (ABI)
# ---------------------------------------------------------------------------


class TestParameters:
	def test_single_param(self) -> None:
		"""int identity(int x) { return x; }"""
		body: list[IRInstruction] = [IRReturn(IRTemp("x"))]
		func = IRFunction("identity", [IRTemp("x")], body, IRType.INT)
		asm = _gen(func)
		# x received in %rdi, stored to stack
		assert "movq %rdi," in asm

	def test_two_params(self) -> None:
		"""int add(int a, int b) { return a + b; }"""
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRTemp("a"), "+", IRTemp("b")),
			IRReturn(IRTemp("t0")),
		]
		func = IRFunction("add", [IRTemp("a"), IRTemp("b")], body, IRType.INT)
		asm = _gen(func)
		assert "movq %rdi," in asm
		assert "movq %rsi," in asm

	def test_six_params(self) -> None:
		"""All six register-passed params."""
		params = [IRTemp(f"p{i}") for i in range(6)]
		body: list[IRInstruction] = [IRReturn(IRTemp("p0"))]
		func = IRFunction("f", params, body, IRType.INT)
		asm = _gen(func)
		assert "movq %rdi," in asm
		assert "movq %rsi," in asm
		assert "movq %rdx," in asm
		assert "movq %rcx," in asm
		assert "movq %r8," in asm
		assert "movq %r9," in asm

	def test_seven_params_stack(self) -> None:
		"""7th param comes from the stack (above saved rbp + ret addr)."""
		params = [IRTemp(f"p{i}") for i in range(7)]
		body: list[IRInstruction] = [IRReturn(IRTemp("p6"))]
		func = IRFunction("f", params, body, IRType.INT)
		asm = _gen(func)
		# p6 is loaded from 16(%rbp) (7th param, first stack param)
		assert "16(%rbp)" in asm


# ---------------------------------------------------------------------------
# Multiple functions
# ---------------------------------------------------------------------------


class TestMultipleFunctions:
	def test_two_functions(self) -> None:
		f1 = IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)
		f2 = IRFunction("helper", [], [IRReturn(IRConst(1))], IRType.INT)
		asm = CodeGenerator().generate(IRProgram([f1, f2]))
		assert ".globl main" in asm
		assert ".globl helper" in asm
		assert "\nmain:" in asm
		assert "\nhelper:" in asm


# ---------------------------------------------------------------------------
# Integration: realistic IR programs
# ---------------------------------------------------------------------------


class TestIntegration:
	def test_add_function(self) -> None:
		"""int add(int a, int b) { return a + b; }"""
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRTemp("a"), "+", IRTemp("b")),
			IRReturn(IRTemp("t0")),
		]
		func = IRFunction("add", [IRTemp("a"), IRTemp("b")], body, IRType.INT)
		asm = _gen(func)
		# Prologue
		assert "pushq %rbp" in asm
		assert "movq %rsp, %rbp" in asm
		# Param storage
		assert "movq %rdi," in asm
		assert "movq %rsi," in asm
		# Add
		assert "addq %rcx, %rax" in asm
		# Return
		assert "ret" in asm

	def test_factorial_loop(self) -> None:
		"""
		int factorial(int n) {
		    int result = 1;
		    while (n > 1) { result = result * n; n = n - 1; }
		    return result;
		}
		"""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("result"), IRConst(1)),
			IRLabelInstr("loop"),
			IRBinOp(IRTemp("cond"), IRTemp("n"), ">", IRConst(1)),
			IRCondJump(IRTemp("cond"), "body", "done"),
			IRLabelInstr("body"),
			IRBinOp(IRTemp("result"), IRTemp("result"), "*", IRTemp("n")),
			IRBinOp(IRTemp("n"), IRTemp("n"), "-", IRConst(1)),
			IRJump("loop"),
			IRLabelInstr("done"),
			IRReturn(IRTemp("result")),
		]
		func = IRFunction("factorial", [IRTemp("n")], body, IRType.INT)
		asm = _gen(func)
		assert ".globl factorial" in asm
		assert "loop:" in asm
		assert "body:" in asm
		assert "done:" in asm
		assert "imulq %rcx, %rax" in asm
		assert "subq %rcx, %rax" in asm
		assert "jmp loop" in asm
		assert "ret" in asm

	def test_pointer_operations(self) -> None:
		"""Alloc, store, load roundtrip."""
		body: list[IRInstruction] = [
			IRAlloc(IRTemp("p"), 8),
			IRStore(IRTemp("p"), IRConst(42)),
			IRLoad(IRTemp("val"), IRTemp("p")),
			IRReturn(IRTemp("val")),
		]
		func = IRFunction("ptr_test", [], body, IRType.INT)
		asm = _gen(func)
		assert "subq $16, %rsp" in asm  # alloc 8 rounds to 16
		assert "movq %rsp, %rax" in asm
		assert "movq %rcx, (%rax)" in asm
		assert "movq (%rax), %rax" in asm
		assert "ret" in asm

	def test_call_with_return(self) -> None:
		"""x = add(1, 2); return x;"""
		body: list[IRInstruction] = [
			IRCall(IRTemp("x"), "add", [IRConst(1), IRConst(2)]),
			IRReturn(IRTemp("x")),
		]
		func = IRFunction("main", [], body, IRType.INT)
		asm = _gen(func)
		assert "movq $1, %rdi" in asm
		assert "movq $2, %rsi" in asm
		assert "call add" in asm
		assert "ret" in asm

	def test_if_else(self) -> None:
		"""
		if (x > 0) return 1; else return -1;
		"""
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("cond"), IRTemp("x"), ">", IRConst(0)),
			IRCondJump(IRTemp("cond"), "then", "else"),
			IRLabelInstr("then"),
			IRReturn(IRConst(1)),
			IRLabelInstr("else"),
			IRReturn(IRConst(-1)),
		]
		func = IRFunction("sign", [IRTemp("x")], body, IRType.INT)
		asm = _gen(func)
		assert "setg %al" in asm
		assert "jne then" in asm
		assert "jmp else" in asm
		assert "then:" in asm
		assert "else:" in asm
		assert "movq $1, %rax" in asm
		assert "movq $-1, %rax" in asm


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestArrayCodegen:
	def test_array_element_read(self) -> None:
		"""Simulate: arr[2] read via pointer arithmetic + load."""
		body: list[IRInstruction] = [
			IRAlloc(IRTemp("arr"), 40),  # int arr[10]
			IRBinOp(IRTemp("offset"), IRConst(2), "*", IRConst(4)),
			IRBinOp(IRTemp("addr"), IRTemp("arr"), "+", IRTemp("offset")),
			IRLoad(IRTemp("val"), IRTemp("addr")),
			IRReturn(IRTemp("val")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		# Should contain alloc, mul for offset, add for addr, load, return
		assert "subq $" in asm  # alloc
		assert "imulq %rcx, %rax" in asm  # offset = 2 * 4
		assert "addq %rcx, %rax" in asm  # addr = arr + offset
		assert "movq (%rax), %rax" in asm  # load
		assert "ret" in asm

	def test_array_element_write(self) -> None:
		"""Simulate: arr[0] = 42 via pointer arithmetic + store."""
		body: list[IRInstruction] = [
			IRAlloc(IRTemp("arr"), 40),
			IRBinOp(IRTemp("offset"), IRConst(0), "*", IRConst(4)),
			IRBinOp(IRTemp("addr"), IRTemp("arr"), "+", IRTemp("offset")),
			IRStore(IRTemp("addr"), IRConst(42)),
			IRReturn(),
		]
		asm = _gen(IRFunction("f", [], body, IRType.VOID))
		assert "movq %rcx, (%rax)" in asm  # store
		assert "ret" in asm

	def test_array_loop_access(self) -> None:
		"""Simulate loop: for i in 0..n, read arr[i]."""
		body: list[IRInstruction] = [
			IRAlloc(IRTemp("arr"), 40),
			IRCopy(IRTemp("i"), IRConst(0)),
			IRLabelInstr("loop"),
			IRBinOp(IRTemp("cond"), IRTemp("i"), "<", IRConst(10)),
			IRCondJump(IRTemp("cond"), "body", "end"),
			IRLabelInstr("body"),
			IRBinOp(IRTemp("off"), IRTemp("i"), "*", IRConst(4)),
			IRBinOp(IRTemp("addr"), IRTemp("arr"), "+", IRTemp("off")),
			IRLoad(IRTemp("val"), IRTemp("addr")),
			IRBinOp(IRTemp("i"), IRTemp("i"), "+", IRConst(1)),
			IRJump("loop"),
			IRLabelInstr("end"),
			IRReturn(IRTemp("val")),
		]
		asm = _gen(IRFunction("f", [], body, IRType.INT))
		assert "loop:" in asm
		assert "body:" in asm
		assert "end:" in asm
		assert "movq (%rax), %rax" in asm
		assert "ret" in asm


class TestErrors:
	def test_unknown_binop_raises(self) -> None:
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("t0"), IRConst(1), "???", IRConst(2)),
			IRReturn(IRTemp("t0")),
		]
		func = IRFunction("f", [], body, IRType.INT)
		try:
			_gen(func)
			assert False, "Should have raised ValueError"
		except ValueError as e:
			assert "Unknown binary operator" in str(e)

	def test_unknown_unaryop_raises(self) -> None:
		body: list[IRInstruction] = [
			IRUnaryOp(IRTemp("t0"), "@", IRConst(1)),
			IRReturn(IRTemp("t0")),
		]
		func = IRFunction("f", [], body, IRType.INT)
		try:
			_gen(func)
			assert False, "Should have raised ValueError"
		except ValueError as e:
			assert "Unknown unary operator" in str(e)


# ---------------------------------------------------------------------------
# Global variables (.data / .bss)
# ---------------------------------------------------------------------------


class TestGlobalVars:
	def test_initialized_global_data_section(self) -> None:
		"""Initialized global should appear in .data with .quad directive."""
		prog = IRProgram(
			functions=[IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)],
			globals=[IRGlobalVar("counter", IRType.INT, initializer=42)],
		)
		asm = CodeGenerator().generate(prog)
		assert ".section .data" in asm
		assert ".globl counter" in asm
		assert "counter:" in asm
		assert ".quad 42" in asm

	def test_uninitialized_global_bss_section(self) -> None:
		"""Uninitialized global should appear in .bss with .zero directive."""
		prog = IRProgram(
			functions=[IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)],
			globals=[IRGlobalVar("buffer", IRType.INT)],
		)
		asm = CodeGenerator().generate(prog)
		assert ".section .bss" in asm
		assert ".globl buffer" in asm
		assert "buffer:" in asm
		assert ".zero 8" in asm

	def test_char_global_uses_byte(self) -> None:
		"""Char-typed initialized global emits .byte instead of .quad."""
		prog = IRProgram(
			functions=[IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)],
			globals=[IRGlobalVar("ch", IRType.CHAR, initializer=65)],
		)
		asm = CodeGenerator().generate(prog)
		assert ".byte 65" in asm

	def test_char_global_bss_uses_zero_1(self) -> None:
		"""Uninitialized char global reserves 1 byte in .bss."""
		prog = IRProgram(
			functions=[IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)],
			globals=[IRGlobalVar("ch", IRType.CHAR)],
		)
		asm = CodeGenerator().generate(prog)
		assert ".zero 1" in asm

	def test_mixed_initialized_and_uninitialized(self) -> None:
		"""Program with both initialized and uninitialized globals."""
		prog = IRProgram(
			functions=[IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)],
			globals=[
				IRGlobalVar("x", IRType.INT, initializer=10),
				IRGlobalVar("y", IRType.INT),
			],
		)
		asm = CodeGenerator().generate(prog)
		assert ".section .data" in asm
		assert ".section .bss" in asm
		assert ".quad 10" in asm
		assert ".zero 8" in asm

	def test_no_data_section_when_no_initialized_globals(self) -> None:
		"""No .data section emitted when there are no initialized globals."""
		prog = IRProgram(
			functions=[IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)],
			globals=[IRGlobalVar("x", IRType.INT)],
		)
		asm = CodeGenerator().generate(prog)
		assert ".section .data" not in asm
		assert ".section .bss" in asm

	def test_no_bss_section_when_no_uninitialized_globals(self) -> None:
		"""No .bss section emitted when all globals are initialized."""
		prog = IRProgram(
			functions=[IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)],
			globals=[IRGlobalVar("x", IRType.INT, initializer=5)],
		)
		asm = CodeGenerator().generate(prog)
		assert ".section .data" in asm
		assert ".section .bss" not in asm


# ---------------------------------------------------------------------------
# String literals (.rodata)
# ---------------------------------------------------------------------------


class TestStringData:
	def test_string_literal_rodata(self) -> None:
		"""String data should appear in .rodata with .asciz directive."""
		prog = IRProgram(
			functions=[IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)],
			string_data=[IRStringData(".LC0", "hello")],
		)
		asm = CodeGenerator().generate(prog)
		assert ".section .rodata" in asm
		assert ".LC0:" in asm
		assert '.asciz "hello"' in asm

	def test_multiple_strings(self) -> None:
		"""Multiple string literals each get their own label."""
		prog = IRProgram(
			functions=[IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)],
			string_data=[
				IRStringData(".LC0", "hello"),
				IRStringData(".LC1", "world"),
			],
		)
		asm = CodeGenerator().generate(prog)
		assert ".LC0:" in asm
		assert '.asciz "hello"' in asm
		assert ".LC1:" in asm
		assert '.asciz "world"' in asm

	def test_no_rodata_section_when_no_strings(self) -> None:
		"""No .rodata section emitted when there are no string literals."""
		prog = IRProgram(
			functions=[IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)],
		)
		asm = CodeGenerator().generate(prog)
		assert ".section .rodata" not in asm


# ---------------------------------------------------------------------------
# IRGlobalRef (loading addresses of globals/strings)
# ---------------------------------------------------------------------------


class TestGlobalRef:
	def test_global_ref_emits_leaq(self) -> None:
		"""IRGlobalRef should emit leaq for RIP-relative address loading."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("addr"), IRGlobalRef("counter")),
			IRReturn(IRTemp("addr")),
		]
		prog = IRProgram(
			functions=[IRFunction("main", [], body, IRType.INT)],
			globals=[IRGlobalVar("counter", IRType.INT, initializer=0)],
		)
		asm = CodeGenerator().generate(prog)
		assert "leaq counter(%rip)" in asm

	def test_load_global_value(self) -> None:
		"""Load value from a global: get address via IRGlobalRef, then IRLoad."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("addr"), IRGlobalRef("x")),
			IRLoad(IRTemp("val"), IRTemp("addr")),
			IRReturn(IRTemp("val")),
		]
		prog = IRProgram(
			functions=[IRFunction("main", [], body, IRType.INT)],
			globals=[IRGlobalVar("x", IRType.INT, initializer=42)],
		)
		asm = CodeGenerator().generate(prog)
		assert "leaq x(%rip)" in asm
		assert "movq (%rax), %rax" in asm

	def test_store_to_global(self) -> None:
		"""Store value to a global: get address via IRGlobalRef, then IRStore."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("addr"), IRGlobalRef("x")),
			IRStore(IRTemp("addr"), IRConst(99)),
			IRReturn(),
		]
		prog = IRProgram(
			functions=[IRFunction("main", [], body, IRType.VOID)],
			globals=[IRGlobalVar("x", IRType.INT, initializer=0)],
		)
		asm = CodeGenerator().generate(prog)
		assert "leaq x(%rip)" in asm
		assert "movq %rcx, (%rax)" in asm

	def test_string_ref_by_label(self) -> None:
		"""Reference a string literal label with IRGlobalRef."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("str_ptr"), IRGlobalRef(".LC0")),
			IRCall(None, "puts", [IRTemp("str_ptr")]),
			IRReturn(IRConst(0)),
		]
		prog = IRProgram(
			functions=[IRFunction("main", [], body, IRType.INT)],
			string_data=[IRStringData(".LC0", "hello world")],
		)
		asm = CodeGenerator().generate(prog)
		assert ".section .rodata" in asm
		assert '.asciz "hello world"' in asm
		assert "leaq .LC0(%rip)" in asm
		assert "call puts" in asm

	def test_full_program_globals_and_strings(self) -> None:
		"""Full program with globals, strings, and functions using them."""
		body: list[IRInstruction] = [
			# Load string address and call printf
			IRCopy(IRTemp("fmt"), IRGlobalRef(".LC0")),
			# Load global value
			IRCopy(IRTemp("g_addr"), IRGlobalRef("count")),
			IRLoad(IRTemp("g_val"), IRTemp("g_addr")),
			# Call printf(fmt, g_val)
			IRCall(None, "printf", [IRTemp("fmt"), IRTemp("g_val")]),
			IRReturn(IRConst(0)),
		]
		prog = IRProgram(
			functions=[IRFunction("main", [], body, IRType.INT)],
			globals=[
				IRGlobalVar("count", IRType.INT, initializer=5),
				IRGlobalVar("result", IRType.INT),
			],
			string_data=[IRStringData(".LC0", "count = %d\\n")],
		)
		asm = CodeGenerator().generate(prog)
		# .data section with initialized global
		assert ".section .data" in asm
		assert ".quad 5" in asm
		# .rodata section with string
		assert ".section .rodata" in asm
		assert '.asciz "count = %d\\n"' in asm
		# .bss section with uninitialized global
		assert ".section .bss" in asm
		assert ".zero 8" in asm
		# .text section with code
		assert ".section .text" in asm
		assert "leaq .LC0(%rip)" in asm
		assert "leaq count(%rip)" in asm
		assert "call printf" in asm

	def test_section_order(self) -> None:
		"""Sections should appear in order: .data, .rodata, .bss, .text."""
		prog = IRProgram(
			functions=[IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)],
			globals=[
				IRGlobalVar("x", IRType.INT, initializer=1),
				IRGlobalVar("y", IRType.INT),
			],
			string_data=[IRStringData(".LC0", "test")],
		)
		asm = CodeGenerator().generate(prog)
		data_pos = asm.index(".section .data")
		rodata_pos = asm.index(".section .rodata")
		bss_pos = asm.index(".section .bss")
		text_pos = asm.index(".section .text")
		assert data_pos < rodata_pos < bss_pos < text_pos
