"""Edge-case tests for codegen correctness of control flow and ABI compliance."""

import re

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRBinOp,
	IRCall,
	IRCondJump,
	IRConst,
	IRCopy,
	IRFloatConst,
	IRFunction,
	IRInstruction,
	IRJump,
	IRLabelInstr,
	IRProgram,
	IRReturn,
	IRTemp,
	IRType,
)


def _gen(func: IRFunction, extra_funcs: list[IRFunction] | None = None) -> str:
	"""Helper: generate assembly for a program with one or more functions."""
	funcs = [func] + (extra_funcs or [])
	return CodeGenerator().generate(IRProgram(funcs))


def _gen_program(funcs: list[IRFunction]) -> str:
	return CodeGenerator().generate(IRProgram(funcs))


def _asm_lines(asm: str) -> list[str]:
	"""Return stripped, non-empty assembly lines."""
	return [line.strip() for line in asm.split("\n") if line.strip()]


# ---------------------------------------------------------------------------
# 1. Functions with many arguments (>6 int args to test stack passing)
# ---------------------------------------------------------------------------


class TestManyIntArgs:
	"""Test that functions with >6 integer args use stack passing correctly."""

	def test_7_int_params_reads_from_stack(self) -> None:
		"""7th integer param should be loaded from 16(%rbp) (first stack slot)."""
		params = [IRTemp(f"p{i}") for i in range(7)]
		param_types = [IRType.INT] * 7
		body: list[IRInstruction] = [IRReturn(IRTemp("p6"))]
		func = IRFunction("many_args", params, body, IRType.INT, param_types)
		asm = _gen(func)
		# First 6 params go to registers, 7th comes from stack at 16(%rbp)
		assert "16(%rbp)" in asm

	def test_8_int_params_uses_multiple_stack_slots(self) -> None:
		"""8th integer param should be at 24(%rbp)."""
		params = [IRTemp(f"p{i}") for i in range(8)]
		param_types = [IRType.INT] * 8
		body: list[IRInstruction] = [
			IRBinOp(IRTemp("sum"), IRTemp("p6"), "+", IRTemp("p7")),
			IRReturn(IRTemp("sum")),
		]
		func = IRFunction("eight_args", params, body, IRType.INT, param_types)
		asm = _gen(func)
		assert "16(%rbp)" in asm
		assert "24(%rbp)" in asm

	def test_calling_function_with_7_args_pushes_to_stack(self) -> None:
		"""When calling a function with 7 int args, the 7th should be pushed."""
		args = [IRConst(i) for i in range(7)]
		arg_types = [IRType.INT] * 7
		body: list[IRInstruction] = [
			IRCall(IRTemp("result"), "target_func", args, arg_types, IRType.INT),
			IRReturn(IRTemp("result")),
		]
		func = IRFunction("caller", [], body, IRType.INT)
		asm = _gen(func)
		# 7th arg should be pushed onto the stack
		assert "pushq" in asm
		# After call, stack cleanup should happen
		assert "addq $" in asm

	def test_calling_with_8_args_maintains_16byte_alignment(self) -> None:
		"""8 stack-overflow args (2 on stack) should maintain 16-byte alignment."""
		args = [IRConst(i) for i in range(8)]
		arg_types = [IRType.INT] * 8
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "target", args, arg_types, IRType.INT),
			IRReturn(IRTemp("r")),
		]
		func = IRFunction("caller8", [], body, IRType.INT)
		asm = _gen(func)
		# 2 stack args => even number => no padding needed
		# Both should be pushed
		pushq_count = asm.count("pushq %rax")
		assert pushq_count == 2

	def test_calling_with_9_args_pads_for_alignment(self) -> None:
		"""9 args (3 on stack, odd) should add padding for 16-byte alignment."""
		args = [IRConst(i) for i in range(9)]
		arg_types = [IRType.INT] * 9
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "target", args, arg_types, IRType.INT),
			IRReturn(IRTemp("r")),
		]
		func = IRFunction("caller9", [], body, IRType.INT)
		asm = _gen(func)
		# 3 stack args => odd => needs 8-byte padding
		assert "subq $8, %rsp" in asm


# ---------------------------------------------------------------------------
# 2. Mixed int/float arguments in correct ABI registers
# ---------------------------------------------------------------------------


class TestMixedIntFloatArgs:
	"""Test that int and float args go to the correct ABI registers."""

	def test_float_arg_uses_xmm0(self) -> None:
		"""A single float argument should be passed in %xmm0."""
		args = [IRFloatConst(1.0)]
		arg_types = [IRType.FLOAT]
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "sinf", args, arg_types, IRType.FLOAT),
			IRReturn(IRTemp("r")),
		]
		func = IRFunction("call_sinf", [], body, IRType.FLOAT)
		asm = _gen(func)
		assert "movss" in asm
		assert "%xmm0" in asm

	def test_int_then_float_args(self) -> None:
		"""int, float args should go to %rdi and %xmm0 respectively."""
		args = [IRConst(42), IRFloatConst(3.14)]
		arg_types = [IRType.INT, IRType.FLOAT]
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "mixed", args, arg_types, IRType.INT),
			IRReturn(IRTemp("r")),
		]
		func = IRFunction("caller_mixed", [], body, IRType.INT)
		asm = _gen(func)
		# Int arg should go to %rdi
		assert "%rdi" in asm
		# Float arg should go to %xmm0
		assert "%xmm0" in asm

	def test_float_then_int_args(self) -> None:
		"""float, int args: float to %xmm0, int to %rdi (independent counters)."""
		args = [IRFloatConst(2.0), IRConst(10)]
		arg_types = [IRType.FLOAT, IRType.INT]
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "mixed2", args, arg_types, IRType.INT),
			IRReturn(IRTemp("r")),
		]
		func = IRFunction("caller_fi", [], body, IRType.INT)
		asm = _gen(func)
		assert "%xmm0" in asm
		assert "%rdi" in asm

	def test_multiple_floats_use_sequential_xmm(self) -> None:
		"""Multiple float args should use %xmm0, %xmm1, %xmm2, etc."""
		args = [IRFloatConst(1.0), IRFloatConst(2.0), IRFloatConst(3.0)]
		arg_types = [IRType.DOUBLE, IRType.DOUBLE, IRType.DOUBLE]
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "triple_float", args, arg_types, IRType.DOUBLE),
			IRReturn(IRTemp("r")),
		]
		func = IRFunction("caller_3f", [], body, IRType.DOUBLE)
		func.return_type = IRType.DOUBLE
		asm = _gen(func)
		assert "%xmm0" in asm
		assert "%xmm1" in asm
		assert "%xmm2" in asm

	def test_sse_arg_count_in_al(self) -> None:
		"""The number of SSE register args must be passed in %al."""
		args = [IRFloatConst(1.0), IRFloatConst(2.0)]
		arg_types = [IRType.FLOAT, IRType.FLOAT]
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "two_floats", args, arg_types, IRType.FLOAT),
			IRReturn(IRTemp("r")),
		]
		func = IRFunction("caller_al", [], body, IRType.FLOAT)
		asm = _gen(func)
		# %al should contain the count of SSE args (2)
		assert "movb $2, %al" in asm

	def test_zero_floats_sets_al_to_zero(self) -> None:
		"""With no float args, %al should be set to 0."""
		args = [IRConst(1), IRConst(2)]
		arg_types = [IRType.INT, IRType.INT]
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "add", args, arg_types, IRType.INT),
			IRReturn(IRTemp("r")),
		]
		func = IRFunction("caller_no_float", [], body, IRType.INT)
		asm = _gen(func)
		assert "movb $0, %al" in asm

	def test_float_param_received_in_xmm(self) -> None:
		"""A function receiving a float param should read from %xmm0."""
		params = [IRTemp("f")]
		param_types = [IRType.FLOAT]
		body: list[IRInstruction] = [IRReturn(IRTemp("f"), ir_type=IRType.FLOAT)]
		func = IRFunction("recv_float", params, body, IRType.FLOAT, param_types)
		asm = _gen(func)
		# Float param should be moved from %xmm0 to its stack slot
		assert "movss %xmm0" in asm


# ---------------------------------------------------------------------------
# 3. Nested function calls as arguments: f(g(x), h(y))
# ---------------------------------------------------------------------------


class TestNestedCallArgs:
	"""Test codegen for nested function calls used as arguments."""

	def test_nested_call_as_arg(self) -> None:
		"""f(g(x)) should first call g, then pass result to f."""
		body: list[IRInstruction] = [
			IRCall(IRTemp("t0"), "g", [IRConst(5)], [IRType.INT], IRType.INT),
			IRCall(IRTemp("t1"), "f", [IRTemp("t0")], [IRType.INT], IRType.INT),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("main", [], body, IRType.INT)
		asm = _gen(func)
		lines = _asm_lines(asm)
		# g should be called before f
		call_g = next(i for i, ln in enumerate(lines) if ln == "call g")
		call_f = next(i for i, ln in enumerate(lines) if ln == "call f")
		assert call_g < call_f

	def test_two_nested_calls_as_args(self) -> None:
		"""f(g(x), h(y)) should call g and h first, then f."""
		body: list[IRInstruction] = [
			IRCall(IRTemp("t0"), "g", [IRConst(1)], [IRType.INT], IRType.INT),
			IRCall(IRTemp("t1"), "h", [IRConst(2)], [IRType.INT], IRType.INT),
			IRCall(
				IRTemp("t2"), "f",
				[IRTemp("t0"), IRTemp("t1")],
				[IRType.INT, IRType.INT],
				IRType.INT,
			),
			IRReturn(IRTemp("t2")),
		]
		func = IRFunction("main", [], body, IRType.INT)
		asm = _gen(func)
		lines = _asm_lines(asm)
		call_g = next(i for i, ln in enumerate(lines) if ln == "call g")
		call_h = next(i for i, ln in enumerate(lines) if ln == "call h")
		call_f = next(i for i, ln in enumerate(lines) if ln == "call f")
		assert call_g < call_f
		assert call_h < call_f

	def test_deeply_nested_calls(self) -> None:
		"""f(g(h(x))) should produce 3 calls in correct order."""
		body: list[IRInstruction] = [
			IRCall(IRTemp("t0"), "h", [IRConst(1)], [IRType.INT], IRType.INT),
			IRCall(IRTemp("t1"), "g", [IRTemp("t0")], [IRType.INT], IRType.INT),
			IRCall(IRTemp("t2"), "f", [IRTemp("t1")], [IRType.INT], IRType.INT),
			IRReturn(IRTemp("t2")),
		]
		func = IRFunction("main", [], body, IRType.INT)
		asm = _gen(func)
		lines = _asm_lines(asm)
		call_h = next(i for i, ln in enumerate(lines) if ln == "call h")
		call_g = next(i for i, ln in enumerate(lines) if ln == "call g")
		call_f = next(i for i, ln in enumerate(lines) if ln == "call f")
		assert call_h < call_g < call_f

	def test_call_result_passed_to_correct_register(self) -> None:
		"""Result of g() passed as first arg to f() should end up in %rdi."""
		body: list[IRInstruction] = [
			IRCall(IRTemp("t0"), "g", [], [], IRType.INT),
			IRCall(IRTemp("t1"), "f", [IRTemp("t0")], [IRType.INT], IRType.INT),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("main", [], body, IRType.INT)
		asm = _gen(func)
		# After call g, result in %rax is stored, then loaded into %rdi for call f
		assert "%rdi" in asm


# ---------------------------------------------------------------------------
# 4. Switch with fall-through and mixed case/default ordering
# ---------------------------------------------------------------------------


class TestSwitchControlFlow:
	"""Test switch-like control flow with labels and conditional jumps."""

	def test_switch_three_cases(self) -> None:
		"""Switch with 3 cases: codegen should emit cmp + je pattern for each case."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("val"), IRConst(2)),
			# Case 0
			IRBinOp(IRTemp("cmp0"), IRTemp("val"), "==", IRConst(0)),
			IRCondJump(IRTemp("cmp0"), "case_0", "check_1"),
			IRLabelInstr("check_1"),
			# Case 1
			IRBinOp(IRTemp("cmp1"), IRTemp("val"), "==", IRConst(1)),
			IRCondJump(IRTemp("cmp1"), "case_1", "check_2"),
			IRLabelInstr("check_2"),
			# Case 2
			IRBinOp(IRTemp("cmp2"), IRTemp("val"), "==", IRConst(2)),
			IRCondJump(IRTemp("cmp2"), "case_2", "default_case"),
			# Case bodies
			IRLabelInstr("case_0"),
			IRCopy(IRTemp("r"), IRConst(10)),
			IRJump("switch_end"),
			IRLabelInstr("case_1"),
			IRCopy(IRTemp("r"), IRConst(20)),
			IRJump("switch_end"),
			IRLabelInstr("case_2"),
			IRCopy(IRTemp("r"), IRConst(30)),
			IRJump("switch_end"),
			IRLabelInstr("default_case"),
			IRCopy(IRTemp("r"), IRConst(99)),
			IRJump("switch_end"),
			IRLabelInstr("switch_end"),
			IRReturn(IRTemp("r")),
		]
		func = IRFunction("switch_test", [], body, IRType.INT)
		asm = _gen(func)
		# Each case label should be present
		assert "case_0:" in asm
		assert "case_1:" in asm
		assert "case_2:" in asm
		assert "default_case:" in asm
		assert "switch_end:" in asm

	def test_switch_fallthrough(self) -> None:
		"""Fall-through: no jmp between case_0 body and case_1 body."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("val"), IRConst(0)),
			IRBinOp(IRTemp("cmp0"), IRTemp("val"), "==", IRConst(0)),
			IRCondJump(IRTemp("cmp0"), "case_0", "check_1"),
			IRLabelInstr("check_1"),
			IRBinOp(IRTemp("cmp1"), IRTemp("val"), "==", IRConst(1)),
			IRCondJump(IRTemp("cmp1"), "case_1", "default_case"),
			# Fall-through: case_0 flows directly into case_1 (no jmp)
			IRLabelInstr("case_0"),
			IRCopy(IRTemp("x"), IRConst(10)),
			IRLabelInstr("case_1"),
			IRCopy(IRTemp("y"), IRConst(20)),
			IRJump("switch_end"),
			IRLabelInstr("default_case"),
			IRCopy(IRTemp("y"), IRConst(99)),
			IRJump("switch_end"),
			IRLabelInstr("switch_end"),
			IRReturn(IRTemp("y")),
		]
		func = IRFunction("fallthrough", [], body, IRType.INT)
		asm = _gen(func)
		lines = _asm_lines(asm)
		# Between case_0: and case_1: there should be no jmp
		case0_idx = next(i for i, ln in enumerate(lines) if ln == "case_0:")
		case1_idx = next(i for i, ln in enumerate(lines) if ln == "case_1:")
		between = lines[case0_idx + 1:case1_idx]
		jmp_lines = [x for x in between if x.startswith("jmp ")]
		assert len(jmp_lines) == 0

	def test_default_before_cases(self) -> None:
		"""Default label can appear before case labels in the IR."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("val"), IRConst(5)),
			IRBinOp(IRTemp("cmp"), IRTemp("val"), "==", IRConst(1)),
			IRCondJump(IRTemp("cmp"), "case_1", "default_lbl"),
			# Default body first
			IRLabelInstr("default_lbl"),
			IRCopy(IRTemp("r"), IRConst(0)),
			IRJump("end"),
			# Then case 1
			IRLabelInstr("case_1"),
			IRCopy(IRTemp("r"), IRConst(1)),
			IRJump("end"),
			IRLabelInstr("end"),
			IRReturn(IRTemp("r")),
		]
		func = IRFunction("default_first", [], body, IRType.INT)
		asm = _gen(func)
		lines = _asm_lines(asm)
		default_idx = next(i for i, ln in enumerate(lines) if ln == "default_lbl:")
		case1_idx = next(i for i, ln in enumerate(lines) if ln == "case_1:")
		assert default_idx < case1_idx


# ---------------------------------------------------------------------------
# 5. Deeply nested ternary expressions
# ---------------------------------------------------------------------------


class TestNestedTernary:
	"""Test deeply nested ternary (conditional) expressions at the IR level."""

	def test_single_ternary(self) -> None:
		"""Simple ternary: cond ? 10 : 20"""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("cond"), IRConst(1)),
			IRCondJump(IRTemp("cond"), "true_1", "false_1"),
			IRLabelInstr("true_1"),
			IRCopy(IRTemp("r"), IRConst(10)),
			IRJump("end_1"),
			IRLabelInstr("false_1"),
			IRCopy(IRTemp("r"), IRConst(20)),
			IRJump("end_1"),
			IRLabelInstr("end_1"),
			IRReturn(IRTemp("r")),
		]
		func = IRFunction("ternary1", [], body, IRType.INT)
		asm = _gen(func)
		assert "true_1:" in asm
		assert "false_1:" in asm
		assert "end_1:" in asm
		# Should have conditional jump
		assert "jne true_1" in asm

	def test_nested_ternary_2_levels(self) -> None:
		"""Nested: cond1 ? (cond2 ? 1 : 2) : 3"""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("c1"), IRConst(1)),
			IRCondJump(IRTemp("c1"), "outer_true", "outer_false"),
			IRLabelInstr("outer_true"),
			IRCopy(IRTemp("c2"), IRConst(0)),
			IRCondJump(IRTemp("c2"), "inner_true", "inner_false"),
			IRLabelInstr("inner_true"),
			IRCopy(IRTemp("r"), IRConst(1)),
			IRJump("end"),
			IRLabelInstr("inner_false"),
			IRCopy(IRTemp("r"), IRConst(2)),
			IRJump("end"),
			IRLabelInstr("outer_false"),
			IRCopy(IRTemp("r"), IRConst(3)),
			IRJump("end"),
			IRLabelInstr("end"),
			IRReturn(IRTemp("r")),
		]
		func = IRFunction("ternary2", [], body, IRType.INT)
		asm = _gen(func)
		assert "outer_true:" in asm
		assert "inner_true:" in asm
		assert "inner_false:" in asm
		assert "outer_false:" in asm
		assert "end:" in asm

	def test_nested_ternary_3_levels(self) -> None:
		"""3-level nesting: a ? (b ? (c ? 1 : 2) : 3) : 4"""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("a"), IRConst(1)),
			IRCondJump(IRTemp("a"), "l1_true", "l1_false"),
			IRLabelInstr("l1_true"),
			IRCopy(IRTemp("b"), IRConst(1)),
			IRCondJump(IRTemp("b"), "l2_true", "l2_false"),
			IRLabelInstr("l2_true"),
			IRCopy(IRTemp("c"), IRConst(0)),
			IRCondJump(IRTemp("c"), "l3_true", "l3_false"),
			IRLabelInstr("l3_true"),
			IRCopy(IRTemp("r"), IRConst(1)),
			IRJump("done"),
			IRLabelInstr("l3_false"),
			IRCopy(IRTemp("r"), IRConst(2)),
			IRJump("done"),
			IRLabelInstr("l2_false"),
			IRCopy(IRTemp("r"), IRConst(3)),
			IRJump("done"),
			IRLabelInstr("l1_false"),
			IRCopy(IRTemp("r"), IRConst(4)),
			IRJump("done"),
			IRLabelInstr("done"),
			IRReturn(IRTemp("r")),
		]
		func = IRFunction("ternary3", [], body, IRType.INT)
		asm = _gen(func)
		# All labels should be emitted
		for label in ["l1_true", "l1_false", "l2_true", "l2_false", "l3_true", "l3_false", "done"]:
			assert f"{label}:" in asm
		# Should have 3 conditional jumps (jne)
		jne_count = len(re.findall(r"\bjne\b", asm))
		assert jne_count == 3

	def test_ternary_both_branches_produce_values(self) -> None:
		"""Both branches of ternary write to the same destination temp."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("cond"), IRConst(1)),
			IRCondJump(IRTemp("cond"), "t_br", "f_br"),
			IRLabelInstr("t_br"),
			IRCopy(IRTemp("result"), IRConst(100)),
			IRJump("merge"),
			IRLabelInstr("f_br"),
			IRCopy(IRTemp("result"), IRConst(200)),
			IRJump("merge"),
			IRLabelInstr("merge"),
			IRReturn(IRTemp("result")),
		]
		func = IRFunction("ternary_val", [], body, IRType.INT)
		asm = _gen(func)
		# Both $100 and $200 should appear as movq immediates
		assert "$100" in asm
		assert "$200" in asm


# ---------------------------------------------------------------------------
# 6. Break/continue in nested loops
# ---------------------------------------------------------------------------


class TestBreakContinueNestedLoops:
	"""Test break/continue control flow in nested loop structures."""

	def test_simple_while_break(self) -> None:
		"""While loop with break: should have jump to loop exit."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("i"), IRConst(0)),
			IRLabelInstr("loop_start"),
			IRBinOp(IRTemp("cond"), IRTemp("i"), "<", IRConst(10)),
			IRCondJump(IRTemp("cond"), "loop_body", "loop_end"),
			IRLabelInstr("loop_body"),
			# if i == 5 break
			IRBinOp(IRTemp("brk_cond"), IRTemp("i"), "==", IRConst(5)),
			IRCondJump(IRTemp("brk_cond"), "loop_end", "loop_cont"),
			IRLabelInstr("loop_cont"),
			IRBinOp(IRTemp("i"), IRTemp("i"), "+", IRConst(1)),
			IRJump("loop_start"),
			IRLabelInstr("loop_end"),
			IRReturn(IRTemp("i")),
		]
		func = IRFunction("while_break", [], body, IRType.INT)
		asm = _gen(func)
		assert "loop_start:" in asm
		assert "loop_end:" in asm
		# Break condition should jump to loop_end
		assert "jne loop_end" in asm

	def test_simple_continue(self) -> None:
		"""Continue should jump back to loop_start."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("i"), IRConst(0)),
			IRCopy(IRTemp("sum"), IRConst(0)),
			IRLabelInstr("loop_start"),
			IRBinOp(IRTemp("cond"), IRTemp("i"), "<", IRConst(10)),
			IRCondJump(IRTemp("cond"), "loop_body", "loop_end"),
			IRLabelInstr("loop_body"),
			IRBinOp(IRTemp("i"), IRTemp("i"), "+", IRConst(1)),
			# if i % 2 == 0 continue
			IRBinOp(IRTemp("mod"), IRTemp("i"), "%", IRConst(2)),
			IRBinOp(IRTemp("skip"), IRTemp("mod"), "==", IRConst(0)),
			IRCondJump(IRTemp("skip"), "loop_start", "no_skip"),
			IRLabelInstr("no_skip"),
			IRBinOp(IRTemp("sum"), IRTemp("sum"), "+", IRTemp("i")),
			IRJump("loop_start"),
			IRLabelInstr("loop_end"),
			IRReturn(IRTemp("sum")),
		]
		func = IRFunction("continue_test", [], body, IRType.INT)
		asm = _gen(func)
		# Continue should jump to loop_start
		assert "jne loop_start" in asm

	def test_nested_loop_break_targets_inner(self) -> None:
		"""Break in inner loop should target inner loop's end, not outer."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("i"), IRConst(0)),
			IRLabelInstr("outer_start"),
			IRBinOp(IRTemp("oc"), IRTemp("i"), "<", IRConst(5)),
			IRCondJump(IRTemp("oc"), "outer_body", "outer_end"),
			IRLabelInstr("outer_body"),
			IRCopy(IRTemp("j"), IRConst(0)),
			IRLabelInstr("inner_start"),
			IRBinOp(IRTemp("ic"), IRTemp("j"), "<", IRConst(5)),
			IRCondJump(IRTemp("ic"), "inner_body", "inner_end"),
			IRLabelInstr("inner_body"),
			# break from inner
			IRBinOp(IRTemp("bc"), IRTemp("j"), "==", IRConst(3)),
			IRCondJump(IRTemp("bc"), "inner_end", "inner_cont"),
			IRLabelInstr("inner_cont"),
			IRBinOp(IRTemp("j"), IRTemp("j"), "+", IRConst(1)),
			IRJump("inner_start"),
			IRLabelInstr("inner_end"),
			IRBinOp(IRTemp("i"), IRTemp("i"), "+", IRConst(1)),
			IRJump("outer_start"),
			IRLabelInstr("outer_end"),
			IRReturn(IRTemp("i")),
		]
		func = IRFunction("nested_break", [], body, IRType.INT)
		asm = _gen(func)
		# Inner break goes to inner_end, not outer_end
		assert "jne inner_end" in asm
		assert "outer_start:" in asm
		assert "inner_start:" in asm
		assert "outer_end:" in asm
		assert "inner_end:" in asm

	def test_nested_loop_continue_targets_inner(self) -> None:
		"""Continue in inner loop should jump to inner_start."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("i"), IRConst(0)),
			IRLabelInstr("outer"),
			IRBinOp(IRTemp("oc"), IRTemp("i"), "<", IRConst(3)),
			IRCondJump(IRTemp("oc"), "outer_body", "outer_done"),
			IRLabelInstr("outer_body"),
			IRCopy(IRTemp("j"), IRConst(0)),
			IRLabelInstr("inner"),
			IRBinOp(IRTemp("ic"), IRTemp("j"), "<", IRConst(3)),
			IRCondJump(IRTemp("ic"), "inner_body", "inner_done"),
			IRLabelInstr("inner_body"),
			IRBinOp(IRTemp("j"), IRTemp("j"), "+", IRConst(1)),
			# continue inner
			IRBinOp(IRTemp("sk"), IRTemp("j"), "==", IRConst(1)),
			IRCondJump(IRTemp("sk"), "inner", "inner_rest"),
			IRLabelInstr("inner_rest"),
			IRJump("inner"),
			IRLabelInstr("inner_done"),
			IRBinOp(IRTemp("i"), IRTemp("i"), "+", IRConst(1)),
			IRJump("outer"),
			IRLabelInstr("outer_done"),
			IRReturn(IRTemp("i")),
		]
		func = IRFunction("nested_continue", [], body, IRType.INT)
		asm = _gen(func)
		assert "jne inner" in asm


# ---------------------------------------------------------------------------
# 7. Goto jumping over variable declarations
# ---------------------------------------------------------------------------


class TestGotoOverDeclarations:
	"""Test goto jumping over variable declarations (codegen level)."""

	def test_goto_forward_over_assignment(self) -> None:
		"""Goto that skips over a variable assignment should still emit the label."""
		body: list[IRInstruction] = [
			IRJump("skip_label"),
			# This assignment is skipped
			IRCopy(IRTemp("x"), IRConst(42)),
			IRLabelInstr("skip_label"),
			IRCopy(IRTemp("y"), IRConst(10)),
			IRReturn(IRTemp("y")),
		]
		func = IRFunction("goto_fwd", [], body, IRType.INT)
		asm = _gen(func)
		assert "jmp skip_label" in asm
		assert "skip_label:" in asm
		# The skipped assignment ($42) should still be in the assembly
		# (it's just dead code, not eliminated at codegen level)
		assert "$42" in asm

	def test_goto_backward(self) -> None:
		"""Backward goto creates a loop-like structure."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("i"), IRConst(0)),
			IRLabelInstr("again"),
			IRBinOp(IRTemp("i"), IRTemp("i"), "+", IRConst(1)),
			IRBinOp(IRTemp("done"), IRTemp("i"), ">=", IRConst(10)),
			IRCondJump(IRTemp("done"), "exit", "again"),
			IRLabelInstr("exit"),
			IRReturn(IRTemp("i")),
		]
		func = IRFunction("goto_back", [], body, IRType.INT)
		asm = _gen(func)
		assert "again:" in asm
		assert "exit:" in asm
		# The backward jump should be "jmp again"
		assert "jmp again" in asm

	def test_multiple_gotos_to_same_label(self) -> None:
		"""Multiple goto statements targeting the same label."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("x"), IRConst(0)),
			IRBinOp(IRTemp("c1"), IRTemp("x"), "==", IRConst(0)),
			IRCondJump(IRTemp("c1"), "target", "check2"),
			IRLabelInstr("check2"),
			IRBinOp(IRTemp("c2"), IRTemp("x"), "==", IRConst(1)),
			IRCondJump(IRTemp("c2"), "target", "fallthrough"),
			IRLabelInstr("fallthrough"),
			IRCopy(IRTemp("r"), IRConst(99)),
			IRJump("done"),
			IRLabelInstr("target"),
			IRCopy(IRTemp("r"), IRConst(42)),
			IRJump("done"),
			IRLabelInstr("done"),
			IRReturn(IRTemp("r")),
		]
		func = IRFunction("multi_goto", [], body, IRType.INT)
		asm = _gen(func)
		assert "target:" in asm
		# Both conditional jumps should reference "target"
		jne_target = len(re.findall(r"jne target\b", asm))
		assert jne_target == 2

	def test_goto_skips_multiple_declarations(self) -> None:
		"""Goto that skips over multiple variable declarations."""
		body: list[IRInstruction] = [
			IRJump("end_label"),
			IRCopy(IRTemp("a"), IRConst(1)),
			IRCopy(IRTemp("b"), IRConst(2)),
			IRCopy(IRTemp("c"), IRConst(3)),
			IRBinOp(IRTemp("sum"), IRTemp("a"), "+", IRTemp("b")),
			IRLabelInstr("end_label"),
			IRCopy(IRTemp("result"), IRConst(0)),
			IRReturn(IRTemp("result")),
		]
		func = IRFunction("goto_skip_multi", [], body, IRType.INT)
		asm = _gen(func)
		assert "jmp end_label" in asm
		assert "end_label:" in asm
		# All skipped assignments should still be in the assembly as dead code
		assert "$1" in asm
		assert "$2" in asm
		assert "$3" in asm


# ---------------------------------------------------------------------------
# Additional edge cases combining multiple features
# ---------------------------------------------------------------------------


class TestCombinedEdgeCases:
	"""Tests combining multiple codegen edge cases."""

	def test_call_with_many_args_and_float_mix(self) -> None:
		"""Function call with 7 int args + 1 float arg."""
		int_args = [IRConst(i) for i in range(7)]
		float_args = [IRFloatConst(3.14)]
		all_args = int_args + float_args
		all_types = [IRType.INT] * 7 + [IRType.FLOAT]
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "big_func", all_args, all_types, IRType.INT),
			IRReturn(IRTemp("r")),
		]
		func = IRFunction("mixed_caller", [], body, IRType.INT)
		asm = _gen(func)
		# 7th int arg goes to stack
		assert "pushq" in asm
		# Float goes to xmm0
		assert "%xmm0" in asm
		# 1 SSE arg
		assert "movb $1, %al" in asm

	def test_condjump_generates_cmp_and_branch(self) -> None:
		"""IRCondJump should produce cmpq $0 + jne/jmp pair."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("flag"), IRConst(1)),
			IRCondJump(IRTemp("flag"), "yes", "no"),
			IRLabelInstr("yes"),
			IRReturn(IRConst(1)),
			IRLabelInstr("no"),
			IRReturn(IRConst(0)),
		]
		func = IRFunction("cond_test", [], body, IRType.INT)
		asm = _gen(func)
		assert "cmpq $0, %rax" in asm
		assert "jne yes" in asm
		assert "jmp no" in asm

	def test_return_value_in_rax(self) -> None:
		"""Integer return value should be placed in %rax."""
		body: list[IRInstruction] = [IRReturn(IRConst(42))]
		func = IRFunction("ret42", [], body, IRType.INT)
		asm = _gen(func)
		assert "movq $42, %rax" in asm
		assert "ret" in asm

	def test_float_return_in_xmm0(self) -> None:
		"""Float return value should be placed in %xmm0."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("f"), IRFloatConst(1.0), ir_type=IRType.FLOAT),
			IRReturn(IRTemp("f"), ir_type=IRType.FLOAT),
		]
		func = IRFunction("retf", [], body, IRType.FLOAT)
		asm = _gen(func)
		assert "movss" in asm
		assert "%xmm0" in asm

	def test_void_function_no_return_value(self) -> None:
		"""Void function should not put a value in %rax before ret."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("x"), IRConst(1)),
			IRReturn(),
		]
		func = IRFunction("void_fn", [], body, IRType.VOID)
		asm = _gen(func)
		lines = _asm_lines(asm)
		# Find the ret instruction
		ret_idx = next(i for i, ln in enumerate(lines) if ln == "ret")
		# The line before ret should be popq %rbp, not movq $something, %rax
		assert lines[ret_idx - 1] == "popq %rbp"
