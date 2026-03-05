"""Tests for loop-invariant code motion (LICM) optimization pass."""

from compiler.ir import (
	IRBinOp,
	IRCondJump,
	IRConst,
	IRConvert,
	IRCopy,
	IRFloatConst,
	IRFunction,
	IRJump,
	IRLabelInstr,
	IRLoad,
	IRProgram,
	IRReturn,
	IRStore,
	IRTemp,
	IRType,
	IRUnaryOp,
	IRCall,
)
from compiler.optimizer import IROptimizer


def _make_func(body, name="test", params=None):
	"""Helper to wrap instructions in a single-function program."""
	return IRProgram(functions=[
		IRFunction(name=name, params=params or [], body=body, return_type=IRType.INT)
	])


def _opt(body):
	"""Optimize a function body and return the resulting instructions."""
	prog = _make_func(body)
	result = IROptimizer().optimize(prog)
	return result.functions[0].body


def _licm_only(body):
	"""Run only the LICM pass (single invocation)."""
	return IROptimizer()._licm(body)


# ── Basic LICM ──


class TestLICMBasic:
	"""Test that simple loop-invariant computations are hoisted."""

	def test_hoist_constant_binop(self):
		"""t2 = 3 + 4 inside a loop should be hoisted before the header."""
		body = [
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_header"),
			IRBinOp(dest=IRTemp("t2"), left=IRConst(3), op="+", right=IRConst(4)),
			IRCopy(dest=IRTemp("t0"), source=IRTemp("t2")),
			IRCondJump(condition=IRTemp("t0"), true_label="loop_header", false_label="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("t0")),
		]
		result = _licm_only(body)
		# The constant binop should appear before the loop_header label
		header_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRLabelInstr) and instr.name == "loop_header")
		binop_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRBinOp) and instr.dest.name == "t2")
		assert binop_idx < header_idx, "Loop-invariant binop should be hoisted before loop header"

	def test_hoist_invariant_with_outside_operand(self):
		"""An instruction using a temp defined outside the loop is invariant."""
		body = [
			IRCopy(dest=IRTemp("x"), source=IRConst(10)),
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_header"),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="+", right=IRConst(5)),
			IRCopy(dest=IRTemp("i"), source=IRTemp("t1")),
			IRCondJump(condition=IRTemp("i"), true_label="loop_header", false_label="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("i")),
		]
		result = _licm_only(body)
		header_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRLabelInstr) and instr.name == "loop_header")
		binop_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRBinOp) and instr.dest.name == "t1")
		assert binop_idx < header_idx

	def test_no_hoist_varying_operand(self):
		"""An instruction using a temp modified inside the loop is NOT invariant."""
		body = [
			IRCopy(dest=IRTemp("i"), source=IRConst(0)),
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_header"),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("i"), op="+", right=IRConst(1)),
			IRCopy(dest=IRTemp("i"), source=IRTemp("t1")),
			IRCondJump(condition=IRTemp("i"), true_label="loop_header", false_label="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("i")),
		]
		result = _licm_only(body)
		# t1 depends on i which is modified in the loop, so nothing should be hoisted
		header_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRLabelInstr) and instr.name == "loop_header")
		binop_positions = [i for i, instr in enumerate(result) if isinstance(instr, IRBinOp)]
		for pos in binop_positions:
			assert pos > header_idx, "Non-invariant binop should NOT be hoisted"

	def test_no_hoist_multiple_defs(self):
		"""An instruction whose dest is defined multiple times in the loop is NOT hoisted."""
		body = [
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_header"),
			IRBinOp(dest=IRTemp("t1"), left=IRConst(1), op="+", right=IRConst(2)),
			IRCopy(dest=IRTemp("t1"), source=IRConst(99)),
			IRCondJump(condition=IRTemp("t1"), true_label="loop_header", false_label="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("t1")),
		]
		result = _licm_only(body)
		# t1 is defined twice in the loop, so the binop should not be hoisted
		header_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRLabelInstr) and instr.name == "loop_header")
		binop_positions = [i for i, instr in enumerate(result) if isinstance(instr, IRBinOp)]
		for pos in binop_positions:
			assert pos > header_idx


class TestLICMTransitive:
	"""Test transitive loop-invariant detection."""

	def test_transitive_invariance(self):
		"""If t1 = const op const, and t2 = t1 op const, both should be hoisted."""
		body = [
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_header"),
			IRBinOp(dest=IRTemp("t1"), left=IRConst(3), op="+", right=IRConst(4)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t1"), op="*", right=IRConst(2)),
			IRCopy(dest=IRTemp("r"), source=IRTemp("t2")),
			IRCondJump(condition=IRTemp("r"), true_label="loop_header", false_label="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("r")),
		]
		result = _licm_only(body)
		header_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRLabelInstr) and instr.name == "loop_header")
		# Both t1 and t2 should be hoisted
		t1_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRBinOp) and instr.dest.name == "t1")
		t2_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRBinOp) and instr.dest.name == "t2")
		assert t1_idx < header_idx
		assert t2_idx < header_idx
		# t1 should come before t2 (dependency order preserved)
		assert t1_idx < t2_idx


class TestLICMInstructionTypes:
	"""Test LICM with different instruction types."""

	def test_hoist_unary_op(self):
		"""Unary operations with invariant operands should be hoisted."""
		body = [
			IRCopy(dest=IRTemp("x"), source=IRConst(42)),
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_header"),
			IRUnaryOp(dest=IRTemp("t1"), op="-", operand=IRTemp("x")),
			IRCopy(dest=IRTemp("r"), source=IRTemp("t1")),
			IRCondJump(condition=IRTemp("r"), true_label="loop_header", false_label="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("r")),
		]
		result = _licm_only(body)
		header_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRLabelInstr) and instr.name == "loop_header")
		unary_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRUnaryOp))
		assert unary_idx < header_idx

	def test_hoist_copy_of_outside_temp(self):
		"""A copy of a temp defined outside the loop is invariant."""
		body = [
			IRCopy(dest=IRTemp("x"), source=IRConst(10)),
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_header"),
			IRCopy(dest=IRTemp("t1"), source=IRTemp("x")),
			IRCondJump(condition=IRTemp("t1"), true_label="loop_header", false_label="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("t1")),
		]
		result = _licm_only(body)
		header_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRLabelInstr) and instr.name == "loop_header")
		copy_idx = next(
			i for i, instr in enumerate(result)
			if isinstance(instr, IRCopy) and instr.dest.name == "t1"
		)
		assert copy_idx < header_idx

	def test_hoist_convert(self):
		"""IRConvert with invariant source should be hoisted."""
		body = [
			IRCopy(dest=IRTemp("x"), source=IRConst(10)),
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_header"),
			IRConvert(dest=IRTemp("t1"), source=IRTemp("x"), from_type=IRType.INT, to_type=IRType.FLOAT),
			IRCondJump(condition=IRTemp("x"), true_label="loop_header", false_label="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("x")),
		]
		result = _licm_only(body)
		header_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRLabelInstr) and instr.name == "loop_header")
		convert_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRConvert))
		assert convert_idx < header_idx

	def test_no_hoist_load(self):
		"""IRLoad should NOT be hoisted (may have side effects via memory)."""
		body = [
			IRCopy(dest=IRTemp("ptr"), source=IRConst(0)),
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_header"),
			IRLoad(dest=IRTemp("t1"), address=IRTemp("ptr")),
			IRCondJump(condition=IRTemp("t1"), true_label="loop_header", false_label="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("t1")),
		]
		result = _licm_only(body)
		header_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRLabelInstr) and instr.name == "loop_header")
		load_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRLoad))
		assert load_idx > header_idx, "Loads should not be hoisted"

	def test_no_hoist_store(self):
		"""IRStore should NOT be hoisted."""
		body = [
			IRCopy(dest=IRTemp("ptr"), source=IRConst(0)),
			IRCopy(dest=IRTemp("val"), source=IRConst(42)),
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_header"),
			IRStore(address=IRTemp("ptr"), value=IRTemp("val")),
			IRBinOp(dest=IRTemp("cond"), left=IRConst(1), op="-", right=IRConst(0)),
			IRCondJump(condition=IRTemp("cond"), true_label="loop_header", false_label="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRConst(0)),
		]
		result = _licm_only(body)
		header_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRLabelInstr) and instr.name == "loop_header")
		store_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRStore))
		assert store_idx > header_idx, "Stores should not be hoisted"

	def test_no_hoist_call(self):
		"""IRCall should NOT be hoisted (side effects)."""
		body = [
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_header"),
			IRCall(dest=IRTemp("t1"), function_name="foo", args=[]),
			IRCondJump(condition=IRTemp("t1"), true_label="loop_header", false_label="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("t1")),
		]
		result = _licm_only(body)
		header_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRLabelInstr) and instr.name == "loop_header")
		call_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRCall))
		assert call_idx > header_idx, "Calls should not be hoisted"


class TestLICMNoLoop:
	"""Test that LICM is a no-op when there are no loops."""

	def test_no_loop_unchanged(self):
		"""Straight-line code without loops should be unchanged."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(1), op="+", right=IRConst(2)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _licm_only(body)
		assert result == body

	def test_forward_jump_only(self):
		"""Forward jumps (no back edges) should not create loops."""
		body = [
			IRCondJump(condition=IRConst(1), true_label="then", false_label="else"),
			IRLabelInstr(name="then"),
			IRBinOp(dest=IRTemp("t0"), left=IRConst(1), op="+", right=IRConst(2)),
			IRJump(target="end"),
			IRLabelInstr(name="else"),
			IRBinOp(dest=IRTemp("t0"), left=IRConst(3), op="+", right=IRConst(4)),
			IRJump(target="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("t0")),
		]
		result = _licm_only(body)
		assert result == body


class TestLICMFloatConst:
	"""Test LICM with float constants."""

	def test_hoist_float_binop(self):
		"""Float constant operations inside a loop should be hoisted."""
		body = [
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_header"),
			IRBinOp(dest=IRTemp("t1"), left=IRFloatConst(1.5), op="+", right=IRFloatConst(2.5)),
			IRCopy(dest=IRTemp("r"), source=IRTemp("t1")),
			IRCondJump(condition=IRTemp("r"), true_label="loop_header", false_label="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("r")),
		]
		result = _licm_only(body)
		header_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRLabelInstr) and instr.name == "loop_header")
		binop_idx = next(i for i, instr in enumerate(result) if isinstance(instr, IRBinOp) and instr.dest.name == "t1")
		assert binop_idx < header_idx


class TestLICMIntegration:
	"""Integration tests: LICM combined with other optimization passes."""

	def test_licm_with_constant_folding(self):
		"""After LICM hoists a constant binop, constant folding can fold it."""
		body = [
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_header"),
			IRBinOp(dest=IRTemp("t1"), left=IRConst(3), op="+", right=IRConst(4)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t1"), op="*", right=IRConst(2)),
			IRReturn(value=IRTemp("t2")),
		]
		result = _opt(body)
		# After full optimization, the constant computation should be folded
		# and the result should be a direct return of 14
		ret = [instr for instr in result if isinstance(instr, IRReturn)]
		assert len(ret) >= 1
		assert isinstance(ret[0].value, IRConst)
		assert ret[0].value.value == 14

	def test_full_pipeline_loop_with_invariant(self):
		"""Full optimization pipeline with a loop containing invariant code."""
		body = [
			IRCopy(dest=IRTemp("n"), source=IRConst(10)),
			IRCopy(dest=IRTemp("i"), source=IRConst(0)),
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_header"),
			# Invariant: n * 2 doesn't change
			IRBinOp(dest=IRTemp("limit"), left=IRTemp("n"), op="*", right=IRConst(2)),
			# Varying: i + 1
			IRBinOp(dest=IRTemp("i_next"), left=IRTemp("i"), op="+", right=IRConst(1)),
			IRCopy(dest=IRTemp("i"), source=IRTemp("i_next")),
			IRBinOp(dest=IRTemp("cond"), left=IRTemp("i"), op="<", right=IRTemp("limit")),
			IRCondJump(condition=IRTemp("cond"), true_label="loop_header", false_label="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("i")),
		]
		result = _opt(body)
		# The loop should still produce a valid result
		# n*2 should have been hoisted and possibly folded to 20
		ret = [instr for instr in result if isinstance(instr, IRReturn)]
		assert len(ret) >= 1

	def test_empty_body(self):
		"""LICM should handle empty body gracefully."""
		result = _licm_only([])
		assert result == []
