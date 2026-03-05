"""Tests for constant folding optimization: binary ops, unary ops, conditional jump
folding, and interaction with copy propagation and DCE to produce fewer instructions."""

from compiler.ir import (
	IRBinOp,
	IRCondJump,
	IRConst,
	IRCopy,
	IRFloatConst,
	IRFunction,
	IRJump,
	IRLabelInstr,
	IRProgram,
	IRReturn,
	IRTemp,
	IRType,
	IRUnaryOp,
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


def _fold_only(body):
	"""Run only constant folding (single pass)."""
	return IROptimizer()._constant_fold(body)


def _condjump_only(body):
	"""Run only constant conditional jump folding."""
	return IROptimizer()._constant_condjump_folding(body)


# ── Constant Folding: Binary Operations ──


class TestConstFoldBinaryOps:
	def test_fold_add(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(3), op="+", right=IRConst(4))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(7)

	def test_fold_subtract(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(10), op="-", right=IRConst(3))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(7)

	def test_fold_multiply(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(6), op="*", right=IRConst(7))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(42)

	def test_fold_divide(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(20), op="/", right=IRConst(4))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(5)

	def test_fold_divide_by_zero_unchanged(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(10), op="/", right=IRConst(0))]
		result = _fold_only(body)
		assert isinstance(result[0], IRBinOp)

	def test_fold_modulo(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(10), op="%", right=IRConst(3))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(1)

	def test_fold_modulo_by_zero_unchanged(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(10), op="%", right=IRConst(0))]
		result = _fold_only(body)
		assert isinstance(result[0], IRBinOp)

	def test_fold_bitwise_and(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(0xFF), op="&", right=IRConst(0x0F))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(0x0F)

	def test_fold_bitwise_or(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(0xF0), op="|", right=IRConst(0x0F))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(0xFF)

	def test_fold_bitwise_xor(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(0xFF), op="^", right=IRConst(0x0F))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(0xF0)

	def test_fold_left_shift(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(1), op="<<", right=IRConst(3))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(8)

	def test_fold_right_shift(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(16), op=">>", right=IRConst(2))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(4)

	def test_fold_comparison_less(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(3), op="<", right=IRConst(5))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(1)

	def test_fold_comparison_equal(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(5), op="==", right=IRConst(5))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(1)

	def test_fold_comparison_not_equal(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(3), op="!=", right=IRConst(5))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(1)

	def test_fold_logical_and(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(1), op="&&", right=IRConst(2))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(1)

	def test_fold_logical_or_both_zero(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(0), op="||", right=IRConst(0))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(0)

	def test_non_const_operand_unchanged(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRTemp("t1"), op="+", right=IRConst(4))]
		result = _fold_only(body)
		assert isinstance(result[0], IRBinOp)


# ── Constant Folding: Unary Operations ──


class TestConstFoldUnaryOps:
	def test_fold_negate(self):
		body = [IRUnaryOp(dest=IRTemp("t0"), op="-", operand=IRConst(5))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(-5)

	def test_fold_bitwise_not(self):
		body = [IRUnaryOp(dest=IRTemp("t0"), op="~", operand=IRConst(0))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(-1)

	def test_fold_logical_not_true(self):
		body = [IRUnaryOp(dest=IRTemp("t0"), op="!", operand=IRConst(5))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(0)

	def test_fold_logical_not_false(self):
		body = [IRUnaryOp(dest=IRTemp("t0"), op="!", operand=IRConst(0))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(1)

	def test_fold_float_negate(self):
		body = [IRUnaryOp(dest=IRTemp("t0"), op="-", operand=IRFloatConst(3.14))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert isinstance(result[0].source, IRFloatConst)
		assert result[0].source.value == -3.14


# ── Constant Conditional Jump Folding ──


class TestConstCondJumpFolding:
	def test_true_condition_becomes_jump(self):
		"""IRCondJump with nonzero constant -> IRJump to true_label."""
		body = [IRCondJump(condition=IRConst(1), true_label="L1", false_label="L2")]
		result = _condjump_only(body)
		assert len(result) == 1
		assert isinstance(result[0], IRJump)
		assert result[0].target == "L1"

	def test_false_condition_becomes_jump(self):
		"""IRCondJump with zero constant -> IRJump to false_label."""
		body = [IRCondJump(condition=IRConst(0), true_label="L1", false_label="L2")]
		result = _condjump_only(body)
		assert len(result) == 1
		assert isinstance(result[0], IRJump)
		assert result[0].target == "L2"

	def test_nonzero_int_is_true(self):
		"""Any nonzero integer is truthy."""
		body = [IRCondJump(condition=IRConst(42), true_label="L1", false_label="L2")]
		result = _condjump_only(body)
		assert isinstance(result[0], IRJump)
		assert result[0].target == "L1"

	def test_negative_int_is_true(self):
		"""Negative integers are truthy."""
		body = [IRCondJump(condition=IRConst(-1), true_label="L1", false_label="L2")]
		result = _condjump_only(body)
		assert isinstance(result[0], IRJump)
		assert result[0].target == "L1"

	def test_float_zero_is_false(self):
		"""Float 0.0 is falsy."""
		body = [IRCondJump(condition=IRFloatConst(0.0), true_label="L1", false_label="L2")]
		result = _condjump_only(body)
		assert isinstance(result[0], IRJump)
		assert result[0].target == "L2"

	def test_float_nonzero_is_true(self):
		"""Float nonzero is truthy."""
		body = [IRCondJump(condition=IRFloatConst(1.5), true_label="L1", false_label="L2")]
		result = _condjump_only(body)
		assert isinstance(result[0], IRJump)
		assert result[0].target == "L1"

	def test_non_const_condition_unchanged(self):
		"""Non-constant conditions are left as-is."""
		body = [IRCondJump(condition=IRTemp("t0"), true_label="L1", false_label="L2")]
		result = _condjump_only(body)
		assert isinstance(result[0], IRCondJump)

	def test_other_instructions_pass_through(self):
		"""Non-condjump instructions are not affected."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRCondJump(condition=IRConst(1), true_label="L1", false_label="L2"),
			IRLabelInstr(name="L1"),
		]
		result = _condjump_only(body)
		assert isinstance(result[0], IRCopy)
		assert isinstance(result[1], IRJump)
		assert isinstance(result[2], IRLabelInstr)


# ── Integration: Constant Folding Produces Fewer Instructions ──


class TestConstFoldReducesInstructions:
	def test_fold_chain_reduces_instructions(self):
		"""t0 = 3 + 4; t1 = t0 * 2; return t1 -> return 14 with fewer instructions."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(3), op="+", right=IRConst(4)),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="*", right=IRConst(2)),
			IRReturn(value=IRTemp("t1")),
		]
		original_count = len(body)
		result = _opt(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert isinstance(ret[0].value, IRConst)
		assert ret[0].value.value == 14
		assert len(result) < original_count

	def test_const_comparison_folds_branch(self):
		"""Constant comparison folded, then condjump becomes unconditional."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(3), op="<", right=IRConst(5)),
			IRCondJump(condition=IRTemp("t0"), true_label="L1", false_label="L2"),
			IRLabelInstr(name="L1"),
			IRReturn(value=IRConst(1)),
			IRLabelInstr(name="L2"),
			IRReturn(value=IRConst(0)),
		]
		result = _opt(body)
		# The comparison should be folded, condjump should become a jump to L1
		jumps = [i for i in result if isinstance(i, IRJump)]
		assert any(j.target == "L1" for j in jumps)
		# Should not have a condjump anymore
		assert not any(isinstance(i, IRCondJump) for i in result)

	def test_false_branch_eliminated(self):
		"""When condition is always false, true branch becomes unreachable."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(5), op=">", right=IRConst(10)),
			IRCondJump(condition=IRTemp("t0"), true_label="L1", false_label="L2"),
			IRLabelInstr(name="L1"),
			IRReturn(value=IRConst(1)),
			IRLabelInstr(name="L2"),
			IRReturn(value=IRConst(0)),
		]
		result = _opt(body)
		# Should jump to L2 (false branch)
		jumps = [i for i in result if isinstance(i, IRJump)]
		assert any(j.target == "L2" for j in jumps)
		assert not any(isinstance(i, IRCondJump) for i in result)

	def test_copy_prop_enables_condjump_folding(self):
		"""Copy propagation propagates constants to condjump, enabling folding."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(1)),
			IRCondJump(condition=IRTemp("t0"), true_label="L1", false_label="L2"),
			IRLabelInstr(name="L1"),
			IRReturn(value=IRConst(10)),
			IRLabelInstr(name="L2"),
			IRReturn(value=IRConst(20)),
		]
		result = _opt(body)
		assert not any(isinstance(i, IRCondJump) for i in result)
		jumps = [i for i in result if isinstance(i, IRJump)]
		assert any(j.target == "L1" for j in jumps)

	def test_nested_const_exprs_fully_folded(self):
		"""Multiple levels of constant expressions all fold down."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(2), op="+", right=IRConst(3)),
			IRBinOp(dest=IRTemp("t1"), left=IRConst(4), op="*", right=IRConst(5)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t0"), op="+", right=IRTemp("t1")),
			IRReturn(value=IRTemp("t2")),
		]
		result = _opt(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert isinstance(ret[0].value, IRConst)
		assert ret[0].value.value == 25  # (2+3) + (4*5) = 5 + 20 = 25
		# All binops and copies should be eliminated, leaving just the return
		assert len(result) == 1

	def test_dce_removes_unused_folded_results(self):
		"""Dead code elimination removes unused results from folded constants."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(1), op="+", right=IRConst(2)),
			IRBinOp(dest=IRTemp("t1"), left=IRConst(3), op="+", right=IRConst(4)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _opt(body)
		# t1 is unused, should be eliminated
		assert len(result) == 1
		assert isinstance(result[0], IRReturn)
		assert result[0].value == IRConst(3)
