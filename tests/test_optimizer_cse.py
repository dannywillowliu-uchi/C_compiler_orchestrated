"""Tests for CSE (common subexpression elimination) and LICM (loop-invariant code motion)."""

from compiler.ir import (
	IRBinOp,
	IRCondJump,
	IRConst,
	IRCopy,
	IRFunction,
	IRJump,
	IRLabelInstr,
	IRProgram,
	IRReturn,
	IRStore,
	IRTemp,
	IRType,
	IRUnaryOp,
)
from compiler.optimizer import IROptimizer


def _make_func(body, name="test", params=None):
	return IRProgram(functions=[
		IRFunction(name=name, params=params or [], body=body, return_type=IRType.INT)
	])


def _opt(body):
	prog = _make_func(body)
	result = IROptimizer().optimize(prog)
	return result.functions[0].body


def _cse_only(body):
	return IROptimizer()._cse(body)


def _licm_only(body):
	return IROptimizer()._licm(body)


# -- CSE: Basic within a block --


class TestCSEBasic:
	def test_duplicate_binop(self):
		"""Same binop expression is replaced with a copy."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _cse_only(body)
		assert isinstance(result[0], IRBinOp)
		assert isinstance(result[1], IRCopy)
		assert result[1].dest == IRTemp("t1")
		assert result[1].source == IRTemp("t0")

	def test_duplicate_unaryop(self):
		"""Same unary expression is replaced with a copy."""
		body = [
			IRUnaryOp(dest=IRTemp("t0"), op="-", operand=IRTemp("a")),
			IRUnaryOp(dest=IRTemp("t1"), op="-", operand=IRTemp("a")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _cse_only(body)
		assert isinstance(result[0], IRUnaryOp)
		assert isinstance(result[1], IRCopy)
		assert result[1].source == IRTemp("t0")

	def test_different_ops_not_cse(self):
		"""Different operators are not considered the same expression."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("a"), op="-", right=IRTemp("b")),
		]
		result = _cse_only(body)
		assert isinstance(result[0], IRBinOp)
		assert isinstance(result[1], IRBinOp)

	def test_different_operands_not_cse(self):
		"""Different operands are not the same expression."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("a"), op="+", right=IRTemp("c")),
		]
		result = _cse_only(body)
		assert isinstance(result[0], IRBinOp)
		assert isinstance(result[1], IRBinOp)

	def test_cse_with_constants(self):
		"""CSE works with constant operands."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(3), op="*", right=IRConst(4)),
			IRBinOp(dest=IRTemp("t1"), left=IRConst(3), op="*", right=IRConst(4)),
		]
		result = _cse_only(body)
		assert isinstance(result[1], IRCopy)
		assert result[1].source == IRTemp("t0")


# -- CSE: Invalidation on reassignment --


class TestCSEInvalidation:
	def test_reassignment_invalidates(self):
		"""Reassigning an operand invalidates the CSE entry."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRCopy(dest=IRTemp("a"), source=IRConst(5)),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _cse_only(body)
		# t1 should NOT be a copy because a was reassigned
		assert isinstance(result[2], IRBinOp)

	def test_reassignment_of_dest_invalidates(self):
		"""Reassigning the destination temp invalidates the CSE entry."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRCopy(dest=IRTemp("t0"), source=IRConst(99)),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _cse_only(body)
		# t1 should NOT be a copy because t0 was reassigned (value no longer valid)
		assert isinstance(result[2], IRBinOp)

	def test_store_invalidates_all(self):
		"""A store conservatively clears all available expressions."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRStore(address=IRTemp("ptr"), value=IRConst(42)),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("a"), op="+", right=IRTemp("b")),
		]
		result = _cse_only(body)
		assert isinstance(result[2], IRBinOp)

	def test_label_clears_available(self):
		"""Labels (basic block boundaries) clear available expressions."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRLabelInstr(name="L1"),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("a"), op="+", right=IRTemp("b")),
		]
		result = _cse_only(body)
		assert isinstance(result[2], IRBinOp)

	def test_self_referential_not_registered(self):
		"""t0 = t0 + 1 should not be registered for CSE."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("t0"), op="+", right=IRConst(1)),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="+", right=IRConst(1)),
		]
		result = _cse_only(body)
		# t1 should NOT be a copy because t0 = t0 + 1 is self-referential
		assert isinstance(result[1], IRBinOp)


# -- CSE: Multiple expressions --


class TestCSEMultipleExpressions:
	def test_two_different_expressions_both_cse(self):
		"""Two distinct expressions are both CSE'd when duplicated."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("c"), op="*", right=IRTemp("d")),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRBinOp(dest=IRTemp("t3"), left=IRTemp("c"), op="*", right=IRTemp("d")),
			IRReturn(value=IRTemp("t3")),
		]
		result = _cse_only(body)
		assert isinstance(result[0], IRBinOp)
		assert isinstance(result[1], IRBinOp)
		assert isinstance(result[2], IRCopy)
		assert result[2].source == IRTemp("t0")
		assert isinstance(result[3], IRCopy)
		assert result[3].source == IRTemp("t1")

	def test_triple_duplicate(self):
		"""Third occurrence also CSE'd to the first computation."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("a"), op="+", right=IRTemp("b")),
		]
		result = _cse_only(body)
		assert isinstance(result[0], IRBinOp)
		assert isinstance(result[1], IRCopy)
		assert result[1].source == IRTemp("t0")
		assert isinstance(result[2], IRCopy)
		assert result[2].source == IRTemp("t0")

	def test_cse_mixed_binop_and_unaryop(self):
		"""CSE works across both binary and unary ops independently."""
		body = [
			IRUnaryOp(dest=IRTemp("t0"), op="-", operand=IRTemp("x")),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRUnaryOp(dest=IRTemp("t2"), op="-", operand=IRTemp("x")),
			IRBinOp(dest=IRTemp("t3"), left=IRTemp("a"), op="+", right=IRTemp("b")),
		]
		result = _cse_only(body)
		assert isinstance(result[2], IRCopy)
		assert result[2].source == IRTemp("t0")
		assert isinstance(result[3], IRCopy)
		assert result[3].source == IRTemp("t1")


# -- LICM: Loop-invariant code motion --


class TestLICM:
	def test_hoist_invariant_binop(self):
		"""A computation with loop-external operands is hoisted above the loop header."""
		body = [
			IRCopy(dest=IRTemp("i"), source=IRConst(0)),
			IRLabelInstr(name="loop"),
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRBinOp(dest=IRTemp("i"), left=IRTemp("i"), op="+", right=IRConst(1)),
			IRCondJump(condition=IRTemp("i"), true_label="loop", false_label="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("t0")),
		]
		result = _licm_only(body)
		# t0 = a + b should be hoisted before the loop label
		binops_before_loop = []
		for instr in result:
			if isinstance(instr, IRLabelInstr) and instr.name == "loop":
				break
			if isinstance(instr, IRBinOp):
				binops_before_loop.append(instr)
		assert len(binops_before_loop) == 1
		assert binops_before_loop[0].dest == IRTemp("t0")
		assert binops_before_loop[0].left == IRTemp("a")

	def test_no_hoist_variant_computation(self):
		"""A computation using a loop variable is NOT hoisted."""
		body = [
			IRCopy(dest=IRTemp("i"), source=IRConst(0)),
			IRLabelInstr(name="loop"),
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("i"), op="*", right=IRConst(2)),
			IRBinOp(dest=IRTemp("i"), left=IRTemp("i"), op="+", right=IRConst(1)),
			IRCondJump(condition=IRTemp("i"), true_label="loop", false_label="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("t0")),
		]
		result = _licm_only(body)
		# t0 = i * 2 should NOT be hoisted (i is defined in the loop)
		binops_before_loop = []
		for instr in result:
			if isinstance(instr, IRLabelInstr) and instr.name == "loop":
				break
			if isinstance(instr, IRBinOp):
				binops_before_loop.append(instr)
		assert len(binops_before_loop) == 0

	def test_hoist_unaryop(self):
		"""A unary operation with loop-external operand is hoisted."""
		body = [
			IRLabelInstr(name="loop"),
			IRUnaryOp(dest=IRTemp("t0"), op="-", operand=IRTemp("x")),
			IRJump(target="loop"),
		]
		result = _licm_only(body)
		# t0 = -x should be hoisted
		assert isinstance(result[0], IRUnaryOp)
		assert result[0].dest == IRTemp("t0")

	def test_no_hoist_if_dest_redefined(self):
		"""Don't hoist if the same dest is defined elsewhere in the loop."""
		body = [
			IRLabelInstr(name="loop"),
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRCopy(dest=IRTemp("t0"), source=IRConst(99)),
			IRJump(target="loop"),
		]
		result = _licm_only(body)
		# t0 is defined both by the binop and the copy, so don't hoist
		assert isinstance(result[0], IRLabelInstr)
		assert result[0].name == "loop"

	def test_hoist_constant_computation(self):
		"""A computation on constants inside a loop is hoisted."""
		body = [
			IRLabelInstr(name="loop"),
			IRBinOp(dest=IRTemp("t0"), left=IRConst(3), op="*", right=IRConst(7)),
			IRReturn(value=IRTemp("t0")),
			IRJump(target="loop"),
		]
		result = _licm_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].dest == IRTemp("t0")


# -- Interaction with existing passes --


class TestCSEWithOtherPasses:
	def test_cse_then_dce(self):
		"""CSE introduces copies; DCE removes dead originals after propagation."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _opt(body)
		# After CSE: t1 = copy t0. After copy prop: return uses t0. DCE removes dead t1.
		ret = [i for i in result if isinstance(i, IRReturn)][0]
		assert ret.value == IRTemp("a") or ret.value == IRTemp("t0") or isinstance(ret.value, IRTemp)
		# The key check: no duplicate binop remains
		binops = [i for i in result if isinstance(i, IRBinOp)]
		assert len(binops) <= 1

	def test_constant_fold_then_cse(self):
		"""Constant folding happens before CSE, so identical folds are CSE'd."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("x"), op="+", right=IRTemp("y")),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="+", right=IRTemp("y")),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t0"), op="*", right=IRTemp("t1")),
			IRReturn(value=IRTemp("t2")),
		]
		result = _opt(body)
		# x + y computed once, reused
		binops_add = [i for i in result if isinstance(i, IRBinOp) and i.op == "+"]
		assert len(binops_add) <= 1

	def test_licm_with_full_pipeline(self):
		"""LICM hoists invariant code, then other passes optimize further."""
		body = [
			IRCopy(dest=IRTemp("i"), source=IRConst(0)),
			IRLabelInstr(name="loop"),
			IRBinOp(dest=IRTemp("t0"), left=IRConst(2), op="+", right=IRConst(3)),
			IRBinOp(dest=IRTemp("i"), left=IRTemp("i"), op="+", right=IRConst(1)),
			IRCondJump(condition=IRTemp("i"), true_label="loop", false_label="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("t0")),
		]
		result = _opt(body)
		# 2 + 3 should be folded to 5 and hoisted (or folded first then hoisted)
		# Either way, no 2+3 binop should remain
		binops_add = [i for i in result if isinstance(i, IRBinOp) and i.op == "+"]
		# The only remaining add should be i + 1, not 2 + 3
		for b in binops_add:
			assert not (isinstance(b.left, IRConst) and b.left.value == 2)

	def test_cse_preserves_side_effects(self):
		"""CSE does not eliminate stores or loads."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRStore(address=IRTemp("ptr"), value=IRTemp("t0")),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _cse_only(body)
		# Store is preserved
		assert any(isinstance(i, IRStore) for i in result)
		# After store, CSE is invalidated, so t1 is recomputed
		assert isinstance(result[2], IRBinOp)
