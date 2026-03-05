"""Tests for global copy propagation across basic blocks."""

from compiler.ir import (
	IRBinOp,
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
)
from compiler.optimizer import IROptimizer


def _make_program(body: list[IRInstruction]) -> IRProgram:
	func = IRFunction(name="test", params=[], body=body, return_type="int")
	return IRProgram(functions=[func])


def _opt(body: list[IRInstruction]) -> list[IRInstruction]:
	prog = _make_program(body)
	opt = IROptimizer()
	result = opt.optimize(prog)
	return result.functions[0].body


class TestGlobalCopyPropagationBasic:
	"""Test cross-block copy propagation."""

	def test_copy_propagated_across_blocks(self):
		"""Copy in block A should propagate to block B when source is not redefined."""
		body = [
			IRCopy(dest=IRTemp("t2"), source=IRTemp("t1")),
			IRJump(target="L1"),
			IRLabelInstr(name="L1"),
			IRBinOp(dest=IRTemp("t3"), left=IRTemp("t2"), op="+", right=IRConst(1)),
			IRReturn(value=IRTemp("t3")),
		]
		result = _opt(body)
		# t2 should be replaced by t1 in the binop
		binops = [i for i in result if isinstance(i, IRBinOp)]
		if binops:
			assert isinstance(binops[0].left, IRTemp)
			assert binops[0].left.name == "t1"

	def test_copy_not_propagated_when_source_redefined_and_used(self):
		"""Copy should NOT propagate if the source is redefined and both values are used."""
		body = [
			IRCopy(dest=IRTemp("t2"), source=IRTemp("t1")),
			IRJump(target="L1"),
			IRLabelInstr(name="L1"),
			# t1 is redefined here
			IRCopy(dest=IRTemp("t1"), source=IRConst(42)),
			# Both t2 (old t1) and t1 (new value) are used
			IRBinOp(dest=IRTemp("t3"), left=IRTemp("t2"), op="+", right=IRTemp("t1")),
			IRReturn(value=IRTemp("t3")),
		]
		result = _opt(body)
		# t2 should NOT be replaced by t1 since t1 was redefined to a different value
		binops = [i for i in result if isinstance(i, IRBinOp)]
		if binops:
			left, right = binops[0].left, binops[0].right
			# The two operands should be different (old t1 value vs new 42)
			if isinstance(left, IRTemp) and isinstance(right, (IRConst, IRTemp)):
				assert not (isinstance(right, IRTemp) and left.name == right.name)

	def test_copy_not_propagated_when_dest_redefined(self):
		"""Copy should NOT propagate if the dest is redefined before use."""
		body = [
			IRCopy(dest=IRTemp("t2"), source=IRTemp("t1")),
			IRJump(target="L1"),
			IRLabelInstr(name="L1"),
			# t2 is redefined, so old copy is dead
			IRCopy(dest=IRTemp("t2"), source=IRConst(99)),
			IRBinOp(dest=IRTemp("t3"), left=IRTemp("t2"), op="+", right=IRConst(1)),
			IRReturn(value=IRTemp("t3")),
		]
		result = _opt(body)
		# The binop should use the constant 99, not t1
		binops = [i for i in result if isinstance(i, IRBinOp)]
		if binops:
			left = binops[0].left
			# Should be IRConst(99) after copy prop + const prop, or t2 if only copy prop ran
			assert not (isinstance(left, IRTemp) and left.name == "t1")


class TestGlobalCopyPropagationMerge:
	"""Test copy propagation at CFG merge points (multiple predecessors)."""

	def test_copy_not_propagated_at_merge_without_agreement(self):
		"""At a merge point, copy should NOT propagate if not available on all paths."""
		body = [
			IRCondJump(condition=IRTemp("cond"), true_label="L_true", false_label="L_false"),
			IRLabelInstr(name="L_true"),
			IRCopy(dest=IRTemp("t2"), source=IRTemp("t1")),
			IRJump(target="L_merge"),
			IRLabelInstr(name="L_false"),
			# No copy of t2=t1 on this path
			IRCopy(dest=IRTemp("t2"), source=IRConst(5)),
			IRJump(target="L_merge"),
			IRLabelInstr(name="L_merge"),
			IRBinOp(dest=IRTemp("t3"), left=IRTemp("t2"), op="+", right=IRConst(1)),
			IRReturn(value=IRTemp("t3")),
		]
		result = _opt(body)
		# t2 should NOT be replaced at L_merge since the copy isn't on both paths
		binops = [i for i in result if isinstance(i, IRBinOp)]
		if binops:
			left = binops[0].left
			assert not (isinstance(left, IRTemp) and left.name == "t1")

	def test_copy_propagated_at_merge_with_agreement(self):
		"""At a merge point, copy SHOULD propagate if available on ALL paths."""
		body = [
			IRCondJump(condition=IRTemp("cond"), true_label="L_true", false_label="L_false"),
			IRLabelInstr(name="L_true"),
			IRCopy(dest=IRTemp("t2"), source=IRTemp("t1")),
			IRJump(target="L_merge"),
			IRLabelInstr(name="L_false"),
			IRCopy(dest=IRTemp("t2"), source=IRTemp("t1")),
			IRJump(target="L_merge"),
			IRLabelInstr(name="L_merge"),
			IRBinOp(dest=IRTemp("t3"), left=IRTemp("t2"), op="+", right=IRConst(1)),
			IRReturn(value=IRTemp("t3")),
		]
		result = _opt(body)
		# t2 should be replaced by t1 since the copy is on both paths
		binops = [i for i in result if isinstance(i, IRBinOp)]
		if binops:
			assert isinstance(binops[0].left, IRTemp)
			assert binops[0].left.name == "t1"


class TestGlobalCopyPropagationChain:
	"""Test chained and transitive copy propagation across blocks."""

	def test_chained_copies_across_blocks(self):
		"""Chain of copies across blocks: t2=t1, then t3=t2 should resolve t3 to t1 or t2."""
		body = [
			IRCopy(dest=IRTemp("t2"), source=IRTemp("t1")),
			IRJump(target="L1"),
			IRLabelInstr(name="L1"),
			IRCopy(dest=IRTemp("t3"), source=IRTemp("t2")),
			IRJump(target="L2"),
			IRLabelInstr(name="L2"),
			IRBinOp(dest=IRTemp("t4"), left=IRTemp("t3"), op="+", right=IRConst(1)),
			IRReturn(value=IRTemp("t4")),
		]
		result = _opt(body)
		# t3 should be resolved (at least to t2, possibly to t1 through chain)
		binops = [i for i in result if isinstance(i, IRBinOp)]
		if binops:
			assert isinstance(binops[0].left, IRTemp)
			# Should resolve to either t1 or t2 (not t3, which is an intermediate)
			assert binops[0].left.name in ("t1", "t2")

	def test_copy_propagation_multiple_uses(self):
		"""Single copy should propagate to multiple uses in successor block."""
		body = [
			IRCopy(dest=IRTemp("t2"), source=IRTemp("t1")),
			IRJump(target="L1"),
			IRLabelInstr(name="L1"),
			IRBinOp(dest=IRTemp("t3"), left=IRTemp("t2"), op="+", right=IRTemp("t2")),
			IRReturn(value=IRTemp("t3")),
		]
		result = _opt(body)
		binops = [i for i in result if isinstance(i, IRBinOp)]
		if binops:
			assert isinstance(binops[0].left, IRTemp) and binops[0].left.name == "t1"
			assert isinstance(binops[0].right, IRTemp) and binops[0].right.name == "t1"


class TestGlobalCopyPropagationInteraction:
	"""Test interaction of global copy propagation with other passes."""

	def test_global_copy_prop_enables_dce(self):
		"""After global copy propagation, the copy instruction itself becomes dead."""
		body = [
			IRCopy(dest=IRTemp("t2"), source=IRTemp("t1")),
			IRJump(target="L1"),
			IRLabelInstr(name="L1"),
			IRBinOp(dest=IRTemp("t3"), left=IRTemp("t2"), op="+", right=IRConst(1)),
			IRReturn(value=IRTemp("t3")),
		]
		result = _opt(body)
		# The copy t2=t1 should be eliminated by DCE since t2 is no longer used
		copies = [i for i in result if isinstance(i, IRCopy) and i.dest.name == "t2"]
		assert len(copies) == 0

	def test_single_block_unchanged(self):
		"""Single-block functions should still work (handled by local copy prop)."""
		body = [
			IRCopy(dest=IRTemp("t2"), source=IRTemp("t1")),
			IRBinOp(dest=IRTemp("t3"), left=IRTemp("t2"), op="+", right=IRConst(1)),
			IRReturn(value=IRTemp("t3")),
		]
		result = _opt(body)
		# Local copy prop should handle this
		binops = [i for i in result if isinstance(i, IRBinOp)]
		if binops:
			assert isinstance(binops[0].left, IRTemp)
			assert binops[0].left.name == "t1"

	def test_empty_body(self):
		"""Empty function body should not crash."""
		result = _opt([])
		assert result == []

	def test_no_copies(self):
		"""Function with no copies should pass through unchanged."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(1), op="+", right=IRConst(2)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _opt(body)
		assert any(isinstance(i, IRReturn) for i in result)
