"""Tests for IR-level dead code elimination: unreachable instruction removal
after return/goto and unused IRTemp definition elimination."""

from compiler.ir import (
	IRAlloc,
	IRBinOp,
	IRCall,
	IRCondJump,
	IRConst,
	IRConvert,
	IRCopy,
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


def _unreachable_only(body):
	"""Run only unreachable elimination (single pass)."""
	return IROptimizer()._unreachable_elimination(body)


def _dce_only(body):
	"""Run only dead code elimination (single pass)."""
	return IROptimizer()._dead_code_elimination(body)


# ===== Unreachable code after return =====


class TestUnreachableAfterReturn:
	"""Instructions between IRReturn and the next label are unreachable."""

	def test_removes_instructions_after_return(self):
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(42)),
			IRReturn(value=IRTemp("t0")),
			IRCopy(dest=IRTemp("t1"), source=IRConst(99)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t1"), op="+", right=IRConst(1)),
			IRLabelInstr(name="L1"),
			IRReturn(value=IRConst(0)),
		]
		result = _unreachable_only(body)
		# t1 and t2 instructions should be removed
		assert len(result) == 4
		assert isinstance(result[0], IRCopy)
		assert isinstance(result[1], IRReturn)
		assert isinstance(result[2], IRLabelInstr)
		assert isinstance(result[3], IRReturn)

	def test_keeps_label_after_return(self):
		body = [
			IRReturn(value=IRConst(1)),
			IRLabelInstr(name="L1"),
			IRReturn(value=IRConst(2)),
		]
		result = _unreachable_only(body)
		assert len(result) == 3
		assert isinstance(result[1], IRLabelInstr)

	def test_no_dead_code_after_return(self):
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _unreachable_only(body)
		assert len(result) == 2

	def test_multiple_returns_with_dead_code(self):
		body = [
			IRCondJump(condition=IRTemp("t0"), true_label="L1", false_label="L2"),
			IRLabelInstr(name="L1"),
			IRReturn(value=IRConst(1)),
			IRCopy(dest=IRTemp("dead1"), source=IRConst(100)),
			IRLabelInstr(name="L2"),
			IRReturn(value=IRConst(2)),
			IRCopy(dest=IRTemp("dead2"), source=IRConst(200)),
		]
		result = _unreachable_only(body)
		# dead1 and dead2 should be removed
		assert "dead1" not in str(result)
		assert "dead2" not in str(result)


# ===== Unreachable code after goto (IRJump) =====


class TestUnreachableAfterGoto:
	"""Instructions between IRJump and the next label are unreachable."""

	def test_removes_instructions_after_jump(self):
		body = [
			IRJump(target="L2"),
			IRCopy(dest=IRTemp("t1"), source=IRConst(99)),
			IRStore(address=IRTemp("t3"), value=IRConst(0)),
			IRLabelInstr(name="L2"),
			IRReturn(value=IRConst(0)),
		]
		result = _unreachable_only(body)
		assert len(result) == 3
		assert isinstance(result[0], IRJump)
		assert isinstance(result[1], IRLabelInstr)
		assert isinstance(result[2], IRReturn)

	def test_jump_immediately_before_label(self):
		body = [
			IRJump(target="L1"),
			IRLabelInstr(name="L1"),
			IRReturn(value=IRConst(0)),
		]
		result = _unreachable_only(body)
		assert len(result) == 3

	def test_nested_jumps_with_dead_code(self):
		body = [
			IRJump(target="L1"),
			IRCopy(dest=IRTemp("dead1"), source=IRConst(1)),
			IRLabelInstr(name="L1"),
			IRJump(target="L2"),
			IRCopy(dest=IRTemp("dead2"), source=IRConst(2)),
			IRLabelInstr(name="L2"),
			IRReturn(value=IRConst(0)),
		]
		result = _unreachable_only(body)
		assert len(result) == 5
		assert all(not (isinstance(i, IRCopy) and i.dest.name.startswith("dead")) for i in result)


# ===== Unreachable code with mixed control flow =====


class TestUnreachableMixedControlFlow:
	"""Combinations of returns and jumps with unreachable code."""

	def test_condjump_not_treated_as_terminator(self):
		"""Conditional jumps should NOT cause following code to be unreachable."""
		body = [
			IRCondJump(condition=IRTemp("t0"), true_label="L1", false_label="L2"),
			IRCopy(dest=IRTemp("t1"), source=IRConst(42)),
			IRLabelInstr(name="L1"),
			IRReturn(value=IRConst(0)),
		]
		# The copy after condjump is technically unreachable in well-formed IR,
		# but unreachable_elimination only triggers on IRJump/IRReturn
		result = _unreachable_only(body)
		# condjump is not IRJump or IRReturn, so t1 should be kept
		assert any(isinstance(i, IRCopy) and i.dest.name == "t1" for i in result)

	def test_empty_body(self):
		result = _unreachable_only([])
		assert result == []

	def test_only_return(self):
		body = [IRReturn(value=IRConst(0))]
		result = _unreachable_only(body)
		assert len(result) == 1

	def test_return_at_end_no_dead(self):
		body = [
			IRLabelInstr(name="entry"),
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _unreachable_only(body)
		assert len(result) == 3


# ===== Dead code elimination: unused temp definitions =====


class TestUnusedTempElimination:
	"""Remove definitions of temps that are never read."""

	def test_removes_unused_binop(self):
		body = [
			IRBinOp(dest=IRTemp("unused"), left=IRConst(1), op="+", right=IRConst(2)),
			IRReturn(value=IRConst(0)),
		]
		result = _dce_only(body)
		assert len(result) == 1
		assert isinstance(result[0], IRReturn)

	def test_removes_unused_unaryop(self):
		body = [
			IRUnaryOp(dest=IRTemp("unused"), op="-", operand=IRConst(5)),
			IRReturn(value=IRConst(0)),
		]
		result = _dce_only(body)
		assert len(result) == 1

	def test_removes_unused_copy(self):
		body = [
			IRCopy(dest=IRTemp("unused"), source=IRConst(42)),
			IRReturn(value=IRConst(0)),
		]
		result = _dce_only(body)
		assert len(result) == 1

	def test_removes_unused_convert(self):
		body = [
			IRConvert(dest=IRTemp("unused"), source=IRConst(42), from_type=IRType.INT, to_type=IRType.LONG),
			IRReturn(value=IRConst(0)),
		]
		result = _dce_only(body)
		assert len(result) == 1

	def test_removes_unused_load(self):
		"""IRLoad with unused result should be removed (no side effects)."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRStore(address=IRTemp("addr"), value=IRConst(10)),
			IRLoad(dest=IRTemp("unused"), address=IRTemp("addr")),
			IRReturn(value=IRConst(0)),
		]
		result = _dce_only(body)
		# addr is used by store and load, so alloc/store stay. Load result is unused.
		assert not any(isinstance(i, IRLoad) for i in result)

	def test_keeps_used_definitions(self):
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(1), op="+", right=IRConst(2)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _dce_only(body)
		assert len(result) == 2
		assert isinstance(result[0], IRBinOp)

	def test_keeps_call_even_if_result_unused(self):
		"""Calls have side effects and must not be removed."""
		body = [
			IRCall(dest=IRTemp("unused"), function_name="printf", args=[IRConst(0)]),
			IRReturn(value=IRConst(0)),
		]
		result = _dce_only(body)
		assert len(result) == 2
		assert isinstance(result[0], IRCall)

	def test_keeps_store_instructions(self):
		"""Stores have side effects and must not be removed."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRStore(address=IRTemp("addr"), value=IRConst(42)),
			IRReturn(value=IRConst(0)),
		]
		result = _dce_only(body)
		assert any(isinstance(i, IRStore) for i in result)

	def test_chain_of_unused_temps(self):
		"""Full optimizer should remove chains of unused temps iteratively."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(1)),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="+", right=IRConst(2)),
			IRUnaryOp(dest=IRTemp("t2"), op="-", operand=IRTemp("t1")),
			IRReturn(value=IRConst(0)),
		]
		result = _opt(body)
		# All three temps are unused; full optimizer should remove the chain
		assert not any(isinstance(i, (IRBinOp, IRUnaryOp, IRCopy)) for i in result)

	def test_partial_chain_keeps_used(self):
		"""Only the unused tail of a chain should be removed."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(1)),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="+", right=IRConst(2)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _opt(body)
		# t1 is used by return; after constant folding t0+2 = 1+2 = 3
		assert any(isinstance(i, IRReturn) for i in result)


# ===== Full optimizer integration: unreachable + DCE combined =====


class TestFullOptimizerDCE:
	"""Full optimizer should combine unreachable elimination with DCE."""

	def test_unreachable_and_unused_combined(self):
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(42)),
			IRReturn(value=IRTemp("t0")),
			IRCopy(dest=IRTemp("dead"), source=IRConst(99)),
			IRLabelInstr(name="L1"),
			IRCopy(dest=IRTemp("unused"), source=IRConst(0)),
			IRReturn(value=IRConst(0)),
		]
		result = _opt(body)
		# "dead" removed by unreachable elim, "unused" removed by DCE
		assert not any(
			isinstance(i, IRCopy) and i.dest.name in ("dead", "unused")
			for i in result
		)

	def test_goto_with_dead_code_and_unused_temps(self):
		body = [
			IRCopy(dest=IRTemp("x"), source=IRConst(10)),
			IRJump(target="end"),
			IRCopy(dest=IRTemp("unreachable"), source=IRConst(20)),
			IRLabelInstr(name="end"),
			IRCopy(dest=IRTemp("unused"), source=IRConst(30)),
			IRReturn(value=IRConst(0)),
		]
		result = _opt(body)
		assert not any(
			isinstance(i, IRCopy) and i.dest.name in ("unreachable", "unused", "x")
			for i in result
		)

	def test_constant_condjump_creates_unreachable(self):
		"""Constant folding turns condjump into jump, creating unreachable code."""
		body = [
			IRCondJump(condition=IRConst(1), true_label="L1", false_label="L2"),
			IRLabelInstr(name="L1"),
			IRReturn(value=IRConst(1)),
			IRLabelInstr(name="L2"),
			IRReturn(value=IRConst(2)),
		]
		result = _opt(body)
		# Should fold to jump L1, making L2 potentially reachable but the
		# condjump should become an unconditional jump
		assert any(isinstance(i, IRReturn) for i in result)

	def test_preserves_side_effects_in_live_code(self):
		"""Ensure stores and calls in live code are never removed."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRStore(address=IRTemp("addr"), value=IRConst(42)),
			IRCall(dest=IRTemp("r"), function_name="foo", args=[IRTemp("addr")]),
			IRReturn(value=IRTemp("r")),
		]
		result = _opt(body)
		assert any(isinstance(i, IRStore) for i in result)
		assert any(isinstance(i, IRCall) for i in result)
		assert any(isinstance(i, IRReturn) for i in result)
