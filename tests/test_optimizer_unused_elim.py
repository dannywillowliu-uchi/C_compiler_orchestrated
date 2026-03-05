"""Tests for liveness-based unused variable elimination in the optimizer."""

from compiler.ir import (
	IRAddrOf,
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
	IRVaStart,
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


def _elim_only(body):
	return IROptimizer()._unused_variable_elimination(body)


class TestUnusedCopyElimination:
	"""Test elimination of unused IRCopy instructions."""

	def test_unused_copy_removed(self):
		"""t0 = 5; return 10 -> copy to t0 is dead."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRReturn(value=IRConst(10)),
		]
		result = _elim_only(body)
		assert len(result) == 1
		assert isinstance(result[0], IRReturn)

	def test_used_copy_kept(self):
		"""t0 = 5; return t0 -> copy must stay."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _elim_only(body)
		assert len(result) == 2
		assert isinstance(result[0], IRCopy)

	def test_overwritten_before_use(self):
		"""t0 = 5; t0 = 10; return t0 -> first copy is dead."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRCopy(dest=IRTemp("t0"), source=IRConst(10)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _elim_only(body)
		assert len(result) == 2
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(10)


class TestUnusedBinopElimination:
	"""Test elimination of unused IRBinOp instructions."""

	def test_unused_binop_removed(self):
		"""t0 = t1 + t2; return t1 -> binop result unused."""
		body = [
			IRCopy(dest=IRTemp("t1"), source=IRConst(3)),
			IRCopy(dest=IRTemp("t2"), source=IRConst(4)),
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("t1"), op="+", right=IRTemp("t2")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _elim_only(body)
		# t0 binop should be removed, t2 copy stays (sources used by binop are resolved separately)
		assert not any(isinstance(i, IRBinOp) for i in result)

	def test_used_binop_kept(self):
		"""t0 = t1 + t2; return t0 -> binop result used."""
		body = [
			IRCopy(dest=IRTemp("t1"), source=IRConst(3)),
			IRCopy(dest=IRTemp("t2"), source=IRConst(4)),
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("t1"), op="+", right=IRTemp("t2")),
			IRReturn(value=IRTemp("t0")),
		]
		result = _elim_only(body)
		assert any(isinstance(i, IRBinOp) for i in result)


class TestUnusedLoadElimination:
	"""Test elimination of unused IRLoad instructions."""

	def test_unused_load_removed(self):
		"""t1 = *t0; return 42 -> load is dead."""
		body = [
			IRAlloc(dest=IRTemp("t0"), size=4),
			IRLoad(dest=IRTemp("t1"), address=IRTemp("t0")),
			IRReturn(value=IRConst(42)),
		]
		result = _elim_only(body)
		assert not any(isinstance(i, IRLoad) for i in result)

	def test_used_load_kept(self):
		"""t1 = *t0; return t1 -> load is live."""
		body = [
			IRAlloc(dest=IRTemp("t0"), size=4),
			IRLoad(dest=IRTemp("t1"), address=IRTemp("t0")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _elim_only(body)
		assert any(isinstance(i, IRLoad) for i in result)


class TestUnusedAllocElimination:
	"""Test elimination of unused IRAlloc instructions."""

	def test_unused_alloc_removed(self):
		"""t0 = alloc 4; return 0 -> alloc is dead."""
		body = [
			IRAlloc(dest=IRTemp("t0"), size=4),
			IRReturn(value=IRConst(0)),
		]
		result = _elim_only(body)
		assert len(result) == 1
		assert isinstance(result[0], IRReturn)

	def test_used_alloc_kept(self):
		"""t0 = alloc 4; *t0 = 5; return -> alloc is used by store."""
		body = [
			IRAlloc(dest=IRTemp("t0"), size=4),
			IRStore(address=IRTemp("t0"), value=IRConst(5)),
			IRReturn(value=IRConst(0)),
		]
		result = _elim_only(body)
		assert any(isinstance(i, IRAlloc) for i in result)


class TestUnusedAddrOfElimination:
	"""Test elimination of unused IRAddrOf instructions."""

	def test_unused_addrof_removed(self):
		"""t1 = &t0; return 0 -> addrof is dead."""
		body = [
			IRAlloc(dest=IRTemp("t0"), size=4),
			IRAddrOf(dest=IRTemp("t1"), source=IRTemp("t0")),
			IRReturn(value=IRConst(0)),
		]
		result = _elim_only(body)
		assert not any(isinstance(i, IRAddrOf) for i in result)

	def test_used_addrof_kept(self):
		"""t1 = &t0; return t1 -> addrof result used."""
		body = [
			IRAlloc(dest=IRTemp("t0"), size=4),
			IRAddrOf(dest=IRTemp("t1"), source=IRTemp("t0")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _elim_only(body)
		assert any(isinstance(i, IRAddrOf) for i in result)


class TestUnusedConvertElimination:
	"""Test elimination of unused IRConvert instructions."""

	def test_unused_convert_removed(self):
		"""t1 = convert t0 INT->FLOAT; return t0 -> convert dead."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(42)),
			IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.INT, to_type=IRType.FLOAT),
			IRReturn(value=IRTemp("t0")),
		]
		result = _elim_only(body)
		assert not any(isinstance(i, IRConvert) for i in result)


class TestSideEffectsPreserved:
	"""Ensure instructions with side effects are NOT eliminated even if dest is unused."""

	def test_call_with_unused_dest_kept(self):
		"""t0 = call foo(); return 0 -> call has side effects, must stay."""
		body = [
			IRCall(dest=IRTemp("t0"), function_name="foo", args=[]),
			IRReturn(value=IRConst(0)),
		]
		result = _elim_only(body)
		assert any(isinstance(i, IRCall) for i in result)

	def test_store_kept(self):
		"""*t0 = 5; return 0 -> store has side effects."""
		body = [
			IRAlloc(dest=IRTemp("t0"), size=4),
			IRStore(address=IRTemp("t0"), value=IRConst(5)),
			IRReturn(value=IRConst(0)),
		]
		result = _elim_only(body)
		assert any(isinstance(i, IRStore) for i in result)

	def test_vastart_kept(self):
		"""va_start is side-effectful, must stay."""
		body = [
			IRAlloc(dest=IRTemp("ap"), size=24),
			IRVaStart(ap_addr=IRTemp("ap"), num_named_gp=1),
			IRReturn(value=IRConst(0)),
		]
		result = _elim_only(body)
		assert any(isinstance(i, IRVaStart) for i in result)


class TestControlFlowLiveness:
	"""Test liveness across branches and joins."""

	def test_dead_in_both_branches(self):
		"""t0 defined but not used in either branch -> dead."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRCopy(dest=IRTemp("cond"), source=IRConst(1)),
			IRCondJump(condition=IRTemp("cond"), true_label="L1", false_label="L2"),
			IRLabelInstr(name="L1"),
			IRReturn(value=IRConst(1)),
			IRLabelInstr(name="L2"),
			IRReturn(value=IRConst(2)),
		]
		result = _elim_only(body)
		# t0 copy should be eliminated
		assert not any(
			isinstance(i, IRCopy) and i.dest.name == "t0" for i in result
		)

	def test_live_in_one_branch(self):
		"""t0 used in one branch -> must stay."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRCopy(dest=IRTemp("cond"), source=IRConst(1)),
			IRCondJump(condition=IRTemp("cond"), true_label="L1", false_label="L2"),
			IRLabelInstr(name="L1"),
			IRReturn(value=IRTemp("t0")),
			IRLabelInstr(name="L2"),
			IRReturn(value=IRConst(2)),
		]
		result = _elim_only(body)
		assert any(
			isinstance(i, IRCopy) and i.dest.name == "t0" for i in result
		)

	def test_loop_liveness(self):
		"""Variable used inside a loop body stays alive."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(0)),
			IRCopy(dest=IRTemp("limit"), source=IRConst(10)),
			IRLabelInstr(name="loop"),
			IRBinOp(dest=IRTemp("cmp"), left=IRTemp("t0"), op="<", right=IRTemp("limit")),
			IRCondJump(condition=IRTemp("cmp"), true_label="body", false_label="end"),
			IRLabelInstr(name="body"),
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("t0"), op="+", right=IRConst(1)),
			IRJump(target="loop"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("t0")),
		]
		result = _elim_only(body)
		# All instructions should remain - everything is live
		assert len(result) == len(body)


class TestCascadingElimination:
	"""Test that the full optimizer cascades: removing one unused def exposes more."""

	def test_chain_elimination(self):
		"""t0 = 1; t1 = t0 + 2; return 0 -> both defs eventually removed."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(1)),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="+", right=IRConst(2)),
			IRReturn(value=IRConst(0)),
		]
		result = _opt(body)
		# After full optimization, only return should remain
		assert len(result) == 1
		assert isinstance(result[0], IRReturn)

	def test_multiple_unused_temps(self):
		"""Several unused temps all eliminated."""
		body = [
			IRCopy(dest=IRTemp("a"), source=IRConst(1)),
			IRCopy(dest=IRTemp("b"), source=IRConst(2)),
			IRCopy(dest=IRTemp("c"), source=IRConst(3)),
			IRCopy(dest=IRTemp("result"), source=IRConst(42)),
			IRReturn(value=IRTemp("result")),
		]
		result = _opt(body)
		# All unused temps eliminated; copy-prop may fold result into return
		assert len(result) <= 2
		assert any(isinstance(i, IRReturn) for i in result)


class TestEmptyAndTrivial:
	"""Edge cases: empty body, single instruction, etc."""

	def test_empty_body(self):
		result = _elim_only([])
		assert result == []

	def test_single_return(self):
		body = [IRReturn(value=IRConst(0))]
		result = _elim_only(body)
		assert len(result) == 1

	def test_no_dead_code(self):
		"""All instructions are live -> no change."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(42)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _elim_only(body)
		assert len(result) == 2
