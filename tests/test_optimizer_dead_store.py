"""Tests for dead store elimination pass in the optimizer.

Covers IRStore elimination for overwritten stack stores (x = 1; x = 2 pattern),
liveness-based dead copy/binop/convert elimination, alias safety, control flow
boundaries, and integration with the full optimizer pipeline.
"""

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
	IRUnaryOp,
)
from compiler.optimizer import IROptimizer


def _make_func(body, name="test"):
	return IRProgram(functions=[
		IRFunction(name=name, params=[], body=body, return_type=IRType.INT)
	])


def _dse(body):
	"""Run only the dead store elimination pass."""
	return IROptimizer()._dead_store_elimination(body)


def _opt(body):
	"""Run the full optimizer pipeline."""
	prog = _make_func(body)
	result = IROptimizer().optimize(prog)
	return result.functions[0].body


class TestBasicDeadStoreElimination:
	"""IRStore to stack location overwritten before any load."""

	def test_simple_overwrite(self):
		"""*addr = 1; *addr = 2 -> first store dead."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRStore(address=IRTemp("addr"), value=IRConst(1)),
			IRStore(address=IRTemp("addr"), value=IRConst(2)),
			IRLoad(dest=IRTemp("t"), address=IRTemp("addr")),
			IRReturn(value=IRTemp("t")),
		]
		result = _dse(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) == 1
		assert stores[0].value == IRConst(2)

	def test_triple_overwrite(self):
		"""*addr = 1; *addr = 2; *addr = 3 -> only last store survives."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRStore(address=IRTemp("addr"), value=IRConst(1)),
			IRStore(address=IRTemp("addr"), value=IRConst(2)),
			IRStore(address=IRTemp("addr"), value=IRConst(3)),
			IRLoad(dest=IRTemp("t"), address=IRTemp("addr")),
			IRReturn(value=IRTemp("t")),
		]
		result = _dse(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) == 1
		assert stores[0].value == IRConst(3)

	def test_different_addresses_preserved(self):
		"""*a = 5; *b = 10 -> both stores preserved."""
		body = [
			IRAlloc(dest=IRTemp("a"), size=4),
			IRAlloc(dest=IRTemp("b"), size=4),
			IRStore(address=IRTemp("a"), value=IRConst(5)),
			IRStore(address=IRTemp("b"), value=IRConst(10)),
			IRLoad(dest=IRTemp("t0"), address=IRTemp("a")),
			IRLoad(dest=IRTemp("t1"), address=IRTemp("b")),
			IRReturn(value=IRTemp("t0")),
		]
		result = _dse(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) == 2

	def test_intervening_load_preserves_store(self):
		"""*addr = 5; t = *addr; *addr = 10 -> first store is live."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRStore(address=IRTemp("addr"), value=IRConst(5)),
			IRLoad(dest=IRTemp("t"), address=IRTemp("addr")),
			IRStore(address=IRTemp("addr"), value=IRConst(10)),
			IRReturn(value=IRTemp("t")),
		]
		result = _dse(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) == 2

	def test_intervening_call_preserves_store(self):
		"""*addr = 5; call foo(); *addr = 10 -> first store preserved."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRStore(address=IRTemp("addr"), value=IRConst(5)),
			IRCall(dest=IRTemp("r"), function_name="foo"),
			IRStore(address=IRTemp("addr"), value=IRConst(10)),
			IRReturn(value=IRTemp("r")),
		]
		result = _dse(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) == 2


class TestAliasingSafety:
	"""Stores to aliased addresses must not be eliminated."""

	def test_addr_taken_prevents_elimination(self):
		"""If addr is taken via IRAddrOf, stores are preserved."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRAddrOf(dest=IRTemp("ptr"), source=IRTemp("addr")),
			IRStore(address=IRTemp("addr"), value=IRConst(5)),
			IRStore(address=IRTemp("addr"), value=IRConst(10)),
			IRReturn(value=IRTemp("ptr")),
		]
		result = _dse(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) == 2

	def test_non_aliased_still_eliminated(self):
		"""Aliasing of one addr doesn't protect a different addr."""
		body = [
			IRAlloc(dest=IRTemp("a"), size=4),
			IRAlloc(dest=IRTemp("b"), size=4),
			IRAddrOf(dest=IRTemp("ptr"), source=IRTemp("a")),
			IRStore(address=IRTemp("b"), value=IRConst(1)),
			IRStore(address=IRTemp("b"), value=IRConst(2)),
			IRLoad(dest=IRTemp("t"), address=IRTemp("b")),
			IRReturn(value=IRTemp("t")),
		]
		result = _dse(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) == 1
		assert stores[0].value == IRConst(2)


class TestLivenessBasedDeadDefs:
	"""Dead copies/binops/converts detected via liveness analysis."""

	def test_dead_copy_overwritten(self):
		"""t0 = 5; t0 = 10; return t0 -> first copy dead."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRCopy(dest=IRTemp("t0"), source=IRConst(10)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _dse(body)
		assert len(result) == 2
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(10)

	def test_dead_binop_overwritten(self):
		"""t0 = t1 + t2; t0 = 42; return t0 -> binop dead."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("t1"), op="+", right=IRTemp("t2")),
			IRCopy(dest=IRTemp("t0"), source=IRConst(42)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _dse(body)
		assert not any(isinstance(i, IRBinOp) for i in result)

	def test_dead_convert_overwritten(self):
		"""Convert result overwritten -> convert dead."""
		body = [
			IRConvert(dest=IRTemp("t0"), source=IRTemp("t1"), from_type=IRType.INT, to_type=IRType.FLOAT),
			IRCopy(dest=IRTemp("t0"), source=IRConst(0)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _dse(body)
		assert not any(isinstance(i, IRConvert) for i in result)

	def test_dead_unary_overwritten(self):
		"""Dead unary op eliminated."""
		body = [
			IRUnaryOp(dest=IRTemp("t0"), op="-", operand=IRTemp("t1")),
			IRCopy(dest=IRTemp("t0"), source=IRConst(0)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _dse(body)
		assert not any(isinstance(i, IRUnaryOp) for i in result)

	def test_live_copy_preserved(self):
		"""t0 = 5; return t0 -> copy is live."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _dse(body)
		assert len(result) == 2

	def test_copy_used_between_defs_preserved(self):
		"""t0 = 5; t1 = t0; t0 = 10; return t1 -> first t0 def is live."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRCopy(dest=IRTemp("t1"), source=IRTemp("t0")),
			IRCopy(dest=IRTemp("t0"), source=IRConst(10)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _dse(body)
		copies = [i for i in result if isinstance(i, IRCopy)]
		assert any(c.source == IRConst(5) for c in copies)


class TestControlFlowBoundaries:
	"""DSE respects basic block boundaries for stores."""

	def test_store_live_across_jump(self):
		"""Store at end of block is preserved when loaded in successor."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRStore(address=IRTemp("addr"), value=IRConst(5)),
			IRJump(target="L1"),
			IRLabelInstr(name="L1"),
			IRLoad(dest=IRTemp("t"), address=IRTemp("addr")),
			IRReturn(value=IRTemp("t")),
		]
		result = _dse(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) == 1

	def test_copy_live_across_branch(self):
		"""Copy used in alternate branch must not be eliminated."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRCondJump(condition=IRTemp("cond"), true_label="L1", false_label="L2"),
			IRLabelInstr(name="L1"),
			IRCopy(dest=IRTemp("t0"), source=IRConst(10)),
			IRJump(target="L3"),
			IRLabelInstr(name="L2"),
			IRReturn(value=IRTemp("t0")),
			IRLabelInstr(name="L3"),
			IRReturn(value=IRTemp("t0")),
		]
		result = _dse(body)
		first_copies = [
			i for i in result
			if isinstance(i, IRCopy) and i.source == IRConst(5)
		]
		assert len(first_copies) == 1

	def test_store_within_single_block_eliminated(self):
		"""Two stores in same block with no load between -> first dead."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRStore(address=IRTemp("addr"), value=IRConst(100)),
			IRStore(address=IRTemp("addr"), value=IRConst(200)),
			IRJump(target="done"),
			IRLabelInstr(name="done"),
			IRLoad(dest=IRTemp("t"), address=IRTemp("addr")),
			IRReturn(value=IRTemp("t")),
		]
		result = _dse(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) == 1
		assert stores[0].value == IRConst(200)


class TestEdgeCases:
	"""Edge cases and boundary conditions."""

	def test_empty_body(self):
		"""DSE handles empty body gracefully."""
		assert _dse([]) == []

	def test_no_dead_stores(self):
		"""Body with no dead stores is unchanged."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _dse(body)
		assert len(result) == 2

	def test_store_to_non_temp_address_preserved(self):
		"""Stores to non-IRTemp addresses are conservatively preserved."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRStore(address=IRConst(0), value=IRConst(5)),
			IRStore(address=IRConst(0), value=IRConst(10)),
			IRReturn(value=None),
		]
		result = _dse(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) == 2

	def test_load_from_different_address_no_effect(self):
		"""Load from different address doesn't prevent elimination."""
		body = [
			IRAlloc(dest=IRTemp("a"), size=4),
			IRAlloc(dest=IRTemp("b"), size=4),
			IRStore(address=IRTemp("a"), value=IRConst(1)),
			IRLoad(dest=IRTemp("t"), address=IRTemp("b")),
			IRStore(address=IRTemp("a"), value=IRConst(2)),
			IRLoad(dest=IRTemp("t2"), address=IRTemp("a")),
			IRReturn(value=IRTemp("t2")),
		]
		result = _dse(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) == 1
		assert stores[0].value == IRConst(2)


class TestFullPipelineIntegration:
	"""DSE as part of the full optimization pipeline."""

	def test_x_eq_1_x_eq_2_pattern(self):
		"""x = 1; x = 2; return x -> dead store eliminated by full optimizer."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRStore(address=IRTemp("addr"), value=IRConst(1)),
			IRStore(address=IRTemp("addr"), value=IRConst(2)),
			IRLoad(dest=IRTemp("t"), address=IRTemp("addr")),
			IRReturn(value=IRTemp("t")),
		]
		result = _opt(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) <= 1

	def test_dead_copy_removed_in_pipeline(self):
		"""Full optimizer removes dead copies."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRCopy(dest=IRTemp("t0"), source=IRConst(10)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _opt(body)
		copies = [i for i in result if isinstance(i, IRCopy)]
		assert len(copies) <= 1

	def test_chained_dead_stores_in_pipeline(self):
		"""Multiple dead stores all removed by full optimizer."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRStore(address=IRTemp("addr"), value=IRConst(1)),
			IRStore(address=IRTemp("addr"), value=IRConst(2)),
			IRStore(address=IRTemp("addr"), value=IRConst(3)),
			IRLoad(dest=IRTemp("t"), address=IRTemp("addr")),
			IRReturn(value=IRTemp("t")),
		]
		result = _opt(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) <= 1
