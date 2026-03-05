"""Tests for dead store elimination pass in the optimizer."""

from compiler.ir import (
	IRAlloc,
	IRAddrOf,
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
	return IRProgram(functions=[
		IRFunction(name=name, params=params or [], body=body, return_type=IRType.INT)
	])


def _opt(body):
	prog = _make_func(body)
	result = IROptimizer().optimize(prog)
	return result.functions[0].body


def _dse_only(body):
	return IROptimizer()._dead_store_elimination(body)


class TestDeadCopyElimination:
	"""Test elimination of IRCopy/IRBinOp/IRConvert where dest is overwritten before use."""

	def test_copy_overwritten_before_use(self):
		"""t0 = 5; t0 = 10; return t0 -> first copy is dead."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRCopy(dest=IRTemp("t0"), source=IRConst(10)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _dse_only(body)
		assert len(result) == 2
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(10)
		assert isinstance(result[1], IRReturn)

	def test_binop_overwritten_before_use(self):
		"""t0 = t1 + t2; t0 = 42; return t0 -> binop is dead."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("t1"), op="+", right=IRTemp("t2")),
			IRCopy(dest=IRTemp("t0"), source=IRConst(42)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _dse_only(body)
		# t0 = t1 + t2 is dead, but t1 and t2 are used by it.
		# The binop should be eliminated since t0 is immediately overwritten.
		copies = [i for i in result if isinstance(i, IRCopy)]
		assert len(copies) == 1
		assert copies[0].source == IRConst(42)
		assert not any(isinstance(i, IRBinOp) for i in result)

	def test_convert_overwritten_before_use(self):
		"""Convert result is overwritten -> convert is dead."""
		body = [
			IRConvert(dest=IRTemp("t0"), source=IRTemp("t1"), from_type=IRType.INT, to_type=IRType.FLOAT),
			IRCopy(dest=IRTemp("t0"), source=IRConst(0)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _dse_only(body)
		assert not any(isinstance(i, IRConvert) for i in result)

	def test_live_copy_not_eliminated(self):
		"""t0 = 5; return t0 -> copy is live, must be kept."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _dse_only(body)
		assert len(result) == 2
		assert isinstance(result[0], IRCopy)

	def test_copy_used_between_defs(self):
		"""t0 = 5; t1 = t0; t0 = 10; return t1 -> first copy is live."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRCopy(dest=IRTemp("t1"), source=IRTemp("t0")),
			IRCopy(dest=IRTemp("t0"), source=IRConst(10)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _dse_only(body)
		# t0 = 5 is live (used by t1 = t0), t0 = 10 is dead (t0 not used after)
		copies = [i for i in result if isinstance(i, IRCopy)]
		assert len(copies) == 2
		assert copies[0].source == IRConst(5)
		assert copies[1].source == IRTemp("t0")


class TestDeadStoreElimination:
	"""Test elimination of IRStore where the same address is stored again without a load."""

	def test_store_overwritten(self):
		"""*addr = 5; *addr = 10 -> first store is dead."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRStore(address=IRTemp("addr"), value=IRConst(5)),
			IRStore(address=IRTemp("addr"), value=IRConst(10)),
			IRReturn(value=IRTemp("addr")),
		]
		result = _dse_only(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) == 1
		assert stores[0].value == IRConst(10)

	def test_store_with_intervening_load(self):
		"""*addr = 5; t = *addr; *addr = 10 -> first store is live."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRStore(address=IRTemp("addr"), value=IRConst(5)),
			IRLoad(dest=IRTemp("t"), address=IRTemp("addr")),
			IRStore(address=IRTemp("addr"), value=IRConst(10)),
			IRReturn(value=IRTemp("t")),
		]
		result = _dse_only(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) == 2

	def test_store_with_intervening_call(self):
		"""*addr = 5; call foo(); *addr = 10 -> first store preserved (call may read)."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRStore(address=IRTemp("addr"), value=IRConst(5)),
			IRCall(dest=IRTemp("r"), function_name="foo"),
			IRStore(address=IRTemp("addr"), value=IRConst(10)),
			IRReturn(value=IRTemp("r")),
		]
		result = _dse_only(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) == 2

	def test_store_to_different_addresses(self):
		"""*a = 5; *b = 10 -> both live (different addresses)."""
		body = [
			IRAlloc(dest=IRTemp("a"), size=4),
			IRAlloc(dest=IRTemp("b"), size=4),
			IRStore(address=IRTemp("a"), value=IRConst(5)),
			IRStore(address=IRTemp("b"), value=IRConst(10)),
			IRLoad(dest=IRTemp("t0"), address=IRTemp("a")),
			IRLoad(dest=IRTemp("t1"), address=IRTemp("b")),
			IRReturn(value=IRTemp("t0")),
		]
		result = _dse_only(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) == 2

	def test_aliased_address_not_eliminated(self):
		"""If address is taken via IRAddrOf, stores are not eliminated."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRAddrOf(dest=IRTemp("ptr"), source=IRTemp("addr")),
			IRStore(address=IRTemp("addr"), value=IRConst(5)),
			IRStore(address=IRTemp("addr"), value=IRConst(10)),
			IRReturn(value=IRTemp("ptr")),
		]
		result = _dse_only(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) == 2

	def test_multiple_dead_stores(self):
		"""*addr = 1; *addr = 2; *addr = 3 -> only last store survives."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRStore(address=IRTemp("addr"), value=IRConst(1)),
			IRStore(address=IRTemp("addr"), value=IRConst(2)),
			IRStore(address=IRTemp("addr"), value=IRConst(3)),
			IRLoad(dest=IRTemp("t"), address=IRTemp("addr")),
			IRReturn(value=IRTemp("t")),
		]
		result = _dse_only(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) == 1
		assert stores[0].value == IRConst(3)


class TestDSEAcrossBlocks:
	"""Test that DSE respects control flow boundaries for liveness."""

	def test_copy_live_across_branch(self):
		"""A copy whose dest is used in another branch must not be eliminated."""
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
		result = _dse_only(body)
		# t0 = 5 is live because it's used in the L2 branch
		first_copies = [
			i for i in result
			if isinstance(i, IRCopy) and i.source == IRConst(5)
		]
		assert len(first_copies) == 1

	def test_store_not_eliminated_across_blocks(self):
		"""IRStore at end of block is preserved (loaded in successor)."""
		body = [
			IRAlloc(dest=IRTemp("addr"), size=4),
			IRStore(address=IRTemp("addr"), value=IRConst(5)),
			IRJump(target="L1"),
			IRLabelInstr(name="L1"),
			IRLoad(dest=IRTemp("t"), address=IRTemp("addr")),
			IRReturn(value=IRTemp("t")),
		]
		result = _dse_only(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert len(stores) == 1


class TestDSEIntegration:
	"""Test DSE as part of the full optimization pipeline."""

	def test_dead_store_removed_in_full_opt(self):
		"""Full optimizer removes dead stores."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRCopy(dest=IRTemp("t0"), source=IRConst(10)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _opt(body)
		# After full optimization, only the live copy and return remain
		copies = [i for i in result if isinstance(i, IRCopy)]
		assert len(copies) <= 1

	def test_dead_irstore_removed_in_full_opt(self):
		"""Full optimizer removes dead IRStore instructions."""
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

	def test_empty_body(self):
		"""DSE handles empty body gracefully."""
		result = _dse_only([])
		assert result == []

	def test_no_dead_stores(self):
		"""When there are no dead stores, body is unchanged."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(5)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _dse_only(body)
		assert len(result) == 2

	def test_unary_op_dead(self):
		"""Dead unary op eliminated by DSE."""
		body = [
			IRUnaryOp(dest=IRTemp("t0"), op="-", operand=IRTemp("t1")),
			IRCopy(dest=IRTemp("t0"), source=IRConst(0)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _dse_only(body)
		assert not any(isinstance(i, IRUnaryOp) for i in result)
