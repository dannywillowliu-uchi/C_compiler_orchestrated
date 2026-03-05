"""Tests for redundant load elimination pass in the optimizer."""

from compiler.ir import (
	IRAlloc,
	IRBinOp,
	IRCall,
	IRConst,
	IRCopy,
	IRFunction,
	IRLabelInstr,
	IRLoad,
	IRProgram,
	IRReturn,
	IRStore,
	IRTemp,
	IRType,
)
from compiler.optimizer import IROptimizer


def _make_func(body, name="test", params=None):
	return IRProgram(functions=[
		IRFunction(name=name, params=params or [], body=body, return_type=IRType.INT)
	])


def _rle_only(body):
	"""Run only the redundant load elimination pass."""
	return IROptimizer()._redundant_load_elimination(body)


def _opt(body):
	"""Run full optimization pipeline."""
	prog = _make_func(body)
	result = IROptimizer().optimize(prog)
	return result.functions[0].body


class TestRedundantLoadElimination:
	"""Test elimination of redundant loads from the same address."""

	def test_two_consecutive_loads_same_address(self):
		"""Two loads from the same address -> second becomes a copy."""
		addr = IRTemp("addr")
		body = [
			IRLoad(dest=IRTemp("t0"), address=addr),
			IRLoad(dest=IRTemp("t1"), address=addr),
			IRReturn(value=IRTemp("t1")),
		]
		result = _rle_only(body)
		assert len(result) == 3
		assert isinstance(result[0], IRLoad)
		assert isinstance(result[1], IRCopy)
		assert result[1].dest == IRTemp("t1")
		assert result[1].source == IRTemp("t0")

	def test_three_consecutive_loads_same_address(self):
		"""Three loads from same address -> second and third become copies."""
		addr = IRTemp("addr")
		body = [
			IRLoad(dest=IRTemp("t0"), address=addr),
			IRLoad(dest=IRTemp("t1"), address=addr),
			IRLoad(dest=IRTemp("t2"), address=addr),
			IRReturn(value=IRTemp("t2")),
		]
		result = _rle_only(body)
		assert isinstance(result[0], IRLoad)
		assert isinstance(result[1], IRCopy)
		assert result[1].source == IRTemp("t0")
		assert isinstance(result[2], IRCopy)
		assert result[2].source == IRTemp("t1")

	def test_intervening_store_same_address_invalidates(self):
		"""A store to the same address between loads prevents elimination."""
		addr = IRTemp("addr")
		body = [
			IRLoad(dest=IRTemp("t0"), address=addr),
			IRStore(address=addr, value=IRConst(42)),
			IRLoad(dest=IRTemp("t1"), address=addr),
			IRReturn(value=IRTemp("t1")),
		]
		result = _rle_only(body)
		assert isinstance(result[0], IRLoad)
		assert isinstance(result[1], IRStore)
		assert isinstance(result[2], IRLoad)  # not eliminated

	def test_intervening_store_different_address_preserves(self):
		"""A store to a different address does not prevent elimination."""
		addr_a = IRTemp("addr_a")
		addr_b = IRTemp("addr_b")
		body = [
			IRLoad(dest=IRTemp("t0"), address=addr_a),
			IRStore(address=addr_b, value=IRConst(42)),
			IRLoad(dest=IRTemp("t1"), address=addr_a),
			IRReturn(value=IRTemp("t1")),
		]
		result = _rle_only(body)
		assert isinstance(result[0], IRLoad)
		assert isinstance(result[1], IRStore)
		assert isinstance(result[2], IRCopy)  # eliminated
		assert result[2].source == IRTemp("t0")

	def test_call_invalidates_all_loads(self):
		"""A function call invalidates all tracked loads."""
		addr = IRTemp("addr")
		body = [
			IRLoad(dest=IRTemp("t0"), address=addr),
			IRCall(dest=IRTemp("t_call"), function_name="foo", args=[]),
			IRLoad(dest=IRTemp("t1"), address=addr),
			IRReturn(value=IRTemp("t1")),
		]
		result = _rle_only(body)
		assert isinstance(result[0], IRLoad)
		assert isinstance(result[2], IRLoad)  # not eliminated due to call

	def test_label_invalidates_all_loads(self):
		"""A label (join point) invalidates all tracked loads."""
		addr = IRTemp("addr")
		body = [
			IRLoad(dest=IRTemp("t0"), address=addr),
			IRLabelInstr(name="L1"),
			IRLoad(dest=IRTemp("t1"), address=addr),
			IRReturn(value=IRTemp("t1")),
		]
		result = _rle_only(body)
		assert isinstance(result[0], IRLoad)
		assert isinstance(result[2], IRLoad)  # not eliminated due to label

	def test_different_addresses_not_eliminated(self):
		"""Loads from different addresses are not eliminated."""
		body = [
			IRLoad(dest=IRTemp("t0"), address=IRTemp("addr_a")),
			IRLoad(dest=IRTemp("t1"), address=IRTemp("addr_b")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _rle_only(body)
		assert isinstance(result[0], IRLoad)
		assert isinstance(result[1], IRLoad)

	def test_address_redefinition_invalidates(self):
		"""If the address temp is redefined, the load map entry is invalidated."""
		body = [
			IRLoad(dest=IRTemp("t0"), address=IRTemp("addr")),
			IRBinOp(dest=IRTemp("addr"), left=IRTemp("addr"), op="+", right=IRConst(4)),
			IRLoad(dest=IRTemp("t1"), address=IRTemp("addr")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _rle_only(body)
		assert isinstance(result[0], IRLoad)
		assert isinstance(result[2], IRLoad)  # not eliminated, addr changed

	def test_load_dest_overwritten_invalidates(self):
		"""If the load dest temp is overwritten, entries referencing it are removed."""
		addr = IRTemp("addr")
		body = [
			IRLoad(dest=IRTemp("t0"), address=addr),
			IRCopy(dest=IRTemp("t0"), source=IRConst(99)),
			IRLoad(dest=IRTemp("t1"), address=addr),
			IRReturn(value=IRTemp("t1")),
		]
		result = _rle_only(body)
		assert isinstance(result[0], IRLoad)
		assert isinstance(result[2], IRLoad)  # not eliminated, t0 was overwritten

	def test_multiple_addresses_tracked(self):
		"""Multiple addresses can be tracked simultaneously."""
		addr_a = IRTemp("addr_a")
		addr_b = IRTemp("addr_b")
		body = [
			IRLoad(dest=IRTemp("t0"), address=addr_a),
			IRLoad(dest=IRTemp("t1"), address=addr_b),
			IRLoad(dest=IRTemp("t2"), address=addr_a),
			IRLoad(dest=IRTemp("t3"), address=addr_b),
			IRReturn(value=IRTemp("t3")),
		]
		result = _rle_only(body)
		assert isinstance(result[0], IRLoad)
		assert isinstance(result[1], IRLoad)
		assert isinstance(result[2], IRCopy)
		assert result[2].source == IRTemp("t0")
		assert isinstance(result[3], IRCopy)
		assert result[3].source == IRTemp("t1")

	def test_full_optimization_eliminates_redundant_load(self):
		"""Redundant load is eliminated through full optimization pipeline."""
		addr = IRTemp("addr")
		body = [
			IRAlloc(dest=addr, size=4),
			IRStore(address=addr, value=IRConst(10)),
			IRLoad(dest=IRTemp("t0"), address=addr),
			IRLoad(dest=IRTemp("t1"), address=addr),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t0"), op="+", right=IRTemp("t1")),
			IRReturn(value=IRTemp("t2")),
		]
		result = _opt(body)
		# After optimization, the redundant load should be gone
		loads = [i for i in result if isinstance(i, IRLoad)]
		assert len(loads) <= 1

	def test_empty_body(self):
		"""Empty body is handled gracefully."""
		result = _rle_only([])
		assert result == []

	def test_no_loads(self):
		"""Body with no loads returns unchanged."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(1)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _rle_only(body)
		assert len(result) == 2
