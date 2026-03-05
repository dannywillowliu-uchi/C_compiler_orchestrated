"""Tests for redundant IRConvert elimination in the optimizer."""

from compiler.ir import (
	IRConst,
	IRConvert,
	IRFloatConst,
	IRFunction,
	IRLabelInstr,
	IRProgram,
	IRReturn,
	IRTemp,
	IRType,
)
from compiler.optimizer import IROptimizer


def _optimize_body(body):
	"""Helper: optimize a single function body and return the optimized body."""
	func = IRFunction(
		name="test", params=[], body=body, return_type=IRType.INT,
	)
	program = IRProgram(functions=[func])
	optimized = IROptimizer().optimize(program)
	return optimized.functions[0].body


class TestNoopConvertElimination:
	"""Pattern 1: Remove IRConvert where from_type == to_type."""

	def test_same_type_int(self) -> None:
		body = [
			IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.INT, to_type=IRType.INT),
			IRReturn(IRTemp("t1")),
		]
		result = _optimize_body(body)
		# The no-op convert should become a copy (or be propagated away)
		assert not any(isinstance(i, IRConvert) for i in result)

	def test_same_type_float(self) -> None:
		body = [
			IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.FLOAT, to_type=IRType.FLOAT),
			IRReturn(IRTemp("t1")),
		]
		result = _optimize_body(body)
		assert not any(isinstance(i, IRConvert) for i in result)

	def test_same_type_long(self) -> None:
		body = [
			IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.LONG, to_type=IRType.LONG),
			IRReturn(IRTemp("t1")),
		]
		result = _optimize_body(body)
		assert not any(isinstance(i, IRConvert) for i in result)

	def test_noop_preserves_value_flow(self) -> None:
		"""After eliminating a no-op convert, the return should use the original source."""
		body = [
			IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.INT, to_type=IRType.INT),
			IRReturn(IRTemp("t1")),
		]
		result = _optimize_body(body)
		ret = [i for i in result if isinstance(i, IRReturn)][0]
		# After copy propagation, the return should reference t0 directly
		assert isinstance(ret.value, IRTemp) and ret.value.name == "t0"


class TestRoundTripConvertElimination:
	"""Pattern 2: Collapse back-to-back conversions where intermediate is wider."""

	def test_int_long_int(self) -> None:
		"""int->long->int with wider intermediate should produce original value."""
		body = [
			IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.INT, to_type=IRType.LONG),
			IRConvert(dest=IRTemp("t2"), source=IRTemp("t1"), from_type=IRType.LONG, to_type=IRType.INT),
			IRReturn(IRTemp("t2")),
		]
		result = _optimize_body(body)
		# Both converts should be eliminated; return should use t0
		ret = [i for i in result if isinstance(i, IRReturn)][0]
		assert isinstance(ret.value, IRTemp) and ret.value.name == "t0"

	def test_char_int_char(self) -> None:
		"""char->int->char with wider intermediate should produce original value."""
		body = [
			IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.CHAR, to_type=IRType.INT),
			IRConvert(dest=IRTemp("t2"), source=IRTemp("t1"), from_type=IRType.INT, to_type=IRType.CHAR),
			IRReturn(IRTemp("t2")),
		]
		result = _optimize_body(body)
		ret = [i for i in result if isinstance(i, IRReturn)][0]
		assert isinstance(ret.value, IRTemp) and ret.value.name == "t0"

	def test_short_long_short(self) -> None:
		"""short->long->short with wider intermediate should produce original value."""
		body = [
			IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.SHORT, to_type=IRType.LONG),
			IRConvert(dest=IRTemp("t2"), source=IRTemp("t1"), from_type=IRType.LONG, to_type=IRType.SHORT),
			IRReturn(IRTemp("t2")),
		]
		result = _optimize_body(body)
		ret = [i for i in result if isinstance(i, IRReturn)][0]
		assert isinstance(ret.value, IRTemp) and ret.value.name == "t0"

	def test_no_converts_remain(self) -> None:
		"""After round-trip elimination, no IRConvert should remain."""
		body = [
			IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.INT, to_type=IRType.LONG),
			IRConvert(dest=IRTemp("t2"), source=IRTemp("t1"), from_type=IRType.LONG, to_type=IRType.INT),
			IRReturn(IRTemp("t2")),
		]
		result = _optimize_body(body)
		assert not any(isinstance(i, IRConvert) for i in result)


class TestChainedConvertCollapse:
	"""Pattern 3: Collapse back-to-back conversions into a single convert."""

	def test_int_long_float(self) -> None:
		"""int->long->float should collapse to int->float."""
		body = [
			IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.INT, to_type=IRType.LONG),
			IRConvert(dest=IRTemp("t2"), source=IRTemp("t1"), from_type=IRType.LONG, to_type=IRType.FLOAT),
			IRReturn(IRTemp("t2")),
		]
		result = _optimize_body(body)
		converts = [i for i in result if isinstance(i, IRConvert)]
		assert len(converts) == 1
		assert converts[0].from_type == IRType.INT
		assert converts[0].to_type == IRType.FLOAT
		assert isinstance(converts[0].source, IRTemp) and converts[0].source.name == "t0"

	def test_char_int_long(self) -> None:
		"""char->int->long should collapse to char->long."""
		body = [
			IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.CHAR, to_type=IRType.INT),
			IRConvert(dest=IRTemp("t2"), source=IRTemp("t1"), from_type=IRType.INT, to_type=IRType.LONG),
			IRReturn(IRTemp("t2")),
		]
		result = _optimize_body(body)
		converts = [i for i in result if isinstance(i, IRConvert)]
		assert len(converts) == 1
		assert converts[0].from_type == IRType.CHAR
		assert converts[0].to_type == IRType.LONG

	def test_chain_with_const_source(self) -> None:
		"""Chained converts with a constant source should be fully folded."""
		body = [
			IRConvert(dest=IRTemp("t1"), source=IRConst(42), from_type=IRType.INT, to_type=IRType.LONG),
			IRConvert(dest=IRTemp("t2"), source=IRTemp("t1"), from_type=IRType.LONG, to_type=IRType.FLOAT),
			IRReturn(IRTemp("t2")),
		]
		result = _optimize_body(body)
		# Both converts fold away: IRConst(42) -> IRFloatConst(42.0)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert isinstance(ret[0].value, IRFloatConst)
		assert ret[0].value.value == 42.0


class TestConvertEliminationEdgeCases:
	"""Edge cases and interactions with other passes."""

	def test_label_breaks_chain(self) -> None:
		"""A label between two converts should prevent chaining."""
		body = [
			IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.INT, to_type=IRType.LONG),
			IRLabelInstr(name="L1"),
			IRConvert(dest=IRTemp("t2"), source=IRTemp("t1"), from_type=IRType.LONG, to_type=IRType.INT),
			IRReturn(IRTemp("t2")),
		]
		result = _optimize_body(body)
		# The label breaks the chain, so the second convert should remain
		converts = [i for i in result if isinstance(i, IRConvert)]
		assert len(converts) >= 1

	def test_dead_intermediate_removed(self) -> None:
		"""When a chained convert is collapsed, DCE should remove the dead first convert."""
		body = [
			IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.INT, to_type=IRType.LONG),
			IRConvert(dest=IRTemp("t2"), source=IRTemp("t1"), from_type=IRType.LONG, to_type=IRType.FLOAT),
			IRReturn(IRTemp("t2")),
		]
		result = _optimize_body(body)
		# t1 is no longer used; DCE should remove its convert
		converts = [i for i in result if isinstance(i, IRConvert)]
		assert len(converts) == 1
		assert converts[0].dest.name == "t2"

	def test_different_convert_not_eliminated(self) -> None:
		"""A genuine type conversion should not be eliminated."""
		body = [
			IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.INT, to_type=IRType.FLOAT),
			IRReturn(IRTemp("t1")),
		]
		result = _optimize_body(body)
		converts = [i for i in result if isinstance(i, IRConvert)]
		assert len(converts) == 1
		assert converts[0].from_type == IRType.INT
		assert converts[0].to_type == IRType.FLOAT

	def test_triple_chain_collapses(self) -> None:
		"""Three chained converts should collapse to one."""
		body = [
			IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.CHAR, to_type=IRType.INT),
			IRConvert(dest=IRTemp("t2"), source=IRTemp("t1"), from_type=IRType.INT, to_type=IRType.LONG),
			IRConvert(dest=IRTemp("t3"), source=IRTemp("t2"), from_type=IRType.LONG, to_type=IRType.FLOAT),
			IRReturn(IRTemp("t3")),
		]
		result = _optimize_body(body)
		converts = [i for i in result if isinstance(i, IRConvert)]
		assert len(converts) == 1
		assert converts[0].from_type == IRType.CHAR
		assert converts[0].to_type == IRType.FLOAT
		assert isinstance(converts[0].source, IRTemp) and converts[0].source.name == "t0"
