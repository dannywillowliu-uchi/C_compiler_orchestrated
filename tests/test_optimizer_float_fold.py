"""Tests for float constant folding in the optimizer."""

from compiler.ir import (
	IRBinOp,
	IRConst,
	IRCopy,
	IRFloatConst,
	IRFunction,
	IRProgram,
	IRReturn,
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


def _fold_only(body):
	return IROptimizer()._constant_fold(body)


# -- Float Arithmetic Folding --


class TestFloatArithmeticFolding:
	def test_float_add(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRFloatConst(1.5), op="+", right=IRFloatConst(2.5))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert isinstance(result[0].source, IRFloatConst)
		assert result[0].source.value == 4.0

	def test_float_sub(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRFloatConst(5.0), op="-", right=IRFloatConst(1.5))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source.value == 3.5

	def test_float_mul(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRFloatConst(3.0), op="*", right=IRFloatConst(2.5))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source.value == 7.5

	def test_float_div(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRFloatConst(10.0), op="/", right=IRFloatConst(4.0))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source.value == 2.5

	def test_float_div_by_zero_preserved(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRFloatConst(1.0), op="/", right=IRFloatConst(0.0))]
		result = _fold_only(body)
		assert isinstance(result[0], IRBinOp)

	def test_float_result_type_preserved(self):
		"""Float folding preserves the ir_type (FLOAT vs DOUBLE)."""
		body = [IRBinOp(
			dest=IRTemp("t0"),
			left=IRFloatConst(1.0, ir_type=IRType.DOUBLE),
			op="+",
			right=IRFloatConst(2.0, ir_type=IRType.DOUBLE),
		)]
		result = _fold_only(body)
		assert isinstance(result[0].source, IRFloatConst)
		assert result[0].source.ir_type == IRType.DOUBLE
		assert result[0].source.value == 3.0


# -- Float Comparison Folding --


class TestFloatComparisonFolding:
	def test_float_less_than(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRFloatConst(1.0), op="<", right=IRFloatConst(2.0))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(1)

	def test_float_greater_than(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRFloatConst(3.0), op=">", right=IRFloatConst(1.0))]
		result = _fold_only(body)
		assert result[0].source == IRConst(1)

	def test_float_less_equal(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRFloatConst(2.0), op="<=", right=IRFloatConst(2.0))]
		result = _fold_only(body)
		assert result[0].source == IRConst(1)

	def test_float_greater_equal(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRFloatConst(1.0), op=">=", right=IRFloatConst(2.0))]
		result = _fold_only(body)
		assert result[0].source == IRConst(0)

	def test_float_equal(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRFloatConst(3.14), op="==", right=IRFloatConst(3.14))]
		result = _fold_only(body)
		assert result[0].source == IRConst(1)

	def test_float_not_equal(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRFloatConst(1.0), op="!=", right=IRFloatConst(2.0))]
		result = _fold_only(body)
		assert result[0].source == IRConst(1)

	def test_float_comparison_returns_int(self):
		"""Float comparisons must return IRConst (int), not IRFloatConst."""
		for op in ["<", ">", "<=", ">=", "==", "!="]:
			body = [IRBinOp(dest=IRTemp("t0"), left=IRFloatConst(1.0), op=op, right=IRFloatConst(2.0))]
			result = _fold_only(body)
			assert isinstance(result[0].source, IRConst), f"Comparison {op} should return IRConst"


# -- Mixed Int/Float Folding --


class TestMixedIntFloatFolding:
	def test_int_left_float_right(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(2), op="+", right=IRFloatConst(1.5))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert isinstance(result[0].source, IRFloatConst)
		assert result[0].source.value == 3.5

	def test_float_left_int_right(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRFloatConst(3.0), op="*", right=IRConst(4))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert isinstance(result[0].source, IRFloatConst)
		assert result[0].source.value == 12.0

	def test_mixed_comparison_returns_int(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(2), op="<", right=IRFloatConst(2.5))]
		result = _fold_only(body)
		assert isinstance(result[0].source, IRConst)
		assert result[0].source.value == 1

	def test_mixed_div(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(7), op="/", right=IRFloatConst(2.0))]
		result = _fold_only(body)
		assert isinstance(result[0].source, IRFloatConst)
		assert result[0].source.value == 3.5


# -- Float Unary Folding --


class TestFloatUnaryFolding:
	def test_float_negate(self):
		body = [IRUnaryOp(dest=IRTemp("t0"), op="-", operand=IRFloatConst(3.14))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert isinstance(result[0].source, IRFloatConst)
		assert result[0].source.value == -3.14

	def test_float_negate_preserves_type(self):
		body = [IRUnaryOp(dest=IRTemp("t0"), op="-", operand=IRFloatConst(1.0, ir_type=IRType.DOUBLE))]
		result = _fold_only(body)
		assert result[0].source.ir_type == IRType.DOUBLE

	def test_float_unsupported_unary_preserved(self):
		"""Bitwise not on float should not fold."""
		body = [IRUnaryOp(dest=IRTemp("t0"), op="~", operand=IRFloatConst(1.0))]
		result = _fold_only(body)
		assert isinstance(result[0], IRUnaryOp)


# -- Full Pipeline Integration --


class TestFloatFoldingFullPipeline:
	def test_chained_float_fold(self):
		"""(1.5 + 2.5) * 3.0 should fold completely through propagation."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRFloatConst(1.5), op="+", right=IRFloatConst(2.5)),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="*", right=IRFloatConst(3.0)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _opt(body)
		assert len(result) == 1
		assert isinstance(result[0], IRReturn)
		assert isinstance(result[0].value, IRFloatConst)
		assert result[0].value.value == 12.0

	def test_float_fold_with_dce(self):
		"""Dead float computation is eliminated."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRFloatConst(1.0), op="+", right=IRFloatConst(2.0)),
			IRReturn(value=None),
		]
		result = _opt(body)
		assert len(result) == 1
		assert isinstance(result[0], IRReturn)

	def test_negate_then_add(self):
		"""-2.0 + 5.0 = 3.0 through full pipeline."""
		body = [
			IRUnaryOp(dest=IRTemp("t0"), op="-", operand=IRFloatConst(2.0)),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="+", right=IRFloatConst(5.0)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _opt(body)
		assert len(result) == 1
		assert isinstance(result[0].value, IRFloatConst)
		assert result[0].value.value == 3.0

	def test_non_const_float_not_folded(self):
		"""Float binop with a non-const operand is not folded."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("x"), op="+", right=IRFloatConst(1.0)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _opt(body)
		binops = [i for i in result if isinstance(i, IRBinOp)]
		assert len(binops) == 1

	def test_float_unsupported_binop_preserved(self):
		"""Unsupported float binop (like modulo) is not folded."""
		body = [IRBinOp(dest=IRTemp("t0"), left=IRFloatConst(5.0), op="%", right=IRFloatConst(2.0))]
		result = _fold_only(body)
		assert isinstance(result[0], IRBinOp)
