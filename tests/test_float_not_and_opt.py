"""Tests for float logical-not codegen and optimizer IRFunction field preservation."""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRFloatConst,
	IRFunction,
	IRProgram,
	IRReturn,
	IRTemp,
	IRType,
	IRUnaryOp,
)
from compiler.optimizer import IROptimizer


def _gen(func: IRFunction) -> str:
	"""Helper: generate assembly for a single-function program."""
	return CodeGenerator().generate(IRProgram([func]))


# ---------------------------------------------------------------------------
# Float logical-not codegen
# ---------------------------------------------------------------------------


class TestFloatLogicalNot:
	def _make_not_func(self, value: float, ir_type: IRType = IRType.DOUBLE) -> IRFunction:
		"""Build a function that returns !value for a float/double constant."""
		return IRFunction(
			name="test_not",
			params=[],
			body=[
				IRUnaryOp(dest=IRTemp("t0"), op="!", operand=IRFloatConst(value, ir_type), ir_type=ir_type),
				IRReturn(IRTemp("t0")),
			],
			return_type=IRType.INT,
		)

	def test_not_zero_produces_one(self) -> None:
		"""!0.0 should produce 1."""
		asm = _gen(self._make_not_func(0.0))
		# Should contain ucomis for the comparison and sete for the result
		assert "ucomisd" in asm
		assert "sete" in asm

	def test_not_positive_produces_zero(self) -> None:
		"""!1.5 should produce 0."""
		asm = _gen(self._make_not_func(1.5))
		assert "ucomisd" in asm
		assert "sete" in asm

	def test_not_negative_produces_zero(self) -> None:
		"""!-3.14 should produce 0."""
		asm = _gen(self._make_not_func(-3.14))
		assert "ucomisd" in asm

	def test_not_float_type(self) -> None:
		"""! on float (not double) uses ucomiss."""
		asm = _gen(self._make_not_func(0.0, IRType.FLOAT))
		assert "ucomiss" in asm
		assert "sete" in asm

	def test_not_stores_integer_result(self) -> None:
		"""Logical not on float stores an integer result via movzbq."""
		asm = _gen(self._make_not_func(1.0))
		assert "movzbq" in asm

	def test_no_crash_on_float_not(self) -> None:
		"""Previously raised ValueError: Unsupported float unary operator: !"""
		# Just confirm no exception is raised for each case
		for val in [0.0, 1.5, -3.14]:
			_gen(self._make_not_func(val))


# ---------------------------------------------------------------------------
# Optimizer IRFunction field preservation
# ---------------------------------------------------------------------------


class TestOptimizerFieldPreservation:
	def test_param_types_preserved(self) -> None:
		"""Optimized function should retain param_types from the original."""
		func = IRFunction(
			name="foo",
			params=[IRTemp("a"), IRTemp("b")],
			body=[IRReturn(IRTemp("a"))],
			return_type=IRType.INT,
			param_types=[IRType.INT, IRType.DOUBLE],
		)
		program = IRProgram(functions=[func])
		optimized = IROptimizer().optimize(program)
		opt_func = optimized.functions[0]
		assert opt_func.param_types == [IRType.INT, IRType.DOUBLE]

	def test_storage_class_preserved(self) -> None:
		"""Optimized function should retain storage_class from the original."""
		func = IRFunction(
			name="bar",
			params=[],
			body=[IRReturn(None)],
			return_type=IRType.INT,
			storage_class="static",
		)
		program = IRProgram(functions=[func])
		optimized = IROptimizer().optimize(program)
		opt_func = optimized.functions[0]
		assert opt_func.storage_class == "static"

	def test_default_fields_preserved(self) -> None:
		"""When param_types/storage_class are defaults, they remain so after optimization."""
		func = IRFunction(
			name="baz",
			params=[],
			body=[IRReturn(None)],
			return_type=IRType.INT,
		)
		program = IRProgram(functions=[func])
		optimized = IROptimizer().optimize(program)
		opt_func = optimized.functions[0]
		assert opt_func.param_types == []
		assert opt_func.storage_class is None

	def test_extern_storage_class_preserved(self) -> None:
		"""Extern storage class should survive optimization."""
		func = IRFunction(
			name="ext_fn",
			params=[IRTemp("x")],
			body=[IRReturn(IRTemp("x"))],
			return_type=IRType.INT,
			param_types=[IRType.INT],
			storage_class="extern",
		)
		program = IRProgram(functions=[func])
		optimized = IROptimizer().optimize(program)
		opt_func = optimized.functions[0]
		assert opt_func.storage_class == "extern"
		assert opt_func.param_types == [IRType.INT]
