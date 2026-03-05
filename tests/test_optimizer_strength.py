"""Tests for IR-level algebraic strength reduction pass."""

from compiler.ir import (
	IRBinOp,
	IRConst,
	IRCopy,
	IRFunction,
	IRProgram,
	IRReturn,
	IRTemp,
	IRType,
)
from compiler.optimizer import IROptimizer


def _make_func(body, name="test", params=None):
	"""Helper to wrap instructions in a single-function program."""
	return IRProgram(functions=[
		IRFunction(name=name, params=params or [], body=body, return_type=IRType.INT)
	])


def _strength_only(body):
	"""Run only strength reduction (single pass)."""
	return IROptimizer()._strength_reduction(body)


def _opt(body):
	"""Optimize a function body and return the resulting instructions."""
	prog = _make_func(body)
	result = IROptimizer().optimize(prog)
	return result.functions[0].body


# -- Multiply by power-of-2 -> left shift --

class TestMultiplyPowerOf2:
	def test_mul_by_2(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="*", right=IRConst(2)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "<<"
		assert isinstance(result[0].right, IRConst) and result[0].right.value == 1

	def test_mul_by_4(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="*", right=IRConst(4)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "<<"
		assert result[0].right.value == 2

	def test_mul_by_8(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="*", right=IRConst(8)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "<<"
		assert result[0].right.value == 3

	def test_mul_by_16(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="*", right=IRConst(16)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "<<"
		assert result[0].right.value == 4

	def test_mul_by_256(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="*", right=IRConst(256)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "<<"
		assert result[0].right.value == 8

	def test_mul_by_1024(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="*", right=IRConst(1024)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "<<"
		assert result[0].right.value == 10

	def test_mul_by_power_of_2_on_left(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(8), op="*", right=IRTemp("x")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "<<"
		assert result[0].left == IRTemp("x")
		assert result[0].right.value == 3

	def test_mul_by_non_power_of_2_unchanged(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="*", right=IRConst(3)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "*"

	def test_mul_by_6_unchanged(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="*", right=IRConst(6)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "*"


# -- Divide by power-of-2 -> right shift --

class TestDividePowerOf2:
	def test_div_by_2(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="/", right=IRConst(2)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == ">>"
		assert result[0].right.value == 1

	def test_div_by_4(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="/", right=IRConst(4)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == ">>"
		assert result[0].right.value == 2

	def test_div_by_8(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="/", right=IRConst(8)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == ">>"
		assert result[0].right.value == 3

	def test_div_by_16(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="/", right=IRConst(16)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == ">>"
		assert result[0].right.value == 4

	def test_div_by_1(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="/", right=IRConst(1)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRTemp("x")

	def test_div_by_non_power_of_2_unchanged(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="/", right=IRConst(3)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "/"

	def test_div_by_5_unchanged(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="/", right=IRConst(5)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "/"

	def test_div_preserves_dest(self):
		body = [
			IRBinOp(dest=IRTemp("result"), left=IRTemp("x"), op="/", right=IRConst(32)),
			IRReturn(value=IRTemp("result")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].dest == IRTemp("result")
		assert result[0].op == ">>"
		assert result[0].right.value == 5


# -- Modulo by power-of-2 -> bitwise AND --

class TestModuloPowerOf2:
	def test_mod_by_2(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="%", right=IRConst(2)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "&"
		assert result[0].right.value == 1

	def test_mod_by_4(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="%", right=IRConst(4)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "&"
		assert result[0].right.value == 3

	def test_mod_by_8(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="%", right=IRConst(8)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "&"
		assert result[0].right.value == 7

	def test_mod_by_16(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="%", right=IRConst(16)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "&"
		assert result[0].right.value == 15

	def test_mod_by_256(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="%", right=IRConst(256)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "&"
		assert result[0].right.value == 255

	def test_mod_by_1(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="%", right=IRConst(1)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(0)

	def test_mod_by_non_power_of_2_unchanged(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="%", right=IRConst(3)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "%"

	def test_mod_preserves_dest(self):
		body = [
			IRBinOp(dest=IRTemp("result"), left=IRTemp("x"), op="%", right=IRConst(64)),
			IRReturn(value=IRTemp("result")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].dest == IRTemp("result")
		assert result[0].op == "&"
		assert result[0].right.value == 63


# -- Multiply by 0 -> 0 --

class TestMultiplyByZero:
	def test_mul_zero_right(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="*", right=IRConst(0)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(0)

	def test_mul_zero_left(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(0), op="*", right=IRTemp("x")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(0)


# -- Multiply by 1 -> copy --

class TestMultiplyByOne:
	def test_mul_one_right(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="*", right=IRConst(1)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRTemp("x")

	def test_mul_one_left(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(1), op="*", right=IRTemp("x")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRTemp("x")


# -- Integration: strength reduction within full optimizer pipeline --

class TestStrengthReductionIntegration:
	def test_mul_power2_in_pipeline(self):
		"""Multiply by power-of-2 gets reduced to shift in full optimization."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="*", right=IRConst(8)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _opt(body)
		# After full optimization, the multiply should become a shift
		binops = [i for i in result if isinstance(i, IRBinOp)]
		shifts = [i for i in binops if i.op == "<<"]
		muls = [i for i in binops if i.op == "*"]
		assert len(muls) == 0, "Multiply should be eliminated"
		assert len(shifts) == 1 or any(isinstance(i, IRCopy) for i in result)

	def test_div_power2_in_pipeline(self):
		"""Divide by power-of-2 gets reduced to shift in full optimization."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="/", right=IRConst(4)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _opt(body)
		binops = [i for i in result if isinstance(i, IRBinOp)]
		shifts = [i for i in binops if i.op == ">>"]
		divs = [i for i in binops if i.op == "/"]
		assert len(divs) == 0, "Divide should be eliminated"
		assert len(shifts) == 1 or any(isinstance(i, IRCopy) for i in result)

	def test_mod_power2_in_pipeline(self):
		"""Modulo by power-of-2 gets reduced to AND in full optimization."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="%", right=IRConst(16)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _opt(body)
		binops = [i for i in result if isinstance(i, IRBinOp)]
		ands = [i for i in binops if i.op == "&"]
		mods = [i for i in binops if i.op == "%"]
		assert len(mods) == 0, "Modulo should be eliminated"
		assert len(ands) == 1

	def test_combined_strength_reductions(self):
		"""Multiple strength reductions applied in the same function."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="*", right=IRConst(4)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("x"), op="/", right=IRConst(8)),
			IRBinOp(dest=IRTemp("t3"), left=IRTemp("x"), op="%", right=IRConst(16)),
			IRBinOp(dest=IRTemp("t4"), left=IRTemp("x"), op="*", right=IRConst(0)),
			IRBinOp(dest=IRTemp("t5"), left=IRTemp("x"), op="*", right=IRConst(1)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		# t1: * 4 -> << 2
		assert isinstance(result[0], IRBinOp) and result[0].op == "<<"
		# t2: / 8 -> >> 3
		assert isinstance(result[1], IRBinOp) and result[1].op == ">>"
		# t3: % 16 -> & 15
		assert isinstance(result[2], IRBinOp) and result[2].op == "&"
		# t4: * 0 -> copy 0
		assert isinstance(result[3], IRCopy) and result[3].source == IRConst(0)
		# t5: * 1 -> copy x
		assert isinstance(result[4], IRCopy) and result[4].source == IRTemp("x")

	def test_large_power_of_2(self):
		"""Strength reduction works with large powers of 2."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="*", right=IRConst(2**20)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("x"), op="/", right=IRConst(2**16)),
			IRBinOp(dest=IRTemp("t3"), left=IRTemp("x"), op="%", right=IRConst(2**12)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp) and result[0].op == "<<" and result[0].right.value == 20
		assert isinstance(result[1], IRBinOp) and result[1].op == ">>" and result[1].right.value == 16
		assert isinstance(result[2], IRBinOp) and result[2].op == "&" and result[2].right.value == 2**12 - 1
