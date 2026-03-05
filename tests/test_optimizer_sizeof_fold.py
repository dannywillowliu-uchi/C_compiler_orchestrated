"""Tests for constant folding of sizeof-based arithmetic in the IR optimizer.

Sizeof expressions resolve to IRConst at IR generation time. The optimizer
should fold binary operations involving these constants into a single constant,
eliminating unnecessary temporaries.
"""

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


def _make_program(body):
	"""Wrap a list of IR instructions in a minimal IRProgram."""
	func = IRFunction(
		name="test",
		params=[],
		body=body,
		return_type=IRType.INT,
		param_types=[],
	)
	return IRProgram(functions=[func])


def _optimized_body(body):
	"""Optimize a function body and return the result instructions."""
	prog = _make_program(body)
	opt = IROptimizer()
	result = opt.optimize(prog)
	return result.functions[0].body


class TestSizeofConstantFolding:
	"""Test that sizeof constants are folded in binary operations."""

	def test_sizeof_plus_literal(self):
		"""sizeof(int) + 1 => IRConst(4) + IRConst(1) => IRConst(5)"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(4), op="+", right=IRConst(1)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) for i in result)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 5

	def test_sizeof_multiply_literal(self):
		"""sizeof(long) * 10 => IRConst(8) * IRConst(10) => IRConst(80)"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(8), op="*", right=IRConst(10)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) for i in result)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 80

	def test_sizeof_subtract(self):
		"""sizeof(long) - sizeof(int) => 8 - 4 => 4"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(8), op="-", right=IRConst(4)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) for i in result)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 4

	def test_sizeof_divide(self):
		"""sizeof(long) / sizeof(int) => 8 / 4 => 2"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(8), op="/", right=IRConst(4)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) for i in result)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 2


class TestSizeofThroughCopyPropagation:
	"""Test that sizeof constants stored in temps are propagated and folded."""

	def test_sizeof_temp_plus_literal(self):
		"""t1 = sizeof(int); t2 = t1 + 5 => folded to 9"""
		body = [
			IRCopy(dest=IRTemp("t1"), source=IRConst(4)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t1"), op="+", right=IRConst(5)),
			IRReturn(value=IRTemp("t2")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) for i in result)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 9

	def test_two_sizeof_temps_added(self):
		"""t1 = sizeof(int); t2 = sizeof(long); t3 = t1 + t2 => 12"""
		body = [
			IRCopy(dest=IRTemp("t1"), source=IRConst(4)),
			IRCopy(dest=IRTemp("t2"), source=IRConst(8)),
			IRBinOp(dest=IRTemp("t3"), left=IRTemp("t1"), op="+", right=IRTemp("t2")),
			IRReturn(value=IRTemp("t3")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) for i in result)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 12

	def test_sizeof_temp_multiply_chain(self):
		"""t1 = sizeof(int); t2 = t1 * 10; t3 = t2 + 1 => 41"""
		body = [
			IRCopy(dest=IRTemp("t1"), source=IRConst(4)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t1"), op="*", right=IRConst(10)),
			IRBinOp(dest=IRTemp("t3"), left=IRTemp("t2"), op="+", right=IRConst(1)),
			IRReturn(value=IRTemp("t3")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) for i in result)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 41

	def test_sizeof_used_multiple_times(self):
		"""Sizeof constant propagated to multiple uses."""
		body = [
			IRCopy(dest=IRTemp("sz"), source=IRConst(4)),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("sz"), op="+", right=IRConst(1)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("sz"), op="*", right=IRConst(3)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		binops = [i for i in result if isinstance(i, IRBinOp)]
		assert len(binops) == 0
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 5


class TestSizeofComparisonFolding:
	"""Test that sizeof comparisons fold to constants."""

	def test_sizeof_equality(self):
		"""sizeof(int) == 4 => 1"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(4), op="==", right=IRConst(4)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 1

	def test_sizeof_inequality(self):
		"""sizeof(int) != sizeof(long) => 1"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(4), op="!=", right=IRConst(8)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 1

	def test_sizeof_less_than(self):
		"""sizeof(char) < sizeof(int) => 1"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(1), op="<", right=IRConst(4)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 1


class TestSizeofBitwiseFolding:
	"""Test that sizeof values in bitwise operations fold correctly."""

	def test_sizeof_shift_left(self):
		"""sizeof(int) << 1 => 4 << 1 => 8"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(4), op="<<", right=IRConst(1)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 8

	def test_sizeof_bitwise_and(self):
		"""sizeof(long) & 0x7 => 8 & 7 => 0"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(8), op="&", right=IRConst(7)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 0


class TestSizeofWithTypedConstants:
	"""Test folding with typed sizeof constants (LONG, CHAR, etc.)."""

	def test_sizeof_long_type_const(self):
		"""sizeof result with LONG type folds correctly."""
		body = [
			IRCopy(dest=IRTemp("t1"), source=IRConst(8, ir_type=IRType.LONG)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t1"), op="+", right=IRConst(2)),
			IRReturn(value=IRTemp("t2")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) for i in result)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 10

	def test_sizeof_char_type_const(self):
		"""sizeof(char) = 1, used in arithmetic."""
		body = [
			IRCopy(dest=IRTemp("t1"), source=IRConst(1, ir_type=IRType.CHAR)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t1"), op="*", right=IRConst(100)),
			IRReturn(value=IRTemp("t2")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) for i in result)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 100


class TestSizeofNoFoldWhenNonConst:
	"""Verify that non-constant operands are NOT incorrectly folded."""

	def test_sizeof_plus_variable_not_folded(self):
		"""sizeof(int) + x should NOT be fully folded (x is unknown)."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(4), op="+", right=IRTemp("x")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		# The return should still reference a temp, not a const
		assert isinstance(ret[0].value, IRTemp)

	def test_variable_times_sizeof_not_folded(self):
		"""n * sizeof(long) should NOT be fully folded."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("n"), op="*", right=IRConst(8)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		# The binop should remain since n is unknown
		binops = [i for i in result if isinstance(i, IRBinOp)]
		assert len(binops) == 1
