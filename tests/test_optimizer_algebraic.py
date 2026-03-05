"""Tests for the IR-level algebraic simplification pass."""

from compiler.ir import (
	IRBinOp,
	IRConst,
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


# -- Identity patterns: x + 0 -> x --

class TestAddZero:
	def test_add_zero_right(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="+", right=IRConst(0)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		# The binop should be eliminated; copy prop may fold the copy into the return
		assert not any(isinstance(i, IRBinOp) and i.op == "+" for i in result)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert isinstance(ret[0].value, IRTemp) and ret[0].value.name == "x"

	def test_add_zero_left(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(0), op="+", right=IRTemp("x")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) and i.op == "+" for i in result)


# -- Identity patterns: x - 0 -> x --

class TestSubZero:
	def test_sub_zero(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="-", right=IRConst(0)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) and i.op == "-" for i in result)


# -- Identity patterns: x * 1 -> x --

class TestMulOne:
	def test_mul_one_right(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="*", right=IRConst(1)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) and i.op == "*" for i in result)

	def test_mul_one_left(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(1), op="*", right=IRTemp("x")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) and i.op == "*" for i in result)


# -- Absorption patterns: x * 0 -> 0 --

class TestMulZero:
	def test_mul_zero_right(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="*", right=IRConst(0)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		# Should produce a constant 0 return
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 0

	def test_mul_zero_left(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(0), op="*", right=IRTemp("x")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 0


# -- Bitwise AND with zero: x & 0 -> 0 --

class TestAndZero:
	def test_and_zero_right(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="&", right=IRConst(0)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 0

	def test_and_zero_left(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(0), op="&", right=IRTemp("x")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 0


# -- Bitwise OR with zero: x | 0 -> x --

class TestOrZero:
	def test_or_zero_right(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="|", right=IRConst(0)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) and i.op == "|" for i in result)

	def test_or_zero_left(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(0), op="|", right=IRTemp("x")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) and i.op == "|" for i in result)


# -- XOR with zero: x ^ 0 -> x --

class TestXorZero:
	def test_xor_zero_right(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="^", right=IRConst(0)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) and i.op == "^" for i in result)

	def test_xor_zero_left(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(0), op="^", right=IRTemp("x")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) and i.op == "^" for i in result)


# -- Shift by zero: x << 0 -> x, x >> 0 -> x --

class TestShiftZero:
	def test_lshift_zero(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="<<", right=IRConst(0)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) and i.op == "<<" for i in result)

	def test_rshift_zero(self):
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op=">>", right=IRConst(0)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) and i.op == ">>" for i in result)


# -- Same-operand patterns --

class TestSameOperand:
	def test_sub_self(self):
		"""x - x -> 0"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="-", right=IRTemp("x")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 0

	def test_and_self(self):
		"""x & x -> x"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="&", right=IRTemp("x")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) and i.op == "&" for i in result)

	def test_or_self(self):
		"""x | x -> x"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="|", right=IRTemp("x")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) and i.op == "|" for i in result)

	def test_xor_self(self):
		"""x ^ x -> 0"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="^", right=IRTemp("x")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert isinstance(ret[0].value, IRConst) and ret[0].value.value == 0


# -- Chained simplifications --

class TestChainedSimplifications:
	def test_chain_add_zero_then_mul_one(self):
		"""(x + 0) * 1 should simplify to just x."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="+", right=IRConst(0)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t1"), op="*", right=IRConst(1)),
			IRReturn(value=IRTemp("t2")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) for i in result)

	def test_chain_or_zero_then_and_self(self):
		"""(x | 0) & (x | 0) -> x."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="|", right=IRConst(0)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t1"), op="&", right=IRTemp("t1")),
			IRReturn(value=IRTemp("t2")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) for i in result)

	def test_chain_sub_zero_then_xor_zero(self):
		"""(x - 0) ^ 0 -> x."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="-", right=IRConst(0)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t1"), op="^", right=IRConst(0)),
			IRReturn(value=IRTemp("t2")),
		]
		result = _optimized_body(body)
		assert not any(isinstance(i, IRBinOp) for i in result)


# -- Non-matching cases (should NOT simplify) --

class TestNoSimplification:
	def test_add_nonzero_preserved(self):
		"""x + 5 should not be simplified."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="+", right=IRConst(5)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert any(isinstance(i, IRBinOp) and i.op == "+" for i in result)

	def test_mul_nonone_preserved(self):
		"""x * 3 should not be simplified to a copy (may become shift+add but stays as binop)."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="*", right=IRConst(3)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		# 3 is not a power of 2, so it stays as multiply
		assert any(isinstance(i, IRBinOp) for i in result)

	def test_and_nonzero_preserved(self):
		"""x & 0xFF should not be simplified."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="&", right=IRConst(0xFF)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert any(isinstance(i, IRBinOp) and i.op == "&" for i in result)

	def test_sub_different_temps_preserved(self):
		"""x - y (different temps) should not be simplified to 0."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="-", right=IRTemp("y")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert any(isinstance(i, IRBinOp) and i.op == "-" for i in result)

	def test_shift_nonzero_preserved(self):
		"""x << 2 should not be simplified."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="<<", right=IRConst(2)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		assert any(isinstance(i, IRBinOp) for i in result)


# -- Edge cases --

class TestEdgeCases:
	def test_multiple_simplifications_in_sequence(self):
		"""Multiple algebraic simplifications in a row."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="+", right=IRConst(0)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("y"), op="*", right=IRConst(0)),
			IRBinOp(dest=IRTemp("t3"), left=IRTemp("z"), op="&", right=IRConst(0)),
			IRBinOp(dest=IRTemp("t4"), left=IRTemp("w"), op="|", right=IRConst(0)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _optimized_body(body)
		# None of these should remain as BinOps
		assert not any(isinstance(i, IRBinOp) for i in result)

	def test_algebraic_interacts_with_copy_prop(self):
		"""Algebraic simplification feeds into copy propagation."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("x"), op="+", right=IRConst(0)),
			# t1 should be copy-propagated to x
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t1"), op="*", right=IRConst(1)),
			IRReturn(value=IRTemp("t2")),
		]
		result = _optimized_body(body)
		# After optimization: t1=x (copy), t2=t1=x (copy), return x
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert isinstance(ret[0].value, IRTemp) and ret[0].value.name == "x"
