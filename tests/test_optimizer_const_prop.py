"""Tests for constant propagation through IRConvert, boolean simplification,
arithmetic constant folding, and redundant IRCopy chain folding."""

from compiler.ir import (
	IRBinOp,
	IRConst,
	IRConvert,
	IRCopy,
	IRFloatConst,
	IRFunction,
	IRLabelInstr,
	IRProgram,
	IRReturn,
	IRTemp,
	IRType,
	IRUnaryOp,
)
from compiler.optimizer import IROptimizer


def _make_func(body, name="test", params=None):
	"""Helper to wrap instructions in a single-function program."""
	return IRProgram(functions=[
		IRFunction(name=name, params=params or [], body=body, return_type=IRType.INT)
	])


def _opt(body):
	"""Optimize a function body and return the resulting instructions."""
	prog = _make_func(body)
	result = IROptimizer().optimize(prog)
	return result.functions[0].body


def _fold_only(body):
	"""Run only the constant folding pass."""
	return IROptimizer()._constant_fold(body)


def _convert_const_only(body):
	"""Run only convert constant propagation."""
	return IROptimizer()._convert_const_propagation(body)


def _bool_only(body):
	"""Run only boolean simplification."""
	return IROptimizer()._boolean_simplification(body)


# ── Arithmetic Constant Propagation (Folding) ──


class TestArithmeticConstantPropagation:
	"""Tests that IRBinOp/IRUnaryOp with all-constant operands are folded."""

	def test_add_two_consts(self):
		"""t0 = 3 + 5 -> t0 = 8"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(3), op="+", right=IRConst(5)),
		]
		result = _fold_only(body)
		assert len(result) == 1
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(8)

	def test_sub_two_consts(self):
		"""t0 = 10 - 3 -> t0 = 7"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(10), op="-", right=IRConst(3)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(7)

	def test_mul_two_consts(self):
		"""t0 = 4 * 6 -> t0 = 24"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(4), op="*", right=IRConst(6)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(24)

	def test_div_two_consts(self):
		"""t0 = 15 / 3 -> t0 = 5"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(15), op="/", right=IRConst(3)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(5)

	def test_div_by_zero_not_folded(self):
		"""t0 = 10 / 0 -> not folded"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(10), op="/", right=IRConst(0)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRBinOp)

	def test_mod_two_consts(self):
		"""t0 = 17 % 5 -> t0 = 2"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(17), op="%", right=IRConst(5)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(2)

	def test_bitwise_and(self):
		"""t0 = 0xFF & 0x0F -> t0 = 15"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(0xFF), op="&", right=IRConst(0x0F)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(15)

	def test_bitwise_or(self):
		"""t0 = 0xF0 | 0x0F -> t0 = 0xFF"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(0xF0), op="|", right=IRConst(0x0F)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(0xFF)

	def test_bitwise_xor(self):
		"""t0 = 0xFF ^ 0x0F -> t0 = 0xF0"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(0xFF), op="^", right=IRConst(0x0F)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(0xF0)

	def test_shift_left(self):
		"""t0 = 1 << 4 -> t0 = 16"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(1), op="<<", right=IRConst(4)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(16)

	def test_shift_right(self):
		"""t0 = 32 >> 2 -> t0 = 8"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(32), op=">>", right=IRConst(2)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(8)

	def test_comparison_less(self):
		"""t0 = 3 < 5 -> t0 = 1"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(3), op="<", right=IRConst(5)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(1)

	def test_comparison_eq(self):
		"""t0 = 3 == 3 -> t0 = 1"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(3), op="==", right=IRConst(3)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(1)

	def test_comparison_neq(self):
		"""t0 = 3 != 5 -> t0 = 1"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(3), op="!=", right=IRConst(5)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(1)

	def test_logical_and_consts(self):
		"""t0 = 1 && 0 -> t0 = 0"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(1), op="&&", right=IRConst(0)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(0)

	def test_logical_or_consts(self):
		"""t0 = 0 || 1 -> t0 = 1"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(0), op="||", right=IRConst(1)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(1)

	def test_unary_negate(self):
		"""t0 = -5 -> t0 = -5"""
		body = [
			IRUnaryOp(dest=IRTemp("t0"), op="-", operand=IRConst(5)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(-5)

	def test_unary_bitwise_not(self):
		"""t0 = ~0 -> t0 = -1"""
		body = [
			IRUnaryOp(dest=IRTemp("t0"), op="~", operand=IRConst(0)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(-1)

	def test_unary_logical_not(self):
		"""t0 = !5 -> t0 = 0"""
		body = [
			IRUnaryOp(dest=IRTemp("t0"), op="!", operand=IRConst(5)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(0)

	def test_non_const_operand_not_folded(self):
		"""t0 = t1 + 5 -> not folded"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("t1"), op="+", right=IRConst(5)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRBinOp)

	def test_float_add_consts(self):
		"""t0 = 1.5 + 2.5 -> t0 = 4.0"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRFloatConst(1.5), op="+", right=IRFloatConst(2.5)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert isinstance(result[0].source, IRFloatConst)
		assert result[0].source.value == 4.0

	def test_float_comparison(self):
		"""t0 = 1.0 < 2.0 -> t0 = 1 (int)"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRFloatConst(1.0), op="<", right=IRFloatConst(2.0)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert isinstance(result[0].source, IRConst)
		assert result[0].source.value == 1

	def test_mixed_int_float(self):
		"""t0 = IRConst(3) + IRFloatConst(2.5) -> t0 = 5.5"""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(3), op="+", right=IRFloatConst(2.5)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert isinstance(result[0].source, IRFloatConst)
		assert result[0].source.value == 5.5

	def test_float_unary_negate(self):
		"""t0 = -3.14 -> t0 = -3.14"""
		body = [
			IRUnaryOp(dest=IRTemp("t0"), op="-", operand=IRFloatConst(3.14)),
		]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert isinstance(result[0].source, IRFloatConst)
		assert result[0].source.value == -3.14

	def test_chained_const_fold_via_full_opt(self):
		"""t1 = 3 + 5; t2 = t1 * 2 -> t2 = 16 after copy prop + refold."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(3), op="+", right=IRConst(5)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t1"), op="*", right=IRConst(2)),
			IRReturn(value=IRTemp("t2")),
		]
		result = _opt(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert ret[0].value == IRConst(16)

	def test_multi_step_propagation(self):
		"""t1 = 10; t2 = t1 + 20; t3 = t2 - 5 -> t3 = 25."""
		body = [
			IRCopy(dest=IRTemp("t1"), source=IRConst(10)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t1"), op="+", right=IRConst(20)),
			IRBinOp(dest=IRTemp("t3"), left=IRTemp("t2"), op="-", right=IRConst(5)),
			IRReturn(value=IRTemp("t3")),
		]
		result = _opt(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert ret[0].value == IRConst(25)

	def test_unary_then_binary_fold(self):
		"""t1 = -3; t2 = t1 + 10 -> t2 = 7."""
		body = [
			IRUnaryOp(dest=IRTemp("t1"), op="-", operand=IRConst(3)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t1"), op="+", right=IRConst(10)),
			IRReturn(value=IRTemp("t2")),
		]
		result = _opt(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert ret[0].value == IRConst(7)


# ── Constant Propagation through IRConvert ──


class TestConvertConstPropagation:
	def test_int_to_int_same_width(self):
		"""convert(int, IRConst(5)) -> IRConst(5)"""
		body = [
			IRConvert(dest=IRTemp("t0"), source=IRConst(5), from_type=IRType.INT, to_type=IRType.INT),
		]
		result = _convert_const_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(5, ir_type=IRType.INT)

	def test_int_to_char_truncation(self):
		"""convert(char, IRConst(300)) -> IRConst(44) (300 & 0xFF = 44, sign-extended)"""
		body = [
			IRConvert(dest=IRTemp("t0"), source=IRConst(300), from_type=IRType.INT, to_type=IRType.CHAR),
		]
		result = _convert_const_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source.value == 44  # 300 & 0xFF = 44

	def test_int_to_char_sign_extension(self):
		"""convert(char, IRConst(200)) -> IRConst(-56) (200 & 0xFF = 200 -> sign extend = -56)"""
		body = [
			IRConvert(dest=IRTemp("t0"), source=IRConst(200), from_type=IRType.INT, to_type=IRType.CHAR),
		]
		result = _convert_const_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source.value == -56

	def test_int_to_short_truncation(self):
		"""convert(short, IRConst(70000)) -> truncated to short range"""
		body = [
			IRConvert(dest=IRTemp("t0"), source=IRConst(70000), from_type=IRType.INT, to_type=IRType.SHORT),
		]
		result = _convert_const_only(body)
		assert isinstance(result[0], IRCopy)
		# 70000 & 0xFFFF = 4464, which is < 32768 so stays positive
		assert result[0].source.value == 4464

	def test_int_to_bool(self):
		"""convert(bool, IRConst(42)) -> IRConst(1)"""
		body = [
			IRConvert(dest=IRTemp("t0"), source=IRConst(42), from_type=IRType.INT, to_type=IRType.BOOL),
		]
		result = _convert_const_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source.value == 1

	def test_int_to_bool_zero(self):
		"""convert(bool, IRConst(0)) -> IRConst(0)"""
		body = [
			IRConvert(dest=IRTemp("t0"), source=IRConst(0), from_type=IRType.INT, to_type=IRType.BOOL),
		]
		result = _convert_const_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source.value == 0

	def test_int_to_float(self):
		"""convert(float, IRConst(5)) -> IRFloatConst(5.0)"""
		body = [
			IRConvert(dest=IRTemp("t0"), source=IRConst(5), from_type=IRType.INT, to_type=IRType.FLOAT),
		]
		result = _convert_const_only(body)
		assert isinstance(result[0], IRCopy)
		assert isinstance(result[0].source, IRFloatConst)
		assert result[0].source.value == 5.0

	def test_int_to_double(self):
		"""convert(double, IRConst(7)) -> IRFloatConst(7.0, DOUBLE)"""
		body = [
			IRConvert(dest=IRTemp("t0"), source=IRConst(7), from_type=IRType.INT, to_type=IRType.DOUBLE),
		]
		result = _convert_const_only(body)
		assert isinstance(result[0], IRCopy)
		assert isinstance(result[0].source, IRFloatConst)
		assert result[0].source.value == 7.0
		assert result[0].source.ir_type == IRType.DOUBLE

	def test_float_to_int(self):
		"""convert(int, IRFloatConst(3.7)) -> IRConst(3)"""
		body = [
			IRConvert(
				dest=IRTemp("t0"), source=IRFloatConst(3.7),
				from_type=IRType.FLOAT, to_type=IRType.INT,
			),
		]
		result = _convert_const_only(body)
		assert isinstance(result[0], IRCopy)
		assert isinstance(result[0].source, IRConst)
		assert result[0].source.value == 3

	def test_float_to_char(self):
		"""convert(char, IRFloatConst(65.9)) -> IRConst(65)"""
		body = [
			IRConvert(
				dest=IRTemp("t0"), source=IRFloatConst(65.9),
				from_type=IRType.FLOAT, to_type=IRType.CHAR,
			),
		]
		result = _convert_const_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source.value == 65

	def test_double_to_float(self):
		"""convert(float, IRFloatConst(3.14, DOUBLE)) -> IRFloatConst(3.14, FLOAT)"""
		body = [
			IRConvert(
				dest=IRTemp("t0"), source=IRFloatConst(3.14, ir_type=IRType.DOUBLE),
				from_type=IRType.DOUBLE, to_type=IRType.FLOAT,
			),
		]
		result = _convert_const_only(body)
		assert isinstance(result[0], IRCopy)
		assert isinstance(result[0].source, IRFloatConst)
		assert result[0].source.ir_type == IRType.FLOAT

	def test_non_const_source_unchanged(self):
		"""convert with non-constant source should not be folded."""
		body = [
			IRConvert(dest=IRTemp("t0"), source=IRTemp("t1"), from_type=IRType.INT, to_type=IRType.CHAR),
		]
		result = _convert_const_only(body)
		assert isinstance(result[0], IRConvert)

	def test_negative_int_to_char(self):
		"""convert(char, IRConst(-1)) -> IRConst(-1) (0xFF sign-extended)"""
		body = [
			IRConvert(dest=IRTemp("t0"), source=IRConst(-1), from_type=IRType.INT, to_type=IRType.CHAR),
		]
		result = _convert_const_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source.value == -1


# ── Boolean Simplification ──


class TestBooleanSimplification:
	def test_double_negation(self):
		"""!!x -> x"""
		body = [
			IRUnaryOp(dest=IRTemp("t1"), op="!", operand=IRTemp("t0")),
			IRUnaryOp(dest=IRTemp("t2"), op="!", operand=IRTemp("t1")),
		]
		result = _bool_only(body)
		assert isinstance(result[1], IRCopy)
		assert result[1].source == IRTemp("t0")

	def test_compare_eq_zero(self):
		"""x == 0 -> !x"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="==", right=IRConst(0)),
		]
		result = _bool_only(body)
		assert isinstance(result[0], IRUnaryOp)
		assert result[0].op == "!"
		assert result[0].operand == IRTemp("t0")

	def test_zero_eq_x(self):
		"""0 == x -> !x"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(0), op="==", right=IRTemp("t0")),
		]
		result = _bool_only(body)
		assert isinstance(result[0], IRUnaryOp)
		assert result[0].op == "!"
		assert result[0].operand == IRTemp("t0")

	def test_and_with_zero(self):
		"""x && 0 -> 0"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="&&", right=IRConst(0)),
		]
		result = _bool_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(0)

	def test_and_with_one(self):
		"""x && 1 -> x"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="&&", right=IRConst(1)),
		]
		result = _bool_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRTemp("t0")

	def test_or_with_zero(self):
		"""x || 0 -> x"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="||", right=IRConst(0)),
		]
		result = _bool_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRTemp("t0")

	def test_or_with_one(self):
		"""x || 1 -> 1"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="||", right=IRConst(1)),
		]
		result = _bool_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(1)

	def test_zero_and_x(self):
		"""0 && x -> 0"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(0), op="&&", right=IRTemp("t0")),
		]
		result = _bool_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(0)

	def test_one_or_x(self):
		"""1 || x -> 1"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRConst(1), op="||", right=IRTemp("t0")),
		]
		result = _bool_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(1)

	def test_label_clears_negation_tracking(self):
		"""Negation map should be cleared at labels."""
		body = [
			IRUnaryOp(dest=IRTemp("t1"), op="!", operand=IRTemp("t0")),
			IRLabelInstr(name="L1"),
			IRUnaryOp(dest=IRTemp("t2"), op="!", operand=IRTemp("t1")),
		]
		result = _bool_only(body)
		# t2 = !t1 should NOT be simplified since label cleared the map
		assert isinstance(result[2], IRUnaryOp)

	def test_non_negation_unary_not_tracked(self):
		"""Only ! is tracked for double negation, not ~ or -."""
		body = [
			IRUnaryOp(dest=IRTemp("t1"), op="~", operand=IRTemp("t0")),
			IRUnaryOp(dest=IRTemp("t2"), op="!", operand=IRTemp("t1")),
		]
		result = _bool_only(body)
		assert isinstance(result[1], IRUnaryOp)
		assert result[1].op == "!"


# ── Redundant IRCopy Chain Folding ──


class TestCopyChainFolding:
	def test_copy_chain_to_constant(self):
		"""t1 = copy 5; t2 = copy t1; t3 = copy t2 -> all resolve to 5."""
		body = [
			IRCopy(dest=IRTemp("t1"), source=IRConst(5)),
			IRCopy(dest=IRTemp("t2"), source=IRTemp("t1")),
			IRCopy(dest=IRTemp("t3"), source=IRTemp("t2")),
			IRReturn(value=IRTemp("t3")),
		]
		result = _opt(body)
		# After full optimization, the return should use the constant directly
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert ret[0].value == IRConst(5)

	def test_copy_chain_to_temp(self):
		"""t2 = copy t1; t3 = copy t2 -> t3 uses t1 directly."""
		body = [
			IRCopy(dest=IRTemp("t2"), source=IRTemp("t1")),
			IRCopy(dest=IRTemp("t3"), source=IRTemp("t2")),
			IRReturn(value=IRTemp("t3")),
		]
		result = _opt(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert ret[0].value == IRTemp("t1")

	def test_convert_then_copy_chain(self):
		"""convert produces copy, then chained copies resolve correctly."""
		body = [
			IRConvert(dest=IRTemp("t1"), source=IRConst(5), from_type=IRType.INT, to_type=IRType.INT),
			IRCopy(dest=IRTemp("t2"), source=IRTemp("t1")),
			IRReturn(value=IRTemp("t2")),
		]
		result = _opt(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert isinstance(ret[0].value, IRConst)
		assert ret[0].value.value == 5


# ── Integration: Multiple Passes Working Together ──


class TestConstPropIntegration:
	def test_convert_fold_then_dce(self):
		"""After folding a convert of a constant, dead code should be eliminated."""
		body = [
			IRConvert(dest=IRTemp("t0"), source=IRConst(10), from_type=IRType.INT, to_type=IRType.CHAR),
			IRReturn(value=IRTemp("t0")),
		]
		result = _opt(body)
		# Should be simplified: return 10 (after char truncation = 10)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert isinstance(ret[0].value, IRConst)
		assert ret[0].value.value == 10

	def test_double_negation_then_dce(self):
		"""!!x collapses to x, and the intermediate negation is DCE'd."""
		body = [
			IRUnaryOp(dest=IRTemp("t1"), op="!", operand=IRTemp("t0")),
			IRUnaryOp(dest=IRTemp("t2"), op="!", operand=IRTemp("t1")),
			IRReturn(value=IRTemp("t2")),
		]
		result = _opt(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		# t2 should resolve to t0, and t1 should be DCE'd
		assert ret[0].value == IRTemp("t0")

	def test_compare_zero_then_double_neg(self):
		"""(x == 0) == 0 -> !!x -> x"""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="==", right=IRConst(0)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t1"), op="==", right=IRConst(0)),
			IRReturn(value=IRTemp("t2")),
		]
		result = _opt(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert ret[0].value == IRTemp("t0")

	def test_logical_and_zero_folded(self):
		"""x && 0 is folded to 0 and propagated through."""
		body = [
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="&&", right=IRConst(0)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _opt(body)
		ret = [i for i in result if isinstance(i, IRReturn)]
		assert len(ret) == 1
		assert ret[0].value == IRConst(0)
