"""Tests for narrow (CHAR/SHORT) arithmetic truncation in codegen."""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRBinOp,
	IRConst,
	IRConvert,
	IRFunction,
	IRProgram,
	IRReturn,
	IRTemp,
	IRType,
	IRUnaryOp,
)


def _gen(func: IRFunction) -> str:
	"""Helper: generate assembly for a single-function program."""
	return CodeGenerator().generate(IRProgram([func], [], []))


# ---------------------------------------------------------------------------
# CHAR arithmetic truncation
# ---------------------------------------------------------------------------


class TestCharArithmeticTruncation:
	def test_char_add_emits_sign_extend(self) -> None:
		"""CHAR addition should emit movsbq truncation."""
		body = [
			IRBinOp(IRTemp("t2"), IRTemp("t0"), "+", IRTemp("t1"), ir_type=IRType.CHAR),
			IRReturn(IRTemp("t2")),
		]
		func = IRFunction("f", [IRTemp("t0"), IRTemp("t1")], body, IRType.INT, [IRType.CHAR, IRType.CHAR])
		asm = _gen(func)
		assert "addq" in asm
		assert "movsbq %al, %rax" in asm

	def test_unsigned_char_add_emits_zero_extend(self) -> None:
		"""Unsigned CHAR addition should emit movzbq truncation."""
		body = [
			IRBinOp(IRTemp("t2"), IRTemp("t0"), "+", IRTemp("t1"), ir_type=IRType.CHAR, is_unsigned=True),
			IRReturn(IRTemp("t2")),
		]
		func = IRFunction("f", [IRTemp("t0"), IRTemp("t1")], body, IRType.INT, [IRType.CHAR, IRType.CHAR])
		asm = _gen(func)
		assert "addq" in asm
		assert "movzbq %al, %rax" in asm

	def test_char_sub_emits_truncation(self) -> None:
		body = [
			IRBinOp(IRTemp("t2"), IRTemp("t0"), "-", IRTemp("t1"), ir_type=IRType.CHAR),
			IRReturn(IRTemp("t2")),
		]
		func = IRFunction("f", [IRTemp("t0"), IRTemp("t1")], body, IRType.INT, [IRType.CHAR, IRType.CHAR])
		asm = _gen(func)
		assert "subq" in asm
		assert "movsbq %al, %rax" in asm

	def test_char_mul_emits_truncation(self) -> None:
		body = [
			IRBinOp(IRTemp("t2"), IRTemp("t0"), "*", IRTemp("t1"), ir_type=IRType.CHAR),
			IRReturn(IRTemp("t2")),
		]
		func = IRFunction("f", [IRTemp("t0"), IRTemp("t1")], body, IRType.INT, [IRType.CHAR, IRType.CHAR])
		asm = _gen(func)
		assert "imulq" in asm
		assert "movsbq %al, %rax" in asm

	def test_char_bitwise_and_emits_truncation(self) -> None:
		body = [
			IRBinOp(IRTemp("t2"), IRTemp("t0"), "&", IRTemp("t1"), ir_type=IRType.CHAR),
			IRReturn(IRTemp("t2")),
		]
		func = IRFunction("f", [IRTemp("t0"), IRTemp("t1")], body, IRType.INT, [IRType.CHAR, IRType.CHAR])
		asm = _gen(func)
		assert "andq" in asm
		assert "movsbq %al, %rax" in asm

	def test_char_comparison_no_truncation(self) -> None:
		"""Comparison ops produce 0/1 and should NOT emit narrow truncation."""
		body = [
			IRBinOp(IRTemp("t2"), IRTemp("t0"), "<", IRTemp("t1"), ir_type=IRType.CHAR),
			IRReturn(IRTemp("t2")),
		]
		func = IRFunction("f", [IRTemp("t0"), IRTemp("t1")], body, IRType.INT, [IRType.CHAR, IRType.CHAR])
		asm = _gen(func)
		# Should have movzbq from setcc, but NOT movsbq from truncation
		assert "movzbq %al, %rax" in asm
		assert "movsbq %al, %rax" not in asm


# ---------------------------------------------------------------------------
# SHORT arithmetic truncation
# ---------------------------------------------------------------------------


class TestShortArithmeticTruncation:
	def test_short_add_emits_sign_extend(self) -> None:
		body = [
			IRBinOp(IRTemp("t2"), IRTemp("t0"), "+", IRTemp("t1"), ir_type=IRType.SHORT),
			IRReturn(IRTemp("t2")),
		]
		func = IRFunction("f", [IRTemp("t0"), IRTemp("t1")], body, IRType.INT, [IRType.SHORT, IRType.SHORT])
		asm = _gen(func)
		assert "addq" in asm
		assert "movswq %ax, %rax" in asm

	def test_unsigned_short_add_emits_zero_extend(self) -> None:
		body = [
			IRBinOp(IRTemp("t2"), IRTemp("t0"), "+", IRTemp("t1"), ir_type=IRType.SHORT, is_unsigned=True),
			IRReturn(IRTemp("t2")),
		]
		func = IRFunction("f", [IRTemp("t0"), IRTemp("t1")], body, IRType.INT, [IRType.SHORT, IRType.SHORT])
		asm = _gen(func)
		assert "addq" in asm
		assert "movzwq %ax, %rax" in asm

	def test_short_sub_emits_truncation(self) -> None:
		body = [
			IRBinOp(IRTemp("t2"), IRTemp("t0"), "-", IRTemp("t1"), ir_type=IRType.SHORT),
			IRReturn(IRTemp("t2")),
		]
		func = IRFunction("f", [IRTemp("t0"), IRTemp("t1")], body, IRType.INT, [IRType.SHORT, IRType.SHORT])
		asm = _gen(func)
		assert "subq" in asm
		assert "movswq %ax, %rax" in asm

	def test_short_mul_emits_truncation(self) -> None:
		body = [
			IRBinOp(IRTemp("t2"), IRTemp("t0"), "*", IRTemp("t1"), ir_type=IRType.SHORT),
			IRReturn(IRTemp("t2")),
		]
		func = IRFunction("f", [IRTemp("t0"), IRTemp("t1")], body, IRType.INT, [IRType.SHORT, IRType.SHORT])
		asm = _gen(func)
		assert "imulq" in asm
		assert "movswq %ax, %rax" in asm

	def test_short_shift_left_emits_truncation(self) -> None:
		body = [
			IRBinOp(IRTemp("t2"), IRTemp("t0"), "<<", IRConst(1), ir_type=IRType.SHORT),
			IRReturn(IRTemp("t2")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.SHORT])
		asm = _gen(func)
		assert "salq" in asm
		assert "movswq %ax, %rax" in asm

	def test_short_comparison_no_truncation(self) -> None:
		body = [
			IRBinOp(IRTemp("t2"), IRTemp("t0"), "==", IRTemp("t1"), ir_type=IRType.SHORT),
			IRReturn(IRTemp("t2")),
		]
		func = IRFunction("f", [IRTemp("t0"), IRTemp("t1")], body, IRType.INT, [IRType.SHORT, IRType.SHORT])
		asm = _gen(func)
		assert "movzbq %al, %rax" in asm
		assert "movswq %ax, %rax" not in asm


# ---------------------------------------------------------------------------
# INT arithmetic should NOT emit truncation
# ---------------------------------------------------------------------------


class TestIntNoTruncation:
	def test_int_add_no_truncation(self) -> None:
		body = [
			IRBinOp(IRTemp("t2"), IRTemp("t0"), "+", IRTemp("t1"), ir_type=IRType.INT),
			IRReturn(IRTemp("t2")),
		]
		func = IRFunction("f", [IRTemp("t0"), IRTemp("t1")], body, IRType.INT, [IRType.INT, IRType.INT])
		asm = _gen(func)
		assert "addq" in asm
		assert "movsbq" not in asm
		assert "movswq" not in asm
		assert "movzbq" not in asm
		assert "movzwq" not in asm


# ---------------------------------------------------------------------------
# Unary ops on narrow types
# ---------------------------------------------------------------------------


class TestUnaryNarrowTruncation:
	def test_char_negate_emits_truncation(self) -> None:
		body = [
			IRUnaryOp(IRTemp("t1"), "-", IRTemp("t0"), ir_type=IRType.CHAR),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.CHAR])
		asm = _gen(func)
		assert "negq" in asm
		assert "movsbq %al, %rax" in asm

	def test_short_bitwise_not_emits_truncation(self) -> None:
		body = [
			IRUnaryOp(IRTemp("t1"), "~", IRTemp("t0"), ir_type=IRType.SHORT),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.SHORT])
		asm = _gen(func)
		assert "notq" in asm
		assert "movswq %ax, %rax" in asm

	def test_char_logical_not_no_truncation(self) -> None:
		"""Logical NOT produces 0/1, no narrow truncation needed."""
		body = [
			IRUnaryOp(IRTemp("t1"), "!", IRTemp("t0"), ir_type=IRType.CHAR),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.CHAR])
		asm = _gen(func)
		assert "sete" in asm
		assert "movsbq" not in asm


# ---------------------------------------------------------------------------
# Chained narrow arithmetic
# ---------------------------------------------------------------------------


class TestChainedNarrowArithmetic:
	def test_chained_char_add_both_truncated(self) -> None:
		"""Two consecutive CHAR additions should each emit truncation."""
		body = [
			IRBinOp(IRTemp("t2"), IRTemp("t0"), "+", IRTemp("t1"), ir_type=IRType.CHAR),
			IRBinOp(IRTemp("t3"), IRTemp("t2"), "+", IRConst(1), ir_type=IRType.CHAR),
			IRReturn(IRTemp("t3")),
		]
		func = IRFunction("f", [IRTemp("t0"), IRTemp("t1")], body, IRType.INT, [IRType.CHAR, IRType.CHAR])
		asm = _gen(func)
		# Count movsbq occurrences - should be 2 (one per addition)
		assert asm.count("movsbq %al, %rax") == 2

	def test_chained_short_ops_both_truncated(self) -> None:
		"""SHORT add then multiply should each emit truncation."""
		body = [
			IRBinOp(IRTemp("t2"), IRTemp("t0"), "+", IRTemp("t1"), ir_type=IRType.SHORT),
			IRBinOp(IRTemp("t3"), IRTemp("t2"), "*", IRConst(2), ir_type=IRType.SHORT),
			IRReturn(IRTemp("t3")),
		]
		func = IRFunction("f", [IRTemp("t0"), IRTemp("t1")], body, IRType.INT, [IRType.SHORT, IRType.SHORT])
		asm = _gen(func)
		assert asm.count("movswq %ax, %rax") == 2


# ---------------------------------------------------------------------------
# Convert with narrowing
# ---------------------------------------------------------------------------


class TestConvertNarrowing:
	def test_int_to_char_convert_truncates(self) -> None:
		body = [
			IRConvert(IRTemp("t1"), IRTemp("t0"), from_type=IRType.INT, to_type=IRType.CHAR),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.INT])
		asm = _gen(func)
		assert "movsbq %al, %rax" in asm

	def test_int_to_short_convert_truncates(self) -> None:
		body = [
			IRConvert(IRTemp("t1"), IRTemp("t0"), from_type=IRType.INT, to_type=IRType.SHORT),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.INT])
		asm = _gen(func)
		assert "movswq %ax, %rax" in asm

	def test_char_to_int_convert_sign_extends(self) -> None:
		body = [
			IRConvert(IRTemp("t1"), IRTemp("t0"), from_type=IRType.CHAR, to_type=IRType.INT),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.CHAR])
		asm = _gen(func)
		assert "movsbq %al, %rax" in asm


# ---------------------------------------------------------------------------
# Overflow / wraparound scenarios (assembly verification)
# ---------------------------------------------------------------------------


class TestOverflowWraparound:
	def test_char_overflow_add_const(self) -> None:
		"""Adding constants that overflow char range should still emit truncation."""
		body = [
			IRBinOp(IRTemp("t1"), IRTemp("t0"), "+", IRConst(200), ir_type=IRType.CHAR),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.CHAR])
		asm = _gen(func)
		assert "movsbq %al, %rax" in asm

	def test_short_overflow_mul_const(self) -> None:
		"""Multiplying by a value that could overflow short should emit truncation."""
		body = [
			IRBinOp(IRTemp("t1"), IRTemp("t0"), "*", IRConst(1000), ir_type=IRType.SHORT),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.SHORT])
		asm = _gen(func)
		assert "movswq %ax, %rax" in asm

	def test_unsigned_char_wrap_emits_movzbq(self) -> None:
		"""Unsigned char arithmetic wrapping should use zero-extend."""
		body = [
			IRBinOp(IRTemp("t1"), IRTemp("t0"), "+", IRConst(255), ir_type=IRType.CHAR, is_unsigned=True),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.CHAR])
		asm = _gen(func)
		assert "movzbq %al, %rax" in asm
		assert "movsbq" not in asm


# ---------------------------------------------------------------------------
# Division and modulo on narrow types
# ---------------------------------------------------------------------------


class TestNarrowDivMod:
	def test_char_div_emits_truncation(self) -> None:
		body = [
			IRBinOp(IRTemp("t2"), IRTemp("t0"), "/", IRTemp("t1"), ir_type=IRType.CHAR),
			IRReturn(IRTemp("t2")),
		]
		func = IRFunction("f", [IRTemp("t0"), IRTemp("t1")], body, IRType.INT, [IRType.CHAR, IRType.CHAR])
		asm = _gen(func)
		assert "idivq" in asm
		assert "movsbq %al, %rax" in asm

	def test_short_mod_emits_truncation(self) -> None:
		body = [
			IRBinOp(IRTemp("t2"), IRTemp("t0"), "%", IRTemp("t1"), ir_type=IRType.SHORT),
			IRReturn(IRTemp("t2")),
		]
		func = IRFunction("f", [IRTemp("t0"), IRTemp("t1")], body, IRType.INT, [IRType.SHORT, IRType.SHORT])
		asm = _gen(func)
		assert "idivq" in asm
		assert "movswq %ax, %rax" in asm

	def test_unsigned_short_div_emits_truncation(self) -> None:
		body = [
			IRBinOp(IRTemp("t2"), IRTemp("t0"), "/", IRTemp("t1"), ir_type=IRType.SHORT, is_unsigned=True),
			IRReturn(IRTemp("t2")),
		]
		func = IRFunction("f", [IRTemp("t0"), IRTemp("t1")], body, IRType.INT, [IRType.SHORT, IRType.SHORT])
		asm = _gen(func)
		assert "divq" in asm
		assert "movzwq %ax, %rax" in asm
