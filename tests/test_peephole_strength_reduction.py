"""Tests for peephole strength reduction: multiply/divide by powers of 2, push/pop, self-moves."""

from compiler.peephole import PeepholeOptimizer


def _opt(asm: str) -> str:
	"""Helper: optimize assembly text."""
	return PeepholeOptimizer().optimize(asm)


# ---------------------------------------------------------------------------
# Multiply strength reduction (imulq -> shlq)
# ---------------------------------------------------------------------------


class TestMultiplyStrengthReduction:
	def test_imulq_2_to_shlq(self) -> None:
		assert _opt("\timulq $2, %rax") == "\tshlq $1, %rax"

	def test_imulq_4_to_shlq(self) -> None:
		assert _opt("\timulq $4, %rbx") == "\tshlq $2, %rbx"

	def test_imulq_8_to_shlq(self) -> None:
		assert _opt("\timulq $8, %rcx") == "\tshlq $3, %rcx"

	def test_imulq_16(self) -> None:
		assert _opt("\timulq $16, %rdx") == "\tshlq $4, %rdx"

	def test_imulq_1024(self) -> None:
		assert _opt("\timulq $1024, %rax") == "\tshlq $10, %rax"

	def test_imulq_non_power_of_2_preserved(self) -> None:
		asm = "\timulq $3, %rax"
		assert _opt(asm) == asm

	def test_imulq_5_preserved(self) -> None:
		asm = "\timulq $5, %rax"
		assert _opt(asm) == asm

	def test_imulq_7_preserved(self) -> None:
		asm = "\timulq $7, %rax"
		assert _opt(asm) == asm

	def test_imulq_1_eliminated(self) -> None:
		lines = ["\tmovq $42, %rax", "\timulq $1, %rax", "\tret"]
		result = _opt("\n".join(lines))
		expected = "\n".join(["\tmovq $42, %rax", "\tret"])
		assert result == expected

	def test_imulq_0_to_xorl(self) -> None:
		assert _opt("\timulq $0, %rax") == "\txorl %eax, %eax"

	def test_imulq_in_context(self) -> None:
		lines = [
			"\tmovq -8(%rbp), %rax",
			"\timulq $8, %rax",
			"\taddq %rbx, %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tmovq -8(%rbp), %rax",
			"\tshlq $3, %rax",
			"\taddq %rbx, %rax",
			"\tret",
		])
		assert result == expected


# ---------------------------------------------------------------------------
# Signed division strength reduction (idivq -> sarq)
# ---------------------------------------------------------------------------


class TestSignedDivStrengthReduction:
	def test_div_by_2(self) -> None:
		asm = "\n".join(["\tmovq $2, %rcx", "\tcqto", "\tidivq %rcx"])
		assert _opt(asm) == "\tsarq $1, %rax"

	def test_div_by_4(self) -> None:
		asm = "\n".join(["\tmovq $4, %rcx", "\tcqto", "\tidivq %rcx"])
		assert _opt(asm) == "\tsarq $2, %rax"

	def test_div_by_8(self) -> None:
		asm = "\n".join(["\tmovq $8, %rcx", "\tcqto", "\tidivq %rcx"])
		assert _opt(asm) == "\tsarq $3, %rax"

	def test_div_by_1024(self) -> None:
		asm = "\n".join(["\tmovq $1024, %rcx", "\tcqto", "\tidivq %rcx"])
		assert _opt(asm) == "\tsarq $10, %rax"

	def test_div_by_1_eliminated(self) -> None:
		lines = [
			"\tmovq -8(%rbp), %rax",
			"\tmovq $1, %rcx",
			"\tcqto",
			"\tidivq %rcx",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join(["\tmovq -8(%rbp), %rax", "\tret"])
		assert result == expected

	def test_non_power_of_2_preserved(self) -> None:
		asm = "\n".join(["\tmovq $3, %rcx", "\tcqto", "\tidivq %rcx"])
		assert _opt(asm) == asm

	def test_div_by_6_preserved(self) -> None:
		asm = "\n".join(["\tmovq $6, %rcx", "\tcqto", "\tidivq %rcx"])
		assert _opt(asm) == asm

	def test_mismatched_register_preserved(self) -> None:
		"""movq to %rcx but idivq %rbx should not match."""
		asm = "\n".join(["\tmovq $4, %rcx", "\tcqto", "\tidivq %rbx"])
		assert _opt(asm) == asm

	def test_in_context(self) -> None:
		lines = [
			"\tmovq -8(%rbp), %rax",
			"\tmovq $16, %rcx",
			"\tcqto",
			"\tidivq %rcx",
			"\tmovq %rax, -16(%rbp)",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tmovq -8(%rbp), %rax",
			"\tsarq $4, %rax",
			"\tmovq %rax, -16(%rbp)",
			"\tret",
		])
		assert result == expected


# ---------------------------------------------------------------------------
# Unsigned division strength reduction (divq -> shrq)
# ---------------------------------------------------------------------------


class TestUnsignedDivStrengthReduction:
	def test_udiv_by_2(self) -> None:
		asm = "\n".join(["\tmovq $2, %rcx", "\txorq %rdx, %rdx", "\tdivq %rcx"])
		assert _opt(asm) == "\tshrq $1, %rax"

	def test_udiv_by_8(self) -> None:
		asm = "\n".join(["\tmovq $8, %rcx", "\txorq %rdx, %rdx", "\tdivq %rcx"])
		assert _opt(asm) == "\tshrq $3, %rax"

	def test_udiv_by_256(self) -> None:
		asm = "\n".join(["\tmovq $256, %rcx", "\txorq %rdx, %rdx", "\tdivq %rcx"])
		assert _opt(asm) == "\tshrq $8, %rax"

	def test_udiv_by_1_eliminated(self) -> None:
		lines = [
			"\tmovq -8(%rbp), %rax",
			"\tmovq $1, %rcx",
			"\txorq %rdx, %rdx",
			"\tdivq %rcx",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join(["\tmovq -8(%rbp), %rax", "\tret"])
		assert result == expected

	def test_non_power_of_2_preserved(self) -> None:
		asm = "\n".join(["\tmovq $5, %rcx", "\txorq %rdx, %rdx", "\tdivq %rcx"])
		assert _opt(asm) == asm

	def test_mismatched_register_preserved(self) -> None:
		asm = "\n".join(["\tmovq $4, %rcx", "\txorq %rdx, %rdx", "\tdivq %rbx"])
		assert _opt(asm) == asm

	def test_in_context(self) -> None:
		lines = [
			"\tmovq -8(%rbp), %rax",
			"\tmovq $32, %rcx",
			"\txorq %rdx, %rdx",
			"\tdivq %rcx",
			"\tmovq %rax, -16(%rbp)",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tmovq -8(%rbp), %rax",
			"\tshrq $5, %rax",
			"\tmovq %rax, -16(%rbp)",
			"\tret",
		])
		assert result == expected


# ---------------------------------------------------------------------------
# Redundant push/pop elimination
# ---------------------------------------------------------------------------


class TestPushPopElimination:
	def test_same_register_eliminated(self) -> None:
		lines = ["\tpushq %rbp", "\tpushq %rax", "\tpopq %rax", "\tret"]
		result = _opt("\n".join(lines))
		expected = "\n".join(["\tpushq %rbp", "\tret"])
		assert result == expected

	def test_different_register_becomes_movq(self) -> None:
		asm = "\n".join(["\tpushq %rax", "\tpopq %rbx"])
		assert _opt(asm) == "\tmovq %rax, %rbx"

	def test_pushq_popq_rax_rax(self) -> None:
		asm = "\n".join(["\tpushq %rax", "\tpopq %rax"])
		assert _opt(asm) == ""

	def test_push_pop_in_context(self) -> None:
		lines = [
			"\tmovq $1, %rax",
			"\tpushq %rcx",
			"\tpopq %rcx",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tmovq $1, %rax",
			"\tret",
		])
		assert result == expected

	def test_push_pop_different_in_context(self) -> None:
		lines = [
			"\tmovq $42, %rax",
			"\tpushq %rax",
			"\tpopq %rbx",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tmovq $42, %rax",
			"\tmovq %rax, %rbx",
			"\tret",
		])
		assert result == expected


# ---------------------------------------------------------------------------
# Self-move elimination (movq %reg, %reg)
# ---------------------------------------------------------------------------


class TestSelfMoveElimination:
	def test_movq_self_move(self) -> None:
		assert _opt("\tmovq %rax, %rax") == ""

	def test_movq_self_move_rbx(self) -> None:
		assert _opt("\tmovq %rbx, %rbx") == ""

	def test_movl_self_move(self) -> None:
		assert _opt("\tmovl %eax, %eax") == ""

	def test_movw_self_move(self) -> None:
		assert _opt("\tmovw %ax, %ax") == ""

	def test_movb_self_move(self) -> None:
		assert _opt("\tmovb %al, %al") == ""

	def test_non_self_move_preserved(self) -> None:
		asm = "\tmovq %rax, %rbx"
		assert _opt(asm) == asm

	def test_self_move_in_context(self) -> None:
		lines = [
			"\taddq %rbx, %rax",
			"\tmovq %rax, %rax",
			"\tmovq %rcx, %rcx",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join(["\taddq %rbx, %rax", "\tret"])
		assert result == expected

	def test_multiple_self_moves(self) -> None:
		lines = [
			"\tmovq %rax, %rax",
			"\tmovq %rbx, %rbx",
			"\tmovq %rcx, %rcx",
		]
		assert _opt("\n".join(lines)) == ""


# ---------------------------------------------------------------------------
# Combined / interaction patterns
# ---------------------------------------------------------------------------


class TestCombinedPatterns:
	def test_mul_and_div_strength_reduction(self) -> None:
		"""Both multiply and divide strength reduction in same function."""
		lines = [
			"\timulq $8, %rax",
			"\tmovq %rax, -8(%rbp)",
			"\tmovq -16(%rbp), %rax",
			"\tmovq $4, %rcx",
			"\tcqto",
			"\tidivq %rcx",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tshlq $3, %rax",
			"\tmovq %rax, -8(%rbp)",
			"\tmovq -16(%rbp), %rax",
			"\tsarq $2, %rax",
			"\tret",
		])
		assert result == expected

	def test_push_pop_self_move_chain(self) -> None:
		"""Push/pop elimination followed by self-move elimination."""
		lines = [
			"\tpushq %rax",
			"\tpopq %rax",
			"\tmovq %rbx, %rbx",
			"\tret",
		]
		result = _opt("\n".join(lines))
		assert result == "\tret"

	def test_all_patterns_in_function(self) -> None:
		"""Realistic function with all pattern types."""
		lines = [
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tmovq -8(%rbp), %rax",
			"\timulq $16, %rax",
			"\tmovq %rax, %rax",
			"\tpushq %rcx",
			"\tpopq %rcx",
			"\tmovq -16(%rbp), %rax",
			"\tmovq $8, %rcx",
			"\txorq %rdx, %rdx",
			"\tdivq %rcx",
			"\tpopq %rbp",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tmovq -8(%rbp), %rax",
			"\tshlq $4, %rax",
			"\tmovq -16(%rbp), %rax",
			"\tshrq $3, %rax",
			"\tpopq %rbp",
			"\tret",
		])
		assert result == expected
