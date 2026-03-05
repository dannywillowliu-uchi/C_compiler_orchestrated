"""Tests for enhanced peephole optimizer: redundant moves, strength reduction, add/sub combining."""

from compiler.peephole import PeepholeOptimizer


def _opt(asm: str) -> str:
	"""Helper: optimize assembly text."""
	return PeepholeOptimizer().optimize(asm)


# ---------------------------------------------------------------------------
# Pattern 1: Redundant reverse move elimination
# ---------------------------------------------------------------------------


class TestReverseMoveElimination:
	def test_basic_reverse_move(self) -> None:
		asm = "\tmovq %rax, %rbx\n\tmovq %rbx, %rax"
		assert _opt(asm) == "\tmovq %rax, %rbx"

	def test_reverse_move_different_regs(self) -> None:
		asm = "\tmovq %rcx, %rdx\n\tmovq %rdx, %rcx"
		assert _opt(asm) == "\tmovq %rcx, %rdx"

	def test_non_reverse_move_preserved(self) -> None:
		"""movq %rA, %rB + movq %rC, %rA is NOT a reverse move."""
		asm = "\tmovq %rax, %rbx\n\tmovq %rcx, %rax"
		assert _opt(asm) == asm

	def test_reverse_move_with_context(self) -> None:
		lines = [
			"\tpushq %rbp",
			"\tmovq %rax, %rbx",
			"\tmovq %rbx, %rax",
			"\taddq %rcx, %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tpushq %rbp",
			"\tmovq %rax, %rbx",
			"\taddq %rcx, %rax",
			"\tret",
		])
		assert result == expected

	def test_multiple_reverse_moves(self) -> None:
		lines = [
			"\tmovq %rax, %rbx",
			"\tmovq %rbx, %rax",
			"\tmovq %rcx, %rdx",
			"\tmovq %rdx, %rcx",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tmovq %rax, %rbx",
			"\tmovq %rcx, %rdx",
		])
		assert result == expected

	def test_same_direction_deduplicated(self) -> None:
		"""Two identical moves are deduplicated to one."""
		asm = "\tmovq %rax, %rbx\n\tmovq %rax, %rbx"
		assert _opt(asm) == "\tmovq %rax, %rbx"


# ---------------------------------------------------------------------------
# Pattern 2: Strength reduction (imulq -> shlq)
# ---------------------------------------------------------------------------


class TestStrengthReduction:
	def test_imulq_2_to_shlq_1(self) -> None:
		asm = "\timulq $2, %rax"
		assert _opt(asm) == "\tshlq $1, %rax"

	def test_imulq_4_to_shlq_2(self) -> None:
		asm = "\timulq $4, %rbx"
		assert _opt(asm) == "\tshlq $2, %rbx"

	def test_imulq_8_to_shlq_3(self) -> None:
		asm = "\timulq $8, %rcx"
		assert _opt(asm) == "\tshlq $3, %rcx"

	def test_imulq_16_to_shlq_4(self) -> None:
		asm = "\timulq $16, %rdx"
		assert _opt(asm) == "\tshlq $4, %rdx"

	def test_imulq_1024_to_shlq_10(self) -> None:
		asm = "\timulq $1024, %rax"
		assert _opt(asm) == "\tshlq $10, %rax"

	def test_imulq_non_power_of_2_preserved(self) -> None:
		asm = "\timulq $3, %rax"
		assert _opt(asm) == asm

	def test_imulq_5_preserved(self) -> None:
		asm = "\timulq $5, %rax"
		assert _opt(asm) == asm

	def test_imulq_1_eliminated(self) -> None:
		"""Multiplication by 1 is identity -- eliminate entirely."""
		lines = [
			"\tmovq $42, %rax",
			"\timulq $1, %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tmovq $42, %rax",
			"\tret",
		])
		assert result == expected

	def test_imulq_0_to_xorq(self) -> None:
		"""Multiplication by 0 should zero the register."""
		asm = "\timulq $0, %rax"
		assert _opt(asm) == "\txorq %rax, %rax"

	def test_strength_reduction_with_context(self) -> None:
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
# Pattern 3: Self-move elimination (already existed, verify it works)
# ---------------------------------------------------------------------------


class TestSelfMoveElimination:
	def test_self_move_rax(self) -> None:
		asm = "\tmovq %rax, %rax"
		assert _opt(asm) == ""

	def test_self_move_in_context(self) -> None:
		lines = [
			"\taddq %rbx, %rax",
			"\tmovq %rax, %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\taddq %rbx, %rax",
			"\tret",
		])
		assert result == expected


# ---------------------------------------------------------------------------
# Pattern 4: Combine adjacent addq/subq to same register
# ---------------------------------------------------------------------------


class TestCombineAddSub:
	def test_combine_two_addq(self) -> None:
		asm = "\taddq $5, %rax\n\taddq $3, %rax"
		assert _opt(asm) == "\taddq $8, %rax"

	def test_combine_two_subq(self) -> None:
		asm = "\tsubq $5, %rax\n\tsubq $3, %rax"
		assert _opt(asm) == "\tsubq $8, %rax"

	def test_combine_addq_subq_positive_result(self) -> None:
		"""addq $10 + subq $3 -> addq $7."""
		asm = "\taddq $10, %rax\n\tsubq $3, %rax"
		assert _opt(asm) == "\taddq $7, %rax"

	def test_combine_addq_subq_negative_result(self) -> None:
		"""addq $3 + subq $10 -> subq $7."""
		asm = "\taddq $3, %rax\n\tsubq $10, %rax"
		assert _opt(asm) == "\tsubq $7, %rax"

	def test_combine_addq_subq_cancel(self) -> None:
		"""addq $5 + subq $5 -> eliminated."""
		lines = [
			"\tpushq %rbp",
			"\taddq $5, %rax",
			"\tsubq $5, %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tpushq %rbp",
			"\tret",
		])
		assert result == expected

	def test_different_registers_not_combined(self) -> None:
		asm = "\taddq $5, %rax\n\taddq $3, %rbx"
		assert _opt(asm) == asm

	def test_combine_subq_addq(self) -> None:
		"""subq $3 + addq $10 -> addq $7."""
		asm = "\tsubq $3, %rax\n\taddq $10, %rax"
		assert _opt(asm) == "\taddq $7, %rax"

	def test_combine_with_context(self) -> None:
		lines = [
			"\tpushq %rbp",
			"\tsubq $32, %rsp",
			"\tsubq $16, %rsp",
			"\tmovq %rdi, -8(%rbp)",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tpushq %rbp",
			"\tsubq $48, %rsp",
			"\tmovq %rdi, -8(%rbp)",
			"\tret",
		])
		assert result == expected

	def test_combine_negative_immediates(self) -> None:
		"""addq $-5 is equivalent to subq $5."""
		asm = "\taddq $-5, %rax\n\taddq $-3, %rax"
		assert _opt(asm) == "\tsubq $8, %rax"


# ---------------------------------------------------------------------------
# Pattern 5: movq $0 -> xorq (already existed, verify it works)
# ---------------------------------------------------------------------------


class TestZeroToXor:
	def test_mov_zero_to_xor(self) -> None:
		asm = "\tmovq $0, %rax"
		assert _opt(asm) == "\txorq %rax, %rax"

	def test_mov_zero_to_xor_different_reg(self) -> None:
		asm = "\tmovq $0, %rcx"
		assert _opt(asm) == "\txorq %rcx, %rcx"

	def test_non_zero_mov_preserved(self) -> None:
		asm = "\tmovq $42, %rax"
		assert _opt(asm) == asm


# ---------------------------------------------------------------------------
# Combined patterns: multiple new optimizations interacting
# ---------------------------------------------------------------------------


class TestCombinedNewPatterns:
	def test_strength_reduction_and_combine(self) -> None:
		"""imulq + adjacent addq should both optimize."""
		lines = [
			"\timulq $4, %rax",
			"\taddq $8, %rax",
			"\taddq $2, %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tshlq $2, %rax",
			"\taddq $10, %rax",
			"\tret",
		])
		assert result == expected

	def test_reverse_move_and_self_move(self) -> None:
		"""Reverse move + self move in sequence."""
		lines = [
			"\tmovq %rax, %rbx",
			"\tmovq %rbx, %rax",
			"\tmovq %rcx, %rcx",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tmovq %rax, %rbx",
			"\tret",
		])
		assert result == expected

	def test_realistic_loop_optimization(self) -> None:
		"""A realistic loop body with multiple optimization opportunities."""
		lines = [
			".L_loop:",
			"\tmovq -8(%rbp), %rax",
			"\timulq $8, %rax",
			"\taddq $16, %rax",
			"\taddq $16, %rax",
			"\tmovq %rax, %rbx",
			"\tmovq %rbx, %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			".L_loop:",
			"\tmovq -8(%rbp), %rax",
			"\tshlq $3, %rax",
			"\taddq $32, %rax",
			"\tmovq %rax, %rbx",
			"\tret",
		])
		assert result == expected
