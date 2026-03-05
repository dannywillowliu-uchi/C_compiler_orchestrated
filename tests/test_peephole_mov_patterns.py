"""Tests for peephole mov-related optimization patterns."""

import pytest

from compiler.peephole import PeepholeOptimizer


@pytest.fixture
def opt() -> PeepholeOptimizer:
	return PeepholeOptimizer()


class TestDuplicateMovElimination:
	"""Pattern: movq %rA, %rB; movq %rA, %rB -> movq %rA, %rB"""

	def test_consecutive_identical_moves(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq %rax, %rbx\n\tmovq %rax, %rbx"
		result = opt.optimize(asm)
		assert result == "\tmovq %rax, %rbx"

	def test_different_moves_not_eliminated_different_dst(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq %rax, %rbx\n\tmovq %rcx, %rdx"
		result = opt.optimize(asm)
		assert "\tmovq %rax, %rbx" in result
		assert "\tmovq %rcx, %rdx" in result

	def test_triple_identical_moves(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq %rax, %rbx\n\tmovq %rax, %rbx\n\tmovq %rax, %rbx"
		result = opt.optimize(asm)
		assert result == "\tmovq %rax, %rbx"


class TestLeaZeroFold:
	"""Pattern: leaq 0(%rA), %rB -> movq %rA, %rB"""

	def test_lea_zero_offset_different_regs(self, opt: PeepholeOptimizer) -> None:
		asm = "\tleaq 0(%rax), %rbx"
		result = opt.optimize(asm)
		assert result == "\tmovq %rax, %rbx"

	def test_lea_zero_offset_same_reg_eliminated(self, opt: PeepholeOptimizer) -> None:
		"""leaq 0(%rax), %rax -> movq %rax, %rax -> eliminated (self-move)."""
		asm = "\tleaq 0(%rax), %rax"
		result = opt.optimize(asm)
		assert result == ""

	def test_lea_nonzero_offset_not_folded(self, opt: PeepholeOptimizer) -> None:
		asm = "\tleaq 8(%rax), %rbx"
		result = opt.optimize(asm)
		assert result == "\tleaq 8(%rax), %rbx"

	def test_lea_zero_with_rsp(self, opt: PeepholeOptimizer) -> None:
		asm = "\tleaq 0(%rsp), %rdi"
		result = opt.optimize(asm)
		assert result == "\tmovq %rsp, %rdi"


class TestDeadRegMove:
	"""Pattern: movq %rA, %rB where %rB is immediately overwritten."""

	def test_move_then_overwrite(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq %rax, %rbx\n\tmovq %rcx, %rbx"
		result = opt.optimize(asm)
		assert result == "\tmovq %rcx, %rbx"

	def test_move_then_lea_overwrite(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq %rax, %rbx\n\tleaq 8(%rsp), %rbx"
		result = opt.optimize(asm)
		assert result == "\tleaq 8(%rsp), %rbx"

	def test_move_not_eliminated_when_read(self, opt: PeepholeOptimizer) -> None:
		"""movq %rax, %rbx; addq %rbx, %rcx - %rbx is read, so move is kept."""
		asm = "\tmovq %rax, %rbx\n\taddq %rbx, %rcx"
		result = opt.optimize(asm)
		assert "\tmovq %rax, %rbx" in result

	def test_move_not_eliminated_when_dst_used_in_src(self, opt: PeepholeOptimizer) -> None:
		"""movq %rax, %rbx; movq (%rbx), %rbx - %rbx is read in source."""
		asm = "\tmovq %rax, %rbx\n\tmovq (%rbx), %rbx"
		result = opt.optimize(asm)
		assert "\tmovq %rax, %rbx" in result

	def test_move_then_xor_zero_not_eliminated(self, opt: PeepholeOptimizer) -> None:
		"""xorq %rbx, %rbx has %rbx in source operand, so dead move doesn't fire."""
		asm = "\tmovq %rax, %rbx\n\txorq %rbx, %rbx"
		result = opt.optimize(asm)
		assert "\tmovq %rax, %rbx" in result
		assert "\txorq %rbx, %rbx" in result

	def test_different_dst_not_eliminated(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq %rax, %rbx\n\tmovq %rcx, %rdx"
		result = opt.optimize(asm)
		assert "\tmovq %rax, %rbx" in result
		assert "\tmovq %rcx, %rdx" in result


class TestMovZeroToXor:
	"""Pattern: movq $0, %rA -> xorq %rA, %rA (already existed, verifying it works)."""

	def test_movq_zero_to_xor(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq $0, %rax"
		result = opt.optimize(asm)
		assert result == "\txorq %rax, %rax"

	def test_movq_nonzero_not_changed(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq $1, %rax"
		result = opt.optimize(asm)
		assert result == "\tmovq $1, %rax"

	def test_movq_zero_various_regs(self, opt: PeepholeOptimizer) -> None:
		for reg in ["%rax", "%rbx", "%rcx", "%rdx", "%rdi", "%rsi"]:
			asm = f"\tmovq $0, {reg}"
			result = opt.optimize(asm)
			assert result == f"\txorq {reg}, {reg}"


class TestPatternInteractions:
	"""Test that new patterns interact correctly with existing ones."""

	def test_lea_zero_fold_then_self_move_elim(self, opt: PeepholeOptimizer) -> None:
		"""leaq 0(%rax), %rax should fold to movq then eliminate as self-move."""
		asm = "\tleaq 0(%rax), %rax"
		result = opt.optimize(asm)
		assert result == ""

	def test_duplicate_move_with_zero(self, opt: PeepholeOptimizer) -> None:
		"""movq $0, %rax; movq $0, %rax -> both become xorq; second is dead move."""
		asm = "\tmovq $0, %rax\n\tmovq $0, %rax"
		result = opt.optimize(asm)
		# Both become xorq %rax, %rax; xorq has %rax in src so dead move won't fire,
		# but duplicate move pattern doesn't apply to xorq either.
		# The self-move check also doesn't apply. So we get two xorq lines.
		assert "\txorq %rax, %rax" in result

	def test_dead_move_before_zero(self, opt: PeepholeOptimizer) -> None:
		"""movq %rcx, %rax; movq $0, %rax -> movq $0 fires, dead move eliminated."""
		asm = "\tmovq %rcx, %rax\n\tmovq $0, %rax"
		result = opt.optimize(asm)
		assert result == "\txorq %rax, %rax"
