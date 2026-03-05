"""Tests for LEA folding and address-mode peephole optimizations."""

import pytest

from compiler.peephole import PeepholeOptimizer


@pytest.fixture
def opt() -> PeepholeOptimizer:
	return PeepholeOptimizer()


# --- movq $0 -> xorq ---

class TestMovZeroToXor:
	def test_movq_zero_to_xorq(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq $0, %rax"
		result = opt.optimize(asm)
		assert result == "\txorq %rax, %rax"

	def test_movq_zero_different_regs(self, opt: PeepholeOptimizer) -> None:
		for reg in ["%rbx", "%rcx", "%rdx", "%rsi", "%rdi", "%r8", "%r9"]:
			asm = f"\tmovq $0, {reg}"
			result = opt.optimize(asm)
			assert result == f"\txorq {reg}, {reg}"

	def test_movl_zero_to_xorl(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovl $0, %eax"
		result = opt.optimize(asm)
		assert result == "\txorl %eax, %eax"

	def test_movq_nonzero_unchanged(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq $1, %rax"
		result = opt.optimize(asm)
		assert result == "\tmovq $1, %rax"


# --- movq $imm, %rA + addq %rA, %rB -> addq $imm, %rB ---

class TestImmAddFold:
	def test_basic_imm_add_fold(self, opt: PeepholeOptimizer) -> None:
		asm = "\n".join([
			"\tmovq $42, %rcx",
			"\taddq %rcx, %rax",
			"\tret",
		])
		result = opt.optimize(asm)
		assert "\taddq $42, %rax" in result
		assert "movq $42" not in result

	def test_negative_imm_add_fold(self, opt: PeepholeOptimizer) -> None:
		asm = "\n".join([
			"\tmovq $-8, %rcx",
			"\taddq %rcx, %rax",
			"\tret",
		])
		result = opt.optimize(asm)
		assert "\taddq $-8, %rax" in result

	def test_imm_add_fold_preserves_when_reg_live(self, opt: PeepholeOptimizer) -> None:
		"""If the temp register is still used, don't fold."""
		asm = "\n".join([
			"\tmovq $42, %rcx",
			"\taddq %rcx, %rax",
			"\tmovq %rcx, %rbx",
		])
		result = opt.optimize(asm)
		# %rcx is still used, so the fold should not happen
		assert "movq $42, %rcx" in result

	def test_imm_add_fold_same_reg_no_fold(self, opt: PeepholeOptimizer) -> None:
		"""movq $imm, %rA + addq %rA, %rA should not fold (src==dst)."""
		asm = "\n".join([
			"\tmovq $10, %rax",
			"\taddq %rax, %rax",
			"\tret",
		])
		result = opt.optimize(asm)
		# Should not produce addq $10, %rax since that would be wrong semantics
		assert "addq $10, %rax" not in result


# --- movq $imm, %rA + addq %rB, %rA -> leaq imm(%rB), %rA ---

class TestImmRegToLea:
	def test_basic_lea_reduction(self, opt: PeepholeOptimizer) -> None:
		asm = "\n".join([
			"\tmovq $16, %rcx",
			"\taddq %rdi, %rcx",
			"\tret",
		])
		result = opt.optimize(asm)
		assert "\tleaq 16(%rdi), %rcx" in result

	def test_negative_offset_lea(self, opt: PeepholeOptimizer) -> None:
		asm = "\n".join([
			"\tmovq $-24, %rax",
			"\taddq %rbp, %rax",
			"\tret",
		])
		result = opt.optimize(asm)
		assert "\tleaq -24(%rbp), %rax" in result

	def test_lea_reduction_same_reg_no_fold(self, opt: PeepholeOptimizer) -> None:
		"""addq src == dst should not fold."""
		asm = "\n".join([
			"\tmovq $8, %rax",
			"\taddq %rax, %rax",
			"\tret",
		])
		result = opt.optimize(asm)
		assert "leaq" not in result


# --- movq %rA, %rB + addq $imm, %rB -> leaq imm(%rA), %rB ---

class TestMovAddImmToLea:
	def test_basic_mov_addimm_lea(self, opt: PeepholeOptimizer) -> None:
		asm = "\n".join([
			"\tmovq %rdi, %rax",
			"\taddq $8, %rax",
			"\tret",
		])
		result = opt.optimize(asm)
		assert "\tleaq 8(%rdi), %rax" in result
		assert "movq %rdi" not in result

	def test_negative_offset(self, opt: PeepholeOptimizer) -> None:
		asm = "\n".join([
			"\tmovq %rsi, %rcx",
			"\taddq $-16, %rcx",
			"\tret",
		])
		result = opt.optimize(asm)
		assert "\tleaq -16(%rsi), %rcx" in result

	def test_zero_offset(self, opt: PeepholeOptimizer) -> None:
		asm = "\n".join([
			"\tmovq %rdi, %rax",
			"\taddq $0, %rax",
			"\tret",
		])
		result = opt.optimize(asm)
		# addq $0 is a noop, should be eliminated; movq remains or becomes self-move
		assert "addq $0" not in result

	def test_no_fold_when_src_live(self, opt: PeepholeOptimizer) -> None:
		"""Don't fold if source register is still used after."""
		asm = "\n".join([
			"\tmovq %rdi, %rax",
			"\taddq $8, %rax",
			"\tmovq %rdi, %rbx",
		])
		result = opt.optimize(asm)
		# %rdi is still live, so fold should not happen
		assert "movq %rdi, %rax" in result

	def test_same_reg_no_fold(self, opt: PeepholeOptimizer) -> None:
		"""movq %rA, %rA is a self-move, don't trigger lea fold."""
		asm = "\n".join([
			"\tmovq %rax, %rax",
			"\taddq $8, %rax",
			"\tret",
		])
		result = opt.optimize(asm)
		# self-move should be eliminated, addq remains
		assert "\taddq $8, %rax" in result

	def test_different_dst_no_fold(self, opt: PeepholeOptimizer) -> None:
		"""addq destination must match movq destination."""
		asm = "\n".join([
			"\tmovq %rdi, %rax",
			"\taddq $8, %rbx",
			"\tret",
		])
		result = opt.optimize(asm)
		assert "leaq" not in result


# --- movq %rA, %rB + addq %rC, %rB -> leaq (%rA,%rC), %rB ---

class TestMovAddRegToLea:
	def test_basic_reg_reg_lea(self, opt: PeepholeOptimizer) -> None:
		asm = "\n".join([
			"\tmovq %rdi, %rax",
			"\taddq %rsi, %rax",
			"\tret",
		])
		result = opt.optimize(asm)
		assert "\tleaq (%rdi,%rsi), %rax" in result

	def test_no_fold_when_src_live(self, opt: PeepholeOptimizer) -> None:
		asm = "\n".join([
			"\tmovq %rdi, %rax",
			"\taddq %rsi, %rax",
			"\tmovq %rdi, %rbx",
		])
		result = opt.optimize(asm)
		assert "movq %rdi, %rax" in result

	def test_no_fold_when_add_src_eq_dst(self, opt: PeepholeOptimizer) -> None:
		"""addq %rB, %rB doubles the value; leaq (%rA,%rB),%rB is different."""
		asm = "\n".join([
			"\tmovq %rdi, %rax",
			"\taddq %rax, %rax",
			"\tret",
		])
		result = opt.optimize(asm)
		assert "leaq" not in result

	def test_no_fold_when_add_dst_different(self, opt: PeepholeOptimizer) -> None:
		asm = "\n".join([
			"\tmovq %rdi, %rax",
			"\taddq %rsi, %rcx",
			"\tret",
		])
		result = opt.optimize(asm)
		assert "leaq" not in result


# --- Combined / chained patterns ---

class TestCombinedPatterns:
	def test_movq_zero_then_cmpq_zero(self, opt: PeepholeOptimizer) -> None:
		"""movq $0 + cmpq $0 folds to xorq (testq eliminated as redundant)."""
		asm = "\n".join([
			"\tmovq $0, %rax",
			"\tcmpq $0, %rax",
		])
		result = opt.optimize(asm)
		assert "\txorq %rax, %rax" in result

	def test_multiple_zero_moves(self, opt: PeepholeOptimizer) -> None:
		asm = "\n".join([
			"\tmovq $0, %rax",
			"\tmovq $0, %rbx",
			"\tmovq $0, %rcx",
		])
		result = opt.optimize(asm)
		assert "\txorq %rax, %rax" in result
		assert "\txorq %rbx, %rbx" in result
		assert "\txorq %rcx, %rcx" in result
		assert "movq $0" not in result

	def test_lea_fold_chain(self, opt: PeepholeOptimizer) -> None:
		"""Multiple lea-foldable sequences in a row."""
		asm = "\n".join([
			"\tmovq $8, %rcx",
			"\taddq %rdi, %rcx",
			"\tmovq $16, %rdx",
			"\taddq %rsi, %rdx",
			"\tret",
		])
		result = opt.optimize(asm)
		assert "\tleaq 8(%rdi), %rcx" in result
		assert "\tleaq 16(%rsi), %rdx" in result

	def test_imm_add_fold_then_use(self, opt: PeepholeOptimizer) -> None:
		"""Folded addq result should still be usable."""
		asm = "\n".join([
			"\tmovq $100, %rcx",
			"\taddq %rcx, %rax",
			"\tmovq %rax, -8(%rbp)",
			"\tret",
		])
		result = opt.optimize(asm)
		assert "\taddq $100, %rax" in result
		assert "\tmovq %rax, -8(%rbp)" in result

	def test_no_false_positive_on_labels(self, opt: PeepholeOptimizer) -> None:
		"""Patterns should not span across labels."""
		asm = "\n".join([
			"\tmovq %rdi, %rax",
			".L1:",
			"\taddq $8, %rax",
			"\tret",
		])
		result = opt.optimize(asm)
		# The movq and addq are separated by a label, no lea fold should happen
		assert "leaq" not in result

	def test_preserve_non_matching_instructions(self, opt: PeepholeOptimizer) -> None:
		"""Instructions that don't match any pattern should pass through unchanged."""
		asm = "\n".join([
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tcallq _foo",
			"\tpopq %rbp",
			"\tret",
		])
		result = opt.optimize(asm)
		assert "\tpushq %rbp" in result
		assert "\tcallq _foo" in result
		assert "\tpopq %rbp" in result
		assert "\tret" in result
