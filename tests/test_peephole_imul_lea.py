"""Tests for lea-based strength reduction of multiply-by-constant."""

import pytest

from compiler.peephole import PeepholeOptimizer


@pytest.fixture
def opt() -> PeepholeOptimizer:
	return PeepholeOptimizer()


def _optimize(opt: PeepholeOptimizer, lines: list[str]) -> str:
	asm = "\n".join(lines) + "\n"
	return opt.optimize(asm)


class TestImulToLea2Operand:
	"""2-operand imulq $imm, %reg -> lea/shift."""

	def test_imulq_by_2(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $2, %rax"])
		assert "leaq (,%rax,2), %rax" in result
		assert "imulq" not in result

	def test_imulq_by_3(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $3, %rax"])
		assert "leaq (%rax,%rax,2), %rax" in result
		assert "imulq" not in result

	def test_imulq_by_4(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $4, %rcx"])
		assert "leaq (,%rcx,4), %rcx" in result
		assert "imulq" not in result

	def test_imulq_by_5(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $5, %rdx"])
		assert "leaq (%rdx,%rdx,4), %rdx" in result
		assert "imulq" not in result

	def test_imulq_by_8(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $8, %rsi"])
		assert "leaq (,%rsi,8), %rsi" in result
		assert "imulq" not in result

	def test_imulq_by_9(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $9, %rdi"])
		assert "leaq (%rdi,%rdi,8), %rdi" in result
		assert "imulq" not in result

	def test_imulq_by_0_zeros_reg(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $0, %rax"])
		assert "xorl %eax, %eax" in result
		assert "imulq" not in result

	def test_imulq_by_1_eliminated(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $1, %rax"])
		assert "imulq" not in result
		assert "leaq" not in result

	def test_imulq_by_16_uses_shift(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $16, %rax"])
		assert "shlq $4, %rax" in result
		assert "imulq" not in result

	def test_imulq_non_optimizable_unchanged(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $7, %rax"])
		assert "imulq" in result


class TestImulToLea3Operand:
	"""3-operand imulq $imm, %src, %dst -> lea."""

	def test_imulq3_by_2(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $2, %rax, %rdx"])
		assert "leaq (,%rax,2), %rdx" in result
		assert "imulq" not in result

	def test_imulq3_by_3(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $3, %rax, %rdx"])
		assert "leaq (%rax,%rax,2), %rdx" in result
		assert "imulq" not in result

	def test_imulq3_by_5(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $5, %rcx, %rbx"])
		assert "leaq (%rcx,%rcx,4), %rbx" in result
		assert "imulq" not in result

	def test_imulq3_by_9(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $9, %rsi, %rdi"])
		assert "leaq (%rsi,%rsi,8), %rdi" in result
		assert "imulq" not in result

	def test_imulq3_by_4(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $4, %r8, %r9"])
		assert "leaq (,%r8,4), %r9" in result
		assert "imulq" not in result

	def test_imulq3_by_8(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $8, %r10, %r11"])
		assert "leaq (,%r10,8), %r11" in result
		assert "imulq" not in result

	def test_imulq3_by_1_becomes_mov(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $1, %rax, %rdx"])
		assert "movq %rax, %rdx" in result
		assert "imulq" not in result

	def test_imulq3_by_0_zeros_dst(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $0, %rax, %rdx"])
		assert "xorl %edx, %edx" in result
		assert "imulq" not in result

	def test_imulq3_non_optimizable_unchanged(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, ["\timulq $7, %rax, %rdx"])
		assert "imulq" in result


class TestImulLeaPreservesContext:
	"""Ensure surrounding instructions are preserved."""

	def test_surrounding_instructions_preserved(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, [
			"\tmovq %rdi, %rax",
			"\timulq $3, %rax",
			"\tret",
		])
		lines = result.strip().split("\n")
		assert len(lines) == 3
		assert "movq" in lines[0]
		assert "leaq" in lines[1]
		assert "ret" in lines[2]

	def test_multiple_imuls_optimized(self, opt: PeepholeOptimizer) -> None:
		result = _optimize(opt, [
			"\timulq $5, %rax",
			"\timulq $9, %rcx",
		])
		assert "leaq" in result
		assert result.count("leaq") == 2
		assert "imulq" not in result
