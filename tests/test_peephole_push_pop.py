"""Tests for peephole push/pop elimination, self-move removal, and mov-zero folding."""

import pytest

from compiler.peephole import PeepholeOptimizer


@pytest.fixture
def opt() -> PeepholeOptimizer:
	return PeepholeOptimizer()


class TestPushPopElimination:
	"""Test redundant pushq/popq same-register pair removal."""

	def test_push_pop_same_register_rax(self, opt: PeepholeOptimizer) -> None:
		asm = "\tpushq %rax\n\tpopq %rax"
		result = opt.optimize(asm)
		assert "pushq" not in result
		assert "popq" not in result

	def test_push_pop_same_register_rbx(self, opt: PeepholeOptimizer) -> None:
		asm = "\tpushq %rbx\n\tpopq %rbx"
		result = opt.optimize(asm)
		assert "pushq" not in result
		assert "popq" not in result

	def test_push_pop_same_register_r12(self, opt: PeepholeOptimizer) -> None:
		asm = "\tpushq %r12\n\tpopq %r12"
		result = opt.optimize(asm)
		assert "pushq" not in result
		assert "popq" not in result

	def test_push_pop_different_registers_becomes_mov(self, opt: PeepholeOptimizer) -> None:
		asm = "\tpushq %rax\n\tpopq %rbx"
		result = opt.optimize(asm)
		assert "pushq" not in result
		assert "popq" not in result
		assert "\tmovq %rax, %rbx" in result

	def test_push_pop_not_adjacent_preserved(self, opt: PeepholeOptimizer) -> None:
		asm = "\tpushq %rax\n\tmovq %rcx, %rdx\n\tpopq %rax"
		result = opt.optimize(asm)
		assert "pushq" in result
		assert "popq" in result

	def test_push_pop_surrounding_code_preserved(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq %rdi, %rsi\n\tpushq %rcx\n\tpopq %rcx\n\tmovq %rax, %rbx"
		result = opt.optimize(asm)
		assert "\tmovq %rdi, %rsi" in result
		assert "\tmovq %rax, %rbx" in result
		assert "pushq" not in result
		assert "popq" not in result

	def test_multiple_push_pop_pairs(self, opt: PeepholeOptimizer) -> None:
		asm = "\tpushq %rax\n\tpopq %rax\n\tpushq %rbx\n\tpopq %rbx"
		result = opt.optimize(asm)
		assert "pushq" not in result
		assert "popq" not in result


class TestSelfMoveElimination:
	"""Test movq/movl/movw/movb self-move removal."""

	def test_movq_self_move(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq %rax, %rax"
		result = opt.optimize(asm)
		assert "movq" not in result

	def test_movl_self_move(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovl %eax, %eax"
		result = opt.optimize(asm)
		assert "movl" not in result

	def test_movw_self_move(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovw %ax, %ax"
		result = opt.optimize(asm)
		assert "movw" not in result

	def test_movb_self_move(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovb %al, %al"
		result = opt.optimize(asm)
		assert "movb" not in result

	def test_movq_different_regs_preserved(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq %rax, %rbx"
		result = opt.optimize(asm)
		assert "\tmovq %rax, %rbx" in result

	def test_movl_different_regs_preserved(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovl %eax, %ebx"
		result = opt.optimize(asm)
		assert "\tmovl %eax, %ebx" in result

	def test_movw_different_regs_preserved(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovw %ax, %bx"
		result = opt.optimize(asm)
		assert "\tmovw %ax, %bx" in result

	def test_movb_different_regs_preserved(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovb %al, %bl"
		result = opt.optimize(asm)
		assert "\tmovb %al, %bl" in result

	def test_self_move_r8_registers(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq %r8, %r8"
		result = opt.optimize(asm)
		assert "movq" not in result

	def test_self_move_with_surrounding_code(self, opt: PeepholeOptimizer) -> None:
		asm = "\taddq $1, %rax\n\tmovq %rbx, %rbx\n\tsubq $2, %rcx"
		result = opt.optimize(asm)
		assert "\taddq $1, %rax" in result
		assert "\tsubq $2, %rcx" in result
		assert "movq %rbx, %rbx" not in result

	def test_multiple_self_moves(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq %rax, %rax\n\tmovl %ebx, %ebx\n\tmovw %cx, %cx"
		result = opt.optimize(asm)
		assert "mov" not in result


class TestMovZeroToXor:
	"""Test movq/movl/movw/movb $0 -> xorl folding."""

	def test_movq_zero_to_xorl(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq $0, %rax"
		result = opt.optimize(asm)
		assert "\txorl %eax, %eax" in result
		assert "movq" not in result

	def test_movl_zero_to_xorl(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovl $0, %eax"
		result = opt.optimize(asm)
		assert "\txorl %eax, %eax" in result
		assert "movl" not in result

	def test_movw_zero_to_xorl(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovw $0, %ax"
		result = opt.optimize(asm)
		assert "\txorl %eax, %eax" in result
		assert "movw" not in result

	def test_movb_zero_to_xorl(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovb $0, %al"
		result = opt.optimize(asm)
		assert "\txorl %eax, %eax" in result
		assert "movb" not in result

	def test_movq_zero_r8_register(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq $0, %r8"
		result = opt.optimize(asm)
		assert "\txorl %r8d, %r8d" in result

	def test_movl_zero_r10_register(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovl $0, %r10d"
		result = opt.optimize(asm)
		assert "\txorl %r10d, %r10d" in result

	def test_movw_zero_r12_register(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovw $0, %r12w"
		result = opt.optimize(asm)
		assert "\txorl %r12d, %r12d" in result

	def test_movb_zero_r9_register(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovb $0, %r9b"
		result = opt.optimize(asm)
		assert "\txorl %r9d, %r9d" in result

	def test_movq_nonzero_preserved(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq $42, %rax"
		result = opt.optimize(asm)
		assert "\tmovq $42, %rax" in result

	def test_movl_nonzero_preserved(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovl $1, %eax"
		result = opt.optimize(asm)
		assert "\tmovl $1, %eax" in result

	def test_multiple_zero_movs(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq $0, %rax\n\tmovl $0, %ebx\n\tmovw $0, %cx\n\tmovb $0, %dl"
		result = opt.optimize(asm)
		assert "\txorl %eax, %eax" in result
		assert "\txorl %ebx, %ebx" in result
		assert "\txorl %ecx, %ecx" in result
		assert "\txorl %edx, %edx" in result
		assert "mov" not in result

	def test_movw_zero_unknown_reg_unchanged(self, opt: PeepholeOptimizer) -> None:
		"""If reg can't be mapped to 32-bit, don't transform."""
		asm = "\tmovw $0, %sp"
		result = opt.optimize(asm)
		# %sp is not in the reg16_to_reg32 map, so should be unchanged
		assert "\tmovw $0, %sp" in result

	def test_movb_zero_unknown_reg_unchanged(self, opt: PeepholeOptimizer) -> None:
		"""If reg can't be mapped to 32-bit, don't transform."""
		asm = "\tmovb $0, %spl"
		result = opt.optimize(asm)
		# %spl is not in the reg8_to_reg32 map, so should be unchanged
		assert "\tmovb $0, %spl" in result


class TestCombinedPatterns:
	"""Test interaction of multiple patterns together."""

	def test_push_pop_then_self_move(self, opt: PeepholeOptimizer) -> None:
		asm = "\tpushq %rax\n\tpopq %rax\n\tmovq %rbx, %rbx"
		result = opt.optimize(asm)
		assert "pushq" not in result
		assert "popq" not in result
		assert "movq" not in result

	def test_self_move_then_zero_mov(self, opt: PeepholeOptimizer) -> None:
		asm = "\tmovq %rax, %rax\n\tmovq $0, %rbx"
		result = opt.optimize(asm)
		assert "movq" not in result
		assert "\txorl %ebx, %ebx" in result

	def test_all_three_patterns(self, opt: PeepholeOptimizer) -> None:
		asm = (
			"\tpushq %rcx\n\tpopq %rcx\n"
			"\tmovl %edx, %edx\n"
			"\tmovq $0, %rsi\n"
			"\tmovb $0, %al"
		)
		result = opt.optimize(asm)
		assert "pushq" not in result
		assert "popq" not in result
		assert "movl %edx" not in result
		assert "\txorl %esi, %esi" in result
		assert "\txorl %eax, %eax" in result
