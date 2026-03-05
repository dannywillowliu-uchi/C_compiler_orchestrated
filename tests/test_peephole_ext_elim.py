"""Tests for peephole optimizer redundant sign/zero extension elimination."""

import pytest

from compiler.peephole import PeepholeOptimizer


@pytest.fixture
def opt() -> PeepholeOptimizer:
	return PeepholeOptimizer()


def _asm(*lines: str) -> str:
	"""Build assembly text from tab-indented instruction lines."""
	return "\n".join(lines)


class TestRedundantMovslqAfter32BitOp:
	"""movslq %eXX, %rXX is redundant after a 32-bit op that wrote %eXX."""

	def test_movl_then_movslq(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\tmovl %edi, %eax", "\tmovslq %eax, %rax")
		result = opt.optimize(asm)
		assert "movslq" not in result
		assert "\tmovl %edi, %eax" in result

	def test_addl_then_movslq(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\taddl %edi, %eax", "\tmovslq %eax, %rax")
		result = opt.optimize(asm)
		assert "movslq" not in result
		assert "\taddl %edi, %eax" in result

	def test_xorl_then_movslq(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\txorl %ecx, %edx", "\tmovslq %edx, %rdx")
		result = opt.optimize(asm)
		assert "movslq" not in result

	def test_subl_then_movslq(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\tsubl $1, %ecx", "\tmovslq %ecx, %rcx")
		result = opt.optimize(asm)
		assert "movslq" not in result

	def test_andl_then_movslq(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\tandl $0xff, %eax", "\tmovslq %eax, %rax")
		result = opt.optimize(asm)
		assert "movslq" not in result

	def test_imull_then_movslq(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\timull %edi, %eax", "\tmovslq %eax, %rax")
		result = opt.optimize(asm)
		assert "movslq" not in result

	def test_shll_then_movslq(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\tshll $2, %eax", "\tmovslq %eax, %rax")
		result = opt.optimize(asm)
		assert "movslq" not in result

	def test_orl_then_movslq(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\torl $0x80, %eax", "\tmovslq %eax, %rax")
		result = opt.optimize(asm)
		assert "movslq" not in result

	def test_movslq_different_reg_kept(self, opt: PeepholeOptimizer) -> None:
		"""movslq %eax, %rbx should NOT be eliminated (different physical reg)."""
		asm = _asm("\tmovl %edi, %eax", "\tmovslq %eax, %rbx")
		result = opt.optimize(asm)
		assert "movslq" in result

	def test_movslq_no_prior_32bit_op_kept(self, opt: PeepholeOptimizer) -> None:
		"""movslq without a preceding 32-bit op should be kept."""
		asm = _asm("\tmovq %rdi, %rax", "\tmovslq %eax, %rax")
		result = opt.optimize(asm)
		assert "movslq" in result

	def test_movslq_wrong_reg_kept(self, opt: PeepholeOptimizer) -> None:
		"""32-bit op on %ecx then movslq %eax, %rax should NOT be eliminated."""
		asm = _asm("\tmovl %edi, %ecx", "\tmovslq %eax, %rax")
		result = opt.optimize(asm)
		assert "movslq" in result

	def test_r8_r15_registers(self, opt: PeepholeOptimizer) -> None:
		"""Works with extended registers like %r8d/%r8."""
		asm = _asm("\tmovl %edi, %r8d", "\tmovslq %r8d, %r8")
		result = opt.optimize(asm)
		assert "movslq" not in result

	def test_sarl_then_movslq(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\tsarl $1, %eax", "\tmovslq %eax, %rax")
		result = opt.optimize(asm)
		assert "movslq" not in result

	def test_shrl_then_movslq(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\tshrl $3, %eax", "\tmovslq %eax, %rax")
		result = opt.optimize(asm)
		assert "movslq" not in result


class TestRedundantCltqAfter32BitOp:
	"""cltq (sign-extend %eax -> %rax) is redundant after 32-bit op writing %eax."""

	def test_movl_then_cltq(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\tmovl %edi, %eax", "\tcltq")
		result = opt.optimize(asm)
		assert "cltq" not in result
		assert "\tmovl %edi, %eax" in result

	def test_addl_then_cltq(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\taddl $1, %eax", "\tcltq")
		result = opt.optimize(asm)
		assert "cltq" not in result

	def test_xorl_then_cltq(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\txorl %eax, %eax", "\tcltq")
		result = opt.optimize(asm)
		assert "cltq" not in result

	def test_andl_then_cltq(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\tandl $0xff, %eax", "\tcltq")
		result = opt.optimize(asm)
		assert "cltq" not in result

	def test_subl_then_cltq(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\tsubl %edx, %eax", "\tcltq")
		result = opt.optimize(asm)
		assert "cltq" not in result

	def test_cltq_without_32bit_op_kept(self, opt: PeepholeOptimizer) -> None:
		"""cltq after a 64-bit operation should be kept."""
		asm = _asm("\tmovq %rdi, %rax", "\tcltq")
		result = opt.optimize(asm)
		assert "cltq" in result

	def test_cltq_after_non_eax_32bit_op_kept(self, opt: PeepholeOptimizer) -> None:
		"""32-bit op on %ecx does not make cltq redundant."""
		asm = _asm("\tmovl %edi, %ecx", "\tcltq")
		result = opt.optimize(asm)
		assert "cltq" in result

	def test_imull_then_cltq(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\timull %ecx, %eax", "\tcltq")
		result = opt.optimize(asm)
		assert "cltq" not in result

	def test_shll_then_cltq(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\tshll $2, %eax", "\tcltq")
		result = opt.optimize(asm)
		assert "cltq" not in result


class TestConsecutiveZeroExtElimination:
	"""Consecutive zero-extension instructions where the second is redundant."""

	def test_movzbl_then_movzwl_same_reg(self, opt: PeepholeOptimizer) -> None:
		"""movzbl %al, %eax; movzwl %ax, %eax -> just movzbl."""
		asm = _asm("\tmovzbl %al, %eax", "\tmovzwl %ax, %eax")
		result = opt.optimize(asm)
		assert "movzwl" not in result
		assert "\tmovzbl %al, %eax" in result

	def test_movzbl_then_movzbl_same_reg(self, opt: PeepholeOptimizer) -> None:
		"""movzbl %al, %eax; movzbl %al, %eax -> single movzbl."""
		asm = _asm("\tmovzbl %al, %eax", "\tmovzbl %al, %eax")
		result = opt.optimize(asm)
		assert result.count("movzbl") == 1

	def test_movzwl_then_movzwl_same_reg(self, opt: PeepholeOptimizer) -> None:
		"""movzwl %ax, %eax; movzwl %ax, %eax -> single movzwl."""
		asm = _asm("\tmovzwl %ax, %eax", "\tmovzwl %ax, %eax")
		result = opt.optimize(asm)
		assert result.count("movzwl") == 1

	def test_movzbl_then_movslq_same_reg(self, opt: PeepholeOptimizer) -> None:
		"""movzbl %al, %eax; movslq %eax, %rax -> just movzbl (already zero-extended)."""
		asm = _asm("\tmovzbl %al, %eax", "\tmovslq %eax, %rax")
		result = opt.optimize(asm)
		assert "movslq" not in result
		assert "\tmovzbl %al, %eax" in result

	def test_movzwl_then_movslq_same_reg(self, opt: PeepholeOptimizer) -> None:
		"""movzwl %ax, %eax; movslq %eax, %rax -> just movzwl."""
		asm = _asm("\tmovzwl %ax, %eax", "\tmovslq %eax, %rax")
		result = opt.optimize(asm)
		assert "movslq" not in result
		assert "\tmovzwl %ax, %eax" in result

	def test_different_dest_regs_kept(self, opt: PeepholeOptimizer) -> None:
		"""movzbl %al, %eax; movzwl %cx, %ecx -> keep both (different regs)."""
		asm = _asm("\tmovzbl %al, %eax", "\tmovzwl %cx, %ecx")
		result = opt.optimize(asm)
		assert "movzbl" in result
		assert "movzwl" in result

	def test_movzbl_ecx_then_movzwl_ecx(self, opt: PeepholeOptimizer) -> None:
		"""movzbl %cl, %ecx; movzwl %cx, %ecx -> just movzbl."""
		asm = _asm("\tmovzbl %cl, %ecx", "\tmovzwl %cx, %ecx")
		result = opt.optimize(asm)
		assert "movzwl" not in result
		assert "\tmovzbl %cl, %ecx" in result

	def test_movzbl_then_movzwl_different_dest_kept(self, opt: PeepholeOptimizer) -> None:
		"""movzbl %al, %eax; movzwl %cx, %eax -> keep both (different source chain)."""
		asm = _asm("\tmovzbl %al, %eax", "\tmovzwl %cx, %eax")
		result = opt.optimize(asm)
		# %cx is not a sub-register of %eax, so this is not redundant
		assert "movzwl" in result


class TestMixedExtElimPatterns:
	"""Integration tests combining extension elimination with other patterns."""

	def test_movl_movslq_in_longer_sequence(self, opt: PeepholeOptimizer) -> None:
		"""Redundant movslq removed even in a longer instruction sequence."""
		asm = _asm(
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tmovl %edi, %eax",
			"\tmovslq %eax, %rax",
			"\tpopq %rbp",
			"\tret",
		)
		result = opt.optimize(asm)
		assert "movslq" not in result

	def test_addl_cltq_in_function(self, opt: PeepholeOptimizer) -> None:
		"""Redundant cltq removed inside a function body."""
		asm = _asm(
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\taddl $1, %eax",
			"\tcltq",
			"\tpopq %rbp",
			"\tret",
		)
		result = opt.optimize(asm)
		assert "cltq" not in result

	def test_chained_movzbl_movzwl_movslq(self, opt: PeepholeOptimizer) -> None:
		"""movzbl %al, %eax; movzwl %ax, %eax; movslq %eax, %rax -> just movzbl."""
		asm = _asm(
			"\tmovzbl %al, %eax",
			"\tmovzwl %ax, %eax",
			"\tmovslq %eax, %rax",
		)
		result = opt.optimize(asm)
		assert "movzwl" not in result
		assert "movslq" not in result
		assert "\tmovzbl %al, %eax" in result

	def test_movl_movslq_multiple_registers(self, opt: PeepholeOptimizer) -> None:
		"""Each register pair optimized independently."""
		asm = _asm(
			"\tmovl %edi, %eax",
			"\tmovslq %eax, %rax",
			"\tmovl %esi, %ecx",
			"\tmovslq %ecx, %rcx",
		)
		result = opt.optimize(asm)
		assert result.count("movslq") == 0

	def test_non_redundant_movslq_preserved(self, opt: PeepholeOptimizer) -> None:
		"""movslq between unrelated instructions is preserved."""
		asm = _asm(
			"\tcallq foo",
			"\tmovslq %eax, %rax",
		)
		result = opt.optimize(asm)
		assert "movslq" in result

	def test_32bit_op_with_intervening_instruction(self, opt: PeepholeOptimizer) -> None:
		"""movslq is NOT removed if there's an intervening instruction."""
		asm = _asm(
			"\tmovl %edi, %eax",
			"\taddq $1, %rbx",
			"\tmovslq %eax, %rax",
		)
		result = opt.optimize(asm)
		# The movslq is not adjacent to the movl, so it should be kept
		assert "movslq" in result
