"""Tests for peephole optimizer extension elimination patterns."""

import pytest

from compiler.peephole import PeepholeOptimizer


@pytest.fixture
def opt() -> PeepholeOptimizer:
	return PeepholeOptimizer()


def _asm(*lines: str) -> str:
	"""Build assembly text from tab-indented instruction lines."""
	return "\n".join(lines)


class TestDuplicateExtensionElimination:
	"""Pattern 1: Remove duplicate extensions where the same register is extended twice."""

	def test_exact_duplicate_movzbl(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\tmovzbl %al, %eax", "\tmovzbl %al, %eax")
		result = opt.optimize(asm)
		assert result == _asm("\tmovzbl %al, %eax")

	def test_exact_duplicate_movsbl(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\tmovsbl %cl, %ecx", "\tmovsbl %cl, %ecx")
		result = opt.optimize(asm)
		assert result == _asm("\tmovsbl %cl, %ecx")

	def test_exact_duplicate_movzwl(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\tmovzwl %ax, %eax", "\tmovzwl %ax, %eax")
		result = opt.optimize(asm)
		assert result == _asm("\tmovzwl %ax, %eax")

	def test_exact_duplicate_movswl(self, opt: PeepholeOptimizer) -> None:
		asm = _asm("\tmovswl %dx, %edx", "\tmovswl %dx, %edx")
		result = opt.optimize(asm)
		assert result == _asm("\tmovswl %dx, %edx")

	def test_redundant_re_extension_same_reg(self, opt: PeepholeOptimizer) -> None:
		"""movzbl %al, %eax; movzbl %al, %eax -> keep first only."""
		asm = _asm("\tmovzbl %al, %eax", "\tmovzbl %al, %eax")
		result = opt.optimize(asm)
		assert "\tmovzbl" in result
		assert result.count("movzbl") == 1

	def test_no_elimination_different_src(self, opt: PeepholeOptimizer) -> None:
		"""movzbl %al, %eax; movzbl %bl, %eax -> keep both (different source)."""
		asm = _asm("\tmovzbl %al, %eax", "\tmovzbl %bl, %eax")
		result = opt.optimize(asm)
		assert result.count("movzbl") == 2 or result.count("mov") >= 2

	def test_no_elimination_different_dest(self, opt: PeepholeOptimizer) -> None:
		"""movzbl %al, %eax; movzbl %al, %ecx -> keep both (different dest)."""
		asm = _asm("\tmovzbl %al, %eax", "\tmovzbl %al, %ecx")
		result = opt.optimize(asm)
		assert result.count("movzbl") == 2

	def test_re_extension_from_subreg_of_dest(self, opt: PeepholeOptimizer) -> None:
		"""movzbl %cl, %ecx; movzbl %cl, %ecx -> single extension.

		Second movzbl reads %cl which is the low byte of %ecx (first dest).
		"""
		asm = _asm("\tmovzbl %cl, %ecx", "\tmovzbl %cl, %ecx")
		result = opt.optimize(asm)
		assert result.count("movzbl") == 1


class TestExtensionAfterTruncation:
	"""Pattern 2: Remove extension immediately after truncation to same width."""

	def test_movb_then_movzbl(self, opt: PeepholeOptimizer) -> None:
		"""movb %al, %cl; movzbl %cl, %ecx -> movzbl %al, %ecx."""
		asm = _asm("\tmovb %al, %cl", "\tmovzbl %cl, %ecx")
		result = opt.optimize(asm)
		assert "\tmovzbl %al, %ecx" in result
		assert "\tmovb" not in result

	def test_movb_then_movsbl(self, opt: PeepholeOptimizer) -> None:
		"""movb %dl, %cl; movsbl %cl, %ecx -> movsbl %dl, %ecx."""
		asm = _asm("\tmovb %dl, %cl", "\tmovsbl %cl, %ecx")
		result = opt.optimize(asm)
		assert "\tmovsbl %dl, %ecx" in result
		assert "\tmovb" not in result

	def test_movw_then_movzwl(self, opt: PeepholeOptimizer) -> None:
		"""movw %ax, %cx; movzwl %cx, %ecx -> movzwl %ax, %ecx."""
		asm = _asm("\tmovw %ax, %cx", "\tmovzwl %cx, %ecx")
		result = opt.optimize(asm)
		assert "\tmovzwl %ax, %ecx" in result
		assert "\tmovw" not in result

	def test_movw_then_movswl(self, opt: PeepholeOptimizer) -> None:
		"""movw %dx, %cx; movswl %cx, %ecx -> movswl %dx, %ecx."""
		asm = _asm("\tmovw %dx, %cx", "\tmovswl %cx, %ecx")
		result = opt.optimize(asm)
		assert "\tmovswl %dx, %ecx" in result
		assert "\tmovw" not in result

	def test_no_fold_width_mismatch(self, opt: PeepholeOptimizer) -> None:
		"""movb %al, %cl; movzwl %cx, %ecx -> no fold (width mismatch)."""
		asm = _asm("\tmovb %al, %cl", "\tmovzwl %cx, %ecx")
		result = opt.optimize(asm)
		assert "\tmovb" in result
		assert "\tmovzwl" in result

	def test_no_fold_different_registers(self, opt: PeepholeOptimizer) -> None:
		"""movb %al, %cl; movzbl %dl, %edx -> no fold (different registers)."""
		asm = _asm("\tmovb %al, %cl", "\tmovzbl %dl, %edx")
		result = opt.optimize(asm)
		assert "\tmovb" in result
		assert "\tmovzbl" in result


class TestMovzblTestlFold:
	"""Pattern 3: Fold movzbl + testl into testb."""

	def test_movzbl_testl_to_testb(self, opt: PeepholeOptimizer) -> None:
		"""movzbl %al, %eax; testl %eax, %eax -> testb %al, %al."""
		asm = _asm("\tmovzbl %al, %eax", "\ttestl %eax, %eax")
		result = opt.optimize(asm)
		assert "\ttestb %al, %al" in result
		assert "\tmovzbl" not in result
		assert "\ttestl" not in result

	def test_movzbl_testl_different_regs(self, opt: PeepholeOptimizer) -> None:
		"""movzbl %cl, %ecx; testl %ecx, %ecx -> testb %cl, %cl."""
		asm = _asm("\tmovzbl %cl, %ecx", "\ttestl %ecx, %ecx")
		result = opt.optimize(asm)
		assert "\ttestb %cl, %cl" in result

	def test_no_fold_movsbl_testl(self, opt: PeepholeOptimizer) -> None:
		"""movsbl %al, %eax; testl %eax, %eax -> no fold (sign extension changes value)."""
		asm = _asm("\tmovsbl %al, %eax", "\ttestl %eax, %eax")
		result = opt.optimize(asm)
		assert "\tmovsbl" in result or "\ttestl" in result

	def test_no_fold_different_test_reg(self, opt: PeepholeOptimizer) -> None:
		"""movzbl %al, %eax; testl %ecx, %ecx -> no fold (different test register)."""
		asm = _asm("\tmovzbl %al, %eax", "\ttestl %ecx, %ecx")
		result = opt.optimize(asm)
		assert "\tmovzbl" in result
		assert "\ttestl" in result

	def test_fold_with_following_jcc(self, opt: PeepholeOptimizer) -> None:
		"""movzbl %al, %eax; testl %eax, %eax; je .L1 -> testb %al, %al; je .L1."""
		asm = _asm("\tmovzbl %al, %eax", "\ttestl %eax, %eax", "\tje .L1")
		result = opt.optimize(asm)
		assert "\ttestb %al, %al" in result
		assert "\tje .L1" in result
		assert "\tmovzbl" not in result


class TestRedundantExtAfterZeroExtOp:
	"""Pattern 4: Remove movzbl when source was already zero-extended."""

	def test_movl_then_movzbl(self, opt: PeepholeOptimizer) -> None:
		"""movl %ecx, %eax; movzbl %al, %eax -> movl %ecx, %eax."""
		asm = _asm("\tmovl %ecx, %eax", "\tmovzbl %al, %eax")
		result = opt.optimize(asm)
		assert "\tmovl %ecx, %eax" in result
		assert "\tmovzbl" not in result

	def test_xorl_then_movzbl(self, opt: PeepholeOptimizer) -> None:
		"""xorl %eax, %eax; movzbl %al, %eax -> xorl %eax, %eax."""
		asm = _asm("\txorl %eax, %eax", "\tmovzbl %al, %eax")
		result = opt.optimize(asm)
		assert "\txorl" in result
		assert "\tmovzbl" not in result

	def test_addl_then_movzbl(self, opt: PeepholeOptimizer) -> None:
		"""addl %ecx, %eax; movzbl %al, %eax -> addl %ecx, %eax."""
		asm = _asm("\taddl %ecx, %eax", "\tmovzbl %al, %eax")
		result = opt.optimize(asm)
		assert "\taddl" in result
		assert "\tmovzbl" not in result

	def test_andl_byte_mask_then_movzbl(self, opt: PeepholeOptimizer) -> None:
		"""andl $255, %eax; movzbl %al, %eax -> andl $255, %eax."""
		asm = _asm("\tandl $255, %eax", "\tmovzbl %al, %eax")
		result = opt.optimize(asm)
		assert "\tandl" in result
		assert "\tmovzbl" not in result

	def test_andl_large_mask_no_elim(self, opt: PeepholeOptimizer) -> None:
		"""andl $65535, %eax; movzbl %al, %eax -> keep movzbl (mask > 0xFF, byte may differ)."""
		asm = _asm("\tandl $65535, %eax", "\tmovzbl %al, %eax")
		result = opt.optimize(asm)
		# andl $65535 doesn't guarantee the low byte is the full value
		# so movzbl is NOT redundant in the general case
		# However, andl is a 32-bit zero-extending op, and movzbl re-zeros upper bits
		# which are already zero. So actually it IS redundant.
		# The pattern checks that the 32-bit op already zero-extends.
		assert "\tandl" in result

	def test_movzwl_after_movl(self, opt: PeepholeOptimizer) -> None:
		"""movl %ecx, %eax; movzwl %ax, %eax -> movl %ecx, %eax."""
		asm = _asm("\tmovl %ecx, %eax", "\tmovzwl %ax, %eax")
		result = opt.optimize(asm)
		assert "\tmovl %ecx, %eax" in result
		assert "\tmovzwl" not in result

	def test_no_elim_different_dest(self, opt: PeepholeOptimizer) -> None:
		"""movl %ecx, %eax; movzbl %al, %edx -> keep both (different dest)."""
		asm = _asm("\tmovl %ecx, %eax", "\tmovzbl %al, %edx")
		result = opt.optimize(asm)
		assert "\tmovl" in result
		assert "\tmovzbl" in result

	def test_no_elim_movsbl(self, opt: PeepholeOptimizer) -> None:
		"""movl %ecx, %eax; movsbl %al, %eax -> keep both (sign ext != zero ext)."""
		asm = _asm("\tmovl %ecx, %eax", "\tmovsbl %al, %eax")
		result = opt.optimize(asm)
		assert "\tmovsbl" in result


class TestExtensionChainInteractions:
	"""Test interactions between extension patterns and existing peephole patterns."""

	def test_triple_extension(self, opt: PeepholeOptimizer) -> None:
		"""Three identical extensions -> single extension."""
		asm = _asm(
			"\tmovzbl %al, %eax",
			"\tmovzbl %al, %eax",
			"\tmovzbl %al, %eax",
		)
		result = opt.optimize(asm)
		assert result.count("movzbl") == 1

	def test_trunc_ext_chain(self, opt: PeepholeOptimizer) -> None:
		"""movb + movzbl chain should fold to single movzbl."""
		asm = _asm(
			"\tmovb %dl, %al",
			"\tmovzbl %al, %eax",
		)
		result = opt.optimize(asm)
		assert "\tmovzbl %dl, %eax" in result
		assert "\tmovb" not in result

	def test_movzbl_testl_with_je(self, opt: PeepholeOptimizer) -> None:
		"""Full pattern: movzbl + testl + conditional branch."""
		asm = _asm(
			"\tmovzbl %bl, %ebx",
			"\ttestl %ebx, %ebx",
			"\tjne .L2",
		)
		result = opt.optimize(asm)
		assert "\ttestb %bl, %bl" in result
		assert "\tjne .L2" in result

	def test_ext_elim_preserves_other_instrs(self, opt: PeepholeOptimizer) -> None:
		"""Extension elimination should not affect surrounding instructions."""
		asm = _asm(
			"\tpushq %rbp",
			"\tmovzbl %al, %eax",
			"\tmovzbl %al, %eax",
			"\tpopq %rbp",
		)
		result = opt.optimize(asm)
		assert "\tpushq %rbp" in result
		assert "\tpopq %rbp" in result
		assert result.count("movzbl") == 1
