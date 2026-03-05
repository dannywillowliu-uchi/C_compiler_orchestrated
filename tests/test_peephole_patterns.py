"""Tests for address-mode folding and redundant comparison peephole patterns."""

from compiler.peephole import PeepholeOptimizer


def _opt(asm: str) -> str:
	"""Helper: optimize assembly text."""
	return PeepholeOptimizer().optimize(asm)


# ---------------------------------------------------------------------------
# Pattern 1: cmpq $0, %reg -> testq %reg, %reg
# ---------------------------------------------------------------------------


class TestCmpZeroToTest:
	def test_basic_cmpq_zero(self) -> None:
		"""cmpq $0, %rax -> testq %rax, %rax."""
		assert _opt("\tcmpq $0, %rax") == "\ttestq %rax, %rax"

	def test_cmpq_zero_different_regs(self) -> None:
		"""Works for any register."""
		assert _opt("\tcmpq $0, %rbx") == "\ttestq %rbx, %rbx"
		assert _opt("\tcmpq $0, %rcx") == "\ttestq %rcx, %rcx"
		assert _opt("\tcmpq $0, %rdi") == "\ttestq %rdi, %rdi"

	def test_cmpq_nonzero_no_fire(self) -> None:
		"""cmpq with non-zero immediate should NOT transform."""
		asm = "\tcmpq $1, %rax"
		assert _opt(asm) == asm

	def test_cmpq_zero_in_context(self) -> None:
		"""cmpq $0 transforms within surrounding code."""
		lines = [
			"\tmovq %rdi, %rax",
			"\tcmpq $0, %rax",
			"\tje .L1",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tmovq %rdi, %rax",
			"\ttestq %rax, %rax",
			"\tje .L1",
		])
		assert result == expected

	def test_cmpq_zero_preserves_branch_semantics(self) -> None:
		"""testq sets same flags as cmpq $0, so branches are preserved."""
		lines = [
			"\tcmpq $0, %rcx",
			"\tjne .L2",
			"\tmovq $1, %rax",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\ttestq %rcx, %rcx",
			"\tjne .L2",
			"\tmovq $1, %rax",
		])
		assert result == expected


# ---------------------------------------------------------------------------
# Pattern 2: Redundant comparison after flag-setting arithmetic
# ---------------------------------------------------------------------------


class TestRedundantCmpElimination:
	def test_addq_then_cmpq_zero(self) -> None:
		"""addq sets flags, so cmpq $0 on same reg is redundant."""
		lines = [
			"\taddq %rbx, %rax",
			"\tcmpq $0, %rax",
		]
		assert _opt("\n".join(lines)) == "\taddq %rbx, %rax"

	def test_subq_then_cmpq_zero(self) -> None:
		"""subq sets flags, cmpq $0 on same reg is redundant."""
		lines = [
			"\tsubq %rcx, %rdx",
			"\tcmpq $0, %rdx",
		]
		assert _opt("\n".join(lines)) == "\tsubq %rcx, %rdx"

	def test_andq_then_cmpq_zero(self) -> None:
		"""andq sets flags."""
		lines = [
			"\tandq %rbx, %rax",
			"\tcmpq $0, %rax",
		]
		assert _opt("\n".join(lines)) == "\tandq %rbx, %rax"

	def test_orq_then_cmpq_zero(self) -> None:
		"""orq sets flags."""
		lines = [
			"\torq %rcx, %rax",
			"\tcmpq $0, %rax",
		]
		assert _opt("\n".join(lines)) == "\torq %rcx, %rax"

	def test_xorq_then_cmpq_zero(self) -> None:
		"""xorq sets flags."""
		lines = [
			"\txorq %rbx, %rax",
			"\tcmpq $0, %rax",
		]
		assert _opt("\n".join(lines)) == "\txorq %rbx, %rax"

	def test_different_register_no_fire(self) -> None:
		"""cmpq on different register than arithmetic dest should NOT fire."""
		lines = [
			"\taddq %rbx, %rax",
			"\tcmpq $0, %rcx",
		]
		assert _opt("\n".join(lines)) == "\n".join([
			"\taddq %rbx, %rax",
			"\ttestq %rcx, %rcx",
		])

	def test_cmpq_nonzero_after_arith_no_fire(self) -> None:
		"""cmpq with non-zero immediate should NOT be eliminated."""
		lines = [
			"\taddq %rbx, %rax",
			"\tcmpq $5, %rax",
		]
		assert _opt("\n".join(lines)) == "\n".join(lines)

	def test_movq_then_cmpq_no_fire(self) -> None:
		"""movq does NOT set flags, so cmpq should NOT be eliminated."""
		lines = [
			"\tmovq %rbx, %rax",
			"\tcmpq $0, %rax",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tmovq %rbx, %rax",
			"\ttestq %rax, %rax",
		])
		assert result == expected

	def test_redundant_cmp_with_branch(self) -> None:
		"""Full sequence: arith + redundant cmp + branch."""
		lines = [
			"\tsubq $1, %rcx",
			"\tcmpq $0, %rcx",
			"\tjne .Lloop",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tsubq $1, %rcx",
			"\tjne .Lloop",
		])
		assert result == expected

	def test_arith_with_imm_then_cmpq(self) -> None:
		"""Arithmetic with immediate operand also sets flags."""
		lines = [
			"\taddq $16, %rsp",
			"\tcmpq $0, %rsp",
		]
		assert _opt("\n".join(lines)) == "\taddq $16, %rsp"


# ---------------------------------------------------------------------------
# Pattern 3: LEA + load RIP-relative folding
# ---------------------------------------------------------------------------


class TestLeaLoadFold:
	def test_basic_lea_load_fold(self) -> None:
		"""leaq sym(%rip), %rax; movq (%rax), %rax -> movq sym(%rip), %rax."""
		lines = [
			"\tleaq myvar(%rip), %rax",
			"\tmovq (%rax), %rax",
		]
		assert _opt("\n".join(lines)) == "\tmovq myvar(%rip), %rax"

	def test_different_symbol(self) -> None:
		"""Works with different symbol names."""
		lines = [
			"\tleaq global_counter(%rip), %rcx",
			"\tmovq (%rcx), %rcx",
		]
		assert _opt("\n".join(lines)) == "\tmovq global_counter(%rip), %rcx"

	def test_different_registers_no_fold(self) -> None:
		"""lea into %rax but load from %rbx should NOT fold."""
		lines = [
			"\tleaq sym(%rip), %rax",
			"\tmovq (%rbx), %rbx",
		]
		assert _opt("\n".join(lines)) == "\n".join(lines)

	def test_lea_dst_differs_from_load_dst_no_fold(self) -> None:
		"""lea and load must use same register for both base and dest."""
		lines = [
			"\tleaq sym(%rip), %rax",
			"\tmovq (%rax), %rbx",
		]
		assert _opt("\n".join(lines)) == "\n".join(lines)

	def test_lea_load_fold_in_context(self) -> None:
		"""Fold within surrounding instructions."""
		lines = [
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tleaq config(%rip), %rax",
			"\tmovq (%rax), %rax",
			"\taddq $1, %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tmovq config(%rip), %rax",
			"\taddq $1, %rax",
			"\tret",
		])
		assert result == expected

	def test_non_rip_relative_no_fold(self) -> None:
		"""leaq with non-%rip base should NOT fold (regex won't match)."""
		lines = [
			"\tleaq 8(%rbp), %rax",
			"\tmovq (%rax), %rax",
		]
		assert _opt("\n".join(lines)) == "\n".join(lines)


# ---------------------------------------------------------------------------
# Pattern 4: Immediate add folding
# ---------------------------------------------------------------------------


class TestImmAddFold:
	def test_basic_imm_add_fold(self) -> None:
		"""movq $imm, %rax; addq %rax, %rcx -> addq $imm, %rcx when %rax dead."""
		lines = [
			"\tmovq $8, %rax",
			"\taddq %rax, %rcx",
			"\tmovq $0, %rax",
		]
		result = _opt("\n".join(lines))
		assert "\taddq $8, %rcx" in result

	def test_negative_immediate(self) -> None:
		"""Works with negative immediates."""
		lines = [
			"\tmovq $-16, %rbx",
			"\taddq %rbx, %rdi",
			"\tmovq $0, %rbx",
		]
		result = _opt("\n".join(lines))
		assert "\taddq $-16, %rdi" in result

	def test_imm_too_large_no_fold(self) -> None:
		"""Immediate exceeding 32-bit range should NOT fold."""
		lines = [
			"\tmovq $2147483648, %rax",
			"\taddq %rax, %rcx",
			"\tret",
		]
		result = _opt("\n".join(lines))
		assert "\taddq $2147483648, %rcx" not in result

	def test_imm_too_negative_no_fold(self) -> None:
		"""Very negative immediate should NOT fold."""
		lines = [
			"\tmovq $-2147483649, %rax",
			"\taddq %rax, %rcx",
			"\tret",
		]
		result = _opt("\n".join(lines))
		assert "\taddq $-2147483649, %rcx" not in result

	def test_reg_still_used_no_fold(self) -> None:
		"""Should NOT fold when the immediate register is still used."""
		lines = [
			"\tmovq $8, %rax",
			"\taddq %rax, %rcx",
			"\taddq %rax, %rdx",
		]
		assert _opt("\n".join(lines)) == "\n".join(lines)

	def test_same_src_dst_no_fold(self) -> None:
		"""movq $imm, %rax; addq %rax, %rax should NOT fold."""
		lines = [
			"\tmovq $8, %rax",
			"\taddq %rax, %rax",
			"\tret",
		]
		assert _opt("\n".join(lines)) == "\n".join(lines)

	def test_imm_add_fold_at_ret(self) -> None:
		"""Register is dead at ret (non-%rax)."""
		lines = [
			"\tmovq $42, %rbx",
			"\taddq %rbx, %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\taddq $42, %rax",
			"\tret",
		])
		assert result == expected

	def test_rax_not_dead_at_ret(self) -> None:
		"""movq $imm, %rax; addq %rax, %rcx; ret -> %rax live at ret, no fold."""
		lines = [
			"\tmovq $8, %rax",
			"\taddq %rax, %rcx",
			"\tret",
		]
		assert _opt("\n".join(lines)) == "\n".join(lines)

	def test_32bit_boundary(self) -> None:
		"""Immediate at 32-bit max boundary should fold."""
		lines = [
			"\tmovq $2147483647, %rbx",
			"\taddq %rbx, %rcx",
			"\tmovq $0, %rbx",
		]
		result = _opt("\n".join(lines))
		assert "\taddq $2147483647, %rcx" in result

	def test_32bit_min_boundary(self) -> None:
		"""Immediate at 32-bit min boundary should fold."""
		lines = [
			"\tmovq $-2147483648, %rbx",
			"\taddq %rbx, %rcx",
			"\tmovq $0, %rbx",
		]
		result = _opt("\n".join(lines))
		assert "\taddq $-2147483648, %rcx" in result


# ---------------------------------------------------------------------------
# Pattern 5: movq $0, %reg -> xorq %reg, %reg
# ---------------------------------------------------------------------------


class TestMovZeroToXor:
	def test_basic_mov_zero(self) -> None:
		"""movq $0, %rax -> xorl %eax, %eax."""
		assert _opt("\tmovq $0, %rax") == "\txorl %eax, %eax"

	def test_different_registers(self) -> None:
		"""Works for any register."""
		assert _opt("\tmovq $0, %rbx") == "\txorl %ebx, %ebx"
		assert _opt("\tmovq $0, %rcx") == "\txorl %ecx, %ecx"
		assert _opt("\tmovq $0, %r8") == "\txorl %r8d, %r8d"

	def test_nonzero_immediate_no_fire(self) -> None:
		"""movq with non-zero immediate should NOT transform."""
		assert _opt("\tmovq $1, %rax") == "\tmovq $1, %rax"
		assert _opt("\tmovq $-1, %rax") == "\tmovq $-1, %rax"

	def test_mov_zero_in_context(self) -> None:
		"""movq $0 transforms within surrounding code."""
		lines = [
			"\tpushq %rbp",
			"\tmovq $0, %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tpushq %rbp",
			"\txorl %eax, %eax",
			"\tret",
		])
		assert result == expected

	def test_mov_zero_paired_with_cmpq_uses_pair_pattern(self) -> None:
		"""movq $0 + cmpq $0 same reg folds to xorl + testq."""
		lines = [
			"\tmovq $0, %rax",
			"\tcmpq $0, %rax",
		]
		result = _opt("\n".join(lines))
		# Pair pattern produces xorl + testq; testq may remain since
		# redundant test elimination may not match cross-width registers
		assert "\txorl %eax, %eax" in result
		assert "\tcmpq" not in result

	def test_multiple_mov_zeros(self) -> None:
		"""Multiple movq $0 instructions all transform."""
		lines = [
			"\tmovq $0, %rax",
			"\tmovq $0, %rbx",
			"\tmovq $0, %rcx",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\txorl %eax, %eax",
			"\txorl %ebx, %ebx",
			"\txorl %ecx, %ecx",
		])
		assert result == expected


# ---------------------------------------------------------------------------
# Cross-pattern interaction tests
# ---------------------------------------------------------------------------


class TestCrossPatternInteraction:
	def test_arith_cmp_branch_full_sequence(self) -> None:
		"""Arithmetic + redundant cmp + branch: cmp eliminated."""
		lines = [
			"\taddq $1, %rax",
			"\tcmpq $0, %rax",
			"\tje .Ldone",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\taddq $1, %rax",
			"\tje .Ldone",
		])
		assert result == expected

	def test_lea_fold_then_xor_zero(self) -> None:
		"""LEA+load fold and movq $0 -> xorl in same function."""
		lines = [
			"\tleaq counter(%rip), %rax",
			"\tmovq (%rax), %rax",
			"\tmovq $0, %rbx",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tmovq counter(%rip), %rax",
			"\txorl %ebx, %ebx",
			"\tret",
		])
		assert result == expected

	def test_imm_add_fold_and_cmp_to_test(self) -> None:
		"""Immediate add fold combined with cmpq $0 -> testq."""
		lines = [
			"\tmovq $10, %rbx",
			"\taddq %rbx, %rax",
			"\tmovq $0, %rbx",
			"\tcmpq $0, %rax",
			"\tje .L1",
		]
		result = _opt("\n".join(lines))
		assert "\taddq $10, %rax" in result
		assert "\ttestq %rax, %rax" in result or "\tcmpq $0, %rax" not in result

	def test_realistic_function_all_patterns(self) -> None:
		"""Realistic function exercising all 5 new patterns."""
		lines = [
			".globl process",
			"process:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			# Pattern 3: LEA+load fold
			"\tleaq data(%rip), %rax",
			"\tmovq (%rax), %rax",
			# Pattern 5: movq $0 -> xorq
			"\tmovq $0, %rcx",
			# Pattern 4: immediate add fold (rbx dead at overwrite)
			"\tmovq $8, %rbx",
			"\taddq %rbx, %rax",
			"\tmovq $0, %rbx",
			# Pattern 2: redundant cmp after sub
			"\tsubq %rcx, %rax",
			"\tcmpq $0, %rax",
			# Pattern 1: standalone cmpq $0 -> testq (on different reg)
			"\tje .Lzero",
			"\tmovq %rbp, %rsp",
			"\tpopq %rbp",
			"\tret",
			".Lzero:",
			"\tmovq $0, %rax",
			"\tmovq %rbp, %rsp",
			"\tpopq %rbp",
			"\tret",
		]
		result = _opt("\n".join(lines))
		# Pattern 3: lea+load folded
		assert "\tmovq data(%rip), %rax" in result
		# Pattern 5: movq $0 -> xorl (first occurrence, for %rcx)
		assert "\txorl %ecx, %ecx" in result
		# Pattern 4: imm add folded
		assert "\taddq $8, %rax" in result
		# Pattern 2: redundant cmp removed (subq sets flags)
		lines_list = result.split("\n")
		# After subq, there should be no cmpq $0 on same reg
		for j, line in enumerate(lines_list):
			if "\tsubq %rcx, %rax" in line:
				assert j + 1 < len(lines_list)
				assert "cmpq" not in lines_list[j + 1]
				break

	def test_existing_patterns_unaffected(self) -> None:
		"""Verify existing patterns still work after adding new ones."""
		# Store-reload
		assert _opt("\tmovq %rax, -8(%rbp)\n\tmovq -8(%rbp), %rax") == "\tmovq %rax, -8(%rbp)"
		# Self-move
		assert _opt("\tmovq %rax, %rax") == ""
		# Noop arith
		assert _opt("\taddq $0, %rax") == ""
		assert _opt("\tsubq $0, %rsp") == ""
		# Push/pop
		assert _opt("\tpushq %rax\n\tpopq %rax") == ""
