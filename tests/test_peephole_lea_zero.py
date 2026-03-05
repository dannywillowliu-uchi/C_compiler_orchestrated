"""Tests for peephole patterns: xorl zeroing, lea address folding, imulq $1 removal."""

from compiler.peephole import PeepholeOptimizer

_opt = PeepholeOptimizer().optimize


# --- Pattern 1: movq $0, %reg -> xorl %e_reg, %e_reg ---


class TestMovqZeroToXorl:
	"""movq $0, %reg should become xorl with 32-bit register (shorter encoding, breaks deps)."""

	def test_movq_zero_rax(self) -> None:
		assert _opt("\tmovq $0, %rax") == "\txorl %eax, %eax"

	def test_movq_zero_rbx(self) -> None:
		assert _opt("\tmovq $0, %rbx") == "\txorl %ebx, %ebx"

	def test_movq_zero_rcx(self) -> None:
		assert _opt("\tmovq $0, %rcx") == "\txorl %ecx, %ecx"

	def test_movq_zero_rdx(self) -> None:
		assert _opt("\tmovq $0, %rdx") == "\txorl %edx, %edx"

	def test_movq_zero_rsi(self) -> None:
		assert _opt("\tmovq $0, %rsi") == "\txorl %esi, %esi"

	def test_movq_zero_rdi(self) -> None:
		assert _opt("\tmovq $0, %rdi") == "\txorl %edi, %edi"

	def test_movq_zero_r8(self) -> None:
		assert _opt("\tmovq $0, %r8") == "\txorl %r8d, %r8d"

	def test_movq_zero_r9(self) -> None:
		assert _opt("\tmovq $0, %r9") == "\txorl %r9d, %r9d"

	def test_movq_zero_r15(self) -> None:
		assert _opt("\tmovq $0, %r15") == "\txorl %r15d, %r15d"

	def test_movq_nonzero_unchanged(self) -> None:
		"""movq $1, %rax should NOT be converted."""
		assert _opt("\tmovq $1, %rax") == "\tmovq $1, %rax"

	def test_movq_zero_in_context(self) -> None:
		"""xorl zeroing works alongside other instructions."""
		asm = "\n".join([
			"\tmovq %rdi, -8(%rbp)",
			"\tmovq $0, %rax",
			"\tret",
		])
		result = _opt(asm)
		assert "\txorl %eax, %eax" in result
		assert "\tmovq %rdi, -8(%rbp)" in result

	def test_movq_zero_plus_cmpq_zero(self) -> None:
		"""movq $0 + cmpq $0 same reg folds to xorl + (testq eliminated as redundant)."""
		asm = "\tmovq $0, %rax\n\tcmpq $0, %rax"
		result = _opt(asm)
		assert "\txorl %eax, %eax" in result

	def test_multiple_zero_moves(self) -> None:
		"""Multiple movq $0 to different regs all become xorl."""
		asm = "\n".join([
			"\tmovq $0, %rax",
			"\tmovq $0, %rbx",
			"\tmovq $0, %rcx",
		])
		result = _opt(asm)
		assert "\txorl %eax, %eax" in result
		assert "\txorl %ebx, %ebx" in result
		assert "\txorl %ecx, %ecx" in result


# --- Pattern 2: movq offset(%rbp), %r1 + addq $imm, %r1 -> leaq offset+imm(%rbp), %r1 ---


class TestLoadAddToLea:
	"""Fold load + add-immediate into single leaq."""

	def test_basic_fold(self) -> None:
		asm = "\tmovq -8(%rbp), %rax\n\taddq $16, %rax"
		assert _opt(asm) == "\tleaq 8(%rbp), %rax"

	def test_negative_offset_positive_imm(self) -> None:
		asm = "\tmovq -24(%rbp), %rcx\n\taddq $8, %rcx"
		assert _opt(asm) == "\tleaq -16(%rbp), %rcx"

	def test_negative_offset_negative_imm(self) -> None:
		asm = "\tmovq -8(%rbp), %rdx\n\taddq $-4, %rdx"
		assert _opt(asm) == "\tleaq -12(%rbp), %rdx"

	def test_zero_offset(self) -> None:
		asm = "\tmovq 0(%rbp), %rax\n\taddq $10, %rax"
		assert _opt(asm) == "\tleaq 10(%rbp), %rax"

	def test_large_positive_offset(self) -> None:
		asm = "\tmovq -128(%rbp), %rsi\n\taddq $64, %rsi"
		assert _opt(asm) == "\tleaq -64(%rbp), %rsi"

	def test_different_register_no_fold(self) -> None:
		"""addq targets different register than load destination -> no fold."""
		asm = "\tmovq -8(%rbp), %rax\n\taddq $16, %rbx"
		result = _opt(asm)
		assert "\tmovq -8(%rbp), %rax" in result
		assert "\taddq $16, %rbx" in result

	def test_non_rbp_base_no_fold(self) -> None:
		"""Load from non-%rbp base should not fold (pattern only matches %rbp)."""
		asm = "\tmovq -8(%rsp), %rax\n\taddq $16, %rax"
		result = _opt(asm)
		assert "leaq" not in result

	def test_fold_with_surrounding_code(self) -> None:
		"""Fold works when surrounded by other instructions."""
		asm = "\n".join([
			"\tpushq %rbp",
			"\tmovq -16(%rbp), %rax",
			"\taddq $4, %rax",
			"\tret",
		])
		result = _opt(asm)
		assert "\tleaq -12(%rbp), %rax" in result
		assert "\tpushq %rbp" in result

	def test_result_zero_offset(self) -> None:
		"""When offset + imm = 0, leaq 0(%rbp) further folds to movq %rbp."""
		asm = "\tmovq -8(%rbp), %rax\n\taddq $8, %rax"
		assert _opt(asm) == "\tmovq %rbp, %rax"

	def test_r_numbered_register(self) -> None:
		"""Fold works with r8-r15 registers."""
		asm = "\tmovq -32(%rbp), %r12\n\taddq $16, %r12"
		assert _opt(asm) == "\tleaq -16(%rbp), %r12"


# --- Pattern 3: imulq $1, %reg -> removed (identity multiply) ---


class TestImulqOneRemoval:
	"""imulq $1, %reg is an identity operation and should be eliminated."""

	def test_imulq_one_rax(self) -> None:
		assert _opt("\timulq $1, %rax") == ""

	def test_imulq_one_rbx(self) -> None:
		assert _opt("\timulq $1, %rbx") == ""

	def test_imulq_one_r10(self) -> None:
		assert _opt("\timulq $1, %r10") == ""

	def test_imulq_one_in_context(self) -> None:
		"""imulq $1 is removed, surrounding instructions preserved."""
		asm = "\n".join([
			"\tmovq -8(%rbp), %rax",
			"\timulq $1, %rax",
			"\tret",
		])
		result = _opt(asm)
		assert "imulq" not in result
		assert "\tmovq -8(%rbp), %rax" in result
		assert "\tret" in result

	def test_imulq_two_unchanged(self) -> None:
		"""imulq $2, %rax should become shlq $1, not removed."""
		result = _opt("\timulq $2, %rax")
		assert result == "\tshlq $1, %rax"

	def test_imulq_zero_to_xorl(self) -> None:
		"""imulq $0, %rax should become xorl %eax, %eax."""
		assert _opt("\timulq $0, %rax") == "\txorl %eax, %eax"

	def test_imulq_nonpower_unchanged(self) -> None:
		"""imulq with non-power-of-2 should remain unchanged."""
		assert _opt("\timulq $3, %rax") == "\timulq $3, %rax"


# --- Combined pattern tests ---


class TestCombinedPatterns:
	"""Test interactions between the three patterns."""

	def test_zero_and_lea_fold_together(self) -> None:
		"""Both xorl zeroing and lea folding in same function."""
		asm = "\n".join([
			"\tmovq $0, %rbx",
			"\tmovq -8(%rbp), %rax",
			"\taddq $4, %rax",
			"\tret",
		])
		result = _opt(asm)
		assert "\txorl %ebx, %ebx" in result
		assert "\tleaq -4(%rbp), %rax" in result

	def test_imulq_one_and_zero(self) -> None:
		"""imulq $1 removal and movq $0 -> xorl in same function."""
		asm = "\n".join([
			"\timulq $1, %rax",
			"\tmovq $0, %rbx",
			"\tret",
		])
		result = _opt(asm)
		assert "imulq" not in result
		assert "\txorl %ebx, %ebx" in result

	def test_all_three_patterns(self) -> None:
		"""All three patterns fire in a single function."""
		asm = "\n".join([
			"\tmovq $0, %rcx",
			"\tmovq -16(%rbp), %rax",
			"\taddq $8, %rax",
			"\timulq $1, %rdx",
			"\tret",
		])
		result = _opt(asm)
		assert "\txorl %ecx, %ecx" in result
		assert "\tleaq -8(%rbp), %rax" in result
		assert "imulq" not in result
