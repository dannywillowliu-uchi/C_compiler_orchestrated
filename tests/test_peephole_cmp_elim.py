"""Tests for redundant comparison elimination and conditional move optimization."""

from compiler.peephole import PeepholeOptimizer


def _opt(asm: str) -> str:
	"""Helper: optimize assembly text."""
	return PeepholeOptimizer().optimize(asm)


# ---------------------------------------------------------------------------
# Redundant cmpq $0 after flag-setting arithmetic (existing pattern)
# ---------------------------------------------------------------------------


class TestRedundantCmpAfterArith:
	def test_addq_then_cmpq_zero(self) -> None:
		"""addq sets ZF/SF, so cmpq $0 on same reg is redundant."""
		lines = ["\taddq %rcx, %rax", "\tcmpq $0, %rax"]
		result = _opt("\n".join(lines))
		assert result == "\taddq %rcx, %rax"

	def test_subq_then_cmpq_zero(self) -> None:
		lines = ["\tsubq $1, %rax", "\tcmpq $0, %rax"]
		result = _opt("\n".join(lines))
		assert result == "\tsubq $1, %rax"

	def test_andq_then_cmpq_zero(self) -> None:
		lines = ["\tandq %rbx, %rax", "\tcmpq $0, %rax"]
		result = _opt("\n".join(lines))
		assert result == "\tandq %rbx, %rax"

	def test_orq_then_cmpq_zero(self) -> None:
		lines = ["\torq %rdx, %rax", "\tcmpq $0, %rax"]
		result = _opt("\n".join(lines))
		assert result == "\torq %rdx, %rax"

	def test_xorq_then_cmpq_zero(self) -> None:
		lines = ["\txorq $0xFF, %rax", "\tcmpq $0, %rax"]
		result = _opt("\n".join(lines))
		assert result == "\txorq $0xFF, %rax"

	def test_different_register_not_eliminated(self) -> None:
		"""cmpq $0 on a different register must NOT be eliminated."""
		lines = ["\taddq %rcx, %rax", "\tcmpq $0, %rbx"]
		result = _opt("\n".join(lines))
		# cmpq $0 gets converted to testq, but NOT eliminated
		assert "\ttestq %rbx, %rbx" in result
		assert "\taddq %rcx, %rax" in result

	def test_addq_imm_then_cmpq_zero(self) -> None:
		lines = ["\taddq $42, %rdi", "\tcmpq $0, %rdi"]
		result = _opt("\n".join(lines))
		assert result == "\taddq $42, %rdi"


# ---------------------------------------------------------------------------
# Redundant testq after flag-setting arithmetic (NEW pattern)
# ---------------------------------------------------------------------------


class TestRedundantTestAfterArith:
	def test_addq_then_testq(self) -> None:
		"""addq sets ZF/SF, so testq %reg, %reg on same reg is redundant."""
		lines = ["\taddq %rcx, %rax", "\ttestq %rax, %rax"]
		result = _opt("\n".join(lines))
		assert result == "\taddq %rcx, %rax"

	def test_subq_then_testq(self) -> None:
		lines = ["\tsubq $1, %rax", "\ttestq %rax, %rax"]
		result = _opt("\n".join(lines))
		assert result == "\tsubq $1, %rax"

	def test_andq_then_testq(self) -> None:
		lines = ["\tandq $0xFF, %rcx", "\ttestq %rcx, %rcx"]
		result = _opt("\n".join(lines))
		assert result == "\tandq $0xFF, %rcx"

	def test_orq_then_testq(self) -> None:
		lines = ["\torq %rdx, %rax", "\ttestq %rax, %rax"]
		result = _opt("\n".join(lines))
		assert result == "\torq %rdx, %rax"

	def test_xorq_then_testq(self) -> None:
		lines = ["\txorq %rbx, %rax", "\ttestq %rax, %rax"]
		result = _opt("\n".join(lines))
		assert result == "\txorq %rbx, %rax"

	def test_different_register_preserved(self) -> None:
		"""testq on a different register must NOT be eliminated."""
		lines = ["\taddq %rcx, %rax", "\ttestq %rbx, %rbx"]
		result = _opt("\n".join(lines))
		assert "\taddq %rcx, %rax" in result
		assert "\ttestq %rbx, %rbx" in result


# ---------------------------------------------------------------------------
# Redundant cmp/test after negq, incq, decq (NEW patterns)
# ---------------------------------------------------------------------------


class TestRedundantCmpAfterUnary:
	def test_negq_then_cmpq_zero(self) -> None:
		"""negq sets ZF/SF, so cmpq $0 is redundant."""
		lines = ["\tnegq %rax", "\tcmpq $0, %rax"]
		result = _opt("\n".join(lines))
		assert result == "\tnegq %rax"

	def test_negq_then_testq(self) -> None:
		lines = ["\tnegq %rax", "\ttestq %rax, %rax"]
		result = _opt("\n".join(lines))
		assert result == "\tnegq %rax"

	def test_incq_then_cmpq_zero(self) -> None:
		lines = ["\tincq %rcx", "\tcmpq $0, %rcx"]
		result = _opt("\n".join(lines))
		assert result == "\tincq %rcx"

	def test_incq_then_testq(self) -> None:
		lines = ["\tincq %rcx", "\ttestq %rcx, %rcx"]
		result = _opt("\n".join(lines))
		assert result == "\tincq %rcx"

	def test_decq_then_cmpq_zero(self) -> None:
		lines = ["\tdecq %rdx", "\tcmpq $0, %rdx"]
		result = _opt("\n".join(lines))
		assert result == "\tdecq %rdx"

	def test_decq_then_testq(self) -> None:
		lines = ["\tdecq %rdx", "\ttestq %rdx, %rdx"]
		result = _opt("\n".join(lines))
		assert result == "\tdecq %rdx"

	def test_negq_different_register_preserved(self) -> None:
		lines = ["\tnegq %rax", "\tcmpq $0, %rbx"]
		result = _opt("\n".join(lines))
		assert "\tnegq %rax" in result
		assert "\ttestq %rbx, %rbx" in result


# ---------------------------------------------------------------------------
# Redundant cmp/test after shift (NEW patterns)
# ---------------------------------------------------------------------------


class TestRedundantCmpAfterShift:
	def test_shlq_then_cmpq_zero(self) -> None:
		lines = ["\tshlq $2, %rax", "\tcmpq $0, %rax"]
		result = _opt("\n".join(lines))
		assert result == "\tshlq $2, %rax"

	def test_shrq_then_cmpq_zero(self) -> None:
		lines = ["\tshrq $1, %rbx", "\tcmpq $0, %rbx"]
		result = _opt("\n".join(lines))
		assert result == "\tshrq $1, %rbx"

	def test_sarq_then_testq(self) -> None:
		lines = ["\tsarq $3, %rcx", "\ttestq %rcx, %rcx"]
		result = _opt("\n".join(lines))
		assert result == "\tsarq $3, %rcx"

	def test_shlq_then_testq(self) -> None:
		lines = ["\tshlq $4, %rdx", "\ttestq %rdx, %rdx"]
		result = _opt("\n".join(lines))
		assert result == "\tshlq $4, %rdx"

	def test_shift_different_register_preserved(self) -> None:
		lines = ["\tshlq $2, %rax", "\ttestq %rbx, %rbx"]
		result = _opt("\n".join(lines))
		assert "\tshlq $2, %rax" in result
		assert "\ttestq %rbx, %rbx" in result


# ---------------------------------------------------------------------------
# Redundant testq after testq / cmpq $0 after testq (existing, but more tests)
# ---------------------------------------------------------------------------


class TestRedundantCmpAfterTest:
	def test_testq_then_cmpq_zero(self) -> None:
		"""testq already sets flags; cmpq $0 on same reg is redundant."""
		lines = ["\ttestq %rax, %rax", "\tcmpq $0, %rax"]
		result = _opt("\n".join(lines))
		assert result == "\ttestq %rax, %rax"

	def test_testq_then_cmpq_zero_different_reg(self) -> None:
		"""Different registers: both instructions kept."""
		lines = ["\ttestq %rax, %rax", "\tcmpq $0, %rbx"]
		result = _opt("\n".join(lines))
		assert "\ttestq %rax, %rax" in result
		assert "\ttestq %rbx, %rbx" in result


# ---------------------------------------------------------------------------
# Conditional branch to cmov optimization (NEW)
# ---------------------------------------------------------------------------


class TestBranchToCmov:
	def test_basic_je_to_cmov(self) -> None:
		"""Simple je branch pattern converted to cmov."""
		lines = [
			"\tje .L1",
			"\tmovq %rax, %rdx",
			"\tjmp .L2",
			".L1:",
			"\tmovq %rbx, %rdx",
			".L2:",
		]
		result = _opt("\n".join(lines))
		assert "cmovneq" in result
		assert ".L2:" in result
		# Should not contain conditional jump or jmp
		assert "\tje " not in result
		assert "\tjmp " not in result

	def test_jne_to_cmov(self) -> None:
		lines = [
			"\tjne .Lfalse",
			"\tmovq %rax, %rcx",
			"\tjmp .Lend",
			".Lfalse:",
			"\tmovq %rbx, %rcx",
			".Lend:",
		]
		result = _opt("\n".join(lines))
		assert "cmoveq" in result
		assert "\tjne " not in result

	def test_jg_to_cmov(self) -> None:
		lines = [
			"\tjg .Lf",
			"\tmovq %r8, %rax",
			"\tjmp .Le",
			".Lf:",
			"\tmovq %r9, %rax",
			".Le:",
		]
		result = _opt("\n".join(lines))
		assert "cmovleq" in result

	def test_jl_to_cmov(self) -> None:
		lines = [
			"\tjl .Lf",
			"\tmovq %rdi, %rax",
			"\tjmp .Le",
			".Lf:",
			"\tmovq %rsi, %rax",
			".Le:",
		]
		result = _opt("\n".join(lines))
		assert "cmovgeq" in result

	def test_cmov_preserves_semantics(self) -> None:
		"""The false path value is loaded first, then conditionally overwritten."""
		lines = [
			"\tje .L1",
			"\tmovq %rax, %rdx",
			"\tjmp .L2",
			".L1:",
			"\tmovq %rbx, %rdx",
			".L2:",
		]
		result = _opt("\n".join(lines))
		result_lines = result.split("\n")
		# First instruction should load the false-branch value
		assert result_lines[0] == "\tmovq %rbx, %rdx"
		# Second instruction should be the conditional move
		assert result_lines[1] == "\tcmovneq %rax, %rdx"
		# End label preserved
		assert result_lines[2] == ".L2:"

	def test_cmov_not_applied_with_immediate_true(self) -> None:
		"""cmov cannot take an immediate source for the true branch."""
		lines = [
			"\tje .L1",
			"\tmovq $42, %rax",
			"\tjmp .L2",
			".L1:",
			"\tmovq %rbx, %rax",
			".L2:",
		]
		result = _opt("\n".join(lines))
		# Pattern should NOT fire because true-branch source is immediate
		assert "\tje .L1" in result or "\tjmp" in result or "cmov" not in result

	def test_cmov_with_memory_false_branch(self) -> None:
		"""False branch can use memory operand."""
		lines = [
			"\tje .L1",
			"\tmovq %rax, %rdx",
			"\tjmp .L2",
			".L1:",
			"\tmovq -8(%rbp), %rdx",
			".L2:",
		]
		result = _opt("\n".join(lines))
		assert "cmovneq" in result
		assert "-8(%rbp)" in result

	def test_cmov_different_destinations_not_applied(self) -> None:
		"""If true and false branches write to different regs, don't optimize."""
		lines = [
			"\tje .L1",
			"\tmovq %rax, %rdx",
			"\tjmp .L2",
			".L1:",
			"\tmovq %rbx, %rcx",
			".L2:",
		]
		result = _opt("\n".join(lines))
		assert "cmov" not in result

	def test_cmov_wrong_label_not_applied(self) -> None:
		"""If false label doesn't match jcc target, don't optimize."""
		lines = [
			"\tje .L1",
			"\tmovq %rax, %rdx",
			"\tjmp .L2",
			".L3:",
			"\tmovq %rbx, %rdx",
			".L2:",
		]
		result = _opt("\n".join(lines))
		assert "cmov" not in result

	def test_cmov_no_end_label_not_applied(self) -> None:
		"""If end label doesn't follow the false mov, don't optimize."""
		lines = [
			"\tje .L1",
			"\tmovq %rax, %rdx",
			"\tjmp .L2",
			".L1:",
			"\tmovq %rbx, %rdx",
			"\taddq $1, %rcx",
		]
		result = _opt("\n".join(lines))
		assert "cmov" not in result


# ---------------------------------------------------------------------------
# Interaction: comparison elimination chains
# ---------------------------------------------------------------------------


class TestCmpElimChains:
	def test_arith_plus_cmpq_zero_in_longer_sequence(self) -> None:
		"""Redundant cmpq elimination works within a larger instruction sequence."""
		lines = [
			"\tmovq -8(%rbp), %rax",
			"\taddq $1, %rax",
			"\tcmpq $0, %rax",
			"\tje .L1",
		]
		result = _opt("\n".join(lines))
		assert "\tcmpq" not in result
		assert "\ttestq" not in result
		assert "\taddq $1, %rax" in result
		assert "\tje .L1" in result

	def test_shift_plus_testq_in_context(self) -> None:
		lines = [
			"\tmovq -16(%rbp), %rcx",
			"\tshlq $2, %rcx",
			"\ttestq %rcx, %rcx",
			"\tjne .L5",
		]
		result = _opt("\n".join(lines))
		assert "\ttestq" not in result
		assert "\tshlq $2, %rcx" in result
		assert "\tjne .L5" in result

	def test_negq_plus_cmpq_zero_with_branch(self) -> None:
		lines = [
			"\tnegq %rax",
			"\tcmpq $0, %rax",
			"\tjl .Lneg",
		]
		result = _opt("\n".join(lines))
		assert "\tcmpq" not in result
		assert "\ttestq" not in result
		assert "\tnegq %rax" in result
		assert "\tjl .Lneg" in result

	def test_incq_plus_testq_with_branch(self) -> None:
		lines = [
			"\tincq %rdi",
			"\ttestq %rdi, %rdi",
			"\tje .Lzero",
		]
		result = _opt("\n".join(lines))
		assert "\ttestq" not in result
		assert "\tincq %rdi" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestCmpElimEdgeCases:
	def test_cmpq_nonzero_not_eliminated(self) -> None:
		"""cmpq with non-zero immediate should NOT be eliminated."""
		lines = ["\taddq %rcx, %rax", "\tcmpq $1, %rax"]
		result = _opt("\n".join(lines))
		assert "\tcmpq $1, %rax" in result

	def test_testq_different_regs_not_eliminated(self) -> None:
		"""testq with two different registers is not a zero-test and should be kept."""
		lines = ["\taddq %rcx, %rax", "\ttestq %rbx, %rax"]
		result = _opt("\n".join(lines))
		assert "\ttestq %rbx, %rax" in result

	def test_label_between_arith_and_cmp_blocks_elimination(self) -> None:
		"""A label between arithmetic and cmp prevents elimination."""
		lines = [
			"\taddq $1, %rax",
			".Lmid:",
			"\tcmpq $0, %rax",
		]
		result = _opt("\n".join(lines))
		# The cmpq $0 gets converted to testq by single-instruction pattern,
		# but should NOT be eliminated since there's a label between
		assert "\taddq $1, %rax" in result
		assert ".Lmid:" in result
		assert "\ttestq %rax, %rax" in result

	def test_multiple_redundant_cmp_eliminations(self) -> None:
		"""Multiple independent redundant comparisons in one function."""
		lines = [
			"\taddq $1, %rax",
			"\tcmpq $0, %rax",
			"\tje .L1",
			"\tsubq $2, %rbx",
			"\ttestq %rbx, %rbx",
			"\tjne .L2",
		]
		result = _opt("\n".join(lines))
		assert "\tcmpq" not in result
		assert "\ttestq" not in result
		assert "\taddq $1, %rax" in result
		assert "\tsubq $2, %rbx" in result
