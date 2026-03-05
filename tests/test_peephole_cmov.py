"""Tests for conditional move (cmov) peephole optimization patterns."""

from compiler.peephole import PeepholeOptimizer


def _opt(asm: str) -> str:
	"""Helper: optimize assembly text."""
	return PeepholeOptimizer().optimize(asm)


# ---------------------------------------------------------------------------
# Simple branch-over-mov -> cmov (3-line pattern)
# ---------------------------------------------------------------------------


class TestSimpleBranchOverMov:
	"""jCC .Lskip + movq %rA, %rDst + .Lskip: -> cmovINV_CCq %rA, %rDst + .Lskip:"""

	def test_je_branch_over_mov(self) -> None:
		asm = "\tje .L1\n\tmovq %rax, %rbx\n.L1:"
		result = _opt(asm)
		assert "\tcmovneq %rax, %rbx" in result
		assert "\tje" not in result

	def test_jne_branch_over_mov(self) -> None:
		asm = "\tjne .L1\n\tmovq %rcx, %rdx\n.L1:"
		result = _opt(asm)
		assert "\tcmoveq %rcx, %rdx" in result
		assert "\tjne" not in result

	def test_jg_branch_over_mov(self) -> None:
		asm = "\tjg .L1\n\tmovq %rsi, %rdi\n.L1:"
		result = _opt(asm)
		assert "\tcmovleq %rsi, %rdi" in result

	def test_jge_branch_over_mov(self) -> None:
		asm = "\tjge .L1\n\tmovq %r8, %r9\n.L1:"
		result = _opt(asm)
		assert "\tcmovlq %r8, %r9" in result

	def test_jl_branch_over_mov(self) -> None:
		asm = "\tjl .L1\n\tmovq %rax, %rcx\n.L1:"
		result = _opt(asm)
		assert "\tcmovgeq %rax, %rcx" in result

	def test_jle_branch_over_mov(self) -> None:
		asm = "\tjle .L1\n\tmovq %rax, %rcx\n.L1:"
		result = _opt(asm)
		assert "\tcmovgq %rax, %rcx" in result

	def test_ja_branch_over_mov(self) -> None:
		asm = "\tja .L1\n\tmovq %rax, %rbx\n.L1:"
		result = _opt(asm)
		assert "\tcmovbeq %rax, %rbx" in result

	def test_jae_branch_over_mov(self) -> None:
		asm = "\tjae .L1\n\tmovq %rax, %rbx\n.L1:"
		result = _opt(asm)
		assert "\tcmovbq %rax, %rbx" in result

	def test_jb_branch_over_mov(self) -> None:
		asm = "\tjb .L1\n\tmovq %rax, %rbx\n.L1:"
		result = _opt(asm)
		assert "\tcmovaeq %rax, %rbx" in result

	def test_jbe_branch_over_mov(self) -> None:
		asm = "\tjbe .L1\n\tmovq %rax, %rbx\n.L1:"
		result = _opt(asm)
		assert "\tcmovaq %rax, %rbx" in result

	def test_js_branch_over_mov(self) -> None:
		asm = "\tjs .L1\n\tmovq %rax, %rbx\n.L1:"
		result = _opt(asm)
		assert "\tcmovnsq %rax, %rbx" in result

	def test_jns_branch_over_mov(self) -> None:
		asm = "\tjns .L1\n\tmovq %rax, %rbx\n.L1:"
		result = _opt(asm)
		assert "\tcmovsq %rax, %rbx" in result

	def test_label_preserved(self) -> None:
		asm = "\tje .L1\n\tmovq %rax, %rbx\n.L1:"
		result = _opt(asm)
		assert ".L1:" in result

	def test_memory_source(self) -> None:
		asm = "\tje .L1\n\tmovq -8(%rbp), %rax\n.L1:"
		result = _opt(asm)
		assert "\tcmovneq -8(%rbp), %rax" in result

	def test_immediate_source_not_converted(self) -> None:
		"""cmov cannot use immediate source - pattern should NOT apply."""
		asm = "\tje .L1\n\tmovq $42, %rax\n.L1:"
		result = _opt(asm)
		assert "\tje .L1" in result or "\tcmov" not in result

	def test_label_mismatch_not_converted(self) -> None:
		"""jCC targets a different label than the one after movq."""
		asm = "\tje .L2\n\tmovq %rax, %rbx\n.L1:"
		result = _opt(asm)
		assert "\tje .L2" in result

	def test_non_mov_instruction_not_converted(self) -> None:
		"""Only movq between jCC and label should trigger conversion."""
		asm = "\tje .L1\n\taddq %rax, %rbx\n.L1:"
		result = _opt(asm)
		assert "\tje .L1" in result


# ---------------------------------------------------------------------------
# Full cmpq/testq + branch-over-mov integration
# ---------------------------------------------------------------------------


class TestCmpBranchCmov:
	"""Full pattern: cmpq/testq + jCC + movq + label."""

	def test_cmpq_je_mov(self) -> None:
		asm = "\tcmpq $0, %rax\n\tje .L1\n\tmovq %rbx, %rcx\n.L1:"
		result = _opt(asm)
		# cmpq $0 -> testq, and je + movq -> cmovne
		assert "testq %rax, %rax" in result or "cmpq" in result
		assert "\tcmovneq %rbx, %rcx" in result

	def test_testq_jne_mov(self) -> None:
		asm = "\ttestq %rdi, %rdi\n\tjne .L1\n\tmovq %rsi, %rdx\n.L1:"
		result = _opt(asm)
		assert "\ttestq %rdi, %rdi" in result
		assert "\tcmoveq %rsi, %rdx" in result

	def test_cmpq_jl_mov(self) -> None:
		asm = "\tcmpq $10, %rax\n\tjl .L1\n\tmovq %rbx, %rcx\n.L1:"
		result = _opt(asm)
		assert "\tcmpq $10, %rax" in result
		assert "\tcmovgeq %rbx, %rcx" in result

	def test_cmpq_jg_mov(self) -> None:
		asm = "\tcmpq %rdx, %rax\n\tjg .L1\n\tmovq %rsi, %rdi\n.L1:"
		result = _opt(asm)
		assert "\tcmpq %rdx, %rax" in result
		assert "\tcmovleq %rsi, %rdi" in result


# ---------------------------------------------------------------------------
# Diamond pattern (existing _try_branch_to_cmov) - 6-line pattern
# ---------------------------------------------------------------------------


class TestDiamondBranchToCmov:
	"""jCC + movq + jmp + label + movq + label -> movq + cmov + label."""

	def test_basic_diamond(self) -> None:
		asm = (
			"\tje .Lfalse\n"
			"\tmovq %rax, %rcx\n"
			"\tjmp .Lend\n"
			".Lfalse:\n"
			"\tmovq %rbx, %rcx\n"
			".Lend:"
		)
		result = _opt(asm)
		assert "\tcmovneq %rax, %rcx" in result
		assert "\tmovq %rbx, %rcx" in result
		assert "\tje" not in result
		assert "\tjmp" not in result

	def test_diamond_jne(self) -> None:
		asm = (
			"\tjne .Lfalse\n"
			"\tmovq %rsi, %rdi\n"
			"\tjmp .Lend\n"
			".Lfalse:\n"
			"\tmovq %rdx, %rdi\n"
			".Lend:"
		)
		result = _opt(asm)
		assert "\tcmoveq %rsi, %rdi" in result
		assert "\tmovq %rdx, %rdi" in result

	def test_diamond_memory_false_branch(self) -> None:
		asm = (
			"\tje .Lfalse\n"
			"\tmovq %rax, %rcx\n"
			"\tjmp .Lend\n"
			".Lfalse:\n"
			"\tmovq -8(%rbp), %rcx\n"
			".Lend:"
		)
		result = _opt(asm)
		assert "\tcmovneq %rax, %rcx" in result
		assert "\tmovq -8(%rbp), %rcx" in result

	def test_diamond_immediate_true_not_converted(self) -> None:
		"""cmov requires register/memory source - immediate in true branch blocks conversion."""
		asm = (
			"\tje .Lfalse\n"
			"\tmovq $1, %rcx\n"
			"\tjmp .Lend\n"
			".Lfalse:\n"
			"\tmovq %rbx, %rcx\n"
			".Lend:"
		)
		result = _opt(asm)
		# Should NOT be converted since true source is $1
		assert "\tje .Lfalse" in result

	def test_diamond_different_dst_not_converted(self) -> None:
		"""True and false branches write to different registers - no conversion."""
		asm = (
			"\tje .Lfalse\n"
			"\tmovq %rax, %rcx\n"
			"\tjmp .Lend\n"
			".Lfalse:\n"
			"\tmovq %rbx, %rdx\n"
			".Lend:"
		)
		result = _opt(asm)
		assert "\tje .Lfalse" in result


# ---------------------------------------------------------------------------
# Edge cases and non-matching patterns
# ---------------------------------------------------------------------------


class TestCmovEdgeCases:
	def test_no_conversion_with_multiple_instructions(self) -> None:
		"""Multiple instructions between jCC and label - should not convert."""
		asm = "\tje .L1\n\tmovq %rax, %rbx\n\taddq $1, %rbx\n.L1:"
		result = _opt(asm)
		# The simple pattern shouldn't match because addq is between movq and label
		assert "\tje .L1" in result

	def test_surrounding_code_preserved(self) -> None:
		"""Code before and after the pattern is preserved."""
		asm = (
			"\tpushq %rbp\n"
			"\tcmpq $5, %rax\n"
			"\tjle .L1\n"
			"\tmovq %rbx, %rcx\n"
			".L1:\n"
			"\tpopq %rbp\n"
			"\tret"
		)
		result = _opt(asm)
		assert "\tpushq %rbp" in result
		assert "\tcmovgq %rbx, %rcx" in result
		assert "\tpopq %rbp" in result
		assert "\tret" in result

	def test_multiple_cmov_conversions(self) -> None:
		"""Multiple independent branch-over-mov patterns in sequence."""
		asm = (
			"\tje .L1\n"
			"\tmovq %rax, %rbx\n"
			".L1:\n"
			"\tjne .L2\n"
			"\tmovq %rcx, %rdx\n"
			".L2:"
		)
		result = _opt(asm)
		assert "\tcmovneq %rax, %rbx" in result
		assert "\tcmoveq %rcx, %rdx" in result
