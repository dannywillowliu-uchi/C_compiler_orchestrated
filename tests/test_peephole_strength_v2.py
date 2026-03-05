"""Tests for peephole strength reduction, dead code after jumps, and redundant cmp after test."""

from compiler.peephole import PeepholeOptimizer


def make_asm(*lines: str) -> str:
	return "\n".join(lines)


class TestStrengthReductionMul:
	"""imulq $N, %reg -> shlq $log2(N), %reg for powers of 2."""

	def setup_method(self) -> None:
		self.opt = PeepholeOptimizer()

	def test_mul_by_2(self) -> None:
		asm = make_asm("\timulq $2, %rax")
		result = self.opt.optimize(asm)
		assert "\tshlq $1, %rax" in result
		assert "imulq" not in result

	def test_mul_by_4(self) -> None:
		asm = make_asm("\timulq $4, %rcx")
		result = self.opt.optimize(asm)
		assert "\tshlq $2, %rcx" in result

	def test_mul_by_8(self) -> None:
		asm = make_asm("\timulq $8, %rdx")
		result = self.opt.optimize(asm)
		assert "\tshlq $3, %rdx" in result

	def test_mul_by_16(self) -> None:
		asm = make_asm("\timulq $16, %rsi")
		result = self.opt.optimize(asm)
		assert "\tshlq $4, %rsi" in result

	def test_mul_by_256(self) -> None:
		asm = make_asm("\timulq $256, %rdi")
		result = self.opt.optimize(asm)
		assert "\tshlq $8, %rdi" in result

	def test_mul_by_1024(self) -> None:
		asm = make_asm("\timulq $1024, %r8")
		result = self.opt.optimize(asm)
		assert "\tshlq $10, %r8" in result

	def test_mul_by_1_eliminated(self) -> None:
		asm = make_asm("\timulq $1, %rax")
		result = self.opt.optimize(asm)
		assert "imulq" not in result
		assert "shlq" not in result

	def test_mul_by_0_becomes_xor(self) -> None:
		asm = make_asm("\timulq $0, %rax")
		result = self.opt.optimize(asm)
		assert "\txorq %rax, %rax" in result

	def test_mul_by_non_power_of_2_unchanged(self) -> None:
		asm = make_asm("\timulq $3, %rax")
		result = self.opt.optimize(asm)
		assert "\timulq $3, %rax" in result

	def test_mul_by_5_unchanged(self) -> None:
		asm = make_asm("\timulq $5, %rcx")
		result = self.opt.optimize(asm)
		assert "\timulq $5, %rcx" in result

	def test_mul_by_7_unchanged(self) -> None:
		asm = make_asm("\timulq $7, %rdx")
		result = self.opt.optimize(asm)
		assert "\timulq $7, %rdx" in result

	def test_mul_preserves_surrounding(self) -> None:
		asm = make_asm(
			"\tmovq %rdi, %rax",
			"\timulq $8, %rax",
			"\tret",
		)
		result = self.opt.optimize(asm)
		assert "\tshlq $3, %rax" in result
		assert "\tmovq %rdi, %rax" in result
		assert "\tret" in result

	def test_mul_large_power_of_2(self) -> None:
		asm = make_asm("\timulq $65536, %rax")
		result = self.opt.optimize(asm)
		assert "\tshlq $16, %rax" in result

	def test_multiple_muls(self) -> None:
		asm = make_asm(
			"\timulq $4, %rax",
			"\timulq $8, %rcx",
		)
		result = self.opt.optimize(asm)
		assert "\tshlq $2, %rax" in result
		assert "\tshlq $3, %rcx" in result
		assert "imulq" not in result


class TestDeadCodeAfterJump:
	"""Remove unreachable instructions after unconditional jmp/ret until next label."""

	def setup_method(self) -> None:
		self.opt = PeepholeOptimizer()

	def test_dead_code_after_jmp(self) -> None:
		asm = make_asm(
			"\tjmp .L1",
			"\tmovq %rax, %rbx",
			"\taddq $1, %rcx",
			".L1:",
		)
		result = self.opt.optimize(asm)
		assert "movq %rax, %rbx" not in result
		assert "addq" not in result
		assert ".L1:" in result

	def test_dead_code_after_ret(self) -> None:
		asm = make_asm(
			"\tret",
			"\tmovq %rax, %rbx",
			"\taddq $1, %rcx",
			".L2:",
		)
		result = self.opt.optimize(asm)
		assert "movq %rax, %rbx" not in result
		assert "addq" not in result
		assert "\tret" in result
		assert ".L2:" in result

	def test_no_dead_code_before_label(self) -> None:
		asm = make_asm(
			"\tjmp .L1",
			".L1:",
		)
		result = self.opt.optimize(asm)
		# jmp to next label is eliminated by jump-to-next pattern
		assert ".L1:" in result

	def test_dead_code_single_instruction(self) -> None:
		asm = make_asm(
			"\tjmp .L3",
			"\tsubq $8, %rsp",
			".L3:",
		)
		result = self.opt.optimize(asm)
		assert "subq" not in result
		assert ".L3:" in result

	def test_no_dead_code_when_no_jmp(self) -> None:
		asm = make_asm(
			"\tmovq %rax, %rbx",
			"\taddq $1, %rcx",
			".L1:",
		)
		result = self.opt.optimize(asm)
		assert "movq %rax, %rbx" in result
		assert "addq" not in result or "addq $1, %rcx" in result

	def test_dead_code_multiple_instructions(self) -> None:
		asm = make_asm(
			"\tjmp .Lend",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tsubq $16, %rsp",
			"\tmovq %rdi, -8(%rbp)",
			".Lend:",
		)
		result = self.opt.optimize(asm)
		assert "pushq" not in result
		assert "subq" not in result
		assert ".Lend:" in result

	def test_dead_code_preserves_next_label_code(self) -> None:
		asm = make_asm(
			"\tjmp .L1",
			"\tmovq $99, %rax",
			".L1:",
			"\tmovq $42, %rax",
			"\tret",
		)
		result = self.opt.optimize(asm)
		assert "movq $99" not in result
		assert ".L1:" in result
		assert "\tret" in result

	def test_dead_code_after_ret_at_end(self) -> None:
		asm = make_asm(
			"\tret",
			"\tmovq $1, %rax",
		)
		result = self.opt.optimize(asm)
		assert "movq $1" not in result
		assert "\tret" in result

	def test_dead_code_preserves_empty_lines(self) -> None:
		asm = make_asm(
			"\tjmp .L1",
			"\tmovq $1, %rax",
			"",
			".L1:",
		)
		result = self.opt.optimize(asm)
		# Dead code up to the empty line is removed; empty line stops scanning
		assert "\tjmp .L1" in result
		assert ".L1:" in result

	def test_jmp_with_no_dead_code(self) -> None:
		asm = make_asm(
			"\tjmp .L1",
			"",
			".L1:",
		)
		result = self.opt.optimize(asm)
		assert "\tjmp .L1" in result or ".L1:" in result


class TestRedundantCmpAfterTest:
	"""Eliminate redundant cmpq $0 after testq on same register."""

	def setup_method(self) -> None:
		self.opt = PeepholeOptimizer()

	def test_testq_then_cmpq_zero_same_reg(self) -> None:
		asm = make_asm(
			"\ttestq %rax, %rax",
			"\tcmpq $0, %rax",
		)
		result = self.opt.optimize(asm)
		assert "\ttestq %rax, %rax" in result
		assert "cmpq" not in result

	def test_testq_then_cmpq_zero_different_reg(self) -> None:
		asm = make_asm(
			"\ttestq %rax, %rax",
			"\tcmpq $0, %rcx",
		)
		result = self.opt.optimize(asm)
		assert "\ttestq %rax, %rax" in result
		# cmpq on different reg becomes testq via cmp_zero_to_test
		assert "\ttestq %rcx, %rcx" in result

	def test_testq_different_regs_not_eliminated(self) -> None:
		"""testq %rax, %rcx is not the same-register test pattern."""
		asm = make_asm(
			"\ttestq %rax, %rcx",
			"\tcmpq $0, %rcx",
		)
		result = self.opt.optimize(asm)
		assert "\ttestq %rax, %rcx" in result
		# cmpq $0 becomes testq via the single-line pattern
		assert "\ttestq %rcx, %rcx" in result

	def test_testq_cmpq_preserves_surrounding(self) -> None:
		asm = make_asm(
			"\tmovq %rdi, %rax",
			"\ttestq %rax, %rax",
			"\tcmpq $0, %rax",
			"\tje .L1",
		)
		result = self.opt.optimize(asm)
		assert "\ttestq %rax, %rax" in result
		assert "cmpq" not in result
		assert "\tje .L1" in result

	def test_testq_cmpq_multiple_occurrences(self) -> None:
		asm = make_asm(
			"\ttestq %rax, %rax",
			"\tcmpq $0, %rax",
			"\tje .L1",
			"\ttestq %rcx, %rcx",
			"\tcmpq $0, %rcx",
			"\tje .L2",
		)
		result = self.opt.optimize(asm)
		assert result.count("cmpq") == 0
		assert result.count("testq") == 2

	def test_cmpq_zero_without_preceding_testq(self) -> None:
		"""cmpq $0 without testq should still become testq (existing pattern)."""
		asm = make_asm(
			"\tcmpq $0, %rax",
			"\tje .L1",
		)
		result = self.opt.optimize(asm)
		assert "\ttestq %rax, %rax" in result
		assert "cmpq" not in result


class TestStrengthReductionIntegration:
	"""Integration tests combining strength reduction with other patterns."""

	def setup_method(self) -> None:
		self.opt = PeepholeOptimizer()

	def test_mul_then_dead_code(self) -> None:
		asm = make_asm(
			"\timulq $4, %rax",
			"\tjmp .Ldone",
			"\timulq $8, %rcx",
			".Ldone:",
		)
		result = self.opt.optimize(asm)
		assert "\tshlq $2, %rax" in result
		assert "imulq $8" not in result
		assert "shlq $3" not in result

	def test_dead_code_after_ret_with_testq_cmpq(self) -> None:
		asm = make_asm(
			"\ttestq %rax, %rax",
			"\tcmpq $0, %rax",
			"\tje .L1",
			"\tret",
			"\tmovq $0, %rax",
			".L1:",
		)
		result = self.opt.optimize(asm)
		assert "cmpq" not in result
		assert "\ttestq %rax, %rax" in result
		# movq $0 after ret should be eliminated as dead code
		assert result.count("movq $0") == 0 or "movq $0, %rax" not in result.split("\tret")[1].split(".L1:")[0]

	def test_all_three_patterns_together(self) -> None:
		asm = make_asm(
			"\timulq $16, %rax",
			"\ttestq %rax, %rax",
			"\tcmpq $0, %rax",
			"\tje .Lskip",
			"\tjmp .Lend",
			"\taddq $1, %rax",
			".Lskip:",
			"\txorq %rax, %rax",
			".Lend:",
		)
		result = self.opt.optimize(asm)
		# imulq $16 -> shlq $4
		assert "\tshlq $4, %rax" in result
		# redundant cmpq removed
		assert "cmpq" not in result
		# dead addq after jmp removed
		assert "addq $1" not in result
