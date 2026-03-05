"""Tests for function prologue/epilogue and stack frame peephole optimizations."""

from compiler.peephole import PeepholeOptimizer


def make_asm(*lines: str) -> str:
	return "\n".join(lines)


class TestUnusedCalleeSavedElimination:
	"""Pattern 1: Eliminate push/pop pairs for callee-saved registers never used in body."""

	def setup_method(self) -> None:
		self.opt = PeepholeOptimizer()

	def test_unused_pushpop_rbx(self) -> None:
		"""pushq %rbx / popq %rbx removed when %rbx is never used."""
		asm = make_asm(
			".globl foo",
			".type foo, @function",
			"foo:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tpushq %rbx",
			"\tmovq $42, %rax",
			"\tpopq %rbx",
			"\tpopq %rbp",
			"\tret",
		)
		result = self.opt.optimize(asm)
		assert "\tpushq %rbx" not in result
		assert "\tpopq %rbx" not in result
		assert "\tmovq $42, %rax" in result

	def test_used_callee_saved_kept(self) -> None:
		"""pushq %rbx / popq %rbx kept when %rbx is used in body."""
		asm = make_asm(
			".globl foo",
			".type foo, @function",
			"foo:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tpushq %rbx",
			"\tmovq %rdi, %rbx",
			"\taddq %rbx, %rax",
			"\tpopq %rbx",
			"\tpopq %rbp",
			"\tret",
		)
		result = self.opt.optimize(asm)
		assert "\tpushq %rbx" in result
		assert "\tpopq %rbx" in result

	def test_unused_r12_pushpop(self) -> None:
		"""pushq %r12 / popq %r12 removed when %r12 is unused."""
		asm = make_asm(
			".globl bar",
			".type bar, @function",
			"bar:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tpushq %r12",
			"\tmovq $1, %rax",
			"\tpopq %r12",
			"\tpopq %rbp",
			"\tret",
		)
		result = self.opt.optimize(asm)
		assert "\tpushq %r12" not in result
		assert "\tpopq %r12" not in result

	def test_unused_movq_save_restore(self) -> None:
		"""movq %rbx, offset(%rbp) / movq offset(%rbp), %rbx removed when unused."""
		asm = make_asm(
			".globl baz",
			".type baz, @function",
			"baz:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tsubq $16, %rsp",
			"\tmovq %rbx, -8(%rbp)",
			"\tmovq $99, %rax",
			"\tmovq -8(%rbp), %rbx",
			"\tmovq %rbp, %rsp",
			"\tpopq %rbp",
			"\tret",
		)
		result = self.opt.optimize(asm)
		assert "movq %rbx, -8(%rbp)" not in result
		assert "movq -8(%rbp), %rbx" not in result

	def test_multiple_unused_callee_saved(self) -> None:
		"""Multiple unused callee-saved registers all get eliminated."""
		asm = make_asm(
			".globl multi",
			".type multi, @function",
			"multi:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tpushq %rbx",
			"\tpushq %r12",
			"\tpushq %r13",
			"\tmovq $0, %rax",
			"\tpopq %r13",
			"\tpopq %r12",
			"\tpopq %rbx",
			"\tpopq %rbp",
			"\tret",
		)
		result = self.opt.optimize(asm)
		assert "\tpushq %rbx" not in result
		assert "\tpushq %r12" not in result
		assert "\tpushq %r13" not in result
		assert "\tpopq %rbx" not in result
		assert "\tpopq %r12" not in result
		assert "\tpopq %r13" not in result


class TestAdjacentRspFolding:
	"""Pattern 2: Fold adjacent subq/addq %rsp operations."""

	def setup_method(self) -> None:
		self.opt = PeepholeOptimizer()

	def test_two_subq_rsp(self) -> None:
		"""subq $16, %rsp + subq $8, %rsp -> subq $24, %rsp."""
		asm = make_asm(
			".globl foo",
			".type foo, @function",
			"foo:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tsubq $16, %rsp",
			"\tsubq $8, %rsp",
			"\tmovq $0, %rax",
			"\tmovq %rbp, %rsp",
			"\tpopq %rbp",
			"\tret",
		)
		result = self.opt.optimize(asm)
		assert "\tsubq $24, %rsp" in result
		assert "\tsubq $16, %rsp" not in result
		assert "\tsubq $8, %rsp" not in result

	def test_subq_addq_rsp_cancel(self) -> None:
		"""subq $16, %rsp + addq $16, %rsp cancel out."""
		asm = make_asm(
			".globl foo",
			".type foo, @function",
			"foo:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tsubq $16, %rsp",
			"\taddq $16, %rsp",
			"\tmovq %rbp, %rsp",
			"\tpopq %rbp",
			"\tret",
		)
		result = self.opt.optimize(asm)
		assert "\tsubq $16, %rsp" not in result
		assert "\taddq $16, %rsp" not in result

	def test_addq_addq_rsp(self) -> None:
		"""addq $8, %rsp + addq $16, %rsp -> addq $24, %rsp."""
		asm = make_asm(
			".globl foo",
			".type foo, @function",
			"foo:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\taddq $8, %rsp",
			"\taddq $16, %rsp",
			"\tmovq %rbp, %rsp",
			"\tpopq %rbp",
			"\tret",
		)
		result = self.opt.optimize(asm)
		assert "\taddq $24, %rsp" in result

	def test_subq_partial_addq_rsp(self) -> None:
		"""subq $32, %rsp + addq $8, %rsp -> subq $24, %rsp."""
		asm = make_asm(
			".globl foo",
			".type foo, @function",
			"foo:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tsubq $32, %rsp",
			"\taddq $8, %rsp",
			"\tmovq $0, %rax",
			"\tmovq %rbp, %rsp",
			"\tpopq %rbp",
			"\tret",
		)
		result = self.opt.optimize(asm)
		assert "\tsubq $24, %rsp" in result


class TestLeafFramePointerElimination:
	"""Pattern 3: Remove frame pointer setup/teardown in leaf functions."""

	def setup_method(self) -> None:
		self.opt = PeepholeOptimizer()

	def test_leaf_function_frame_removed(self) -> None:
		"""Leaf function with no %rbp usage has frame pointer eliminated."""
		asm = make_asm(
			".globl leaf",
			".type leaf, @function",
			"leaf:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tmovq %rdi, %rax",
			"\tmovq %rbp, %rsp",
			"\tpopq %rbp",
			"\tret",
		)
		result = self.opt.optimize(asm)
		assert "\tpushq %rbp" not in result
		assert "\tmovq %rsp, %rbp" not in result
		assert "\tmovq %rbp, %rsp" not in result
		assert "\tpopq %rbp" not in result
		assert "\tmovq %rdi, %rax" in result
		assert "\tret" in result

	def test_non_leaf_function_frame_kept(self) -> None:
		"""Function with call instruction keeps frame pointer."""
		asm = make_asm(
			".globl nonleaf",
			".type nonleaf, @function",
			"nonleaf:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tcall other_func",
			"\tmovq %rbp, %rsp",
			"\tpopq %rbp",
			"\tret",
		)
		result = self.opt.optimize(asm)
		assert "\tpushq %rbp" in result
		assert "\tmovq %rsp, %rbp" in result

	def test_rbp_used_in_body_frame_kept(self) -> None:
		"""Leaf function that uses %rbp for stack access keeps frame pointer."""
		asm = make_asm(
			".globl uses_rbp",
			".type uses_rbp, @function",
			"uses_rbp:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tsubq $16, %rsp",
			"\tmovq %rdi, -8(%rbp)",
			"\tmovq -8(%rbp), %rax",
			"\tmovq %rbp, %rsp",
			"\tpopq %rbp",
			"\tret",
		)
		result = self.opt.optimize(asm)
		assert "\tpushq %rbp" in result
		assert "\tmovq %rsp, %rbp" in result

	def test_empty_leaf_function(self) -> None:
		"""Minimal leaf function with only ret."""
		asm = make_asm(
			".globl noop",
			".type noop, @function",
			"noop:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tmovq %rbp, %rsp",
			"\tpopq %rbp",
			"\tret",
		)
		result = self.opt.optimize(asm)
		assert "\tpushq %rbp" not in result
		assert "\tpopq %rbp" not in result
		assert "\tret" in result


class TestConsecutivePushCombine:
	"""Pattern 4: Combine consecutive pushq into subq + mov sequence."""

	def setup_method(self) -> None:
		self.opt = PeepholeOptimizer()

	def test_three_pushes_combined(self) -> None:
		"""Three consecutive pushq -> subq $24, %rsp + movq sequence."""
		asm = make_asm(
			".globl foo",
			".type foo, @function",
			"foo:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tpushq %rdi",
			"\tpushq %rsi",
			"\tpushq %rdx",
			"\tret",
		)
		result = self.opt.optimize(asm)
		assert "\tsubq $24, %rsp" in result
		assert "\tmovq %rdi, 16(%rsp)" in result
		assert "\tmovq %rsi, 8(%rsp)" in result
		assert "\tmovq %rdx, (%rsp)" in result
		# The original pushq instructions should be gone
		assert "\tpushq %rdi" not in result
		assert "\tpushq %rsi" not in result
		assert "\tpushq %rdx" not in result

	def test_two_pushes_not_combined(self) -> None:
		"""Two consecutive pushq are not combined (threshold is 3)."""
		asm = make_asm(
			".globl foo",
			".type foo, @function",
			"foo:",
			"\tpushq %rdi",
			"\tpushq %rsi",
			"\tret",
		)
		result = self.opt.optimize(asm)
		# With 2 pushes, they might get combined by push/pop or stay as-is
		# The combine pattern requires 3+, so these should remain as pushq
		# (unless other patterns like push-pop elimination fire)
		assert "\tsubq $16, %rsp" not in result

	def test_four_pushes_combined(self) -> None:
		"""Four consecutive pushq -> subq $32, %rsp + movq sequence."""
		asm = make_asm(
			".globl foo",
			".type foo, @function",
			"foo:",
			"\tpushq %rdi",
			"\tpushq %rsi",
			"\tpushq %rdx",
			"\tpushq %rcx",
			"\tret",
		)
		result = self.opt.optimize(asm)
		assert "\tsubq $32, %rsp" in result
		assert "\tmovq %rdi, 24(%rsp)" in result
		assert "\tmovq %rsi, 16(%rsp)" in result
		assert "\tmovq %rdx, 8(%rsp)" in result
		assert "\tmovq %rcx, (%rsp)" in result

	def test_non_consecutive_pushes_not_combined(self) -> None:
		"""Pushes separated by other instructions are not combined."""
		asm = make_asm(
			".globl foo",
			".type foo, @function",
			"foo:",
			"\tpushq %rdi",
			"\tmovq $1, %rax",
			"\tpushq %rsi",
			"\tpushq %rdx",
			"\tret",
		)
		result = self.opt.optimize(asm)
		# Only the last two are consecutive (< 3), so no combine
		assert "\tpushq %rdi" in result


class TestCombinedFrameOptimizations:
	"""Integration tests combining multiple frame optimization patterns."""

	def setup_method(self) -> None:
		self.opt = PeepholeOptimizer()

	def test_leaf_with_unused_callee_saved(self) -> None:
		"""Leaf function with unused callee-saved regs gets both optimizations."""
		asm = make_asm(
			".globl leaf_clean",
			".type leaf_clean, @function",
			"leaf_clean:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tpushq %rbx",
			"\tmovq %rdi, %rax",
			"\tpopq %rbx",
			"\tmovq %rbp, %rsp",
			"\tpopq %rbp",
			"\tret",
		)
		result = self.opt.optimize(asm)
		# Unused %rbx push/pop removed
		assert "\tpushq %rbx" not in result
		assert "\tpopq %rbx" not in result
		# Leaf frame pointer removed
		assert "\tpushq %rbp" not in result
		assert "\tpopq %rbp" not in result
		# Body preserved
		assert "\tmovq %rdi, %rax" in result
		assert "\tret" in result

	def test_realistic_function_prologue(self) -> None:
		"""Realistic function with stack frame and used callee-saved register."""
		asm = make_asm(
			".globl compute",
			".type compute, @function",
			"compute:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tsubq $16, %rsp",
			"\tmovq %rbx, -8(%rbp)",
			"\tmovq %rdi, %rbx",
			"\taddq %rsi, %rbx",
			"\tmovq %rbx, %rax",
			"\tmovq -8(%rbp), %rbx",
			"\tmovq %rbp, %rsp",
			"\tpopq %rbp",
			"\tret",
		)
		result = self.opt.optimize(asm)
		# %rbx is used, so save/restore should be kept
		assert "movq %rbx, -8(%rbp)" in result
		assert "movq -8(%rbp), %rbx" in result
		# Frame pointer used (%rbp referenced), so kept
		assert "\tpushq %rbp" in result
