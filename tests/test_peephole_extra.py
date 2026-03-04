"""Tests for extra peephole optimizer patterns: movl self-moves, movl $0 -> xorl,
push/pop elimination, and redundant consecutive loads."""

from compiler.peephole import PeepholeOptimizer


def _opt(asm: str) -> str:
	"""Helper: optimize assembly text."""
	return PeepholeOptimizer().optimize(asm)


# ---------------------------------------------------------------------------
# Pattern 1: Redundant same-register moves (movl/movw/movb)
# ---------------------------------------------------------------------------


class TestMovlSelfMoveElimination:
	def test_movl_self_move_eax(self) -> None:
		asm = "\tmovl %eax, %eax"
		assert _opt(asm) == ""

	def test_movl_self_move_ecx(self) -> None:
		asm = "\tmovl %ecx, %ecx"
		assert _opt(asm) == ""

	def test_movl_self_move_r8d(self) -> None:
		asm = "\tmovl %r8d, %r8d"
		assert _opt(asm) == ""

	def test_movl_different_regs_preserved(self) -> None:
		asm = "\tmovl %eax, %ecx"
		assert _opt(asm) == asm

	def test_movq_self_move_still_works(self) -> None:
		"""Existing movq self-move elimination still fires."""
		asm = "\tmovq %rax, %rax"
		assert _opt(asm) == ""

	def test_movl_self_move_in_context(self) -> None:
		lines = [
			"\tpushq %rbp",
			"\tmovl %eax, %eax",
			"\taddl %ecx, %eax",
			"\tmovl %ebx, %ebx",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tpushq %rbp",
			"\taddl %ecx, %eax",
			"\tret",
		])
		assert result == expected

	def test_movw_self_move(self) -> None:
		asm = "\tmovw %ax, %ax"
		assert _opt(asm) == ""

	def test_movb_self_move(self) -> None:
		asm = "\tmovb %al, %al"
		assert _opt(asm) == ""

	def test_movb_different_regs_preserved(self) -> None:
		asm = "\tmovb %al, %cl"
		assert _opt(asm) == asm


# ---------------------------------------------------------------------------
# Pattern 2: movl $0, %reg -> xorl %reg, %reg
# ---------------------------------------------------------------------------


class TestMovlZeroToXorl:
	def test_movl_zero_eax(self) -> None:
		asm = "\tmovl $0, %eax"
		assert _opt(asm) == "\txorl %eax, %eax"

	def test_movl_zero_ecx(self) -> None:
		asm = "\tmovl $0, %ecx"
		assert _opt(asm) == "\txorl %ecx, %ecx"

	def test_movl_zero_r8d(self) -> None:
		asm = "\tmovl $0, %r8d"
		assert _opt(asm) == "\txorl %r8d, %r8d"

	def test_movl_nonzero_preserved(self) -> None:
		asm = "\tmovl $42, %eax"
		assert _opt(asm) == asm

	def test_movl_negative_preserved(self) -> None:
		asm = "\tmovl $-1, %eax"
		assert _opt(asm) == asm

	def test_movq_zero_still_uses_xorq(self) -> None:
		"""Existing movq $0 -> xorq pattern still works."""
		asm = "\tmovq $0, %rax"
		assert _opt(asm) == "\txorq %rax, %rax"

	def test_movl_zero_in_context(self) -> None:
		lines = [
			"\tpushq %rbp",
			"\tmovl $0, %eax",
			"\taddl %ecx, %eax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tpushq %rbp",
			"\txorl %eax, %eax",
			"\taddl %ecx, %eax",
			"\tret",
		])
		assert result == expected

	def test_multiple_movl_zeros(self) -> None:
		lines = [
			"\tmovl $0, %eax",
			"\tmovl $0, %ecx",
			"\tmovl $0, %edx",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\txorl %eax, %eax",
			"\txorl %ecx, %ecx",
			"\txorl %edx, %edx",
		])
		assert result == expected


# ---------------------------------------------------------------------------
# Pattern 3: Push/pop elimination (same register, no intervening instructions)
# ---------------------------------------------------------------------------


class TestPushPopElimination:
	def test_push_pop_rbx(self) -> None:
		asm = "\tpushq %rbx\n\tpopq %rbx"
		assert _opt(asm) == ""

	def test_push_pop_r12(self) -> None:
		asm = "\tpushq %r12\n\tpopq %r12"
		assert _opt(asm) == ""

	def test_push_pop_rbp(self) -> None:
		asm = "\tpushq %rbp\n\tpopq %rbp"
		assert _opt(asm) == ""

	def test_push_pop_different_regs_preserved(self) -> None:
		"""pushq %rbx + popq %rcx is a move, not redundant."""
		asm = "\tpushq %rbx\n\tpopq %rcx"
		assert _opt(asm) == asm

	def test_push_pop_with_intervening_preserved(self) -> None:
		"""Push/pop with instructions between them are NOT eliminated."""
		lines = [
			"\tpushq %rbx",
			"\tmovq %rax, %rcx",
			"\tpopq %rbx",
		]
		result = _opt("\n".join(lines))
		assert "\tpushq %rbx" in result
		assert "\tpopq %rbx" in result

	def test_push_pop_in_context(self) -> None:
		lines = [
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tpushq %rbx",
			"\tpopq %rbx",
			"\tpopq %rbp",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tpopq %rbp",
			"\tret",
		])
		assert result == expected

	def test_multiple_push_pop_pairs(self) -> None:
		lines = [
			"\tpushq %rbx",
			"\tpopq %rbx",
			"\tpushq %r12",
			"\tpopq %r12",
		]
		result = _opt("\n".join(lines))
		assert result == ""


# ---------------------------------------------------------------------------
# Pattern 4: Redundant consecutive loads
# ---------------------------------------------------------------------------


class TestRedundantConsecutiveLoads:
	def test_identical_movq_loads(self) -> None:
		asm = "\tmovq -8(%rbp), %rax\n\tmovq -8(%rbp), %rax"
		assert _opt(asm) == "\tmovq -8(%rbp), %rax"

	def test_identical_movl_loads(self) -> None:
		asm = "\tmovl -4(%rbp), %eax\n\tmovl -4(%rbp), %eax"
		assert _opt(asm) == "\tmovl -4(%rbp), %eax"

	def test_different_offsets_preserved(self) -> None:
		asm = "\tmovq -8(%rbp), %rax\n\tmovq -16(%rbp), %rax"
		assert _opt(asm) == asm

	def test_different_dest_regs_preserved(self) -> None:
		asm = "\tmovq -8(%rbp), %rax\n\tmovq -8(%rbp), %rcx"
		assert _opt(asm) == asm

	def test_different_base_regs_preserved(self) -> None:
		asm = "\tmovq -8(%rbp), %rax\n\tmovq -8(%rsp), %rax"
		assert _opt(asm) == asm

	def test_redundant_load_in_context(self) -> None:
		lines = [
			"\tpushq %rbp",
			"\tmovq -8(%rbp), %rax",
			"\tmovq -8(%rbp), %rax",
			"\taddq %rcx, %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tpushq %rbp",
			"\tmovq -8(%rbp), %rax",
			"\taddq %rcx, %rax",
			"\tret",
		])
		assert result == expected

	def test_non_redundant_load_preserved(self) -> None:
		"""Different instructions are not redundant loads."""
		lines = [
			"\tmovq -8(%rbp), %rax",
			"\tmovq %rax, -16(%rbp)",
		]
		result = _opt("\n".join(lines))
		assert result == "\n".join(lines)

	def test_triple_redundant_loads(self) -> None:
		"""Three identical loads should collapse to one."""
		lines = [
			"\tmovq -8(%rbp), %rax",
			"\tmovq -8(%rbp), %rax",
			"\tmovq -8(%rbp), %rax",
		]
		result = _opt("\n".join(lines))
		assert result == "\tmovq -8(%rbp), %rax"

	def test_redundant_load_with_rsp_base(self) -> None:
		asm = "\tmovq 16(%rsp), %rdi\n\tmovq 16(%rsp), %rdi"
		assert _opt(asm) == "\tmovq 16(%rsp), %rdi"


# ---------------------------------------------------------------------------
# Combined: multiple new patterns interacting
# ---------------------------------------------------------------------------


class TestCombinedExtraPatterns:
	def test_movl_self_move_and_zero(self) -> None:
		"""Both movl self-move and movl $0 in same sequence."""
		lines = [
			"\tmovl %eax, %eax",
			"\tmovl $0, %ecx",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\txorl %ecx, %ecx",
			"\tret",
		])
		assert result == expected

	def test_push_pop_and_redundant_load(self) -> None:
		lines = [
			"\tpushq %rbx",
			"\tpopq %rbx",
			"\tmovq -8(%rbp), %rax",
			"\tmovq -8(%rbp), %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tmovq -8(%rbp), %rax",
			"\tret",
		])
		assert result == expected

	def test_all_four_patterns_together(self) -> None:
		"""Exercise all four new patterns in a single function."""
		lines = [
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			# Pattern 1: movl self-move
			"\tmovl %eax, %eax",
			# Pattern 2: movl $0 -> xorl
			"\tmovl $0, %ecx",
			# Pattern 3: push/pop same register
			"\tpushq %rbx",
			"\tpopq %rbx",
			# Pattern 4: redundant consecutive load
			"\tmovq -8(%rbp), %rax",
			"\tmovq -8(%rbp), %rax",
			"\tpopq %rbp",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\txorl %ecx, %ecx",
			"\tmovq -8(%rbp), %rax",
			"\tpopq %rbp",
			"\tret",
		])
		assert result == expected

	def test_existing_patterns_not_broken(self) -> None:
		"""Verify existing patterns still work alongside new ones."""
		lines = [
			# Existing: movq self-move
			"\tmovq %rax, %rax",
			# Existing: movq $0 -> xorq
			"\tmovq $0, %rbx",
			# Existing: addq $0 elimination
			"\taddq $0, %rcx",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\txorq %rbx, %rbx",
			"\tret",
		])
		assert result == expected
