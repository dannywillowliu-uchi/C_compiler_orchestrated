"""Tests for peephole optimizer: mov chains, self-moves, push/pop, dead loads."""

from compiler.peephole import PeepholeOptimizer


def _opt(asm: str) -> str:
	"""Helper: optimize assembly text."""
	return PeepholeOptimizer().optimize(asm)


# ---------------------------------------------------------------------------
# Self-move elimination
# ---------------------------------------------------------------------------


class TestSelfMoveElimination:
	def test_movq_self_move(self) -> None:
		asm = "\tmovq %rax, %rax"
		assert _opt(asm) == ""

	def test_movl_self_move(self) -> None:
		asm = "\tmovl %eax, %eax"
		assert _opt(asm) == ""

	def test_movw_self_move(self) -> None:
		asm = "\tmovw %ax, %ax"
		assert _opt(asm) == ""

	def test_movb_self_move(self) -> None:
		asm = "\tmovb %al, %al"
		assert _opt(asm) == ""

	def test_self_move_preserves_other_instructions(self) -> None:
		asm = "\tmovq $42, %rax\n\tmovq %rax, %rax\n\tret"
		result = _opt(asm)
		assert "\tmovq %rax, %rax" not in result
		assert "\tmovq $42, %rax" in result
		assert "\tret" in result

	def test_different_regs_not_eliminated(self) -> None:
		asm = "\tmovq %rax, %rbx"
		assert _opt(asm) == asm

	def test_multiple_self_moves(self) -> None:
		asm = "\tmovq %rax, %rax\n\tmovq %rbx, %rbx"
		assert _opt(asm) == ""


# ---------------------------------------------------------------------------
# Mov chain elimination
# ---------------------------------------------------------------------------


class TestMovChainElimination:
	def test_basic_chain(self) -> None:
		"""movq %rax, %rbx + movq %rbx, %rcx -> movq %rax, %rcx (when %rbx dead)."""
		asm = "\tmovq %rax, %rbx\n\tmovq %rbx, %rcx\n\tret"
		result = _opt(asm)
		assert "\tmovq %rax, %rcx" in result
		assert "\tmovq %rax, %rbx" not in result
		assert "\tmovq %rbx, %rcx" not in result

	def test_chain_not_optimized_when_middle_reg_used(self) -> None:
		"""Chain should not fire if the middle register is used afterward."""
		asm = "\tmovq %rax, %rbx\n\tmovq %rbx, %rcx\n\taddq %rbx, %rdx"
		result = _opt(asm)
		assert "\tmovq %rax, %rbx" in result
		assert "\tmovq %rbx, %rcx" in result

	def test_chain_with_rax_at_ret(self) -> None:
		"""If middle reg is %rax and next is ret, %rax is live so chain should not fire."""
		asm = "\tmovq %rcx, %rax\n\tmovq %rax, %rbx\n\tret"
		result = _opt(asm)
		# %rax is live at ret, so chain should NOT be collapsed
		assert "\tmovq %rcx, %rax" in result

	def test_chain_middle_reg_dead_before_overwrite(self) -> None:
		"""Middle reg is overwritten before being read -> chain fires."""
		asm = "\tmovq %rax, %rbx\n\tmovq %rbx, %rcx\n\tmovq $0, %rbx\n\tret"
		result = _opt(asm)
		assert "\tmovq %rax, %rcx" in result


# ---------------------------------------------------------------------------
# Push/pop pair elimination
# ---------------------------------------------------------------------------


class TestPushPopElimination:
	def test_same_register(self) -> None:
		"""pushq %rax + popq %rax -> eliminated."""
		asm = "\tpushq %rax\n\tpopq %rax"
		assert _opt(asm) == ""

	def test_different_register_not_eliminated(self) -> None:
		"""pushq %rax + popq %rbx should NOT be eliminated."""
		asm = "\tpushq %rax\n\tpopq %rbx"
		assert _opt(asm) == asm

	def test_push_pop_with_surrounding_code(self) -> None:
		asm = "\tmovq $1, %rax\n\tpushq %rbx\n\tpopq %rbx\n\tret"
		result = _opt(asm)
		assert "\tpushq" not in result
		assert "\tpopq" not in result
		assert "\tmovq $1, %rax" in result
		assert "\tret" in result


# ---------------------------------------------------------------------------
# Dead load elimination (load followed by overwrite)
# ---------------------------------------------------------------------------


class TestDeadLoadElimination:
	def test_load_then_overwrite_movq(self) -> None:
		"""Load to %rax then movq $42, %rax -> just movq $42, %rax."""
		asm = "\tmovq -8(%rbp), %rax\n\tmovq $42, %rax"
		result = _opt(asm)
		assert result == "\tmovq $42, %rax"

	def test_load_then_overwrite_leaq(self) -> None:
		"""Load to %rax then leaq overwrites %rax -> just leaq."""
		asm = "\tmovq -8(%rbp), %rax\n\tleaq 8(%rbx), %rax"
		result = _opt(asm)
		assert result == "\tleaq 8(%rbx), %rax"

	def test_load_then_overwrite_xorq(self) -> None:
		"""Load to %rax then xorq %rax, %rax -> just xorq."""
		asm = "\tmovq -8(%rbp), %rcx\n\txorq %rbx, %rcx"
		# xorq reads %rbx and %rcx, so %rcx is read - should NOT eliminate
		result = _opt(asm)
		assert "\tmovq -8(%rbp), %rcx" in result

	def test_load_not_eliminated_when_read(self) -> None:
		"""Load to %rax then instruction using %rax should NOT eliminate."""
		asm = "\tmovq -8(%rbp), %rax\n\tmovq %rax, %rbx"
		result = _opt(asm)
		assert "\tmovq -8(%rbp), %rax" in result

	def test_load_then_overwrite_different_reg(self) -> None:
		"""Load to %rax, overwrite of %rbx should NOT eliminate the load."""
		asm = "\tmovq -8(%rbp), %rax\n\tmovq $42, %rbx"
		result = _opt(asm)
		assert "\tmovq -8(%rbp), %rax" in result

	def test_load_then_overwrite_with_self_read(self) -> None:
		"""Load to %rax then leaq using %rax should NOT eliminate."""
		asm = "\tmovq -8(%rbp), %rax\n\tleaq 8(%rax), %rax"
		result = _opt(asm)
		assert "\tmovq -8(%rbp), %rax" in result

	def test_multiple_dead_loads(self) -> None:
		"""Multiple dead loads in sequence."""
		asm = "\tmovq -8(%rbp), %rax\n\tmovq -16(%rbp), %rax\n\tmovq $0, %rax"
		result = _opt(asm)
		# First load dead (overwritten by second), second dead (overwritten by third)
		# After optimization, the $0 gets turned to xorq
		assert "movq -8(%rbp)" not in result
		assert "movq -16(%rbp)" not in result


# ---------------------------------------------------------------------------
# Reverse move elimination
# ---------------------------------------------------------------------------


class TestReverseMoveElimination:
	def test_basic_reverse(self) -> None:
		"""movq %rax, %rbx + movq %rbx, %rax -> just movq %rax, %rbx."""
		asm = "\tmovq %rax, %rbx\n\tmovq %rbx, %rax"
		result = _opt(asm)
		assert result == "\tmovq %rax, %rbx"

	def test_non_reverse_kept(self) -> None:
		"""movq %rax, %rbx + movq %rcx, %rax should be kept."""
		asm = "\tmovq %rax, %rbx\n\tmovq %rcx, %rax"
		result = _opt(asm)
		assert "\tmovq %rax, %rbx" in result
		assert "\tmovq %rcx, %rax" in result


# ---------------------------------------------------------------------------
# Combined / interaction tests
# ---------------------------------------------------------------------------


class TestCombinedPatterns:
	def test_self_move_and_chain(self) -> None:
		"""Self-move followed by a useful instruction."""
		asm = "\tmovq %rax, %rax\n\tmovq %rax, %rbx"
		result = _opt(asm)
		assert "\tmovq %rax, %rax" not in result
		assert "\tmovq %rax, %rbx" in result

	def test_chain_then_self_move(self) -> None:
		"""Chain producing a self-move should collapse fully."""
		asm = "\tmovq %rax, %rbx\n\tmovq %rbx, %rax\n\tret"
		result = _opt(asm)
		# This is a reverse-move pair, so second line eliminated
		assert "\tmovq %rbx, %rax" not in result

	def test_dead_load_then_push_pop(self) -> None:
		"""Dead load and push/pop in sequence."""
		asm = "\tmovq -8(%rbp), %rax\n\tmovq $1, %rax\n\tpushq %rbx\n\tpopq %rbx\n\tret"
		result = _opt(asm)
		assert "movq -8(%rbp)" not in result
		assert "\tpushq" not in result
		assert "\tpopq" not in result
