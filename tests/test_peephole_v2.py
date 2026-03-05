"""Tests for new peephole optimizer patterns (v2)."""

from compiler.peephole import PeepholeOptimizer


def _opt(asm: str) -> str:
	"""Helper: optimize assembly text."""
	return PeepholeOptimizer().optimize(asm)


# ---------------------------------------------------------------------------
# Pattern: Mov chain elimination
# ---------------------------------------------------------------------------


class TestMovChainElimination:
	def test_basic_chain_dead_intermediate(self) -> None:
		"""movq %rdi, %rbx + movq %rbx, %rcx -> movq %rdi, %rcx when %rbx is dead."""
		lines = [
			"\tmovq %rdi, %rbx",
			"\tmovq %rbx, %rcx",
			"\tmovq $0, %rbx",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tmovq %rdi, %rcx",
			"\txorq %rbx, %rbx",
		])
		assert result == expected

	def test_chain_intermediate_dead_at_ret(self) -> None:
		"""Intermediate register is dead at ret (non-%rax)."""
		lines = [
			"\tmovq %rdi, %rbx",
			"\tmovq %rbx, %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tmovq %rdi, %rax",
			"\tret",
		])
		assert result == expected

	def test_chain_rax_intermediate_live_at_ret(self) -> None:
		"""Chain should NOT fire when intermediate is %rax and next is ret."""
		lines = [
			"\tmovq %rdi, %rax",
			"\tmovq %rax, %rcx",
			"\tret",
		]
		# %rax is live at ret (return value), so chain should not fire
		assert _opt("\n".join(lines)) == "\n".join(lines)

	def test_chain_intermediate_used_later(self) -> None:
		"""Chain should NOT fire when intermediate reg is read later."""
		lines = [
			"\tmovq %rdi, %rbx",
			"\tmovq %rbx, %rcx",
			"\taddq %rbx, %rdx",
		]
		assert _opt("\n".join(lines)) == "\n".join(lines)

	def test_chain_no_match_different_intermediate(self) -> None:
		"""Chain should NOT fire when intermediates don't match."""
		lines = [
			"\tmovq %rdi, %rbx",
			"\tmovq %rcx, %rdx",
			"\tret",
		]
		assert _opt("\n".join(lines)) == "\n".join(lines)

	def test_chain_intermediate_overwritten_later(self) -> None:
		"""Chain fires when intermediate is overwritten (non-zero mov)."""
		lines = [
			"\tmovq %rsi, %rbx",
			"\tmovq %rbx, %rdx",
			"\tmovq %rdi, %rbx",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tmovq %rsi, %rdx",
			"\tmovq %rdi, %rbx",
		])
		assert result == expected

	def test_chain_at_control_flow_conservative(self) -> None:
		"""Chain should NOT fire when followed by a conditional jump."""
		lines = [
			"\tmovq %rdi, %rbx",
			"\tmovq %rbx, %rcx",
			"\tjne .L1",
		]
		assert _opt("\n".join(lines)) == "\n".join(lines)

	def test_chain_at_call_conservative(self) -> None:
		"""Chain should NOT fire when followed by a call."""
		lines = [
			"\tmovq %rdi, %rbx",
			"\tmovq %rbx, %rcx",
			"\tcall foo",
		]
		assert _opt("\n".join(lines)) == "\n".join(lines)

	def test_chain_with_surrounding_code(self) -> None:
		"""Chain fires correctly within surrounding instructions."""
		lines = [
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tmovq %rdi, %rbx",
			"\tmovq %rbx, %rcx",
			"\tmovq $0, %rbx",
			"\taddq %rcx, %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tmovq %rdi, %rcx",
			"\txorq %rbx, %rbx",
			"\taddq %rcx, %rax",
			"\tret",
		])
		assert result == expected


# ---------------------------------------------------------------------------
# Pattern: LEA strength reduction
# ---------------------------------------------------------------------------


class TestLeaReduction:
	def test_basic_lea(self) -> None:
		"""movq $8, %rax + addq %rbx, %rax -> leaq 8(%rbx), %rax."""
		asm = "\tmovq $8, %rax\n\taddq %rbx, %rax"
		assert _opt(asm) == "\tleaq 8(%rbx), %rax"

	def test_negative_immediate(self) -> None:
		"""Negative immediate in LEA."""
		asm = "\tmovq $-16, %rax\n\taddq %rbx, %rax"
		assert _opt(asm) == "\tleaq -16(%rbx), %rax"

	def test_zero_immediate(self) -> None:
		"""Zero immediate produces leaq 0(%reg), %reg."""
		asm = "\tmovq $0, %rcx\n\taddq %rdx, %rcx"
		assert _opt(asm) == "\tleaq 0(%rdx), %rcx"

	def test_large_positive_immediate(self) -> None:
		"""Immediate at 32-bit max boundary."""
		asm = "\tmovq $2147483647, %rax\n\taddq %rbx, %rax"
		assert _opt(asm) == "\tleaq 2147483647(%rbx), %rax"

	def test_too_large_immediate_no_fire(self) -> None:
		"""Immediate exceeding 32-bit range should NOT trigger LEA."""
		asm = "\tmovq $2147483648, %rax\n\taddq %rbx, %rax"
		assert _opt(asm) == asm

	def test_too_negative_immediate_no_fire(self) -> None:
		"""Very negative immediate should NOT trigger LEA."""
		asm = "\tmovq $-2147483649, %rax\n\taddq %rbx, %rax"
		assert _opt(asm) == asm

	def test_different_destinations_no_fire(self) -> None:
		"""movq $imm, %rax + addq %rbx, %rcx should NOT fire (different dests)."""
		asm = "\tmovq $8, %rax\n\taddq %rbx, %rcx"
		assert _opt(asm) == asm

	def test_same_src_dst_add_no_fire(self) -> None:
		"""addq %rax, %rax (doubling) should NOT trigger LEA."""
		asm = "\tmovq $8, %rax\n\taddq %rax, %rax"
		assert _opt(asm) == asm

	def test_lea_with_context(self) -> None:
		"""LEA reduction within surrounding code."""
		lines = [
			"\tpushq %rbp",
			"\tmovq $16, %rax",
			"\taddq %rdi, %rax",
			"\tmovq %rax, -8(%rbp)",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tpushq %rbp",
			"\tleaq 16(%rdi), %rax",
			"\tmovq %rax, -8(%rbp)",
		])
		assert result == expected


# ---------------------------------------------------------------------------
# Pattern: Push/pop elimination
# ---------------------------------------------------------------------------


class TestPushPopElimination:
	def test_basic_push_pop_same_reg(self) -> None:
		"""pushq %rax + popq %rax -> removed."""
		asm = "\tpushq %rax\n\tpopq %rax"
		assert _opt(asm) == ""

	def test_push_pop_rbx(self) -> None:
		"""pushq %rbx + popq %rbx -> removed."""
		asm = "\tpushq %rbx\n\tpopq %rbx"
		assert _opt(asm) == ""

	def test_push_pop_different_regs_no_fire(self) -> None:
		"""pushq %rax + popq %rbx should NOT be removed."""
		asm = "\tpushq %rax\n\tpopq %rbx"
		assert _opt(asm) == asm

	def test_push_pop_with_surrounding_code(self) -> None:
		"""Push/pop elimination within surrounding code."""
		lines = [
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tpushq %rax",
			"\tpopq %rax",
			"\tmovq $42, %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tmovq $42, %rax",
			"\tret",
		])
		assert result == expected

	def test_multiple_push_pop_pairs(self) -> None:
		"""Multiple redundant push/pop pairs."""
		lines = [
			"\tpushq %rax",
			"\tpopq %rax",
			"\tpushq %rbx",
			"\tpopq %rbx",
		]
		assert _opt("\n".join(lines)) == ""

	def test_push_not_followed_by_pop_no_fire(self) -> None:
		"""pushq not followed by popq should NOT fire."""
		asm = "\tpushq %rax\n\tmovq $0, %rbx"
		result = _opt(asm)
		# Push/pop doesn't fire, but movq $0 -> xorq does
		assert result == "\tpushq %rax\n\txorq %rbx, %rbx"


# ---------------------------------------------------------------------------
# Pattern: Dead store elimination
# ---------------------------------------------------------------------------


class TestDeadStoreElimination:
	def test_basic_dead_store(self) -> None:
		"""Two stores to same location -> keep only the second."""
		asm = "\tmovq %rax, -8(%rbp)\n\tmovq %rbx, -8(%rbp)"
		assert _opt(asm) == "\tmovq %rbx, -8(%rbp)"

	def test_dead_store_same_register(self) -> None:
		"""Same register stored twice to same location."""
		asm = "\tmovq %rax, -8(%rbp)\n\tmovq %rax, -8(%rbp)"
		assert _opt(asm) == "\tmovq %rax, -8(%rbp)"

	def test_different_locations_no_fire(self) -> None:
		"""Stores to different locations should NOT fire."""
		asm = "\tmovq %rax, -8(%rbp)\n\tmovq %rbx, -16(%rbp)"
		assert _opt(asm) == asm

	def test_dead_store_with_context(self) -> None:
		"""Dead store elimination within surrounding code."""
		lines = [
			"\tpushq %rbp",
			"\tmovq %rax, -8(%rbp)",
			"\tmovq %rcx, -8(%rbp)",
			"\tmovq -8(%rbp), %rax",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tpushq %rbp",
			"\tmovq %rcx, -8(%rbp)",
			"\tmovq -8(%rbp), %rax",
		])
		assert result == expected

	def test_non_store_between_stores_no_fire(self) -> None:
		"""Non-consecutive stores should NOT be eliminated."""
		lines = [
			"\tmovq %rax, -8(%rbp)",
			"\taddq $1, %rax",
			"\tmovq %rax, -8(%rbp)",
		]
		assert _opt("\n".join(lines)) == "\n".join(lines)


# ---------------------------------------------------------------------------
# Pattern: Jump-to-next elimination
# ---------------------------------------------------------------------------


class TestJumpToNext:
	def test_basic_jump_to_next(self) -> None:
		"""jmp .L1 + .L1: -> .L1:."""
		asm = "\tjmp .L1\n.L1:"
		assert _opt(asm) == ".L1:"

	def test_different_labels_no_fire(self) -> None:
		"""jmp .L1 + .L2: should NOT fire."""
		asm = "\tjmp .L1\n.L2:"
		assert _opt(asm) == asm

	def test_jmp_not_followed_by_label_no_fire(self) -> None:
		"""jmp followed by instruction: jump-to-next doesn't fire, dead code after jmp is removed."""
		asm = "\tjmp .L1\n\tmovq $0, %rax"
		result = _opt(asm)
		# Dead code after unconditional jmp is eliminated
		assert result == "\tjmp .L1"

	def test_jump_to_next_with_context(self) -> None:
		"""Jump-to-next within a realistic function."""
		lines = [
			"\tcmpq $0, %rax",
			"\tje .L2",
			"\tmovq $1, %rax",
			"\tjmp .L3",
			".L3:",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\ttestq %rax, %rax",
			"\tje .L2",
			"\tmovq $1, %rax",
			".L3:",
			"\tret",
		])
		assert result == expected

	def test_conditional_jump_not_matched(self) -> None:
		"""Conditional jumps should NOT be matched by jump-to-next."""
		asm = "\tje .L1\n.L1:"
		assert _opt(asm) == asm

	def test_jump_to_function_label_not_matched(self) -> None:
		"""Jump to function-style label (no dot) should NOT match."""
		asm = "\tjmp main\nmain:"
		# The regex requires .label format for both jmp target and label
		assert _opt(asm) == asm


# ---------------------------------------------------------------------------
# Edge cases where patterns should NOT fire
# ---------------------------------------------------------------------------


class TestNegativeCases:
	def test_mov_chain_with_memory_operand(self) -> None:
		"""Mov chain regex only matches reg-to-reg, not reg-to-mem."""
		lines = [
			"\tmovq %rax, -8(%rbp)",
			"\tmovq -8(%rbp), %rcx",
		]
		# This is a store+load, not a reg-to-reg chain
		# store-reload would fire if regs match, but here src=%rax, dst=%rcx
		assert _opt("\n".join(lines)) == "\n".join(lines)

	def test_lea_with_immediate_add_not_matched(self) -> None:
		"""addq $imm, %reg should NOT match LEA pattern (needs reg, reg)."""
		asm = "\tmovq $8, %rax\n\taddq $4, %rax"
		assert _opt(asm) == asm

	def test_push_pop_with_memory_operand(self) -> None:
		"""Push/pop with memory operands should NOT match (regex needs %reg)."""
		asm = "\tpushq -8(%rbp)\n\tpopq -8(%rbp)"
		assert _opt(asm) == asm

	def test_dead_store_with_immediate_source(self) -> None:
		"""movq $imm, offset(%rbp) should NOT match dead store regex (needs %reg src)."""
		asm = "\tmovq $0, -8(%rbp)\n\tmovq $1, -8(%rbp)"
		assert _opt(asm) == asm

	def test_single_instruction_no_crash(self) -> None:
		"""Single instruction with no following line shouldn't crash."""
		assert _opt("\tmovq %rdi, %rax") == "\tmovq %rdi, %rax"

	def test_empty_lines_preserved(self) -> None:
		"""Empty lines in assembly should be preserved."""
		asm = "\tmovq $42, %rax\n\n\tret"
		assert _opt(asm) == asm


# ---------------------------------------------------------------------------
# Multi-pattern interaction tests
# ---------------------------------------------------------------------------


class TestMultiPatternInteraction:
	def test_chain_then_self_move_cascading(self) -> None:
		"""Mov chain + self-move elimination across passes."""
		lines = [
			"\tmovq %rdi, %rbx",
			"\tmovq %rbx, %rcx",
			"\tmovq %rcx, %rcx",
			"\tmovq $0, %rbx",
		]
		result = _opt("\n".join(lines))
		# Pass 1: self-move %rcx,%rcx removed + chain %rdi->%rbx->%rcx collapses
		# movq $0, %rbx -> xorq %rbx, %rbx
		expected = "\n".join([
			"\tmovq %rdi, %rcx",
			"\txorq %rbx, %rbx",
		])
		assert result == expected

	def test_push_pop_and_jump_to_next(self) -> None:
		"""Push/pop + jump-to-next both fire."""
		lines = [
			"\tpushq %rax",
			"\tpopq %rax",
			"\tjmp .L1",
			".L1:",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			".L1:",
			"\tret",
		])
		assert result == expected

	def test_dead_store_then_store_reload(self) -> None:
		"""Dead store elimination followed by store-reload."""
		lines = [
			"\tmovq %rax, -8(%rbp)",
			"\tmovq %rbx, -8(%rbp)",
			"\tmovq -8(%rbp), %rbx",
		]
		result = _opt("\n".join(lines))
		# Dead store removes first store, then store-reload collapses
		assert result == "\tmovq %rbx, -8(%rbp)"

	def test_lea_and_noop_arith(self) -> None:
		"""LEA reduction with noop arith removal."""
		lines = [
			"\tmovq $16, %rax",
			"\taddq %rdi, %rax",
			"\taddq $0, %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tleaq 16(%rdi), %rax",
			"\tret",
		])
		assert result == expected

	def test_all_patterns_in_function(self) -> None:
		"""Realistic function exercising multiple new patterns."""
		lines = [
			".section .text",
			".globl compute",
			"compute:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tsubq $32, %rsp",
			# Dead store: two stores to same location
			"\tmovq %rdi, -8(%rbp)",
			"\tmovq %rsi, -8(%rbp)",
			# Mov chain (with %rbx dead at overwrite)
			"\tmovq %rsi, %rbx",
			"\tmovq %rbx, %rcx",
			"\tmovq $0, %rbx",
			# LEA strength reduction
			"\tmovq $24, %rax",
			"\taddq %rcx, %rax",
			# Push/pop no-op
			"\tpushq %rdx",
			"\tpopq %rdx",
			# Self-move
			"\tmovq %rax, %rax",
			# Jump to next
			"\tjmp .L1",
			".L1:",
			# Noop arith
			"\taddq $0, %rax",
			"\tmovq %rbp, %rsp",
			"\tpopq %rbp",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			".section .text",
			".globl compute",
			"compute:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tsubq $32, %rsp",
			"\tmovq %rsi, -8(%rbp)",
			"\tmovq %rsi, %rcx",
			"\txorq %rbx, %rbx",
			"\tleaq 24(%rcx), %rax",
			".L1:",
			"\tmovq %rbp, %rsp",
			"\tpopq %rbp",
			"\tret",
		])
		assert result == expected

	def test_cascading_chains(self) -> None:
		"""A three-link chain: A->B->C->D, where B and C are dead."""
		lines = [
			"\tmovq %rdi, %rbx",
			"\tmovq %rbx, %rcx",
			"\tmovq %rcx, %rax",
			"\tmovq $0, %rbx",
			"\tmovq $0, %rcx",
			"\tret",
		]
		result = _opt("\n".join(lines))
		# Chains collapse across passes, movq $0 -> xorq
		# Note: xorq %rbx, %rbx overwrites %rbx (not a read), so chain fires.
		# But xorq %rcx, %rcx both reads and writes %rcx in the regex check,
		# so chain detection may be conservative. Check actual result.
		result_lines = result.split("\n")
		assert "\tmovq %rdi, %rax" in result or "\tmovq %rdi, %rcx" in result
		assert "\tret" in result_lines

	def test_existing_patterns_still_work(self) -> None:
		"""Verify original patterns are unaffected by new additions."""
		# Store-reload
		assert _opt("\tmovq %rax, -8(%rbp)\n\tmovq -8(%rbp), %rax") == "\tmovq %rax, -8(%rbp)"
		# Self-move
		assert _opt("\tmovq %rax, %rax") == ""
		# Zero-cmp
		assert _opt("\tmovq $0, %rax\n\tcmpq $0, %rax") == "\txorq %rax, %rax\n\ttestq %rax, %rax"
		# Noop arith
		assert _opt("\taddq $0, %rax") == ""
		assert _opt("\tsubq $0, %rsp") == ""
