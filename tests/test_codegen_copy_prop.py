"""Tests for codegen copy propagation pass."""

from compiler.codegen import copy_propagate_asm


class TestCopyPropagateBasic:
	"""Basic copy propagation: eliminate redundant movq reg,reg."""

	def test_eliminate_self_move(self):
		lines = ["\tmovq %rax, %rax"]
		result = copy_propagate_asm(lines)
		assert result == []

	def test_eliminate_chain_self_move(self):
		"""movq %rax, %rbx; movq %rbx, %rax -> keep first, eliminate second."""
		lines = [
			"\tmovq %rax, %rbx",
			"\tmovq %rbx, %rax",
		]
		result = copy_propagate_asm(lines)
		assert len(result) == 1
		assert result[0] == "\tmovq %rax, %rbx"

	def test_shorten_chain(self):
		"""movq %rax, %rbx; movq %rbx, %rcx -> movq %rax, %rbx; movq %rax, %rcx."""
		lines = [
			"\tmovq %rax, %rbx",
			"\tmovq %rbx, %rcx",
		]
		result = copy_propagate_asm(lines)
		assert len(result) == 2
		assert result[0] == "\tmovq %rax, %rbx"
		assert result[1] == "\tmovq %rax, %rcx"

	def test_three_step_chain(self):
		"""movq %rax, %rbx; movq %rbx, %rcx; movq %rcx, %rdx -> resolves to %rax."""
		lines = [
			"\tmovq %rax, %rbx",
			"\tmovq %rbx, %rcx",
			"\tmovq %rcx, %rdx",
		]
		result = copy_propagate_asm(lines)
		assert len(result) == 3
		assert result[2] == "\tmovq %rax, %rdx"

	def test_chain_ending_in_self_move(self):
		"""movq %rax, %rbx; movq %rbx, %rcx; movq %rcx, %rax -> last is eliminated."""
		lines = [
			"\tmovq %rax, %rbx",
			"\tmovq %rbx, %rcx",
			"\tmovq %rcx, %rax",
		]
		result = copy_propagate_asm(lines)
		assert len(result) == 2


class TestCopyPropagateInvalidation:
	"""Copy tracking is properly invalidated on writes."""

	def test_invalidate_on_arithmetic(self):
		"""After addq writes to %rbx, the copy is invalid."""
		lines = [
			"\tmovq %rax, %rbx",
			"\taddq $1, %rbx",
			"\tmovq %rbx, %rcx",
		]
		result = copy_propagate_asm(lines)
		# After addq, %rbx != %rax, so movq %rbx, %rcx should NOT resolve to %rax
		assert result[-1] == "\tmovq %rbx, %rcx"

	def test_invalidate_on_label(self):
		"""Labels reset copy tracking (branch target)."""
		lines = [
			"\tmovq %rax, %rbx",
			".L1:",
			"\tmovq %rbx, %rcx",
		]
		result = copy_propagate_asm(lines)
		# After label, copy tracking is reset
		assert "\tmovq %rbx, %rcx" in result

	def test_invalidate_on_jump(self):
		"""Jumps reset copy tracking."""
		lines = [
			"\tmovq %rax, %rbx",
			"\tjmp .L1",
			"\tmovq %rbx, %rcx",
		]
		result = copy_propagate_asm(lines)
		# After jump, copies are cleared
		assert "\tmovq %rbx, %rcx" in result

	def test_call_clobbers_caller_saved(self):
		"""Call invalidates caller-saved registers but preserves callee-saved."""
		lines = [
			"\tmovq %r12, %rbx",
			"\tmovq %rax, %rcx",
			"\tcall foo",
			"\tmovq %rbx, %rsi",  # rbx -> r12 should be preserved (both callee-saved)
			"\tmovq %rcx, %rdi",  # rcx was clobbered by call, no propagation
		]
		result = copy_propagate_asm(lines)
		# %rbx -> %r12 should be preserved across call (callee-saved)
		assert "\tmovq %r12, %rsi" in result
		# %rcx was clobbered, so movq %rcx, %rdi stays
		assert "\tmovq %rcx, %rdi" in result

	def test_invalidate_on_setcc(self):
		"""setCC writes to a sub-register, invalidating the parent."""
		lines = [
			"\tmovq %rcx, %rax",
			"\tsete %al",
			"\tmovq %rax, %rbx",
		]
		result = copy_propagate_asm(lines)
		# After sete %al, %rax is no longer a copy of %rcx
		assert "\tmovq %rax, %rbx" in result

	def test_invalidate_on_movzbq(self):
		"""movzbq writes to destination register."""
		lines = [
			"\tmovq %rcx, %rax",
			"\tmovzbq %dl, %rax",
			"\tmovq %rax, %rbx",
		]
		result = copy_propagate_asm(lines)
		assert "\tmovq %rax, %rbx" in result

	def test_invalidate_on_negq(self):
		"""negq modifies its operand, invalidating copies."""
		lines = [
			"\tmovq %rax, %rbx",
			"\tnegq %rbx",
			"\tmovq %rbx, %rcx",
		]
		result = copy_propagate_asm(lines)
		assert "\tmovq %rbx, %rcx" in result


class TestCopyPropagatePreservation:
	"""Non-mov instructions and memory ops are preserved unchanged."""

	def test_non_mov_lines_preserved(self):
		"""Arithmetic and other instructions pass through unchanged."""
		lines = [
			"\taddq %rcx, %rax",
			"\tsubq $8, %rsp",
			"\timulq %rcx, %rax",
		]
		result = copy_propagate_asm(lines)
		assert result == lines

	def test_memory_mov_preserved(self):
		"""movq involving memory operands is not affected."""
		lines = [
			"\tmovq %rax, -8(%rbp)",
			"\tmovq -8(%rbp), %rcx",
		]
		result = copy_propagate_asm(lines)
		assert result == lines

	def test_directives_preserved(self):
		"""Assembler directives pass through unchanged."""
		lines = [
			".section .text",
			".globl main",
			"\t.align 16",
		]
		result = copy_propagate_asm(lines)
		assert result == lines

	def test_label_and_function_structure(self):
		"""Function structure with labels is preserved."""
		lines = [
			".globl foo",
			".type foo, @function",
			"foo:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tmovq %rdi, %rax",
			"\tmovq %rax, %rbx",
			"\tpopq %rbp",
			"\tret",
		]
		result = copy_propagate_asm(lines)
		# movq %rax, %rbx should resolve to movq %rdi, %rbx
		# because after "foo:" label copies are reset, then movq %rsp, %rbp is not reg-reg (rbp/rsp tracked)
		# then movq %rdi, %rax establishes copy, movq %rax, %rbx -> movq %rdi, %rbx
		assert "\tmovq %rdi, %rbx" in result


class TestCopyPropagateRealPatterns:
	"""Test patterns that actually occur in the compiler's codegen output."""

	def test_load_compute_store_pattern(self):
		"""Common pattern: load from reg, compute, store to reg."""
		lines = [
			"\tmovq %rbx, %rax",   # _load_value from allocated reg
			"\taddq %rcx, %rax",   # compute
			"\tmovq %rax, %r12",   # _store_to_temp to allocated reg
		]
		result = copy_propagate_asm(lines)
		# addq writes %rax so the copy %rax <- %rbx is invalidated
		# movq %rax, %r12 should remain unchanged
		assert result == lines

	def test_copy_through_rax(self):
		"""IRCopy that goes through %rax when temps are in registers."""
		lines = [
			"\tmovq %rbx, %rax",
			"\tmovq %rax, %r12",
		]
		result = copy_propagate_asm(lines)
		# Should shorten to movq %rbx, %r12
		assert len(result) == 2
		assert result[1] == "\tmovq %rbx, %r12"

	def test_redundant_copy_back(self):
		"""Copy to %rax then back: should eliminate the second mov."""
		lines = [
			"\tmovq %rbx, %rax",
			"\tmovq %rax, %rbx",
		]
		result = copy_propagate_asm(lines)
		assert len(result) == 1
		assert result[0] == "\tmovq %rbx, %rax"

	def test_multiple_functions_independent(self):
		"""Each function label resets copy state."""
		lines = [
			"foo:",
			"\tmovq %rax, %rbx",
			"\tmovq %rbx, %rcx",
			".size foo, .-foo",
			"bar:",
			"\tmovq %rbx, %rcx",  # fresh context, no propagation
		]
		result = copy_propagate_asm(lines)
		# In foo: movq %rbx, %rcx -> movq %rax, %rcx
		assert result[2] == "\tmovq %rax, %rcx"
		# In bar: no copy info, stays as-is
		assert result[-1] == "\tmovq %rbx, %rcx"


class TestCopyPropagateMoveCount:
	"""Verify that copy propagation reduces total mov count for simple patterns."""

	def test_fewer_movs_simple_chain(self):
		lines = [
			"\tmovq %rax, %rbx",
			"\tmovq %rbx, %rax",
			"\tmovq %rax, %rcx",
		]
		original_movs = sum(1 for ln in lines if "\tmovq" in ln)
		result = copy_propagate_asm(lines)
		result_movs = sum(1 for ln in result if "\tmovq" in ln)
		assert result_movs < original_movs

	def test_fewer_movs_copy_chain(self):
		lines = [
			"\tmovq %rdi, %rax",
			"\tmovq %rax, %rbx",
			"\tmovq %rbx, %rcx",
			"\tmovq %rcx, %rdi",  # resolves to self-move, eliminated
		]
		original_movs = sum(1 for ln in lines if "\tmovq" in ln)
		result = copy_propagate_asm(lines)
		result_movs = sum(1 for ln in result if "\tmovq" in ln)
		assert result_movs < original_movs


class TestCopyPropagateEdgeCases:
	"""Edge cases for robustness."""

	def test_empty_input(self):
		assert copy_propagate_asm([]) == []

	def test_no_movs(self):
		lines = ["\taddq $1, %rax", "\tret"]
		assert copy_propagate_asm(lines) == lines

	def test_cmpq_does_not_invalidate(self):
		"""cmpq is read-only and should not invalidate copies."""
		lines = [
			"\tmovq %rax, %rbx",
			"\tcmpq $0, %rbx",
			"\tmovq %rbx, %rcx",
		]
		result = copy_propagate_asm(lines)
		assert "\tmovq %rax, %rcx" in result

	def test_testq_does_not_invalidate(self):
		"""testq is read-only and should not invalidate copies."""
		lines = [
			"\tmovq %rax, %rbx",
			"\ttestq %rbx, %rbx",
			"\tmovq %rbx, %rcx",
		]
		result = copy_propagate_asm(lines)
		assert "\tmovq %rax, %rcx" in result

	def test_r8_through_r15(self):
		"""Extended registers work correctly."""
		lines = [
			"\tmovq %r8, %r9",
			"\tmovq %r9, %r10",
		]
		result = copy_propagate_asm(lines)
		assert result[1] == "\tmovq %r8, %r10"

	def test_pushq_does_not_invalidate(self):
		"""pushq reads but does not write to its register operand."""
		lines = [
			"\tmovq %rax, %rbx",
			"\tpushq %rbx",
			"\tmovq %rbx, %rcx",
		]
		result = copy_propagate_asm(lines)
		assert "\tmovq %rax, %rcx" in result
