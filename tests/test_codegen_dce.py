"""Tests for assembly-level dead code elimination pass."""

from compiler.codegen import dead_code_eliminate_asm


class TestUnreachableCodeElimination:
	"""Test removal of unreachable instructions after unconditional jumps."""

	def test_removes_code_after_jmp_until_label(self):
		lines = [
			"\tjmp .L1",
			"\tmovq $1, %rax",
			"\tmovq $2, %rcx",
			".L1:",
			"\tmovq $3, %rax",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == [
			"\tjmp .L1",
			".L1:",
			"\tmovq $3, %rax",
		]

	def test_preserves_code_after_conditional_jump(self):
		lines = [
			"\tje .L1",
			"\tmovq $1, %rax",
			".L1:",
			"\tmovq $2, %rax",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == lines

	def test_multiple_jmp_dead_zones(self):
		lines = [
			"\tjmp .L1",
			"\tmovq $1, %rax",
			".L1:",
			"\tjmp .L2",
			"\tmovq $2, %rax",
			"\taddq %rcx, %rax",
			".L2:",
			"\tret",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == [
			"\tjmp .L1",
			".L1:",
			"\tjmp .L2",
			".L2:",
			"\tret",
		]

	def test_preserves_directives_after_jmp(self):
		"""Directives like .size and .globl should survive dead zones."""
		lines = [
			"\tjmp .L1",
			"\tmovq $1, %rax",
			".size foo, .-foo",
			".globl bar",
		]
		result = dead_code_eliminate_asm(lines)
		assert ".size foo, .-foo" in result
		assert ".globl bar" in result

	def test_empty_dead_zone(self):
		"""jmp immediately followed by label should be preserved as-is."""
		lines = [
			"\tjmp .L1",
			".L1:",
			"\tmovq $1, %rax",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == lines

	def test_no_jmp_no_changes(self):
		lines = [
			"\tmovq $1, %rax",
			"\taddq %rcx, %rax",
			"\tret",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == lines


class TestDeadStoreElimination:
	"""Test removal of dead register stores that are immediately overwritten."""

	def test_removes_dead_movq_to_same_register(self):
		lines = [
			"\tmovq $1, %rax",
			"\tmovq $2, %rax",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == ["\tmovq $2, %rax"]

	def test_preserves_store_when_register_is_read(self):
		"""If the next instruction reads the register, the store is live."""
		lines = [
			"\tmovq $1, %rax",
			"\taddq %rax, %rcx",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == lines

	def test_removes_dead_leaq(self):
		lines = [
			"\tleaq 8(%rbp), %rax",
			"\tmovq $42, %rax",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == ["\tmovq $42, %rax"]

	def test_preserves_store_to_different_register(self):
		lines = [
			"\tmovq $1, %rax",
			"\tmovq $2, %rcx",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == lines

	def test_preserves_store_to_memory(self):
		"""Stores to memory should never be eliminated."""
		lines = [
			"\tmovq %rax, -8(%rbp)",
			"\tmovq $2, %rax",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == lines

	def test_preserves_store_before_label(self):
		"""Don't eliminate stores before labels (branch targets)."""
		lines = [
			"\tmovq $1, %rax",
			".L1:",
			"\tmovq $2, %rax",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == lines

	def test_preserves_store_before_jump(self):
		lines = [
			"\tmovq $1, %rax",
			"\tjmp .L1",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == lines

	def test_preserves_store_before_ret(self):
		lines = [
			"\tmovq $1, %rax",
			"\tret",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == lines

	def test_preserves_rbp_writes(self):
		"""Frame pointer writes should never be eliminated."""
		lines = [
			"\tmovq %rsp, %rbp",
			"\tmovq $0, %rbp",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == lines

	def test_preserves_rsp_writes(self):
		"""Stack pointer writes should never be eliminated."""
		lines = [
			"\tsubq $16, %rsp",
			"\tsubq $8, %rsp",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == lines

	def test_removes_dead_sub_register_write(self):
		"""Writing to %eax then %rax: the %eax write is dead."""
		lines = [
			"\tmovl $1, %eax",
			"\tmovq $2, %rax",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == ["\tmovq $2, %rax"]

	def test_preserves_call_instructions(self):
		"""Call instructions have side effects and should never be eliminated."""
		lines = [
			"\tcall foo",
			"\tmovq $1, %rax",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == lines

	def test_preserves_store_before_call(self):
		"""Stores before calls are live (call may read/clobber registers)."""
		lines = [
			"\tmovq $1, %rax",
			"\tcall foo",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == lines

	def test_xor_zeroing_idiom_kills_previous_store(self):
		"""xorq %rax, %rax zeroes the register -- previous write is dead."""
		lines = [
			"\tmovq $1, %rax",
			"\txorq %rax, %rax",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == ["\txorq %rax, %rax"]

	def test_next_reads_via_memory_operand(self):
		"""If next instruction reads register through memory operand, keep the store."""
		lines = [
			"\tmovq $1000, %rax",
			"\tmovq (%rax), %rax",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == lines


class TestCombinedOptimizations:
	"""Test that both passes work together correctly."""

	def test_unreachable_and_dead_store(self):
		lines = [
			"\tmovq $1, %rax",
			"\tmovq $2, %rax",
			"\tjmp .L1",
			"\tmovq $3, %rax",
			".L1:",
			"\tret",
		]
		result = dead_code_eliminate_asm(lines)
		# Dead store: first movq removed; Unreachable: movq $3 removed
		assert result == [
			"\tmovq $2, %rax",
			"\tjmp .L1",
			".L1:",
			"\tret",
		]

	def test_realistic_function_snippet(self):
		"""Simulate a realistic codegen pattern with redundant moves."""
		lines = [
			".globl foo",
			".type foo, @function",
			"foo:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tsubq $16, %rsp",
			"\tmovq %rdi, -8(%rbp)",
			"\tmovq -8(%rbp), %rax",
			"\tmovq $42, %rax",
			"\tmovq %rbp, %rsp",
			"\tpopq %rbp",
			"\tret",
			".size foo, .-foo",
		]
		result = dead_code_eliminate_asm(lines)
		# The load from -8(%rbp) into %rax is dead since %rax is immediately overwritten
		assert "\tmovq -8(%rbp), %rax" not in result
		assert "\tmovq $42, %rax" in result

	def test_preserves_complete_function_structure(self):
		"""Ensure function prologue/epilogue is never disturbed."""
		lines = [
			".globl main",
			".type main, @function",
			"main:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tmovq $0, %rax",
			"\tmovq %rbp, %rsp",
			"\tpopq %rbp",
			"\tret",
			".size main, .-main",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == lines

	def test_empty_input(self):
		assert dead_code_eliminate_asm([]) == []

	def test_only_labels_and_directives(self):
		lines = [
			".section .text",
			".globl foo",
			"foo:",
			".size foo, .-foo",
		]
		result = dead_code_eliminate_asm(lines)
		assert result == lines
