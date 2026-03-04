"""Integration tests: register allocator wired into the compile pipeline."""

from compiler.__main__ import compile_source
from compiler.regalloc import ALLOCATABLE_REGS, CALLEE_SAVED_REGS


def _count_stack_accesses(assembly: str) -> int:
	"""Count lines that reference stack slots via (%rbp)."""
	return sum(1 for line in assembly.splitlines() if "(%rbp)" in line)


class TestPipelineRegalloc:
	"""Verify that --optimize activates register allocation end-to-end."""

	def test_optimized_uses_callee_saved_registers(self) -> None:
		"""Optimized output should contain callee-saved register references."""
		source = """
		int compute(int a, int b) {
			int x = a + b;
			int y = x * 2;
			int z = y - a;
			return z + b;
		}
		"""
		asm_opt = compile_source(source, optimize=True)
		has_callee_saved = any(reg in asm_opt for reg in CALLEE_SAVED_REGS)
		assert has_callee_saved, (
			f"Optimized assembly should use callee-saved registers.\n{asm_opt}"
		)

	def test_optimized_reduces_stack_spills(self) -> None:
		"""Optimized output should have fewer stack accesses than unoptimized."""
		source = """
		int work(int a, int b) {
			int c = a + b;
			int d = c * a;
			int e = d - b;
			return e;
		}
		"""
		asm_plain = compile_source(source, optimize=False)
		asm_opt = compile_source(source, optimize=True)

		spills_plain = _count_stack_accesses(asm_plain)
		spills_opt = _count_stack_accesses(asm_opt)

		assert spills_opt < spills_plain, (
			f"Optimized ({spills_opt} stack refs) should have fewer stack "
			f"accesses than unoptimized ({spills_plain}).\n"
			f"--- unoptimized ---\n{asm_plain}\n--- optimized ---\n{asm_opt}"
		)

	def test_optimized_uses_allocatable_registers(self) -> None:
		"""Optimized output should reference registers from the allocatable set."""
		source = """
		int add(int x, int y) {
			return x + y;
		}
		"""
		asm_opt = compile_source(source, optimize=True)
		has_alloc = any(reg in asm_opt for reg in ALLOCATABLE_REGS)
		assert has_alloc, (
			f"Optimized assembly should use allocatable registers.\n{asm_opt}"
		)

	def test_unoptimized_does_not_use_allocatable_registers(self) -> None:
		"""Without --optimize, assembly should not contain allocatable registers."""
		source = """
		int add(int x, int y) {
			return x + y;
		}
		"""
		asm_plain = compile_source(source, optimize=False)
		has_alloc = any(reg in asm_plain for reg in ALLOCATABLE_REGS)
		assert not has_alloc, (
			f"Unoptimized assembly should NOT use allocatable registers.\n{asm_plain}"
		)

	def test_loop_with_regalloc_pipeline(self) -> None:
		"""Loop code compiled with --optimize should use registers and reduce spills."""
		source = """
		int sum_to(int n) {
			int sum = 0;
			int i = 1;
			while (i <= n) {
				sum = sum + i;
				i = i + 1;
			}
			return sum;
		}
		"""
		asm_opt = compile_source(source, optimize=True)
		asm_plain = compile_source(source, optimize=False)

		assert any(reg in asm_opt for reg in CALLEE_SAVED_REGS), (
			"Loop with --optimize should use callee-saved registers"
		)
		assert _count_stack_accesses(asm_opt) < _count_stack_accesses(asm_plain), (
			"Loop with --optimize should reduce stack accesses"
		)

	def test_multi_function_regalloc(self) -> None:
		"""Multiple functions should each get register allocation."""
		source = """
		int foo(int a) {
			int b = a * 2;
			return b;
		}
		int bar(int x) {
			int y = x + 1;
			return y;
		}
		"""
		asm_opt = compile_source(source, optimize=True)
		assert ".globl foo" in asm_opt
		assert ".globl bar" in asm_opt
		assert any(reg in asm_opt for reg in ALLOCATABLE_REGS)

	def test_call_crossing_uses_callee_saved(self) -> None:
		"""Temps live across a call should be placed in callee-saved registers."""
		source = """
		int helper(int x) {
			return x * 2;
		}
		int caller(int a) {
			int before = a + 1;
			int result = helper(before);
			return before + result;
		}
		"""
		asm_opt = compile_source(source, optimize=True)
		# 'before' is live across the call to helper, must use callee-saved
		has_callee = any(reg in asm_opt for reg in CALLEE_SAVED_REGS)
		assert has_callee, (
			f"Temp live across call should use callee-saved register.\n{asm_opt}"
		)
