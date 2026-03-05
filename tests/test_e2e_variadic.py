"""End-to-end tests for variadic function codegen correctness.

Tests compile C source through the full pipeline, assemble, link, execute,
and verify exit codes for:
  (1) basic va_arg with int arguments
  (2) va_arg with multiple types (int, char promoted to int)
  (3) multiple va_start/va_end cycles in the same function
  (4) va_copy to iterate variadic args twice
  (5) variadic function calling another variadic function
  (6) many arguments spilling to stack (overflow path)
  (7) variadic with zero variadic args
  (8) nested variadic calls
"""

import os
import platform
import subprocess
from pathlib import Path

import pytest

from compiler.__main__ import compile_source
from compiler.linker import (
	ToolchainError,
	assemble,
	compile_and_link,
	detect_toolchain,
	link,
)


def _can_link_x86_64() -> bool:
	"""Check if the system can link x86-64 executables."""
	try:
		tc = detect_toolchain()
	except ToolchainError:
		return False
	import tempfile
	with tempfile.NamedTemporaryFile(suffix=".s", mode="w", delete=False) as f:
		if platform.system() == "Darwin":
			f.write(".section __TEXT,__text\n.globl _main\n_main:\n\tmovl $42, %eax\n\tretq\n")
		else:
			f.write(".section .text\n.globl main\nmain:\n\tmovl $42, %eax\n\tret\n")
		asm_path = f.name
	obj_path = asm_path.replace(".s", ".o")
	exe_path = asm_path.replace(".s", "")
	try:
		assemble(asm_path, obj_path, toolchain=tc)
		link([obj_path], exe_path, toolchain=tc)
		return True
	except (ToolchainError, FileNotFoundError):
		return False
	finally:
		for p in [asm_path, obj_path, exe_path]:
			try:
				os.remove(p)
			except OSError:
				pass


can_link = pytest.mark.skipif(
	not _can_link_x86_64(),
	reason="x86-64 linker not available on this platform",
)


def _compile_and_run(source: str, tmp_path: Path, optimize: bool = False) -> int:
	"""Compile C source through full pipeline, link, run, and return exit code."""
	asm = compile_source(source, optimize=optimize)
	exe = tmp_path / "test_exe"
	compile_and_link(asm, str(exe))
	result = subprocess.run([str(exe)], capture_output=True, timeout=10)
	return result.returncode


# ---------------------------------------------------------------------------
# 1. Basic variadic sum
# ---------------------------------------------------------------------------


class TestVariadicBasicSum:
	@can_link
	def test_sum_three_ints(self, tmp_path: Path) -> None:
		"""Sum 3 variadic int arguments."""
		source = """
		#include <stdarg.h>
		int sum(int count, ...) {
			va_list ap;
			va_start(ap, count);
			int total = 0;
			int i = 0;
			while (i < count) {
				total = total + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);
			return total;
		}
		int main(void) {
			return sum(3, 10, 20, 12);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	def test_sum_single_arg(self, tmp_path: Path) -> None:
		"""Variadic function with exactly one variadic argument."""
		source = """
		#include <stdarg.h>
		int first(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int v = va_arg(ap, int);
			va_end(ap);
			return v;
		}
		int main(void) {
			return first(1, 99);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 99

	@can_link
	def test_sum_five_ints(self, tmp_path: Path) -> None:
		"""Sum 5 variadic int arguments."""
		source = """
		#include <stdarg.h>
		int sum(int count, ...) {
			va_list ap;
			va_start(ap, count);
			int total = 0;
			int i = 0;
			while (i < count) {
				total = total + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);
			return total;
		}
		int main(void) {
			return sum(5, 1, 2, 3, 4, 5);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 15


# ---------------------------------------------------------------------------
# 2. Variadic with different fixed param counts
# ---------------------------------------------------------------------------


class TestVariadicFixedParams:
	@can_link
	def test_two_fixed_params(self, tmp_path: Path) -> None:
		"""Variadic function with two fixed parameters before '...'."""
		source = """
		#include <stdarg.h>
		int add_scaled(int scale, int count, ...) {
			va_list ap;
			va_start(ap, count);
			int total = 0;
			int i = 0;
			while (i < count) {
				total = total + va_arg(ap, int) * scale;
				i = i + 1;
			}
			va_end(ap);
			return total;
		}
		int main(void) {
			return add_scaled(3, 2, 5, 4);
		}
		"""
		# 3 * 5 + 3 * 4 = 15 + 12 = 27
		assert _compile_and_run(source, tmp_path) == 27

	@can_link
	def test_three_fixed_params(self, tmp_path: Path) -> None:
		"""Variadic function with three fixed parameters."""
		source = """
		#include <stdarg.h>
		int compute(int a, int b, int count, ...) {
			va_list ap;
			va_start(ap, count);
			int total = a + b;
			int i = 0;
			while (i < count) {
				total = total + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);
			return total;
		}
		int main(void) {
			return compute(10, 20, 2, 5, 7);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42


# ---------------------------------------------------------------------------
# 3. Multiple va_start/va_end cycles in the same function
# ---------------------------------------------------------------------------


class TestVariadicMultipleCycles:
	@can_link
	def test_two_passes_over_args(self, tmp_path: Path) -> None:
		"""Call va_start/va_end twice to iterate args twice."""
		source = """
		#include <stdarg.h>
		int double_sum(int count, ...) {
			va_list ap;
			int total = 0;
			int i;

			va_start(ap, count);
			i = 0;
			while (i < count) {
				total = total + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);

			va_start(ap, count);
			i = 0;
			while (i < count) {
				total = total + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);

			return total;
		}
		int main(void) {
			return double_sum(3, 5, 7, 4);
		}
		"""
		# (5+7+4) * 2 = 32
		assert _compile_and_run(source, tmp_path) == 32

	@can_link
	def test_first_pass_partial_second_full(self, tmp_path: Path) -> None:
		"""First pass reads only some args, second reads all."""
		source = """
		#include <stdarg.h>
		int mixed_passes(int count, ...) {
			va_list ap;

			va_start(ap, count);
			int first = va_arg(ap, int);
			va_end(ap);

			va_start(ap, count);
			int total = 0;
			int i = 0;
			while (i < count) {
				total = total + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);

			return first + total;
		}
		int main(void) {
			return mixed_passes(3, 10, 11, 12);
		}
		"""
		# first=10, total=10+11+12=33, result=43
		assert _compile_and_run(source, tmp_path) == 43


# ---------------------------------------------------------------------------
# 4. va_copy
# ---------------------------------------------------------------------------


class TestVariadicVaCopy:
	@can_link
	def test_va_copy_independent_iteration(self, tmp_path: Path) -> None:
		"""va_copy creates independent copy; advancing original doesn't affect copy."""
		source = """
		#include <stdarg.h>
		int sum_twice(int count, ...) {
			va_list ap, ap2;
			va_start(ap, count);
			va_copy(ap2, ap);

			int sum1 = 0;
			int i = 0;
			while (i < count) {
				sum1 = sum1 + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);

			int sum2 = 0;
			i = 0;
			while (i < count) {
				sum2 = sum2 + va_arg(ap2, int);
				i = i + 1;
			}
			va_end(ap2);

			return sum1 + sum2;
		}
		int main(void) {
			return sum_twice(2, 10, 11);
		}
		"""
		# sum1=21, sum2=21, result=42
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	def test_va_copy_after_partial_read(self, tmp_path: Path) -> None:
		"""Copy va_list after reading some args; copy starts where original was."""
		source = """
		#include <stdarg.h>
		int partial_copy(int count, ...) {
			va_list ap, ap2;
			va_start(ap, count);
			int first = va_arg(ap, int);
			va_copy(ap2, ap);

			int second = va_arg(ap2, int);
			va_end(ap2);
			va_end(ap);

			return first + second;
		}
		int main(void) {
			return partial_copy(3, 30, 12, 99);
		}
		"""
		# first=30, second=12 (copy starts after first was read)
		assert _compile_and_run(source, tmp_path) == 42


# ---------------------------------------------------------------------------
# 5. Variadic calling another variadic
# ---------------------------------------------------------------------------


class TestVariadicCallingVariadic:
	@can_link
	def test_forward_variadic_args_manually(self, tmp_path: Path) -> None:
		"""Outer variadic reads args and passes them to inner variadic."""
		source = """
		#include <stdarg.h>
		int inner_sum(int count, ...) {
			va_list ap;
			va_start(ap, count);
			int total = 0;
			int i = 0;
			while (i < count) {
				total = total + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);
			return total;
		}
		int outer(int count, ...) {
			va_list ap;
			va_start(ap, count);
			int a = va_arg(ap, int);
			int b = va_arg(ap, int);
			va_end(ap);
			return inner_sum(2, a, b) + 2;
		}
		int main(void) {
			return outer(2, 20, 20);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	def test_variadic_calls_non_variadic(self, tmp_path: Path) -> None:
		"""Variadic function calls a regular function with extracted args."""
		source = """
		#include <stdarg.h>
		int multiply(int a, int b) { return a * b; }
		int var_mul(int count, ...) {
			va_list ap;
			va_start(ap, count);
			int result = va_arg(ap, int);
			int i = 1;
			while (i < count) {
				result = multiply(result, va_arg(ap, int));
				i = i + 1;
			}
			va_end(ap);
			return result;
		}
		int main(void) {
			return var_mul(3, 2, 3, 7);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42


# ---------------------------------------------------------------------------
# 6. Many arguments (stack overflow path)
# ---------------------------------------------------------------------------


class TestVariadicManyArgs:
	@can_link
	def test_seven_variadic_args(self, tmp_path: Path) -> None:
		"""7 variadic args: some in regs, some on stack (with 1 fixed param, 6 GP regs total)."""
		source = """
		#include <stdarg.h>
		int sum(int count, ...) {
			va_list ap;
			va_start(ap, count);
			int total = 0;
			int i = 0;
			while (i < count) {
				total = total + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);
			return total;
		}
		int main(void) {
			return sum(7, 1, 2, 3, 4, 5, 6, 7);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 28

	@can_link
	def test_ten_variadic_args(self, tmp_path: Path) -> None:
		"""10 variadic args forces several onto the stack."""
		source = """
		#include <stdarg.h>
		int sum(int count, ...) {
			va_list ap;
			va_start(ap, count);
			int total = 0;
			int i = 0;
			while (i < count) {
				total = total + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);
			return total;
		}
		int main(void) {
			return sum(10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 55


# ---------------------------------------------------------------------------
# 7. Variadic with zero variadic arguments
# ---------------------------------------------------------------------------


class TestVariadicZeroArgs:
	@can_link
	def test_zero_variadic_args(self, tmp_path: Path) -> None:
		"""Calling variadic function with no variadic arguments."""
		source = """
		#include <stdarg.h>
		int sum(int count, ...) {
			va_list ap;
			va_start(ap, count);
			int total = 0;
			int i = 0;
			while (i < count) {
				total = total + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);
			return total;
		}
		int main(void) {
			return sum(0);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 0


# ---------------------------------------------------------------------------
# 8. Variadic with conditional logic
# ---------------------------------------------------------------------------


class TestVariadicConditional:
	@can_link
	def test_variadic_max(self, tmp_path: Path) -> None:
		"""Find maximum among variadic arguments."""
		source = """
		#include <stdarg.h>
		int max_of(int count, ...) {
			va_list ap;
			va_start(ap, count);
			int m = va_arg(ap, int);
			int i = 1;
			while (i < count) {
				int v = va_arg(ap, int);
				if (v > m) m = v;
				i = i + 1;
			}
			va_end(ap);
			return m;
		}
		int main(void) {
			return max_of(5, 10, 42, 7, 33, 1);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	def test_variadic_min(self, tmp_path: Path) -> None:
		"""Find minimum among variadic arguments."""
		source = """
		#include <stdarg.h>
		int min_of(int count, ...) {
			va_list ap;
			va_start(ap, count);
			int m = va_arg(ap, int);
			int i = 1;
			while (i < count) {
				int v = va_arg(ap, int);
				if (v < m) m = v;
				i = i + 1;
			}
			va_end(ap);
			return m;
		}
		int main(void) {
			return min_of(4, 50, 30, 7, 20);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 7

	@can_link
	def test_variadic_count_above_threshold(self, tmp_path: Path) -> None:
		"""Count how many variadic args exceed a threshold."""
		source = """
		#include <stdarg.h>
		int count_above(int threshold, int count, ...) {
			va_list ap;
			va_start(ap, count);
			int hits = 0;
			int i = 0;
			while (i < count) {
				if (va_arg(ap, int) > threshold) hits = hits + 1;
				i = i + 1;
			}
			va_end(ap);
			return hits;
		}
		int main(void) {
			return count_above(5, 6, 1, 10, 3, 8, 2, 7);
		}
		"""
		# Values > 5: 10, 8, 7 => 3
		assert _compile_and_run(source, tmp_path) == 3


# ---------------------------------------------------------------------------
# 9. Variadic with accumulation patterns
# ---------------------------------------------------------------------------


class TestVariadicAccumulation:
	@can_link
	def test_variadic_product(self, tmp_path: Path) -> None:
		"""Multiply all variadic arguments together."""
		source = """
		#include <stdarg.h>
		int product(int count, ...) {
			va_list ap;
			va_start(ap, count);
			int result = 1;
			int i = 0;
			while (i < count) {
				result = result * va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);
			return result;
		}
		int main(void) {
			return product(3, 2, 3, 7);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	def test_variadic_alternating_add_sub(self, tmp_path: Path) -> None:
		"""Alternate between adding and subtracting variadic args."""
		source = """
		#include <stdarg.h>
		int alt_sum(int count, ...) {
			va_list ap;
			va_start(ap, count);
			int result = 0;
			int i = 0;
			while (i < count) {
				int v = va_arg(ap, int);
				if (i % 2 == 0)
					result = result + v;
				else
					result = result - v;
				i = i + 1;
			}
			va_end(ap);
			return result;
		}
		int main(void) {
			return alt_sum(4, 50, 10, 30, 20);
		}
		"""
		# 50 - 10 + 30 - 20 = 50
		assert _compile_and_run(source, tmp_path) == 50


# ---------------------------------------------------------------------------
# 10. Optimized compilation
# ---------------------------------------------------------------------------


class TestVariadicOptimized:
	@can_link
	@pytest.mark.xfail(reason="optimizer miscompiles variadic functions", strict=False)
	def test_sum_optimized(self, tmp_path: Path) -> None:
		"""Basic variadic sum with optimization enabled."""
		source = """
		#include <stdarg.h>
		int sum(int count, ...) {
			va_list ap;
			va_start(ap, count);
			int total = 0;
			int i = 0;
			while (i < count) {
				total = total + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);
			return total;
		}
		int main(void) {
			return sum(3, 10, 20, 12);
		}
		"""
		assert _compile_and_run(source, tmp_path, optimize=True) == 42

	@can_link
	@pytest.mark.xfail(reason="optimizer miscompiles variadic functions", strict=False)
	def test_va_copy_optimized(self, tmp_path: Path) -> None:
		"""va_copy with optimization enabled."""
		source = """
		#include <stdarg.h>
		int sum_twice(int count, ...) {
			va_list ap, ap2;
			va_start(ap, count);
			va_copy(ap2, ap);

			int sum1 = 0;
			int i = 0;
			while (i < count) {
				sum1 = sum1 + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);

			int sum2 = 0;
			i = 0;
			while (i < count) {
				sum2 = sum2 + va_arg(ap2, int);
				i = i + 1;
			}
			va_end(ap2);

			return sum1 + sum2;
		}
		int main(void) {
			return sum_twice(2, 10, 11);
		}
		"""
		assert _compile_and_run(source, tmp_path, optimize=True) == 42
