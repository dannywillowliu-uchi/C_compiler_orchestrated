"""E2E compile-assemble-link-run tests for do-while, ternary, comma, char ops, and globals."""

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


def _compile_and_run(source: str, tmp_path: Path) -> int:
	"""Compile C source through full pipeline, link, run, and return exit code."""
	asm = compile_source(source)
	exe = tmp_path / "test_exe"
	compile_and_link(asm, str(exe))
	result = subprocess.run([str(exe)], capture_output=True, timeout=10)
	return result.returncode


class TestDoWhileLoop:
	"""Do-while loop: countdown from 10, verify final value."""

	@can_link
	def test_do_while_countdown(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int n = 10;
			do {
				n = n - 1;
			} while (n > 0);
			return n;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 0

	@can_link
	def test_do_while_sum(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int i = 1;
			int sum = 0;
			do {
				sum = sum + i;
				i = i + 1;
			} while (i <= 10);
			return sum;
		}
		"""
		# 1+2+...+10 = 55
		assert _compile_and_run(source, tmp_path) == 55

	@can_link
	def test_do_while_runs_at_least_once(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int x = 0;
			do {
				x = 42;
			} while (0);
			return x;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42


class TestTernaryExpression:
	"""Ternary expression: compute max/min of two values."""

	@can_link
	def test_ternary_max(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int a = 30;
			int b = 50;
			int max = (a > b) ? a : b;
			return max;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 50

	@can_link
	def test_ternary_min(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int a = 30;
			int b = 50;
			int min = (a < b) ? a : b;
			return min;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 30

	@can_link
	def test_ternary_in_return(self, tmp_path: Path) -> None:
		source = """
		int abs_val(int x) {
			return (x >= 0) ? x : (0 - x);
		}
		int main() {
			return abs_val(-17);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 17

	@can_link
	def test_ternary_as_function_arg(self, tmp_path: Path) -> None:
		source = """
		int identity(int x) { return x; }
		int main() {
			int a = 10;
			int b = 20;
			return identity((a > b) ? a : b);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 20


class TestCommaExpression:
	"""Comma expression in for-loop update."""

	@can_link
	def test_comma_for_loop_dual_counters(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int i;
			int j;
			i = 0;
			j = 10;
			for (; i < 5; i = i + 1, j = j - 1) {
			}
			return i + j;
		}
		"""
		# i=5, j=5 => 10
		assert _compile_and_run(source, tmp_path) == 10

	@can_link
	def test_comma_for_loop_product(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int i;
			int j;
			i = 1;
			j = 100;
			for (; i < 4; i = i + 1, j = j - 10) {
			}
			return j;
		}
		"""
		# 3 iterations: j goes 100->90->80->70
		assert _compile_and_run(source, tmp_path) == 70

	@can_link
	def test_comma_expr_evaluates_to_last(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int x;
			x = (1, 2, 42);
			return x;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42


class TestCharOperations:
	"""Char type: comparisons, arithmetic, char-to-int promotion."""

	@can_link
	def test_char_literal_value(self, tmp_path: Path) -> None:
		source = """
		int main() {
			char c = 'A';
			return c;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 65

	@can_link
	def test_char_arithmetic(self, tmp_path: Path) -> None:
		source = """
		int main() {
			char c = 'A';
			char d = c + 3;
			return d;
		}
		"""
		# 'A' + 3 = 68 = 'D'
		assert _compile_and_run(source, tmp_path) == 68

	@can_link
	def test_char_comparison(self, tmp_path: Path) -> None:
		source = """
		int main() {
			char a = 'a';
			char z = 'z';
			if (a < z) {
				return 1;
			}
			return 0;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 1

	@can_link
	def test_char_to_int_promotion(self, tmp_path: Path) -> None:
		source = """
		int main() {
			char c = '0';
			int val = c - 48;
			return val;
		}
		"""
		# '0' is 48, so 48 - 48 = 0
		assert _compile_and_run(source, tmp_path) == 0

	@can_link
	def test_char_digit_to_int(self, tmp_path: Path) -> None:
		source = """
		int main() {
			char c = '7';
			int val = c - '0';
			return val;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 7

	@can_link
	def test_char_case_offset(self, tmp_path: Path) -> None:
		source = """
		int main() {
			char upper = 'A';
			char lower = upper + 32;
			return lower;
		}
		"""
		# 'A'(65) + 32 = 97 = 'a'
		assert _compile_and_run(source, tmp_path) == 97


class TestGlobalVariables:
	"""Global variables used across functions."""

	@can_link
	def test_global_counter_incremented_by_functions(self, tmp_path: Path) -> None:
		source = """
		int counter = 0;
		void inc() {
			counter = counter + 1;
		}
		int main() {
			inc();
			inc();
			inc();
			return counter;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 3

	@can_link
	def test_global_accumulator(self, tmp_path: Path) -> None:
		source = """
		int total = 0;
		void add(int x) {
			total = total + x;
		}
		int main() {
			add(10);
			add(20);
			add(30);
			return total;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 60

	@can_link
	def test_global_read_and_write_from_multiple_functions(self, tmp_path: Path) -> None:
		source = """
		int g = 5;
		int double_g() {
			g = g * 2;
			return g;
		}
		int add_to_g(int x) {
			g = g + x;
			return g;
		}
		int main() {
			double_g();
			add_to_g(3);
			return g;
		}
		"""
		# g=5 -> double_g: g=10 -> add_to_g(3): g=13
		assert _compile_and_run(source, tmp_path) == 13


class TestDoWhileWithBreak:
	"""Do-while with break: verify break exits correctly."""

	@can_link
	def test_do_while_break_on_condition(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int i = 0;
			do {
				if (i == 5) {
					break;
				}
				i = i + 1;
			} while (i < 100);
			return i;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 5

	@can_link
	def test_do_while_break_first_iteration(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int x = 99;
			do {
				break;
				x = 0;
			} while (1);
			return x;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 99

	@can_link
	def test_do_while_break_accumulate(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int sum = 0;
			int i = 1;
			do {
				sum = sum + i;
				i = i + 1;
				if (i > 5) {
					break;
				}
			} while (1);
			return sum;
		}
		"""
		# 1+2+3+4+5 = 15
		assert _compile_and_run(source, tmp_path) == 15


class TestNestedTernary:
	"""Nested ternary: a ? (b ? c : d) : e."""

	@can_link
	def test_nested_ternary_true_true(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int a = 1;
			int b = 1;
			int result = a ? (b ? 10 : 20) : 30;
			return result;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 10

	@can_link
	def test_nested_ternary_true_false(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int a = 1;
			int b = 0;
			int result = a ? (b ? 10 : 20) : 30;
			return result;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 20

	@can_link
	def test_nested_ternary_false(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int a = 0;
			int b = 1;
			int result = a ? (b ? 10 : 20) : 30;
			return result;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 30

	@can_link
	def test_nested_ternary_clamp(self, tmp_path: Path) -> None:
		"""Clamp a value between 0 and 100 using nested ternary."""
		source = """
		int main() {
			int val = 150;
			int clamped = (val < 0) ? 0 : ((val > 100) ? 100 : val);
			return clamped;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 100

	@can_link
	def test_nested_ternary_clamp_low(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int val = 50;
			int clamped = (val < 0) ? 0 : ((val > 100) ? 100 : val);
			return clamped;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 50
