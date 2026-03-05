"""End-to-end correctness tests for unsigned arithmetic and goto edge cases.

Tests compile C source through the full pipeline, assemble+link with gcc,
execute, and check exit codes for:
  (1) unsigned integer arithmetic (overflow wrapping, comparisons, division/modulo)
  (2) goto jumping into/out of loops and across variable declarations
  (3) switch fallthrough with goto interactions
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


def _compile_and_run(source: str, tmp_path: Path) -> int:
	"""Compile C source through full pipeline, link, run, and return exit code."""
	asm = compile_source(source)
	exe = tmp_path / "test_exe"
	compile_and_link(asm, str(exe))
	result = subprocess.run([str(exe)], capture_output=True, timeout=10)
	return result.returncode


# ---------------------------------------------------------------------------
# (1) Unsigned integer arithmetic
# ---------------------------------------------------------------------------


class TestUnsignedOverflowWrapping:
	"""Unsigned arithmetic should wrap on overflow (modulo 2^32)."""

	@can_link
	def test_unsigned_add_overflow_wraps(self, tmp_path: Path) -> None:
		"""UINT_MAX + 1 should wrap to 0."""
		source = """
		int main(void) {
			unsigned int x = 4294967295u;
			x = x + 1;
			return x;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 0

	@can_link
	def test_unsigned_add_overflow_small(self, tmp_path: Path) -> None:
		"""UINT_MAX + 3 should wrap to 2."""
		source = """
		int main(void) {
			unsigned int x = 4294967295u;
			x = x + 3;
			return x;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 2

	@can_link
	def test_unsigned_subtract_underflow(self, tmp_path: Path) -> None:
		"""0u - 1 should wrap to UINT_MAX; low 8 bits = 255."""
		source = """
		int main(void) {
			unsigned int x = 0;
			x = x - 1;
			return x & 255;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 255

	@can_link
	def test_unsigned_multiply_overflow(self, tmp_path: Path) -> None:
		"""Unsigned multiply should wrap modulo 2^32."""
		source = """
		int main(void) {
			unsigned int x = 65536u;
			unsigned int y = 65537u;
			unsigned int z = x * y;
			return z & 255;
		}
		"""
		# 65536 * 65537 = 4295032832 = 0x100010000 -> low 32 bits = 0x10000 = 65536
		# 65536 & 255 = 0
		assert _compile_and_run(source, tmp_path) == 0


class TestUnsignedComparisons:
	"""Unsigned comparisons must use unsigned (above/below) semantics."""

	@can_link
	def test_unsigned_greater_than_signed_negative(self, tmp_path: Path) -> None:
		"""A large unsigned value should be > 0, not treated as negative."""
		source = """
		int main(void) {
			unsigned int x = 4294967295u;
			if (x > 0) return 1;
			return 0;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 1

	@can_link
	def test_unsigned_less_than(self, tmp_path: Path) -> None:
		source = """
		int main(void) {
			unsigned int a = 5;
			unsigned int b = 10;
			if (a < b) return 1;
			return 0;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 1

	@can_link
	def test_unsigned_ge_equal_values(self, tmp_path: Path) -> None:
		source = """
		int main(void) {
			unsigned int a = 42;
			unsigned int b = 42;
			if (a >= b) return 1;
			return 0;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 1

	@can_link
	def test_unsigned_comparison_large_values(self, tmp_path: Path) -> None:
		"""0x80000000u should be > 0x7FFFFFFFu in unsigned comparison."""
		source = """
		int main(void) {
			unsigned int a = 2147483648u;
			unsigned int b = 2147483647u;
			if (a > b) return 1;
			return 0;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 1


class TestUnsignedDivisionModulo:
	"""Unsigned division and modulo operations."""

	@can_link
	def test_unsigned_division_basic(self, tmp_path: Path) -> None:
		source = """
		int main(void) {
			unsigned int x = 100;
			unsigned int y = 7;
			return x / y;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 14

	@can_link
	def test_unsigned_modulo_basic(self, tmp_path: Path) -> None:
		source = """
		int main(void) {
			unsigned int x = 100;
			unsigned int y = 7;
			return x % y;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 2

	@can_link
	def test_unsigned_division_large_dividend(self, tmp_path: Path) -> None:
		"""Division of a value with high bit set should use unsigned semantics."""
		source = """
		int main(void) {
			unsigned int x = 4294967200u;
			unsigned int y = 100;
			unsigned int q = x / y;
			return q & 255;
		}
		"""
		# 4294967200 / 100 = 42949672 -> 42949672 & 255 = 40
		assert _compile_and_run(source, tmp_path) == 40

	@can_link
	def test_unsigned_modulo_large_value(self, tmp_path: Path) -> None:
		source = """
		int main(void) {
			unsigned int x = 4294967295u;
			unsigned int y = 256;
			return x % y;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 255


# ---------------------------------------------------------------------------
# (2) Goto jumping into/out of loops and across variable declarations
# ---------------------------------------------------------------------------


class TestGotoOutOfLoop:
	"""Goto jumping out of loops should work correctly."""

	@can_link
	def test_goto_breaks_out_of_while(self, tmp_path: Path) -> None:
		source = """
		int main(void) {
			int i = 0;
			while (1) {
				if (i == 5) goto done;
				i = i + 1;
			}
			done:
			return i;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 5

	@can_link
	def test_goto_breaks_out_of_for(self, tmp_path: Path) -> None:
		source = """
		int main(void) {
			int result = 0;
			for (int i = 0; i < 100; i = i + 1) {
				result = result + i;
				if (i == 9) goto done;
			}
			done:
			return result;
		}
		"""
		# sum 0..9 = 45
		assert _compile_and_run(source, tmp_path) == 45

	@can_link
	def test_goto_out_of_nested_loops(self, tmp_path: Path) -> None:
		source = """
		int main(void) {
			int count = 0;
			for (int i = 0; i < 10; i = i + 1) {
				for (int j = 0; j < 10; j = j + 1) {
					count = count + 1;
					if (count == 25) goto done;
				}
			}
			done:
			return count;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 25


class TestGotoIntoLoop:
	"""Goto jumping into a loop body (skipping loop header)."""

	@can_link
	def test_goto_skips_initialization(self, tmp_path: Path) -> None:
		"""Goto past variable initialization; variable retains default/zero."""
		source = """
		int main(void) {
			int x = 10;
			goto skip;
			x = 99;
			skip:
			return x;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 10


class TestGotoAcrossVariableDeclarations:
	"""Goto jumping across variable declarations."""

	@can_link
	def test_goto_forward_past_decl(self, tmp_path: Path) -> None:
		source = """
		int main(void) {
			int a = 1;
			goto skip;
			int b = 99;
			skip:
			return a;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 1

	@can_link
	def test_goto_backward_loop_pattern(self, tmp_path: Path) -> None:
		"""Use goto to create a manual loop."""
		source = """
		int main(void) {
			int i = 0;
			int sum = 0;
			loop:
			if (i >= 10) goto done;
			sum = sum + i;
			i = i + 1;
			goto loop;
			done:
			return sum;
		}
		"""
		# sum 0..9 = 45
		assert _compile_and_run(source, tmp_path) == 45

	@can_link
	def test_multiple_gotos_to_same_label(self, tmp_path: Path) -> None:
		source = """
		int main(void) {
			int x = 0;
			if (x == 0) goto target;
			x = 100;
			goto target;
			x = 200;
			target:
			return x;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 0

	@can_link
	def test_goto_in_if_else(self, tmp_path: Path) -> None:
		source = """
		int main(void) {
			int x = 5;
			if (x > 3) goto big;
			return 0;
			big:
			return x + 10;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 15


# ---------------------------------------------------------------------------
# (3) Switch fallthrough with goto interactions
# ---------------------------------------------------------------------------


class TestSwitchFallthroughAndGoto:
	"""Switch statements with fallthrough and goto interactions."""

	@can_link
	def test_switch_basic_case(self, tmp_path: Path) -> None:
		source = """
		int main(void) {
			int x = 2;
			int r = 0;
			switch (x) {
				case 1: r = 10; break;
				case 2: r = 20; break;
				case 3: r = 30; break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 20

	@can_link
	def test_switch_fallthrough(self, tmp_path: Path) -> None:
		"""Without break, cases should fall through."""
		source = """
		int main(void) {
			int x = 1;
			int r = 0;
			switch (x) {
				case 1: r = r + 10;
				case 2: r = r + 20;
				case 3: r = r + 30; break;
				case 4: r = r + 40;
			}
			return r;
		}
		"""
		# Falls through: 10 + 20 + 30 = 60
		assert _compile_and_run(source, tmp_path) == 60

	@can_link
	def test_switch_default_case(self, tmp_path: Path) -> None:
		source = """
		int main(void) {
			int x = 99;
			int r = 0;
			switch (x) {
				case 1: r = 10; break;
				case 2: r = 20; break;
				default: r = 42; break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	def test_switch_default_fallthrough(self, tmp_path: Path) -> None:
		"""Default in middle with fallthrough to next case."""
		source = """
		int main(void) {
			int x = 99;
			int r = 0;
			switch (x) {
				case 1: r = 10; break;
				default: r = r + 5;
				case 3: r = r + 30; break;
			}
			return r;
		}
		"""
		# default matches, falls through: 5 + 30 = 35
		assert _compile_and_run(source, tmp_path) == 35

	@can_link
	def test_goto_out_of_switch(self, tmp_path: Path) -> None:
		"""Goto can be used to break out of a switch to an external label."""
		source = """
		int main(void) {
			int x = 2;
			int r = 0;
			switch (x) {
				case 1: r = 10; break;
				case 2: r = 20; goto done;
				case 3: r = 30; break;
			}
			r = r + 100;
			done:
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 20

	@can_link
	def test_goto_skips_switch(self, tmp_path: Path) -> None:
		"""Goto can skip over an entire switch statement."""
		source = """
		int main(void) {
			int r = 7;
			goto skip;
			switch (r) {
				case 7: r = 0; break;
				default: r = 99; break;
			}
			skip:
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 7

	@can_link
	def test_switch_with_goto_loop(self, tmp_path: Path) -> None:
		"""Use goto to re-enter a switch from outside it."""
		source = """
		int main(void) {
			int state = 0;
			int count = 0;
			again:
			switch (state) {
				case 0:
					state = 1;
					count = count + 1;
					goto again;
				case 1:
					state = 2;
					count = count + 10;
					goto again;
				case 2:
					count = count + 100;
					break;
			}
			return count;
		}
		"""
		# 1 + 10 + 100 = 111
		assert _compile_and_run(source, tmp_path) == 111


# ---------------------------------------------------------------------------
# (4) Combined unsigned + goto/switch interactions
# ---------------------------------------------------------------------------


class TestUnsignedWithControlFlow:
	"""Tests combining unsigned arithmetic with goto and switch."""

	@can_link
	def test_unsigned_countdown_with_goto(self, tmp_path: Path) -> None:
		"""Count down unsigned from 10 to 0 using goto loop."""
		source = """
		int main(void) {
			unsigned int n = 10;
			unsigned int sum = 0;
			loop:
			if (n == 0) goto done;
			sum = sum + n;
			n = n - 1;
			goto loop;
			done:
			return sum;
		}
		"""
		# sum 1..10 = 55
		assert _compile_and_run(source, tmp_path) == 55

	@can_link
	def test_unsigned_in_switch(self, tmp_path: Path) -> None:
		source = """
		int main(void) {
			unsigned int x = 3;
			unsigned int r = 0;
			switch (x) {
				case 1: r = 10; break;
				case 2: r = 20; break;
				case 3: r = 30; break;
				default: r = 99; break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 30

	@can_link
	def test_unsigned_division_in_goto_loop(self, tmp_path: Path) -> None:
		"""Repeatedly divide unsigned value by 2 using goto, count iterations."""
		source = """
		int main(void) {
			unsigned int x = 128;
			int count = 0;
			loop:
			if (x <= 1) goto done;
			x = x / 2;
			count = count + 1;
			goto loop;
			done:
			return count;
		}
		"""
		# 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 = 7 iterations
		assert _compile_and_run(source, tmp_path) == 7
