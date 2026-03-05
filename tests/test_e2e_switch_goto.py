"""End-to-end tests for switch/case with fallthrough and goto interactions.

Tests compile C source through the full pipeline, assemble+link, execute,
and verify exit codes for:
  (1) fallthrough between cases
  (2) goto jumping into/out of switch cases
  (3) switch with enum values
  (4) nested switch statements
  (5) switch with compound expressions
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
# 1. Fallthrough between cases
# ---------------------------------------------------------------------------


class TestSwitchFallthrough:
	@can_link
	def test_fallthrough_accumulates(self, tmp_path: Path) -> None:
		"""Case 1 falls through to case 2, accumulating result."""
		source = """
		int main(void) {
			int x = 1;
			int r = 0;
			switch (x) {
				case 1: r = r + 10;
				case 2: r = r + 20;
				case 3: r = r + 30;
					break;
				default: r = 99;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 60

	@can_link
	def test_fallthrough_from_middle(self, tmp_path: Path) -> None:
		"""Entering at case 2 falls through to case 3."""
		source = """
		int main(void) {
			int x = 2;
			int r = 0;
			switch (x) {
				case 1: r = r + 10;
					break;
				case 2: r = r + 20;
				case 3: r = r + 30;
					break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 50

	@can_link
	def test_no_fallthrough_with_break(self, tmp_path: Path) -> None:
		"""Break prevents fallthrough."""
		source = """
		int main(void) {
			int x = 1;
			int r = 0;
			switch (x) {
				case 1: r = 10; break;
				case 2: r = 20; break;
				case 3: r = 30; break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 10

	@can_link
	def test_fallthrough_to_default(self, tmp_path: Path) -> None:
		"""Case falls through into default."""
		source = """
		int main(void) {
			int x = 2;
			int r = 0;
			switch (x) {
				case 1: r = 10; break;
				case 2: r = 20;
				default: r = r + 5;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 25

	@can_link
	def test_default_only(self, tmp_path: Path) -> None:
		"""No matching case, hits default."""
		source = """
		int main(void) {
			int x = 99;
			int r = 0;
			switch (x) {
				case 1: r = 10; break;
				case 2: r = 20; break;
				default: r = 42;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42


# ---------------------------------------------------------------------------
# 2. Goto jumping into/out of switch cases
# ---------------------------------------------------------------------------


class TestSwitchGotoInteraction:
	@can_link
	def test_goto_out_of_switch(self, tmp_path: Path) -> None:
		"""Goto jumps out of a switch body to a label after it."""
		source = """
		int main(void) {
			int r = 0;
			switch (1) {
				case 1:
					r = 10;
					goto done;
				case 2:
					r = 20;
					break;
			}
			r = 99;
			done:
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 10

	@can_link
	def test_goto_skips_cases(self, tmp_path: Path) -> None:
		"""Goto inside a case skips remaining cases."""
		source = """
		int main(void) {
			int r = 0;
			int x = 1;
			switch (x) {
				case 1:
					r = 5;
					goto skip;
				case 2:
					r = r + 20;
					break;
				case 3:
					r = r + 30;
					break;
			}
			skip:
			r = r + 1;
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 6

	@can_link
	def test_goto_before_switch(self, tmp_path: Path) -> None:
		"""Goto before switch creates a loop-like pattern."""
		source = """
		int main(void) {
			int r = 0;
			int x = 0;
			again:
			switch (x) {
				case 0:
					r = r + 1;
					x = 1;
					goto again;
				case 1:
					r = r + 10;
					break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 11

	@can_link
	def test_goto_between_functions_of_switch(self, tmp_path: Path) -> None:
		"""Goto from one case to a label after the switch, skipping default."""
		source = """
		int main(void) {
			int r = 0;
			switch (3) {
				case 1: r = 10; break;
				case 2: r = 20; break;
				case 3: r = 30; goto end;
				default: r = 99;
			}
			end:
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 30


# ---------------------------------------------------------------------------
# 3. Switch with enum values
# ---------------------------------------------------------------------------


class TestSwitchWithEnum:
	@can_link
	def test_enum_cases_basic(self, tmp_path: Path) -> None:
		"""Switch on int variable with enum constant case labels."""
		source = """
		enum Color { RED, GREEN, BLUE };
		int main(void) {
			int c = GREEN;
			int r = 0;
			switch (c) {
				case RED: r = 10; break;
				case GREEN: r = 20; break;
				case BLUE: r = 30; break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 20

	@can_link
	def test_enum_with_explicit_values(self, tmp_path: Path) -> None:
		"""Enum constants with explicit values in switch."""
		source = """
		enum Status { OK = 0, ERR = 5, WARN = 10 };
		int main(void) {
			int s = ERR;
			int r = 0;
			switch (s) {
				case OK: r = 1; break;
				case ERR: r = 50; break;
				case WARN: r = 100; break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 50

	@can_link
	def test_enum_fallthrough(self, tmp_path: Path) -> None:
		"""Enum cases with fallthrough behavior."""
		source = """
		enum Level { LOW, MED, HIGH };
		int main(void) {
			int l = LOW;
			int r = 0;
			switch (l) {
				case LOW: r = r + 1;
				case MED: r = r + 2;
				case HIGH: r = r + 4;
					break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 7

	@can_link
	def test_enum_with_default(self, tmp_path: Path) -> None:
		"""Enum value not covered by cases falls to default."""
		source = """
		enum Dir { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };
		int main(void) {
			int d = 5;
			int r = 0;
			switch (d) {
				case UP: r = 10; break;
				case DOWN: r = 20; break;
				default: r = 99; break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 99


# ---------------------------------------------------------------------------
# 4. Nested switch statements
# ---------------------------------------------------------------------------


class TestNestedSwitch:
	@can_link
	def test_nested_switch_basic(self, tmp_path: Path) -> None:
		"""Inner switch inside outer switch case."""
		source = """
		int main(void) {
			int a = 1;
			int b = 2;
			int r = 0;
			switch (a) {
				case 1:
					switch (b) {
						case 1: r = 11; break;
						case 2: r = 12; break;
						case 3: r = 13; break;
					}
					break;
				case 2:
					r = 20;
					break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 12

	@can_link
	def test_nested_switch_inner_break_doesnt_exit_outer(self, tmp_path: Path) -> None:
		"""Break in inner switch doesn't exit outer switch; outer fallthrough continues."""
		source = """
		int main(void) {
			int a = 1;
			int b = 1;
			int r = 0;
			switch (a) {
				case 1:
					switch (b) {
						case 1: r = r + 10; break;
						case 2: r = r + 20; break;
					}
					r = r + 5;
					break;
				case 2:
					r = r + 100;
					break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 15

	@can_link
	def test_nested_switch_outer_fallthrough(self, tmp_path: Path) -> None:
		"""Outer case without break falls through after inner switch."""
		source = """
		int main(void) {
			int a = 1;
			int b = 1;
			int r = 0;
			switch (a) {
				case 1:
					switch (b) {
						case 1: r = r + 10; break;
					}
				case 2:
					r = r + 20;
					break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 30

	@can_link
	def test_deeply_nested_switch(self, tmp_path: Path) -> None:
		"""Three levels of nested switch."""
		source = """
		int main(void) {
			int a = 2;
			int b = 1;
			int c = 3;
			int r = 0;
			switch (a) {
				case 1: r = 100; break;
				case 2:
					switch (b) {
						case 1:
							switch (c) {
								case 1: r = 211; break;
								case 2: r = 212; break;
								case 3: r = 213; break;
							}
							break;
						case 2: r = 220; break;
					}
					break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 213


# ---------------------------------------------------------------------------
# 5. Switch with compound expressions
# ---------------------------------------------------------------------------


class TestSwitchCompoundExpr:
	@can_link
	def test_switch_on_arithmetic_expr(self, tmp_path: Path) -> None:
		"""Switch on an arithmetic expression."""
		source = """
		int main(void) {
			int x = 3;
			int y = 2;
			int r = 0;
			switch (x + y) {
				case 4: r = 10; break;
				case 5: r = 20; break;
				case 6: r = 30; break;
				default: r = 99;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 20

	@can_link
	def test_switch_on_modulo(self, tmp_path: Path) -> None:
		"""Switch on modulo expression."""
		source = """
		int main(void) {
			int x = 17;
			int r = 0;
			switch (x % 4) {
				case 0: r = 10; break;
				case 1: r = 20; break;
				case 2: r = 30; break;
				case 3: r = 40; break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 20

	@can_link
	def test_switch_on_bitwise_and(self, tmp_path: Path) -> None:
		"""Switch on bitwise AND expression."""
		source = """
		int main(void) {
			int x = 7;
			int r = 0;
			switch (x & 3) {
				case 0: r = 10; break;
				case 1: r = 20; break;
				case 2: r = 30; break;
				case 3: r = 40; break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 40

	@can_link
	def test_switch_on_ternary(self, tmp_path: Path) -> None:
		"""Switch on result of a ternary expression."""
		source = """
		int main(void) {
			int a = 1;
			int r = 0;
			switch (a ? 2 : 3) {
				case 1: r = 10; break;
				case 2: r = 20; break;
				case 3: r = 30; break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 20

	@can_link
	def test_switch_no_matching_case_no_default(self, tmp_path: Path) -> None:
		"""No matching case and no default leaves result unchanged."""
		source = """
		int main(void) {
			int x = 99;
			int r = 7;
			switch (x) {
				case 1: r = 10; break;
				case 2: r = 20; break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 7

	@can_link
	def test_switch_multiple_stmts_per_case(self, tmp_path: Path) -> None:
		"""Multiple statements within a single case."""
		source = """
		int main(void) {
			int x = 2;
			int r = 0;
			switch (x) {
				case 1:
					r = 1;
					r = r + 1;
					break;
				case 2:
					r = 10;
					r = r * 2;
					r = r + 3;
					break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 23


# ---------------------------------------------------------------------------
# Combined: switch + goto + enum interactions
# ---------------------------------------------------------------------------


class TestSwitchGotoEnumCombined:
	@can_link
	def test_enum_switch_with_goto_exit(self, tmp_path: Path) -> None:
		"""Switch on enum with goto to exit early."""
		source = """
		enum Op { ADD, SUB, MUL };
		int main(void) {
			int op = MUL;
			int r = 0;
			switch (op) {
				case ADD: r = 1; goto done;
				case SUB: r = 2; goto done;
				case MUL: r = 3; goto done;
			}
			r = 99;
			done:
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 3

	@can_link
	def test_nested_switch_with_goto(self, tmp_path: Path) -> None:
		"""Goto from inner nested switch exits both switches."""
		source = """
		int main(void) {
			int a = 1;
			int b = 2;
			int r = 0;
			switch (a) {
				case 1:
					switch (b) {
						case 1: r = 11; break;
						case 2: r = 12; goto end;
						case 3: r = 13; break;
					}
					r = r + 100;
					break;
				case 2:
					r = 20;
					break;
			}
			r = r + 200;
			end:
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 12

	@can_link
	def test_switch_goto_loop_pattern(self, tmp_path: Path) -> None:
		"""Switch inside a goto-based loop pattern (state machine)."""
		source = """
		int main(void) {
			int state = 0;
			int r = 0;
			loop:
			switch (state) {
				case 0:
					r = r + 1;
					state = 1;
					goto loop;
				case 1:
					r = r + 10;
					state = 2;
					goto loop;
				case 2:
					r = r + 100;
					break;
			}
			return r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 111
