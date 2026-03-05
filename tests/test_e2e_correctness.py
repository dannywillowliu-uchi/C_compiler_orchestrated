"""End-to-end compile-execute correctness tests for arithmetic, control flow, and type system.

Tests compile C source through the full pipeline, assemble, link, execute,
and verify the exit code matches expected behavior.
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


class TestIntegerPromotionConversion:
	"""Integer promotion and conversion edge cases."""

	@can_link
	def test_char_plus_char_promotes_to_int(self, tmp_path: Path) -> None:
		"""char + char should promote to int, allowing results > 127."""
		source = """
		int main() {
			char a = 100;
			char b = 50;
			int result = a + b;
			return result;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 150

	@can_link
	def test_char_multiplication_promotion(self, tmp_path: Path) -> None:
		"""char * char should promote and not overflow at char boundary."""
		source = """
		int main() {
			char a = 10;
			char b = 12;
			int result = a * b;
			return result;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 120

	@can_link
	def test_unsigned_char_no_sign_extension(self, tmp_path: Path) -> None:
		"""unsigned char 200 should stay 200 when assigned to int."""
		source = """
		int main() {
			unsigned char c = 200;
			int x = c;
			return x;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 200

	@can_link
	def test_signed_char_sign_extension(self, tmp_path: Path) -> None:
		"""Negative signed char should sign-extend to int properly."""
		source = """
		int main() {
			char c = -1;
			int x = c;
			if (x == -1) return 0;
			return 1;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 0

	@can_link
	def test_int_to_char_truncation(self, tmp_path: Path) -> None:
		"""Assigning int 257 to char should truncate to 1."""
		source = """
		int main() {
			int x = 257;
			char c = x;
			return c;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 1

	@can_link
	def test_char_comparison_after_promotion(self, tmp_path: Path) -> None:
		"""Comparing chars should work correctly after promotion."""
		source = """
		int main() {
			char a = 65;
			char b = 66;
			if (a < b) return 0;
			return 1;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 0


class TestNestedStructMemberAccess:
	"""Nested struct member access chains."""

	@can_link
	def test_two_level_nested_struct(self, tmp_path: Path) -> None:
		source = """
		struct Inner { int val; };
		struct Outer { struct Inner inner; int extra; };
		int main() {
			struct Outer o;
			o.inner.val = 42;
			o.extra = 10;
			return o.inner.val;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	def test_three_level_nested_struct(self, tmp_path: Path) -> None:
		source = """
		struct A { int x; };
		struct B { struct A a; int y; };
		struct C { struct B b; int z; };
		int main() {
			struct C c;
			c.b.a.x = 7;
			c.b.y = 14;
			c.z = 21;
			return c.b.a.x + c.b.y + c.z;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	def test_nested_struct_assignment_and_read(self, tmp_path: Path) -> None:
		source = """
		struct Point { int x; int y; };
		struct Rect { struct Point origin; struct Point size; };
		int main() {
			struct Rect r;
			r.origin.x = 1;
			r.origin.y = 2;
			r.size.x = 10;
			r.size.y = 20;
			return r.origin.x + r.origin.y + r.size.x + r.size.y;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 33

	@can_link
	def test_nested_struct_in_computation(self, tmp_path: Path) -> None:
		source = """
		struct Vec2 { int x; int y; };
		struct Line { struct Vec2 start; struct Vec2 end; };
		int main() {
			struct Line l;
			l.start.x = 1;
			l.start.y = 2;
			l.end.x = 4;
			l.end.y = 6;
			int dx = l.end.x - l.start.x;
			int dy = l.end.y - l.start.y;
			return dx + dy;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 7


class TestPointerArithmetic:
	"""Pointer arithmetic with different pointee sizes."""

	@can_link
	def test_int_pointer_increment(self, tmp_path: Path) -> None:
		"""Incrementing int* should advance by sizeof(int)."""
		source = """
		int main() {
			int arr[3];
			arr[0] = 10;
			arr[1] = 20;
			arr[2] = 30;
			int *p = arr;
			p = p + 1;
			return *p;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 20

	@can_link
	def test_char_pointer_increment(self, tmp_path: Path) -> None:
		"""Incrementing char* should advance by 1 byte."""
		source = """
		int main() {
			char arr[4];
			arr[0] = 5;
			arr[1] = 10;
			arr[2] = 15;
			arr[3] = 20;
			char *p = arr;
			p = p + 2;
			return *p;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 15

	@can_link
	def test_pointer_subtraction(self, tmp_path: Path) -> None:
		"""Pointer difference should give element count, not byte count."""
		source = """
		int main() {
			int arr[5];
			int *p1 = &arr[1];
			int *p2 = &arr[4];
			int diff = p2 - p1;
			return diff;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 3

	@can_link
	def test_pointer_to_array_element_write(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int arr[3];
			arr[0] = 0;
			arr[1] = 0;
			arr[2] = 0;
			int *p = arr;
			*(p + 2) = 99;
			return arr[2];
		}
		"""
		assert _compile_and_run(source, tmp_path) == 99

	@can_link
	def test_pointer_decrement(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int arr[3];
			arr[0] = 11;
			arr[1] = 22;
			arr[2] = 33;
			int *p = &arr[2];
			p = p - 2;
			return *p;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 11


class TestSwitchFallthrough:
	"""Switch statement fallthrough behavior."""

	@can_link
	def test_switch_with_break(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int x = 2;
			int result = 0;
			switch (x) {
				case 1: result = 10; break;
				case 2: result = 20; break;
				case 3: result = 30; break;
			}
			return result;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 20

	@can_link
	def test_switch_fallthrough_no_break(self, tmp_path: Path) -> None:
		"""Without break, execution falls through to subsequent cases."""
		source = """
		int main() {
			int x = 1;
			int result = 0;
			switch (x) {
				case 1: result = result + 1;
				case 2: result = result + 2;
				case 3: result = result + 4;
			}
			return result;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 7

	@can_link
	def test_switch_default_case(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int x = 99;
			int result = 0;
			switch (x) {
				case 1: result = 10; break;
				case 2: result = 20; break;
				default: result = 42; break;
			}
			return result;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	def test_switch_fallthrough_into_default(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int x = 3;
			int result = 0;
			switch (x) {
				case 1: result = 1; break;
				case 3: result = 3;
				default: result = result + 10;
			}
			return result;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 13

	@can_link
	def test_switch_multiple_cases_same_body(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int x = 2;
			int result = 0;
			switch (x) {
				case 1:
				case 2:
				case 3:
					result = 50;
					break;
				default:
					result = 0;
					break;
			}
			return result;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 50


class TestComplexForLoops:
	"""Complex for-loop patterns including comma expressions."""

	@can_link
	def test_basic_for_loop_sum(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int sum = 0;
			int i;
			for (i = 1; i <= 10; i = i + 1) {
				sum = sum + i;
			}
			return sum;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 55

	@can_link
	def test_for_loop_multiple_init(self, tmp_path: Path) -> None:
		"""Multiple initializations using comma expression."""
		source = """
		int main() {
			int i;
			int j;
			int result = 0;
			for (i = 0, j = 10; i < j; i = i + 1, j = j - 1) {
				result = result + 1;
			}
			return result;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 5

	@can_link
	def test_nested_for_loops(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int count = 0;
			int i;
			int j;
			for (i = 0; i < 4; i = i + 1) {
				for (j = 0; j < 3; j = j + 1) {
					count = count + 1;
				}
			}
			return count;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 12

	@can_link
	def test_for_loop_with_break(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int i;
			int last = 0;
			for (i = 0; i < 100; i = i + 1) {
				if (i == 7) break;
				last = i;
			}
			return last;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 6

	@can_link
	def test_for_loop_with_continue(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int sum = 0;
			int i;
			for (i = 0; i < 10; i = i + 1) {
				if (i % 2 == 0) continue;
				sum = sum + i;
			}
			return sum;
		}
		"""
		# Sum of odd numbers 1+3+5+7+9 = 25
		assert _compile_and_run(source, tmp_path) == 25

	@can_link
	def test_for_loop_empty_body(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int i;
			for (i = 0; i < 10; i = i + 1) {}
			return i;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 10


class TestFunctionPointerCalls:
	"""Function pointer calls through variables."""

	@can_link
	def test_basic_function_pointer(self, tmp_path: Path) -> None:
		source = """
		int double_it(int x) {
			return x * 2;
		}
		int main() {
			int (*fp)(int) = double_it;
			return fp(21);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	def test_function_pointer_swap(self, tmp_path: Path) -> None:
		"""Call different functions through the same pointer variable."""
		source = """
		int add_one(int x) { return x + 1; }
		int add_two(int x) { return x + 2; }
		int main() {
			int (*fp)(int);
			fp = add_one;
			int a = fp(10);
			fp = add_two;
			int b = fp(10);
			return a + b;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 23

	@can_link
	def test_function_pointer_from_condition(self, tmp_path: Path) -> None:
		source = """
		int square(int x) { return x * x; }
		int negate(int x) { return 0 - x; }
		int main() {
			int flag = 1;
			int (*fp)(int);
			if (flag) {
				fp = square;
			} else {
				fp = negate;
			}
			return fp(5);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 25

	@can_link
	def test_function_pointer_two_args(self, tmp_path: Path) -> None:
		source = """
		int add(int a, int b) { return a + b; }
		int mul(int a, int b) { return a * b; }
		int main() {
			int (*op)(int, int);
			op = add;
			int sum = op(3, 4);
			op = mul;
			int prod = op(3, 4);
			return sum + prod;
		}
		"""
		# 7 + 12 = 19
		assert _compile_and_run(source, tmp_path) == 19

	@can_link
	def test_function_pointer_returning_zero(self, tmp_path: Path) -> None:
		source = """
		int zero(void) { return 0; }
		int main() {
			int (*fp)(void) = zero;
			return fp();
		}
		"""
		assert _compile_and_run(source, tmp_path) == 0
