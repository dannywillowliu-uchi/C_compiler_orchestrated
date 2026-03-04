"""E2E compile-assemble-link-run tests for pointer arithmetic, arrays, and string operations."""

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


class TestPointerArithmeticScaling:
	"""Pointer arithmetic with proper element-size scaling."""

	@can_link
	@pytest.mark.xfail(reason="compiler bug: pointer arithmetic on local array name does not scale by element size")
	def test_int_pointer_plus_offset_deref(self, tmp_path: Path) -> None:
		"""*(arr + 2) reads the third int element correctly."""
		source = """
		int main() {
			int arr[5];
			arr[0] = 10;
			arr[1] = 20;
			arr[2] = 30;
			arr[3] = 40;
			arr[4] = 50;
			return *(arr + 2);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 30

	@can_link
	def test_int_pointer_plus_offset_via_function(self, tmp_path: Path) -> None:
		"""Function receives int* and accesses elements via pointer arithmetic."""
		source = """
		int get_element(int *p, int idx) {
			return *(p + idx);
		}
		int main() {
			int arr[4];
			arr[0] = 5;
			arr[1] = 15;
			arr[2] = 25;
			arr[3] = 35;
			return get_element(arr, 3);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 35

	@can_link
	@pytest.mark.xfail(reason="compiler bug: pointer arithmetic on local array name does not scale by element size")
	def test_pointer_add_commutative(self, tmp_path: Path) -> None:
		"""3 + arr is equivalent to arr + 3 (pointer + int commutativity)."""
		source = """
		int main() {
			int arr[4];
			arr[0] = 1;
			arr[1] = 2;
			arr[2] = 3;
			arr[3] = 77;
			return *(3 + arr);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 77

	@can_link
	@pytest.mark.xfail(reason="compiler bug: arr + N on local int array does not scale N by element size")
	def test_pointer_subtract_offset(self, tmp_path: Path) -> None:
		"""int *p pointing at arr[4]; *(p - 2) reads arr[2]."""
		source = """
		int read_back(int *end, int offset) {
			return *(end - offset);
		}
		int main() {
			int arr[5];
			arr[0] = 1;
			arr[1] = 2;
			arr[2] = 42;
			arr[3] = 4;
			arr[4] = 5;
			return read_back(arr + 4, 2);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	def test_char_pointer_arithmetic_no_scaling(self, tmp_path: Path) -> None:
		"""char* arithmetic advances by 1 byte (sizeof(char) == 1)."""
		source = """
		int main() {
			char buf[4];
			buf[0] = 10;
			buf[1] = 20;
			buf[2] = 30;
			buf[3] = 40;
			return *(buf + 2);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 30


class TestArrayIndexing:
	"""Array indexing with various element types."""

	@can_link
	def test_int_array_read_write(self, tmp_path: Path) -> None:
		"""Basic int array: write and read back values."""
		source = """
		int main() {
			int a[3];
			a[0] = 11;
			a[1] = 22;
			a[2] = 33;
			return a[0] + a[1] + a[2];
		}
		"""
		assert _compile_and_run(source, tmp_path) == 66

	@can_link
	def test_char_array_indexing(self, tmp_path: Path) -> None:
		"""Char array: write individual chars and read back."""
		source = """
		int main() {
			char s[5];
			s[0] = 'H';
			s[1] = 'e';
			s[2] = 'l';
			s[3] = 'l';
			s[4] = 'o';
			return s[0];
		}
		"""
		# 'H' == 72
		assert _compile_and_run(source, tmp_path) == 72

	@can_link
	def test_char_array_sum(self, tmp_path: Path) -> None:
		"""Sum of char values verifies each element is stored/loaded correctly."""
		source = """
		int main() {
			char c[3];
			c[0] = 10;
			c[1] = 20;
			c[2] = 30;
			return c[0] + c[1] + c[2];
		}
		"""
		assert _compile_and_run(source, tmp_path) == 60

	@can_link
	def test_array_index_with_expression(self, tmp_path: Path) -> None:
		"""Array index is a computed expression, not just a literal."""
		source = """
		int main() {
			int arr[6];
			int i;
			for (i = 0; i < 6; i++) {
				arr[i] = i * 10;
			}
			return arr[2 + 3];
		}
		"""
		assert _compile_and_run(source, tmp_path) == 50

	@can_link
	def test_array_write_via_pointer_read_via_index(self, tmp_path: Path) -> None:
		"""Write via pointer arithmetic, read via array indexing."""
		source = """
		void fill(int *p, int n) {
			int i;
			for (i = 0; i < n; i++) {
				*(p + i) = (i + 1) * 5;
			}
		}
		int main() {
			int arr[4];
			fill(arr, 4);
			return arr[0] + arr[1] + arr[2] + arr[3];
		}
		"""
		# 5 + 10 + 15 + 20 = 50
		assert _compile_and_run(source, tmp_path) == 50


class TestMultiDimensionalArrays:
	"""Multi-dimensional array access: a[i][j]."""

	@can_link
	@pytest.mark.xfail(reason="compiler bug: semantic analyzer rejects multi-dimensional array subscripts")
	def test_2d_array_basic_access(self, tmp_path: Path) -> None:
		"""Write and read a 2x3 int array."""
		source = """
		int main() {
			int a[2][3];
			a[0][0] = 1;
			a[0][1] = 2;
			a[0][2] = 3;
			a[1][0] = 4;
			a[1][1] = 5;
			a[1][2] = 6;
			return a[1][2];
		}
		"""
		assert _compile_and_run(source, tmp_path) == 6

	@can_link
	@pytest.mark.xfail(reason="compiler bug: semantic analyzer rejects multi-dimensional array subscripts")
	def test_2d_array_sum_all_elements(self, tmp_path: Path) -> None:
		"""Sum all elements of a 2x3 array."""
		source = """
		int main() {
			int m[2][3];
			int sum;
			int i;
			int j;
			m[0][0] = 1;
			m[0][1] = 2;
			m[0][2] = 3;
			m[1][0] = 4;
			m[1][1] = 5;
			m[1][2] = 6;
			sum = 0;
			for (i = 0; i < 2; i++) {
				for (j = 0; j < 3; j++) {
					sum = sum + m[i][j];
				}
			}
			return sum;
		}
		"""
		# 1+2+3+4+5+6 = 21
		assert _compile_and_run(source, tmp_path) == 21

	@can_link
	@pytest.mark.xfail(reason="compiler bug: semantic analyzer rejects multi-dimensional array subscripts")
	def test_2d_array_row_column_independence(self, tmp_path: Path) -> None:
		"""Verify row/column layout: a[0][2] != a[1][0]."""
		source = """
		int main() {
			int a[2][3];
			a[0][0] = 10;
			a[0][1] = 20;
			a[0][2] = 30;
			a[1][0] = 40;
			a[1][1] = 50;
			a[1][2] = 60;
			return a[0][2] + a[1][0];
		}
		"""
		# 30 + 40 = 70
		assert _compile_and_run(source, tmp_path) == 70

	@can_link
	@pytest.mark.xfail(reason="compiler bug: semantic analyzer rejects multi-dimensional array subscripts")
	def test_3d_array_access(self, tmp_path: Path) -> None:
		"""A 2x2x2 3D array, write and read back a specific element."""
		source = """
		int main() {
			int cube[2][2][2];
			cube[0][0][0] = 1;
			cube[0][0][1] = 2;
			cube[0][1][0] = 3;
			cube[0][1][1] = 4;
			cube[1][0][0] = 5;
			cube[1][0][1] = 6;
			cube[1][1][0] = 7;
			cube[1][1][1] = 8;
			return cube[1][1][0] + cube[0][0][1];
		}
		"""
		# 7 + 2 = 9
		assert _compile_and_run(source, tmp_path) == 9


class TestPointerSubtraction:
	"""Pointer subtraction yields element count, not byte count."""

	@can_link
	@pytest.mark.xfail(reason="compiler bug: arr + N on local int array does not scale N by element size")
	def test_int_pointer_diff_basic(self, tmp_path: Path) -> None:
		"""Subtracting two int pointers gives element count."""
		source = """
		int ptr_diff(int *end, int *start) {
			return end - start;
		}
		int main() {
			int arr[5];
			return ptr_diff(arr + 4, arr);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 4

	@can_link
	@pytest.mark.xfail(reason="compiler bug: arr + N on local int array does not scale N by element size")
	def test_int_pointer_diff_adjacent(self, tmp_path: Path) -> None:
		"""Adjacent int pointers have a difference of 1."""
		source = """
		int ptr_diff(int *a, int *b) {
			return a - b;
		}
		int main() {
			int arr[3];
			return ptr_diff(arr + 1, arr);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 1

	@can_link
	def test_char_pointer_diff(self, tmp_path: Path) -> None:
		"""char pointer difference counts chars (1 byte each)."""
		source = """
		int ptr_diff(char *a, char *b) {
			return a - b;
		}
		int main() {
			char buf[10];
			return ptr_diff(buf + 7, buf);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 7

	@can_link
	@pytest.mark.xfail(reason="compiler bug: arr + N on local int array does not scale N by element size")
	def test_pointer_diff_negative(self, tmp_path: Path) -> None:
		"""Subtracting higher pointer from lower gives negative (wraps in exit code)."""
		source = """
		int ptr_diff(int *a, int *b) {
			return a - b;
		}
		int main() {
			int arr[5];
			int diff;
			diff = ptr_diff(arr + 1, arr + 4);
			return 0 - diff;
		}
		"""
		# diff = -3, return 3
		assert _compile_and_run(source, tmp_path) == 3


class TestStringCharacterAccess:
	"""String character access via pointer and array indexing."""

	@can_link
	def test_string_first_char(self, tmp_path: Path) -> None:
		"""Access first character of a string literal via char pointer."""
		source = """
		int main() {
			char *s = "Hello";
			return *s;
		}
		"""
		# 'H' == 72
		assert _compile_and_run(source, tmp_path) == 72

	@can_link
	def test_string_char_at_offset(self, tmp_path: Path) -> None:
		"""Access character at offset via pointer arithmetic."""
		source = """
		int main() {
			char *s = "ABCDE";
			return *(s + 2);
		}
		"""
		# 'C' == 67
		assert _compile_and_run(source, tmp_path) == 67

	@can_link
	@pytest.mark.xfail(reason="compiler bug: array subscript on pointer variable uses wrong element stride")
	def test_string_char_via_index(self, tmp_path: Path) -> None:
		"""Access string character via array subscript on pointer."""
		source = """
		int main() {
			char *s = "abcdef";
			return s[4];
		}
		"""
		# 'e' == 101
		assert _compile_and_run(source, tmp_path) == 101

	@can_link
	def test_string_null_terminator(self, tmp_path: Path) -> None:
		"""String literal is null-terminated: character after last is 0."""
		source = """
		int main() {
			char *s = "Hi";
			return *(s + 2);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 0

	@can_link
	def test_string_length_via_pointer_walk(self, tmp_path: Path) -> None:
		"""Compute string length by walking pointer until null."""
		source = """
		int strlen_manual(char *s) {
			int len = 0;
			while (*(s + len) != 0) {
				len = len + 1;
			}
			return len;
		}
		int main() {
			return strlen_manual("Hello");
		}
		"""
		assert _compile_and_run(source, tmp_path) == 5


class TestArrayDecayToPointer:
	"""Array names decay to pointers when passed to functions."""

	@can_link
	def test_array_passed_as_pointer_param(self, tmp_path: Path) -> None:
		"""Array decays to pointer when passed to function expecting int*."""
		source = """
		int sum(int *p, int n) {
			int total = 0;
			int i;
			for (i = 0; i < n; i++) {
				total = total + *(p + i);
			}
			return total;
		}
		int main() {
			int arr[3];
			arr[0] = 10;
			arr[1] = 20;
			arr[2] = 30;
			return sum(arr, 3);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 60

	@can_link
	def test_array_decay_modification_in_function(self, tmp_path: Path) -> None:
		"""Function modifies array via pointer; caller sees changes."""
		source = """
		void double_elements(int *p, int n) {
			int i;
			for (i = 0; i < n; i++) {
				*(p + i) = *(p + i) * 2;
			}
		}
		int main() {
			int arr[3];
			arr[0] = 5;
			arr[1] = 10;
			arr[2] = 15;
			double_elements(arr, 3);
			return arr[0] + arr[1] + arr[2];
		}
		"""
		# 10 + 20 + 30 = 60
		assert _compile_and_run(source, tmp_path) == 60

	@can_link
	@pytest.mark.xfail(reason="compiler bug: arr + N on local int array does not scale N by element size")
	def test_array_decay_swap_elements(self, tmp_path: Path) -> None:
		"""Swap two array elements via pointer parameters."""
		source = """
		void swap(int *a, int *b) {
			int tmp = *a;
			*a = *b;
			*b = tmp;
		}
		int main() {
			int arr[3];
			arr[0] = 100;
			arr[1] = 200;
			arr[2] = 300;
			swap(arr + 0, arr + 2);
			return arr[0];
		}
		"""
		# After swap: arr[0]=300, arr[2]=100
		assert _compile_and_run(source, tmp_path) == 300 % 256

	@can_link
	def test_array_decay_bubble_sort(self, tmp_path: Path) -> None:
		"""Bubble sort verifies pointer decay + pointer arithmetic + dereference all work together."""
		source = """
		void swap(int *a, int *b) {
			int tmp = *a;
			*a = *b;
			*b = tmp;
		}
		void sort(int *arr, int n) {
			int i;
			int j;
			for (i = 0; i < n - 1; i++) {
				for (j = 0; j < n - 1 - i; j++) {
					if (*(arr + j) > *(arr + j + 1)) {
						swap(arr + j, arr + j + 1);
					}
				}
			}
		}
		int main() {
			int arr[5];
			arr[0] = 5;
			arr[1] = 3;
			arr[2] = 1;
			arr[3] = 4;
			arr[4] = 2;
			sort(arr, 5);
			return arr[0] + arr[4] * 10;
		}
		"""
		# Sorted: [1,2,3,4,5]. arr[0]=1, arr[4]=5. 1 + 5*10 = 51
		assert _compile_and_run(source, tmp_path) == 51


class TestPointerArithmeticCombined:
	"""Combined pointer/array scenarios exercising multiple features together."""

	@can_link
	def test_reverse_array_via_pointers(self, tmp_path: Path) -> None:
		"""Reverse an array in-place using two pointer parameters."""
		source = """
		void swap(int *a, int *b) {
			int tmp = *a;
			*a = *b;
			*b = tmp;
		}
		void reverse(int *arr, int n) {
			int i;
			for (i = 0; i < n / 2; i++) {
				swap(arr + i, arr + (n - 1 - i));
			}
		}
		int main() {
			int arr[4];
			arr[0] = 10;
			arr[1] = 20;
			arr[2] = 30;
			arr[3] = 40;
			reverse(arr, 4);
			return arr[0];
		}
		"""
		assert _compile_and_run(source, tmp_path) == 40

	@can_link
	def test_find_max_via_pointer_scan(self, tmp_path: Path) -> None:
		"""Find max element by scanning array through pointer."""
		source = """
		int find_max(int *arr, int n) {
			int max;
			int i;
			max = *arr;
			for (i = 1; i < n; i++) {
				if (*(arr + i) > max) {
					max = *(arr + i);
				}
			}
			return max;
		}
		int main() {
			int arr[5];
			arr[0] = 12;
			arr[1] = 45;
			arr[2] = 7;
			arr[3] = 89;
			arr[4] = 23;
			return find_max(arr, 5);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 89

	@can_link
	def test_copy_array_via_pointers(self, tmp_path: Path) -> None:
		"""Copy one array to another using pointer-based function."""
		source = """
		void copy(int *dst, int *src, int n) {
			int i;
			for (i = 0; i < n; i++) {
				*(dst + i) = *(src + i);
			}
		}
		int main() {
			int a[3];
			int b[3];
			a[0] = 11;
			a[1] = 22;
			a[2] = 33;
			copy(b, a, 3);
			return b[0] + b[1] + b[2];
		}
		"""
		assert _compile_and_run(source, tmp_path) == 66

	@can_link
	def test_dot_product_via_pointers(self, tmp_path: Path) -> None:
		"""Dot product of two arrays through pointer parameters."""
		source = """
		int dot(int *a, int *b, int n) {
			int sum = 0;
			int i;
			for (i = 0; i < n; i++) {
				sum = sum + *(a + i) * *(b + i);
			}
			return sum;
		}
		int main() {
			int x[3];
			int y[3];
			x[0] = 1;
			x[1] = 2;
			x[2] = 3;
			y[0] = 4;
			y[1] = 5;
			y[2] = 6;
			return dot(x, y, 3);
		}
		"""
		# 1*4 + 2*5 + 3*6 = 4+10+18 = 32
		assert _compile_and_run(source, tmp_path) == 32
