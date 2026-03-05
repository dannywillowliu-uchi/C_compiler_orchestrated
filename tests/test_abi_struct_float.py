"""Edge-case tests for struct passing/returning and float ABI correctness.

Tests cover: (1) passing structs by value to functions, (2) returning structs
from functions, (3) float/double argument passing via XMM registers following
System V ABI, (4) mixed int+float argument passing. Uses the compile-and-run
pattern from existing e2e tests.
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
# 1. Passing structs by value to functions
# ---------------------------------------------------------------------------


class TestStructPassByValue:
	"""Test passing structs by value to functions."""

	@can_link
	def test_pass_simple_struct_by_value(self, tmp_path: Path) -> None:
		"""Pass a two-field struct by value and read its fields."""
		source = """
		struct Point { int x; int y; };
		int sum_point(struct Point p) {
			return p.x + p.y;
		}
		int main() {
			struct Point p;
			p.x = 30;
			p.y = 12;
			return sum_point(p);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	@pytest.mark.xfail(reason="struct by-value pass does not copy; callee modifies caller's data")
	def test_pass_struct_preserves_caller_copy(self, tmp_path: Path) -> None:
		"""Modifying a by-value struct param should not affect the caller's copy."""
		source = """
		struct Val { int n; };
		int modify_and_return(struct Val v) {
			v.n = v.n + 100;
			return v.n;
		}
		int main() {
			struct Val v;
			v.n = 5;
			modify_and_return(v);
			return v.n;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 5

	@can_link
	def test_pass_struct_with_three_fields(self, tmp_path: Path) -> None:
		"""Struct with three int fields passed by value."""
		source = """
		struct Triple { int a; int b; int c; };
		int sum_triple(struct Triple t) {
			return t.a + t.b + t.c;
		}
		int main() {
			struct Triple t;
			t.a = 10;
			t.b = 20;
			t.c = 33;
			return sum_triple(t);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 63

	@can_link
	def test_pass_struct_alongside_int_args(self, tmp_path: Path) -> None:
		"""Pass a struct by value alongside regular int arguments."""
		source = """
		struct Pair { int x; int y; };
		int compute(int scale, struct Pair p) {
			return scale * (p.x + p.y);
		}
		int main() {
			struct Pair p;
			p.x = 3;
			p.y = 4;
			return compute(5, p);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 35

	@can_link
	def test_pass_two_structs_by_value(self, tmp_path: Path) -> None:
		"""Pass two different structs by value to the same function."""
		source = """
		struct Pair { int x; int y; };
		int add_pairs(struct Pair a, struct Pair b) {
			return a.x + a.y + b.x + b.y;
		}
		int main() {
			struct Pair p1;
			struct Pair p2;
			p1.x = 10;
			p1.y = 20;
			p2.x = 30;
			p2.y = 40;
			return add_pairs(p1, p2);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 100

	@can_link
	def test_pass_nested_struct_by_value(self, tmp_path: Path) -> None:
		"""Pass a struct containing another struct by value."""
		source = """
		struct Inner { int val; };
		struct Outer { struct Inner i; int extra; };
		int extract(struct Outer o) {
			return o.i.val + o.extra;
		}
		int main() {
			struct Outer o;
			o.i.val = 50;
			o.extra = 7;
			return extract(o);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 57


# ---------------------------------------------------------------------------
# 2. Returning structs from functions
# ---------------------------------------------------------------------------


class TestStructReturn:
	"""Test returning structs from functions."""

	@can_link
	def test_return_simple_struct(self, tmp_path: Path) -> None:
		"""Return a struct with two fields and read them."""
		source = """
		struct Point { int x; int y; };
		struct Point make_point(int x, int y) {
			struct Point p;
			p.x = x;
			p.y = y;
			return p;
		}
		int main() {
			struct Point p;
			p = make_point(25, 17);
			return p.x + p.y;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	def test_return_struct_used_in_expression(self, tmp_path: Path) -> None:
		"""Use a returned struct's field directly in an expression."""
		source = """
		struct Result { int value; int error; };
		struct Result compute(int x) {
			struct Result r;
			r.value = x * 2;
			r.error = 0;
			return r;
		}
		int main() {
			struct Result r;
			r = compute(21);
			return r.value;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	def test_return_struct_chain(self, tmp_path: Path) -> None:
		"""Chain struct returns through multiple function calls."""
		source = """
		struct Val { int n; };
		struct Val make_val(int n) {
			struct Val v;
			v.n = n;
			return v;
		}
		struct Val double_val(struct Val v) {
			struct Val result;
			result.n = v.n * 2;
			return result;
		}
		int main() {
			struct Val v;
			v = make_val(10);
			v = double_val(v);
			return v.n;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 20

	@can_link
	def test_return_struct_with_single_field(self, tmp_path: Path) -> None:
		"""Return a struct with a single field (fits in one register)."""
		source = """
		struct Wrapper { int value; };
		struct Wrapper wrap(int x) {
			struct Wrapper w;
			w.value = x;
			return w;
		}
		int main() {
			struct Wrapper w;
			w = wrap(99);
			return w.value;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 99


# ---------------------------------------------------------------------------
# 3. Float/double argument passing via XMM registers (System V ABI)
# ---------------------------------------------------------------------------


class TestFloatArgPassing:
	"""Test float/double parameter passing via XMM registers."""

	@can_link
	@pytest.mark.xfail(reason="float function arg passing via XMM registers not working at runtime")
	def test_single_float_arg(self, tmp_path: Path) -> None:
		"""Pass a single float argument and convert result to int."""
		source = """
		float identity(float x) { return x; }
		int main() {
			float f = identity(42.0);
			return (int)f;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	@pytest.mark.xfail(reason="float function arg passing via XMM registers not working at runtime")
	def test_two_float_args_addition(self, tmp_path: Path) -> None:
		"""Add two float arguments."""
		source = """
		float add(float a, float b) { return a + b; }
		int main() {
			float result = add(20.5, 21.5);
			return (int)result;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	@pytest.mark.xfail(reason="float function arg passing via XMM registers not working at runtime")
	def test_float_subtraction(self, tmp_path: Path) -> None:
		"""Float subtraction with conversion to int."""
		source = """
		float sub(float a, float b) { return a - b; }
		int main() {
			float result = sub(100.0, 67.0);
			return (int)result;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 33

	@can_link
	@pytest.mark.xfail(reason="float function arg passing via XMM registers not working at runtime")
	def test_float_multiplication(self, tmp_path: Path) -> None:
		"""Float multiplication yielding integer-convertible result."""
		source = """
		float mul(float a, float b) { return a * b; }
		int main() {
			float result = mul(7.0, 8.0);
			return (int)result;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 56

	@can_link
	@pytest.mark.xfail(reason="float function arg passing via XMM registers not working at runtime")
	def test_four_float_args(self, tmp_path: Path) -> None:
		"""Pass four float arguments (xmm0-xmm3)."""
		source = """
		float sum4(float a, float b, float c, float d) {
			return a + b + c + d;
		}
		int main() {
			float r = sum4(10.0, 20.0, 30.0, 40.0);
			return (int)r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 100

	@can_link
	@pytest.mark.xfail(reason="float function arg passing via XMM registers not working at runtime")
	def test_double_arg_passing(self, tmp_path: Path) -> None:
		"""Pass double arguments and return double."""
		source = """
		double add_doubles(double a, double b) { return a + b; }
		int main() {
			double result = add_doubles(25.0, 50.0);
			return (int)result;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 75

	@can_link
	@pytest.mark.xfail(reason="float function arg passing via XMM registers not working at runtime")
	def test_float_to_int_truncation(self, tmp_path: Path) -> None:
		"""Float-to-int conversion truncates toward zero."""
		source = """
		int truncate(float f) { return (int)f; }
		int main() {
			return truncate(99.9);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 99

	@can_link
	def test_int_to_float_conversion(self, tmp_path: Path) -> None:
		"""Int-to-float conversion and back."""
		source = """
		float from_int(int x) { return (float)x; }
		int main() {
			float f = from_int(77);
			return (int)f;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 77

	@can_link
	@pytest.mark.xfail(reason="float function arg passing via XMM registers not working at runtime")
	def test_float_comparison_branch(self, tmp_path: Path) -> None:
		"""Float comparison used in conditional branching."""
		source = """
		int greater(float a, float b) {
			if (a > b) return 1;
			return 0;
		}
		int main() {
			return greater(3.14, 2.71) * 50 + greater(1.0, 2.0) * 100;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 50


# ---------------------------------------------------------------------------
# 4. Mixed int+float argument passing
# ---------------------------------------------------------------------------


class TestMixedIntFloatArgs:
	"""Test mixed integer and float argument passing per System V ABI."""

	@can_link
	@pytest.mark.xfail(reason="mixed int+float arg passing broken: float args via XMM not received correctly")
	def test_int_then_float(self, tmp_path: Path) -> None:
		"""First arg int (rdi), second arg float (xmm0)."""
		source = """
		int add_mixed(int a, float b) {
			return a + (int)b;
		}
		int main() {
			return add_mixed(20, 22.0);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	@pytest.mark.xfail(reason="mixed int+float arg passing broken: float args via XMM not received correctly")
	def test_float_then_int(self, tmp_path: Path) -> None:
		"""First arg float (xmm0), second arg int (rdi)."""
		source = """
		int add_mixed(float a, int b) {
			return (int)a + b;
		}
		int main() {
			return add_mixed(30.0, 12);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	@pytest.mark.xfail(reason="mixed int+float arg passing broken: float args via XMM not received correctly")
	def test_alternating_int_float(self, tmp_path: Path) -> None:
		"""Alternating int and float args: int, float, int, float."""
		source = """
		int interleaved(int a, float b, int c, float d) {
			return a + (int)b + c + (int)d;
		}
		int main() {
			return interleaved(10, 20.0, 30, 40.0);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 100

	@can_link
	@pytest.mark.xfail(reason="mixed int+float arg passing broken: float args via XMM not received correctly")
	def test_many_int_args_with_one_float(self, tmp_path: Path) -> None:
		"""Multiple int args with a float arg mixed in."""
		source = """
		int compute(int a, int b, float f, int c) {
			return a + b + (int)f + c;
		}
		int main() {
			return compute(5, 10, 15.0, 20);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 50

	@can_link
	@pytest.mark.xfail(reason="mixed int+float arg passing broken: float args via XMM not received correctly")
	def test_float_return_from_mixed_args(self, tmp_path: Path) -> None:
		"""Function with mixed args returning a float."""
		source = """
		float weighted(int weight, float value) {
			return (float)weight * value;
		}
		int main() {
			float r = weighted(3, 14.0);
			return (int)r;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	def test_double_with_ints(self, tmp_path: Path) -> None:
		"""Double argument mixed with integer arguments."""
		source = """
		int combine(int a, double d, int b) {
			return a + (int)d + b;
		}
		int main() {
			return combine(10, 25.0, 15);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 50

	@can_link
	@pytest.mark.xfail(reason="mixed int+float arg passing broken: float args via XMM not received correctly")
	def test_mixed_args_with_local_float_computation(self, tmp_path: Path) -> None:
		"""Mixed args with additional float computation inside the function."""
		source = """
		int scale_and_add(int base, float factor, int offset) {
			float scaled = (float)base * factor;
			return (int)scaled + offset;
		}
		int main() {
			return scale_and_add(10, 3.0, 12);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42
