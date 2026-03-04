"""Tests for assembler and linker integration (src/compiler/linker.py and CLI flags)."""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from compiler.__main__ import compile_source
from compiler.linker import (
	ToolchainError,
	assemble,
	compile_and_link,
	compile_to_object,
	detect_toolchain,
	link,
)

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")

SIMPLE_MAIN = "int main() { return 42; }"


def run_compiler(*args: str) -> subprocess.CompletedProcess[str]:
	"""Run `python -m compiler` with the given arguments."""
	env = {**os.environ, "PYTHONPATH": SRC_DIR}
	return subprocess.run(
		[sys.executable, "-m", "compiler", *args],
		capture_output=True,
		text=True,
		timeout=30,
		env=env,
	)


def _can_assemble_x86_64() -> bool:
	"""Check if the system can assemble x86-64 code."""
	try:
		tc = detect_toolchain()
	except ToolchainError:
		return False
	# Quick test: try to assemble a minimal x86-64 file
	import tempfile
	with tempfile.NamedTemporaryFile(suffix=".s", mode="w", delete=False) as f:
		f.write(".section .text\n.globl _test\n_test:\n\tret\n")
		asm_path = f.name
	obj_path = asm_path.replace(".s", ".o")
	try:
		assemble(asm_path, obj_path, toolchain=tc)
		return True
	except (ToolchainError, FileNotFoundError):
		return False
	finally:
		for p in [asm_path, obj_path]:
			try:
				os.remove(p)
			except OSError:
				pass


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


can_assemble = pytest.mark.skipif(
	not _can_assemble_x86_64(),
	reason="x86-64 assembler not available on this platform",
)
can_link = pytest.mark.skipif(
	not _can_link_x86_64(),
	reason="x86-64 linker not available on this platform",
)


# --- Unit tests for detect_toolchain ---


class TestDetectToolchain:
	"""Tests for toolchain detection."""

	def test_detects_system(self) -> None:
		tc = detect_toolchain()
		assert tc.system in ("Darwin", "Linux")

	def test_detects_arch(self) -> None:
		tc = detect_toolchain()
		assert tc.arch in ("x86_64", "arm64", "aarch64")

	def test_has_assembler(self) -> None:
		tc = detect_toolchain()
		assert tc.assembler is not None
		assert Path(tc.assembler).exists() or shutil.which(tc.assembler) is not None

	def test_has_cc(self) -> None:
		tc = detect_toolchain()
		assert tc.cc is not None
		assert Path(tc.cc).exists() or shutil.which(tc.cc) is not None

	def test_cross_compile_set_on_arm64_mac(self) -> None:
		tc = detect_toolchain()
		if tc.system == "Darwin" and tc.arch == "arm64":
			assert tc.cross_compile is True
			assert "-arch" in tc.extra_as_flags
			assert "x86_64" in tc.extra_as_flags
		elif tc.system == "Darwin" and tc.arch == "x86_64":
			assert tc.cross_compile is False

	def test_unsupported_platform(self) -> None:
		with patch("compiler.linker.platform.system", return_value="Windows"):
			with pytest.raises(ToolchainError, match="Unsupported platform"):
				detect_toolchain()

	def test_missing_cc_darwin(self) -> None:
		with patch("compiler.linker.platform.system", return_value="Darwin"), \
			patch("compiler.linker.shutil.which", return_value=None):
			with pytest.raises(ToolchainError, match="No C compiler found"):
				detect_toolchain()

	def test_missing_as_linux(self) -> None:
		with patch("compiler.linker.platform.system", return_value="Linux"), \
			patch("compiler.linker.platform.machine", return_value="x86_64"), \
			patch("compiler.linker.shutil.which", return_value=None):
			with pytest.raises(ToolchainError, match="assembler.*not found"):
				detect_toolchain()


# --- Unit tests for assemble() ---


class TestAssemble:
	"""Tests for the assemble() function."""

	def test_missing_input_file(self, tmp_path: Path) -> None:
		with pytest.raises(FileNotFoundError, match="Assembly file not found"):
			assemble(str(tmp_path / "nonexistent.s"), str(tmp_path / "out.o"))

	@can_assemble
	def test_assemble_produces_object_file(self, tmp_path: Path) -> None:
		asm_file = tmp_path / "test.s"
		obj_file = tmp_path / "test.o"
		tc = detect_toolchain()
		if tc.system == "Darwin":
			asm_file.write_text(".section __TEXT,__text\n.globl _test\n_test:\n\tretq\n")
		else:
			asm_file.write_text(".section .text\n.globl test\ntest:\n\tret\n")
		assemble(str(asm_file), str(obj_file))
		assert obj_file.exists()
		assert obj_file.stat().st_size > 0

	@can_assemble
	def test_assemble_invalid_asm_raises(self, tmp_path: Path) -> None:
		asm_file = tmp_path / "bad.s"
		asm_file.write_text("this is not valid assembly!!!\n%%%garbage%%%\n")
		with pytest.raises(ToolchainError, match="Assembly failed"):
			assemble(str(asm_file), str(tmp_path / "bad.o"))


# --- Unit tests for link() ---


class TestLink:
	"""Tests for the link() function."""

	def test_missing_object_file(self, tmp_path: Path) -> None:
		with pytest.raises(FileNotFoundError, match="Object file not found"):
			link([str(tmp_path / "nonexistent.o")], str(tmp_path / "out"))

	@can_link
	def test_link_produces_executable(self, tmp_path: Path) -> None:
		tc = detect_toolchain()
		asm_file = tmp_path / "main.s"
		obj_file = tmp_path / "main.o"
		exe_file = tmp_path / "main"
		if tc.system == "Darwin":
			asm_file.write_text(
				".section __TEXT,__text\n.globl _main\n_main:\n"
				"\tmovl $0, %eax\n\tretq\n"
			)
		else:
			asm_file.write_text(
				".section .text\n.globl main\nmain:\n"
				"\tmovl $0, %eax\n\tret\n"
			)
		assemble(str(asm_file), str(obj_file), toolchain=tc)
		link([str(obj_file)], str(exe_file), toolchain=tc)
		assert exe_file.exists()
		assert os.access(str(exe_file), os.X_OK)


# --- Unit tests for compile_to_object() ---


class TestCompileToObject:
	"""Tests for compile_to_object()."""

	@can_assemble
	def test_produces_object_file(self, tmp_path: Path) -> None:
		asm = compile_source(SIMPLE_MAIN)
		obj_file = tmp_path / "test.o"
		compile_to_object(asm, str(obj_file))
		assert obj_file.exists()
		assert obj_file.stat().st_size > 0

	@can_assemble
	def test_cleans_intermediate_asm(self, tmp_path: Path) -> None:
		asm = compile_source(SIMPLE_MAIN)
		obj_file = tmp_path / "test.o"
		asm_file = tmp_path / "test.s"
		compile_to_object(asm, str(obj_file))
		assert not asm_file.exists()

	@can_assemble
	def test_keeps_asm_when_requested(self, tmp_path: Path) -> None:
		asm = compile_source(SIMPLE_MAIN)
		obj_file = tmp_path / "test.o"
		asm_file = tmp_path / "test.s"
		compile_to_object(asm, str(obj_file), keep_asm=True)
		assert asm_file.exists()
		assert "main" in asm_file.read_text()


# --- Unit tests for compile_and_link() ---


class TestCompileAndLink:
	"""Tests for compile_and_link()."""

	@can_link
	def test_produces_executable(self, tmp_path: Path) -> None:
		asm = compile_source(SIMPLE_MAIN)
		exe = tmp_path / "test_exe"
		compile_and_link(asm, str(exe))
		assert exe.exists()
		assert os.access(str(exe), os.X_OK)

	@can_link
	def test_cleans_intermediates(self, tmp_path: Path) -> None:
		asm = compile_source(SIMPLE_MAIN)
		exe = tmp_path / "test_exe"
		compile_and_link(asm, str(exe))
		assert not (tmp_path / "test_exe.s").exists()
		assert not (tmp_path / "test_exe.o").exists()

	@can_link
	def test_keeps_intermediates_when_requested(self, tmp_path: Path) -> None:
		asm = compile_source(SIMPLE_MAIN)
		exe = tmp_path / "test_exe"
		compile_and_link(asm, str(exe), keep_intermediates=True)
		assert (tmp_path / "test_exe.s").exists()
		assert (tmp_path / "test_exe.o").exists()

	@can_link
	def test_executable_runs_and_returns_exit_code(self, tmp_path: Path) -> None:
		asm = compile_source(SIMPLE_MAIN)
		exe = tmp_path / "test_exe"
		compile_and_link(asm, str(exe))
		result = subprocess.run([str(exe)], capture_output=True, timeout=10)
		assert result.returncode == 42


# --- CLI integration tests ---


@pytest.fixture()
def tmp_c_file(tmp_path: Path) -> Path:
	p = tmp_path / "test.c"
	p.write_text(SIMPLE_MAIN)
	return p


class TestCLIAsmFlag:
	"""Tests for -S flag (emit assembly only)."""

	def test_s_flag_outputs_assembly_to_stdout(self, tmp_c_file: Path) -> None:
		result = run_compiler("-S", str(tmp_c_file))
		assert result.returncode == 0
		assert "main:" in result.stdout
		assert "$42" in result.stdout

	def test_s_flag_with_output_file(self, tmp_c_file: Path, tmp_path: Path) -> None:
		out = tmp_path / "out.s"
		result = run_compiler("-S", str(tmp_c_file), "-o", str(out))
		assert result.returncode == 0
		assert out.exists()
		assert "main:" in out.read_text()


class TestCLICompileOnlyFlag:
	"""Tests for -c flag (compile to object file)."""

	@can_assemble
	def test_c_flag_produces_object_file(self, tmp_c_file: Path, tmp_path: Path) -> None:
		out = tmp_path / "test.o"
		result = run_compiler("-c", str(tmp_c_file), "-o", str(out))
		assert result.returncode == 0
		assert out.exists()
		assert out.stat().st_size > 0

	@can_assemble
	def test_c_flag_default_output_name(self, tmp_c_file: Path) -> None:
		result = run_compiler("-c", str(tmp_c_file))
		assert result.returncode == 0
		expected = Path("test.o")
		assert expected.exists()
		expected.unlink()


class TestCLILinkFlag:
	"""Tests for default link mode."""

	@can_link
	def test_default_produces_executable(self, tmp_c_file: Path, tmp_path: Path) -> None:
		out = tmp_path / "test_exe"
		result = run_compiler(str(tmp_c_file), "-o", str(out))
		assert result.returncode == 0
		assert out.exists()
		assert os.access(str(out), os.X_OK)

	@can_link
	def test_executable_returns_correct_exit_code(self, tmp_c_file: Path, tmp_path: Path) -> None:
		out = tmp_path / "test_exe"
		result = run_compiler(str(tmp_c_file), "-o", str(out))
		assert result.returncode == 0
		run_result = subprocess.run([str(out)], capture_output=True, timeout=10)
		assert run_result.returncode == 42

	@can_link
	def test_l_flag_accepted(self, tmp_c_file: Path, tmp_path: Path) -> None:
		out = tmp_path / "test_exe"
		# -lm should be accepted (math library, linked but not used)
		result = run_compiler(str(tmp_c_file), "-o", str(out), "-lm")
		assert result.returncode == 0
		assert out.exists()


class TestCLIKeepIntermediates:
	"""Tests for --keep-intermediates flag."""

	@can_link
	def test_keep_intermediates(self, tmp_c_file: Path, tmp_path: Path) -> None:
		out = tmp_path / "test_exe"
		result = run_compiler(str(tmp_c_file), "-o", str(out), "--keep-intermediates")
		assert result.returncode == 0
		assert (tmp_path / "test_exe.s").exists()
		assert (tmp_path / "test_exe.o").exists()

	@can_link
	def test_intermediates_cleaned_by_default(self, tmp_c_file: Path, tmp_path: Path) -> None:
		out = tmp_path / "test_exe"
		result = run_compiler(str(tmp_c_file), "-o", str(out))
		assert result.returncode == 0
		assert not (tmp_path / "test_exe.s").exists()
		assert not (tmp_path / "test_exe.o").exists()


class TestCLIErrorHandling:
	"""Tests for error handling in new modes."""

	def test_missing_input(self) -> None:
		result = run_compiler("nonexistent.c")
		assert result.returncode != 0
		assert "error" in result.stderr.lower()

	def test_s_with_invalid_source(self, tmp_path: Path) -> None:
		bad = tmp_path / "bad.c"
		bad.write_text("int main() { return x; }")
		result = run_compiler("-S", str(bad))
		assert result.returncode != 0
		assert "error" in result.stderr.lower()

	def test_help_shows_new_flags(self) -> None:
		result = run_compiler("--help")
		assert result.returncode == 0
		assert "-S" in result.stdout
		assert "-c" in result.stdout
		assert "-l" in result.stdout


# --- End-to-end compile-assemble-link-run tests ---


def _compile_and_run(source: str, tmp_path: Path) -> int:
	"""Compile C source through full pipeline, link, run, and return exit code."""
	asm = compile_source(source)
	exe = tmp_path / "test_exe"
	compile_and_link(asm, str(exe))
	result = subprocess.run([str(exe)], capture_output=True, timeout=10)
	return result.returncode


class TestE2EFibonacci:
	"""Fibonacci loop computing fib(10)=55."""

	@can_link
	def test_fibonacci_loop_returns_55(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int a = 0;
			int b = 1;
			int temp;
			int i;
			for (i = 0; i < 10; i++) {
				temp = b;
				b = a + b;
				a = temp;
			}
			return a;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 55

	@can_link
	def test_fibonacci_function_returns_55(self, tmp_path: Path) -> None:
		source = """
		int fibonacci(int n) {
			int a = 0;
			int b = 1;
			int temp;
			int i;
			for (i = 0; i < n; i++) {
				temp = b;
				b = a + b;
				a = temp;
			}
			return a;
		}
		int main() {
			return fibonacci(10);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 55


class TestE2EArraySumPointerArithmetic:
	"""Array sum with pointer arithmetic."""

	@can_link
	def test_array_sum_indexing(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int arr[5];
			int sum;
			int i;
			arr[0] = 10;
			arr[1] = 20;
			arr[2] = 30;
			arr[3] = 40;
			arr[4] = 50;
			sum = 0;
			for (i = 0; i < 5; i++) {
				sum = sum + arr[i];
			}
			return sum;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 150

	@can_link
	def test_array_sum_via_function(self, tmp_path: Path) -> None:
		source = """
		int sum_array(int *arr, int n) {
			int total = 0;
			int i;
			for (i = 0; i < n; i++) {
				total = total + *(arr + i);
			}
			return total;
		}
		int main() {
			int arr[4];
			arr[0] = 5;
			arr[1] = 15;
			arr[2] = 25;
			arr[3] = 35;
			return sum_array(arr, 4);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 80


class TestE2ENestedFunctionCalls:
	"""Nested function calls f(g(x))."""

	@can_link
	def test_nested_call_double_then_add(self, tmp_path: Path) -> None:
		source = """
		int double_val(int x) {
			return x * 2;
		}
		int add_ten(int x) {
			return x + 10;
		}
		int main() {
			return add_ten(double_val(7));
		}
		"""
		assert _compile_and_run(source, tmp_path) == 24

	@can_link
	def test_triple_nesting(self, tmp_path: Path) -> None:
		source = """
		int add_one(int x) { return x + 1; }
		int mul_two(int x) { return x * 2; }
		int sub_three(int x) { return x - 3; }
		int main() {
			return sub_three(mul_two(add_one(10)));
		}
		"""
		# add_one(10)=11, mul_two(11)=22, sub_three(22)=19
		assert _compile_and_run(source, tmp_path) == 19

	@can_link
	def test_nested_with_multiple_args(self, tmp_path: Path) -> None:
		source = """
		int add(int a, int b) { return a + b; }
		int mul(int a, int b) { return a * b; }
		int main() {
			return add(mul(3, 4), mul(5, 6));
		}
		"""
		# mul(3,4)=12, mul(5,6)=30, add(12,30)=42
		assert _compile_and_run(source, tmp_path) == 42


class TestE2EStructMemberAccessAndSizeof:
	"""Struct member access and sizeof."""

	@can_link
	def test_struct_member_access(self, tmp_path: Path) -> None:
		source = """
		struct Point {
			int x;
			int y;
		};
		int main() {
			struct Point p;
			p.x = 30;
			p.y = 12;
			return p.x + p.y;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	def test_struct_pointer_member_access(self, tmp_path: Path) -> None:
		source = """
		struct Pair {
			int first;
			int second;
		};
		int sum_pair(struct Pair *p) {
			return p->first + p->second;
		}
		int main() {
			struct Pair pair;
			pair.first = 25;
			pair.second = 75;
			return sum_pair(&pair);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 100

	@can_link
	def test_sizeof_int(self, tmp_path: Path) -> None:
		source = """
		int main() {
			return sizeof(int);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 4

	@can_link
	def test_sizeof_struct(self, tmp_path: Path) -> None:
		source = """
		struct Vec2 {
			int x;
			int y;
		};
		int main() {
			return sizeof(struct Vec2);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 8


class TestE2ESwitchCaseComputed:
	"""Switch/case with computed values."""

	@can_link
	def test_switch_returns_computed_value(self, tmp_path: Path) -> None:
		source = """
		int compute(int op, int a, int b) {
			int result;
			switch (op) {
				case 0:
					result = a + b;
					break;
				case 1:
					result = a - b;
					break;
				case 2:
					result = a * b;
					break;
				default:
					result = 0;
					break;
			}
			return result;
		}
		int main() {
			return compute(2, 7, 8);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 56

	@can_link
	def test_switch_accumulates_via_loop(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int state = 0;
			int acc = 0;
			while (state != 3) {
				switch (state) {
					case 0:
						acc = acc + 10;
						state = 1;
						break;
					case 1:
						acc = acc + 20;
						state = 2;
						break;
					case 2:
						acc = acc + 30;
						state = 3;
						break;
					default:
						state = 3;
						break;
				}
			}
			return acc;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 60

	@can_link
	def test_switch_default_branch(self, tmp_path: Path) -> None:
		source = """
		int classify(int x) {
			int r;
			switch (x) {
				case 1: r = 10; break;
				case 2: r = 20; break;
				case 3: r = 30; break;
				default: r = 99; break;
			}
			return r;
		}
		int main() {
			return classify(7);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 99


class TestE2ELoopsBreakContinue:
	"""While/for loops with break and continue."""

	@can_link
	def test_for_loop_with_break(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int sum = 0;
			int i;
			for (i = 1; i <= 100; i++) {
				if (i > 10) {
					break;
				}
				sum = sum + i;
			}
			return sum;
		}
		"""
		# 1+2+...+10 = 55
		assert _compile_and_run(source, tmp_path) == 55

	@can_link
	def test_for_loop_with_continue(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int sum = 0;
			int i;
			for (i = 1; i <= 10; i++) {
				if (i % 2 == 0) {
					continue;
				}
				sum = sum + i;
			}
			return sum;
		}
		"""
		# 1+3+5+7+9 = 25
		assert _compile_and_run(source, tmp_path) == 25

	@can_link
	def test_while_loop_with_break(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int n = 1;
			while (1) {
				if (n >= 64) {
					break;
				}
				n = n * 2;
			}
			return n;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 64

	@can_link
	def test_nested_loops_with_break(self, tmp_path: Path) -> None:
		source = """
		int main() {
			int count = 0;
			int i;
			int j;
			for (i = 0; i < 5; i++) {
				for (j = 0; j < 5; j++) {
					if (j >= 3) {
						break;
					}
					count = count + 1;
				}
			}
			return count;
		}
		"""
		# 5 outer iterations * 3 inner iterations (break at j>=3) = 15
		assert _compile_and_run(source, tmp_path) == 15
