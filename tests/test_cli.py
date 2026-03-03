"""Tests for the compiler CLI entry point (__main__.py)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from compiler.__main__ import compile_source

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")

SIMPLE_PROGRAM = "int main() { return 42; }"
TWO_FUNC_PROGRAM = """\
int double_it(int x) {
	return x + x;
}
int main() {
	return double_it(21);
}
"""
INVALID_PROGRAM = "int main() { return x; }"


# --- compile_source() unit tests ---


class TestCompileSource:
	"""Tests for the compile_source() function directly."""

	def test_basic_compilation(self) -> None:
		asm = compile_source(SIMPLE_PROGRAM)
		assert ".section .text" in asm
		assert "main:" in asm
		assert "$42" in asm
		assert "ret" in asm

	def test_multiple_functions(self) -> None:
		asm = compile_source(TWO_FUNC_PROGRAM)
		assert "double_it:" in asm
		assert "main:" in asm
		assert "call double_it" in asm

	def test_optimize_flag(self) -> None:
		source = """\
int main() {
	int x;
	x = 2 + 3;
	return x;
}
"""
		asm_no_opt = compile_source(source, optimize=False)
		asm_opt = compile_source(source, optimize=True)
		# Both should produce valid assembly
		assert "main:" in asm_no_opt
		assert "main:" in asm_opt
		assert "ret" in asm_no_opt
		assert "ret" in asm_opt

	def test_semantic_error_raises(self) -> None:
		from compiler.semantic import SemanticError
		with pytest.raises(SemanticError):
			compile_source(INVALID_PROGRAM)

	def test_preprocessor_runs(self) -> None:
		source = """\
#define VALUE 10
int main() {
	return VALUE;
}
"""
		asm = compile_source(source)
		assert "main:" in asm
		assert "$10" in asm

	def test_with_if_else(self) -> None:
		source = """\
int max(int a, int b) {
	if (a > b) {
		return a;
	} else {
		return b;
	}
}
"""
		asm = compile_source(source)
		assert "max:" in asm
		assert "cmpq" in asm

	def test_with_while_loop(self) -> None:
		source = """\
int sum_to(int n) {
	int s;
	s = 0;
	while (n > 0) {
		s = s + n;
		n = n - 1;
	}
	return s;
}
"""
		asm = compile_source(source)
		assert "sum_to:" in asm
		assert "jmp" in asm

	def test_with_for_loop(self) -> None:
		source = """\
int factorial(int n) {
	int result;
	int i;
	result = 1;
	for (i = 2; i <= n; i = i + 1) {
		result = result * i;
	}
	return result;
}
"""
		asm = compile_source(source)
		assert "factorial:" in asm
		assert "imulq" in asm


# --- CLI integration tests (subprocess) ---


@pytest.fixture()
def tmp_c_file(tmp_path: Path) -> Path:
	"""Create a temp .c file with a simple program."""
	p = tmp_path / "test.c"
	p.write_text(SIMPLE_PROGRAM)
	return p


@pytest.fixture()
def tmp_multi_func_file(tmp_path: Path) -> Path:
	p = tmp_path / "multi.c"
	p.write_text(TWO_FUNC_PROGRAM)
	return p


@pytest.fixture()
def tmp_invalid_file(tmp_path: Path) -> Path:
	p = tmp_path / "bad.c"
	p.write_text(INVALID_PROGRAM)
	return p


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


class TestCLIBasic:
	"""Basic CLI invocation tests."""

	def test_compile_to_stdout(self, tmp_c_file: Path) -> None:
		result = run_compiler(str(tmp_c_file))
		assert result.returncode == 0
		assert "main:" in result.stdout
		assert "$42" in result.stdout
		assert "ret" in result.stdout

	def test_compile_to_output_file(self, tmp_c_file: Path, tmp_path: Path) -> None:
		out = tmp_path / "output.s"
		result = run_compiler(str(tmp_c_file), "-o", str(out))
		assert result.returncode == 0
		assert out.exists()
		asm = out.read_text()
		assert "main:" in asm
		assert "$42" in asm
		assert "ret" in asm

	def test_compile_multiple_functions(self, tmp_multi_func_file: Path) -> None:
		result = run_compiler(str(tmp_multi_func_file))
		assert result.returncode == 0
		assert "double_it:" in result.stdout
		assert "main:" in result.stdout
		assert "call double_it" in result.stdout

	def test_optimize_flag(self, tmp_c_file: Path) -> None:
		result = run_compiler(str(tmp_c_file), "--optimize")
		assert result.returncode == 0
		assert "main:" in result.stdout
		assert "ret" in result.stdout


class TestCLIErrorHandling:
	"""CLI error handling tests."""

	def test_missing_input_file(self) -> None:
		result = run_compiler("nonexistent.c")
		assert result.returncode != 0
		assert "error" in result.stderr.lower()

	def test_semantic_error(self, tmp_invalid_file: Path) -> None:
		result = run_compiler(str(tmp_invalid_file))
		assert result.returncode != 0
		assert "error" in result.stderr.lower()

	def test_no_arguments(self) -> None:
		result = run_compiler()
		assert result.returncode != 0

	def test_help_flag(self) -> None:
		result = run_compiler("--help")
		assert result.returncode == 0
		assert "compiler" in result.stdout.lower() or "usage" in result.stdout.lower()


class TestCLIOutputFile:
	"""Tests for output file handling."""

	def test_output_with_long_flag(self, tmp_c_file: Path, tmp_path: Path) -> None:
		out = tmp_path / "result.s"
		result = run_compiler(str(tmp_c_file), "--output", str(out))
		assert result.returncode == 0
		assert out.exists()
		asm = out.read_text()
		assert "main:" in asm

	def test_output_overwrites_existing(self, tmp_c_file: Path, tmp_path: Path) -> None:
		out = tmp_path / "output.s"
		out.write_text("old content")
		result = run_compiler(str(tmp_c_file), "-o", str(out))
		assert result.returncode == 0
		asm = out.read_text()
		assert "old content" not in asm
		assert "main:" in asm

	def test_stdout_when_no_output_flag(self, tmp_c_file: Path) -> None:
		result = run_compiler(str(tmp_c_file))
		assert result.returncode == 0
		# Should print to stdout, not create a file
		assert ".section .text" in result.stdout
