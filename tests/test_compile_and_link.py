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
