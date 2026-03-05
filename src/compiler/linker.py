"""Assembler and linker integration for producing executable binaries.

Invokes the system assembler (as) and C compiler (cc/clang/gcc) to convert
generated assembly into object files and executables.
"""

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class ToolchainError(Exception):
	"""Raised when assembler or linker tools are unavailable or fail."""


@dataclass
class ToolchainInfo:
	"""Detected system toolchain configuration."""
	system: str  # "Darwin" or "Linux"
	arch: str  # "arm64", "x86_64", etc.
	assembler: str  # Path to assembler
	cc: str  # Path to C compiler (used as linker)
	cross_compile: bool  # True if we need cross-compilation flags
	extra_as_flags: list[str] = field(default_factory=list)
	extra_cc_flags: list[str] = field(default_factory=list)


def detect_toolchain() -> ToolchainInfo:
	"""Detect available assembler and linker, raising ToolchainError if missing."""
	system = platform.system()
	arch = platform.machine()

	if system == "Darwin":
		cc = shutil.which("cc") or shutil.which("clang")
		if not cc:
			raise ToolchainError(
				"No C compiler found. Install Xcode command-line tools: xcode-select --install"
			)
		# On macOS, use clang as the assembler too (handles cross-arch)
		assembler = cc
		cross_compile = arch != "x86_64"
		extra_as_flags: list[str] = []
		extra_cc_flags: list[str] = []
		if cross_compile:
			extra_as_flags = ["-arch", "x86_64"]
			extra_cc_flags = ["-arch", "x86_64"]
		return ToolchainInfo(
			system=system,
			arch=arch,
			assembler=assembler,
			cc=cc,
			cross_compile=cross_compile,
			extra_as_flags=extra_as_flags,
			extra_cc_flags=extra_cc_flags,
		)
	elif system == "Linux":
		assembler_path = shutil.which("as")
		if not assembler_path:
			raise ToolchainError(
				"GNU assembler (as) not found. Install binutils: sudo apt install binutils"
			)
		cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
		if not cc:
			raise ToolchainError(
				"No C compiler found. Install gcc or clang: sudo apt install gcc"
			)
		return ToolchainInfo(
			system=system,
			arch=arch,
			assembler=assembler_path,
			cc=cc,
			cross_compile=False,
		)
	else:
		raise ToolchainError(f"Unsupported platform: {system}")


def transform_asm_for_macos(asm: str) -> str:
	"""Transform GNU/ELF-style assembly to macOS Mach-O compatible syntax.

	The compiler's codegen emits GNU as syntax (`.section .text`, `.type`,
	`.size`, etc.). macOS's clang assembler requires Mach-O directives and
	underscore-prefixed C symbols.
	"""
	# Collect symbol names that need underscore prefix:
	# 1. Global symbols (from .globl directives)
	# 2. Call targets (static functions still need _ prefix on macOS)
	globals_set: set[str] = set()
	for m in re.finditer(r"\.globl\s+(\w+)", asm):
		globals_set.add(m.group(1))

	# Also collect all call targets so static function labels get _ prefix too
	call_targets: set[str] = set()
	for m in re.finditer(r"\bcall\s+(\w+)\s*$", asm, re.MULTILINE):
		call_targets.add(m.group(1))
	# All symbols that need _ prefix (globals + call targets that have labels)
	all_c_symbols = globals_set | call_targets

	lines = asm.splitlines()
	out: list[str] = []

	for line in lines:
		stripped = line.strip()

		# Remove ELF-only directives
		if re.match(r"\.type\s+\w+,\s*@\w+", stripped):
			continue
		if re.match(r"\.size\s+\w+,\s*\.\s*-\s*\w+", stripped):
			continue

		# Transform section directives
		if stripped == ".section .text":
			out.append(".text")
			continue
		if stripped == ".section .data":
			out.append(".data")
			continue
		if stripped == ".section .bss":
			out.append(".bss")
			continue
		if stripped == ".section .rodata":
			out.append(".section __TEXT,__const")
			continue

		# Prefix .globl symbols with underscore
		m = re.match(r"\.globl\s+(\w+)", stripped)
		if m and m.group(1) in globals_set:
			out.append(f".globl _{m.group(1)}")
			continue

		# Prefix function label definitions (global symbols and call targets)
		m = re.match(r"^(\w+):$", stripped)
		if m and m.group(1) in all_c_symbols:
			out.append(f"_{m.group(1)}:")
			continue

		# Prefix call targets with underscore (all C function calls)
		m = re.match(r"(\s*)call\s+(\w+)$", line)
		if m:
			indent, target = m.group(1), m.group(2)
			out.append(f"{indent}call _{target}")
			continue

		# Prefix lea references to C symbols (e.g. leaq symbol(%rip), %reg)
		m = re.match(r"(\s*lea[qlwb]?\s+)(\w+)(\(%rip\).*)$", line)
		if m and m.group(2) in all_c_symbols:
			out.append(f"{m.group(1)}_{m.group(2)}{m.group(3)}")
			continue

		out.append(line)

	return "\n".join(out) + "\n"


def assemble(
	asm_file: str,
	obj_file: str,
	toolchain: ToolchainInfo | None = None,
) -> None:
	"""Assemble a .s file into a .o object file.

	Args:
		asm_file: Path to input assembly file.
		obj_file: Path to output object file.
		toolchain: Detected toolchain info. Auto-detected if None.

	Raises:
		ToolchainError: If assembler is not available or assembly fails.
		FileNotFoundError: If asm_file does not exist.
	"""
	if not Path(asm_file).exists():
		raise FileNotFoundError(f"Assembly file not found: {asm_file}")

	if toolchain is None:
		toolchain = detect_toolchain()

	if toolchain.system == "Darwin":
		# Use clang to assemble (handles cross-arch on macOS)
		cmd = [toolchain.assembler, "-c", "-x", "assembler", *toolchain.extra_as_flags, "-o", obj_file, asm_file]
	else:
		cmd = [toolchain.assembler, *toolchain.extra_as_flags, "-o", obj_file, asm_file]

	try:
		result = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			timeout=30,
		)
	except FileNotFoundError:
		raise ToolchainError(f"Assembler not found: {toolchain.assembler}")
	except subprocess.TimeoutExpired:
		raise ToolchainError("Assembler timed out")

	if result.returncode != 0:
		stderr = result.stderr.strip()
		raise ToolchainError(f"Assembly failed:\n{stderr}")


def link(
	obj_files: list[str],
	output: str,
	libraries: list[str] | None = None,
	toolchain: ToolchainInfo | None = None,
) -> None:
	"""Link object files into an executable binary.

	Uses the system C compiler (cc/clang/gcc) to link, which automatically
	links libc and the C runtime startup code.

	Args:
		obj_files: List of .o file paths to link.
		output: Path to output executable.
		libraries: Optional list of library names (without -l prefix).
		toolchain: Detected toolchain info. Auto-detected if None.

	Raises:
		ToolchainError: If linker is not available or linking fails.
		FileNotFoundError: If any object file does not exist.
	"""
	for obj in obj_files:
		if not Path(obj).exists():
			raise FileNotFoundError(f"Object file not found: {obj}")

	if toolchain is None:
		toolchain = detect_toolchain()

	cmd = [toolchain.cc, *toolchain.extra_cc_flags, "-o", output, *obj_files]
	if libraries:
		for lib in libraries:
			cmd.append(f"-l{lib}")

	try:
		result = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			timeout=30,
		)
	except FileNotFoundError:
		raise ToolchainError(f"C compiler not found: {toolchain.cc}")
	except subprocess.TimeoutExpired:
		raise ToolchainError("Linker timed out")

	if result.returncode != 0:
		stderr = result.stderr.strip()
		raise ToolchainError(f"Linking failed:\n{stderr}")


def compile_and_link(
	asm_source: str,
	output: str,
	libraries: list[str] | None = None,
	keep_intermediates: bool = False,
	toolchain: ToolchainInfo | None = None,
) -> None:
	"""Assemble and link assembly source into an executable.

	Writes assembly to a temporary .s file, assembles to .o, then links
	to the final executable. Cleans up intermediate files unless requested.

	Args:
		asm_source: Assembly source code string.
		output: Path to output executable.
		libraries: Optional list of library names to link.
		keep_intermediates: If True, keep .s and .o files.
		toolchain: Detected toolchain info. Auto-detected if None.

	Raises:
		ToolchainError: If any tool is unavailable or a step fails.
	"""
	if toolchain is None:
		toolchain = detect_toolchain()

	output_path = Path(output)
	asm_file = str(output_path.with_suffix(".s"))
	obj_file = str(output_path.with_suffix(".o"))

	intermediates = [asm_file, obj_file]

	try:
		# If no main function is defined, add a stub that returns 0
		if not re.search(r"^main:|^_main:", asm_source, re.MULTILINE):
			asm_source += "\n.globl main\nmain:\n\txorl %eax, %eax\n\tret\n"

		# Write assembly (transform for macOS if needed)
		asm_text = transform_asm_for_macos(asm_source) if toolchain.system == "Darwin" else asm_source
		Path(asm_file).write_text(asm_text)

		# Assemble
		assemble(asm_file, obj_file, toolchain=toolchain)

		# Link
		link([obj_file], output, libraries=libraries, toolchain=toolchain)
	finally:
		if not keep_intermediates:
			for f in intermediates:
				try:
					os.remove(f)
				except OSError:
					pass


def compile_to_object(
	asm_source: str,
	output: str,
	keep_asm: bool = False,
	toolchain: ToolchainInfo | None = None,
) -> None:
	"""Assemble assembly source into an object file.

	Args:
		asm_source: Assembly source code string.
		output: Path to output object file.
		keep_asm: If True, keep the intermediate .s file.
		toolchain: Detected toolchain info. Auto-detected if None.

	Raises:
		ToolchainError: If assembler is unavailable or assembly fails.
	"""
	if toolchain is None:
		toolchain = detect_toolchain()

	output_path = Path(output)
	asm_file = str(output_path.with_suffix(".s"))

	try:
		asm_text = transform_asm_for_macos(asm_source) if toolchain.system == "Darwin" else asm_source
		Path(asm_file).write_text(asm_text)
		assemble(asm_file, output, toolchain=toolchain)
	finally:
		if not keep_asm:
			try:
				os.remove(asm_file)
			except OSError:
				pass
