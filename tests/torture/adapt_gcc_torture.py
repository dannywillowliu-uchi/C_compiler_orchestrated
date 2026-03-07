"""Re-adapt GCC torture tests with proper dg-directive filtering.

Only includes tests that are:
1. Marked 'dg-do run' (not compile-only or link-only)
2. Not target-restricted (no arm, aarch64, x86-only, etc.)
3. Not requiring special compiler flags (-std=gnu23, etc.)
4. Free of GCC extensions we can't support
5. Self-contained (no #include of GCC test infrastructure headers)
"""

import glob
import re
from pathlib import Path

GCC_TORTURE_DIR = Path("/tmp/gcc-torture-src/gcc/testsuite/gcc.dg/torture")
OUTPUT_DIR = Path("/Users/dannyliu/personal_projects/C_compiler_orchestrated/tests/torture/cases")

# Content patterns that indicate unsupported GCC extensions
SKIP_CONTENT_PATTERNS = [
	r"__builtin_(?!va_)",
	r"__attribute__",
	r"\basm\b",
	r"__asm__",
	r"#pragma",
	r"_Complex",
	r"__int128",
	r"__float128",
	r"__vector",
	r"__extension__",
	r"__typeof__",
	r"\btypeof\b",
	r"__label__",
	r"__SIZEOF_",
	r"__alignof__",
	r"_Alignof",
	r"__atomic_",
	r"__sync_",
	r"_Atomic",
	r"__thread",
	r"_Thread_local",
	r"_Generic",
	r"_Static_assert",
	r"_BitInt",
	r"__float80",
	r"__STDC_VERSION__",
]

# System headers we can't provide
SKIP_INCLUDE_PATTERNS = [
	r"#\s*include\s*<signal\.h>",
	r"#\s*include\s*<setjmp\.h>",
	r"#\s*include\s*<pthread\.h>",
	r"#\s*include\s*<complex\.h>",
	r"#\s*include\s*<fenv\.h>",
	r"#\s*include\s*<math\.h>",
	r"#\s*include\s*<float\.h>",
	r"#\s*include\s*<wchar\.h>",
	r"#\s*include\s*<immintrin\.h>",
]

# dg- directives that make a test incompatible
SKIP_DG_PATTERNS = [
	r"dg-require",
	r"dg-additional-sources",
	r"dg-additional-options.*-m",
	r"dg-additional-options.*-f",
]

SKIP_CONTENT_RE = re.compile("|".join(SKIP_CONTENT_PATTERNS))
SKIP_INCLUDE_RE = re.compile("|".join(SKIP_INCLUDE_PATTERNS))
SKIP_DG_RE = re.compile("|".join(SKIP_DG_PATTERNS))

# Relative includes to GCC test infrastructure (not self-contained)
RELATIVE_INCLUDE_RE = re.compile(r'#\s*include\s*"\.\./')


def get_dg_do(source: str) -> str | None:
	"""Extract the dg-do directive value (run, compile, link, etc.)."""
	m = re.search(r"dg-do\s+(run|compile|link|assemble)", source)
	if m:
		return m.group(1)
	return None


def has_target_restriction(source: str) -> bool:
	"""Check if the test has target-specific restrictions."""
	# dg-do run { target arm*-*-* } or similar
	m = re.search(r"dg-do\s+\w+\s*\{[^}]*target\s+", source)
	if m:
		return True
	# dg-options with architecture flags
	if re.search(r"dg-options.*-march=", source):
		return True
	# dg-options with -std= (language version requirements)
	if re.search(r"dg-options.*-std=", source):
		return True
	return False


def is_compatible(source: str) -> tuple[bool, str]:
	"""Check if a test is compatible. Returns (compatible, reason)."""
	if "main" not in source:
		return False, "no main"

	# Must be a 'dg-do run' test (or have no dg-do, defaulting to run)
	dg_do = get_dg_do(source)
	if dg_do and dg_do != "run":
		return False, f"dg-do {dg_do} (not run)"

	# No target restrictions
	if has_target_restriction(source):
		return False, "target-restricted"

	# No unsupported content
	if SKIP_CONTENT_RE.search(source):
		m = SKIP_CONTENT_RE.search(source)
		return False, f"unsupported: {m.group(0)}"

	# No system includes we can't provide
	if SKIP_INCLUDE_RE.search(source):
		return False, "unsupported system include"

	# No dg- directives that make it incompatible
	if SKIP_DG_RE.search(source):
		return False, "incompatible dg- directive"

	# No relative includes to GCC test infrastructure
	if RELATIVE_INCLUDE_RE.search(source):
		return False, "relative include to GCC infra"

	# No #include of non-standard headers (fp-int-convert.h, etc.)
	# Allow "stdlib.h", "string.h", "stdio.h" style but catch test-infra headers
	infra_include = re.search(r'#\s*include\s*"[^"]*\.h"', source)
	if infra_include:
		return False, f"test-infra include: {infra_include.group(0)}"

	return True, "ok"


def adapt_source(source: str, original_name: str) -> str:
	"""Adapt source: strip includes and dg- directives, keep everything else."""
	lines = source.split("\n")
	out = []

	out.append(f"/* Adapted from gcc.dg/torture/{original_name} */")

	for line in lines:
		# Strip system includes (we declare externs inline)
		if re.match(r"\s*#\s*include\s*<", line):
			continue
		# Strip dg- directive comments
		if re.match(r"\s*/\*.*dg-", line):
			continue
		if re.match(r"\s*//.*dg-", line):
			continue
		out.append(line)

	return "\n".join(out)


def main():
	files = sorted(glob.glob(str(GCC_TORTURE_DIR / "*.c")))
	print(f"Total GCC torture .c files: {len(files)}")

	compatible = []
	rejected = {}
	for fpath in files:
		source = Path(fpath).read_text(errors="replace")
		ok, reason = is_compatible(source)
		if ok:
			compatible.append((Path(fpath).name, source))
		else:
			rejected[reason] = rejected.get(reason, 0) + 1

	print(f"Compatible tests: {len(compatible)}")
	print(f"\nRejection reasons:")
	for reason, count in sorted(rejected.items(), key=lambda x: -x[1]):
		print(f"  {reason}: {count}")

	# Clear old test files
	old_files = list(OUTPUT_DIR.glob("*.c"))
	print(f"\nRemoving {len(old_files)} old test files...")
	for f in old_files:
		f.unlink()

	# Write adapted tests
	written = 0
	for original_name, source in compatible:
		stem = Path(original_name).stem
		safe_name = stem.replace("-", "_")
		out_name = f"gcc_{safe_name}.c"
		adapted = adapt_source(source, original_name)
		(OUTPUT_DIR / out_name).write_text(adapted)
		written += 1

	print(f"Wrote {written} adapted test files to {OUTPUT_DIR}")


if __name__ == "__main__":
	main()
