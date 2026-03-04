#!/usr/bin/env python3
"""Torture test runner for the C compiler.

Usage:
	PYTHONPATH=src .venv/bin/python tests/torture/runner.py [OPTIONS]

Options:
	--baseline PATH    Path to previous results JSON (default: tests/torture/baseline.json)
	--output PATH      Path to write new results JSON (default: tests/torture/results.json)
	--state-file PATH  Path to write TORTURE_RESULTS.md (default: TORTURE_RESULTS.md in repo root)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PREFIX_CATEGORY_MAP = {
	"call_": "calling_convention",
	"decl_": "declarations",
	"struct_": "struct_operations",
	"arith_": "arithmetic",
	"ptr_": "pointers_arrays",
	"flow_": "control_flow",
	"va_": "variadic",
	"misc_": "misc",
}

# Content-based category detection for GCC-sourced tests
CONTENT_CATEGORY_RULES = [
	("struct ", "struct_operations"),
	("union ", "struct_operations"),
	("->", "pointers_arrays"),
	("*p ", "pointers_arrays"),
	("*p)", "pointers_arrays"),
	("*p;", "pointers_arrays"),
	("&x", "pointers_arrays"),
	("&a", "pointers_arrays"),
	("switch", "control_flow"),
	("goto ", "control_flow"),
	("for (", "control_flow"),
	("for(", "control_flow"),
	("while (", "control_flow"),
	("while(", "control_flow"),
	("va_list", "variadic"),
	("va_arg", "variadic"),
	("<<", "arithmetic"),
	(">>", "arithmetic"),
	("unsigned", "arithmetic"),
]


def categorize(test_name: str, source: str = "") -> str:
	"""Categorize a test by prefix first, then by source content."""
	for prefix, category in PREFIX_CATEGORY_MAP.items():
		if test_name.startswith(prefix):
			return category

	if source:
		# Count hits per category from content
		hits: dict[str, int] = {}
		for pattern, cat in CONTENT_CATEGORY_RULES:
			if pattern in source:
				hits[cat] = hits.get(cat, 0) + 1
		if hits:
			return max(hits, key=hits.get)

	return "misc"


def run_single_test(c_file: Path) -> dict:
	"""Run a single torture test case and return its result."""
	from compiler.__main__ import compile_source
	from compiler.linker import compile_and_link

	test_name = c_file.stem
	source = c_file.read_text()
	category = categorize(test_name, source)

	# Step 1: Try to compile source to assembly
	try:
		asm = compile_source(source)
	except Exception as e:
		return {
			"status": "SKIP",
			"category": category,
			"error_msg": f"compile_source failed: {e}",
			"exit_code": None,
		}

	# Step 2: Try to link assembly into an executable
	exe_path = None
	try:
		fd, exe_path = tempfile.mkstemp(prefix=f"torture_{test_name}_", suffix="")
		os.close(fd)
		try:
			compile_and_link(asm, exe_path)
		except Exception as e:
			return {
				"status": "SKIP",
				"category": category,
				"error_msg": f"compile_and_link failed: {e}",
				"exit_code": None,
			}

		# Step 3: Execute the compiled binary
		try:
			result = subprocess.run(
				[exe_path],
				timeout=5,
				capture_output=True,
			)
		except subprocess.TimeoutExpired:
			return {
				"status": "FAIL",
				"category": category,
				"error_msg": "timeout",
				"exit_code": None,
			}

		if result.returncode < 0:
			return {
				"status": "FAIL",
				"category": category,
				"error_msg": f"crash (signal {-result.returncode})",
				"exit_code": result.returncode,
			}
		elif result.returncode == 0:
			return {
				"status": "PASS",
				"category": category,
				"error_msg": None,
				"exit_code": 0,
			}
		else:
			return {
				"status": "FAIL",
				"category": category,
				"error_msg": f"wrong_output (exit code {result.returncode})",
				"exit_code": result.returncode,
			}
	finally:
		if exe_path and os.path.exists(exe_path):
			os.unlink(exe_path)


def compute_deltas(current_tests: dict, baseline_tests: dict) -> dict:
	"""Compute deltas between current and baseline results."""
	newly_passing = []
	newly_failing = []
	newly_compiling = []

	for name, result in sorted(current_tests.items()):
		prev = baseline_tests.get(name)
		if prev is None:
			continue

		prev_status = prev.get("status")
		cur_status = result["status"]

		if cur_status == "PASS" and prev_status in ("FAIL", "SKIP"):
			newly_passing.append(name)
		elif cur_status == "FAIL" and prev_status == "PASS":
			newly_failing.append(name)
		elif prev_status == "SKIP" and cur_status in ("FAIL", "PASS"):
			newly_compiling.append(name)

	return {
		"newly_passing": sorted(newly_passing),
		"newly_failing": sorted(newly_failing),
		"newly_compiling": sorted(newly_compiling),
	}


def write_results_json(output_path: Path, summary: dict, tests: dict, deltas: dict) -> None:
	"""Write the results JSON file."""
	data = {
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"summary": summary,
		"tests": {k: tests[k] for k in sorted(tests)},
		"deltas": deltas,
	}
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(data, indent=2) + "\n")


def write_state_markdown(state_path: Path, summary: dict, tests: dict, deltas: dict) -> None:
	"""Write the TORTURE_RESULTS.md file."""
	total = summary["total"]
	passed = summary["passed"]
	failed = summary["failed"]
	skipped = summary["skipped"]
	pass_rate = (passed / total * 100) if total > 0 else 0.0

	lines = [
		"# Torture Test Results",
		"",
		f"**Score: PASS: {passed}/{total} | FAIL: {failed}/{total} | SKIP: {skipped}/{total}** ({pass_rate:.1f}% pass rate)",
		"",
	]

	# Delta section
	lines.append("## Delta vs Baseline")
	if not any(deltas.values()):
		lines.append("No changes vs baseline.")
	else:
		if deltas["newly_passing"]:
			names = ", ".join(deltas["newly_passing"])
			lines.append(f"- +{len(deltas['newly_passing'])} newly passing: {names}")
		if deltas["newly_failing"]:
			names = ", ".join(deltas["newly_failing"])
			lines.append(f"- -{len(deltas['newly_failing'])} regression: {names}")
		if deltas["newly_compiling"]:
			names = ", ".join(deltas["newly_compiling"])
			lines.append(f"- +{len(deltas['newly_compiling'])} newly compiling (now testable): {names}")
	lines.append("")

	# Group tests by category
	failures_by_cat: dict[str, list] = {}
	skipped_by_cat: dict[str, list] = {}
	for name in sorted(tests):
		t = tests[name]
		if t["status"] == "FAIL":
			failures_by_cat.setdefault(t["category"], []).append((name, t))
		elif t["status"] == "SKIP":
			skipped_by_cat.setdefault(t["category"], []).append(name)

	# Failures by category
	lines.append("## Failures by Category")
	if not failures_by_cat:
		lines.append("No failures!")
	else:
		# Sort categories by failure count descending
		sorted_cats = sorted(failures_by_cat.items(), key=lambda x: -len(x[1]))
		for cat, fail_list in sorted_cats:
			lines.append("")
			lines.append(f"### {cat} ({len(fail_list)} failure{'s' if len(fail_list) != 1 else ''})")
			lines.append("| Test | Error | Detail |")
			lines.append("|------|-------|--------|")
			for name, t in fail_list:
				error_type = "timeout" if t["error_msg"] == "timeout" else (
					"crash" if "crash" in (t["error_msg"] or "") else "wrong_output"
				)
				detail = t["error_msg"] or ""
				lines.append(f"| {name} | {error_type} | {detail} |")
	lines.append("")

	# Skipped tests
	lines.append("## Skipped Tests (not yet compilable)")
	if not skipped_by_cat:
		lines.append("All tests are compilable!")
	else:
		for cat in sorted(skipped_by_cat):
			names = ", ".join(skipped_by_cat[cat])
			lines.append(f"### {cat}")
			lines.append(names)
			lines.append("")
	lines.append("")

	# Recommended priorities
	lines.append("## Recommended Priorities")
	if failures_by_cat:
		sorted_cats = sorted(failures_by_cat.items(), key=lambda x: -len(x[1]))
		for i, (cat, fail_list) in enumerate(sorted_cats, 1):
			lines.append(f"{i}. **{cat}** - {len(fail_list)} failure{'s' if len(fail_list) != 1 else ''}")
	else:
		lines.append("No failures to prioritize.")
	lines.append("")

	state_path.write_text("\n".join(lines))


def main() -> None:
	parser = argparse.ArgumentParser(description="Torture test runner for the C compiler")
	parser.add_argument(
		"--baseline",
		type=Path,
		default=REPO_ROOT / "tests" / "torture" / "baseline.json",
		help="Path to previous results JSON",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=REPO_ROOT / "tests" / "torture" / "results.json",
		help="Path to write new results JSON",
	)
	parser.add_argument(
		"--state-file",
		type=Path,
		default=REPO_ROOT / "TORTURE_RESULTS.md",
		help="Path to write TORTURE_RESULTS.md",
	)
	args = parser.parse_args()

	# Discover test cases
	cases_dir = REPO_ROOT / "tests" / "torture" / "cases"
	c_files = sorted(cases_dir.glob("*.c")) if cases_dir.is_dir() else []

	# Run all tests
	tests: dict[str, dict] = {}
	for c_file in c_files:
		test_name = c_file.stem
		tests[test_name] = run_single_test(c_file)

	# Compute summary
	total = len(tests)
	passed = sum(1 for t in tests.values() if t["status"] == "PASS")
	failed = sum(1 for t in tests.values() if t["status"] == "FAIL")
	skipped = sum(1 for t in tests.values() if t["status"] == "SKIP")
	summary = {"total": total, "passed": passed, "failed": failed, "skipped": skipped}

	# Load baseline and compute deltas
	baseline_tests: dict = {}
	if args.baseline.is_file():
		try:
			baseline_data = json.loads(args.baseline.read_text())
			baseline_tests = baseline_data.get("tests", {})
		except (json.JSONDecodeError, KeyError):
			pass

	deltas = compute_deltas(tests, baseline_tests)

	# Write outputs
	write_results_json(args.output, summary, tests, deltas)
	write_state_markdown(args.state_file, summary, tests, deltas)

	# Print one-line summary
	print(f"Torture tests: {passed}/{total} passed, {failed} failed, {skipped} skipped")


if __name__ == "__main__":
	main()
