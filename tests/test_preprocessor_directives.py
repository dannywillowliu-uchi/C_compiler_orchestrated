"""Tests for #error, #warning, #line, and #pragma once directives."""

import textwrap

import pytest

from compiler.preprocessor import Preprocessor, PreprocessorError


def preprocess(source: str, **kwargs) -> str:
	pp = Preprocessor(**kwargs)
	return pp.process(textwrap.dedent(source))


# ── #error ────────────────────────────────────────────────────────────────────


class TestErrorDirective:
	def test_error_raises_with_message(self):
		with pytest.raises(PreprocessorError, match="#error stop here"):
			preprocess("#error stop here")

	def test_error_inside_false_ifdef_is_skipped(self):
		source = """\
			#define FOO
			#ifndef FOO
			#error should not fire
			#endif
			int x;
		"""
		result = preprocess(source)
		assert "int x;" in result

	def test_error_inside_true_ifdef_fires(self):
		source = """\
			#define FOO
			#ifdef FOO
			#error should fire
			#endif
		"""
		with pytest.raises(PreprocessorError, match="#error should fire"):
			preprocess(source)

	def test_error_preserves_full_message(self):
		with pytest.raises(PreprocessorError, match="#error this is a detailed message"):
			preprocess('#error this is a detailed message')


# ── #warning ──────────────────────────────────────────────────────────────────


class TestWarningDirective:
	def test_warning_collects_warning(self):
		pp = Preprocessor()
		pp.process("#warning watch out")
		assert len(pp.warnings) == 1
		assert "#warning watch out" in pp.warnings[0]

	def test_warning_continues_processing(self):
		pp = Preprocessor()
		result = pp.process("int a;\n#warning careful\nint b;")
		assert "int a;" in result
		assert "int b;" in result
		assert len(pp.warnings) == 1

	def test_warning_in_false_branch_is_skipped(self):
		source = textwrap.dedent("""\
			#ifdef NOPE
			#warning should not appear
			#endif
			int x;
		""")
		pp = Preprocessor()
		result = pp.process(source)
		assert len(pp.warnings) == 0
		assert "int x;" in result

	def test_multiple_warnings(self):
		source = textwrap.dedent("""\
			#warning first
			#warning second
		""")
		pp = Preprocessor()
		pp.process(source)
		assert len(pp.warnings) == 2
		assert "#warning first" in pp.warnings[0]
		assert "#warning second" in pp.warnings[1]


# ── #line ─────────────────────────────────────────────────────────────────────


class TestLineDirective:
	def test_line_changes_line_value(self):
		source = textwrap.dedent("""\
			#line 100
			int x = __LINE__;
		""")
		pp = Preprocessor()
		result = pp.process(source)
		assert "int x = 100;" in result

	def test_line_with_filename_changes_file(self):
		source = textwrap.dedent("""\
			#line 50 "myfile.c"
			char *f = __FILE__;
		""")
		pp = Preprocessor()
		result = pp.process(source)
		assert '"myfile.c"' in result

	def test_line_with_filename_changes_both(self):
		source = textwrap.dedent("""\
			#line 200 "other.h"
			int line = __LINE__;
			char *file = __FILE__;
		""")
		pp = Preprocessor()
		result = pp.process(source)
		assert "int line = 200;" in result
		assert '"other.h"' in result

	def test_line_increments_after_directive(self):
		source = textwrap.dedent("""\
			#line 10
			int a = __LINE__;
			int b = __LINE__;
		""")
		pp = Preprocessor()
		result = pp.process(source)
		assert "int a = 10;" in result
		assert "int b = 11;" in result

	def test_line_affects_error_location(self):
		source = textwrap.dedent("""\
			#line 500 "generated.c"
			#error boom
		""")
		with pytest.raises(PreprocessorError, match="generated.c:500"):
			pp = Preprocessor()
			pp.process(source)

	def test_line_affects_warning_location(self):
		source = textwrap.dedent("""\
			#line 42 "header.h"
			#warning check this
		""")
		pp = Preprocessor()
		pp.process(source)
		assert "header.h:42" in pp.warnings[0]


# ── #pragma once ──────────────────────────────────────────────────────────────


class TestPragmaOnce:
	def test_pragma_once_prevents_double_inclusion(self, tmp_path):
		header = tmp_path / "header.h"
		header.write_text("#pragma once\nint HEADER_VAR;")

		main_source = f'#include "{header}"\n#include "{header}"\nint x;'
		pp = Preprocessor()
		# Reset _included_files so second include relies on #pragma once
		result = pp.process(main_source, str(tmp_path / "main.c"))
		lines = [line.strip() for line in result.split("\n") if line.strip()]
		# HEADER_VAR should appear only once
		count = sum(1 for line in lines if "HEADER_VAR" in line)
		assert count == 1

	def test_pragma_once_allows_different_files(self, tmp_path):
		h1 = tmp_path / "a.h"
		h1.write_text("#pragma once\nint A_VAR;")
		h2 = tmp_path / "b.h"
		h2.write_text("#pragma once\nint B_VAR;")

		source = f'#include "{h1}"\n#include "{h2}"\n'
		pp = Preprocessor()
		result = pp.process(source, str(tmp_path / "main.c"))
		assert "A_VAR" in result
		assert "B_VAR" in result

	def test_pragma_once_works_with_include_paths(self, tmp_path):
		inc = tmp_path / "inc"
		inc.mkdir()
		header = inc / "util.h"
		header.write_text("#pragma once\nint UTIL;")

		source = '#include "util.h"\n#include "util.h"\n'
		pp = Preprocessor(include_paths=[str(inc)])
		result = pp.process(source, str(tmp_path / "main.c"))
		count = sum(1 for line in result.split("\n") if "UTIL" in line)
		assert count == 1
