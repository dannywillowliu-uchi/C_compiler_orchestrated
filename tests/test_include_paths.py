"""Tests for #include path search (-I flag) support in the preprocessor."""

import pytest

from compiler.preprocessor import Preprocessor, PreprocessorError


@pytest.fixture()
def include_tree(tmp_path):
	"""Create a directory tree with header files for include path testing."""
	# Primary include dir
	inc1 = tmp_path / "inc1"
	inc1.mkdir()
	(inc1 / "alpha.h").write_text("#define ALPHA 1\n")
	(inc1 / "beta.h").write_text("#define BETA 2\n")

	# Secondary include dir (lower priority)
	inc2 = tmp_path / "inc2"
	inc2.mkdir()
	(inc2 / "beta.h").write_text("#define BETA 999\n")  # shadowed by inc1
	(inc2 / "gamma.h").write_text("#define GAMMA 3\n")

	# Nested subdir header
	sub = inc1 / "sub"
	sub.mkdir()
	(sub / "nested.h").write_text("#define NESTED 42\n")

	# Source dir (for quoted include relative search)
	src = tmp_path / "src"
	src.mkdir()
	(src / "local.h").write_text("#define LOCAL 100\n")
	(src / "main.c").write_text('#include "local.h"\nint x = LOCAL;\n')

	return tmp_path


class TestAngleBracketIncludes:
	"""Test <file> includes resolved via -I paths."""

	def test_single_include_path(self, include_tree):
		inc1 = str(include_tree / "inc1")
		pp = Preprocessor(include_paths=[inc1])
		result = pp.process("#include <alpha.h>\nint x = ALPHA;")
		assert "int x = 1;" in result

	def test_multiple_include_paths(self, include_tree):
		inc1 = str(include_tree / "inc1")
		inc2 = str(include_tree / "inc2")
		pp = Preprocessor(include_paths=[inc1, inc2])
		result = pp.process("#include <gamma.h>\nint x = GAMMA;")
		assert "int x = 3;" in result

	def test_include_path_priority_order(self, include_tree):
		"""First -I path wins when multiple paths contain the same header."""
		inc1 = str(include_tree / "inc1")
		inc2 = str(include_tree / "inc2")
		pp = Preprocessor(include_paths=[inc1, inc2])
		result = pp.process("#include <beta.h>\nint x = BETA;")
		assert "int x = 2;" in result

	def test_include_path_priority_reversed(self, include_tree):
		"""Reversed order gives different result."""
		inc1 = str(include_tree / "inc1")
		inc2 = str(include_tree / "inc2")
		pp = Preprocessor(include_paths=[inc2, inc1])
		result = pp.process("#include <beta.h>\nint x = BETA;")
		assert "int x = 999;" in result

	def test_subdirectory_include(self, include_tree):
		inc1 = str(include_tree / "inc1")
		pp = Preprocessor(include_paths=[inc1])
		result = pp.process("#include <sub/nested.h>\nint x = NESTED;")
		assert "int x = 42;" in result

	def test_angle_bracket_not_found_error(self, include_tree):
		inc1 = str(include_tree / "inc1")
		pp = Preprocessor(include_paths=[inc1])
		with pytest.raises(PreprocessorError, match="Cannot find include file"):
			pp.process("#include <nonexistent.h>\n")

	def test_no_include_paths_error(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="Cannot find include file"):
			pp.process("#include <something.h>\n")


class TestQuotedIncludes:
	"""Test "file" includes resolved relative to current file then -I paths."""

	def test_quoted_relative_to_source(self, include_tree):
		main_c = str(include_tree / "src" / "main.c")
		with open(main_c) as f:
			source = f.read()
		pp = Preprocessor()
		result = pp.process(source, filename=main_c)
		assert "int x = 100;" in result

	def test_quoted_falls_back_to_include_path(self, include_tree):
		"""Quoted includes search -I paths if not found relative to source."""
		inc1 = str(include_tree / "inc1")
		pp = Preprocessor(include_paths=[inc1])
		result = pp.process('#include "alpha.h"\nint x = ALPHA;')
		assert "int x = 1;" in result

	def test_quoted_prefers_relative(self, include_tree):
		"""Quoted include prefers file relative to source over -I path."""
		src_dir = include_tree / "src"
		# Create a conflicting header in inc1
		inc1 = include_tree / "inc1"
		(inc1 / "local.h").write_text("#define LOCAL 777\n")

		main_c = str(src_dir / "main.c")
		with open(main_c) as f:
			source = f.read()
		pp = Preprocessor(include_paths=[str(inc1)])
		result = pp.process(source, filename=main_c)
		# Should use src/local.h (LOCAL=100), not inc1/local.h (LOCAL=777)
		assert "int x = 100;" in result


class TestBuiltinHeaders:
	"""Test that built-in headers take precedence over -I paths."""

	def test_builtin_stdbool_overrides_path(self, include_tree):
		inc1 = include_tree / "inc1"
		(inc1 / "stdbool.h").write_text("#define bool int\n#define true 99\n")
		pp = Preprocessor(include_paths=[str(inc1)])
		result = pp.process("#include <stdbool.h>\nint x = true;")
		# Built-in stdbool.h defines true as 1
		assert "int x = 1;" in result

	def test_builtin_stdarg_available(self):
		pp = Preprocessor()
		result = pp.process("#include <stdarg.h>\n")
		assert "va_list" in result


class TestIncludeGuards:
	"""Test include guard / pragma once interaction with -I paths."""

	def test_include_guard_prevents_double_include(self, include_tree):
		inc1 = include_tree / "inc1"
		(inc1 / "guarded.h").write_text(
			"#ifndef GUARDED_H\n#define GUARDED_H\nint guarded_var;\n#endif\n"
		)
		pp = Preprocessor(include_paths=[str(inc1)])
		result = pp.process(
			"#include <guarded.h>\n#include <guarded.h>\nint x;\n"
		)
		assert result.count("int guarded_var;") == 1

	def test_pragma_once_prevents_double_include(self, include_tree):
		inc1 = include_tree / "inc1"
		(inc1 / "once.h").write_text("#pragma once\nint once_var;\n")
		pp = Preprocessor(include_paths=[str(inc1)])
		result = pp.process(
			"#include <once.h>\n#include <once.h>\nint x;\n"
		)
		assert result.count("int once_var;") == 1


class TestChainedIncludes:
	"""Test includes that include other files."""

	def test_nested_include_via_path(self, include_tree):
		"""Header in inc1 includes another header also in inc1."""
		inc1 = include_tree / "inc1"
		(inc1 / "outer.h").write_text('#include "alpha.h"\n#define OUTER 10\n')
		pp = Preprocessor(include_paths=[str(inc1)])
		result = pp.process("#include <outer.h>\nint a = ALPHA;\nint b = OUTER;")
		assert "int a = 1;" in result
		assert "int b = 10;" in result

	def test_cross_directory_include(self, include_tree):
		"""Header in inc1 includes a header only in inc2."""
		inc1 = include_tree / "inc1"
		inc2 = include_tree / "inc2"
		(inc1 / "cross.h").write_text('#include <gamma.h>\n#define CROSS 55\n')
		pp = Preprocessor(include_paths=[str(inc1), str(inc2)])
		result = pp.process("#include <cross.h>\nint g = GAMMA;\nint c = CROSS;")
		assert "int g = 3;" in result
		assert "int c = 55;" in result


class TestEdgeCases:
	"""Edge cases for include path resolution."""

	def test_empty_include_paths_list(self):
		pp = Preprocessor(include_paths=[])
		with pytest.raises(PreprocessorError, match="Cannot find include file"):
			pp.process("#include <missing.h>\n")

	def test_absolute_path_include(self, tmp_path):
		header = tmp_path / "abs.h"
		header.write_text("#define ABS_VAL 42\n")
		pp = Preprocessor()
		result = pp.process(f'#include "{header}"\nint x = ABS_VAL;')
		assert "int x = 42;" in result

	def test_include_path_with_trailing_slash(self, include_tree):
		inc1 = str(include_tree / "inc1") + "/"
		pp = Preprocessor(include_paths=[inc1])
		result = pp.process("#include <alpha.h>\nint x = ALPHA;")
		assert "int x = 1;" in result

	def test_macro_expanded_include(self, include_tree):
		"""Include argument is a macro that expands to the filename."""
		inc1 = str(include_tree / "inc1")
		pp = Preprocessor(include_paths=[inc1])
		result = pp.process(
			'#define HEADER "alpha.h"\n#include HEADER\nint x = ALPHA;'
		)
		assert "int x = 1;" in result

	def test_include_from_stdin(self, include_tree):
		"""Include search works even when source filename is <stdin>."""
		inc1 = str(include_tree / "inc1")
		pp = Preprocessor(include_paths=[inc1])
		result = pp.process("#include <alpha.h>\nint x = ALPHA;", filename="<stdin>")
		assert "int x = 1;" in result


class TestCompileSourceIntegration:
	"""Integration: compile_source accepts include_paths."""

	def test_compile_source_with_include_paths(self, include_tree):
		inc1 = include_tree / "inc1"
		(inc1 / "defs.h").write_text("#define RETVAL 0\n")
		from compiler.__main__ import compile_source
		asm = compile_source(
			"#include <defs.h>\nint main(void) { return RETVAL; }",
			include_paths=[str(inc1)],
		)
		assert "main" in asm
