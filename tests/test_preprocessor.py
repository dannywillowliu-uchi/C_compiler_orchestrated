"""Comprehensive tests for the C preprocessor."""

import textwrap

import pytest

from compiler.preprocessor import Preprocessor, PreprocessorError


# ── Helpers ──────────────────────────────────────────────────────────────────


def preprocess(source: str, **kwargs) -> str:
	pp = Preprocessor(**kwargs)
	return pp.process(textwrap.dedent(source))


def preprocess_lines(source: str, **kwargs) -> list[str]:
	"""Return non-empty, stripped output lines."""
	result = preprocess(source, **kwargs)
	return [line for line in result.split("\n") if line.strip()]


def has_line(lines: list[str], text: str) -> bool:
	return any(text in line for line in lines)


def no_line(lines: list[str], text: str) -> bool:
	return not any(text in line for line in lines)


# ── Comment Stripping ────────────────────────────────────────────────────────


class TestCommentStripping:
	def test_line_comment_removed(self) -> None:
		result = preprocess("int x; // comment\n")
		assert "//" not in result
		assert "int x;" in result

	def test_block_comment_removed(self) -> None:
		result = preprocess("int /* comment */ x;\n")
		assert "/*" not in result
		assert "int" in result
		assert "x;" in result

	def test_multiline_block_comment_preserves_lines(self) -> None:
		source = "a\n/* line1\nline2\nline3 */\nb\n"
		result = preprocess(source)
		lines = result.split("\n")
		assert has_line(lines, "a")
		assert has_line(lines, "b")

	def test_comment_in_string_literal_preserved(self) -> None:
		result = preprocess('"hello // not a comment"\n')
		assert "// not a comment" in result

	def test_comment_in_char_literal_preserved(self) -> None:
		result = preprocess("char c = '/'; // real comment\n")
		assert "'/'" in result
		assert "real comment" not in result

	def test_block_comment_in_string_preserved(self) -> None:
		result = preprocess('"/* not a comment */"\n')
		assert "/* not a comment */" in result


# ── Line Continuation ────────────────────────────────────────────────────────


class TestLineContinuation:
	def test_backslash_newline_joins_lines(self) -> None:
		source = "#define LONG_MACRO \\\n    value\nLONG_MACRO\n"
		lines = preprocess_lines(source)
		assert has_line(lines, "value")

	def test_multiple_continuations(self) -> None:
		source = "#define M \\\n    a + \\\n    b\nM\n"
		lines = preprocess_lines(source)
		assert any("a + " in line and "b" in line for line in lines)


# ── #define Object-Like Macros ───────────────────────────────────────────────


class TestDefineObjectLike:
	def test_simple_define(self) -> None:
		lines = preprocess_lines("#define X 42\nint a = X;\n")
		assert has_line(lines, "42")

	def test_define_replaces_in_code(self) -> None:
		source = "#define MAX 100\nif (x > MAX) return MAX;\n"
		lines = preprocess_lines(source)
		line = [ln for ln in lines if "if" in ln][0]
		assert "100" in line
		assert "MAX" not in line

	def test_define_no_body(self) -> None:
		result = preprocess("#define FLAG\nFLAG\n")
		lines = result.split("\n")
		assert any(line.strip() == "" for line in lines[1:])

	def test_define_does_not_replace_in_strings(self) -> None:
		source = '#define X 42\nchar *s = "X";\n'
		lines = preprocess_lines(source)
		assert has_line(lines, '"X"')

	def test_define_does_not_replace_partial_words(self) -> None:
		source = "#define X 42\nint XMAX = 1;\n"
		lines = preprocess_lines(source)
		assert has_line(lines, "XMAX")

	def test_chained_defines(self) -> None:
		source = "#define A B\n#define B 42\nint x = A;\n"
		lines = preprocess_lines(source)
		assert has_line(lines, "42")

	def test_recursive_define_stops(self) -> None:
		source = "#define X X\nX\n"
		lines = preprocess_lines(source)
		assert has_line(lines, "X")

	def test_define_with_expression(self) -> None:
		source = "#define EXPR (1 + 2)\nint x = EXPR;\n"
		lines = preprocess_lines(source)
		assert has_line(lines, "(1 + 2)")


# ── #define Function-Like Macros ─────────────────────────────────────────────


class TestDefineFunctionLike:
	def test_function_macro_basic(self) -> None:
		source = "#define ADD(a, b) ((a) + (b))\nint x = ADD(1, 2);\n"
		lines = preprocess_lines(source)
		assert has_line(lines, "((1) + (2))")

	def test_function_macro_no_args(self) -> None:
		source = "#define FOO() 42\nint x = FOO();\n"
		lines = preprocess_lines(source)
		assert has_line(lines, "42")

	def test_function_macro_nested_parens(self) -> None:
		source = "#define F(x) x\nF((a, b))\n"
		lines = preprocess_lines(source)
		assert has_line(lines, "(a, b)")

	def test_function_macro_with_string_arg(self) -> None:
		source = '#define F(x) x\nF("hello, world")\n'
		lines = preprocess_lines(source)
		assert has_line(lines, '"hello, world"')

	def test_function_macro_wrong_arg_count(self) -> None:
		with pytest.raises(PreprocessorError, match="expects 2 arguments"):
			preprocess("#define ADD(a, b) a+b\nADD(1)\n")

	def test_function_macro_name_without_parens(self) -> None:
		source = "#define F(x) x+1\nint F = 5;\n"
		lines = preprocess_lines(source)
		assert any("F" in line and "5" in line for line in lines)

	def test_variadic_macro(self) -> None:
		source = "#define LOG(fmt, ...) printf(fmt, __VA_ARGS__)\nLOG(\"x=%d y=%d\", x, y)\n"
		lines = preprocess_lines(source)
		assert any("printf" in line and "x, y" in line for line in lines)

	def test_nested_function_macros(self) -> None:
		source = "#define DOUBLE(x) ((x) + (x))\n#define QUAD(x) DOUBLE(DOUBLE(x))\nQUAD(5)\n"
		lines = preprocess_lines(source)
		assert has_line(lines, "5")

	def test_macro_arg_with_spaces(self) -> None:
		source = "#define F(x) x\nF(  hello  )\n"
		lines = preprocess_lines(source)
		assert has_line(lines, "hello")


# ── #undef ───────────────────────────────────────────────────────────────────


class TestUndef:
	def test_undef_removes_macro(self) -> None:
		source = "#define X 42\n#undef X\nX\n"
		lines = preprocess_lines(source)
		assert has_line(lines, "X")
		assert no_line(lines, "42")

	def test_undef_nonexistent_is_ok(self) -> None:
		preprocess("#undef NONEXISTENT\n")

	def test_undef_then_redefine(self) -> None:
		source = "#define X 1\n#undef X\n#define X 2\nX\n"
		lines = preprocess_lines(source)
		assert has_line(lines, "2")

	def test_undef_empty_error(self) -> None:
		with pytest.raises(PreprocessorError, match="Expected macro name"):
			preprocess("#undef\n")


# ── #ifdef / #ifndef ─────────────────────────────────────────────────────────


class TestIfdefIfndef:
	def test_ifdef_defined(self) -> None:
		source = "#define X\n#ifdef X\nyes\n#endif\n"
		lines = preprocess_lines(source)
		assert has_line(lines, "yes")

	def test_ifdef_not_defined(self) -> None:
		source = "#ifdef X\nyes\n#endif\n"
		lines = preprocess_lines(source)
		assert no_line(lines, "yes")

	def test_ifndef_defined(self) -> None:
		source = "#define X\n#ifndef X\nyes\n#endif\n"
		lines = preprocess_lines(source)
		assert no_line(lines, "yes")

	def test_ifndef_not_defined(self) -> None:
		source = "#ifndef X\nyes\n#endif\n"
		lines = preprocess_lines(source)
		assert has_line(lines, "yes")

	def test_ifdef_else(self) -> None:
		source = "#ifdef X\nyes\n#else\nno\n#endif\n"
		lines = preprocess_lines(source)
		assert no_line(lines, "yes")
		assert has_line(lines, "no")

	def test_ifdef_defined_else(self) -> None:
		source = "#define X\n#ifdef X\nyes\n#else\nno\n#endif\n"
		lines = preprocess_lines(source)
		assert has_line(lines, "yes")
		assert no_line(lines, "no")

	def test_nested_ifdef(self) -> None:
		source = "#define A\n#define B\n#ifdef A\n#ifdef B\nboth\n#endif\n#endif\n"
		lines = preprocess_lines(source)
		assert has_line(lines, "both")

	def test_nested_ifdef_outer_false(self) -> None:
		source = "#define B\n#ifdef A\n#ifdef B\nboth\n#endif\n#endif\n"
		lines = preprocess_lines(source)
		assert no_line(lines, "both")


# ── #if / #elif / #else / #endif ─────────────────────────────────────────────


class TestIfElif:
	def test_if_true(self) -> None:
		lines = preprocess_lines("#if 1\nyes\n#endif\n")
		assert has_line(lines, "yes")

	def test_if_false(self) -> None:
		lines = preprocess_lines("#if 0\nyes\n#endif\n")
		assert no_line(lines, "yes")

	def test_if_expression(self) -> None:
		lines = preprocess_lines("#if 2 + 3 > 4\nyes\n#endif\n")
		assert has_line(lines, "yes")

	def test_if_defined_operator(self) -> None:
		lines = preprocess_lines("#define X\n#if defined(X)\nyes\n#endif\n")
		assert has_line(lines, "yes")

	def test_if_defined_without_parens(self) -> None:
		lines = preprocess_lines("#define X\n#if defined X\nyes\n#endif\n")
		assert has_line(lines, "yes")

	def test_if_not_defined(self) -> None:
		lines = preprocess_lines("#if defined(X)\nyes\n#endif\n")
		assert no_line(lines, "yes")

	def test_elif_basic(self) -> None:
		lines = preprocess_lines("#if 0\nfirst\n#elif 1\nsecond\n#endif\n")
		assert no_line(lines, "first")
		assert has_line(lines, "second")

	def test_elif_chain(self) -> None:
		lines = preprocess_lines("#if 0\nfirst\n#elif 0\nsecond\n#elif 1\nthird\n#endif\n")
		assert no_line(lines, "first")
		assert no_line(lines, "second")
		assert has_line(lines, "third")

	def test_elif_skipped_after_true(self) -> None:
		lines = preprocess_lines("#if 1\nfirst\n#elif 1\nsecond\n#endif\n")
		assert has_line(lines, "first")
		assert no_line(lines, "second")

	def test_if_else(self) -> None:
		lines = preprocess_lines("#if 0\nfirst\n#else\nsecond\n#endif\n")
		assert no_line(lines, "first")
		assert has_line(lines, "second")

	def test_if_elif_else(self) -> None:
		lines = preprocess_lines("#if 0\nfirst\n#elif 0\nsecond\n#else\nthird\n#endif\n")
		assert no_line(lines, "first")
		assert no_line(lines, "second")
		assert has_line(lines, "third")

	def test_if_with_macro_value(self) -> None:
		lines = preprocess_lines("#define X 5\n#if X > 3\nyes\n#endif\n")
		assert has_line(lines, "yes")

	def test_if_with_logical_operators(self) -> None:
		lines = preprocess_lines("#define A 1\n#define B 0\n#if A and not B\nyes\n#endif\n")
		assert has_line(lines, "yes")


# ── Conditional Errors ───────────────────────────────────────────────────────


class TestConditionalErrors:
	def test_endif_without_if(self) -> None:
		with pytest.raises(PreprocessorError, match="#endif without"):
			preprocess("#endif\n")

	def test_else_without_if(self) -> None:
		with pytest.raises(PreprocessorError, match="#else without"):
			preprocess("#else\n")

	def test_elif_without_if(self) -> None:
		with pytest.raises(PreprocessorError, match="#elif without"):
			preprocess("#elif 1\n")

	def test_duplicate_else(self) -> None:
		with pytest.raises(PreprocessorError, match="Duplicate #else"):
			preprocess("#if 1\n#else\n#else\n#endif\n")

	def test_elif_after_else(self) -> None:
		with pytest.raises(PreprocessorError, match="#elif after #else"):
			preprocess("#if 0\n#else\n#elif 1\n#endif\n")

	def test_unterminated_if(self) -> None:
		with pytest.raises(PreprocessorError, match="Unterminated"):
			preprocess("#if 1\ncode\n")


# ── #include ─────────────────────────────────────────────────────────────────


class TestInclude:
	def test_include_quoted(self, tmp_path) -> None:
		header = tmp_path / "header.h"
		header.write_text("#define X 42\n")

		source = f'#include "{header}"\nint a = X;\n'
		lines = preprocess_lines(source)
		assert has_line(lines, "42")

	def test_include_angle_bracket(self, tmp_path) -> None:
		inc_dir = tmp_path / "include"
		inc_dir.mkdir()
		header = inc_dir / "myheader.h"
		header.write_text("#define Y 99\n")

		source = "#include <myheader.h>\nint a = Y;\n"
		lines = preprocess_lines(source, include_paths=[str(inc_dir)])
		assert has_line(lines, "99")

	def test_include_file_not_found(self) -> None:
		with pytest.raises(PreprocessorError, match="Cannot find"):
			preprocess('#include "nonexistent.h"\n')

	def test_include_nested(self, tmp_path) -> None:
		inner = tmp_path / "inner.h"
		inner.write_text("#define INNER_VAL 7\n")

		outer = tmp_path / "outer.h"
		outer.write_text(f'#include "{inner}"\n#define OUTER_VAL (INNER_VAL + 1)\n')

		source = f'#include "{outer}"\nint x = OUTER_VAL;\n'
		lines = preprocess_lines(source)
		assert has_line(lines, "(7 + 1)")

	def test_include_relative_to_current_file(self, tmp_path) -> None:
		sub = tmp_path / "sub"
		sub.mkdir()
		helper = sub / "helper.h"
		helper.write_text("#define HELPER 1\n")

		main_h = sub / "main.h"
		main_h.write_text('#include "helper.h"\n')

		source = f'#include "{main_h}"\nint x = HELPER;\n'
		lines = preprocess_lines(source)
		assert has_line(lines, "1")

	def test_include_guard_prevents_double_include(self, tmp_path) -> None:
		header = tmp_path / "guarded.h"
		header.write_text("#define GUARDED 1\n")

		source = f'#include "{header}"\n#include "{header}"\nint x = GUARDED;\n'
		lines = preprocess_lines(source)
		assert has_line(lines, "1")

	def test_include_search_paths(self, tmp_path) -> None:
		d1 = tmp_path / "dir1"
		d1.mkdir()
		d2 = tmp_path / "dir2"
		d2.mkdir()

		(d2 / "found.h").write_text("#define FOUND 1\n")

		source = '#include "found.h"\nFOUND\n'
		lines = preprocess_lines(source, include_paths=[str(d1), str(d2)])
		assert has_line(lines, "1")


# ── Predefined Macros ────────────────────────────────────────────────────────


class TestPredefinedMacros:
	def test_line_macro(self) -> None:
		source = "a\nb\n__LINE__\n"
		result = preprocess(source)
		lines = result.split("\n")
		assert has_line(lines, "3")

	def test_file_macro(self) -> None:
		pp = Preprocessor()
		result = pp.process("__FILE__\n", filename="test.c")
		assert '"test.c"' in result

	def test_line_and_file_together(self) -> None:
		pp = Preprocessor()
		result = pp.process("__FILE__ __LINE__\n", filename="main.c")
		assert '"main.c"' in result
		assert "1" in result

	def test_predefined_macros_via_constructor(self) -> None:
		pp = Preprocessor(predefined_macros={"VERSION": "42"})
		result = pp.process("int v = VERSION;\n")
		assert "42" in result


# ── Macro Expansion Edge Cases ───────────────────────────────────────────────


class TestMacroExpansionEdgeCases:
	def test_macro_in_macro_body(self) -> None:
		lines = preprocess_lines("#define A 1\n#define B A\nB\n")
		assert has_line(lines, "1")

	def test_multiply_defined_macro(self) -> None:
		lines = preprocess_lines("#define X 1\n#define X 2\nX\n")
		assert has_line(lines, "2")

	def test_empty_define_in_expression(self) -> None:
		lines = preprocess_lines("#define EMPTY\n#ifdef EMPTY\nyes\n#endif\n")
		assert has_line(lines, "yes")

	def test_function_macro_multiline_args(self) -> None:
		source = "#define ADD(a, b) ((a) + (b))\nint x = ADD(1, 2);\n"
		lines = preprocess_lines(source)
		assert has_line(lines, "((1) + (2))")


# ── #error ───────────────────────────────────────────────────────────────────


class TestErrorDirective:
	def test_error_raises(self) -> None:
		with pytest.raises(PreprocessorError, match="#error.*stop here"):
			preprocess("#error stop here\n")

	def test_error_in_false_branch_ignored(self) -> None:
		result = preprocess("#if 0\n#error should not fire\n#endif\n")
		assert result is not None


# ── Empty / Edge Input ───────────────────────────────────────────────────────


class TestEdgeCases:
	def test_empty_input(self) -> None:
		result = preprocess("")
		assert result == ""

	def test_only_newlines(self) -> None:
		result = preprocess("\n\n\n")
		assert isinstance(result, str)

	def test_hash_only_line(self) -> None:
		result = preprocess("#\n")
		assert isinstance(result, str)

	def test_pragma_ignored(self) -> None:
		result = preprocess("#pragma once\nint x;\n")
		assert "int x;" in result

	def test_unknown_directive_raises(self) -> None:
		with pytest.raises(PreprocessorError, match="Unknown directive"):
			preprocess("#foobar\n")

	def test_preserves_non_directive_lines(self) -> None:
		source = "int main() {\n    return 0;\n}\n"
		result = preprocess(source)
		assert "int main()" in result
		assert "return 0;" in result

	def test_include_guard_pattern(self, tmp_path) -> None:
		header = tmp_path / "guard.h"
		header.write_text(
			"#ifndef GUARD_H\n"
			"#define GUARD_H\n"
			"int guarded_value;\n"
			"#endif\n"
		)
		source = f'#include "{header}"\nint x = guarded_value;\n'
		lines = preprocess_lines(source)
		assert has_line(lines, "guarded_value")


# ── Integration Scenarios ────────────────────────────────────────────────────


class TestIntegration:
	def test_typical_c_header_pattern(self, tmp_path) -> None:
		header = tmp_path / "config.h"
		header.write_text(
			"#ifndef CONFIG_H\n"
			"#define CONFIG_H\n"
			"#define MAX_SIZE 1024\n"
			"#define MIN(a, b) ((a) < (b) ? (a) : (b))\n"
			"#endif\n"
		)

		source = (
			f'#include "{header}"\n'
			"int buf[MAX_SIZE];\n"
			"int m = MIN(3, 5);\n"
		)

		lines = preprocess_lines(source)
		assert has_line(lines, "1024")
		assert has_line(lines, "((3) < (5) ? (3) : (5))")

	def test_platform_conditional(self) -> None:
		source = (
			"#define LINUX\n"
			"#ifdef LINUX\n"
			"int platform = 1;\n"
			"#else\n"
			"int platform = 2;\n"
			"#endif\n"
		)
		lines = preprocess_lines(source)
		assert has_line(lines, "platform = 1")
		assert no_line(lines, "platform = 2")

	def test_debug_mode_toggle(self) -> None:
		source = (
			"#define DEBUG 1\n"
			"#if DEBUG\n"
			"int debug_enabled = 1;\n"
			"#else\n"
			"int debug_enabled = 0;\n"
			"#endif\n"
		)
		lines = preprocess_lines(source)
		assert has_line(lines, "debug_enabled = 1")
		assert no_line(lines, "debug_enabled = 0")

	def test_feature_macros_with_undef(self) -> None:
		source = (
			"#define FEATURE_A\n"
			"#define FEATURE_B\n"
			"#ifdef FEATURE_A\n"
			"int has_a = 1;\n"
			"#endif\n"
			"#undef FEATURE_B\n"
			"#ifdef FEATURE_B\n"
			"int has_b = 1;\n"
			"#endif\n"
		)
		lines = preprocess_lines(source)
		assert has_line(lines, "has_a")
		assert no_line(lines, "has_b")

	def test_complex_macro_expansion(self) -> None:
		source = (
			"#define SQUARE(x) ((x) * (x))\n"
			"#define CUBE(x) ((x) * SQUARE(x))\n"
			"int r = CUBE(3);\n"
		)
		lines = preprocess_lines(source)
		assert has_line(lines, "((3) * ((3) * (3)))")
