"""Comprehensive preprocessor edge-case tests.

Covers: recursive macro expansion prevention, stringification (#) operator,
token pasting (##) with edge cases, variadic macros with __VA_ARGS__,
nested #if/#elif chains, #include with search paths, empty macro arguments,
and macro redefinition warnings.
"""

import os
import tempfile

import pytest

from compiler.preprocessor import Preprocessor, PreprocessorError


# ---------------------------------------------------------------------------
# Recursive macro expansion prevention
# ---------------------------------------------------------------------------

class TestRecursiveMacroExpansion:
	def test_direct_self_reference(self):
		"""A macro that references itself should not infinitely recurse."""
		pp = Preprocessor()
		src = "#define FOO FOO + 1\nFOO"
		result = pp.process(src)
		# FOO should appear unexpanded in the output since it's self-referential
		assert "FOO" in result.strip()

	def test_mutual_recursion_a_b(self):
		"""Two macros referencing each other should not infinitely recurse."""
		pp = Preprocessor()
		src = "#define A B\n#define B A\nA"
		result = pp.process(src)
		# Should terminate; one of A or B should remain unexpanded
		assert result.strip() in ("A", "B")

	def test_indirect_self_reference_three_levels(self):
		"""Indirect recursion through three macros."""
		pp = Preprocessor()
		src = "#define X Y\n#define Y Z\n#define Z X\nX"
		result = pp.process(src)
		# Should terminate without error
		assert result.strip() in ("X", "Y", "Z")

	def test_self_ref_in_function_macro(self):
		"""Function-like macro whose body invokes itself."""
		pp = Preprocessor()
		src = "#define F(x) F(x+1)\nF(0)"
		result = pp.process(src)
		# F should appear in output since it can't re-expand itself
		assert "F" in result.strip()

	def test_non_recursive_nested_expansion(self):
		"""Ensure normal nested expansion still works when no recursion."""
		pp = Preprocessor()
		src = "#define INNER 42\n#define OUTER INNER\nOUTER"
		result = pp.process(src)
		assert result.strip() == "42"

	def test_self_ref_with_args(self):
		"""Self-referencing function macro with arguments."""
		pp = Preprocessor()
		src = "#define M(a, b) M(b, a)\nM(1, 2)"
		result = pp.process(src)
		assert "M" in result.strip()


# ---------------------------------------------------------------------------
# Stringification (#) operator
# ---------------------------------------------------------------------------

class TestStringification:
	def test_basic_stringify(self):
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR(hello)'
		result = pp.process(src)
		assert result.strip() == '"hello"'

	def test_stringify_number(self):
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR(42)'
		result = pp.process(src)
		assert result.strip() == '"42"'

	def test_stringify_expression(self):
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR(a + b)'
		result = pp.process(src)
		assert result.strip() == '"a + b"'

	def test_stringify_with_quotes_in_arg(self):
		"""Quotes in the argument should be escaped in the stringified result."""
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR(say "hi")'
		result = pp.process(src)
		assert result.strip() == r'"say \"hi\""'

	def test_stringify_with_backslash(self):
		"""Backslashes in the argument should be escaped."""
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR(a\\b)'
		result = pp.process(src)
		assert result.strip() == r'"a\\b"'

	def test_stringify_empty_arg(self):
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR()'
		result = pp.process(src)
		assert result.strip() == '""'

	def test_stringify_preserves_other_params(self):
		"""# only applies to the immediately following parameter."""
		pp = Preprocessor()
		src = '#define PAIR(a, b) #a " and " #b\nPAIR(hello, world)'
		result = pp.process(src)
		assert '"hello"' in result.strip()
		assert '"world"' in result.strip()

	def test_stringify_does_not_affect_double_hash(self):
		"""# followed by # should be treated as token pasting, not stringify."""
		pp = Preprocessor()
		src = "#define PASTE(a, b) a ## b\nPASTE(foo, bar)"
		result = pp.process(src)
		assert "foobar" in result.strip()

	def test_stringify_multiword(self):
		pp = Preprocessor()
		src = '#define S(x) #x\nS(int *p = NULL)'
		result = pp.process(src)
		assert result.strip() == '"int *p = NULL"'


# ---------------------------------------------------------------------------
# Token pasting (##) operator with edge cases
# ---------------------------------------------------------------------------

class TestTokenPasting:
	def test_basic_paste(self):
		pp = Preprocessor()
		src = "#define PASTE(a, b) a ## b\nPASTE(foo, bar)"
		result = pp.process(src)
		assert "foobar" in result.strip()

	def test_paste_identifier_number(self):
		pp = Preprocessor()
		src = "#define VAR(n) var_ ## n\nVAR(1)"
		result = pp.process(src)
		assert "var_1" in result.strip()

	def test_paste_creates_valid_token(self):
		pp = Preprocessor()
		src = "#define CONCAT(a, b) a ## b\nCONCAT(my, Func)"
		result = pp.process(src)
		assert "myFunc" in result.strip()

	def test_paste_with_empty_left(self):
		"""Pasting with empty left side."""
		pp = Preprocessor()
		src = "#define PRE(x) ## x\nPRE(hello)"
		result = pp.process(src)
		assert "hello" in result.strip()

	def test_paste_in_object_macro(self):
		"""## in an object-like macro body."""
		pp = Preprocessor()
		src = "#define AB a ## b\nAB"
		result = pp.process(src)
		assert "ab" in result.strip()

	def test_paste_number_to_number(self):
		pp = Preprocessor()
		src = "#define JOIN(a, b) a ## b\nJOIN(1, 2)"
		result = pp.process(src)
		assert "12" in result.strip()

	def test_paste_multiple(self):
		"""Multiple ## in one macro body."""
		pp = Preprocessor()
		src = "#define TRIPLE(a, b, c) a ## b ## c\nTRIPLE(x, y, z)"
		result = pp.process(src)
		assert "xyz" in result.strip()

	def test_paste_preserves_surrounding_text(self):
		pp = Preprocessor()
		src = "#define MAKE(x) prefix_ ## x ## _suffix\nMAKE(mid)"
		result = pp.process(src)
		assert "prefix_mid_suffix" in result.strip()

	def test_paste_inside_string_literal_not_applied(self):
		"""## inside a string literal should not be treated as pasting."""
		pp = Preprocessor()
		src = '#define M "a ## b"\nM'
		result = pp.process(src)
		assert "##" in result.strip()


# ---------------------------------------------------------------------------
# Variadic macros with __VA_ARGS__
# ---------------------------------------------------------------------------

class TestVariadicMacros:
	def test_basic_variadic(self):
		pp = Preprocessor()
		src = "#define LOG(...) log(__VA_ARGS__)\nLOG(1, 2, 3)"
		result = pp.process(src)
		assert "log(1, 2, 3)" in result.strip()

	def test_variadic_with_fixed_params(self):
		pp = Preprocessor()
		src = "#define LOG(fmt, ...) printf(fmt, __VA_ARGS__)\nLOG(\"x=%d\", x)"
		result = pp.process(src)
		assert 'printf("x=%d", x)' in result.strip()

	def test_variadic_single_arg(self):
		pp = Preprocessor()
		src = "#define W(...) wrap(__VA_ARGS__)\nW(only_one)"
		result = pp.process(src)
		assert "wrap(only_one)" in result.strip()

	def test_variadic_no_extra_args(self):
		"""Variadic with fixed params and zero variadic args."""
		pp = Preprocessor()
		src = "#define LOG(fmt, ...) printf(fmt, __VA_ARGS__)\nLOG(\"hello\")"
		# The __VA_ARGS__ should expand to empty
		result = pp.process(src)
		assert 'printf("hello", )' in result.strip()

	def test_variadic_many_args(self):
		pp = Preprocessor()
		src = "#define LIST(...) {__VA_ARGS__}\nLIST(a, b, c, d, e)"
		result = pp.process(src)
		assert "{a, b, c, d, e}" in result.strip()

	def test_variadic_with_complex_args(self):
		"""Variadic args containing parentheses."""
		pp = Preprocessor()
		src = "#define CALL(...) f(__VA_ARGS__)\nCALL(g(1), h(2, 3))"
		result = pp.process(src)
		assert "f(g(1), h(2, 3))" in result.strip()

	def test_variadic_stringify_va_args(self):
		"""__VA_ARGS__ is not a regular param, so # doesn't apply directly."""
		pp = Preprocessor()
		src = "#define DBG(...) puts(__VA_ARGS__)\nDBG(\"test\")"
		result = pp.process(src)
		assert 'puts("test")' in result.strip()


# ---------------------------------------------------------------------------
# Nested #if/#elif chains
# ---------------------------------------------------------------------------

class TestNestedIfElifChains:
	def test_nested_ifdef(self):
		pp = Preprocessor()
		src = "\n".join([
			"#define A",
			"#define B",
			"#ifdef A",
			"#ifdef B",
			"both",
			"#endif",
			"#endif",
		])
		result = pp.process(src)
		assert "both" in result

	def test_nested_ifdef_inner_false(self):
		pp = Preprocessor()
		src = "\n".join([
			"#define A",
			"#ifdef A",
			"#ifdef B",
			"inner",
			"#else",
			"inner_else",
			"#endif",
			"#endif",
		])
		result = pp.process(src)
		assert "inner_else" in result
		assert "inner\n" not in result

	def test_elif_chain(self):
		pp = Preprocessor()
		src = "\n".join([
			"#define X 2",
			"#if X == 1",
			"one",
			"#elif X == 2",
			"two",
			"#elif X == 3",
			"three",
			"#else",
			"other",
			"#endif",
		])
		result = pp.process(src)
		assert "two" in result
		assert "one" not in result
		assert "three" not in result
		assert "other" not in result

	def test_elif_falls_through_to_else(self):
		pp = Preprocessor()
		src = "\n".join([
			"#define X 99",
			"#if X == 1",
			"one",
			"#elif X == 2",
			"two",
			"#else",
			"fallback",
			"#endif",
		])
		result = pp.process(src)
		assert "fallback" in result
		assert "one" not in result
		assert "two" not in result

	def test_nested_if_elif(self):
		pp = Preprocessor()
		src = "\n".join([
			"#define A 1",
			"#define B 2",
			"#if A == 1",
			"#if B == 1",
			"A1_B1",
			"#elif B == 2",
			"A1_B2",
			"#else",
			"A1_Bother",
			"#endif",
			"#else",
			"A_not_1",
			"#endif",
		])
		result = pp.process(src)
		assert "A1_B2" in result
		assert "A1_B1" not in result
		assert "A1_Bother" not in result
		assert "A_not_1" not in result

	def test_three_deep_nesting(self):
		pp = Preprocessor()
		src = "\n".join([
			"#define L1 1",
			"#define L2 1",
			"#define L3 1",
			"#if L1",
			"#if L2",
			"#if L3",
			"deep",
			"#endif",
			"#endif",
			"#endif",
		])
		result = pp.process(src)
		assert "deep" in result

	def test_outer_false_skips_inner(self):
		"""If outer #if is false, inner conditionals should be skipped entirely."""
		pp = Preprocessor()
		src = "\n".join([
			"#if 0",
			"#if 1",
			"should_not_appear",
			"#endif",
			"#endif",
		])
		result = pp.process(src)
		assert "should_not_appear" not in result

	def test_elif_after_true_branch_not_evaluated(self):
		pp = Preprocessor()
		src = "\n".join([
			"#if 1",
			"first",
			"#elif 1",
			"second",
			"#else",
			"third",
			"#endif",
		])
		result = pp.process(src)
		assert "first" in result
		assert "second" not in result
		assert "third" not in result

	def test_if_with_defined_operator(self):
		pp = Preprocessor()
		src = "\n".join([
			"#define HAS_FEATURE",
			"#if defined(HAS_FEATURE)",
			"yes",
			"#else",
			"no",
			"#endif",
		])
		result = pp.process(src)
		assert "yes" in result
		assert "no" not in result

	def test_if_with_negated_defined(self):
		pp = Preprocessor()
		src = "\n".join([
			"#if !defined(MISSING)",
			"not_defined",
			"#endif",
		])
		result = pp.process(src)
		assert "not_defined" in result

	def test_if_combined_conditions(self):
		pp = Preprocessor()
		src = "\n".join([
			"#define A 1",
			"#define B 0",
			"#if A && !B",
			"correct",
			"#endif",
		])
		result = pp.process(src)
		assert "correct" in result

	def test_unterminated_if_raises_error(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="Unterminated"):
			pp.process("#if 1\nhello")

	def test_elif_without_if_raises_error(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="#elif without"):
			pp.process("#elif 1\nhello")

	def test_else_without_if_raises_error(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="#else without"):
			pp.process("#else\nhello")

	def test_endif_without_if_raises_error(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="#endif without"):
			pp.process("#endif")

	def test_elif_after_else_raises_error(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="#elif after #else"):
			pp.process("#if 1\n#else\n#elif 1\n#endif")

	def test_duplicate_else_raises_error(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="Duplicate #else"):
			pp.process("#if 1\n#else\n#else\n#endif")


# ---------------------------------------------------------------------------
# #include with search paths
# ---------------------------------------------------------------------------

class TestIncludeSearchPaths:
	def test_include_from_search_path(self):
		with tempfile.TemporaryDirectory() as tmpdir:
			header = os.path.join(tmpdir, "myheader.h")
			with open(header, "w") as f:
				f.write("#define MYVAL 99\n")
			pp = Preprocessor(include_paths=[tmpdir])
			src = '#include "myheader.h"\nMYVAL'
			result = pp.process(src)
			assert "99" in result

	def test_angle_bracket_include_from_search_path(self):
		with tempfile.TemporaryDirectory() as tmpdir:
			header = os.path.join(tmpdir, "sys.h")
			with open(header, "w") as f:
				f.write("#define SYS_OK 1\n")
			pp = Preprocessor(include_paths=[tmpdir])
			src = "#include <sys.h>\nSYS_OK"
			result = pp.process(src)
			assert "1" in result

	def test_multiple_search_paths_priority(self):
		"""First matching path should win."""
		with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
			h1 = os.path.join(dir1, "common.h")
			h2 = os.path.join(dir2, "common.h")
			with open(h1, "w") as f:
				f.write("#define SOURCE 1\n")
			with open(h2, "w") as f:
				f.write("#define SOURCE 2\n")
			pp = Preprocessor(include_paths=[dir1, dir2])
			src = '#include "common.h"\nSOURCE'
			result = pp.process(src)
			assert "1" in result

	def test_include_not_found_raises_error(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="Cannot find"):
			pp.process('#include "nonexistent.h"')

	def test_include_relative_to_current_file(self):
		"""Quoted include should search relative to the including file."""
		with tempfile.TemporaryDirectory() as tmpdir:
			subdir = os.path.join(tmpdir, "sub")
			os.makedirs(subdir)
			helper = os.path.join(subdir, "helper.h")
			with open(helper, "w") as f:
				f.write("#define HELPER 1\n")
			main_file = os.path.join(subdir, "main.c")
			with open(main_file, "w") as f:
				f.write('#include "helper.h"\nHELPER')
			pp = Preprocessor()
			with open(main_file) as f:
				content = f.read()
			result = pp.process(content, filename=main_file)
			assert "1" in result

	def test_builtin_stdbool(self):
		pp = Preprocessor()
		src = "#include <stdbool.h>\ntrue"
		result = pp.process(src)
		assert "1" in result

	def test_builtin_stdarg(self):
		pp = Preprocessor()
		src = "#include <stdarg.h>\ntypedef void *va_list;"
		result = pp.process(src)
		# Should not error, stdarg.h is built-in
		assert "va_list" in result

	def test_nested_include(self):
		"""Include file that itself includes another file."""
		with tempfile.TemporaryDirectory() as tmpdir:
			inner = os.path.join(tmpdir, "inner.h")
			with open(inner, "w") as f:
				f.write("#define INNER_VAL 42\n")
			outer = os.path.join(tmpdir, "outer.h")
			with open(outer, "w") as f:
				f.write('#include "inner.h"\n#define OUTER_VAL INNER_VAL\n')
			pp = Preprocessor(include_paths=[tmpdir])
			src = '#include "outer.h"\nOUTER_VAL'
			result = pp.process(src)
			assert "42" in result

	def test_include_guard_prevents_double_include(self):
		with tempfile.TemporaryDirectory() as tmpdir:
			header = os.path.join(tmpdir, "guarded.h")
			with open(header, "w") as f:
				f.write("#ifndef GUARDED_H\n#define GUARDED_H\nint x;\n#endif\n")
			pp = Preprocessor(include_paths=[tmpdir])
			src = '#include "guarded.h"\n#include "guarded.h"\n'
			result = pp.process(src)
			# Second include should be empty (already included)
			lines = [line for line in result.split("\n") if line.strip()]
			int_x_count = sum(1 for line in lines if "int x" in line)
			assert int_x_count <= 1

	def test_pragma_once(self):
		with tempfile.TemporaryDirectory() as tmpdir:
			header = os.path.join(tmpdir, "once.h")
			with open(header, "w") as f:
				f.write("#pragma once\n#define ONCE_VAL 7\n")
			pp = Preprocessor(include_paths=[tmpdir])
			src = '#include "once.h"\n#include "once.h"\nONCE_VAL'
			result = pp.process(src)
			assert "7" in result


# ---------------------------------------------------------------------------
# Empty macro arguments
# ---------------------------------------------------------------------------

class TestEmptyMacroArguments:
	def test_empty_single_arg(self):
		pp = Preprocessor()
		src = "#define F(x) [x]\nF()"
		result = pp.process(src)
		assert "[]" in result.strip()

	def test_empty_first_of_two(self):
		pp = Preprocessor()
		src = "#define F(a, b) (a)(b)\nF(, world)"
		result = pp.process(src)
		assert "()(world)" in result.strip()

	def test_empty_second_of_two(self):
		pp = Preprocessor()
		src = "#define F(a, b) (a)(b)\nF(hello, )"
		result = pp.process(src)
		assert "(hello)()" in result.strip()

	def test_both_args_empty(self):
		pp = Preprocessor()
		src = "#define F(a, b) [a|b]\nF(, )"
		result = pp.process(src)
		assert "[|]" in result.strip()

	def test_empty_arg_stringify(self):
		pp = Preprocessor()
		src = '#define S(x) #x\nS()'
		result = pp.process(src)
		assert result.strip() == '""'

	def test_zero_param_macro_with_empty_call(self):
		"""Calling a zero-param function macro with empty parens."""
		pp = Preprocessor()
		src = "#define NOP() 42\nNOP()"
		result = pp.process(src)
		assert "42" in result.strip()


# ---------------------------------------------------------------------------
# Macro redefinition
# ---------------------------------------------------------------------------

class TestMacroRedefinition:
	def test_redefine_object_macro(self):
		"""Redefining a macro should use the new definition."""
		pp = Preprocessor()
		src = "#define X 1\n#define X 2\nX"
		result = pp.process(src)
		assert result.strip() == "2"

	def test_undef_and_redefine(self):
		pp = Preprocessor()
		src = "#define X 1\n#undef X\n#define X 2\nX"
		result = pp.process(src)
		assert result.strip() == "2"

	def test_redefine_function_to_object(self):
		"""Redefine a function-like macro as object-like."""
		pp = Preprocessor()
		src = "#define F(x) (x)\n#define F 99\nF"
		result = pp.process(src)
		assert "99" in result.strip()

	def test_undef_nonexistent_no_error(self):
		"""#undef of undefined macro should not raise."""
		pp = Preprocessor()
		src = "#undef NOPE\n42"
		result = pp.process(src)
		assert "42" in result.strip()

	def test_predefined_macro_override(self):
		pp = Preprocessor(predefined_macros={"VER": "1"})
		src = "#define VER 2\nVER"
		result = pp.process(src)
		assert result.strip() == "2"


# ---------------------------------------------------------------------------
# Predefined macros __LINE__ and __FILE__
# ---------------------------------------------------------------------------

class TestPredefinedMacros:
	def test_line_macro(self):
		pp = Preprocessor()
		src = "a\nb\n__LINE__"
		result = pp.process(src)
		lines = result.split("\n")
		assert lines[2].strip() == "3"

	def test_file_macro(self):
		pp = Preprocessor()
		src = '__FILE__'
		result = pp.process(src, filename="test.c")
		assert '"test.c"' in result.strip()

	def test_line_directive_affects_line(self):
		pp = Preprocessor()
		src = '#line 100\n__LINE__'
		result = pp.process(src)
		lines = [line for line in result.split("\n") if line.strip()]
		assert "100" in lines[-1]

	def test_line_directive_with_filename(self):
		pp = Preprocessor()
		src = '#line 50 "other.c"\n__FILE__'
		result = pp.process(src)
		assert '"other.c"' in result


# ---------------------------------------------------------------------------
# Comment stripping edge cases
# ---------------------------------------------------------------------------

class TestCommentStripping:
	def test_block_comment_preserves_lines(self):
		pp = Preprocessor()
		src = "a\n/* comment\nspanning\nlines */\nb"
		result = pp.process(src)
		lines = result.split("\n")
		# Line count should be preserved
		assert len(lines) == 5

	def test_line_comment(self):
		pp = Preprocessor()
		src = "code // comment\nmore"
		result = pp.process(src)
		assert "comment" not in result
		assert "code" in result
		assert "more" in result

	def test_comment_in_string_preserved(self):
		pp = Preprocessor()
		src = '"hello /* not a comment */"'
		result = pp.process(src)
		assert "/* not a comment */" in result

	def test_comment_in_char_preserved(self):
		pp = Preprocessor()
		src = "char c = '/';"
		result = pp.process(src)
		assert "'/'" in result


# ---------------------------------------------------------------------------
# Continuation lines
# ---------------------------------------------------------------------------

class TestContinuationLines:
	def test_define_continuation(self):
		pp = Preprocessor()
		src = "#define MULTI \\\n  (1 + \\\n  2)\nMULTI"
		result = pp.process(src)
		assert "(1 +   2)" in result.strip()

	def test_regular_line_continuation(self):
		pp = Preprocessor()
		src = "hello \\\nworld"
		result = pp.process(src)
		assert "hello world" in result


# ---------------------------------------------------------------------------
# Error directive
# ---------------------------------------------------------------------------

class TestErrorDirective:
	def test_error_raises(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="#error"):
			pp.process('#error "something went wrong"')

	def test_error_in_false_branch_ignored(self):
		pp = Preprocessor()
		src = "#if 0\n#error should not fire\n#endif\nok"
		result = pp.process(src)
		assert "ok" in result


# ---------------------------------------------------------------------------
# Complex integration scenarios
# ---------------------------------------------------------------------------

class TestComplexScenarios:
	def test_macro_expanding_to_directive_not_processed(self):
		"""Macros expanding to # directives should not be re-processed as directives."""
		pp = Preprocessor()
		src = '#define X #define Y 1\nX'
		pp.process(src)
		# The expanded text should appear literally, not as a directive
		assert "Y" not in pp.macros

	def test_function_macro_not_expanded_without_parens(self):
		"""Function-like macro name without () should not expand."""
		pp = Preprocessor()
		src = "#define F(x) (x)\nint F = 1;"
		result = pp.process(src)
		assert "int F = 1;" in result.strip()

	def test_macro_in_if_condition(self):
		pp = Preprocessor()
		src = "#define VERSION 3\n#if VERSION >= 2\nnew_api\n#endif"
		result = pp.process(src)
		assert "new_api" in result

	def test_chained_elif_with_macros(self):
		pp = Preprocessor()
		src = "\n".join([
			"#define PLATFORM 3",
			"#if PLATFORM == 1",
			"linux",
			"#elif PLATFORM == 2",
			"mac",
			"#elif PLATFORM == 3",
			"windows",
			"#else",
			"unknown",
			"#endif",
		])
		result = pp.process(src)
		assert "windows" in result
		assert "linux" not in result
		assert "mac" not in result

	def test_ifndef_include_guard_pattern(self):
		pp = Preprocessor()
		src = "\n".join([
			"#ifndef MY_HEADER_H",
			"#define MY_HEADER_H",
			"int guarded_content;",
			"#endif",
		])
		result = pp.process(src)
		assert "guarded_content" in result

	def test_ifdef_with_undef(self):
		pp = Preprocessor()
		src = "\n".join([
			"#define FEATURE",
			"#ifdef FEATURE",
			"enabled",
			"#endif",
			"#undef FEATURE",
			"#ifdef FEATURE",
			"should_not_appear",
			"#endif",
		])
		result = pp.process(src)
		assert "enabled" in result
		assert "should_not_appear" not in result

	def test_paste_and_stringify_combined(self):
		"""Macro using both # and ## operators."""
		pp = Preprocessor()
		src = "#define DECL(type, name) type var_ ## name = 0; /* #name */\nDECL(int, count)"
		result = pp.process(src)
		assert "var_count" in result.strip()
		assert "int" in result.strip()

	def test_nested_function_macro_expansion(self):
		pp = Preprocessor()
		src = "\n".join([
			"#define ADD(a, b) ((a) + (b))",
			"#define MUL(a, b) ((a) * (b))",
			"#define EXPR MUL(ADD(1, 2), ADD(3, 4))",
			"EXPR",
		])
		result = pp.process(src)
		assert "((1) + (2))" in result
		assert "((3) + (4))" in result

	def test_warning_directive(self):
		pp = Preprocessor()
		src = '#warning deprecated API\ncode'
		result = pp.process(src)
		assert "code" in result
		assert len(pp.warnings) == 1
		assert "deprecated API" in pp.warnings[0]
