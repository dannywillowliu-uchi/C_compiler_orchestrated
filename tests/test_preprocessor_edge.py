"""Edge-case tests for preprocessor features.

Covers: nested macro expansion, empty args, variadic macros with __VA_ARGS__,
#if with complex expressions (&&, ||, ternary, defined()), #include with
macro-expanded paths, stringification edge cases, token pasting edge cases,
recursive include guards, pragma once.
"""

import os
import tempfile

import pytest

from compiler.preprocessor import Preprocessor, PreprocessorError


# ---------------------------------------------------------------------------
# Nested macro expansion
# ---------------------------------------------------------------------------

class TestNestedMacroExpansion:
	def test_two_level_nesting(self):
		pp = Preprocessor()
		src = "#define A 1\n#define B A\nB"
		result = pp.process(src)
		assert result.strip() == "1"

	def test_three_level_nesting(self):
		pp = Preprocessor()
		src = "#define X 42\n#define Y X\n#define Z Y\nZ"
		result = pp.process(src)
		assert result.strip() == "42"

	def test_nested_function_macro(self):
		pp = Preprocessor()
		src = "#define ADD(a,b) (a+b)\n#define DOUBLE(x) ADD(x,x)\nDOUBLE(5)"
		result = pp.process(src)
		assert result.strip() == "(5+5)"

	def test_self_referencing_macro_stops(self):
		"""A macro referencing itself should not infinitely recurse."""
		pp = Preprocessor()
		src = "#define FOO (1 + FOO)\nFOO"
		result = pp.process(src)
		assert "FOO" in result.strip()

	def test_mutual_recursion_stops(self):
		"""Mutually recursive macros should not infinitely recurse."""
		pp = Preprocessor()
		src = "#define A B\n#define B A\nA"
		result = pp.process(src)
		# Should terminate; exact output depends on expansion order
		assert result.strip() in ("A", "B")

	def test_nested_with_function_and_object_macros(self):
		pp = Preprocessor()
		src = (
			"#define VAL 10\n"
			"#define INC(x) (x + 1)\n"
			"#define RESULT INC(VAL)\n"
			"RESULT"
		)
		result = pp.process(src)
		assert result.strip() == "(10 + 1)"


# ---------------------------------------------------------------------------
# Macro with empty args
# ---------------------------------------------------------------------------

class TestEmptyArgs:
	def test_function_macro_empty_body(self):
		pp = Preprocessor()
		src = "#define NOOP(x)\nNOOP(hello) world"
		result = pp.process(src)
		assert "world" in result.strip()

	def test_zero_arg_function_macro(self):
		pp = Preprocessor()
		src = "#define GET() 99\nGET()"
		result = pp.process(src)
		assert result.strip() == "99"

	def test_function_macro_with_empty_arg_value(self):
		"""Calling a macro with an empty argument string."""
		pp = Preprocessor()
		src = "#define ID(x) x\nID()"
		result = pp.process(src)
		assert result.strip() == ""

	def test_multi_arg_one_empty(self):
		pp = Preprocessor()
		src = "#define PAIR(a,b) a-b\nPAIR(x,)"
		result = pp.process(src)
		assert result.strip() == "x-"


# ---------------------------------------------------------------------------
# Variadic macros with __VA_ARGS__
# ---------------------------------------------------------------------------

class TestVariadicMacros:
	def test_basic_variadic(self):
		pp = Preprocessor()
		src = "#define LOG(...) log(__VA_ARGS__)\nLOG(1, 2, 3)"
		result = pp.process(src)
		assert result.strip() == "log(1, 2, 3)"

	def test_variadic_with_named_params(self):
		pp = Preprocessor()
		src = "#define FMT(fmt, ...) printf(fmt, __VA_ARGS__)\nFMT(\"x=%d\", 42)"
		result = pp.process(src)
		assert 'printf("x=%d", 42)' in result.strip()

	def test_variadic_single_arg(self):
		pp = Preprocessor()
		src = "#define WRAP(...) {__VA_ARGS__}\nWRAP(only)"
		result = pp.process(src)
		assert result.strip() == "{only}"

	def test_variadic_no_extra_args(self):
		"""Variadic with named params and no variadic args provided."""
		pp = Preprocessor()
		src = "#define F(a, ...) a __VA_ARGS__\nF(hello)"
		result = pp.process(src)
		assert "hello" in result.strip()

	def test_variadic_many_args(self):
		pp = Preprocessor()
		src = "#define LIST(...) [__VA_ARGS__]\nLIST(a, b, c, d, e)"
		result = pp.process(src)
		assert result.strip() == "[a, b, c, d, e]"


# ---------------------------------------------------------------------------
# #if with complex expressions
# ---------------------------------------------------------------------------

class TestIfComplexExpressions:
	def test_logical_and(self):
		pp = Preprocessor()
		src = "#define A 1\n#define B 1\n#if A && B\nyes\n#endif"
		result = pp.process(src)
		assert "yes" in result

	def test_logical_and_false(self):
		pp = Preprocessor()
		src = "#define A 1\n#define B 0\n#if A && B\nyes\n#else\nno\n#endif"
		result = pp.process(src)
		assert "no" in result
		assert "yes" not in result

	def test_logical_or(self):
		pp = Preprocessor()
		src = "#define A 0\n#define B 1\n#if A || B\nyes\n#endif"
		result = pp.process(src)
		assert "yes" in result

	def test_logical_or_both_false(self):
		pp = Preprocessor()
		src = "#define A 0\n#define B 0\n#if A || B\nyes\n#else\nno\n#endif"
		result = pp.process(src)
		assert "no" in result

	def test_logical_not(self):
		pp = Preprocessor()
		src = "#define A 0\n#if !A\nyes\n#endif"
		result = pp.process(src)
		assert "yes" in result

	def test_defined_in_if(self):
		pp = Preprocessor()
		src = "#define FOO\n#if defined(FOO)\nyes\n#endif"
		result = pp.process(src)
		assert "yes" in result

	def test_not_defined_in_if(self):
		pp = Preprocessor()
		src = "#if defined(NOPE)\nyes\n#else\nno\n#endif"
		result = pp.process(src)
		assert "no" in result

	def test_defined_without_parens(self):
		pp = Preprocessor()
		src = "#define BAR\n#if defined BAR\nyes\n#endif"
		result = pp.process(src)
		assert "yes" in result

	def test_combined_defined_and_logic(self):
		pp = Preprocessor()
		src = "#define X 1\n#if defined(X) && X > 0\nyes\n#endif"
		result = pp.process(src)
		assert "yes" in result

	def test_complex_or_and(self):
		pp = Preprocessor()
		src = "#define A 1\n#define B 0\n#define C 1\n#if (A || B) && C\nyes\n#endif"
		result = pp.process(src)
		assert "yes" in result

	def test_not_equal(self):
		pp = Preprocessor()
		src = "#define VER 2\n#if VER != 1\nyes\n#endif"
		result = pp.process(src)
		assert "yes" in result

	def test_nested_if_elif_else(self):
		pp = Preprocessor()
		src = (
			"#define V 3\n"
			"#if V == 1\none\n"
			"#elif V == 2\ntwo\n"
			"#elif V == 3\nthree\n"
			"#else\nother\n"
			"#endif"
		)
		result = pp.process(src)
		assert "three" in result
		assert "one" not in result
		assert "two" not in result

	def test_undefined_identifier_becomes_zero(self):
		"""Undefined identifiers in #if expressions should evaluate to 0."""
		pp = Preprocessor()
		src = "#if UNDEF\nyes\n#else\nno\n#endif"
		result = pp.process(src)
		assert "no" in result

	def test_arithmetic_in_if(self):
		pp = Preprocessor()
		src = "#define A 2\n#define B 3\n#if A + B == 5\nyes\n#endif"
		result = pp.process(src)
		assert "yes" in result


# ---------------------------------------------------------------------------
# #include with macro-expanded paths
# ---------------------------------------------------------------------------

class TestIncludeMacroExpanded:
	def test_include_with_macro_path(self, tmp_path):
		header = tmp_path / "myheader.h"
		header.write_text("#define FROM_HEADER 100\n")

		pp = Preprocessor(include_paths=[str(tmp_path)])
		src = f'#define HEADER "myheader.h"\n#include HEADER\nFROM_HEADER'
		result = pp.process(src)
		assert "100" in result


# ---------------------------------------------------------------------------
# Stringification edge cases
# ---------------------------------------------------------------------------

class TestStringification:
	def test_basic_stringify(self):
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR(hello)'
		result = pp.process(src)
		assert result.strip() == '"hello"'

	def test_stringify_with_spaces(self):
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR(  hello  )'
		result = pp.process(src)
		# Stringification should strip leading/trailing whitespace from arg
		assert '"hello"' in result.strip()

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
		"""Stringifying an arg that contains a double quote should escape it."""
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR(say "hi")'
		result = pp.process(src)
		assert '\\"hi\\"' in result

	def test_stringify_empty_arg(self):
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR()'
		result = pp.process(src)
		assert result.strip() == '""'

	def test_stringify_preserves_other_tokens(self):
		pp = Preprocessor()
		src = '#define SHOW(x) puts(#x)\nSHOW(value)'
		result = pp.process(src)
		assert result.strip() == 'puts("value")'

	def test_stringify_backslash_in_arg(self):
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR(a\\b)'
		result = pp.process(src)
		assert "\\\\b" in result


# ---------------------------------------------------------------------------
# Token pasting edge cases
# ---------------------------------------------------------------------------

class TestTokenPasting:
	def test_basic_paste(self):
		pp = Preprocessor()
		src = "#define PASTE(a,b) a ## b\nPASTE(foo, bar)"
		result = pp.process(src)
		assert result.strip() == "foobar"

	def test_paste_with_numbers(self):
		pp = Preprocessor()
		src = "#define VAR(n) x ## n\nVAR(1)"
		result = pp.process(src)
		assert result.strip() == "x1"

	def test_paste_creates_identifier(self):
		"""Token pasting creates a new identifier that can be further expanded."""
		pp = Preprocessor()
		src = "#define AB 99\n#define PASTE(a,b) a ## b\nPASTE(A, B)"
		result = pp.process(src)
		assert result.strip() == "99"

	def test_paste_prefix(self):
		pp = Preprocessor()
		src = "#define PREFIX(name) my_ ## name\nPREFIX(func)"
		result = pp.process(src)
		assert result.strip() == "my_func"

	def test_paste_suffix(self):
		pp = Preprocessor()
		src = "#define SUFFIX(name) name ## _t\nSUFFIX(data)"
		result = pp.process(src)
		assert result.strip() == "data_t"

	def test_paste_in_object_macro(self):
		pp = Preprocessor()
		src = "#define JOINED foo ## bar\nJOINED"
		result = pp.process(src)
		assert result.strip() == "foobar"

	def test_paste_empty_side(self):
		"""Pasting where one side is effectively empty."""
		pp = Preprocessor()
		src = "#define PASTE(a,b) a ## b\nPASTE(, hello)"
		result = pp.process(src)
		assert "hello" in result.strip()


# ---------------------------------------------------------------------------
# Recursive include guards
# ---------------------------------------------------------------------------

class TestIncludeGuards:
	def test_ifndef_include_guard(self, tmp_path):
		header = tmp_path / "guarded.h"
		header.write_text(
			"#ifndef GUARDED_H\n"
			"#define GUARDED_H\n"
			"int guarded_val;\n"
			"#endif\n"
		)
		pp = Preprocessor(include_paths=[str(tmp_path)])
		src = '#include "guarded.h"\n#include "guarded.h"\n'
		result = pp.process(src, filename=str(tmp_path / "main.c"))
		# Should only appear once due to include guard
		assert result.count("int guarded_val;") == 1

	def test_mutual_include(self, tmp_path):
		"""Two headers that include each other should not infinite loop."""
		a_h = tmp_path / "a.h"
		b_h = tmp_path / "b.h"
		a_h.write_text('#ifndef A_H\n#define A_H\n#include "b.h"\nint a_val;\n#endif\n')
		b_h.write_text('#ifndef B_H\n#define B_H\n#include "a.h"\nint b_val;\n#endif\n')
		pp = Preprocessor(include_paths=[str(tmp_path)])
		src = '#include "a.h"\n'
		result = pp.process(src, filename=str(tmp_path / "main.c"))
		assert "a_val" in result
		assert "b_val" in result


# ---------------------------------------------------------------------------
# Pragma once
# ---------------------------------------------------------------------------

class TestPragmaOnce:
	def test_pragma_once_prevents_double_include(self, tmp_path):
		header = tmp_path / "once.h"
		header.write_text("#pragma once\nint once_val;\n")
		pp = Preprocessor(include_paths=[str(tmp_path)])
		src = '#include "once.h"\n#include "once.h"\n'
		result = pp.process(src, filename=str(tmp_path / "main.c"))
		assert result.count("int once_val;") == 1

	def test_pragma_once_ignored_for_stdin(self):
		"""#pragma once in stdin should be silently ignored (no crash)."""
		pp = Preprocessor()
		src = "#pragma once\nint x;"
		result = pp.process(src)
		assert "int x;" in result

	def test_pragma_once_different_files(self, tmp_path):
		"""Two different files with pragma once should both be included."""
		h1 = tmp_path / "h1.h"
		h2 = tmp_path / "h2.h"
		h1.write_text("#pragma once\nint h1_val;\n")
		h2.write_text("#pragma once\nint h2_val;\n")
		pp = Preprocessor(include_paths=[str(tmp_path)])
		src = '#include "h1.h"\n#include "h2.h"\n'
		result = pp.process(src, filename=str(tmp_path / "main.c"))
		assert "h1_val" in result
		assert "h2_val" in result


# ---------------------------------------------------------------------------
# Error edge cases
# ---------------------------------------------------------------------------

class TestPreprocessorErrors:
	def test_unterminated_if(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="Unterminated"):
			pp.process("#if 1\nstuff")

	def test_else_without_if(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="#else without"):
			pp.process("#else\n")

	def test_endif_without_if(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="#endif without"):
			pp.process("#endif\n")

	def test_elif_without_if(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="#elif without"):
			pp.process("#elif 1\n")

	def test_duplicate_else(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="Duplicate #else"):
			pp.process("#if 1\n#else\n#else\n#endif")

	def test_error_directive(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="#error"):
			pp.process('#error "stop here"')

	def test_empty_define(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError):
			pp.process("#define\n")

	def test_wrong_arg_count(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="expects 2 arguments"):
			pp.process("#define F(a,b) a+b\nF(1)")


# ---------------------------------------------------------------------------
# Miscellaneous edge cases
# ---------------------------------------------------------------------------

class TestMiscEdgeCases:
	def test_continuation_lines(self):
		pp = Preprocessor()
		src = "#define LONG_MACRO \\\n  42\nLONG_MACRO"
		result = pp.process(src)
		assert "42" in result

	def test_empty_directive(self):
		"""A lone # on a line should be allowed (null directive)."""
		pp = Preprocessor()
		result = pp.process("#\nint x;")
		assert "int x;" in result

	def test_predefined_macros(self):
		pp = Preprocessor(predefined_macros={"VERSION": "3"})
		result = pp.process("VERSION")
		assert result.strip() == "3"

	def test_undef_then_redefine(self):
		pp = Preprocessor()
		src = "#define X 1\n#undef X\n#define X 2\nX"
		result = pp.process(src)
		assert result.strip() == "2"

	def test_ifdef_after_undef(self):
		pp = Preprocessor()
		src = "#define X\n#undef X\n#ifdef X\nyes\n#else\nno\n#endif"
		result = pp.process(src)
		assert "no" in result

	def test_builtin_line_macro(self):
		pp = Preprocessor()
		src = "a\nb\n__LINE__"
		result = pp.process(src)
		lines = result.strip().split("\n")
		assert lines[-1].strip() == "3"

	def test_builtin_file_macro(self):
		pp = Preprocessor()
		src = '__FILE__'
		result = pp.process(src, filename="test.c")
		assert '"test.c"' in result

	def test_comments_stripped(self):
		pp = Preprocessor()
		src = "int x; // comment\nint y; /* block */\n"
		result = pp.process(src)
		assert "// comment" not in result
		assert "/* block */" not in result
		assert "int x;" in result
		assert "int y;" in result

	def test_string_not_expanded(self):
		"""Macros inside string literals should not be expanded."""
		pp = Preprocessor()
		src = '#define FOO bar\n"FOO"'
		result = pp.process(src)
		assert '"FOO"' in result

	def test_warning_directive(self):
		pp = Preprocessor()
		pp.process("#warning test message")
		assert len(pp.warnings) == 1
		assert "test message" in pp.warnings[0]
