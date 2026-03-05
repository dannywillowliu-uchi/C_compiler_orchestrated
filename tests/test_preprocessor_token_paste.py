"""Tests for token pasting (##), nested macro expansion, stringification edge cases,
variadic macros, empty arguments, macro redefinition, and #define/#include interaction.
"""

import os
import tempfile

import pytest

from compiler.preprocessor import Preprocessor, PreprocessorError


# ---------------------------------------------------------------------------
# ## Token pasting operator
# ---------------------------------------------------------------------------

class TestTokenPasting:
	def test_basic_paste_two_identifiers(self):
		pp = Preprocessor()
		src = "#define PASTE(a, b) a ## b\nint PASTE(foo, bar);"
		result = pp.process(src)
		assert "foobar" in result

	def test_paste_identifier_with_number(self):
		pp = Preprocessor()
		src = "#define VAR(n) var_ ## n\nint VAR(42);"
		result = pp.process(src)
		assert "var_42" in result

	def test_paste_creates_valid_token(self):
		pp = Preprocessor()
		src = "#define MAKE(prefix, id) prefix ## _ ## id\nMAKE(my, func);"
		result = pp.process(src)
		assert "my_func" in result

	def test_paste_three_tokens(self):
		pp = Preprocessor()
		src = "#define JOIN3(a, b, c) a ## b ## c\nJOIN3(x, y, z);"
		result = pp.process(src)
		assert "xyz" in result

	def test_paste_with_empty_arg(self):
		pp = Preprocessor()
		src = "#define PASTE(a, b) a ## b\nPASTE(hello, );"
		result = pp.process(src)
		assert "hello" in result

	def test_paste_in_object_macro(self):
		pp = Preprocessor()
		src = "#define GLUED ab ## cd\nGLUED;"
		result = pp.process(src)
		assert "abcd" in result

	def test_paste_preserves_surrounding_text(self):
		pp = Preprocessor()
		src = "#define PASTE(a, b) prefix_ ## a ## _ ## b ## _suffix\nPASTE(mid, end);"
		result = pp.process(src)
		assert "prefix_mid_end_suffix" in result

	def test_paste_inside_string_literal_not_expanded(self):
		"""## inside a string literal should not be treated as token pasting."""
		pp = Preprocessor()
		src = '#define STR "a ## b"\nSTR;'
		result = pp.process(src)
		assert '"a ## b"' in result


# ---------------------------------------------------------------------------
# Nested macro expansion
# ---------------------------------------------------------------------------

class TestNestedMacroExpansion:
	def test_simple_nested_expansion(self):
		pp = Preprocessor()
		src = "#define A 10\n#define B A + 5\nB"
		result = pp.process(src)
		assert "10 + 5" in result

	def test_deeply_nested_expansion(self):
		pp = Preprocessor()
		src = "#define L1 42\n#define L2 L1\n#define L3 L2\n#define L4 L3\nL4"
		result = pp.process(src)
		assert "42" in result

	def test_nested_function_macro(self):
		pp = Preprocessor()
		src = "#define ADD(a, b) (a + b)\n#define DOUBLE(x) ADD(x, x)\nDOUBLE(3)"
		result = pp.process(src)
		assert "(3 + 3)" in result

	def test_nested_with_different_macro_types(self):
		pp = Preprocessor()
		src = "#define VALUE 100\n#define WRAP(x) (x)\nWRAP(VALUE)"
		result = pp.process(src)
		assert "(100)" in result

	def test_macro_arg_expanded_before_substitution(self):
		pp = Preprocessor()
		src = "#define X 5\n#define IDENT(a) a\nIDENT(X)"
		result = pp.process(src)
		assert "5" in result

	def test_expansion_stops_at_self_reference(self):
		pp = Preprocessor()
		src = "#define LOOP LOOP\nLOOP"
		result = pp.process(src)
		assert "LOOP" in result

	def test_nested_function_in_object_macro(self):
		pp = Preprocessor()
		src = "#define INNER(x) (x * 2)\n#define OUTER INNER(7)\nOUTER"
		result = pp.process(src)
		assert "(7 * 2)" in result


# ---------------------------------------------------------------------------
# Macro expansion in #if conditions
# ---------------------------------------------------------------------------

class TestMacroInIfConditions:
	def test_if_with_defined_macro_value(self):
		pp = Preprocessor()
		src = "#define VER 3\n#if VER > 2\nyes\n#else\nno\n#endif"
		result = pp.process(src)
		lines = [line for line in result.split("\n") if line.strip()]
		assert "yes" in lines[-1]

	def test_if_with_nested_macro_in_condition(self):
		pp = Preprocessor()
		src = "#define A 1\n#define B A\n#if B\nyes\n#endif"
		result = pp.process(src)
		assert "yes" in result

	def test_if_undefined_macro_treated_as_zero(self):
		pp = Preprocessor()
		src = "#if UNDEFINED_MACRO\nyes\n#else\nno\n#endif"
		result = pp.process(src)
		lines = [line for line in result.split("\n") if line.strip()]
		assert "no" in lines[-1]

	def test_elif_with_macro_expansion(self):
		pp = Preprocessor()
		src = "#define MODE 2\n#if MODE == 1\na\n#elif MODE == 2\nb\n#else\nc\n#endif"
		result = pp.process(src)
		lines = [line for line in result.split("\n") if line.strip()]
		assert "b" in lines[-1]

	def test_if_with_defined_operator_and_macro(self):
		pp = Preprocessor()
		src = "#define FEAT 1\n#if defined(FEAT) && FEAT\nyes\n#endif"
		result = pp.process(src)
		assert "yes" in result

	def test_if_arithmetic_with_macros(self):
		pp = Preprocessor()
		src = "#define X 3\n#define Y 4\n#if X + Y == 7\nok\n#endif"
		result = pp.process(src)
		assert "ok" in result


# ---------------------------------------------------------------------------
# Stringification edge cases (#)
# ---------------------------------------------------------------------------

class TestStringificationEdgeCases:
	def test_basic_stringify(self):
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR(hello)'
		result = pp.process(src)
		assert '"hello"' in result

	def test_stringify_with_spaces(self):
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR(  hello  world  )'
		result = pp.process(src)
		assert '"hello  world"' in result

	def test_stringify_number(self):
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR(42)'
		result = pp.process(src)
		assert '"42"' in result

	def test_stringify_expression(self):
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR(a + b * c)'
		result = pp.process(src)
		assert '"a + b * c"' in result

	def test_stringify_with_quotes_escaped(self):
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR(say "hi")'
		result = pp.process(src)
		assert r'\"hi\"' in result

	def test_stringify_empty_arg(self):
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR()'
		result = pp.process(src)
		assert '""' in result

	def test_stringify_combined_with_paste(self):
		"""# and ## used in the same macro."""
		pp = Preprocessor()
		src = '#define COMBO(a, b) #a " and " #b\nCOMBO(hello, world)'
		result = pp.process(src)
		assert '"hello"' in result
		assert '"world"' in result

	def test_stringify_preserves_backslash(self):
		pp = Preprocessor()
		src = r'#define STR(x) #x' + "\n" + r'STR(path\to\file)'
		result = pp.process(src)
		assert r"\\to" in result or r"\to" in result


# ---------------------------------------------------------------------------
# Variadic macros with __VA_ARGS__
# ---------------------------------------------------------------------------

class TestVariadicMacros:
	def test_basic_variadic(self):
		pp = Preprocessor()
		src = "#define LOG(...) printf(__VA_ARGS__)\nLOG(\"hello %d\", 42)"
		result = pp.process(src)
		assert 'printf("hello %d", 42)' in result

	def test_variadic_with_named_params(self):
		pp = Preprocessor()
		src = '#define LOG(fmt, ...) printf(fmt, __VA_ARGS__)\nLOG("x=%d", x)'
		result = pp.process(src)
		assert 'printf("x=%d", x)' in result

	def test_variadic_multiple_extra_args(self):
		pp = Preprocessor()
		src = '#define LOG(fmt, ...) printf(fmt, __VA_ARGS__)\nLOG("a%d%d", 1, 2)'
		result = pp.process(src)
		assert 'printf("a%d%d", 1, 2)' in result

	def test_variadic_no_extra_args(self):
		pp = Preprocessor()
		src = "#define LOG(...) printf(__VA_ARGS__)\nLOG(\"hello\")"
		result = pp.process(src)
		assert 'printf("hello")' in result

	def test_variadic_only_ellipsis_multiple(self):
		pp = Preprocessor()
		src = "#define WRAP(...) f(__VA_ARGS__)\nWRAP(a, b, c)"
		result = pp.process(src)
		assert "f(a, b, c)" in result

	def test_variadic_too_few_args_error(self):
		pp = Preprocessor()
		src = '#define LOG(fmt, x, ...) printf(fmt, x, __VA_ARGS__)\nLOG("hi")'
		with pytest.raises(PreprocessorError, match="at least"):
			pp.process(src)


# ---------------------------------------------------------------------------
# Empty macro arguments
# ---------------------------------------------------------------------------

class TestEmptyMacroArgs:
	def test_empty_single_arg(self):
		pp = Preprocessor()
		src = "#define F(x) [x]\nF()"
		result = pp.process(src)
		assert "[]" in result

	def test_empty_first_of_two_args(self):
		pp = Preprocessor()
		src = "#define F(a, b) a-b\nF(, world)"
		result = pp.process(src)
		assert "-world" in result

	def test_empty_second_of_two_args(self):
		pp = Preprocessor()
		src = "#define F(a, b) a-b\nF(hello, )"
		result = pp.process(src)
		assert "hello-" in result

	def test_both_args_empty(self):
		pp = Preprocessor()
		src = "#define F(a, b) [a][b]\nF(, )"
		result = pp.process(src)
		assert "[][]" in result

	def test_empty_arg_with_paste(self):
		pp = Preprocessor()
		src = "#define PASTE(a, b) a ## b\nPASTE(, suffix)"
		result = pp.process(src)
		assert "suffix" in result

	def test_empty_body_macro(self):
		pp = Preprocessor()
		src = "#define EMPTY\nbeforeEMPTYafter"
		result = pp.process(src)
		# EMPTY expands to nothing, but only on word boundaries
		assert "before" in result


# ---------------------------------------------------------------------------
# Macro redefinition warnings
# ---------------------------------------------------------------------------

class TestMacroRedefinition:
	def test_redefine_same_value_no_issue(self):
		"""Redefining a macro with the same body should work."""
		pp = Preprocessor()
		src = "#define FOO 1\n#define FOO 1\nFOO"
		result = pp.process(src)
		assert "1" in result

	def test_redefine_different_value(self):
		"""Redefining with a different value should use the new value."""
		pp = Preprocessor()
		src = "#define FOO 1\n#define FOO 2\nFOO"
		result = pp.process(src)
		assert "2" in result

	def test_undef_then_redefine(self):
		pp = Preprocessor()
		src = "#define FOO 1\n#undef FOO\n#define FOO 2\nFOO"
		result = pp.process(src)
		assert "2" in result

	def test_redefine_object_to_function(self):
		pp = Preprocessor()
		src = "#define FOO 1\n#define FOO(x) x\nFOO(42)"
		result = pp.process(src)
		assert "42" in result

	def test_redefine_function_to_object(self):
		pp = Preprocessor()
		src = "#define FOO(x) x\n#define FOO 99\nFOO"
		result = pp.process(src)
		assert "99" in result


# ---------------------------------------------------------------------------
# Interaction between #define and #include
# ---------------------------------------------------------------------------

class TestDefineIncludeInteraction:
	def test_include_sees_prior_defines(self):
		with tempfile.NamedTemporaryFile(mode="w", suffix=".h", delete=False) as f:
			f.write("int x = VALUE;\n")
			f.flush()
			header_path = f.name
		try:
			pp = Preprocessor()
			src = f'#define VALUE 42\n#include "{header_path}"'
			result = pp.process(src)
			assert "int x = 42;" in result
		finally:
			os.unlink(header_path)

	def test_define_in_include_visible_after(self):
		with tempfile.NamedTemporaryFile(mode="w", suffix=".h", delete=False) as f:
			f.write("#define INCLUDED_VAL 99\n")
			f.flush()
			header_path = f.name
		try:
			pp = Preprocessor()
			src = f'#include "{header_path}"\nint x = INCLUDED_VAL;'
			result = pp.process(src)
			assert "int x = 99;" in result
		finally:
			os.unlink(header_path)

	def test_include_guard_prevents_double_include(self):
		with tempfile.NamedTemporaryFile(mode="w", suffix=".h", delete=False) as f:
			f.write("#ifndef MY_GUARD\n#define MY_GUARD\nint guarded;\n#endif\n")
			f.flush()
			header_path = f.name
		try:
			pp = Preprocessor()
			src = f'#include "{header_path}"\n#include "{header_path}"\nint after;'
			result = pp.process(src)
			assert result.count("int guarded;") == 1
		finally:
			os.unlink(header_path)

	def test_undef_after_include(self):
		with tempfile.NamedTemporaryFile(mode="w", suffix=".h", delete=False) as f:
			f.write("#define TEMP 123\n")
			f.flush()
			header_path = f.name
		try:
			pp = Preprocessor()
			src = f'#include "{header_path}"\n#undef TEMP\n#ifdef TEMP\nyes\n#else\nno\n#endif'
			result = pp.process(src)
			lines = [line for line in result.split("\n") if line.strip()]
			assert "no" in lines[-1]
		finally:
			os.unlink(header_path)

	def test_macro_expanded_include_path(self):
		"""Macro-expanded include argument."""
		with tempfile.NamedTemporaryFile(mode="w", suffix=".h", delete=False) as f:
			f.write("int included_val = 1;\n")
			f.flush()
			header_path = f.name
		try:
			pp = Preprocessor()
			src = f'#define HEADER "{header_path}"\n#include HEADER'
			result = pp.process(src)
			assert "int included_val = 1;" in result
		finally:
			os.unlink(header_path)

	def test_builtin_header_stdbool(self):
		pp = Preprocessor()
		src = '#include <stdbool.h>\nbool x = true;'
		result = pp.process(src)
		assert "_Bool" in result
		assert "1" in result


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------

class TestAdditionalEdgeCases:
	def test_paste_with_macro_arg_that_is_macro(self):
		"""Token pasting where args are themselves macros - paste happens before expansion."""
		pp = Preprocessor()
		src = "#define A x\n#define PASTE(a, b) a ## b\nPASTE(A, B)"
		result = pp.process(src)
		assert "AB" in result

	def test_multiple_stringify_in_one_macro(self):
		pp = Preprocessor()
		src = '#define PAIR(a, b) #a "=" #b\nPAIR(key, val)'
		result = pp.process(src)
		assert '"key"' in result
		assert '"val"' in result

	def test_empty_body_function_macro(self):
		pp = Preprocessor()
		src = "#define NOOP(x)\nint y = NOOP(42) 5;"
		result = pp.process(src)
		assert "int y =  5;" in result

	def test_predefined_macros_available(self):
		pp = Preprocessor(predefined_macros={"VERSION": "3"})
		src = "#if VERSION >= 3\nyes\n#endif"
		result = pp.process(src)
		assert "yes" in result

	def test_continuation_line_in_define(self):
		pp = Preprocessor()
		src = "#define MULTI a + \\\nb + c\nMULTI"
		result = pp.process(src)
		assert "a + b + c" in result

	def test_nested_paste_via_indirection(self):
		"""Paste via an intermediate macro."""
		pp = Preprocessor()
		src = "#define GLUE(a, b) a ## b\n#define MAKE(n) GLUE(item_, n)\nMAKE(5)"
		result = pp.process(src)
		assert "item_5" in result

	def test_chained_function_macros(self):
		pp = Preprocessor()
		src = "#define A(x) (x)\n#define B(x) A(x + 1)\n#define C(x) B(x + 2)\nC(0)"
		result = pp.process(src)
		assert "(0 + 2 + 1)" in result

	def test_if_with_logical_operators_and_macros(self):
		pp = Preprocessor()
		src = "#define A 1\n#define B 0\n#if A && !B\nyes\n#endif"
		result = pp.process(src)
		assert "yes" in result

	def test_function_macro_wrong_arg_count(self):
		pp = Preprocessor()
		src = "#define F(a, b) a + b\nF(1)"
		with pytest.raises(PreprocessorError, match="expects 2"):
			pp.process(src)

	def test_unterminated_macro_args(self):
		pp = Preprocessor()
		src = "#define F(a) a\nF(oops"
		with pytest.raises(PreprocessorError, match="Unterminated"):
			pp.process(src)
