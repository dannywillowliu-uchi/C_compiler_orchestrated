"""Tricky edge-case tests for preprocessor macro expansion and conditionals.

Covers: recursive macro expansion limits, macro with empty body,
function-like macro with 0 args, variadic macro with 0 variadic args,
nested #ifdef/#ifndef, #if with complex constant expressions (&&, ||, !),
#elif chains, redefinition of macros, __LINE__ and __FILE__ accuracy,
token pasting edge cases, stringification of special characters,
include of builtin stdbool.h and stddef.h.
"""

import pytest

from compiler.preprocessor import Preprocessor, PreprocessorError


# ---------------------------------------------------------------------------
# Recursive macro expansion limits
# ---------------------------------------------------------------------------

class TestRecursiveMacroExpansionLimits:
	def test_self_referencing_macro_not_infinite(self):
		"""A macro referencing itself should not expand infinitely."""
		pp = Preprocessor()
		src = "#define FOO FOO\nFOO"
		result = pp.process(src)
		assert "FOO" in result.strip()

	def test_mutual_recursion_terminates(self):
		"""Two macros referencing each other should terminate."""
		pp = Preprocessor()
		src = "#define A B\n#define B A\nA"
		result = pp.process(src)
		# Should terminate without error; the unexpanded token remains
		assert result.strip() in ("A", "B")

	def test_deep_chain_expansion(self):
		"""A long chain of macros should expand fully."""
		pp = Preprocessor()
		lines = []
		for i in range(20):
			lines.append(f"#define M{i} M{i + 1}")
		lines.append("#define M20 999")
		lines.append("M0")
		result = pp.process("\n".join(lines))
		assert result.strip() == "999"

	def test_indirect_self_reference(self):
		"""A -> B -> C -> A should not infinite loop."""
		pp = Preprocessor()
		src = "#define A B\n#define B C\n#define C A\nA"
		result = pp.process(src)
		# Should terminate; the unexpanded token remains
		assert result.strip() in ("A", "B", "C")


# ---------------------------------------------------------------------------
# Macro with empty body
# ---------------------------------------------------------------------------

class TestMacroEmptyBody:
	def test_object_macro_empty_body(self):
		pp = Preprocessor()
		src = "#define EMPTY\nint x = EMPTY 5;"
		result = pp.process(src)
		assert "5" in result.strip()
		# EMPTY should expand to nothing
		assert "EMPTY" not in result.strip()

	def test_function_macro_empty_body(self):
		pp = Preprocessor()
		src = "#define NOP(x)\nNOP(42) int y;"
		result = pp.process(src)
		assert "int y;" in result.strip()
		assert "42" not in result.strip()

	def test_empty_body_in_expression(self):
		pp = Preprocessor()
		src = "#define NOTHING\nint x = 1 NOTHING + 2;"
		result = pp.process(src)
		assert "1" in result.strip()
		assert "NOTHING" not in result.strip()


# ---------------------------------------------------------------------------
# Function-like macro with 0 args
# ---------------------------------------------------------------------------

class TestFunctionMacroZeroArgs:
	def test_zero_arg_function_macro(self):
		pp = Preprocessor()
		src = "#define GETVAL() 42\nint x = GETVAL();"
		result = pp.process(src)
		assert "42" in result.strip()

	def test_zero_arg_macro_without_parens_no_expand(self):
		"""GETVAL without () should NOT expand for function-like macros."""
		pp = Preprocessor()
		src = "#define GETVAL() 42\nint x = GETVAL;"
		result = pp.process(src)
		assert "GETVAL" in result.strip()

	def test_zero_arg_macro_with_whitespace(self):
		pp = Preprocessor()
		src = "#define F() 100\nF(  )"
		result = pp.process(src)
		assert result.strip() == "100"


# ---------------------------------------------------------------------------
# Variadic macro with 0 variadic args
# ---------------------------------------------------------------------------

class TestVariadicMacroZeroArgs:
	def test_variadic_with_no_variadic_args(self):
		pp = Preprocessor()
		src = "#define LOG(fmt, ...) printf(fmt, __VA_ARGS__)\nLOG(\"hello\")"
		result = pp.process(src)
		assert "printf" in result.strip()

	def test_variadic_with_one_variadic_arg(self):
		pp = Preprocessor()
		src = "#define LOG(fmt, ...) printf(fmt, __VA_ARGS__)\nLOG(\"hello\", 42)"
		result = pp.process(src)
		assert "42" in result.strip()

	def test_variadic_with_multiple_variadic_args(self):
		pp = Preprocessor()
		src = "#define LOG(fmt, ...) printf(fmt, __VA_ARGS__)\nLOG(\"hello\", 1, 2, 3)"
		result = pp.process(src)
		assert "1, 2, 3" in result.strip()

	def test_variadic_only(self):
		"""Macro with only variadic param and no fixed params."""
		pp = Preprocessor()
		src = "#define ALL(...) __VA_ARGS__\nALL(a, b, c)"
		result = pp.process(src)
		assert "a, b, c" in result.strip()

	def test_variadic_only_zero_args(self):
		pp = Preprocessor()
		src = "#define ALL(...) __VA_ARGS__\nALL()"
		result = pp.process(src)
		# Should expand to empty
		assert result.strip() == ""


# ---------------------------------------------------------------------------
# Nested #ifdef / #ifndef
# ---------------------------------------------------------------------------

class TestNestedIfdefIfndef:
	def test_nested_ifdef_both_defined(self):
		pp = Preprocessor()
		src = "#define A\n#define B\n#ifdef A\n#ifdef B\nBOTH\n#endif\n#endif"
		result = pp.process(src)
		assert "BOTH" in result

	def test_nested_ifdef_outer_undefined(self):
		pp = Preprocessor()
		src = "#define B\n#ifdef A\n#ifdef B\nBOTH\n#endif\n#endif"
		result = pp.process(src)
		assert "BOTH" not in result

	def test_nested_ifdef_inner_undefined(self):
		pp = Preprocessor()
		src = "#define A\n#ifdef A\n#ifdef B\nBOTH\n#else\nONLY_A\n#endif\n#endif"
		result = pp.process(src)
		assert "ONLY_A" in result
		assert "BOTH" not in result

	def test_nested_ifndef_inside_ifdef(self):
		pp = Preprocessor()
		src = "#define A\n#ifdef A\n#ifndef B\nA_NOT_B\n#endif\n#endif"
		result = pp.process(src)
		assert "A_NOT_B" in result

	def test_deeply_nested_conditionals(self):
		pp = Preprocessor()
		src = (
			"#define X\n"
			"#ifdef X\n"
			"#ifndef Y\n"
			"#ifdef X\n"
			"DEEP\n"
			"#endif\n"
			"#endif\n"
			"#endif\n"
		)
		result = pp.process(src)
		assert "DEEP" in result

	def test_ifndef_with_define_inside(self):
		"""#ifndef guard pattern: define inside the guard."""
		pp = Preprocessor()
		src = "#ifndef GUARD\n#define GUARD\nCONTENT\n#endif\nOUTSIDE"
		result = pp.process(src)
		assert "CONTENT" in result
		assert "OUTSIDE" in result


# ---------------------------------------------------------------------------
# #if with complex constant expressions
# ---------------------------------------------------------------------------

class TestIfComplexExpressions:
	def test_if_logical_and_true(self):
		pp = Preprocessor()
		src = "#define A 1\n#define B 1\n#if A && B\nYES\n#endif"
		result = pp.process(src)
		assert "YES" in result

	def test_if_logical_and_false(self):
		pp = Preprocessor()
		src = "#define A 1\n#define B 0\n#if A && B\nYES\n#else\nNO\n#endif"
		result = pp.process(src)
		assert "NO" in result
		assert "YES" not in result

	def test_if_logical_or_true(self):
		pp = Preprocessor()
		src = "#define A 0\n#define B 1\n#if A || B\nYES\n#endif"
		result = pp.process(src)
		assert "YES" in result

	def test_if_logical_or_false(self):
		pp = Preprocessor()
		src = "#define A 0\n#define B 0\n#if A || B\nYES\n#else\nNO\n#endif"
		result = pp.process(src)
		assert "NO" in result

	def test_if_logical_not_true(self):
		pp = Preprocessor()
		src = "#define A 0\n#if !A\nYES\n#endif"
		result = pp.process(src)
		assert "YES" in result

	def test_if_logical_not_false(self):
		pp = Preprocessor()
		src = "#define A 1\n#if !A\nYES\n#else\nNO\n#endif"
		result = pp.process(src)
		assert "NO" in result

	def test_if_combined_and_or(self):
		pp = Preprocessor()
		src = "#define A 1\n#define B 0\n#define C 1\n#if A && B || C\nYES\n#endif"
		result = pp.process(src)
		assert "YES" in result

	def test_if_not_equal(self):
		pp = Preprocessor()
		src = "#define VER 2\n#if VER != 1\nYES\n#endif"
		result = pp.process(src)
		assert "YES" in result

	def test_if_parenthesized_expression(self):
		pp = Preprocessor()
		src = "#define A 1\n#define B 0\n#if (A || B) && !(A && B)\nXOR\n#endif"
		result = pp.process(src)
		assert "XOR" in result

	def test_if_arithmetic_comparison(self):
		pp = Preprocessor()
		src = "#define X 5\n#if X > 3\nBIG\n#endif"
		result = pp.process(src)
		assert "BIG" in result

	def test_if_defined_combined_with_value(self):
		pp = Preprocessor()
		src = "#define A 2\n#if defined(A) && A > 1\nYES\n#endif"
		result = pp.process(src)
		assert "YES" in result

	def test_if_undefined_identifier_is_zero(self):
		"""Undefined identifiers in #if should evaluate to 0 per C standard."""
		pp = Preprocessor()
		src = "#if UNDEFINED\nYES\n#else\nNO\n#endif"
		result = pp.process(src)
		assert "NO" in result

	def test_if_zero(self):
		pp = Preprocessor()
		src = "#if 0\nHIDDEN\n#endif"
		result = pp.process(src)
		assert "HIDDEN" not in result

	def test_if_one(self):
		pp = Preprocessor()
		src = "#if 1\nVISIBLE\n#endif"
		result = pp.process(src)
		assert "VISIBLE" in result


# ---------------------------------------------------------------------------
# #elif chains
# ---------------------------------------------------------------------------

class TestElifChains:
	def test_elif_first_true(self):
		pp = Preprocessor()
		src = "#define X 1\n#if X == 1\nFIRST\n#elif X == 2\nSECOND\n#else\nOTHER\n#endif"
		result = pp.process(src)
		assert "FIRST" in result
		assert "SECOND" not in result
		assert "OTHER" not in result

	def test_elif_second_true(self):
		pp = Preprocessor()
		src = "#define X 2\n#if X == 1\nFIRST\n#elif X == 2\nSECOND\n#else\nOTHER\n#endif"
		result = pp.process(src)
		assert "SECOND" in result
		assert "FIRST" not in result

	def test_elif_else_fallthrough(self):
		pp = Preprocessor()
		src = "#define X 3\n#if X == 1\nFIRST\n#elif X == 2\nSECOND\n#else\nOTHER\n#endif"
		result = pp.process(src)
		assert "OTHER" in result
		assert "FIRST" not in result
		assert "SECOND" not in result

	def test_multiple_elif(self):
		pp = Preprocessor()
		src = (
			"#define V 3\n"
			"#if V == 1\nONE\n"
			"#elif V == 2\nTWO\n"
			"#elif V == 3\nTHREE\n"
			"#elif V == 4\nFOUR\n"
			"#else\nOTHER\n"
			"#endif"
		)
		result = pp.process(src)
		assert "THREE" in result
		assert "ONE" not in result
		assert "TWO" not in result
		assert "FOUR" not in result

	def test_elif_only_first_match_taken(self):
		"""If multiple elif could match, only the first should be taken."""
		pp = Preprocessor()
		src = (
			"#define V 5\n"
			"#if V > 10\nA\n"
			"#elif V > 3\nB\n"
			"#elif V > 1\nC\n"
			"#else\nD\n"
			"#endif"
		)
		result = pp.process(src)
		assert "B" in result
		assert "C" not in result

	def test_elif_after_else_raises(self):
		pp = Preprocessor()
		src = "#if 1\nA\n#else\nB\n#elif 1\nC\n#endif"
		with pytest.raises(PreprocessorError, match="elif after #else"):
			pp.process(src)


# ---------------------------------------------------------------------------
# Redefinition of macros
# ---------------------------------------------------------------------------

class TestMacroRedefinition:
	def test_redefine_object_macro(self):
		pp = Preprocessor()
		src = "#define X 1\n#define X 2\nX"
		result = pp.process(src)
		assert result.strip() == "2"

	def test_redefine_to_different_type(self):
		"""Redefine from object-like to function-like."""
		pp = Preprocessor()
		src = "#define X 1\n#define X(a) (a+1)\nX(5)"
		result = pp.process(src)
		assert "(5+1)" in result.strip()

	def test_undef_and_redefine(self):
		pp = Preprocessor()
		src = "#define X 1\n#undef X\n#define X 99\nX"
		result = pp.process(src)
		assert result.strip() == "99"

	def test_undef_nonexistent_is_ok(self):
		pp = Preprocessor()
		src = "#undef DOESNT_EXIST\nOK"
		result = pp.process(src)
		assert "OK" in result

	def test_redefine_preserves_last(self):
		pp = Preprocessor()
		src = "#define V 1\n#define V 2\n#define V 3\nV"
		result = pp.process(src)
		assert result.strip() == "3"


# ---------------------------------------------------------------------------
# __LINE__ and __FILE__ accuracy
# ---------------------------------------------------------------------------

class TestLineAndFile:
	def test_line_basic(self):
		pp = Preprocessor()
		src = "line1\n__LINE__\nline3"
		result = pp.process(src)
		lines = result.strip().split("\n")
		assert "2" in lines[1]

	def test_line_after_directives(self):
		"""__LINE__ should reflect the actual source line number."""
		pp = Preprocessor()
		src = "#define X 1\n#define Y 2\n__LINE__"
		result = pp.process(src)
		# Line 3 is where __LINE__ is
		non_empty = [line for line in result.split("\n") if line.strip()]
		assert "3" in non_empty[-1]

	def test_file_default(self):
		pp = Preprocessor()
		src = "__FILE__"
		result = pp.process(src)
		assert '"<stdin>"' in result

	def test_file_custom(self):
		pp = Preprocessor()
		src = "__FILE__"
		result = pp.process(src, filename="test.c")
		assert '"test.c"' in result

	def test_line_in_macro_expansion(self):
		"""__LINE__ in macro body should use the expansion site line."""
		pp = Preprocessor()
		src = "#define WHERE __LINE__\nfirst\nWHERE"
		result = pp.process(src)
		lines = result.strip().split("\n")
		# WHERE is on line 3
		non_empty = [line for line in lines if line.strip()]
		assert "3" in non_empty[-1]

	def test_file_in_macro_expansion(self):
		pp = Preprocessor()
		src = '#define FNAME __FILE__\nFNAME'
		result = pp.process(src, filename="myfile.c")
		assert '"myfile.c"' in result


# ---------------------------------------------------------------------------
# Token pasting edge cases
# ---------------------------------------------------------------------------

class TestTokenPastingEdgeCases:
	def test_paste_identifier_parts(self):
		pp = Preprocessor()
		src = "#define PASTE(a, b) a ## b\nPASTE(foo, bar)"
		result = pp.process(src)
		assert "foobar" in result.strip()

	def test_paste_with_number(self):
		pp = Preprocessor()
		src = "#define VAR(n) x ## n\nVAR(1)"
		result = pp.process(src)
		assert "x1" in result.strip()

	def test_paste_empty_arg(self):
		"""Pasting with an empty argument should yield just the other token."""
		pp = Preprocessor()
		src = "#define PASTE(a, b) a ## b\nPASTE(prefix, )"
		result = pp.process(src)
		assert "prefix" in result.strip()

	def test_paste_in_object_macro(self):
		pp = Preprocessor()
		src = "#define JOINED foo ## bar\nJOINED"
		result = pp.process(src)
		assert "foobar" in result.strip()

	def test_double_paste(self):
		pp = Preprocessor()
		src = "#define TRIPLE(a, b, c) a ## b ## c\nTRIPLE(x, y, z)"
		result = pp.process(src)
		assert "xyz" in result.strip()

	def test_paste_preserves_surrounding(self):
		pp = Preprocessor()
		src = "#define MK(n) var_ ## n = 0\nMK(1);"
		result = pp.process(src)
		assert "var_1" in result.strip()


# ---------------------------------------------------------------------------
# Stringification of special characters
# ---------------------------------------------------------------------------

class TestStringificationSpecialChars:
	def test_stringify_basic(self):
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR(hello)'
		result = pp.process(src)
		assert '"hello"' in result.strip()

	def test_stringify_with_quotes(self):
		"""Stringifying text containing quotes should escape them."""
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR(say "hi")'
		result = pp.process(src)
		assert '\\"hi\\"' in result.strip()

	def test_stringify_with_backslash(self):
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR(a\\b)'
		result = pp.process(src)
		assert "\\\\" in result.strip()

	def test_stringify_with_spaces(self):
		pp = Preprocessor()
		src = '#define STR(x) #x\nSTR(  hello   world  )'
		result = pp.process(src)
		assert '"hello   world"' in result.strip()

	def test_stringify_number(self):
		pp = Preprocessor()
		src = "#define STR(x) #x\nSTR(42)"
		result = pp.process(src)
		assert '"42"' in result.strip()

	def test_stringify_expression(self):
		pp = Preprocessor()
		src = "#define STR(x) #x\nSTR(a + b)"
		result = pp.process(src)
		assert '"a + b"' in result.strip()

	def test_stringify_empty(self):
		pp = Preprocessor()
		src = "#define STR(x) #x\nSTR()"
		result = pp.process(src)
		assert '""' in result.strip()


# ---------------------------------------------------------------------------
# Include of builtin headers
# ---------------------------------------------------------------------------

class TestBuiltinIncludes:
	def test_include_stdbool_h(self):
		pp = Preprocessor()
		src = '#include <stdbool.h>\nbool x = true;'
		result = pp.process(src)
		assert "_Bool" in result
		assert "1" in result  # true -> 1

	def test_stdbool_defines_true_false(self):
		pp = Preprocessor()
		src = "#include <stdbool.h>\ntrue false"
		result = pp.process(src)
		non_empty = [line.strip() for line in result.split("\n") if line.strip()]
		assert "1 0" in non_empty[-1]

	def test_stdbool_bool_true_false_are_defined(self):
		pp = Preprocessor()
		src = "#include <stdbool.h>\n__bool_true_false_are_defined"
		result = pp.process(src)
		non_empty = [line.strip() for line in result.split("\n") if line.strip()]
		assert "1" in non_empty[-1]

	def test_stdbool_include_guard(self):
		"""Including stdbool.h twice should be safe (include guard)."""
		pp = Preprocessor()
		src = '#include <stdbool.h>\n#include <stdbool.h>\nbool x = true;'
		result = pp.process(src)
		assert "_Bool" in result

	def test_stdbool_quoted_include(self):
		pp = Preprocessor()
		src = '#include "stdbool.h"\nbool x = false;'
		result = pp.process(src)
		assert "0" in result


# ---------------------------------------------------------------------------
# Miscellaneous tricky cases
# ---------------------------------------------------------------------------

class TestMiscTrickyCases:
	def test_empty_if_expression_raises(self):
		pp = Preprocessor()
		src = "#if\nX\n#endif"
		with pytest.raises(PreprocessorError):
			pp.process(src)

	def test_unterminated_if_raises(self):
		pp = Preprocessor()
		src = "#ifdef FOO\nstuff"
		with pytest.raises(PreprocessorError, match="Unterminated"):
			pp.process(src)

	def test_endif_without_if_raises(self):
		pp = Preprocessor()
		src = "#endif"
		with pytest.raises(PreprocessorError):
			pp.process(src)

	def test_else_without_if_raises(self):
		pp = Preprocessor()
		src = "#else"
		with pytest.raises(PreprocessorError):
			pp.process(src)

	def test_double_else_raises(self):
		pp = Preprocessor()
		src = "#if 1\nA\n#else\nB\n#else\nC\n#endif"
		with pytest.raises(PreprocessorError, match="Duplicate #else"):
			pp.process(src)

	def test_define_no_name_raises(self):
		pp = Preprocessor()
		src = "#define"
		with pytest.raises(PreprocessorError):
			pp.process(src)

	def test_continuation_line_in_define(self):
		pp = Preprocessor()
		src = "#define MULTI 1 + \\\n2\nMULTI"
		result = pp.process(src)
		assert "1 + 2" in result.strip()

	def test_empty_directive_line(self):
		"""A line with just '#' and nothing else should be accepted."""
		pp = Preprocessor()
		src = "#\nint x;"
		result = pp.process(src)
		assert "int x;" in result

	def test_predefined_macros(self):
		pp = Preprocessor(predefined_macros={"VERSION": "42"})
		src = "VERSION"
		result = pp.process(src)
		assert "42" in result.strip()

	def test_macro_expansion_in_ifdef(self):
		"""#ifdef should check macro name literally, not expand."""
		pp = Preprocessor()
		src = "#define A 1\n#ifdef A\nYES\n#endif"
		result = pp.process(src)
		assert "YES" in result

	def test_nested_function_macro_calls(self):
		pp = Preprocessor()
		src = "#define ADD(a,b) (a+b)\n#define MUL(a,b) (a*b)\nADD(MUL(2,3), 4)"
		result = pp.process(src)
		assert "((2*3)+4)" in result.strip()

	def test_macro_arg_with_commas_in_parens(self):
		"""Commas inside parentheses in macro args should not split args."""
		pp = Preprocessor()
		src = "#define F(x) x\nF(func(a, b))"
		result = pp.process(src)
		assert "func(a, b)" in result.strip()

	def test_if_with_defined_no_parens(self):
		pp = Preprocessor()
		src = "#define FOO\n#if defined FOO\nYES\n#endif"
		result = pp.process(src)
		assert "YES" in result

	def test_if_with_defined_parens(self):
		pp = Preprocessor()
		src = "#define FOO\n#if defined(FOO)\nYES\n#endif"
		result = pp.process(src)
		assert "YES" in result

	def test_if_not_defined(self):
		pp = Preprocessor()
		src = "#if !defined(BAR)\nYES\n#endif"
		result = pp.process(src)
		assert "YES" in result
