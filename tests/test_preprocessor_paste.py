"""Tests for preprocessor ## (token pasting) and # (stringification) operators.

Covers: basic stringification, basic token pasting, nested macro expansion
with paste/stringify, and edge cases (empty args, multiple ## in one macro).
"""

import textwrap

from compiler.preprocessor import Preprocessor


def preprocess(source: str) -> str:
	pp = Preprocessor()
	return pp.process(textwrap.dedent(source))


def preprocess_lines(source: str) -> list[str]:
	result = preprocess(source)
	return [line for line in result.split("\n") if line.strip()]


# ── Basic Stringification (#) ────────────────────────────────────────────────


class TestBasicStringification:
	def test_simple_stringify(self):
		lines = preprocess_lines("""\
			#define STR(x) #x
			STR(hello)
		""")
		assert any('"hello"' in line for line in lines)

	def test_stringify_expression(self):
		lines = preprocess_lines("""\
			#define STR(x) #x
			STR(a + b * c)
		""")
		assert any('"a + b * c"' in line for line in lines)

	def test_stringify_number(self):
		lines = preprocess_lines("""\
			#define STR(x) #x
			STR(42)
		""")
		assert any('"42"' in line for line in lines)

	def test_stringify_does_not_expand_macros(self):
		"""The # operator should stringify the raw argument, not the expanded form."""
		lines = preprocess_lines("""\
			#define VALUE 42
			#define STR(x) #x
			STR(VALUE)
		""")
		assert any('"VALUE"' in line for line in lines)

	def test_stringify_escapes_backslash(self):
		result = preprocess("""\
			#define STR(x) #x
			STR(a\\b)
		""")
		assert '"a\\\\b"' in result

	def test_stringify_escapes_embedded_quotes(self):
		lines = preprocess_lines("""\
			#define STR(x) #x
			STR(say "hi")
		""")
		assert any('"say \\"hi\\""' in line for line in lines)

	def test_stringify_preserves_other_params(self):
		lines = preprocess_lines("""\
			#define PAIR(a, b) #a, b
			PAIR(name, 42)
		""")
		assert any('"name", 42' in line for line in lines)

	def test_stringify_with_space_after_hash(self):
		"""Whitespace between # and param name is allowed."""
		lines = preprocess_lines("""\
			#define STR(x) # x
			STR(test)
		""")
		assert any('"test"' in line for line in lines)


# ── Basic Token Pasting (##) ────────────────────────────────────────────────


class TestBasicTokenPasting:
	def test_simple_paste(self):
		lines = preprocess_lines("""\
			#define CONCAT(a, b) a##b
			CONCAT(foo, bar)
		""")
		assert any("foobar" in line for line in lines)

	def test_paste_with_spaces(self):
		lines = preprocess_lines("""\
			#define CONCAT(a, b) a ## b
			CONCAT(foo, bar)
		""")
		assert any("foobar" in line for line in lines)

	def test_paste_builds_identifier(self):
		lines = preprocess_lines("""\
			#define MAKE_VAR(n) var_##n
			MAKE_VAR(count)
		""")
		assert any("var_count" in line for line in lines)

	def test_paste_number_suffix(self):
		lines = preprocess_lines("""\
			#define REG(n) r##n
			REG(15)
		""")
		assert any("r15" in line for line in lines)

	def test_paste_result_is_re_expanded(self):
		"""After pasting, the result should be subject to further macro expansion."""
		lines = preprocess_lines("""\
			#define AB 100
			#define PASTE(a, b) a##b
			PASTE(A, B)
		""")
		assert any("100" in line for line in lines)

	def test_paste_in_object_like_macro(self):
		lines = preprocess_lines("""\
			#define JOINED he##llo
			JOINED
		""")
		assert any("hello" in line for line in lines)


# ── Multiple ## in One Macro ─────────────────────────────────────────────────


class TestMultiplePasteOps:
	def test_two_paste_ops(self):
		lines = preprocess_lines("""\
			#define MAKE(a, b, c) a##_##b##_##c
			MAKE(my, func, v2)
		""")
		assert any("my_func_v2" in line for line in lines)

	def test_three_way_paste(self):
		lines = preprocess_lines("""\
			#define CAT3(a, b, c) a##b##c
			CAT3(x, y, z)
		""")
		assert any("xyz" in line for line in lines)

	def test_paste_prefix_and_suffix(self):
		lines = preprocess_lines("""\
			#define WRAP(prefix, name, suffix) prefix##_##name##_##suffix
			WRAP(get, value, impl)
		""")
		assert any("get_value_impl" in line for line in lines)


# ── Nested Macro Expansion with Paste/Stringify ──────────────────────────────


class TestNestedExpansion:
	def test_paste_then_expand(self):
		"""MAKE(INN, ER) -> INNER -> 99 via re-expansion."""
		lines = preprocess_lines("""\
			#define INNER 99
			#define MAKE(a, b) a##b
			MAKE(INN, ER)
		""")
		assert any("99" in line for line in lines)

	def test_outer_wrapping_paste(self):
		lines = preprocess_lines("""\
			#define INNER 99
			#define OUTER(x) x
			#define MAKE(a, b) a##b
			OUTER(MAKE(INN, ER))
		""")
		assert any("99" in line for line in lines)

	def test_stringify_result_of_paste(self):
		"""Stringify a paste result: inner macro expands first in text-based preprocessor."""
		lines = preprocess_lines("""\
			#define CONCAT(a, b) a##b
			#define STR(x) #x
			STR(CONCAT(foo, bar))
		""")
		# In our text-based preprocessor, CONCAT expands first, then STR stringifies
		assert any('"foobar"' in line for line in lines)

	def test_paste_inside_stringify_macro(self):
		"""A single macro that uses both # and ## on different params."""
		lines = preprocess_lines("""\
			#define MIXED(a, b) #a, a##b
			MIXED(hello, World)
		""")
		assert any('"hello"' in line for line in lines)
		assert any("helloWorld" in line for line in lines)

	def test_paste_creates_defined_macro(self):
		"""Paste creates a token that is itself a defined macro."""
		lines = preprocess_lines("""\
			#define XY 42
			#define GLUE(a, b) a##b
			int val = GLUE(X, Y);
		""")
		assert any("int val = 42;" in line for line in lines)

	def test_chained_macro_with_paste(self):
		"""One macro calls another that uses ##."""
		lines = preprocess_lines("""\
			#define FIELD(s, f) s##.##f
			#define GET(obj, member) FIELD(obj, member)
			GET(self, x)
		""")
		assert any("self.x" in line for line in lines)


# ── Edge Cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
	def test_stringify_empty_arg(self):
		lines = preprocess_lines("""\
			#define STR(x) #x
			STR()
		""")
		assert any('""' in line for line in lines)

	def test_paste_empty_left(self):
		"""Pasting with empty left operand should yield the right operand."""
		lines = preprocess_lines("""\
			#define PASTE(a, b) a##b
			PASTE(, hello)
		""")
		assert any("hello" in line for line in lines)

	def test_paste_empty_right(self):
		"""Pasting with empty right operand should yield the left operand."""
		lines = preprocess_lines("""\
			#define PASTE(a, b) a##b
			PASTE(hello, )
		""")
		assert any("hello" in line for line in lines)

	def test_paste_both_empty(self):
		"""Pasting two empty operands should produce nothing."""
		result = preprocess("""\
			#define PASTE(a, b) a##b
			PASTE(, )
		""")
		# Result should be blank/whitespace only on that line
		lines = [line for line in result.split("\n") if line.strip()]
		# No non-empty content lines beyond the #define
		assert len(lines) == 0 or all(line.strip() == "" for line in lines)

	def test_paste_at_body_start(self):
		lines = preprocess_lines("""\
			#define PREFIX(x) ##x
			PREFIX(hello)
		""")
		assert any("hello" in line for line in lines)

	def test_paste_at_body_end(self):
		lines = preprocess_lines("""\
			#define SUFFIX(x) x##
			SUFFIX(hello)
		""")
		assert any("hello" in line for line in lines)

	def test_variadic_with_stringify(self):
		lines = preprocess_lines("""\
			#define LOG(fmt, ...) printf(#fmt, __VA_ARGS__)
			LOG(hello %d, 42)
		""")
		assert any('printf("hello %d", 42)' in line for line in lines)

	def test_paste_preserves_string_literal(self):
		"""## inside a string literal should not be treated as token pasting."""
		lines = preprocess_lines("""\
			#define X "a##b"
			X
		""")
		assert any('"a##b"' in line for line in lines)

	def test_hash_in_string_not_stringify(self):
		"""# inside a string literal should not be treated as stringification."""
		lines = preprocess_lines("""\
			#define MSG "#hello"
			MSG
		""")
		assert any('"#hello"' in line for line in lines)

	def test_hash_non_param_is_preserved(self):
		"""# followed by a non-parameter token stays as #."""
		lines = preprocess_lines("""\
			#define FOO(x) #y
			FOO(test)
		""")
		assert any("#y" in line for line in lines)

	def test_multiple_stringify_same_param(self):
		lines = preprocess_lines("""\
			#define TWICE(x) #x #x
			TWICE(hi)
		""")
		assert any('"hi" "hi"' in line for line in lines)

	def test_paste_with_underscore(self):
		lines = preprocess_lines("""\
			#define NAME(prefix, suffix) prefix##_##suffix
			NAME(my, var)
		""")
		assert any("my_var" in line for line in lines)
