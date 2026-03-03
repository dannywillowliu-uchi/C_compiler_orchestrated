"""Tests for preprocessor # (stringification) and ## (token pasting) operators."""

import textwrap

from compiler.preprocessor import Preprocessor


def preprocess(source: str) -> str:
	pp = Preprocessor()
	return pp.process(textwrap.dedent(source))


def preprocess_lines(source: str) -> list[str]:
	result = preprocess(source)
	return [line for line in result.split("\n") if line.strip()]


# ── Stringification (#) ─────────────────────────────────────────────────────


class TestStringification:
	def test_basic_stringification(self):
		lines = preprocess_lines("""\
			#define STR(x) #x
			STR(hello)
		""")
		assert any('"hello"' in line for line in lines)

	def test_stringify_multiple_tokens(self):
		lines = preprocess_lines("""\
			#define STR(x) #x
			STR(a + b)
		""")
		assert any('"a + b"' in line for line in lines)

	def test_stringify_empty_argument(self):
		lines = preprocess_lines("""\
			#define STR(x) #x
			STR()
		""")
		assert any('""' in line for line in lines)

	def test_stringify_with_quotes_in_argument(self):
		lines = preprocess_lines("""\
			#define STR(x) #x
			STR(hello "world")
		""")
		assert any('"hello \\"world\\""' in line for line in lines)

	def test_stringify_with_backslash_in_argument(self):
		result = preprocess("""\
			#define STR(x) #x
			STR(path\\to\\file)
		""")
		assert '"path\\\\to\\\\file"' in result

	def test_stringify_preserves_other_params(self):
		lines = preprocess_lines("""\
			#define FOO(x, y) #x + y
			FOO(hello, 42)
		""")
		assert any('"hello" + 42' in line for line in lines)

	def test_stringify_does_not_expand_argument(self):
		lines = preprocess_lines("""\
			#define X hello
			#define STR(a) #a
			STR(X)
		""")
		assert any('"X"' in line for line in lines)

	def test_stringify_with_whitespace_after_hash(self):
		lines = preprocess_lines("""\
			#define STR(x) # x
			STR(test)
		""")
		assert any('"test"' in line for line in lines)


# ── Token Pasting (##) ──────────────────────────────────────────────────────


class TestTokenPasting:
	def test_basic_concatenation(self):
		lines = preprocess_lines("""\
			#define CONCAT(a, b) a ## b
			CONCAT(foo, bar)
		""")
		assert any("foobar" in line for line in lines)

	def test_no_spaces_concatenation(self):
		lines = preprocess_lines("""\
			#define CONCAT(a, b) a##b
			CONCAT(foo, bar)
		""")
		assert any("foobar" in line for line in lines)

	def test_build_function_name(self):
		lines = preprocess_lines("""\
			#define FUNC(name) func_ ## name
			FUNC(init)
		""")
		assert any("func_init" in line for line in lines)

	def test_build_function_name_prefix(self):
		lines = preprocess_lines("""\
			#define MAKE(prefix, name) prefix ## _ ## name
			MAKE(my, function)
		""")
		assert any("my_function" in line for line in lines)

	def test_numeric_suffix(self):
		lines = preprocess_lines("""\
			#define VAR(n) var ## n
			VAR(1)
		""")
		assert any("var1" in line for line in lines)

	def test_numeric_suffix_multi_digit(self):
		lines = preprocess_lines("""\
			#define REG(n) r ## n
			REG(10)
		""")
		assert any("r10" in line for line in lines)

	def test_paste_at_start_of_body(self):
		lines = preprocess_lines("""\
			#define PREFIX(x) ## x
			PREFIX(hello)
		""")
		assert any("hello" in line for line in lines)

	def test_paste_at_end_of_body(self):
		lines = preprocess_lines("""\
			#define SUFFIX(x) x ##
			SUFFIX(hello)
		""")
		assert any("hello" in line for line in lines)

	def test_macro_expansion_after_pasting(self):
		lines = preprocess_lines("""\
			#define VALUE_X 42
			#define GET(name) VALUE_ ## name
			GET(X)
		""")
		assert any("42" in line for line in lines)

	def test_paste_creates_macro_then_expands(self):
		lines = preprocess_lines("""\
			#define AB 100
			#define PASTE(a, b) a ## b
			PASTE(A, B)
		""")
		assert any("100" in line for line in lines)


# ── Interaction with existing macro expansion ────────────────────────────────


class TestInteraction:
	def test_normal_expansion_still_works(self):
		lines = preprocess_lines("""\
			#define ADD(a, b) (a + b)
			ADD(1, 2)
		""")
		assert any("(1 + 2)" in line for line in lines)

	def test_stringify_and_paste_together(self):
		lines = preprocess_lines("""\
			#define BOTH(a, b) #a ## b
			BOTH(hello, _world)
		""")
		# #a becomes "hello", then ## pastes with _world
		assert any('"hello"_world' in line for line in lines)

	def test_nested_paste_with_expansion(self):
		lines = preprocess_lines("""\
			#define INNER 99
			#define OUTER(x) x
			#define MAKE(a, b) a ## b
			OUTER(MAKE(INN, ER))
		""")
		assert any("99" in line for line in lines)

	def test_paste_in_complex_expression(self):
		lines = preprocess_lines("""\
			#define FIELD(obj, member) obj ## . ## member
			FIELD(self, value)
		""")
		assert any("self.value" in line for line in lines)

	def test_variadic_with_stringify(self):
		lines = preprocess_lines("""\
			#define LOG(fmt, ...) printf(#fmt, __VA_ARGS__)
			LOG(hello %d, 42)
		""")
		assert any('printf("hello %d", 42)' in line for line in lines)

	def test_object_like_with_paste(self):
		lines = preprocess_lines("""\
			#define JOINED he ## llo
			JOINED
		""")
		assert any("hello" in line for line in lines)
