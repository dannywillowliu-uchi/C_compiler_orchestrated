"""Tests for string.h and assert.h builtin headers in the preprocessor."""

from compiler.preprocessor import Preprocessor


class TestStringHeader:
	"""Tests for #include <string.h> builtin header."""

	def test_include_string_h(self):
		pp = Preprocessor()
		result = pp.process('#include <string.h>\n')
		assert "size_t" in result

	def test_strlen_declared(self):
		pp = Preprocessor()
		result = pp.process('#include <string.h>\n')
		assert "strlen" in result

	def test_strcpy_declared(self):
		pp = Preprocessor()
		result = pp.process('#include <string.h>\n')
		assert "strcpy" in result

	def test_strcmp_declared(self):
		pp = Preprocessor()
		result = pp.process('#include <string.h>\n')
		assert "strcmp" in result

	def test_memcpy_declared(self):
		pp = Preprocessor()
		result = pp.process('#include <string.h>\n')
		assert "memcpy" in result

	def test_memset_declared(self):
		pp = Preprocessor()
		result = pp.process('#include <string.h>\n')
		assert "memset" in result

	def test_strncpy_declared(self):
		pp = Preprocessor()
		result = pp.process('#include <string.h>\n')
		assert "strncpy" in result

	def test_strncmp_declared(self):
		pp = Preprocessor()
		result = pp.process('#include <string.h>\n')
		assert "strncmp" in result

	def test_memmove_declared(self):
		pp = Preprocessor()
		result = pp.process('#include <string.h>\n')
		assert "memmove" in result

	def test_null_defined(self):
		pp = Preprocessor()
		result = pp.process('#include <string.h>\nNULL\n')
		assert "((void *)0)" in result

	def test_include_guard(self):
		pp = Preprocessor()
		source = '#include <string.h>\n#include <string.h>\n'
		result = pp.process(source)
		assert result.count("size_t strlen") == 1

	def test_memcmp_declared(self):
		pp = Preprocessor()
		result = pp.process('#include <string.h>\n')
		assert "memcmp" in result

	def test_strcat_declared(self):
		pp = Preprocessor()
		result = pp.process('#include <string.h>\n')
		assert "strcat" in result

	def test_strchr_declared(self):
		pp = Preprocessor()
		result = pp.process('#include <string.h>\n')
		assert "strchr" in result

	def test_strstr_declared(self):
		pp = Preprocessor()
		result = pp.process('#include <string.h>\n')
		assert "strstr" in result


class TestAssertHeader:
	"""Tests for #include <assert.h> builtin header."""

	def test_include_assert_h(self):
		pp = Preprocessor()
		result = pp.process('#include <assert.h>\n')
		assert result is not None

	def test_assert_macro_defined(self):
		pp = Preprocessor()
		result = pp.process('#include <assert.h>\nassert(1)\n')
		assert "((void)0)" in result

	def test_assert_macro_with_expression(self):
		pp = Preprocessor()
		result = pp.process('#include <assert.h>\nassert(x > 0)\n')
		assert "((void)0)" in result

	def test_include_guard(self):
		pp = Preprocessor()
		source = '#include <assert.h>\n#include <assert.h>\n'
		result = pp.process(source)
		# Should not error on double include
		assert result is not None

	def test_assert_in_code_context(self):
		pp = Preprocessor()
		source = '#include <assert.h>\nint foo(int x) { assert(x); return x; }\n'
		result = pp.process(source)
		assert "((void)0)" in result
		assert "return x" in result


class TestStringAndAssertTogether:
	"""Tests for including both string.h and assert.h."""

	def test_include_both(self):
		pp = Preprocessor()
		source = '#include <string.h>\n#include <assert.h>\n'
		result = pp.process(source)
		assert "strlen" in result

	def test_use_both(self):
		pp = Preprocessor()
		source = (
			'#include <string.h>\n'
			'#include <assert.h>\n'
			'assert(1)\n'
		)
		result = pp.process(source)
		assert "strlen" in result
		assert "((void)0)" in result
