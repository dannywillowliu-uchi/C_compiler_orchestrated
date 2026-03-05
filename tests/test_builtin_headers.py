"""Tests for builtin header definitions (stddef.h, stdint.h, limits.h)."""

import pytest

from compiler.preprocessor import Preprocessor


@pytest.fixture
def pp() -> Preprocessor:
	return Preprocessor()


class TestStddefH:
	"""Tests for stddef.h builtin header."""

	def test_include_stddef(self, pp: Preprocessor) -> None:
		result = pp.process('#include <stddef.h>\nsize_t x;')
		assert "typedef unsigned long size_t;" in result
		assert "typedef long ptrdiff_t;" in result

	def test_null_defined(self, pp: Preprocessor) -> None:
		result = pp.process('#include <stddef.h>\nvoid *p = NULL;')
		assert "((void *)0)" in result

	def test_size_t_typedef(self, pp: Preprocessor) -> None:
		result = pp.process('#include <stddef.h>\nsize_t n = 42;')
		assert "size_t n = 42;" in result

	def test_ptrdiff_t_typedef(self, pp: Preprocessor) -> None:
		result = pp.process('#include <stddef.h>\nptrdiff_t d;')
		assert "ptrdiff_t d;" in result

	def test_wchar_t_typedef(self, pp: Preprocessor) -> None:
		result = pp.process('#include <stddef.h>\nwchar_t c;')
		assert "wchar_t c;" in result

	def test_offsetof_defined(self, pp: Preprocessor) -> None:
		pp.process('#include <stddef.h>')
		assert "offsetof" in pp.macros

	def test_include_guard(self, pp: Preprocessor) -> None:
		source = '#include <stddef.h>\n#include <stddef.h>\nsize_t x;'
		result = pp.process(source)
		count = result.count("typedef unsigned long size_t;")
		assert count == 1

	def test_quoted_include(self, pp: Preprocessor) -> None:
		result = pp.process('#include "stddef.h"\nsize_t x;')
		assert "typedef unsigned long size_t;" in result


class TestStdintH:
	"""Tests for stdint.h builtin header."""

	def test_include_stdint(self, pp: Preprocessor) -> None:
		result = pp.process('#include <stdint.h>')
		assert "typedef signed char int8_t;" in result
		assert "typedef short int16_t;" in result
		assert "typedef int int32_t;" in result
		assert "typedef long long int64_t;" in result

	def test_unsigned_types(self, pp: Preprocessor) -> None:
		result = pp.process('#include <stdint.h>')
		assert "typedef unsigned char uint8_t;" in result
		assert "typedef unsigned short uint16_t;" in result
		assert "typedef unsigned int uint32_t;" in result
		assert "typedef unsigned long long uint64_t;" in result

	def test_pointer_types(self, pp: Preprocessor) -> None:
		result = pp.process('#include <stdint.h>')
		assert "typedef long intptr_t;" in result
		assert "typedef unsigned long uintptr_t;" in result

	def test_max_types(self, pp: Preprocessor) -> None:
		result = pp.process('#include <stdint.h>')
		assert "typedef long long intmax_t;" in result
		assert "typedef unsigned long long uintmax_t;" in result

	def test_int_min_max_macros(self, pp: Preprocessor) -> None:
		result = pp.process('#include <stdint.h>\nint x = INT32_MAX;')
		assert "2147483647" in result

	def test_int8_limits(self, pp: Preprocessor) -> None:
		result = pp.process('#include <stdint.h>\nint a = INT8_MIN;\nint b = INT8_MAX;')
		assert "(-128)" in result
		assert "127" in result

	def test_int16_limits(self, pp: Preprocessor) -> None:
		result = pp.process('#include <stdint.h>\nint a = INT16_MIN;\nint b = INT16_MAX;')
		assert "(-32768)" in result
		assert "32767" in result

	def test_uint_max_macros(self, pp: Preprocessor) -> None:
		result = pp.process('#include <stdint.h>\nunsigned x = UINT8_MAX;\nunsigned y = UINT16_MAX;')
		assert "255" in result
		assert "65535" in result

	def test_uint32_max(self, pp: Preprocessor) -> None:
		result = pp.process('#include <stdint.h>\nunsigned x = UINT32_MAX;')
		assert "4294967295U" in result

	def test_size_max(self, pp: Preprocessor) -> None:
		result = pp.process('#include <stdint.h>\nunsigned long x = SIZE_MAX;')
		assert "18446744073709551615UL" in result

	def test_include_guard(self, pp: Preprocessor) -> None:
		source = '#include <stdint.h>\n#include <stdint.h>'
		result = pp.process(source)
		count = result.count("typedef signed char int8_t;")
		assert count == 1

	def test_use_types_in_code(self, pp: Preprocessor) -> None:
		source = '#include <stdint.h>\nint32_t foo(uint8_t bar) { return (int32_t)bar; }'
		result = pp.process(source)
		assert "int32_t foo(uint8_t bar)" in result


class TestLimitsH:
	"""Tests for limits.h builtin header."""

	def test_include_limits(self, pp: Preprocessor) -> None:
		pp.process('#include <limits.h>')
		assert "CHAR_BIT" in pp.macros
		assert "INT_MAX" in pp.macros

	def test_char_bit(self, pp: Preprocessor) -> None:
		result = pp.process('#include <limits.h>\nint x = CHAR_BIT;')
		assert "int x = 8;" in result

	def test_char_min_max(self, pp: Preprocessor) -> None:
		result = pp.process('#include <limits.h>\nint a = CHAR_MIN;\nint b = CHAR_MAX;')
		assert "(-128)" in result
		assert "127" in result

	def test_schar_limits(self, pp: Preprocessor) -> None:
		result = pp.process('#include <limits.h>\nint a = SCHAR_MIN;\nint b = SCHAR_MAX;')
		assert "(-128)" in result
		assert "127" in result

	def test_uchar_max(self, pp: Preprocessor) -> None:
		result = pp.process('#include <limits.h>\nint x = UCHAR_MAX;')
		assert "255" in result

	def test_short_limits(self, pp: Preprocessor) -> None:
		result = pp.process('#include <limits.h>\nint a = SHRT_MIN;\nint b = SHRT_MAX;')
		assert "(-32768)" in result
		assert "32767" in result

	def test_ushrt_max(self, pp: Preprocessor) -> None:
		result = pp.process('#include <limits.h>\nint x = USHRT_MAX;')
		assert "65535" in result

	def test_int_min_max(self, pp: Preprocessor) -> None:
		result = pp.process('#include <limits.h>\nint a = INT_MIN;\nint b = INT_MAX;')
		assert "(-2147483647 - 1)" in result
		assert "2147483647" in result

	def test_uint_max(self, pp: Preprocessor) -> None:
		result = pp.process('#include <limits.h>\nunsigned x = UINT_MAX;')
		assert "4294967295U" in result

	def test_long_limits(self, pp: Preprocessor) -> None:
		result = pp.process('#include <limits.h>\nlong a = LONG_MIN;\nlong b = LONG_MAX;')
		assert "(-9223372036854775807L - 1)" in result
		assert "9223372036854775807L" in result

	def test_ulong_max(self, pp: Preprocessor) -> None:
		result = pp.process('#include <limits.h>\nunsigned long x = ULONG_MAX;')
		assert "18446744073709551615UL" in result

	def test_llong_limits(self, pp: Preprocessor) -> None:
		result = pp.process('#include <limits.h>\nlong long a = LLONG_MIN;\nlong long b = LLONG_MAX;')
		assert "(-9223372036854775807LL - 1)" in result
		assert "9223372036854775807LL" in result

	def test_ullong_max(self, pp: Preprocessor) -> None:
		result = pp.process('#include <limits.h>\nunsigned long long x = ULLONG_MAX;')
		assert "18446744073709551615ULL" in result

	def test_include_guard(self, pp: Preprocessor) -> None:
		source = '#include <limits.h>\n#include <limits.h>'
		result = pp.process(source)
		assert result is not None


class TestMultipleHeaders:
	"""Tests for including multiple builtin headers together."""

	def test_stddef_and_stdint(self, pp: Preprocessor) -> None:
		source = '#include <stddef.h>\n#include <stdint.h>\nsize_t x;\nint32_t y;'
		result = pp.process(source)
		assert "typedef unsigned long size_t;" in result
		assert "typedef int int32_t;" in result

	def test_all_three_headers(self, pp: Preprocessor) -> None:
		source = '#include <stddef.h>\n#include <stdint.h>\n#include <limits.h>\nint x = INT_MAX;'
		result = pp.process(source)
		assert "2147483647" in result
		assert "typedef unsigned long size_t;" in result

	def test_stdint_with_stdbool(self, pp: Preprocessor) -> None:
		source = '#include <stdbool.h>\n#include <stdint.h>\nbool flag;\nint32_t val;'
		result = pp.process(source)
		assert "int32_t val;" in result

	def test_ifdef_on_header_guard(self, pp: Preprocessor) -> None:
		source = '#include <stdint.h>\n#ifdef _STDINT_H\nint guarded = 1;\n#endif'
		result = pp.process(source)
		assert "int guarded = 1;" in result

	def test_ifndef_after_include(self, pp: Preprocessor) -> None:
		source = '#include <limits.h>\n#ifndef _LIMITS_H\nint bad = 1;\n#endif'
		result = pp.process(source)
		assert "int bad = 1;" not in result
