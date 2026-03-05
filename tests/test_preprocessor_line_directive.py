"""Tests for #line directive support in the preprocessor."""

import pytest

from compiler.preprocessor import Preprocessor, PreprocessorError


class TestLineDirectiveBasic:
	"""Basic #line directive functionality."""

	def test_line_number_only(self) -> None:
		"""#line N sets line number for subsequent lines."""
		pp = Preprocessor()
		source = "#line 100\n__LINE__"
		result = pp.process(source)
		assert result.strip() == "100"

	def test_line_number_and_filename(self) -> None:
		"""#line N "file" sets both line number and filename."""
		pp = Preprocessor()
		source = '#line 50 "myfile.c"\n__LINE__ __FILE__'
		result = pp.process(source)
		assert "50" in result
		assert '"myfile.c"' in result

	def test_line_affects_subsequent_lines(self) -> None:
		"""#line affects all subsequent lines, incrementing normally."""
		pp = Preprocessor()
		source = "#line 10\n__LINE__\n__LINE__\n__LINE__"
		result = pp.process(source)
		lines = [ln for ln in result.strip().split("\n") if ln.strip()]
		assert lines[0].strip() == "10"
		assert lines[1].strip() == "11"
		assert lines[2].strip() == "12"

	def test_line_number_resets(self) -> None:
		"""A second #line directive overrides the first."""
		pp = Preprocessor()
		source = "#line 100\n__LINE__\n#line 200\n__LINE__"
		result = pp.process(source)
		lines = [ln for ln in result.strip().split("\n") if ln.strip()]
		assert lines[0].strip() == "100"
		assert lines[1].strip() == "200"


class TestLineDirectiveFilename:
	"""Tests for filename override via #line."""

	def test_filename_override(self) -> None:
		"""#line N "file" overrides __FILE__."""
		pp = Preprocessor()
		source = '#line 1 "override.c"\n__FILE__'
		result = pp.process(source)
		assert '"override.c"' in result

	def test_filename_persists(self) -> None:
		"""Filename override persists across lines."""
		pp = Preprocessor()
		source = '#line 1 "test.h"\n__FILE__\n__FILE__'
		result = pp.process(source)
		lines = [ln for ln in result.strip().split("\n") if ln.strip()]
		assert '"test.h"' in lines[0]
		assert '"test.h"' in lines[1]

	def test_filename_can_be_changed(self) -> None:
		"""A second #line with filename changes the filename."""
		pp = Preprocessor()
		source = '#line 1 "a.c"\n__FILE__\n#line 1 "b.c"\n__FILE__'
		result = pp.process(source)
		lines = [ln for ln in result.strip().split("\n") if ln.strip()]
		assert '"a.c"' in lines[0]
		assert '"b.c"' in lines[1]

	def test_line_without_filename_keeps_previous(self) -> None:
		"""#line N without filename keeps the current filename."""
		pp = Preprocessor()
		source = '#line 1 "custom.c"\n__FILE__\n#line 50\n__FILE__'
		result = pp.process(source)
		lines = [ln for ln in result.strip().split("\n") if ln.strip()]
		assert '"custom.c"' in lines[0]
		assert '"custom.c"' in lines[1]


class TestLineDirectiveWithDiagnostics:
	"""Tests for #line interaction with error/warning diagnostics."""

	def test_error_uses_line_directive(self) -> None:
		"""#error should use the #line-adjusted line number."""
		pp = Preprocessor()
		source = '#line 999 "virtual.c"\n#error boom'
		with pytest.raises(PreprocessorError) as exc_info:
			pp.process(source)
		assert "virtual.c" in str(exc_info.value)
		assert "999" in str(exc_info.value)

	def test_warning_uses_line_directive(self) -> None:
		"""#warning should use the #line-adjusted line/file."""
		pp = Preprocessor()
		source = '#line 42 "warn.c"\n#warning test warning'
		pp.process(source)
		assert len(pp.warnings) == 1
		assert "warn.c" in pp.warnings[0]
		assert "42" in pp.warnings[0]


class TestLineDirectiveErrors:
	"""Tests for invalid #line directives."""

	def test_missing_line_number(self) -> None:
		"""#line with no arguments is an error."""
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="Expected line number"):
			pp.process("#line")

	def test_non_numeric_line_number(self) -> None:
		"""#line with non-numeric argument is an error."""
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="Invalid line number"):
			pp.process("#line abc")

	def test_invalid_filename_format(self) -> None:
		"""#line N with unquoted filename is an error."""
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="Invalid filename"):
			pp.process("#line 10 myfile.c")


class TestLineDirectiveWithConditionals:
	"""Tests for #line interaction with conditional compilation."""

	def test_line_inside_ifdef(self) -> None:
		"""#line inside active #ifdef works."""
		pp = Preprocessor(predefined_macros={"ACTIVE": "1"})
		source = "#ifdef ACTIVE\n#line 500\n__LINE__\n#endif"
		result = pp.process(source)
		lines = [ln for ln in result.strip().split("\n") if ln.strip()]
		assert lines[0].strip() == "500"

	def test_line_inside_inactive_ifdef(self) -> None:
		"""#line inside inactive #ifdef is skipped."""
		pp = Preprocessor()
		source = "#ifdef UNDEFINED\n#line 500\n#endif\n__LINE__"
		result = pp.process(source)
		lines = [ln for ln in result.strip().split("\n") if ln.strip()]
		# Line 4 in source, no offset applied
		assert lines[0].strip() == "4"


class TestLineDirectiveEdgeCases:
	"""Edge cases for #line directive."""

	def test_line_one(self) -> None:
		"""#line 1 resets to line 1."""
		pp = Preprocessor()
		source = "x\ny\nz\n#line 1\n__LINE__"
		result = pp.process(source)
		lines = [ln for ln in result.strip().split("\n") if ln.strip()]
		assert lines[-1].strip() == "1"

	def test_large_line_number(self) -> None:
		"""#line with a large number works."""
		pp = Preprocessor()
		source = "#line 999999\n__LINE__"
		result = pp.process(source)
		assert result.strip() == "999999"

	def test_line_with_define(self) -> None:
		"""#line interacts correctly with #define using __LINE__."""
		pp = Preprocessor()
		source = "#line 42\n#define HERE __LINE__\nHERE"
		result = pp.process(source)
		lines = [ln for ln in result.strip().split("\n") if ln.strip()]
		# HERE expands on line 3 of source, offset makes it 43
		assert lines[0].strip() == "43"

	def test_multiple_line_directives(self) -> None:
		"""Multiple #line directives in sequence."""
		pp = Preprocessor()
		source = "#line 10\n__LINE__\n#line 20\n__LINE__\n#line 30\n__LINE__"
		result = pp.process(source)
		lines = [ln for ln in result.strip().split("\n") if ln.strip()]
		assert lines[0].strip() == "10"
		assert lines[1].strip() == "20"
		assert lines[2].strip() == "30"
