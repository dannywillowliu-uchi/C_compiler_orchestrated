"""Tests for #error and #warning preprocessor directives."""

import pytest

from compiler.preprocessor import Preprocessor, PreprocessorError


class TestErrorDirective:
	"""Tests for #error directive."""

	def test_basic_error(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="#error this is an error"):
			pp.process("#error this is an error")

	def test_error_inside_inactive_ifdef(self):
		"""#error inside inactive #ifdef should not trigger."""
		pp = Preprocessor()
		source = "#ifdef UNDEFINED\n#error should not fire\n#endif\nint x;"
		result = pp.process(source)
		assert "int x;" in result

	def test_error_inside_inactive_ifndef(self):
		pp = Preprocessor(predefined_macros={"DEFINED": "1"})
		source = "#ifndef DEFINED\n#error should not fire\n#endif\nint x;"
		result = pp.process(source)
		assert "int x;" in result

	def test_error_inside_active_ifdef(self):
		pp = Preprocessor(predefined_macros={"ACTIVE": "1"})
		with pytest.raises(PreprocessorError, match="#error triggered"):
			pp.process("#ifdef ACTIVE\n#error triggered\n#endif")

	def test_error_inside_inactive_if(self):
		pp = Preprocessor()
		source = "#if 0\n#error should not fire\n#endif\nint y;"
		result = pp.process(source)
		assert "int y;" in result

	def test_error_inside_else_inactive(self):
		"""#error in else branch when if branch is true should not trigger."""
		pp = Preprocessor(predefined_macros={"X": "1"})
		source = "#ifdef X\nint a;\n#else\n#error nope\n#endif"
		result = pp.process(source)
		assert "int a;" in result

	def test_error_preserves_message(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError) as exc_info:
			pp.process("#error custom message here")
		assert "custom message here" in str(exc_info.value)

	def test_error_includes_filename_and_line(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError) as exc_info:
			pp.process("\n\n#error oops", filename="test.c")
		assert "test.c" in str(exc_info.value)
		assert ":3:" in str(exc_info.value)

	def test_error_with_empty_message(self):
		pp = Preprocessor()
		with pytest.raises(PreprocessorError, match="#error"):
			pp.process("#error")


class TestWarningDirective:
	"""Tests for #warning directive."""

	def test_basic_warning(self):
		pp = Preprocessor()
		pp.process("#warning this is a warning")
		assert len(pp.warnings) == 1
		assert "this is a warning" in pp.warnings[0]

	def test_warning_inside_inactive_ifdef(self):
		"""#warning inside inactive block should not trigger."""
		pp = Preprocessor()
		pp.process("#ifdef UNDEFINED\n#warning should not appear\n#endif")
		assert len(pp.warnings) == 0

	def test_warning_inside_inactive_if(self):
		pp = Preprocessor()
		pp.process("#if 0\n#warning nope\n#endif")
		assert len(pp.warnings) == 0

	def test_warning_inside_active_ifdef(self):
		pp = Preprocessor(predefined_macros={"ACTIVE": "1"})
		pp.process("#ifdef ACTIVE\n#warning hello\n#endif")
		assert len(pp.warnings) == 1
		assert "hello" in pp.warnings[0]

	def test_warning_does_not_stop_processing(self):
		"""#warning should not halt preprocessing."""
		pp = Preprocessor()
		result = pp.process("#warning caution\nint x = 42;")
		assert "int x = 42;" in result
		assert len(pp.warnings) == 1

	def test_multiple_warnings(self):
		pp = Preprocessor()
		pp.process("#warning first\n#warning second\n#warning third")
		assert len(pp.warnings) == 3
		assert "first" in pp.warnings[0]
		assert "second" in pp.warnings[1]
		assert "third" in pp.warnings[2]

	def test_warning_includes_filename_and_line(self):
		pp = Preprocessor()
		pp.process("\n#warning check", filename="foo.h")
		assert "foo.h" in pp.warnings[0]
		assert ":2:" in pp.warnings[0]

	def test_warning_with_empty_message(self):
		pp = Preprocessor()
		pp.process("#warning")
		assert len(pp.warnings) == 1
		assert "#warning" in pp.warnings[0]


class TestErrorWarningWithMacroExpansion:
	"""Tests for macro expansion in #error/#warning messages."""

	def test_error_with_macro_in_message(self):
		"""#error message is taken as literal text (no macro expansion per C standard)."""
		pp = Preprocessor(predefined_macros={"VERSION": "2"})
		with pytest.raises(PreprocessorError, match="#error VERSION is too old"):
			pp.process("#error VERSION is too old")

	def test_warning_with_macro_in_message(self):
		"""#warning message is taken as literal text."""
		pp = Preprocessor(predefined_macros={"FOO": "bar"})
		pp.process("#warning FOO is deprecated")
		assert "FOO is deprecated" in pp.warnings[0]


class TestConditionalInteraction:
	"""Tests for #error/#warning with nested conditionals."""

	def test_error_in_nested_inactive_block(self):
		pp = Preprocessor()
		source = (
			"#ifdef OUTER\n"
			"#ifdef INNER\n"
			"#error deep error\n"
			"#endif\n"
			"#endif\n"
			"int z;"
		)
		result = pp.process(source)
		assert "int z;" in result

	def test_warning_in_elif_branch(self):
		pp = Preprocessor()
		source = (
			"#if 0\nint a;\n"
			"#elif 1\n#warning in elif\n"
			"#endif"
		)
		pp.process(source)
		assert len(pp.warnings) == 1
		assert "in elif" in pp.warnings[0]

	def test_error_in_inactive_elif(self):
		pp = Preprocessor()
		source = (
			"#if 1\nint a;\n"
			"#elif 1\n#error should not fire\n"
			"#endif"
		)
		result = pp.process(source)
		assert "int a;" in result
