"""Extended tests for #error, #warning, #undef, and #line directives."""

import textwrap

import pytest

from compiler.preprocessor import Preprocessor, PreprocessorError


def pp_process(source: str, filename: str = "<stdin>", **kwargs) -> tuple[str, Preprocessor]:
	"""Process source and return (output, preprocessor) for inspection."""
	pp = Preprocessor(**kwargs)
	result = pp.process(textwrap.dedent(source), filename)
	return result, pp


# ── #error ────────────────────────────────────────────────────────────────────


class TestErrorDirective:
	def test_error_with_empty_message(self):
		with pytest.raises(PreprocessorError, match="#error"):
			pp_process("#error")

	def test_error_with_empty_message_trailing_space(self):
		with pytest.raises(PreprocessorError, match="#error"):
			pp_process("#error   ")

	def test_error_with_special_characters(self):
		with pytest.raises(PreprocessorError, match="#error @!\\$%"):
			pp_process("#error @!$%")

	def test_error_reports_correct_file_and_line(self):
		source = "int a;\nint b;\n#error oops\n"
		with pytest.raises(PreprocessorError) as exc_info:
			pp_process(source, filename="test.c")
		assert "test.c:3" in str(exc_info.value)

	def test_error_skipped_in_false_elif_branch(self):
		source = """\
			#define X 1
			#if 0
			#error branch0
			#elif 0
			#error branch1
			#else
			int ok;
			#endif
		"""
		result, _ = pp_process(source)
		assert "int ok;" in result

	def test_error_fires_in_true_elif_branch(self):
		source = """\
			#if 0
			int skip;
			#elif 1
			#error found it
			#endif
		"""
		with pytest.raises(PreprocessorError, match="#error found it"):
			pp_process(source)

	def test_error_respects_line_directive(self):
		source = """\
			#line 999 "generated.h"
			#error stop
		"""
		with pytest.raises(PreprocessorError) as exc_info:
			pp_process(source)
		assert "generated.h:999" in str(exc_info.value)


# ── #warning ──────────────────────────────────────────────────────────────────


class TestWarningDirective:
	def test_warning_with_empty_message(self):
		_, pp = pp_process("#warning")
		assert len(pp.warnings) == 1
		assert "#warning" in pp.warnings[0]

	def test_warning_with_empty_message_trailing_space(self):
		_, pp = pp_process("#warning   ")
		assert len(pp.warnings) == 1

	def test_warning_does_not_discard_output(self):
		source = """\
			int a = 1;
			#warning caution
			int b = 2;
			#warning again
			int c = 3;
		"""
		result, pp = pp_process(source)
		assert "int a = 1;" in result
		assert "int b = 2;" in result
		assert "int c = 3;" in result
		assert len(pp.warnings) == 2

	def test_warning_reports_correct_location(self):
		source = "line1\nline2\n#warning here\n"
		_, pp = pp_process(source, filename="myfile.c")
		assert "myfile.c:3" in pp.warnings[0]

	def test_warning_skipped_in_nested_false_conditional(self):
		source = """\
			#ifdef OUTER
			#ifdef INNER
			#warning deep
			#endif
			#endif
		"""
		_, pp = pp_process(source)
		assert len(pp.warnings) == 0

	def test_warning_respects_line_directive(self):
		source = """\
			#line 200 "foo.h"
			#warning check
		"""
		_, pp = pp_process(source)
		assert "foo.h:200" in pp.warnings[0]

	def test_warning_with_special_characters(self):
		_, pp = pp_process('#warning "quotes" and <angles>')
		assert len(pp.warnings) == 1
		assert '"quotes"' in pp.warnings[0]


# ── #undef ────────────────────────────────────────────────────────────────────


class TestUndefDirective:
	def test_undef_removes_object_macro(self):
		source = """\
			#define FOO 42
			#undef FOO
			int x = FOO;
		"""
		result, _ = pp_process(source)
		# FOO should NOT be expanded since it was undefined
		assert "int x = FOO;" in result

	def test_undef_removes_function_macro(self):
		source = """\
			#define ADD(a, b) ((a) + (b))
			#undef ADD
			int x = ADD;
		"""
		result, _ = pp_process(source)
		assert "int x = ADD;" in result

	def test_undef_of_undefined_macro_is_noop(self):
		source = """\
			#undef NONEXISTENT
			int x = 1;
		"""
		result, _ = pp_process(source)
		assert "int x = 1;" in result

	def test_undef_then_redefine(self):
		source = """\
			#define VAL 10
			int a = VAL;
			#undef VAL
			#define VAL 20
			int b = VAL;
		"""
		result, _ = pp_process(source)
		assert "int a = 10;" in result
		assert "int b = 20;" in result

	def test_undef_only_affects_target(self):
		source = """\
			#define A 1
			#define B 2
			#undef A
			int x = B;
		"""
		result, _ = pp_process(source)
		assert "int x = 2;" in result

	def test_undef_affects_ifdef(self):
		source = """\
			#define FLAG
			#undef FLAG
			#ifdef FLAG
			int bad;
			#else
			int good;
			#endif
		"""
		result, _ = pp_process(source)
		assert "int good;" in result
		assert "int bad;" not in result

	def test_undef_without_name_raises(self):
		with pytest.raises(PreprocessorError, match="Expected macro name"):
			pp_process("#undef")

	def test_undef_skipped_in_false_branch(self):
		source = """\
			#define KEEP 99
			#if 0
			#undef KEEP
			#endif
			int x = KEEP;
		"""
		result, _ = pp_process(source)
		assert "int x = 99;" in result

	def test_undef_predefined_macro(self):
		source = """\
			#undef MY_CONST
			int x = 1;
		"""
		result, _ = pp_process(source, predefined_macros={"MY_CONST": "42"})
		assert "int x = 1;" in result
		# MY_CONST should no longer expand if used after undef
		source2 = """\
			int a = MY_CONST;
			#undef MY_CONST
			int b = MY_CONST;
		"""
		result2, _ = pp_process(source2, predefined_macros={"MY_CONST": "42"})
		assert "int a = 42;" in result2
		assert "int b = MY_CONST;" in result2


# ── #line ─────────────────────────────────────────────────────────────────────


class TestLineDirective:
	def test_line_without_filename(self):
		source = """\
			#line 50
			int x = __LINE__;
		"""
		result, _ = pp_process(source)
		assert "int x = 50;" in result

	def test_line_with_filename(self):
		source = """\
			#line 10 "other.c"
			const char *f = __FILE__;
		"""
		result, _ = pp_process(source)
		assert '"other.c"' in result

	def test_line_consecutive_increments(self):
		source = """\
			#line 100
			int a = __LINE__;
			int b = __LINE__;
			int c = __LINE__;
		"""
		result, _ = pp_process(source)
		assert "int a = 100;" in result
		assert "int b = 101;" in result
		assert "int c = 102;" in result

	def test_line_can_be_overridden_multiple_times(self):
		source = """\
			#line 10
			int a = __LINE__;
			#line 500
			int b = __LINE__;
		"""
		result, _ = pp_process(source)
		assert "int a = 10;" in result
		assert "int b = 500;" in result

	def test_line_filename_persists_across_lines(self):
		source = """\
			#line 1 "custom.h"
			int a = __LINE__;
			const char *f1 = __FILE__;
			const char *f2 = __FILE__;
		"""
		result, _ = pp_process(source)
		assert result.count('"custom.h"') >= 2

	def test_line_invalid_number_raises(self):
		with pytest.raises(PreprocessorError, match="Invalid line number"):
			pp_process("#line abc")

	def test_line_missing_number_raises(self):
		with pytest.raises(PreprocessorError, match="Expected line number"):
			pp_process("#line")

	def test_line_invalid_filename_raises(self):
		with pytest.raises(PreprocessorError, match="Invalid filename"):
			pp_process('#line 10 foo.c')

	def test_line_filename_override_then_new_line_number_only(self):
		"""A #line with only a number should keep the previous filename override."""
		source = """\
			#line 1 "first.h"
			int a = __LINE__;
			#line 99
			const char *f = __FILE__;
		"""
		result, _ = pp_process(source)
		assert "int a = 1;" in result
		# The file override from first #line should persist
		assert '"first.h"' in result

	def test_line_affects_warning_location(self):
		source = """\
			#line 77 "warn.h"
			#warning check
		"""
		_, pp = pp_process(source)
		assert "warn.h:77" in pp.warnings[0]

	def test_line_affects_error_location(self):
		source = """\
			#line 300 "err.c"
			#error fail
		"""
		with pytest.raises(PreprocessorError) as exc_info:
			pp_process(source)
		assert "err.c:300" in str(exc_info.value)
