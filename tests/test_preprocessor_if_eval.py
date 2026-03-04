"""Tests for preprocessor #if expression evaluation with C logical operators."""

from src.compiler.preprocessor import Preprocessor


def test_if_defined_and():
	"""Test #if with && (logical AND) on defined macros."""
	source = (
		"#define X 1\n"
		"#define Y 1\n"
		"#if defined(X) && defined(Y)\n"
		"both_defined\n"
		"#endif\n"
	)
	pp = Preprocessor()
	result = pp.process(source)
	assert "both_defined" in result


def test_if_defined_and_one_missing():
	"""Test #if && where one macro is not defined."""
	source = (
		"#define X 1\n"
		"#if defined(X) && defined(Y)\n"
		"both_defined\n"
		"#else\n"
		"not_both\n"
		"#endif\n"
	)
	pp = Preprocessor()
	result = pp.process(source)
	assert "not_both" in result
	assert "both_defined" not in result


def test_if_not_defined_or():
	"""Test #if with ! and || operators."""
	source = (
		"#define Y 1\n"
		"#if !defined(X) || defined(Y)\n"
		"included\n"
		"#endif\n"
	)
	pp = Preprocessor()
	result = pp.process(source)
	assert "included" in result


def test_if_not_defined_or_both_false():
	"""Test #if !defined(A) || defined(B) where A is defined and B is not."""
	source = (
		"#define A 1\n"
		"#if !defined(A) || defined(B)\n"
		"included\n"
		"#else\n"
		"excluded\n"
		"#endif\n"
	)
	pp = Preprocessor()
	result = pp.process(source)
	assert "excluded" in result
	assert "included" not in result


def test_if_arithmetic_expression():
	"""Test #if with arithmetic comparison (1+2 > 3)."""
	source = (
		"#if (1+2 > 3)\n"
		"greater\n"
		"#else\n"
		"not_greater\n"
		"#endif\n"
	)
	pp = Preprocessor()
	result = pp.process(source)
	assert "not_greater" in result
	lines = [line.strip() for line in result.strip().splitlines()]
	assert "greater" not in lines


def test_if_arithmetic_expression_true():
	"""Test #if with arithmetic comparison that is true."""
	source = (
		"#if (1+2 >= 3)\n"
		"ge\n"
		"#endif\n"
	)
	pp = Preprocessor()
	result = pp.process(source)
	assert "ge" in result


def test_if_nested_logical_operators():
	"""Test nested combinations of &&, ||, and !."""
	source = (
		"#define A 1\n"
		"#define B 1\n"
		"#if (defined(A) && defined(B)) || defined(C)\n"
		"pass1\n"
		"#endif\n"
		"#if defined(C) || (defined(A) && !defined(D))\n"
		"pass2\n"
		"#endif\n"
		"#if !defined(A) && !defined(B)\n"
		"fail1\n"
		"#endif\n"
	)
	pp = Preprocessor()
	result = pp.process(source)
	assert "pass1" in result
	assert "pass2" in result
	assert "fail1" not in result


def test_if_not_equal_preserved():
	"""Test that != is NOT mangled by the ! replacement."""
	source = (
		"#define X 5\n"
		"#if X != 3\n"
		"not_equal\n"
		"#endif\n"
	)
	pp = Preprocessor()
	result = pp.process(source)
	assert "not_equal" in result


def test_if_double_negation():
	"""Test !!expr evaluates correctly."""
	source = (
		"#define X 1\n"
		"#if !!X\n"
		"double_neg\n"
		"#endif\n"
	)
	pp = Preprocessor()
	result = pp.process(source)
	assert "double_neg" in result


def test_if_complex_mixed():
	"""Test complex expression mixing &&, ||, !, and arithmetic."""
	source = (
		"#define VER 3\n"
		"#define FEAT 1\n"
		"#if (VER > 2 && FEAT) || !defined(MISSING)\n"
		"complex_pass\n"
		"#endif\n"
	)
	pp = Preprocessor()
	result = pp.process(source)
	assert "complex_pass" in result
