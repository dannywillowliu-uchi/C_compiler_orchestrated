"""Tests for illegal type modifier combination validation in semantic analysis."""

from compiler.ast_nodes import (
	FunctionDecl,
	CompoundStmt,
	ParamDecl,
	Program,
	ReturnStmt,
	IntLiteral,
	SourceLocation,
	TypeSpec,
	VarDecl,
)
from compiler.semantic import SemanticAnalyzer


def loc() -> SourceLocation:
	return SourceLocation(line=1, col=1)


def _analyze_program(program: Program) -> list[str]:
	"""Run semantic analysis and return error messages."""
	analyzer = SemanticAnalyzer()
	try:
		analyzer.analyze(program)
	except Exception:
		pass
	return [str(e) for e in analyzer.errors]


def _make_var_decl_program(base_type: str, signedness: str | None = None, width_modifier: str | None = None) -> Program:
	"""Build a minimal program with a variable declaration using the given type modifiers."""
	ts = TypeSpec(base_type=base_type, signedness=signedness, width_modifier=width_modifier, loc=loc())
	var = VarDecl(name="x", type_spec=ts, loc=loc())
	func = FunctionDecl(
		name="f",
		return_type=TypeSpec(base_type="void", loc=loc()),
		params=[],
		body=CompoundStmt(statements=[var], loc=loc()),
		loc=loc(),
	)
	return Program(declarations=[func], loc=loc())


def _make_param_program(base_type: str, signedness: str | None = None, width_modifier: str | None = None) -> Program:
	"""Build a program with a function parameter using the given type modifiers."""
	ts = TypeSpec(base_type=base_type, signedness=signedness, width_modifier=width_modifier, loc=loc())
	param = ParamDecl(name="x", type_spec=ts, loc=loc())
	func = FunctionDecl(
		name="f",
		return_type=TypeSpec(base_type="void", loc=loc()),
		params=[param],
		body=CompoundStmt(statements=[], loc=loc()),
		loc=loc(),
	)
	return Program(declarations=[func], loc=loc())


def _make_return_type_program(base_type: str, signedness: str | None = None, width_modifier: str | None = None) -> Program:
	"""Build a program with a function return type using the given type modifiers."""
	ts = TypeSpec(base_type=base_type, signedness=signedness, width_modifier=width_modifier, loc=loc())
	func = FunctionDecl(
		name="f",
		return_type=ts,
		params=[],
		body=CompoundStmt(
			statements=[ReturnStmt(expression=IntLiteral(value=0, loc=loc()), has_expression=True, loc=loc())],
			loc=loc(),
		),
		loc=loc(),
	)
	return Program(declarations=[func], loc=loc())


# --- Illegal signedness + float/double ---

class TestSignednessWithFloat:
	def test_signed_float_rejected(self) -> None:
		errors = _analyze_program(_make_var_decl_program("float", signedness="signed"))
		assert any("'signed' cannot be used with 'float'" in e for e in errors)

	def test_unsigned_float_rejected(self) -> None:
		errors = _analyze_program(_make_var_decl_program("float", signedness="unsigned"))
		assert any("'unsigned' cannot be used with 'float'" in e for e in errors)

	def test_signed_double_rejected(self) -> None:
		errors = _analyze_program(_make_var_decl_program("double", signedness="signed"))
		assert any("'signed' cannot be used with 'double'" in e for e in errors)

	def test_unsigned_double_rejected(self) -> None:
		errors = _analyze_program(_make_var_decl_program("double", signedness="unsigned"))
		assert any("'unsigned' cannot be used with 'double'" in e for e in errors)


# --- Illegal short + long combinations ---

class TestShortLongConflict:
	def test_short_long_rejected(self) -> None:
		# 'short' width modifier with base type 'long'
		errors = _analyze_program(_make_var_decl_program("long", width_modifier="short"))
		assert any("'short' cannot be used with 'long'" in e for e in errors)


# --- Short with float/double ---

class TestShortWithFloatingPoint:
	def test_short_float_rejected(self) -> None:
		errors = _analyze_program(_make_var_decl_program("float", width_modifier="short"))
		assert any("'short' cannot be used with 'float'" in e for e in errors)

	def test_short_double_rejected(self) -> None:
		errors = _analyze_program(_make_var_decl_program("double", width_modifier="short"))
		assert any("'short' cannot be used with 'double'" in e for e in errors)


# --- Long with char ---

class TestLongWithChar:
	def test_long_char_rejected(self) -> None:
		errors = _analyze_program(_make_var_decl_program("char", width_modifier="long"))
		assert any("'long' cannot be used with 'char'" in e for e in errors)

	def test_long_long_char_rejected(self) -> None:
		errors = _analyze_program(_make_var_decl_program("char", width_modifier="long long"))
		assert any("'long long' cannot be used with 'char'" in e for e in errors)


# --- Short with char ---

class TestShortWithChar:
	# 'short char' is not standard C, but the parser may allow it.
	# The width modifier is 'short' and base is 'char' -- this doesn't match
	# our explicit checks but the spec only calls for short+float, short+double,
	# short+long. If the parser never generates short+char TypeSpecs, that's fine.
	pass


# --- Valid combinations should not error ---

class TestValidCombinations:
	def test_signed_int_accepted(self) -> None:
		errors = _analyze_program(_make_var_decl_program("int", signedness="signed"))
		assert not errors

	def test_unsigned_int_accepted(self) -> None:
		errors = _analyze_program(_make_var_decl_program("int", signedness="unsigned"))
		assert not errors

	def test_signed_char_accepted(self) -> None:
		errors = _analyze_program(_make_var_decl_program("char", signedness="signed"))
		assert not errors

	def test_unsigned_char_accepted(self) -> None:
		errors = _analyze_program(_make_var_decl_program("char", signedness="unsigned"))
		assert not errors

	def test_short_int_accepted(self) -> None:
		errors = _analyze_program(_make_var_decl_program("int", width_modifier="short"))
		assert not errors

	def test_long_int_accepted(self) -> None:
		errors = _analyze_program(_make_var_decl_program("int", width_modifier="long"))
		assert not errors

	def test_long_long_int_accepted(self) -> None:
		errors = _analyze_program(_make_var_decl_program("int", width_modifier="long long"))
		assert not errors

	def test_unsigned_short_int_accepted(self) -> None:
		errors = _analyze_program(_make_var_decl_program("int", signedness="unsigned", width_modifier="short"))
		assert not errors

	def test_unsigned_long_int_accepted(self) -> None:
		errors = _analyze_program(_make_var_decl_program("int", signedness="unsigned", width_modifier="long"))
		assert not errors

	def test_plain_float_accepted(self) -> None:
		errors = _analyze_program(_make_var_decl_program("float"))
		assert not errors

	def test_plain_double_accepted(self) -> None:
		errors = _analyze_program(_make_var_decl_program("double"))
		assert not errors

	def test_long_double_accepted(self) -> None:
		errors = _analyze_program(_make_var_decl_program("double", width_modifier="long"))
		# long double is valid in C, we don't reject it
		assert not any("'long' cannot be used with 'double'" in e for e in errors)


# --- Validation also applies to function params and return types ---

class TestValidationInParams:
	def test_unsigned_float_param_rejected(self) -> None:
		errors = _analyze_program(_make_param_program("float", signedness="unsigned"))
		assert any("'unsigned' cannot be used with 'float'" in e for e in errors)

	def test_short_double_param_rejected(self) -> None:
		errors = _analyze_program(_make_param_program("double", width_modifier="short"))
		assert any("'short' cannot be used with 'double'" in e for e in errors)


class TestValidationInReturnTypes:
	def test_signed_float_return_rejected(self) -> None:
		errors = _analyze_program(_make_return_type_program("float", signedness="signed"))
		assert any("'signed' cannot be used with 'float'" in e for e in errors)

	def test_long_char_return_rejected(self) -> None:
		errors = _analyze_program(_make_return_type_program("char", width_modifier="long"))
		assert any("'long' cannot be used with 'char'" in e for e in errors)
