"""Tests for semantic checks: void return, register address-of, and missing return."""

import pytest

from compiler.ast_nodes import (
	CompoundStmt,
	ExprStmt,
	FunctionDecl,
	Identifier,
	IntLiteral,
	Program,
	ReturnStmt,
	SourceLocation,
	TypeSpec,
	UnaryOp,
	VarDecl,
)
from compiler.semantic import SemanticAnalyzer, SemanticError


def loc(line: int = 1, col: int = 1) -> SourceLocation:
	return SourceLocation(line=line, col=col)


def int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def void_type() -> TypeSpec:
	return TypeSpec(base_type="void")


# ---------------------------------------------------------------------------
# 1. Void function with return value / bare return
# ---------------------------------------------------------------------------


class TestVoidReturn:
	def test_void_function_return_value_errors(self) -> None:
		"""void f() { return 42; } should error."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					ReturnStmt(
						expression=IntLiteral(value=42, loc=loc()),
						has_expression=True,
						loc=loc(2, 1),
					),
				], loc=loc()),
				loc=loc(),
			),
		], loc=loc())
		analyzer = SemanticAnalyzer()
		with pytest.raises(SemanticError, match="void function should not return a value"):
			analyzer.analyze(prog)

	def test_void_function_bare_return_ok(self) -> None:
		"""void f() { return; } should be accepted."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					ReturnStmt(
						expression=IntLiteral(value=0, loc=loc()),
						has_expression=False,
						loc=loc(2, 1),
					),
				], loc=loc()),
				loc=loc(),
			),
		], loc=loc())
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)  # should not raise

	def test_void_function_no_return_ok(self) -> None:
		"""void f() { } should be accepted (no return at all)."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[], loc=loc()),
				loc=loc(),
			),
		], loc=loc())
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)  # should not raise


# ---------------------------------------------------------------------------
# 2. Address-of register variable
# ---------------------------------------------------------------------------


class TestRegisterAddressOf:
	def test_address_of_register_errors(self) -> None:
		"""register int x; &x; should error."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=int_type(),
						name="x",
						storage_class="register",
						loc=loc(2, 1),
					),
					ExprStmt(
						expression=UnaryOp(
							op="&",
							operand=Identifier(name="x", loc=loc(3, 2)),
							loc=loc(3, 1),
						),
						loc=loc(3, 1),
					),
					ReturnStmt(
						expression=IntLiteral(value=0, loc=loc()),
						has_expression=True,
						loc=loc(4, 1),
					),
				], loc=loc()),
				loc=loc(),
			),
		], loc=loc())
		analyzer = SemanticAnalyzer()
		with pytest.raises(SemanticError, match="address of register variable 'x' requested"):
			analyzer.analyze(prog)

	def test_register_variable_without_address_ok(self) -> None:
		"""register int x; x + 1; should be accepted."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=int_type(),
						name="x",
						initializer=IntLiteral(value=5, loc=loc()),
						storage_class="register",
						loc=loc(2, 1),
					),
					ReturnStmt(
						expression=Identifier(name="x", loc=loc(3, 1)),
						has_expression=True,
						loc=loc(3, 1),
					),
				], loc=loc()),
				loc=loc(),
			),
		], loc=loc())
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)  # should not raise


# ---------------------------------------------------------------------------
# 3. Missing return in non-void function
# ---------------------------------------------------------------------------


class TestMissingReturn:
	def test_non_void_missing_return_warns(self) -> None:
		"""int f() { } should produce a warning."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[], loc=loc()),
				loc=loc(),
			),
		], loc=loc())
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)
		assert len(analyzer.warnings) == 1
		assert "control reaches end of non-void function 'f'" in analyzer.warnings[0]

	def test_non_void_with_return_no_warning(self) -> None:
		"""int f() { return 0; } should produce no warning."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					ReturnStmt(
						expression=IntLiteral(value=0, loc=loc()),
						has_expression=True,
						loc=loc(2, 1),
					),
				], loc=loc()),
				loc=loc(),
			),
		], loc=loc())
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)
		assert len(analyzer.warnings) == 0

	def test_non_void_last_stmt_not_return_warns(self) -> None:
		"""int f() { int x = 1; } should warn (last stmt is VarDecl, not return)."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=int_type(),
						name="x",
						initializer=IntLiteral(value=1, loc=loc()),
						loc=loc(2, 1),
					),
				], loc=loc()),
				loc=loc(),
			),
		], loc=loc())
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)
		assert len(analyzer.warnings) == 1
		assert "control reaches end of non-void function 'f'" in analyzer.warnings[0]

	def test_void_function_no_return_no_warning(self) -> None:
		"""void f() { } should produce no warning."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[], loc=loc()),
				loc=loc(),
			),
		], loc=loc())
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)
		assert len(analyzer.warnings) == 0
