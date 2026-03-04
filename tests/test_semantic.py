"""Tests for semantic analysis: symbol table and type checking."""

import pytest

from compiler.ast_nodes import (
	ArraySubscript,
	Assignment,
	BinaryOp,
	BreakStmt,
	CaseClause,
	CharLiteral,
	CompoundAssignment,
	CompoundStmt,
	ContinueStmt,
	DoWhileStmt,
	ExprStmt,
	ForStmt,
	FunctionCall,
	FunctionDecl,
	Identifier,
	IfStmt,
	IntLiteral,
	MemberAccess,
	ParamDecl,
	PostfixExpr,
	Program,
	ReturnStmt,
	SizeofExpr,
	SourceLocation,
	StringLiteral,
	StructDecl,
	StructMember,
	SwitchStmt,
	TernaryExpr,
	TypeSpec,
	UnaryOp,
	VarDecl,
	WhileStmt,
)
from compiler.semantic import SemanticAnalyzer, SemanticError, Symbol, SymbolTable


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def loc(line: int = 1, col: int = 1) -> SourceLocation:
	return SourceLocation(line=line, col=col)


def int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def char_type() -> TypeSpec:
	return TypeSpec(base_type="char")


def void_type() -> TypeSpec:
	return TypeSpec(base_type="void")


def ptr_type(base: str = "int", depth: int = 1) -> TypeSpec:
	return TypeSpec(base_type=base, pointer_count=depth)


# ---------------------------------------------------------------------------
# SymbolTable unit tests
# ---------------------------------------------------------------------------


class TestSymbolTable:
	def test_define_and_lookup(self) -> None:
		st = SymbolTable()
		sym = Symbol(name="x", type_spec=int_type(), scope_depth=0)
		st.define(sym)
		assert st.lookup("x") is sym

	def test_lookup_missing_returns_none(self) -> None:
		st = SymbolTable()
		assert st.lookup("missing") is None

	def test_nested_scope_shadows(self) -> None:
		st = SymbolTable()
		outer = Symbol(name="x", type_spec=int_type(), scope_depth=0)
		st.define(outer)
		st.push_scope()
		inner = Symbol(name="x", type_spec=char_type(), scope_depth=1)
		st.define(inner)
		assert st.lookup("x") is inner
		st.pop_scope()
		assert st.lookup("x") is outer

	def test_redefinition_same_scope_raises(self) -> None:
		st = SymbolTable()
		st.define(Symbol(name="x", type_spec=int_type(), scope_depth=0))
		with pytest.raises(SemanticError, match="redefinition"):
			st.define(Symbol(name="x", type_spec=int_type(), scope_depth=0))

	def test_same_name_different_scopes_ok(self) -> None:
		st = SymbolTable()
		st.define(Symbol(name="x", type_spec=int_type(), scope_depth=0))
		st.push_scope()
		st.define(Symbol(name="x", type_spec=char_type(), scope_depth=1))  # no error

	def test_pop_global_scope_raises(self) -> None:
		st = SymbolTable()
		with pytest.raises(RuntimeError):
			st.pop_scope()

	def test_depth_tracking(self) -> None:
		st = SymbolTable()
		assert st.depth == 0
		st.push_scope()
		assert st.depth == 1
		st.push_scope()
		assert st.depth == 2
		st.pop_scope()
		assert st.depth == 1

	def test_lookup_current_scope(self) -> None:
		st = SymbolTable()
		st.define(Symbol(name="x", type_spec=int_type(), scope_depth=0))
		st.push_scope()
		assert st.lookup_current_scope("x") is None
		assert st.lookup("x") is not None


# ---------------------------------------------------------------------------
# Symbol construction
# ---------------------------------------------------------------------------


class TestSymbol:
	def test_basic(self) -> None:
		sym = Symbol(name="foo", type_spec=int_type(), scope_depth=0)
		assert sym.name == "foo"
		assert sym.is_function is False
		assert sym.param_types == []

	def test_function_symbol(self) -> None:
		sym = Symbol(
			name="add",
			type_spec=int_type(),
			scope_depth=0,
			is_function=True,
			param_types=[int_type(), int_type()],
		)
		assert sym.is_function is True
		assert len(sym.param_types) == 2


# ---------------------------------------------------------------------------
# SemanticAnalyzer — valid programs (should not raise)
# ---------------------------------------------------------------------------


class TestSemanticValid:
	def test_empty_program(self) -> None:
		prog = Program(declarations=[])
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)

	def test_simple_function(self) -> None:
		"""int main() { return 0; }"""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="main",
				params=[],
				body=CompoundStmt(statements=[
					ReturnStmt(expression=IntLiteral(value=0)),
				]),
			),
		])
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)

	def test_variable_declaration_and_use(self) -> None:
		"""int main() { int x = 5; return x; }"""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="main",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=5)),
					ReturnStmt(expression=Identifier(name="x")),
				]),
			),
		])
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)

	def test_function_call_correct_args(self) -> None:
		"""int add(int a, int b) { return a + b; } int main() { return add(1, 2); }"""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="add",
				params=[
					ParamDecl(type_spec=int_type(), name="a"),
					ParamDecl(type_spec=int_type(), name="b"),
				],
				body=CompoundStmt(statements=[
					ReturnStmt(expression=BinaryOp(
						left=Identifier(name="a"),
						op="+",
						right=Identifier(name="b"),
					)),
				]),
			),
			FunctionDecl(
				return_type=int_type(),
				name="main",
				params=[],
				body=CompoundStmt(statements=[
					ReturnStmt(expression=FunctionCall(
						name="add",
						arguments=[IntLiteral(value=1), IntLiteral(value=2)],
					)),
				]),
			),
		])
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)

	def test_nested_scopes(self) -> None:
		"""int main() { int x = 1; { int x = 2; } return x; }"""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="main",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=1)),
					CompoundStmt(statements=[
						VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=2)),
					]),
					ReturnStmt(expression=Identifier(name="x")),
				]),
			),
		])
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)

	def test_if_else(self) -> None:
		"""int main() { int x = 1; if (x) { return 1; } else { return 0; } }"""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="main",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=1)),
					IfStmt(
						condition=Identifier(name="x"),
						then_branch=CompoundStmt(statements=[
							ReturnStmt(expression=IntLiteral(value=1)),
						]),
						else_branch=CompoundStmt(statements=[
							ReturnStmt(expression=IntLiteral(value=0)),
						]),
					),
				]),
			),
		])
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)

	def test_while_loop(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="loop",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="i", initializer=IntLiteral(value=0)),
					WhileStmt(
						condition=BinaryOp(
							left=Identifier(name="i"), op="<", right=IntLiteral(value=10)
						),
						body=CompoundStmt(statements=[
							ExprStmt(expression=Assignment(
								target=Identifier(name="i"),
								value=BinaryOp(
									left=Identifier(name="i"), op="+", right=IntLiteral(value=1)
								),
							)),
						]),
					),
				]),
			),
		])
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)

	def test_for_loop_with_var_decl(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					ForStmt(
						init=VarDecl(type_spec=int_type(), name="i", initializer=IntLiteral(value=0)),
						condition=BinaryOp(
							left=Identifier(name="i"), op="<", right=IntLiteral(value=10)
						),
						update=Assignment(
							target=Identifier(name="i"),
							value=BinaryOp(
								left=Identifier(name="i"), op="+", right=IntLiteral(value=1)
							),
						),
						body=CompoundStmt(statements=[]),
					),
				]),
			),
		])
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)

	def test_char_to_int_compatible(self) -> None:
		"""int main() { int x = 'a'; return x; }"""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="main",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=CharLiteral(value="a")),
					ReturnStmt(expression=Identifier(name="x")),
				]),
			),
		])
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)

	def test_assignment(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x"),
					ExprStmt(expression=Assignment(
						target=Identifier(name="x"),
						value=IntLiteral(value=42),
					)),
				]),
			),
		])
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)


# ---------------------------------------------------------------------------
# SemanticAnalyzer — error detection
# ---------------------------------------------------------------------------


class TestUndeclaredVariable:
	def test_undeclared_in_return(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="main",
				params=[],
				body=CompoundStmt(statements=[
					ReturnStmt(expression=Identifier(name="y", loc=loc(3, 10))),
				]),
			),
		])
		with pytest.raises(SemanticError, match="undeclared identifier 'y'"):
			SemanticAnalyzer().analyze(prog)

	def test_undeclared_in_assignment(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					ExprStmt(expression=Assignment(
						target=Identifier(name="z", loc=loc(2, 3)),
						value=IntLiteral(value=1),
					)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="undeclared identifier 'z'"):
			SemanticAnalyzer().analyze(prog)

	def test_var_not_visible_after_scope_exit(self) -> None:
		"""{ int x = 1; } return x;  -- x should be out of scope"""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="main",
				params=[],
				body=CompoundStmt(statements=[
					CompoundStmt(statements=[
						VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=1)),
					]),
					ReturnStmt(expression=Identifier(name="x", loc=loc(4, 10))),
				]),
			),
		])
		with pytest.raises(SemanticError, match="undeclared identifier 'x'"):
			SemanticAnalyzer().analyze(prog)


class TestRedefinition:
	def test_redefinition_same_scope(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x"),
					VarDecl(type_spec=int_type(), name="x", loc=loc(3, 5)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="redefinition of 'x'"):
			SemanticAnalyzer().analyze(prog)

	def test_duplicate_function(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(return_type=void_type(), name="f", params=[], body=CompoundStmt()),
			FunctionDecl(
				return_type=void_type(), name="f", params=[], body=CompoundStmt(), loc=loc(5, 1)
			),
		])
		with pytest.raises(SemanticError, match="redefinition of 'f'"):
			SemanticAnalyzer().analyze(prog)

	def test_duplicate_param(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[
					ParamDecl(type_spec=int_type(), name="a"),
					ParamDecl(type_spec=int_type(), name="a", loc=loc(1, 20)),
				],
				body=CompoundStmt(),
			),
		])
		with pytest.raises(SemanticError, match="redefinition of 'a'"):
			SemanticAnalyzer().analyze(prog)


class TestTypeMismatch:
	def test_incompatible_binary_op(self) -> None:
		"""Pointer % int is not valid."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=ptr_type("int"), name="p"),
					ExprStmt(expression=BinaryOp(
						left=Identifier(name="p"),
						op="%",
						right=IntLiteral(value=2),
						loc=loc(3, 5),
					)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="(incompatible types|pointer type not allowed)"):
			SemanticAnalyzer().analyze(prog)

	def test_return_type_mismatch(self) -> None:
		"""int func() { return "hello"; }  -- string where int expected"""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					ReturnStmt(
						expression=StringLiteral(value="hello"),
						loc=loc(2, 3),
					),
				]),
			),
		])
		with pytest.raises(SemanticError, match="incompatible return type"):
			SemanticAnalyzer().analyze(prog)

	def test_assignment_type_incompatible(self) -> None:
		"""int x; x = "hello";"""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x"),
					ExprStmt(expression=Assignment(
						target=Identifier(name="x"),
						value=StringLiteral(value="hi"),
						loc=loc(3, 5),
					)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="incompatible types in assignment"):
			SemanticAnalyzer().analyze(prog)

	def test_init_type_incompatible(self) -> None:
		"""int x = "hello";"""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=int_type(),
						name="x",
						initializer=StringLiteral(value="hi"),
						loc=loc(2, 3),
					),
				]),
			),
		])
		with pytest.raises(SemanticError, match="incompatible types in initialization"):
			SemanticAnalyzer().analyze(prog)


class TestFunctionCallErrors:
	def test_too_few_arguments(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="add",
				params=[
					ParamDecl(type_spec=int_type(), name="a"),
					ParamDecl(type_spec=int_type(), name="b"),
				],
				body=CompoundStmt(statements=[
					ReturnStmt(expression=BinaryOp(
						left=Identifier(name="a"), op="+", right=Identifier(name="b"),
					)),
				]),
			),
			FunctionDecl(
				return_type=int_type(),
				name="main",
				params=[],
				body=CompoundStmt(statements=[
					ReturnStmt(expression=FunctionCall(
						name="add",
						arguments=[IntLiteral(value=1)],
						loc=loc(5, 10),
					)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="expects 2 arguments, got 1"):
			SemanticAnalyzer().analyze(prog)

	def test_too_many_arguments(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="noop",
				params=[],
				body=CompoundStmt(),
			),
			FunctionDecl(
				return_type=void_type(),
				name="main",
				params=[],
				body=CompoundStmt(statements=[
					ExprStmt(expression=FunctionCall(
						name="noop",
						arguments=[IntLiteral(value=1)],
						loc=loc(4, 3),
					)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="expects 0 arguments, got 1"):
			SemanticAnalyzer().analyze(prog)

	def test_call_undeclared_function(self) -> None:
		"""C89: calling an undeclared function emits a warning (implicit declaration)."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="main",
				params=[],
				body=CompoundStmt(statements=[
					ExprStmt(expression=FunctionCall(
						name="unknown",
						arguments=[],
						loc=loc(2, 3),
					)),
				]),
			),
		])
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)
		assert any("implicit declaration of function 'unknown'" in w for w in analyzer.warnings)

	def test_call_non_function(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x"),
					ExprStmt(expression=FunctionCall(
						name="x",
						arguments=[],
						loc=loc(3, 3),
					)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="'x' is not a function"):
			SemanticAnalyzer().analyze(prog)


class TestScopeManagement:
	def test_for_loop_var_scoped(self) -> None:
		"""Variable declared in for-init should not be visible after the loop."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					ForStmt(
						init=VarDecl(type_spec=int_type(), name="i", initializer=IntLiteral(value=0)),
						condition=BinaryOp(
							left=Identifier(name="i"), op="<", right=IntLiteral(value=10),
						),
						update=Assignment(
							target=Identifier(name="i"),
							value=BinaryOp(
								left=Identifier(name="i"), op="+", right=IntLiteral(value=1),
							),
						),
						body=CompoundStmt(statements=[]),
					),
					ExprStmt(expression=Identifier(name="i", loc=loc(5, 3))),
				]),
			),
		])
		with pytest.raises(SemanticError, match="undeclared identifier 'i'"):
			SemanticAnalyzer().analyze(prog)

	def test_param_visible_in_body(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="identity",
				params=[ParamDecl(type_spec=int_type(), name="x")],
				body=CompoundStmt(statements=[
					ReturnStmt(expression=Identifier(name="x")),
				]),
			),
		])
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)

	def test_param_not_visible_outside_function(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[ParamDecl(type_spec=int_type(), name="a")],
				body=CompoundStmt(),
			),
			FunctionDecl(
				return_type=void_type(),
				name="g",
				params=[],
				body=CompoundStmt(statements=[
					ExprStmt(expression=Identifier(name="a", loc=loc(5, 3))),
				]),
			),
		])
		with pytest.raises(SemanticError, match="undeclared identifier 'a'"):
			SemanticAnalyzer().analyze(prog)


class TestErrorLocationInfo:
	def test_error_has_line_and_col(self) -> None:
		err = SemanticError("test error", line=10, col=5)
		assert err.line == 10
		assert err.col == 5
		assert "10:5" in str(err)

	def test_undeclared_error_preserves_location(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					ReturnStmt(expression=Identifier(name="nope", loc=loc(7, 12))),
				]),
			),
		])
		with pytest.raises(SemanticError) as exc_info:
			SemanticAnalyzer().analyze(prog)
		assert exc_info.value.line == 7
		assert exc_info.value.col == 12


class TestUnaryOpErrors:
	def test_deref_non_pointer(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=1)),
					ExprStmt(expression=UnaryOp(
						op="*", operand=Identifier(name="x"), loc=loc(3, 3),
					)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="dereference of non-pointer"):
			SemanticAnalyzer().analyze(prog)


# ---------------------------------------------------------------------------
# Array support tests
# ---------------------------------------------------------------------------


class TestArrayDeclSemantic:
	def test_array_decl_valid(self) -> None:
		"""int arr[10]; -- should pass without errors."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=int_type(),
						name="arr",
						array_sizes=[IntLiteral(value=10)],
					),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_multidim_array_valid(self) -> None:
		"""int matrix[3][4]; -- should pass."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=int_type(),
						name="matrix",
						array_sizes=[IntLiteral(value=3), IntLiteral(value=4)],
					),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)


class TestArraySubscriptSemantic:
	def test_valid_array_subscript(self) -> None:
		"""int arr[10]; return arr[0]; -- valid."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=int_type(),
						name="arr",
						array_sizes=[IntLiteral(value=10)],
					),
					ReturnStmt(expression=ArraySubscript(
						array=Identifier(name="arr"),
						index=IntLiteral(value=0),
					)),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_pointer_subscript_valid(self) -> None:
		"""int *p; return p[0]; -- valid (pointer subscript)."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=ptr_type("int"), name="p"),
					ReturnStmt(expression=ArraySubscript(
						array=Identifier(name="p"),
						index=IntLiteral(value=0),
					)),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_subscript_non_array_non_pointer(self) -> None:
		"""int x; x[0]; -- error: subscript on non-array/non-pointer."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x"),
					ExprStmt(expression=ArraySubscript(
						array=Identifier(name="x"),
						index=IntLiteral(value=0),
						loc=loc(3, 3),
					)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="subscript requires array or pointer"):
			SemanticAnalyzer().analyze(prog)

	def test_subscript_non_integer_index(self) -> None:
		"""int arr[10]; arr["hello"]; -- error: index must be integer."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=int_type(),
						name="arr",
						array_sizes=[IntLiteral(value=10)],
					),
					ExprStmt(expression=ArraySubscript(
						array=Identifier(name="arr"),
						index=StringLiteral(value="hello"),
						loc=loc(3, 3),
					)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="array index must be an integer"):
			SemanticAnalyzer().analyze(prog)

	def test_array_subscript_result_type(self) -> None:
		"""int *p; p[0] should have type int (dereference pointer)."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=ptr_type("int"), name="p"),
					ReturnStmt(expression=ArraySubscript(
						array=Identifier(name="p"),
						index=IntLiteral(value=0),
					)),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)


# ---------------------------------------------------------------------------
# Break / Continue validation
# ---------------------------------------------------------------------------


class TestBreakContinueValid:
	def test_break_in_while(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					WhileStmt(
						condition=IntLiteral(value=1),
						body=CompoundStmt(statements=[BreakStmt()]),
					),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_continue_in_while(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					WhileStmt(
						condition=IntLiteral(value=1),
						body=CompoundStmt(statements=[ContinueStmt()]),
					),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_break_in_for(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					ForStmt(
						init=None,
						condition=None,
						update=None,
						body=CompoundStmt(statements=[BreakStmt()]),
					),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_continue_in_for(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					ForStmt(
						init=None,
						condition=None,
						update=None,
						body=CompoundStmt(statements=[ContinueStmt()]),
					),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_break_in_do_while(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					DoWhileStmt(
						body=CompoundStmt(statements=[BreakStmt()]),
						condition=IntLiteral(value=0),
					),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_continue_in_do_while(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					DoWhileStmt(
						body=CompoundStmt(statements=[ContinueStmt()]),
						condition=IntLiteral(value=0),
					),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_break_in_nested_loop(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					WhileStmt(
						condition=IntLiteral(value=1),
						body=CompoundStmt(statements=[
							ForStmt(
								init=None,
								condition=None,
								update=None,
								body=CompoundStmt(statements=[BreakStmt()]),
							),
						]),
					),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_break_inside_if_inside_loop(self) -> None:
		"""break inside if inside while is valid."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					WhileStmt(
						condition=IntLiteral(value=1),
						body=CompoundStmt(statements=[
							IfStmt(
								condition=IntLiteral(value=1),
								then_branch=CompoundStmt(statements=[BreakStmt()]),
							),
						]),
					),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)


class TestBreakContinueInvalid:
	def test_break_outside_loop(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					BreakStmt(loc=loc(2, 3)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="break statement not within a loop"):
			SemanticAnalyzer().analyze(prog)

	def test_continue_outside_loop(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					ContinueStmt(loc=loc(2, 3)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="continue statement not within a loop"):
			SemanticAnalyzer().analyze(prog)

	def test_break_in_if_outside_loop(self) -> None:
		"""break inside if but NOT inside any loop is invalid."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					IfStmt(
						condition=IntLiteral(value=1),
						then_branch=CompoundStmt(statements=[
							BreakStmt(loc=loc(3, 5)),
						]),
					),
				]),
			),
		])
		with pytest.raises(SemanticError, match="break statement not within a loop"):
			SemanticAnalyzer().analyze(prog)

	def test_continue_in_if_outside_loop(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					IfStmt(
						condition=IntLiteral(value=1),
						then_branch=CompoundStmt(statements=[
							ContinueStmt(loc=loc(3, 5)),
						]),
					),
				]),
			),
		])
		with pytest.raises(SemanticError, match="continue statement not within a loop"):
			SemanticAnalyzer().analyze(prog)


# ---------------------------------------------------------------------------
# Do-while semantic tests
# ---------------------------------------------------------------------------


class TestDoWhileSemantic:
	def test_do_while_valid(self) -> None:
		"""do { ... } while (x); with declared x should pass."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=1)),
					DoWhileStmt(
						body=CompoundStmt(statements=[
							ExprStmt(expression=Assignment(
								target=Identifier(name="x"),
								value=IntLiteral(value=0),
							)),
						]),
						condition=Identifier(name="x"),
					),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_do_while_undeclared_condition(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					DoWhileStmt(
						body=CompoundStmt(statements=[]),
						condition=Identifier(name="unknown", loc=loc(3, 10)),
					),
				]),
			),
		])
		with pytest.raises(SemanticError, match="undeclared identifier 'unknown'"):
			SemanticAnalyzer().analyze(prog)

	def test_do_while_undeclared_in_body(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					DoWhileStmt(
						body=CompoundStmt(statements=[
							ExprStmt(expression=Identifier(name="nope", loc=loc(2, 5))),
						]),
						condition=IntLiteral(value=1),
					),
				]),
			),
		])
		with pytest.raises(SemanticError, match="undeclared identifier 'nope'"):
			SemanticAnalyzer().analyze(prog)


# ---------------------------------------------------------------------------
# Compound assignment semantic tests
# ---------------------------------------------------------------------------


class TestCompoundAssignmentSemantic:
	def test_valid_int_plus_assign(self) -> None:
		"""int x = 1; x += 2; -- valid."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=1)),
					ExprStmt(expression=CompoundAssignment(
						target=Identifier(name="x"),
						op="+",
						value=IntLiteral(value=2),
					)),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_valid_all_arithmetic_ops(self) -> None:
		"""Test +=, -=, *=, /=, %= all work on int."""
		for op in ["+", "-", "*", "/", "%"]:
			prog = Program(declarations=[
				FunctionDecl(
					return_type=void_type(),
					name="f",
					params=[],
					body=CompoundStmt(statements=[
						VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=10)),
						ExprStmt(expression=CompoundAssignment(
							target=Identifier(name="x"),
							op=op,
							value=IntLiteral(value=2),
						)),
					]),
				),
			])
			SemanticAnalyzer().analyze(prog)

	def test_valid_char_compound_assign(self) -> None:
		"""char c = 'a'; c += 1; -- valid (char is numeric)."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=char_type(), name="c", initializer=CharLiteral(value="a")),
					ExprStmt(expression=CompoundAssignment(
						target=Identifier(name="c"),
						op="+",
						value=IntLiteral(value=1),
					)),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_valid_pointer_plus_assign(self) -> None:
		"""int *p; p += 1; -- valid pointer arithmetic."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=ptr_type("int"), name="p"),
					ExprStmt(expression=CompoundAssignment(
						target=Identifier(name="p"),
						op="+",
						value=IntLiteral(value=1),
					)),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_invalid_string_compound_assign(self) -> None:
		"""int x = 1; x += "hello"; -- error: string is not numeric."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=1)),
					ExprStmt(expression=CompoundAssignment(
						target=Identifier(name="x"),
						op="+",
						value=StringLiteral(value="hello"),
						loc=loc(3, 5),
					)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="incompatible type for compound assignment"):
			SemanticAnalyzer().analyze(prog)

	def test_invalid_undeclared_target(self) -> None:
		"""z += 1; -- error: undeclared."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					ExprStmt(expression=CompoundAssignment(
						target=Identifier(name="z", loc=loc(2, 3)),
						op="+",
						value=IntLiteral(value=1),
					)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="undeclared identifier 'z'"):
			SemanticAnalyzer().analyze(prog)

	def test_compound_assign_returns_target_type(self) -> None:
		"""Compound assignment result type should match target."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=1)),
					ReturnStmt(expression=CompoundAssignment(
						target=Identifier(name="x"),
						op="+",
						value=IntLiteral(value=2),
					)),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)


# ---------------------------------------------------------------------------
# Switch/case semantic tests
# ---------------------------------------------------------------------------


class TestSwitchSemanticValid:
	def test_switch_basic(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=1)),
					SwitchStmt(
						expression=Identifier(name="x"),
						cases=[
							CaseClause(value=IntLiteral(value=1), statements=[
								BreakStmt(),
							]),
							CaseClause(value=IntLiteral(value=2), statements=[
								BreakStmt(),
							]),
						],
					),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_switch_with_default(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=1)),
					SwitchStmt(
						expression=Identifier(name="x"),
						cases=[
							CaseClause(value=IntLiteral(value=1), statements=[BreakStmt()]),
							CaseClause(value=None, statements=[BreakStmt()]),
						],
					),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_break_valid_in_switch(self) -> None:
		"""break inside switch is valid even without a loop."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=0)),
					SwitchStmt(
						expression=Identifier(name="x"),
						cases=[
							CaseClause(value=IntLiteral(value=0), statements=[BreakStmt()]),
						],
					),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)


class TestSwitchSemanticInvalid:
	def test_duplicate_case(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=1)),
					SwitchStmt(
						expression=Identifier(name="x"),
						cases=[
							CaseClause(value=IntLiteral(value=1), statements=[BreakStmt()]),
							CaseClause(value=IntLiteral(value=1), statements=[BreakStmt()], loc=loc(3, 5)),
						],
					),
				]),
			),
		])
		with pytest.raises(SemanticError, match="duplicate case value"):
			SemanticAnalyzer().analyze(prog)

	def test_duplicate_default(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=1)),
					SwitchStmt(
						expression=Identifier(name="x"),
						cases=[
							CaseClause(value=None, statements=[BreakStmt()]),
							CaseClause(value=None, statements=[BreakStmt()], loc=loc(4, 5)),
						],
					),
				]),
			),
		])
		with pytest.raises(SemanticError, match="duplicate default label"):
			SemanticAnalyzer().analyze(prog)

	def test_non_constant_case(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=1)),
					VarDecl(type_spec=int_type(), name="y", initializer=IntLiteral(value=2)),
					SwitchStmt(
						expression=Identifier(name="x"),
						cases=[
							CaseClause(value=Identifier(name="y"), statements=[BreakStmt()], loc=loc(3, 5)),
						],
					),
				]),
			),
		])
		with pytest.raises(SemanticError, match="case expression must be a constant integer"):
			SemanticAnalyzer().analyze(prog)


# ---------------------------------------------------------------------------
# Ternary expression semantic tests
# ---------------------------------------------------------------------------


class TestTernarySemantic:
	def test_ternary_valid(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=1)),
					ReturnStmt(expression=TernaryExpr(
						condition=Identifier(name="x"),
						true_expr=IntLiteral(value=1),
						false_expr=IntLiteral(value=0),
					)),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_ternary_incompatible_branches(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=1)),
					ReturnStmt(expression=TernaryExpr(
						condition=Identifier(name="x"),
						true_expr=IntLiteral(value=1),
						false_expr=StringLiteral(value="hello"),
						loc=loc(3, 5),
					)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="incompatible types in ternary"):
			SemanticAnalyzer().analyze(prog)


# ---------------------------------------------------------------------------
# Sizeof semantic tests
# ---------------------------------------------------------------------------


class TestSizeofSemantic:
	def test_sizeof_type_returns_int(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					ReturnStmt(expression=SizeofExpr(type_operand=int_type())),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_sizeof_expr_returns_int(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=0)),
					ReturnStmt(expression=SizeofExpr(operand=Identifier(name="x"))),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)


# ---------------------------------------------------------------------------
# Postfix ++/-- semantic tests
# ---------------------------------------------------------------------------


class TestPostfixSemantic:
	def test_postfix_on_identifier(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x", initializer=IntLiteral(value=0)),
					ExprStmt(expression=PostfixExpr(operand=Identifier(name="x"), op="++")),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_postfix_on_non_lvalue(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					ExprStmt(expression=PostfixExpr(
						operand=IntLiteral(value=1),
						op="++",
						loc=loc(2, 3),
					)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="operand of postfix operator must be an lvalue"):
			SemanticAnalyzer().analyze(prog)

	def test_postfix_on_array_subscript(self) -> None:
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="arr", array_sizes=[IntLiteral(value=10)]),
					ExprStmt(expression=PostfixExpr(
						operand=ArraySubscript(
							array=Identifier(name="arr"),
							index=IntLiteral(value=0),
						),
						op="++",
					)),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)


# ---------------------------------------------------------------------------
# Function prototype semantic tests
# ---------------------------------------------------------------------------


class TestFunctionPrototypeSemantic:
	def test_prototype_registers_function(self) -> None:
		"""Prototype should register function so it can be called."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="foo",
				params=[ParamDecl(type_spec=int_type(), name="x")],
				body=None,
			),
			FunctionDecl(
				return_type=int_type(),
				name="main",
				params=[],
				body=CompoundStmt(statements=[
					ReturnStmt(expression=FunctionCall(
						name="foo",
						arguments=[IntLiteral(value=42)],
					)),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_prototype_then_definition(self) -> None:
		"""Prototype followed by definition should not error."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=int_type(),
				name="foo",
				params=[ParamDecl(type_spec=int_type(), name="x")],
				body=None,
			),
			FunctionDecl(
				return_type=int_type(),
				name="foo",
				params=[ParamDecl(type_spec=int_type(), name="x")],
				body=CompoundStmt(statements=[
					ReturnStmt(expression=Identifier(name="x")),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)


# ---------------------------------------------------------------------------
# Struct declaration and member access semantic tests
# ---------------------------------------------------------------------------


def _make_point_struct() -> StructDecl:
	"""Helper: struct point { int x; int y; }."""
	return StructDecl(
		name="point",
		members=[
			StructMember(type_spec=int_type(), name="x"),
			StructMember(type_spec=int_type(), name="y"),
		],
	)


class TestStructDeclSemantic:
	def test_valid_struct_decl(self) -> None:
		"""struct point { int x; int y; }; -- valid."""
		prog = Program(declarations=[_make_point_struct()])
		SemanticAnalyzer().analyze(prog)

	def test_duplicate_member(self) -> None:
		"""struct bad { int x; int x; }; -- duplicate member error."""
		prog = Program(declarations=[
			StructDecl(
				name="bad",
				members=[
					StructMember(type_spec=int_type(), name="x"),
					StructMember(type_spec=int_type(), name="x", loc=loc(3, 5)),
				],
			),
		])
		with pytest.raises(SemanticError, match="duplicate member 'x' in struct 'bad'"):
			SemanticAnalyzer().analyze(prog)

	def test_redefinition_of_struct(self) -> None:
		"""Two structs with the same name is an error."""
		prog = Program(declarations=[
			_make_point_struct(),
			StructDecl(name="point", members=[
				StructMember(type_spec=int_type(), name="a"),
			], loc=loc(5, 1)),
		])
		with pytest.raises(SemanticError, match="redefinition of struct 'point'"):
			SemanticAnalyzer().analyze(prog)


class TestMemberAccessSemantic:
	def test_valid_dot_access(self) -> None:
		"""struct point p; p.x -- valid."""
		prog = Program(declarations=[
			_make_point_struct(),
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=TypeSpec(base_type="point"), name="p"),
					ReturnStmt(expression=MemberAccess(
						object=Identifier(name="p"),
						member="x",
						is_arrow=False,
					)),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_valid_arrow_access(self) -> None:
		"""struct point *pp; pp->y -- valid."""
		prog = Program(declarations=[
			_make_point_struct(),
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=TypeSpec(base_type="point", pointer_count=1), name="pp"),
					ReturnStmt(expression=MemberAccess(
						object=Identifier(name="pp"),
						member="y",
						is_arrow=True,
					)),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_invalid_member_name(self) -> None:
		"""p.z where z is not a member of struct point -- error."""
		prog = Program(declarations=[
			_make_point_struct(),
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=TypeSpec(base_type="point"), name="p"),
					ExprStmt(expression=MemberAccess(
						object=Identifier(name="p"),
						member="z",
						is_arrow=False,
						loc=loc(3, 5),
					)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="struct point has no member z"):
			SemanticAnalyzer().analyze(prog)

	def test_arrow_on_non_pointer(self) -> None:
		"""struct point p; p->x -- error: arrow requires pointer."""
		prog = Program(declarations=[
			_make_point_struct(),
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=TypeSpec(base_type="point"), name="p"),
					ExprStmt(expression=MemberAccess(
						object=Identifier(name="p"),
						member="x",
						is_arrow=True,
						loc=loc(3, 5),
					)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="requires pointer to struct"):
			SemanticAnalyzer().analyze(prog)

	def test_dot_on_pointer(self) -> None:
		"""struct point *pp; pp.x -- error: use -> instead."""
		prog = Program(declarations=[
			_make_point_struct(),
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=TypeSpec(base_type="point", pointer_count=1), name="pp"),
					ExprStmt(expression=MemberAccess(
						object=Identifier(name="pp"),
						member="x",
						is_arrow=False,
						loc=loc(3, 5),
					)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="requires non-pointer struct"):
			SemanticAnalyzer().analyze(prog)

	def test_member_access_on_non_struct(self) -> None:
		"""int x; x.y -- error: not a struct."""
		prog = Program(declarations=[
			FunctionDecl(
				return_type=void_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=int_type(), name="x"),
					ExprStmt(expression=MemberAccess(
						object=Identifier(name="x"),
						member="y",
						is_arrow=False,
						loc=loc(3, 5),
					)),
				]),
			),
		])
		with pytest.raises(SemanticError, match="member access on non-struct type"):
			SemanticAnalyzer().analyze(prog)

	def test_nested_struct_access(self) -> None:
		"""struct inner { int val; }; struct outer { inner in; }; o.in.val -- valid."""
		inner = StructDecl(
			name="inner",
			members=[StructMember(type_spec=int_type(), name="val")],
		)
		outer = StructDecl(
			name="outer",
			members=[StructMember(type_spec=TypeSpec(base_type="inner"), name="in_")],
		)
		prog = Program(declarations=[
			inner,
			outer,
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=TypeSpec(base_type="outer"), name="o"),
					ReturnStmt(expression=MemberAccess(
						object=MemberAccess(
							object=Identifier(name="o"),
							member="in_",
							is_arrow=False,
						),
						member="val",
						is_arrow=False,
					)),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_member_access_result_type(self) -> None:
		"""Dot access returns the member's type."""
		prog = Program(declarations=[
			StructDecl(
				name="s",
				members=[
					StructMember(type_spec=TypeSpec(base_type="char", pointer_count=1), name="name"),
				],
			),
			FunctionDecl(
				return_type=TypeSpec(base_type="char", pointer_count=1),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=TypeSpec(base_type="s"), name="obj"),
					ReturnStmt(expression=MemberAccess(
						object=Identifier(name="obj"),
						member="name",
						is_arrow=False,
					)),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)

	def test_struct_prefix_in_base_type(self) -> None:
		"""TypeSpec with base_type='struct point' should also work."""
		prog = Program(declarations=[
			_make_point_struct(),
			FunctionDecl(
				return_type=int_type(),
				name="f",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(type_spec=TypeSpec(base_type="struct point"), name="p"),
					ReturnStmt(expression=MemberAccess(
						object=Identifier(name="p"),
						member="x",
						is_arrow=False,
					)),
				]),
			),
		])
		SemanticAnalyzer().analyze(prog)
