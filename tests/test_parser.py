"""Comprehensive tests for the recursive descent parser."""

import pytest

from compiler.ast_nodes import (
	ArraySubscript,
	Assignment,
	BinaryOp,
	BreakStmt,
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
	Program,
	ReturnStmt,
	StringLiteral,
	UnaryOp,
	VarDecl,
	WhileStmt,
)
from compiler.lexer import Lexer
from compiler.parser import ParseError, Parser


def parse(source: str) -> Program:
	"""Helper: lex + parse source into a Program node."""
	tokens = Lexer(source).tokenize()
	return Parser(tokens).parse()


def parse_single_func(source: str) -> FunctionDecl:
	"""Helper: parse source expecting a single function declaration."""
	prog = parse(source)
	assert len(prog.declarations) == 1
	decl = prog.declarations[0]
	assert isinstance(decl, FunctionDecl)
	return decl


def body_stmts(func: FunctionDecl) -> list:
	"""Helper: extract statements from a function's body."""
	return func.body.statements


# -- Simple return -----------------------------------------------------------


class TestSimpleReturn:
	def test_return_zero(self) -> None:
		func = parse_single_func("int main() { return 0; }")
		assert func.name == "main"
		assert func.return_type.base_type == "int"
		assert len(func.params) == 0
		stmts = body_stmts(func)
		assert len(stmts) == 1
		ret = stmts[0]
		assert isinstance(ret, ReturnStmt)
		assert isinstance(ret.expression, IntLiteral)
		assert ret.expression.value == 0

	def test_return_integer(self) -> None:
		func = parse_single_func("int foo() { return 42; }")
		ret = body_stmts(func)[0]
		assert isinstance(ret, ReturnStmt)
		assert isinstance(ret.expression, IntLiteral)
		assert ret.expression.value == 42

	def test_return_expression(self) -> None:
		func = parse_single_func("int f() { return 1 + 2; }")
		ret = body_stmts(func)[0]
		assert isinstance(ret, ReturnStmt)
		assert isinstance(ret.expression, BinaryOp)

	def test_void_function_bare_return(self) -> None:
		func = parse_single_func("void f() { return; }")
		ret = body_stmts(func)[0]
		assert isinstance(ret, ReturnStmt)


# -- Arithmetic expressions with precedence ----------------------------------


class TestArithmeticPrecedence:
	def test_addition(self) -> None:
		func = parse_single_func("int f() { return 1 + 2; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, BinaryOp)
		assert expr.op == "+"
		assert isinstance(expr.left, IntLiteral)
		assert expr.left.value == 1
		assert isinstance(expr.right, IntLiteral)
		assert expr.right.value == 2

	def test_mul_before_add(self) -> None:
		# 1 + 2 * 3 should parse as 1 + (2 * 3)
		func = parse_single_func("int f() { return 1 + 2 * 3; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, BinaryOp)
		assert expr.op == "+"
		assert isinstance(expr.left, IntLiteral)
		assert expr.left.value == 1
		assert isinstance(expr.right, BinaryOp)
		assert expr.right.op == "*"

	def test_left_associativity(self) -> None:
		# 1 - 2 - 3 should parse as (1 - 2) - 3
		func = parse_single_func("int f() { return 1 - 2 - 3; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, BinaryOp)
		assert expr.op == "-"
		assert isinstance(expr.left, BinaryOp)
		assert expr.left.op == "-"
		assert isinstance(expr.right, IntLiteral)
		assert expr.right.value == 3

	def test_parenthesized_override(self) -> None:
		# (1 + 2) * 3 => (* (+ 1 2) 3)
		func = parse_single_func("int f() { return (1 + 2) * 3; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, BinaryOp)
		assert expr.op == "*"
		assert isinstance(expr.left, BinaryOp)
		assert expr.left.op == "+"

	def test_modulo(self) -> None:
		func = parse_single_func("int f() { return 10 % 3; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, BinaryOp)
		assert expr.op == "%"

	def test_comparison_operators(self) -> None:
		func = parse_single_func("int f() { return 1 < 2; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, BinaryOp)
		assert expr.op == "<"

	def test_equality_operators(self) -> None:
		func = parse_single_func("int f() { return 1 == 2; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, BinaryOp)
		assert expr.op == "=="

	def test_logical_and_or(self) -> None:
		# a && b || c should parse as (a && b) || c
		func = parse_single_func("int f() { return 1 && 2 || 3; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, BinaryOp)
		assert expr.op == "||"
		assert isinstance(expr.left, BinaryOp)
		assert expr.left.op == "&&"

	def test_complex_precedence(self) -> None:
		# 1 + 2 * 3 == 7 && 4 < 5
		func = parse_single_func("int f() { return 1 + 2 * 3 == 7 && 4 < 5; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, BinaryOp)
		assert expr.op == "&&"


# -- Variable declarations ---------------------------------------------------


class TestVarDecl:
	def test_simple_decl(self) -> None:
		func = parse_single_func("int f() { int x; return 0; }")
		decl = body_stmts(func)[0]
		assert isinstance(decl, VarDecl)
		assert decl.name == "x"
		assert decl.type_spec.base_type == "int"
		assert decl.initializer is None

	def test_decl_with_init(self) -> None:
		func = parse_single_func("int f() { int x = 5; return x; }")
		decl = body_stmts(func)[0]
		assert isinstance(decl, VarDecl)
		assert decl.name == "x"
		assert isinstance(decl.initializer, IntLiteral)
		assert decl.initializer.value == 5

	def test_pointer_decl(self) -> None:
		func = parse_single_func("int f() { int *p; return 0; }")
		decl = body_stmts(func)[0]
		assert isinstance(decl, VarDecl)
		assert decl.type_spec.pointer_count == 1

	def test_double_pointer(self) -> None:
		func = parse_single_func("int f() { char **pp; return 0; }")
		decl = body_stmts(func)[0]
		assert isinstance(decl, VarDecl)
		assert decl.type_spec.base_type == "char"
		assert decl.type_spec.pointer_count == 2

	def test_decl_with_expression_init(self) -> None:
		func = parse_single_func("int f() { int x = 1 + 2; return x; }")
		decl = body_stmts(func)[0]
		assert isinstance(decl, VarDecl)
		assert isinstance(decl.initializer, BinaryOp)

	def test_global_var_decl(self) -> None:
		prog = parse("int x = 10;")
		assert len(prog.declarations) == 1
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert decl.name == "x"
		assert isinstance(decl.initializer, IntLiteral)


# -- If/else -----------------------------------------------------------------


class TestIfElse:
	def test_simple_if(self) -> None:
		func = parse_single_func("int f() { if (1) return 0; }")
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, IfStmt)
		assert isinstance(stmt.condition, IntLiteral)
		assert isinstance(stmt.then_branch, ReturnStmt)
		assert stmt.else_branch is None

	def test_if_else(self) -> None:
		src = "int f() { if (x) return 1; else return 0; }"
		func = parse_single_func(src)
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, IfStmt)
		assert isinstance(stmt.then_branch, ReturnStmt)
		assert isinstance(stmt.else_branch, ReturnStmt)

	def test_if_with_block(self) -> None:
		src = "int f() { if (x) { return 1; } }"
		func = parse_single_func(src)
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, IfStmt)
		assert isinstance(stmt.then_branch, CompoundStmt)

	def test_if_else_if(self) -> None:
		src = "int f() { if (a) return 1; else if (b) return 2; else return 3; }"
		func = parse_single_func(src)
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, IfStmt)
		assert isinstance(stmt.else_branch, IfStmt)
		assert isinstance(stmt.else_branch.else_branch, ReturnStmt)

	def test_if_with_comparison(self) -> None:
		func = parse_single_func("int f() { if (x > 0) return 1; }")
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, IfStmt)
		assert isinstance(stmt.condition, BinaryOp)
		assert stmt.condition.op == ">"


# -- While -------------------------------------------------------------------


class TestWhile:
	def test_simple_while(self) -> None:
		src = "int f() { while (1) { return 0; } }"
		func = parse_single_func(src)
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, WhileStmt)
		assert isinstance(stmt.condition, IntLiteral)
		assert isinstance(stmt.body, CompoundStmt)

	def test_while_with_expression(self) -> None:
		src = "int f() { while (x < 10) { x = x + 1; } }"
		func = parse_single_func(src)
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, WhileStmt)
		assert isinstance(stmt.condition, BinaryOp)


# -- For loop ----------------------------------------------------------------


class TestForLoop:
	def test_full_for(self) -> None:
		src = "int f() { for (int i = 0; i < 10; i = i + 1) { return i; } }"
		func = parse_single_func(src)
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, ForStmt)
		assert isinstance(stmt.init, VarDecl)
		assert isinstance(stmt.condition, BinaryOp)
		assert isinstance(stmt.update, Assignment)
		assert isinstance(stmt.body, CompoundStmt)

	def test_for_empty_init(self) -> None:
		src = "int f() { int i; for (; i < 10; i = i + 1) { return 0; } }"
		func = parse_single_func(src)
		stmt = body_stmts(func)[1]
		assert isinstance(stmt, ForStmt)
		assert stmt.init is None

	def test_for_empty_condition(self) -> None:
		src = "int f() { for (int i = 0; ; i = i + 1) { return 0; } }"
		func = parse_single_func(src)
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, ForStmt)
		assert stmt.condition is None

	def test_for_empty_update(self) -> None:
		src = "int f() { for (int i = 0; i < 10; ) { return 0; } }"
		func = parse_single_func(src)
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, ForStmt)
		assert stmt.update is None

	def test_for_expr_init(self) -> None:
		src = "int f() { int i; for (i = 0; i < 10; i = i + 1) { return 0; } }"
		func = parse_single_func(src)
		stmt = body_stmts(func)[1]
		assert isinstance(stmt, ForStmt)
		assert isinstance(stmt.init, Assignment)


# -- Function calls ----------------------------------------------------------


class TestFunctionCall:
	def test_no_args(self) -> None:
		func = parse_single_func("int f() { foo(); }")
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, ExprStmt)
		call = stmt.expression
		assert isinstance(call, FunctionCall)
		assert call.name == "foo"
		assert len(call.arguments) == 0

	def test_one_arg(self) -> None:
		func = parse_single_func("int f() { foo(42); }")
		call = body_stmts(func)[0].expression
		assert isinstance(call, FunctionCall)
		assert len(call.arguments) == 1
		assert isinstance(call.arguments[0], IntLiteral)

	def test_multiple_args(self) -> None:
		func = parse_single_func("int f() { bar(1, 2, 3); }")
		call = body_stmts(func)[0].expression
		assert isinstance(call, FunctionCall)
		assert len(call.arguments) == 3

	def test_nested_call(self) -> None:
		func = parse_single_func("int f() { foo(bar(1)); }")
		call = body_stmts(func)[0].expression
		assert isinstance(call, FunctionCall)
		assert isinstance(call.arguments[0], FunctionCall)

	def test_call_with_expression_arg(self) -> None:
		func = parse_single_func("int f() { foo(1 + 2); }")
		call = body_stmts(func)[0].expression
		assert isinstance(call, FunctionCall)
		assert isinstance(call.arguments[0], BinaryOp)

	def test_call_in_return(self) -> None:
		func = parse_single_func("int f() { return foo(1); }")
		ret = body_stmts(func)[0]
		assert isinstance(ret, ReturnStmt)
		assert isinstance(ret.expression, FunctionCall)


# -- Nested expressions ------------------------------------------------------


class TestNestedExpressions:
	def test_deeply_nested_parens(self) -> None:
		func = parse_single_func("int f() { return ((((1)))); }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, IntLiteral)
		assert expr.value == 1

	def test_unary_negation(self) -> None:
		func = parse_single_func("int f() { return -1; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, UnaryOp)
		assert expr.op == "-"
		assert isinstance(expr.operand, IntLiteral)

	def test_logical_not(self) -> None:
		func = parse_single_func("int f() { return !x; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, UnaryOp)
		assert expr.op == "!"

	def test_bitwise_not(self) -> None:
		func = parse_single_func("int f() { return ~x; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, UnaryOp)
		assert expr.op == "~"

	def test_dereference(self) -> None:
		func = parse_single_func("int f() { return *p; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, UnaryOp)
		assert expr.op == "*"

	def test_address_of(self) -> None:
		func = parse_single_func("int f() { return &x; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, UnaryOp)
		assert expr.op == "&"

	def test_double_negation(self) -> None:
		func = parse_single_func("int f() { return - -1; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, UnaryOp)
		assert expr.op == "-"
		assert isinstance(expr.operand, UnaryOp)
		assert expr.operand.op == "-"

	def test_assignment(self) -> None:
		func = parse_single_func("int f() { x = 5; }")
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, ExprStmt)
		expr = stmt.expression
		assert isinstance(expr, Assignment)
		assert isinstance(expr.target, Identifier)
		assert expr.target.name == "x"
		assert isinstance(expr.value, IntLiteral)

	def test_chained_assignment(self) -> None:
		func = parse_single_func("int f() { x = y = 5; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, Assignment)
		assert isinstance(expr.value, Assignment)

	def test_string_literal(self) -> None:
		func = parse_single_func('int f() { return "hello"; }')
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, StringLiteral)
		assert expr.value == "hello"

	def test_char_literal(self) -> None:
		func = parse_single_func("int f() { return 'a'; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, CharLiteral)
		assert expr.value == "a"


# -- Function parameters ----------------------------------------------------


class TestFunctionParams:
	def test_single_param(self) -> None:
		func = parse_single_func("int f(int x) { return x; }")
		assert len(func.params) == 1
		assert func.params[0].name == "x"
		assert func.params[0].type_spec.base_type == "int"

	def test_multiple_params(self) -> None:
		func = parse_single_func("int add(int a, int b) { return a + b; }")
		assert len(func.params) == 2
		assert func.params[0].name == "a"
		assert func.params[1].name == "b"

	def test_pointer_param(self) -> None:
		func = parse_single_func("void f(int *p) { return; }")
		assert func.params[0].type_spec.pointer_count == 1

	def test_void_params(self) -> None:
		func = parse_single_func("int f(void) { return 0; }")
		assert len(func.params) == 0

	def test_mixed_types(self) -> None:
		func = parse_single_func("int f(int x, char *s, void *p) { return 0; }")
		assert len(func.params) == 3
		assert func.params[0].type_spec.base_type == "int"
		assert func.params[1].type_spec.base_type == "char"
		assert func.params[1].type_spec.pointer_count == 1
		assert func.params[2].type_spec.base_type == "void"
		assert func.params[2].type_spec.pointer_count == 1


# -- Multiple declarations --------------------------------------------------


class TestMultipleDeclarations:
	def test_two_functions(self) -> None:
		src = "int foo() { return 1; } int bar() { return 2; }"
		prog = parse(src)
		assert len(prog.declarations) == 2
		assert isinstance(prog.declarations[0], FunctionDecl)
		assert isinstance(prog.declarations[1], FunctionDecl)
		assert prog.declarations[0].name == "foo"
		assert prog.declarations[1].name == "bar"

	def test_global_var_and_function(self) -> None:
		src = "int x = 10; int main() { return x; }"
		prog = parse(src)
		assert len(prog.declarations) == 2
		assert isinstance(prog.declarations[0], VarDecl)
		assert isinstance(prog.declarations[1], FunctionDecl)


# -- Source location tracking ------------------------------------------------


class TestSourceLocations:
	def test_function_location(self) -> None:
		func = parse_single_func("int main() { return 0; }")
		assert func.loc.line == 1
		assert func.loc.col == 5  # 'main' starts at col 5

	def test_error_location(self) -> None:
		with pytest.raises(ParseError) as exc_info:
			parse("int f() { return 0 }")
		err = exc_info.value
		assert err.line > 0
		assert err.column > 0


# -- Error cases -------------------------------------------------------------


class TestErrorCases:
	def test_missing_semicolon(self) -> None:
		with pytest.raises(ParseError, match="Expected ';'"):
			parse("int f() { return 0 }")

	def test_missing_closing_paren(self) -> None:
		with pytest.raises(ParseError, match="Expected '\\)'"):
			parse("int f() { if (1 { return 0; } }")

	def test_missing_closing_brace(self) -> None:
		with pytest.raises(ParseError, match="Expected '}'"):
			parse("int f() { return 0;")

	def test_unexpected_token(self) -> None:
		with pytest.raises(ParseError):
			parse("int f() { return +; }")

	def test_missing_type_specifier(self) -> None:
		with pytest.raises(ParseError, match="Expected type specifier"):
			parse("f() { return 0; }")

	def test_missing_function_name(self) -> None:
		with pytest.raises(ParseError, match="Expected declaration name"):
			parse("int () { return 0; }")

	def test_missing_param_name(self) -> None:
		with pytest.raises(ParseError, match="Expected parameter name"):
			parse("int f(int) { return 0; }")

	def test_error_has_line_col(self) -> None:
		with pytest.raises(ParseError) as exc_info:
			parse("int f() { return 0 }")
		err = exc_info.value
		assert hasattr(err, "line")
		assert hasattr(err, "column")
		assert "at" in str(err)


# -- Parser.from_source convenience -----------------------------------------


class TestFromSource:
	def test_from_source(self) -> None:
		parser = Parser.from_source("int main() { return 0; }")
		prog = parser.parse()
		assert isinstance(prog, Program)
		assert len(prog.declarations) == 1


# -- Expression statements ---------------------------------------------------


class TestExpressionStatements:
	def test_identifier_stmt(self) -> None:
		func = parse_single_func("int f() { x; }")
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, ExprStmt)
		assert isinstance(stmt.expression, Identifier)

	def test_call_stmt(self) -> None:
		func = parse_single_func("int f() { printf(42); }")
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, ExprStmt)
		assert isinstance(stmt.expression, FunctionCall)


# -- Compound statements -----------------------------------------------------


class TestCompoundStmt:
	def test_empty_block(self) -> None:
		func = parse_single_func("int f() { }")
		assert len(body_stmts(func)) == 0

	def test_nested_blocks(self) -> None:
		func = parse_single_func("int f() { { { return 0; } } }")
		outer = body_stmts(func)[0]
		assert isinstance(outer, CompoundStmt)
		inner = outer.statements[0]
		assert isinstance(inner, CompoundStmt)
		assert isinstance(inner.statements[0], ReturnStmt)


# -- Array declarations and subscripts ----------------------------------------


class TestArrayDecl:
	def test_simple_array(self) -> None:
		func = parse_single_func("int f() { int arr[10]; return 0; }")
		decl = body_stmts(func)[0]
		assert isinstance(decl, VarDecl)
		assert decl.name == "arr"
		assert decl.array_sizes is not None
		assert len(decl.array_sizes) == 1
		assert isinstance(decl.array_sizes[0], IntLiteral)
		assert decl.array_sizes[0].value == 10

	def test_multidim_array(self) -> None:
		func = parse_single_func("int f() { int matrix[3][4]; return 0; }")
		decl = body_stmts(func)[0]
		assert isinstance(decl, VarDecl)
		assert decl.array_sizes is not None
		assert len(decl.array_sizes) == 2
		assert decl.array_sizes[0].value == 3
		assert decl.array_sizes[1].value == 4

	def test_char_array(self) -> None:
		func = parse_single_func("int f() { char buf[256]; return 0; }")
		decl = body_stmts(func)[0]
		assert isinstance(decl, VarDecl)
		assert decl.type_spec.base_type == "char"
		assert decl.array_sizes is not None
		assert decl.array_sizes[0].value == 256

	def test_global_array(self) -> None:
		prog = parse("int arr[5];")
		assert len(prog.declarations) == 1
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert decl.array_sizes is not None
		assert decl.array_sizes[0].value == 5

	def test_no_array_sizes_for_normal_var(self) -> None:
		func = parse_single_func("int f() { int x; return 0; }")
		decl = body_stmts(func)[0]
		assert isinstance(decl, VarDecl)
		assert decl.array_sizes is None


class TestArraySubscript:
	def test_simple_subscript(self) -> None:
		func = parse_single_func("int f() { return arr[0]; }")
		ret = body_stmts(func)[0]
		assert isinstance(ret, ReturnStmt)
		expr = ret.expression
		assert isinstance(expr, ArraySubscript)
		assert isinstance(expr.array, Identifier)
		assert expr.array.name == "arr"
		assert isinstance(expr.index, IntLiteral)
		assert expr.index.value == 0

	def test_subscript_with_variable_index(self) -> None:
		func = parse_single_func("int f() { return arr[i]; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, ArraySubscript)
		assert isinstance(expr.index, Identifier)
		assert expr.index.name == "i"

	def test_subscript_with_expression_index(self) -> None:
		func = parse_single_func("int f() { return arr[i + 1]; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, ArraySubscript)
		assert isinstance(expr.index, BinaryOp)
		assert expr.index.op == "+"

	def test_nested_subscript(self) -> None:
		func = parse_single_func("int f() { return matrix[i][j]; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, ArraySubscript)
		assert isinstance(expr.array, ArraySubscript)
		inner = expr.array
		assert isinstance(inner.array, Identifier)
		assert inner.array.name == "matrix"

	def test_subscript_assignment(self) -> None:
		func = parse_single_func("int f() { arr[0] = 42; }")
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, ExprStmt)
		expr = stmt.expression
		assert isinstance(expr, Assignment)
		assert isinstance(expr.target, ArraySubscript)
		assert isinstance(expr.value, IntLiteral)

	def test_subscript_in_expression(self) -> None:
		func = parse_single_func("int f() { return arr[0] + arr[1]; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, BinaryOp)
		assert isinstance(expr.left, ArraySubscript)
		assert isinstance(expr.right, ArraySubscript)


# -- Break statement ---------------------------------------------------------


class TestBreakStmt:
	def test_break_in_while(self) -> None:
		src = "int f() { while (1) { break; } }"
		func = parse_single_func(src)
		loop = body_stmts(func)[0]
		assert isinstance(loop, WhileStmt)
		inner = loop.body.statements[0]
		assert isinstance(inner, BreakStmt)

	def test_break_in_for(self) -> None:
		src = "int f() { for (int i = 0; i < 10; i = i + 1) { break; } }"
		func = parse_single_func(src)
		loop = body_stmts(func)[0]
		assert isinstance(loop, ForStmt)
		inner = loop.body.statements[0]
		assert isinstance(inner, BreakStmt)

	def test_break_with_if(self) -> None:
		src = "int f() { while (1) { if (x) break; } }"
		func = parse_single_func(src)
		loop = body_stmts(func)[0]
		assert isinstance(loop, WhileStmt)
		if_stmt = loop.body.statements[0]
		assert isinstance(if_stmt, IfStmt)
		assert isinstance(if_stmt.then_branch, BreakStmt)

	def test_break_missing_semicolon(self) -> None:
		with pytest.raises(ParseError, match="Expected ';' after 'break'"):
			parse("int f() { while (1) { break } }")


# -- Continue statement ------------------------------------------------------


class TestContinueStmt:
	def test_continue_in_while(self) -> None:
		src = "int f() { while (1) { continue; } }"
		func = parse_single_func(src)
		loop = body_stmts(func)[0]
		assert isinstance(loop, WhileStmt)
		inner = loop.body.statements[0]
		assert isinstance(inner, ContinueStmt)

	def test_continue_in_for(self) -> None:
		src = "int f() { for (int i = 0; i < 10; i = i + 1) { continue; } }"
		func = parse_single_func(src)
		loop = body_stmts(func)[0]
		assert isinstance(loop, ForStmt)
		inner = loop.body.statements[0]
		assert isinstance(inner, ContinueStmt)

	def test_continue_with_if(self) -> None:
		src = "int f() { while (1) { if (x) continue; } }"
		func = parse_single_func(src)
		loop = body_stmts(func)[0]
		if_stmt = loop.body.statements[0]
		assert isinstance(if_stmt, IfStmt)
		assert isinstance(if_stmt.then_branch, ContinueStmt)

	def test_continue_missing_semicolon(self) -> None:
		with pytest.raises(ParseError, match="Expected ';' after 'continue'"):
			parse("int f() { while (1) { continue } }")


# -- Do-while statement ------------------------------------------------------


class TestDoWhileStmt:
	def test_simple_do_while(self) -> None:
		src = "int f() { do { x = x + 1; } while (x < 10); }"
		func = parse_single_func(src)
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, DoWhileStmt)
		assert isinstance(stmt.body, CompoundStmt)
		assert isinstance(stmt.condition, BinaryOp)
		assert stmt.condition.op == "<"

	def test_do_while_single_stmt(self) -> None:
		src = "int f() { do x = x + 1; while (x < 10); }"
		func = parse_single_func(src)
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, DoWhileStmt)
		assert isinstance(stmt.body, ExprStmt)

	def test_do_while_with_break(self) -> None:
		src = "int f() { do { if (x) break; } while (1); }"
		func = parse_single_func(src)
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, DoWhileStmt)
		body_inner = stmt.body.statements[0]
		assert isinstance(body_inner, IfStmt)
		assert isinstance(body_inner.then_branch, BreakStmt)

	def test_do_while_with_continue(self) -> None:
		src = "int f() { do { continue; } while (x); }"
		func = parse_single_func(src)
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, DoWhileStmt)
		assert isinstance(stmt.body.statements[0], ContinueStmt)

	def test_do_while_missing_while(self) -> None:
		with pytest.raises(ParseError, match="Expected 'while'"):
			parse("int f() { do { x = 1; } (x < 10); }")

	def test_do_while_missing_semicolon(self) -> None:
		with pytest.raises(ParseError, match="Expected ';' after do-while"):
			parse("int f() { do { x = 1; } while (x < 10) }")

	def test_nested_do_while(self) -> None:
		src = "int f() { do { do { x = 1; } while (a); } while (b); }"
		func = parse_single_func(src)
		outer = body_stmts(func)[0]
		assert isinstance(outer, DoWhileStmt)
		inner = outer.body.statements[0]
		assert isinstance(inner, DoWhileStmt)


# -- Compound assignment -----------------------------------------------------


class TestCompoundAssignment:
	def test_plus_assign(self) -> None:
		func = parse_single_func("int f() { x += 1; }")
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, ExprStmt)
		expr = stmt.expression
		assert isinstance(expr, CompoundAssignment)
		assert isinstance(expr.target, Identifier)
		assert expr.target.name == "x"
		assert expr.op == "+="
		assert isinstance(expr.value, IntLiteral)
		assert expr.value.value == 1

	def test_minus_assign(self) -> None:
		func = parse_single_func("int f() { x -= 5; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, CompoundAssignment)
		assert expr.op == "-="

	def test_star_assign(self) -> None:
		func = parse_single_func("int f() { x *= 2; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, CompoundAssignment)
		assert expr.op == "*="

	def test_slash_assign(self) -> None:
		func = parse_single_func("int f() { x /= 3; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, CompoundAssignment)
		assert expr.op == "/="

	def test_percent_assign(self) -> None:
		func = parse_single_func("int f() { x %= 4; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, CompoundAssignment)
		assert expr.op == "%="

	def test_compound_assign_with_expression(self) -> None:
		func = parse_single_func("int f() { x += 1 + 2; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, CompoundAssignment)
		assert expr.op == "+="
		assert isinstance(expr.value, BinaryOp)

	def test_compound_assign_array(self) -> None:
		func = parse_single_func("int f() { arr[0] += 1; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, CompoundAssignment)
		assert isinstance(expr.target, ArraySubscript)
		assert expr.op == "+="

	def test_compound_assign_in_for_update(self) -> None:
		src = "int f() { for (int i = 0; i < 10; i += 1) { return i; } }"
		func = parse_single_func(src)
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, ForStmt)
		assert isinstance(stmt.update, CompoundAssignment)
		assert stmt.update.op == "+="


# -- Prefix increment/decrement ---------------------------------------------


class TestPrefixIncDec:
	def test_prefix_increment(self) -> None:
		func = parse_single_func("int f() { ++x; }")
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, ExprStmt)
		expr = stmt.expression
		assert isinstance(expr, UnaryOp)
		assert expr.op == "++"
		assert expr.prefix is True
		assert isinstance(expr.operand, Identifier)
		assert expr.operand.name == "x"

	def test_prefix_decrement(self) -> None:
		func = parse_single_func("int f() { --x; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, UnaryOp)
		assert expr.op == "--"
		assert expr.prefix is True

	def test_prefix_increment_in_expression(self) -> None:
		func = parse_single_func("int f() { return ++x + 1; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, BinaryOp)
		assert expr.op == "+"
		assert isinstance(expr.left, UnaryOp)
		assert expr.left.op == "++"

	def test_prefix_increment_in_for(self) -> None:
		src = "int f() { for (int i = 0; i < 10; ++i) { return i; } }"
		func = parse_single_func(src)
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, ForStmt)
		assert isinstance(stmt.update, UnaryOp)
		assert stmt.update.op == "++"

	def test_prefix_decrement_in_while(self) -> None:
		src = "int f() { while (--n) { return n; } }"
		func = parse_single_func(src)
		stmt = body_stmts(func)[0]
		assert isinstance(stmt, WhileStmt)
		assert isinstance(stmt.condition, UnaryOp)
		assert stmt.condition.op == "--"

	def test_double_prefix_increment(self) -> None:
		func = parse_single_func("int f() { ++ ++x; }")
		expr = body_stmts(func)[0].expression
		assert isinstance(expr, UnaryOp)
		assert expr.op == "++"
		assert isinstance(expr.operand, UnaryOp)
		assert expr.operand.op == "++"
