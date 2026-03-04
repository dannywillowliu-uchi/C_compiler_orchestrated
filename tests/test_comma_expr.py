"""Tests for comma operator expression support."""

from compiler.ast_nodes import (
	Assignment,
	CommaExpr,
	CompoundStmt,
	ExprStmt,
	FunctionDecl,
	IntLiteral,
	ReturnStmt,
	VarDecl,
)
from compiler.ir_gen import IRGenerator
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer


def _parse(source: str):
	return Parser.from_source(source).parse()


def _func_body(source: str) -> CompoundStmt:
	prog = _parse(source)
	func = prog.declarations[0]
	assert isinstance(func, FunctionDecl)
	assert func.body is not None
	return func.body


class TestCommaExprParser:
	"""Test that the parser produces CommaExpr nodes correctly."""

	def test_simple_comma(self):
		body = _func_body("int main() { 1, 2; }")
		stmt = body.statements[0]
		assert isinstance(stmt, ExprStmt)
		expr = stmt.expression
		assert isinstance(expr, CommaExpr)
		assert isinstance(expr.left, IntLiteral) and expr.left.value == 1
		assert isinstance(expr.right, IntLiteral) and expr.right.value == 2

	def test_chained_comma(self):
		body = _func_body("int main() { 1, 2, 3; }")
		stmt = body.statements[0]
		assert isinstance(stmt, ExprStmt)
		expr = stmt.expression
		# Should be left-associative: (1, 2), 3
		assert isinstance(expr, CommaExpr)
		assert isinstance(expr.right, IntLiteral) and expr.right.value == 3
		left = expr.left
		assert isinstance(left, CommaExpr)
		assert isinstance(left.left, IntLiteral) and left.left.value == 1
		assert isinstance(left.right, IntLiteral) and left.right.value == 2

	def test_comma_with_assignment(self):
		body = _func_body("int main() { int a; int b; a = 1, b = 2; }")
		stmt = body.statements[2]
		assert isinstance(stmt, ExprStmt)
		expr = stmt.expression
		assert isinstance(expr, CommaExpr)
		assert isinstance(expr.left, Assignment)
		assert isinstance(expr.right, Assignment)

	def test_comma_in_parentheses(self):
		body = _func_body("int main() { int x; x = (1, 2); }")
		stmt = body.statements[1]
		assert isinstance(stmt, ExprStmt)
		assert isinstance(stmt.expression, Assignment)
		val = stmt.expression.value
		assert isinstance(val, CommaExpr)
		assert isinstance(val.left, IntLiteral) and val.left.value == 1
		assert isinstance(val.right, IntLiteral) and val.right.value == 2

	def test_comma_not_in_function_args(self):
		"""Comma in function arguments should be a separator, not an operator."""
		prog = _parse("int add(int a, int b); int main() { add(1, 2); }")
		func = prog.declarations[1]
		assert isinstance(func, FunctionDecl)
		body = func.body
		assert body is not None
		stmt = body.statements[0]
		assert isinstance(stmt, ExprStmt)
		# The expression should be a FunctionCall, not a CommaExpr
		from compiler.ast_nodes import FunctionCall
		assert isinstance(stmt.expression, FunctionCall)
		assert len(stmt.expression.arguments) == 2

	def test_comma_not_in_var_init(self):
		"""Comma in multi-variable declaration should be a separator."""
		body = _func_body("int main() { int a = 1, b = 2; }")
		# Should produce two VarDecl nodes, not one with CommaExpr
		assert isinstance(body.statements[0], VarDecl)
		assert body.statements[0].name == "a"
		assert isinstance(body.statements[1], VarDecl)
		assert body.statements[1].name == "b"

	def test_comma_in_return(self):
		body = _func_body("int main() { return 1, 2; }")
		stmt = body.statements[0]
		assert isinstance(stmt, ReturnStmt)
		assert isinstance(stmt.expression, CommaExpr)

	def test_comma_in_for_update(self):
		body = _func_body("int main() { int i; int j; for (i = 0; i < 10; i = i + 1, j = j + 1) i; }")
		from compiler.ast_nodes import ForStmt
		stmt = body.statements[2]
		assert isinstance(stmt, ForStmt)
		assert isinstance(stmt.update, CommaExpr)


class TestCommaExprSemantic:
	"""Test that semantic analysis handles CommaExpr correctly."""

	def test_comma_type_is_right_operand(self):
		prog = _parse("int main() { 1, 2; return 0; }")
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)

	def test_comma_with_variables(self):
		prog = _parse("int main() { int a; int b; a = 1, b = 2; return 0; }")
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)

	def test_nested_comma(self):
		prog = _parse("int main() { int x; x = (1, 2, 3); return x; }")
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)


class TestCommaExprIR:
	"""Test that IR generation handles CommaExpr correctly."""

	def test_comma_generates_ir(self):
		prog = _parse("int main() { return (1, 2); }")
		SemanticAnalyzer().analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		assert len(ir_prog.functions) == 1
		func = ir_prog.functions[0]
		# The function should produce IR that returns 2
		from compiler.ir import IRReturn, IRConst
		returns = [i for i in func.body if isinstance(i, IRReturn)]
		assert len(returns) == 1
		ret_val = returns[0].value
		# The return value should be the constant 2 (right operand of comma)
		assert isinstance(ret_val, IRConst) and ret_val.value == 2

	def test_comma_side_effects(self):
		prog = _parse("int main() { int x; x = 1, x = 2; return x; }")
		SemanticAnalyzer().analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		assert len(ir_prog.functions) == 1

	def test_chained_comma_ir(self):
		prog = _parse("int main() { return (1, 2, 3); }")
		SemanticAnalyzer().analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		func = ir_prog.functions[0]
		from compiler.ir import IRReturn, IRConst
		returns = [i for i in func.body if isinstance(i, IRReturn)]
		assert len(returns) == 1
		ret_val = returns[0].value
		assert isinstance(ret_val, IRConst) and ret_val.value == 3
