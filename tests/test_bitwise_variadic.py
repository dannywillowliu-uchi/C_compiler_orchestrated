"""Tests for bitwise compound assignments and variadic function declarations."""

import pytest

from compiler.ast_nodes import Assignment, BinaryOp, FunctionDecl, Identifier
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer, SemanticError


# ---------------------------------------------------------------------------
# Bitwise compound assignments
# ---------------------------------------------------------------------------

class TestBitwiseCompoundAssignments:
	"""Verify all 5 bitwise compound assignment operators parse and analyze."""

	@pytest.mark.parametrize(
		"op_src, expected_op",
		[
			("x &= mask;", "&"),
			("x |= bits;", "|"),
			("x ^= toggle;", "^"),
			("x <<= shift;", "<<"),
			("x >>= shift;", ">>"),
		],
	)
	def test_parse_bitwise_compound(self, op_src: str, expected_op: str) -> None:
		src = f"int main() {{ int x; int mask; int bits; int toggle; int shift; {op_src} return 0; }}"
		prog = Parser.from_source(src).parse()
		func = prog.declarations[0]
		assert isinstance(func, FunctionDecl)
		# Compound assignments are desugared to Assignment(target, BinaryOp(...))
		assign_stmts = [
			s.expression for s in func.body.statements
			if hasattr(s, "expression") and isinstance(s.expression, Assignment)
			and isinstance(s.expression.value, BinaryOp)
		]
		assert len(assign_stmts) == 1
		a = assign_stmts[0]
		assert a.value.op == expected_op
		assert isinstance(a.target, Identifier)
		assert a.target.name == "x"

	@pytest.mark.parametrize(
		"op_src, expected_op",
		[
			("x &= 0xFF;", "&"),
			("x |= 0x01;", "|"),
			("x ^= 0xAA;", "^"),
			("x <<= 2;", "<<"),
			("x >>= 3;", ">>"),
		],
	)
	def test_semantic_bitwise_compound(self, op_src: str, expected_op: str) -> None:
		src = f"int main() {{ int x; {op_src} return 0; }}"
		prog = Parser.from_source(src).parse()
		analyzer = SemanticAnalyzer()
		errors = analyzer.analyze(prog)
		assert errors == []

	def test_bitwise_compound_chained(self) -> None:
		"""Multiple bitwise compound assignments in sequence."""
		src = """
		int main() {
			int flags;
			flags |= 1;
			flags &= 3;
			flags ^= 2;
			flags <<= 1;
			flags >>= 1;
			return flags;
		}
		"""
		prog = Parser.from_source(src).parse()
		analyzer = SemanticAnalyzer()
		errors = analyzer.analyze(prog)
		assert errors == []
		func = prog.declarations[0]
		assert isinstance(func, FunctionDecl)
		# 1 var decl + 5 compound assigns + 1 return = 7 statements
		assert len(func.body.statements) == 7

	def test_bitwise_compound_with_expression_rhs(self) -> None:
		"""Bitwise compound assignment with a non-trivial RHS expression."""
		src = """
		int main() {
			int x;
			int y;
			x &= y | 3;
			x |= y ^ 5;
			return x;
		}
		"""
		prog = Parser.from_source(src).parse()
		analyzer = SemanticAnalyzer()
		errors = analyzer.analyze(prog)
		assert errors == []


# ---------------------------------------------------------------------------
# Variadic function declarations
# ---------------------------------------------------------------------------

class TestVariadicFunctions:
	"""Verify parsing and semantic analysis of variadic function declarations."""

	def test_parse_variadic_prototype(self) -> None:
		"""Parse a variadic function prototype."""
		src = "int printf(char *fmt, ...);"
		prog = Parser.from_source(src).parse()
		func = prog.declarations[0]
		assert isinstance(func, FunctionDecl)
		assert func.is_variadic is True
		assert len(func.params) == 1
		assert func.params[0].name == "fmt"
		assert func.body is None

	def test_parse_variadic_definition(self) -> None:
		"""Parse a variadic function definition."""
		src = """
		int myprintf(char *fmt, ...) {
			return 0;
		}
		"""
		prog = Parser.from_source(src).parse()
		func = prog.declarations[0]
		assert isinstance(func, FunctionDecl)
		assert func.is_variadic is True
		assert len(func.params) == 1
		assert func.body is not None

	def test_parse_non_variadic(self) -> None:
		"""Non-variadic function should have is_variadic=False."""
		src = "int add(int a, int b) { return a; }"
		prog = Parser.from_source(src).parse()
		func = prog.declarations[0]
		assert isinstance(func, FunctionDecl)
		assert func.is_variadic is False

	def test_variadic_multiple_params(self) -> None:
		"""Variadic function with multiple named params before ellipsis."""
		src = "int foo(int a, int b, char *c, ...);"
		prog = Parser.from_source(src).parse()
		func = prog.declarations[0]
		assert isinstance(func, FunctionDecl)
		assert func.is_variadic is True
		assert len(func.params) == 3

	def test_semantic_variadic_exact_args(self) -> None:
		"""Calling variadic function with exactly the required args succeeds."""
		src = """
		int printf(char *fmt, ...);
		int main() {
			printf("hello");
			return 0;
		}
		"""
		prog = Parser.from_source(src).parse()
		analyzer = SemanticAnalyzer()
		errors = analyzer.analyze(prog)
		assert errors == []

	def test_semantic_variadic_extra_args(self) -> None:
		"""Calling variadic function with extra args succeeds."""
		src = """
		int printf(char *fmt, ...);
		int main() {
			printf("hello %d %d", 1, 2);
			return 0;
		}
		"""
		prog = Parser.from_source(src).parse()
		analyzer = SemanticAnalyzer()
		errors = analyzer.analyze(prog)
		assert errors == []

	def test_semantic_variadic_many_extra_args(self) -> None:
		"""Calling variadic function with many extra args succeeds."""
		src = """
		int printf(char *fmt, ...);
		int main() {
			printf("a%db%dc%d", 1, 2, 3);
			return 0;
		}
		"""
		prog = Parser.from_source(src).parse()
		analyzer = SemanticAnalyzer()
		errors = analyzer.analyze(prog)
		assert errors == []

	def test_semantic_variadic_too_few_args(self) -> None:
		"""Calling variadic function with too few args is an error."""
		src = """
		int mylog(int level, char *fmt, ...);
		int main() {
			mylog(1);
			return 0;
		}
		"""
		prog = Parser.from_source(src).parse()
		analyzer = SemanticAnalyzer()
		with pytest.raises(SemanticError, match="requires at least"):
			analyzer.analyze(prog)

	def test_semantic_non_variadic_wrong_args(self) -> None:
		"""Non-variadic function still rejects wrong arg count."""
		src = """
		int add(int a, int b);
		int main() {
			add(1, 2, 3);
			return 0;
		}
		"""
		prog = Parser.from_source(src).parse()
		analyzer = SemanticAnalyzer()
		with pytest.raises(SemanticError, match="expects 2 arguments"):
			analyzer.analyze(prog)
