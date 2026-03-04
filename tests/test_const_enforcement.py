"""Tests for const qualifier enforcement in semantic analysis."""

import pytest

from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer, SemanticError


def _analyze(source: str) -> None:
	"""Parse and run semantic analysis on *source*. Raises SemanticError on violation."""
	parser = Parser.from_source(source)
	ast = parser.parse()
	analyzer = SemanticAnalyzer()
	analyzer.analyze(ast)


# ---------------------------------------------------------------------------
# Negative tests: should raise SemanticError
# ---------------------------------------------------------------------------


class TestConstAssignment:
	"""Direct assignment to a const variable should be rejected."""

	def test_assign_to_const_int(self) -> None:
		with pytest.raises(SemanticError, match="assignment to const variable"):
			_analyze("int main() { const int x = 10; x = 20; return 0; }")

	def test_assign_to_const_char(self) -> None:
		with pytest.raises(SemanticError, match="assignment to const variable"):
			_analyze("int main() { const char c = 'a'; c = 'b'; return 0; }")


class TestConstCompoundAssignment:
	"""Compound assignment (+=, -=, etc.) to a const variable should be rejected."""

	def test_plus_equals_const(self) -> None:
		with pytest.raises(SemanticError, match="assignment to const variable"):
			_analyze("int main() { const int x = 5; x += 1; return 0; }")

	def test_minus_equals_const(self) -> None:
		with pytest.raises(SemanticError, match="assignment to const variable"):
			_analyze("int main() { const int x = 5; x -= 1; return 0; }")

	def test_bitwise_or_equals_const(self) -> None:
		with pytest.raises(SemanticError, match="assignment to const variable"):
			_analyze("int main() { const int x = 5; x |= 2; return 0; }")


class TestConstIncrementDecrement:
	"""Postfix and prefix ++ / -- on a const variable should be rejected."""

	def test_postfix_increment_const(self) -> None:
		with pytest.raises(SemanticError, match="increment/decrement of const variable"):
			_analyze("int main() { const int x = 5; x++; return 0; }")

	def test_postfix_decrement_const(self) -> None:
		with pytest.raises(SemanticError, match="increment/decrement of const variable"):
			_analyze("int main() { const int x = 5; x--; return 0; }")

	def test_prefix_increment_const(self) -> None:
		with pytest.raises(SemanticError, match="increment/decrement of const variable"):
			_analyze("int main() { const int x = 5; ++x; return 0; }")

	def test_prefix_decrement_const(self) -> None:
		with pytest.raises(SemanticError, match="increment/decrement of const variable"):
			_analyze("int main() { const int x = 5; --x; return 0; }")


class TestConstParameter:
	"""Const function parameters should prevent modification in the function body."""

	def test_assign_to_const_param(self) -> None:
		with pytest.raises(SemanticError, match="assignment to const variable"):
			_analyze("int foo(const int x) { x = 10; return x; }")

	def test_compound_assign_to_const_param(self) -> None:
		with pytest.raises(SemanticError, match="assignment to const variable"):
			_analyze("int foo(const int x) { x += 1; return x; }")

	def test_postfix_increment_const_param(self) -> None:
		with pytest.raises(SemanticError, match="increment/decrement of const variable"):
			_analyze("int foo(const int x) { x++; return x; }")

	def test_prefix_decrement_const_param(self) -> None:
		with pytest.raises(SemanticError, match="increment/decrement of const variable"):
			_analyze("int foo(const int x) { --x; return x; }")


class TestConstPointerDistinction:
	"""Pointer-to-const: the pointer itself is not const, only the data it points to."""

	def test_pointer_to_const_can_be_reassigned(self) -> None:
		# const int *p = &x; -- p is NOT const, it can point somewhere else
		_analyze("""
			int main() {
				int x = 10;
				int y = 20;
				const int *p = &x;
				p = &y;
				return 0;
			}
		""")


# ---------------------------------------------------------------------------
# Positive tests: should NOT raise
# ---------------------------------------------------------------------------


class TestConstPositive:
	"""Non-const variables should still be freely assignable."""

	def test_assign_non_const(self) -> None:
		_analyze("int main() { int x = 10; x = 20; return x; }")

	def test_compound_assign_non_const(self) -> None:
		_analyze("int main() { int x = 10; x += 5; return x; }")

	def test_postfix_increment_non_const(self) -> None:
		_analyze("int main() { int x = 10; x++; return x; }")

	def test_prefix_decrement_non_const(self) -> None:
		_analyze("int main() { int x = 10; --x; return x; }")

	def test_const_var_read_only_usage(self) -> None:
		_analyze("int main() { const int x = 10; int y = x + 1; return y; }")

	def test_non_const_param_can_be_modified(self) -> None:
		_analyze("int foo(int x) { x = 42; return x; }")

	def test_const_init_is_ok(self) -> None:
		_analyze("int main() { const int x = 100; return x; }")
