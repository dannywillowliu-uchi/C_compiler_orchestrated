"""Tests for C89 compatibility features: switch pre-case statements and implicit function declarations."""

from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer
from compiler.ast_nodes import SwitchStmt


class TestSwitchStatementsBeforeFirstCase:
	"""Switch bodies may contain statements before the first case/default label."""

	def test_statements_before_first_case(self):
		"""Statements before the first case should parse without error."""
		source = """
		int main() {
			int x = 1;
			switch (x) {
				x = 2;
				case 1:
					x = 10;
					break;
				case 2:
					x = 20;
					break;
			}
			return x;
		}
		"""
		parser = Parser.from_source(source)
		program = parser.parse()
		func = program.declarations[0]
		switch_stmt = func.body.statements[1]
		assert isinstance(switch_stmt, SwitchStmt)
		# First case should be the pre-switch statements
		assert switch_stmt.cases[0].is_pre_switch is True
		assert switch_stmt.cases[0].value is None
		assert len(switch_stmt.cases[0].statements) == 1  # x = 2;
		# Regular cases follow
		assert switch_stmt.cases[1].value is not None
		assert switch_stmt.cases[2].value is not None

	def test_switch_with_only_default(self):
		"""Switch with only a default label should work."""
		source = """
		int main() {
			int x = 5;
			switch (x) {
				default:
					x = 99;
					break;
			}
			return x;
		}
		"""
		parser = Parser.from_source(source)
		program = parser.parse()
		func = program.declarations[0]
		switch_stmt = func.body.statements[1]
		assert isinstance(switch_stmt, SwitchStmt)
		assert len(switch_stmt.cases) == 1
		assert switch_stmt.cases[0].value is None
		assert switch_stmt.cases[0].is_pre_switch is False

	def test_switch_pre_case_semantic_analysis(self):
		"""Semantic analysis should pass for switch with pre-case statements."""
		source = """
		int main() {
			int x = 1;
			switch (x) {
				x = 2;
				case 1:
					x = 10;
					break;
				default:
					x = 0;
					break;
			}
			return x;
		}
		"""
		parser = Parser.from_source(source)
		program = parser.parse()
		analyzer = SemanticAnalyzer()
		# Should not raise
		analyzer.analyze(program)

	def test_switch_pre_case_not_counted_as_default(self):
		"""Pre-switch statements should not count as a default label."""
		source = """
		int main() {
			int x = 1;
			switch (x) {
				x = 2;
				default:
					x = 0;
					break;
			}
			return x;
		}
		"""
		parser = Parser.from_source(source)
		program = parser.parse()
		analyzer = SemanticAnalyzer()
		# Should not raise - only one default clause
		analyzer.analyze(program)


class TestImplicitFunctionDeclaration:
	"""C89 allows calling undeclared functions, implicitly declaring them as returning int."""

	def test_implicit_declaration_no_error(self):
		"""Calling an undeclared function should not raise a semantic error."""
		source = """
		int main() {
			int x = foo(1, 2);
			return x;
		}
		"""
		parser = Parser.from_source(source)
		program = parser.parse()
		analyzer = SemanticAnalyzer()
		# Should not raise
		analyzer.analyze(program)

	def test_implicit_declaration_emits_warning(self):
		"""Calling an undeclared function should emit a warning."""
		source = """
		int main() {
			bar();
			return 0;
		}
		"""
		parser = Parser.from_source(source)
		program = parser.parse()
		analyzer = SemanticAnalyzer()
		analyzer.analyze(program)
		assert any("implicit declaration of function 'bar'" in w for w in analyzer.warnings)

	def test_implicit_declaration_returns_int(self):
		"""Implicitly declared functions should have int return type."""
		source = """
		int main() {
			int x = implicit_ret(42);
			return x;
		}
		"""
		parser = Parser.from_source(source)
		program = parser.parse()
		analyzer = SemanticAnalyzer()
		# Should not raise - implicit int return is compatible with int variable
		analyzer.analyze(program)
		assert any("implicit declaration of function 'implicit_ret'" in w for w in analyzer.warnings)

	def test_implicit_declaration_accepts_any_args(self):
		"""Implicitly declared functions should accept any number of arguments."""
		source = """
		int main() {
			implicit_fn();
			implicit_fn(1);
			implicit_fn(1, 2, 3);
			return 0;
		}
		"""
		parser = Parser.from_source(source)
		program = parser.parse()
		analyzer = SemanticAnalyzer()
		# Should not raise - implicit declarations accept any arguments
		analyzer.analyze(program)

	def test_explicit_then_no_implicit(self):
		"""Explicitly declared functions should not trigger implicit declaration."""
		source = """
		int foo(int a) { return a; }
		int main() {
			int x = foo(1);
			return x;
		}
		"""
		parser = Parser.from_source(source)
		program = parser.parse()
		analyzer = SemanticAnalyzer()
		analyzer.analyze(program)
		assert not any("implicit declaration" in w for w in analyzer.warnings)
