"""Tests for adjacent string literal concatenation in the parser."""

from compiler.parser import Parser
from compiler.ast_nodes import (
	StringLiteral,
	VarDecl,
	FunctionCall,
	FunctionDecl,
	ExprStmt,
	ReturnStmt,
)


def _parse(source: str):
	return Parser.from_source(source).parse()


def _get_string_value(source: str) -> str:
	"""Parse a program and extract the string literal value from the first var decl initializer."""
	program = _parse(source)
	for decl in program.declarations:
		if isinstance(decl, VarDecl) and isinstance(decl.initializer, StringLiteral):
			return decl.initializer.value
		if isinstance(decl, FunctionDecl):
			for stmt in decl.body.statements:
				if isinstance(stmt, VarDecl) and isinstance(stmt.initializer, StringLiteral):
					return stmt.initializer.value
				if isinstance(stmt, ReturnStmt) and isinstance(stmt.value, StringLiteral):
					return stmt.value.value
				if isinstance(stmt, ExprStmt):
					if isinstance(stmt.expression, StringLiteral):
						return stmt.expression.value
					if isinstance(stmt.expression, FunctionCall):
						for arg in stmt.expression.args:
							if isinstance(arg, StringLiteral):
								return arg.value
	raise AssertionError("No string literal found")


class TestBasicConcatenation:
	def test_two_adjacent_strings(self):
		program = _parse('char *s = "hello" " world";')
		decl = program.declarations[0]
		assert isinstance(decl, VarDecl)
		assert isinstance(decl.initializer, StringLiteral)
		assert decl.initializer.value == "hello world"

	def test_two_adjacent_no_space(self):
		val = _get_string_value('char *s = "foo""bar";')
		assert val == "foobar"

	def test_single_string_unchanged(self):
		program = _parse('char *s = "hello";')
		decl = program.declarations[0]
		assert isinstance(decl.initializer, StringLiteral)
		assert decl.initializer.value == "hello"

	def test_empty_string_concat(self):
		val = _get_string_value('char *s = "" "hello";')
		assert val == "hello"

	def test_both_empty(self):
		val = _get_string_value('char *s = "" "";')
		assert val == ""


class TestMultiPartConcatenation:
	def test_three_strings(self):
		val = _get_string_value('char *s = "a" "b" "c";')
		assert val == "abc"

	def test_four_strings(self):
		val = _get_string_value('char *s = "one" " " "two" " three";')
		assert val == "one two three"

	def test_five_strings(self):
		val = _get_string_value('char *s = "a" "b" "c" "d" "e";')
		assert val == "abcde"

	def test_mixed_empty_and_nonempty(self):
		val = _get_string_value('char *s = "" "a" "" "b" "";')
		assert val == "ab"


class TestConcatenationInInitializers:
	def test_global_var_init(self):
		val = _get_string_value('char *msg = "error: " "file not found";')
		assert val == "error: file not found"

	def test_local_var_init(self):
		source = """
		int main() {
			char *s = "hello" " " "world";
			return 0;
		}
		"""
		program = _parse(source)
		func = program.declarations[0]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl, VarDecl)
		assert isinstance(var_decl.initializer, StringLiteral)
		assert var_decl.initializer.value == "hello world"


class TestConcatenationInFunctionArgs:
	def test_concat_as_function_arg(self):
		source = """
		void foo(char *s);
		int main() {
			foo("hello" " world");
			return 0;
		}
		"""
		program = _parse(source)
		func = program.declarations[1]
		expr_stmt = func.body.statements[0]
		assert isinstance(expr_stmt, ExprStmt)
		call = expr_stmt.expression
		assert isinstance(call, FunctionCall)
		assert len(call.arguments) == 1
		assert isinstance(call.arguments[0], StringLiteral)
		assert call.arguments[0].value == "hello world"

	def test_concat_multiple_args(self):
		source = """
		void foo(char *a, char *b);
		int main() {
			foo("a" "b", "c" "d");
			return 0;
		}
		"""
		program = _parse(source)
		func = program.declarations[1]
		expr_stmt = func.body.statements[0]
		call = expr_stmt.expression
		assert isinstance(call, FunctionCall)
		assert call.arguments[0].value == "ab"
		assert call.arguments[1].value == "cd"


class TestEscapeSequenceInteraction:
	def test_escape_in_first_part(self):
		val = _get_string_value(r'char *s = "hello\n" "world";')
		assert val == "hello\nworld"

	def test_escape_in_second_part(self):
		val = _get_string_value(r'char *s = "hello" "\nworld";')
		assert val == "hello\nworld"

	def test_escape_in_both_parts(self):
		val = _get_string_value(r'char *s = "hello\t" "\tworld";')
		assert val == "hello\t\tworld"

	def test_null_escape(self):
		val = _get_string_value(r'char *s = "abc\0" "def";')
		assert val == "abc\x00def"

	def test_hex_escape(self):
		val = _get_string_value(r'char *s = "\x41" "\x42";')
		assert val == "AB"

	def test_backslash_at_boundary(self):
		val = _get_string_value(r'char *s = "line1\n" "line2\n";')
		assert val == "line1\nline2\n"
