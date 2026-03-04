"""Tests for adjacent string literal concatenation (C translation phase 6)."""

from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.tokens import TokenType
from compiler.ast_nodes import StringLiteral


class TestLexerStringConcat:
	"""Test that the lexer merges adjacent STRING_LITERAL tokens."""

	def test_two_adjacent_strings(self) -> None:
		tokens = Lexer('"hello" "world"').tokenize()
		string_tokens = [t for t in tokens if t.type == TokenType.STRING_LITERAL]
		assert len(string_tokens) == 1
		assert string_tokens[0].value == '"helloworld"'

	def test_three_adjacent_strings(self) -> None:
		tokens = Lexer('"hello" " " "world"').tokenize()
		string_tokens = [t for t in tokens if t.type == TokenType.STRING_LITERAL]
		assert len(string_tokens) == 1
		assert string_tokens[0].value == '"hello world"'

	def test_no_concat_when_separated(self) -> None:
		tokens = Lexer('"hello", "world"').tokenize()
		string_tokens = [t for t in tokens if t.type == TokenType.STRING_LITERAL]
		assert len(string_tokens) == 2
		assert string_tokens[0].value == '"hello"'
		assert string_tokens[1].value == '"world"'

	def test_single_string_unchanged(self) -> None:
		tokens = Lexer('"hello"').tokenize()
		string_tokens = [t for t in tokens if t.type == TokenType.STRING_LITERAL]
		assert len(string_tokens) == 1
		assert string_tokens[0].value == '"hello"'

	def test_preserves_first_token_location(self) -> None:
		tokens = Lexer('"hello" "world"').tokenize()
		string_tokens = [t for t in tokens if t.type == TokenType.STRING_LITERAL]
		assert string_tokens[0].line == 1
		assert string_tokens[0].column == 1

	def test_concat_with_escape_sequences(self) -> None:
		tokens = Lexer('"hello\\n" "world"').tokenize()
		string_tokens = [t for t in tokens if t.type == TokenType.STRING_LITERAL]
		assert len(string_tokens) == 1
		assert string_tokens[0].value == '"hello\\nworld"'

	def test_concat_across_whitespace_and_newlines(self) -> None:
		tokens = Lexer('"hello"\n"world"').tokenize()
		string_tokens = [t for t in tokens if t.type == TokenType.STRING_LITERAL]
		assert len(string_tokens) == 1
		assert string_tokens[0].value == '"helloworld"'

	def test_many_adjacent_strings(self) -> None:
		tokens = Lexer('"a" "b" "c" "d" "e"').tokenize()
		string_tokens = [t for t in tokens if t.type == TokenType.STRING_LITERAL]
		assert len(string_tokens) == 1
		assert string_tokens[0].value == '"abcde"'

	def test_empty_strings_concat(self) -> None:
		tokens = Lexer('"" ""').tokenize()
		string_tokens = [t for t in tokens if t.type == TokenType.STRING_LITERAL]
		assert len(string_tokens) == 1
		assert string_tokens[0].value == '""'

	def test_empty_and_nonempty_concat(self) -> None:
		tokens = Lexer('"" "hello" ""').tokenize()
		string_tokens = [t for t in tokens if t.type == TokenType.STRING_LITERAL]
		assert len(string_tokens) == 1
		assert string_tokens[0].value == '"hello"'

	def test_non_string_tokens_preserved(self) -> None:
		tokens = Lexer('"hello" "world" ;').tokenize()
		types = [t.type for t in tokens]
		assert types == [TokenType.STRING_LITERAL, TokenType.SEMICOLON, TokenType.EOF]


class TestFullPipelineStringConcat:
	"""Test adjacent string concat through the full parse pipeline."""

	def test_string_concat_in_variable_init(self) -> None:
		source = 'int main() { char *s = "hello" " " "world"; return 0; }'
		parser = Parser.from_source(source)
		program = parser.parse()
		func = program.declarations[0]
		body = func.body.statements
		var_decl = body[0]
		assert isinstance(var_decl.initializer, StringLiteral)
		assert var_decl.initializer.value == "hello world"

	def test_string_concat_in_function_call(self) -> None:
		source = 'void f(char *s); int main() { f("hello" "world"); return 0; }'
		parser = Parser.from_source(source)
		program = parser.parse()
		func = program.declarations[1]
		body = func.body.statements
		call_stmt = body[0]
		call = call_stmt.expression
		assert len(call.arguments) == 1
		assert isinstance(call.arguments[0], StringLiteral)
		assert call.arguments[0].value == "helloworld"

	def test_string_concat_in_return(self) -> None:
		source = 'char* f() { return "hello" " " "world"; }'
		parser = Parser.from_source(source)
		program = parser.parse()
		func = program.declarations[0]
		ret = func.body.statements[0]
		assert isinstance(ret.expression, StringLiteral)
		assert ret.expression.value == "hello world"
