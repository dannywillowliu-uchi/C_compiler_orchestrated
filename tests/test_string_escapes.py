"""Tests for C string and char literal escape sequence handling.

Covers the full pipeline: lexer tokenization, parser interpretation,
IR generation, and codegen .asciz emission.
"""

from compiler.lexer import Lexer, interpret_c_escapes
from compiler.tokens import TokenType
from compiler.parser import Parser
from compiler.ast_nodes import CharLiteral, StringLiteral
from compiler.ir import IRStringData
from compiler.ir_gen import IRGenerator
from compiler.codegen import CodeGenerator, _escape_for_gas
from compiler.semantic import SemanticAnalyzer


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def compile_to_asm(source: str) -> str:
	"""Run the full pipeline from C source to assembly string."""
	parser = Parser.from_source(source)
	ast = parser.parse()
	analyzer = SemanticAnalyzer()
	analyzer.analyze(ast)
	ir_gen = IRGenerator()
	ir_prog = ir_gen.generate(ast)
	return CodeGenerator().generate(ir_prog)


def parse_expr(source: str):
	"""Parse a single function and return the first expression statement's expression."""
	parser = Parser.from_source(source)
	prog = parser.parse()
	func = prog.declarations[0]
	return func.body.statements[0].expression


# ---------------------------------------------------------------------------
# interpret_c_escapes unit tests
# ---------------------------------------------------------------------------

class TestInterpretCEscapes:
	def test_no_escapes(self) -> None:
		assert interpret_c_escapes("hello") == "hello"

	def test_newline(self) -> None:
		assert interpret_c_escapes(r"\n") == "\n"

	def test_tab(self) -> None:
		assert interpret_c_escapes(r"\t") == "\t"

	def test_null(self) -> None:
		assert interpret_c_escapes(r"\0") == "\0"

	def test_backslash(self) -> None:
		assert interpret_c_escapes("\\\\") == "\\"

	def test_single_quote(self) -> None:
		assert interpret_c_escapes(r"\'") == "'"

	def test_double_quote(self) -> None:
		assert interpret_c_escapes(r'\"') == '"'

	def test_alert(self) -> None:
		assert interpret_c_escapes(r"\a") == "\a"

	def test_backspace(self) -> None:
		assert interpret_c_escapes(r"\b") == "\b"

	def test_form_feed(self) -> None:
		assert interpret_c_escapes(r"\f") == "\f"

	def test_carriage_return(self) -> None:
		assert interpret_c_escapes(r"\r") == "\r"

	def test_vertical_tab(self) -> None:
		assert interpret_c_escapes(r"\v") == "\v"

	def test_hex_escape(self) -> None:
		assert interpret_c_escapes(r"\x41") == "A"

	def test_hex_escape_lowercase(self) -> None:
		assert interpret_c_escapes(r"\x61") == "a"

	def test_hex_escape_null(self) -> None:
		assert interpret_c_escapes(r"\x00") == "\x00"

	def test_hex_escape_ff(self) -> None:
		assert interpret_c_escapes(r"\xff") == "\xff"

	def test_octal_escape_012(self) -> None:
		assert interpret_c_escapes(r"\012") == "\n"

	def test_octal_escape_101(self) -> None:
		assert interpret_c_escapes(r"\101") == "A"

	def test_octal_escape_single_digit(self) -> None:
		assert interpret_c_escapes(r"\7") == "\x07"

	def test_octal_escape_two_digits(self) -> None:
		assert interpret_c_escapes(r"\77") == "?"

	def test_multiple_escapes(self) -> None:
		assert interpret_c_escapes(r"hello\nworld\t!") == "hello\nworld\t!"

	def test_mixed_escapes(self) -> None:
		assert interpret_c_escapes(r"\a\b\f\r\v") == "\a\b\f\r\v"

	def test_escaped_backslash_before_n(self) -> None:
		assert interpret_c_escapes("\\\\n") == "\\n"

	def test_empty(self) -> None:
		assert interpret_c_escapes("") == ""

	def test_hex_truncates_to_byte(self) -> None:
		assert interpret_c_escapes(r"\x1FF") == "\xff"

	def test_null_with_trailing_octal(self) -> None:
		"""\\012 should be interpreted as octal 012 = newline."""
		assert interpret_c_escapes(r"\012") == "\n"


# ---------------------------------------------------------------------------
# _escape_for_gas unit tests
# ---------------------------------------------------------------------------

class TestEscapeForGas:
	def test_plain_string(self) -> None:
		assert _escape_for_gas("hello") == "hello"

	def test_newline(self) -> None:
		assert _escape_for_gas("\n") == "\\n"

	def test_tab(self) -> None:
		assert _escape_for_gas("\t") == "\\t"

	def test_null(self) -> None:
		assert _escape_for_gas("\0") == "\\0"

	def test_backslash(self) -> None:
		assert _escape_for_gas("\\") == "\\\\"

	def test_double_quote(self) -> None:
		assert _escape_for_gas('"') == '\\"'

	def test_bell(self) -> None:
		assert _escape_for_gas("\a") == "\\a"

	def test_backspace(self) -> None:
		assert _escape_for_gas("\b") == "\\b"

	def test_form_feed(self) -> None:
		assert _escape_for_gas("\f") == "\\f"

	def test_carriage_return(self) -> None:
		assert _escape_for_gas("\r") == "\\r"

	def test_vertical_tab(self) -> None:
		assert _escape_for_gas("\v") == "\\v"

	def test_non_printable_uses_octal(self) -> None:
		assert _escape_for_gas("\x01") == "\\001"

	def test_roundtrip_all_simple_escapes(self) -> None:
		"""interpret then escape should produce a GAS-compatible string."""
		raw = r"\n\t\0\\\"\a\b\f\r\v"
		interpreted = interpret_c_escapes(raw)
		gas_escaped = _escape_for_gas(interpreted)
		assert gas_escaped == r"\n\t\0\\\"\a\b\f\r\v"


# ---------------------------------------------------------------------------
# Lexer tokenization (escapes preserved as raw text)
# ---------------------------------------------------------------------------

class TestLexerEscapeTokenization:
	def test_string_escape_preserved_raw(self) -> None:
		tok = Lexer(r'"hello\nworld"').tokenize()[0]
		assert tok.type == TokenType.STRING_LITERAL
		assert tok.value == r'"hello\nworld"'

	def test_char_escape_preserved_raw(self) -> None:
		tok = Lexer(r"'\n'").tokenize()[0]
		assert tok.type == TokenType.CHAR_LITERAL
		assert tok.value == r"'\n'"

	def test_hex_escape_in_string(self) -> None:
		tok = Lexer(r'"\x41"').tokenize()[0]
		assert tok.type == TokenType.STRING_LITERAL
		assert tok.value == r'"\x41"'

	def test_octal_escape_in_string(self) -> None:
		tok = Lexer(r'"\101"').tokenize()[0]
		assert tok.type == TokenType.STRING_LITERAL
		assert tok.value == r'"\101"'

	def test_all_simple_escapes_in_string(self) -> None:
		tok = Lexer(r'"\a\b\f\n\r\t\v\\\"\0"').tokenize()[0]
		assert tok.type == TokenType.STRING_LITERAL
		assert tok.value == r'"\a\b\f\n\r\t\v\\\"\0"'


# ---------------------------------------------------------------------------
# Parser escape interpretation
# ---------------------------------------------------------------------------

class TestParserEscapeInterpretation:
	def test_char_newline(self) -> None:
		expr = parse_expr(r"int f() { return '\n'; }")
		assert isinstance(expr, CharLiteral)
		assert expr.value == "\n"

	def test_char_tab(self) -> None:
		expr = parse_expr(r"int f() { return '\t'; }")
		assert isinstance(expr, CharLiteral)
		assert expr.value == "\t"

	def test_char_null(self) -> None:
		expr = parse_expr(r"int f() { return '\0'; }")
		assert isinstance(expr, CharLiteral)
		assert expr.value == "\0"

	def test_char_backslash(self) -> None:
		expr = parse_expr(r"int f() { return '\\'; }")
		assert isinstance(expr, CharLiteral)
		assert expr.value == "\\"

	def test_char_single_quote(self) -> None:
		expr = parse_expr("int f() { return '\\''; }")
		assert isinstance(expr, CharLiteral)
		assert expr.value == "'"

	def test_char_hex(self) -> None:
		expr = parse_expr(r"int f() { return '\x41'; }")
		assert isinstance(expr, CharLiteral)
		assert expr.value == "A"

	def test_char_octal(self) -> None:
		expr = parse_expr(r"int f() { return '\101'; }")
		assert isinstance(expr, CharLiteral)
		assert expr.value == "A"

	def test_string_with_escapes(self) -> None:
		expr = parse_expr(r'int f() { return "hello\nworld"; }')
		assert isinstance(expr, StringLiteral)
		assert expr.value == "hello\nworld"

	def test_string_with_hex_escape(self) -> None:
		expr = parse_expr(r'int f() { return "\x48\x49"; }')
		assert isinstance(expr, StringLiteral)
		assert expr.value == "HI"

	def test_string_with_octal_escape(self) -> None:
		expr = parse_expr(r'int f() { return "\110\111"; }')
		assert isinstance(expr, StringLiteral)
		assert expr.value == "HI"

	def test_string_all_escapes(self) -> None:
		expr = parse_expr(r'int f() { return "\a\b\f\n\r\t\v"; }')
		assert isinstance(expr, StringLiteral)
		assert expr.value == "\a\b\f\n\r\t\v"


# ---------------------------------------------------------------------------
# IR generation (char literal ord values)
# ---------------------------------------------------------------------------

class TestIRGenEscapes:
	def test_char_newline_ordinal(self) -> None:
		"""'\\n' should produce IRConst(10) (newline ordinal)."""
		source = r"""
		int main() {
			return '\n';
		}
		"""
		parser = Parser.from_source(source)
		ast = parser.parse()
		SemanticAnalyzer().analyze(ast)
		ir_prog = IRGenerator().generate(ast)
		func = ir_prog.functions[0]
		from compiler.ir import IRReturn, IRConst
		ret = [i for i in func.body if isinstance(i, IRReturn)][0]
		assert isinstance(ret.value, IRConst)
		assert ret.value.value == 10

	def test_char_tab_ordinal(self) -> None:
		source = r"""
		int main() {
			return '\t';
		}
		"""
		parser = Parser.from_source(source)
		ast = parser.parse()
		SemanticAnalyzer().analyze(ast)
		ir_prog = IRGenerator().generate(ast)
		func = ir_prog.functions[0]
		from compiler.ir import IRReturn, IRConst
		ret = [i for i in func.body if isinstance(i, IRReturn)][0]
		assert isinstance(ret.value, IRConst)
		assert ret.value.value == 9

	def test_char_null_ordinal(self) -> None:
		source = r"""
		int main() {
			return '\0';
		}
		"""
		parser = Parser.from_source(source)
		ast = parser.parse()
		SemanticAnalyzer().analyze(ast)
		ir_prog = IRGenerator().generate(ast)
		func = ir_prog.functions[0]
		from compiler.ir import IRReturn, IRConst
		ret = [i for i in func.body if isinstance(i, IRReturn)][0]
		assert isinstance(ret.value, IRConst)
		assert ret.value.value == 0

	def test_char_hex_ordinal(self) -> None:
		source = r"""
		int main() {
			return '\x41';
		}
		"""
		parser = Parser.from_source(source)
		ast = parser.parse()
		SemanticAnalyzer().analyze(ast)
		ir_prog = IRGenerator().generate(ast)
		func = ir_prog.functions[0]
		from compiler.ir import IRReturn, IRConst
		ret = [i for i in func.body if isinstance(i, IRReturn)][0]
		assert isinstance(ret.value, IRConst)
		assert ret.value.value == 65

	def test_string_escape_in_ir_data(self) -> None:
		"""String with escapes should have interpreted value in IRStringData."""
		source = r'''
		int main() {
			char *s = "hello\nworld";
			return 0;
		}
		'''
		parser = Parser.from_source(source)
		ast = parser.parse()
		SemanticAnalyzer().analyze(ast)
		ir_prog = IRGenerator().generate(ast)
		assert len(ir_prog.string_data) == 1
		assert ir_prog.string_data[0].value == "hello\nworld"


# ---------------------------------------------------------------------------
# Codegen .asciz emission
# ---------------------------------------------------------------------------

class TestCodegenEscapes:
	def test_asciz_with_newline(self) -> None:
		"""Actual newline in string data should emit \\n in .asciz."""
		from compiler.ir import IRProgram, IRFunction, IRReturn, IRConst, IRType
		prog = IRProgram(
			functions=[IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)],
			string_data=[IRStringData(".LC0", "hello\nworld")],
		)
		asm = CodeGenerator().generate(prog)
		assert r'.asciz "hello\nworld"' in asm

	def test_asciz_with_tab(self) -> None:
		from compiler.ir import IRProgram, IRFunction, IRReturn, IRConst, IRType
		prog = IRProgram(
			functions=[IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)],
			string_data=[IRStringData(".LC0", "hello\tworld")],
		)
		asm = CodeGenerator().generate(prog)
		assert r'.asciz "hello\tworld"' in asm

	def test_asciz_with_embedded_quote(self) -> None:
		from compiler.ir import IRProgram, IRFunction, IRReturn, IRConst, IRType
		prog = IRProgram(
			functions=[IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)],
			string_data=[IRStringData(".LC0", 'say "hi"')],
		)
		asm = CodeGenerator().generate(prog)
		assert r'.asciz "say \"hi\""' in asm

	def test_asciz_with_backslash(self) -> None:
		from compiler.ir import IRProgram, IRFunction, IRReturn, IRConst, IRType
		prog = IRProgram(
			functions=[IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)],
			string_data=[IRStringData(".LC0", "path\\to")],
		)
		asm = CodeGenerator().generate(prog)
		assert r'.asciz "path\\to"' in asm

	def test_asciz_with_null_byte(self) -> None:
		from compiler.ir import IRProgram, IRFunction, IRReturn, IRConst, IRType
		prog = IRProgram(
			functions=[IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)],
			string_data=[IRStringData(".LC0", "a\0b")],
		)
		asm = CodeGenerator().generate(prog)
		assert r'.asciz "a\0b"' in asm


# ---------------------------------------------------------------------------
# Full pipeline integration tests
# ---------------------------------------------------------------------------

class TestFullPipelineEscapes:
	def test_string_with_newline_full_pipeline(self) -> None:
		source = r'''
		int main() {
			char *s = "hello\n";
			return 0;
		}
		'''
		asm = compile_to_asm(source)
		assert r'.asciz "hello\n"' in asm

	def test_string_with_all_escapes_full_pipeline(self) -> None:
		source = r'''
		int main() {
			char *s = "\a\b\f\n\r\t\v";
			return 0;
		}
		'''
		asm = compile_to_asm(source)
		assert r'.asciz "\a\b\f\n\r\t\v"' in asm

	def test_string_with_hex_escape_full_pipeline(self) -> None:
		source = r'''
		int main() {
			char *s = "\x48\x45\x4c\x4c\x4f";
			return 0;
		}
		'''
		asm = compile_to_asm(source)
		assert '.asciz "HELLO"' in asm

	def test_string_with_octal_escape_full_pipeline(self) -> None:
		source = r'''
		int main() {
			char *s = "\110\105\114\114\117";
			return 0;
		}
		'''
		asm = compile_to_asm(source)
		assert '.asciz "HELLO"' in asm

	def test_char_escape_in_return(self) -> None:
		source = r"""
		int main() {
			return '\n';
		}
		"""
		asm = compile_to_asm(source)
		assert "movq $10, %rax" in asm

	def test_string_with_escaped_backslash_and_n(self) -> None:
		"""C source \\\\n means literal backslash followed by n."""
		source = r'''
		int main() {
			char *s = "\\n";
			return 0;
		}
		'''
		asm = compile_to_asm(source)
		assert r'.asciz "\\n"' in asm
