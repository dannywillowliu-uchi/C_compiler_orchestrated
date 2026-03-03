"""Tests for enum support, compound assignment fix, and string literal codegen."""

from compiler.ast_nodes import (
	CompoundAssignment,
	EnumDecl,
	IntLiteral,
)
from compiler.codegen import CodeGenerator
from compiler.ir_gen import IRGenerator
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer


# ---------------------------------------------------------------------------
# Enum parsing
# ---------------------------------------------------------------------------


class TestEnumParsing:
	def test_basic_enum(self) -> None:
		src = "enum Color { RED, GREEN, BLUE };"
		prog = Parser.from_source(src).parse()
		assert len(prog.declarations) == 1
		decl = prog.declarations[0]
		assert isinstance(decl, EnumDecl)
		assert decl.name == "Color"
		assert len(decl.constants) == 3
		assert decl.constants[0].name == "RED"
		assert decl.constants[1].name == "GREEN"
		assert decl.constants[2].name == "BLUE"

	def test_enum_with_values(self) -> None:
		src = "enum Status { OK = 0, ERR = 1, WARN = 5 };"
		prog = Parser.from_source(src).parse()
		decl = prog.declarations[0]
		assert isinstance(decl, EnumDecl)
		assert len(decl.constants) == 3
		assert isinstance(decl.constants[0].value, IntLiteral)
		assert decl.constants[0].value.value == 0
		assert isinstance(decl.constants[2].value, IntLiteral)
		assert decl.constants[2].value.value == 5

	def test_enum_mixed_values(self) -> None:
		src = "enum Foo { A, B = 5, C };"
		prog = Parser.from_source(src).parse()
		decl = prog.declarations[0]
		assert isinstance(decl, EnumDecl)
		assert decl.constants[0].value is None
		assert isinstance(decl.constants[1].value, IntLiteral)
		assert decl.constants[1].value.value == 5
		assert decl.constants[2].value is None

	def test_enum_in_function(self) -> None:
		src = """
		int main() {
			enum Dir { UP, DOWN, LEFT, RIGHT };
			return 0;
		}
		"""
		prog = Parser.from_source(src).parse()
		body = prog.declarations[0].body.statements
		assert isinstance(body[0], EnumDecl)
		assert body[0].name == "Dir"


# ---------------------------------------------------------------------------
# Enum semantic analysis
# ---------------------------------------------------------------------------


class TestEnumSemantic:
	def test_enum_constants_registered(self) -> None:
		src = """
		enum Color { RED, GREEN, BLUE };
		int main() {
			int x = RED;
			return x;
		}
		"""
		prog = Parser.from_source(src).parse()
		analyzer = SemanticAnalyzer()
		errors = analyzer.analyze(prog)
		assert len(errors) == 0

	def test_enum_constant_in_expression(self) -> None:
		src = """
		enum Vals { A, B = 10, C };
		int main() {
			int x = A + B + C;
			return x;
		}
		"""
		prog = Parser.from_source(src).parse()
		analyzer = SemanticAnalyzer()
		errors = analyzer.analyze(prog)
		assert len(errors) == 0


# ---------------------------------------------------------------------------
# Enum IR generation
# ---------------------------------------------------------------------------


class TestEnumIR:
	def test_enum_constant_resolves_to_int(self) -> None:
		src = """
		enum Color { RED, GREEN, BLUE };
		int main() {
			return GREEN;
		}
		"""
		prog = Parser.from_source(src).parse()
		SemanticAnalyzer().analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		assert len(ir_prog.functions) == 1

	def test_enum_with_explicit_values(self) -> None:
		src = """
		enum Foo { A, B = 5, C };
		int main() {
			return C;
		}
		"""
		prog = Parser.from_source(src).parse()
		SemanticAnalyzer().analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		# C should be 6 (B=5, C=6)
		fn = ir_prog.functions[0]
		# Find the return instruction; the value should be const 6
		from compiler.ir import IRReturn
		returns = [i for i in fn.body if isinstance(i, IRReturn)]
		assert len(returns) == 1
		from compiler.ir import IRConst
		assert isinstance(returns[0].value, IRConst)
		assert returns[0].value.value == 6


# ---------------------------------------------------------------------------
# Bug fix: compound assignment
# ---------------------------------------------------------------------------


class TestCompoundAssignment:
	def test_parser_uses_base_operator(self) -> None:
		src = """
		int main() {
			int x = 10;
			x += 1;
			x -= 2;
			x *= 3;
			x /= 4;
			x %= 5;
			return x;
		}
		"""
		prog = Parser.from_source(src).parse()
		stmts = prog.declarations[0].body.statements
		# stmts: VarDecl, ExprStmt(+=), ExprStmt(-=), ExprStmt(*=), ExprStmt(/=), ExprStmt(%=), Return
		ops = []
		for s in stmts:
			if hasattr(s, "expression") and isinstance(s.expression, CompoundAssignment):
				ops.append(s.expression.op)
		assert ops == ["+", "-", "*", "/", "%"]

	def test_compound_assignment_semantic(self) -> None:
		src = """
		int main() {
			int x = 10;
			x += 1;
			return x;
		}
		"""
		prog = Parser.from_source(src).parse()
		errors = SemanticAnalyzer().analyze(prog)
		assert len(errors) == 0

	def test_compound_assignment_ir(self) -> None:
		src = """
		int main() {
			int x = 10;
			x += 5;
			return x;
		}
		"""
		prog = Parser.from_source(src).parse()
		SemanticAnalyzer().analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		assert len(ir_prog.functions) == 1

	def test_compound_assignment_codegen(self) -> None:
		src = """
		int main() {
			int x = 10;
			x += 5;
			return x;
		}
		"""
		prog = Parser.from_source(src).parse()
		SemanticAnalyzer().analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		asm = CodeGenerator().generate(ir_prog)
		assert "addq" in asm


# ---------------------------------------------------------------------------
# Bug fix: string literal codegen
# ---------------------------------------------------------------------------


class TestStringLiteral:
	def test_string_literal_ir_returns_global_ref(self) -> None:
		src = """
		int main() {
			char* s = "hello";
			return 0;
		}
		"""
		prog = Parser.from_source(src).parse()
		SemanticAnalyzer().analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		assert len(ir_prog.string_data) == 1
		assert ir_prog.string_data[0].label == ".str0"
		assert ir_prog.string_data[0].value == "hello"

	def test_string_literal_codegen_asciz(self) -> None:
		src = """
		int main() {
			char* s = "world";
			return 0;
		}
		"""
		prog = Parser.from_source(src).parse()
		SemanticAnalyzer().analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		asm = CodeGenerator().generate(ir_prog)
		assert ".asciz" in asm
		assert ".rodata" in asm

	def test_multiple_string_literals(self) -> None:
		src = """
		int main() {
			char* a = "foo";
			char* b = "bar";
			return 0;
		}
		"""
		prog = Parser.from_source(src).parse()
		SemanticAnalyzer().analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		assert len(ir_prog.string_data) == 2
		assert ir_prog.string_data[0].label == ".str0"
		assert ir_prog.string_data[1].label == ".str1"
