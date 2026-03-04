"""Tests for goto/label support across parser, semantic analysis, and IR generation."""

import pytest

from compiler.ast_nodes import (
	FunctionDecl,
	GotoStmt,
	LabelStmt,
	Program,
	ReturnStmt,
	SourceLocation,
	TypeSpec,
)
from compiler.codegen import CodeGenerator
from compiler.ir import IRJump, IRLabelInstr
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer, SemanticError


def loc() -> SourceLocation:
	return SourceLocation(line=1, col=1)


def int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def parse(source: str) -> Program:
	tokens = Lexer(source).tokenize()
	return Parser(tokens).parse()


def compile_to_ir(source: str):
	ast = parse(source)
	SemanticAnalyzer().analyze(ast)
	return IRGenerator().generate(ast)


def compile_source(source: str) -> str:
	ir = compile_to_ir(source)
	return CodeGenerator().generate(ir)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParserGoto:
	def test_parse_goto(self) -> None:
		ast = parse("int main() { goto end; end: return 0; }")
		func = ast.declarations[0]
		assert isinstance(func, FunctionDecl)
		stmts = func.body.statements
		assert isinstance(stmts[0], GotoStmt)
		assert stmts[0].label == "end"

	def test_parse_label(self) -> None:
		ast = parse("int main() { start: return 0; }")
		func = ast.declarations[0]
		stmts = func.body.statements
		assert isinstance(stmts[0], LabelStmt)
		assert stmts[0].label == "start"
		assert isinstance(stmts[0].statement, ReturnStmt)

	def test_parse_forward_goto(self) -> None:
		source = """
		int main() {
			goto skip;
			int x = 1;
			skip: return 0;
		}
		"""
		ast = parse(source)
		func = ast.declarations[0]
		stmts = func.body.statements
		assert isinstance(stmts[0], GotoStmt)
		assert stmts[0].label == "skip"
		assert isinstance(stmts[2], LabelStmt)
		assert stmts[2].label == "skip"

	def test_parse_label_before_loop(self) -> None:
		source = """
		int main() {
			int i = 0;
			loop: while (i < 10) { i = i + 1; }
			return i;
		}
		"""
		ast = parse(source)
		func = ast.declarations[0]
		stmts = func.body.statements
		assert isinstance(stmts[1], LabelStmt)
		assert stmts[1].label == "loop"

	def test_parse_multiple_labels(self) -> None:
		source = """
		int main() {
			goto b;
			a: return 1;
			b: return 0;
		}
		"""
		ast = parse(source)
		func = ast.declarations[0]
		stmts = func.body.statements
		assert isinstance(stmts[0], GotoStmt)
		assert isinstance(stmts[1], LabelStmt)
		assert stmts[1].label == "a"
		assert isinstance(stmts[2], LabelStmt)
		assert stmts[2].label == "b"


# ---------------------------------------------------------------------------
# Semantic analysis tests
# ---------------------------------------------------------------------------


class TestSemanticGoto:
	def test_valid_goto(self) -> None:
		source = """
		int main() {
			goto end;
			end: return 0;
		}
		"""
		ast = parse(source)
		errors = SemanticAnalyzer().analyze(ast)
		assert not errors

	def test_forward_goto(self) -> None:
		source = """
		int main() {
			goto skip;
			int x = 1;
			skip: return 0;
		}
		"""
		ast = parse(source)
		errors = SemanticAnalyzer().analyze(ast)
		assert not errors

	def test_goto_undefined_label(self) -> None:
		source = """
		int main() {
			goto nowhere;
			return 0;
		}
		"""
		ast = parse(source)
		with pytest.raises(SemanticError, match="undeclared label 'nowhere'"):
			SemanticAnalyzer().analyze(ast)

	def test_duplicate_label(self) -> None:
		source = """
		int main() {
			done: return 1;
			done: return 0;
		}
		"""
		ast = parse(source)
		with pytest.raises(SemanticError, match="redefinition of label 'done'"):
			SemanticAnalyzer().analyze(ast)

	def test_label_scope_per_function(self) -> None:
		source = """
		int foo() {
			end: return 1;
		}
		int bar() {
			end: return 2;
		}
		"""
		ast = parse(source)
		errors = SemanticAnalyzer().analyze(ast)
		assert not errors

	def test_goto_to_other_function_label(self) -> None:
		source = """
		int foo() {
			target: return 1;
		}
		int bar() {
			goto target;
			return 0;
		}
		"""
		ast = parse(source)
		with pytest.raises(SemanticError, match="undeclared label 'target'"):
			SemanticAnalyzer().analyze(ast)

	def test_label_in_nested_block(self) -> None:
		source = """
		int main() {
			if (1) {
				inner: return 1;
			}
			goto inner;
			return 0;
		}
		"""
		ast = parse(source)
		errors = SemanticAnalyzer().analyze(ast)
		assert not errors

	def test_goto_across_blocks(self) -> None:
		source = """
		int main() {
			int x = 0;
			if (x) {
				goto done;
			}
			x = 1;
			done: return x;
		}
		"""
		ast = parse(source)
		errors = SemanticAnalyzer().analyze(ast)
		assert not errors


# ---------------------------------------------------------------------------
# IR generation tests
# ---------------------------------------------------------------------------


class TestIRGenGoto:
	def test_goto_generates_jump(self) -> None:
		source = """
		int main() {
			goto end;
			end: return 0;
		}
		"""
		ir = compile_to_ir(source)
		func = ir.functions[0]
		jumps = [i for i in func.body if isinstance(i, IRJump)]
		labels = [i for i in func.body if isinstance(i, IRLabelInstr)]
		assert len(jumps) >= 1
		assert len(labels) >= 1
		# The goto should jump to the label
		label_names = {lbl.name for lbl in labels}
		assert any(j.target in label_names for j in jumps)

	def test_forward_goto_ir(self) -> None:
		source = """
		int main() {
			goto skip;
			int x = 1;
			skip: return 0;
		}
		"""
		ir = compile_to_ir(source)
		func = ir.functions[0]
		jumps = [i for i in func.body if isinstance(i, IRJump)]
		labels = [i for i in func.body if isinstance(i, IRLabelInstr)]
		# Forward goto: jump target should be in the labels
		label_names = {lbl.name for lbl in labels}
		assert any(j.target in label_names for j in jumps)

	def test_multiple_gotos_same_label(self) -> None:
		source = """
		int main() {
			int x = 0;
			if (x) goto end;
			x = 1;
			goto end;
			end: return x;
		}
		"""
		ir = compile_to_ir(source)
		func = ir.functions[0]
		label_instrs = [i for i in func.body if isinstance(i, IRLabelInstr)]
		label_names = [lbl.name for lbl in label_instrs]
		# Should use the same IR label for both gotos
		jumps = [i for i in func.body if isinstance(i, IRJump)]
		goto_targets = {j.target for j in jumps}
		# At least one IR label should be a target of a goto
		assert goto_targets & set(label_names)


# ---------------------------------------------------------------------------
# Full pipeline (codegen) tests
# ---------------------------------------------------------------------------


class TestCodegenGoto:
	def test_simple_goto_codegen(self) -> None:
		source = """
		int main() {
			goto end;
			end: return 42;
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm
		assert "jmp" in asm
		assert "ret" in asm

	def test_goto_skip_code(self) -> None:
		source = """
		int main() {
			int x = 10;
			goto done;
			x = 99;
			done: return x;
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm
		assert "jmp" in asm

	def test_label_before_loop(self) -> None:
		source = """
		int main() {
			int i = 0;
			start:
			if (i >= 5) goto end;
			i = i + 1;
			goto start;
			end: return i;
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm
		# Should have multiple jumps for the goto statements
		assert asm.count("jmp") >= 2

	def test_goto_in_nested_if(self) -> None:
		source = """
		int main() {
			int x = 1;
			if (x) {
				goto done;
			}
			x = 0;
			done: return x;
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm
		assert "jmp" in asm
