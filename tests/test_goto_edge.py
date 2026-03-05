"""Edge-case tests for goto/label: forward, backward, nested blocks, cross-scope, multiple labels."""

import pytest

from compiler.codegen import CodeGenerator
from compiler.ir import IRJump, IRLabelInstr
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer, SemanticError


def _parse(source: str):
	tokens = Lexer(source).tokenize()
	return Parser(tokens).parse()


def _compile_to_ir(source: str):
	ast = _parse(source)
	errors = SemanticAnalyzer().analyze(ast)
	assert not errors, f"Semantic errors: {errors}"
	return IRGenerator().generate(ast)


def _compile_to_asm(source: str) -> str:
	ir = _compile_to_ir(source)
	return CodeGenerator().generate(ir)


# ---------------------------------------------------------------------------
# Forward goto
# ---------------------------------------------------------------------------


class TestForwardGoto:
	def test_forward_goto_skips_assignment(self) -> None:
		source = """
		int main(void) {
			int x = 10;
			goto skip;
			x = 99;
			skip: return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm
		assert "main:" in asm

	def test_forward_goto_to_end(self) -> None:
		source = """
		int main(void) {
			goto done;
			done: return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_forward_goto_skips_multiple_statements(self) -> None:
		source = """
		int main(void) {
			int a = 1;
			int b = 2;
			goto end;
			a = 100;
			b = 200;
			int c = 300;
			end: return a + b;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm
		assert "jmp" in asm

	def test_forward_goto_ir_has_jump(self) -> None:
		source = """
		int main(void) {
			goto target;
			int x = 5;
			target: return 0;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		jumps = [i for i in func.body if isinstance(i, IRJump)]
		labels = [i for i in func.body if isinstance(i, IRLabelInstr)]
		label_names = {lbl.name for lbl in labels}
		assert any(j.target in label_names for j in jumps)


# ---------------------------------------------------------------------------
# Backward goto (loop-like patterns)
# ---------------------------------------------------------------------------


class TestBackwardGoto:
	def test_backward_goto_loop(self) -> None:
		"""Use goto to create a manual loop that counts up."""
		source = """
		int main(void) {
			int i = 0;
			top:
			if (i >= 5) goto done;
			i = i + 1;
			goto top;
			done: return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert asm.count("jmp") >= 2

	def test_backward_goto_ir_labels(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			again:
			x = x + 1;
			if (x < 3) goto again;
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		labels = [i for i in func.body if isinstance(i, IRLabelInstr)]
		assert len(labels) >= 1

	def test_backward_goto_with_accumulator(self) -> None:
		source = """
		int main(void) {
			int sum = 0;
			int i = 1;
			loop:
			sum = sum + i;
			i = i + 1;
			if (i <= 10) goto loop;
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Goto in nested blocks
# ---------------------------------------------------------------------------


class TestGotoNestedBlocks:
	def test_goto_inside_if_block(self) -> None:
		source = """
		int main(void) {
			int x = 1;
			if (x) {
				goto done;
			}
			x = 0;
			done: return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_goto_inside_else_block(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			if (x) {
				x = 1;
			} else {
				goto skip;
			}
			x = 99;
			skip: return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_goto_inside_while_body(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			while (i < 10) {
				if (i == 5) goto out;
				i = i + 1;
			}
			out: return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_goto_inside_for_body(self) -> None:
		source = """
		int main(void) {
			int i;
			for (i = 0; i < 10; i = i + 1) {
				if (i == 3) goto found;
			}
			return -1;
			found: return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_label_inside_nested_if(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			goto inner;
			if (1) {
				inner: x = 42;
			}
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Goto across scopes
# ---------------------------------------------------------------------------


class TestGotoAcrossScopes:
	def test_goto_from_inner_to_outer_label(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			{
				x = 1;
				goto done;
			}
			x = 99;
			done: return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_goto_from_outer_to_inner_label(self) -> None:
		"""Goto into a block (valid in C, label has function scope)."""
		source = """
		int main(void) {
			goto inside;
			{
				inside: return 42;
			}
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_goto_between_sibling_blocks(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			{
				x = 1;
				goto target;
			}
			{
				target: x = x + 10;
			}
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_label_scope_is_function_wide(self) -> None:
		"""Labels defined in nested blocks are visible to gotos at function scope."""
		source = """
		int main(void) {
			goto deep;
			if (1) {
				if (1) {
					deep: return 99;
				}
			}
			return 0;
		}
		"""
		ast = _parse(source)
		errors = SemanticAnalyzer().analyze(ast)
		assert not errors


# ---------------------------------------------------------------------------
# Multiple labels
# ---------------------------------------------------------------------------


class TestMultipleLabels:
	def test_multiple_labels_same_function(self) -> None:
		source = """
		int main(void) {
			goto second;
			first: return 1;
			second: return 2;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_multiple_gotos_same_label(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			if (x == 0) goto end;
			if (x == 1) goto end;
			x = 99;
			end: return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		jumps = [i for i in func.body if isinstance(i, IRJump)]
		assert len(jumps) >= 2

	def test_chain_of_gotos(self) -> None:
		source = """
		int main(void) {
			goto a;
			b: return 2;
			a: goto b;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert asm.count("jmp") >= 2

	def test_labels_in_separate_functions(self) -> None:
		source = """
		int foo(void) {
			done: return 1;
		}
		int bar(void) {
			done: return 2;
		}
		int main(void) {
			return foo() + bar();
		}
		"""
		asm = _compile_to_asm(source)
		assert "foo:" in asm
		assert "bar:" in asm

	def test_five_labels(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			goto l3;
			l1: x = x + 1;
			goto l4;
			l2: x = x + 2;
			goto l5;
			l3: x = x + 3;
			goto l1;
			l4: x = x + 4;
			goto l2;
			l5: return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert asm.count("jmp") >= 5


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestGotoErrors:
	def test_goto_undefined_label(self) -> None:
		source = """
		int main(void) {
			goto missing;
			return 0;
		}
		"""
		ast = _parse(source)
		with pytest.raises(SemanticError, match="undeclared label 'missing'"):
			SemanticAnalyzer().analyze(ast)

	def test_duplicate_label_error(self) -> None:
		source = """
		int main(void) {
			lbl: return 1;
			lbl: return 2;
		}
		"""
		ast = _parse(source)
		with pytest.raises(SemanticError, match="redefinition of label 'lbl'"):
			SemanticAnalyzer().analyze(ast)

	def test_goto_label_in_other_function(self) -> None:
		source = """
		int foo(void) {
			target: return 1;
		}
		int main(void) {
			goto target;
			return 0;
		}
		"""
		ast = _parse(source)
		with pytest.raises(SemanticError, match="undeclared label 'target'"):
			SemanticAnalyzer().analyze(ast)
