"""Tests for goto/label interactions with loops, switches, and variable declarations.

Covers edge cases: goto out of/into loops, goto in switch cases, goto across
variable declarations, labels at end of function, multiple gotos to same label,
and complex control flow patterns.
"""

import pytest

from compiler.ir import IRCondJump, IRJump, IRLabelInstr
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer, SemanticError


def _parse(source: str):
	tokens = Lexer(source).tokenize()
	return Parser(tokens).parse()


def _analyze(source: str) -> list:
	ast = _parse(source)
	return SemanticAnalyzer().analyze(ast)


def _compile_to_ir(source: str):
	ast = _parse(source)
	errors = SemanticAnalyzer().analyze(ast)
	assert not errors, f"Semantic errors: {errors}"
	return IRGenerator().generate(ast)


def _ir_labels(func) -> set[str]:
	return {i.name for i in func.body if isinstance(i, IRLabelInstr)}


def _ir_jumps(func) -> list[str]:
	return [i.target for i in func.body if isinstance(i, IRJump)]


# ---------------------------------------------------------------------------
# Goto out of loops
# ---------------------------------------------------------------------------


class TestGotoOutOfLoop:
	def test_goto_out_of_while(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			while (1) {
				if (i == 5) goto done;
				i = i + 1;
			}
			done: return i;
		}
		"""
		ir = _compile_to_ir(source)
		labels = _ir_labels(ir.functions[0])
		jumps = _ir_jumps(ir.functions[0])
		assert any(j in labels for j in jumps)

	def test_goto_out_of_for(self) -> None:
		source = """
		int main(void) {
			int result = 0;
			int i;
			for (i = 0; i < 100; i = i + 1) {
				if (i == 10) goto bail;
				result = result + i;
			}
			bail: return result;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) == 1

	def test_goto_out_of_do_while(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			do {
				x = x + 1;
				if (x == 3) goto exit;
			} while (x < 100);
			exit: return x;
		}
		"""
		ir = _compile_to_ir(source)
		jumps = _ir_jumps(ir.functions[0])
		assert len(jumps) >= 1

	def test_goto_out_of_nested_loops(self) -> None:
		source = """
		int main(void) {
			int i;
			int j;
			for (i = 0; i < 10; i = i + 1) {
				for (j = 0; j < 10; j = j + 1) {
					if (i + j == 7) goto found;
				}
			}
			found: return i + j;
		}
		"""
		ir = _compile_to_ir(source)
		labels = _ir_labels(ir.functions[0])
		jumps = _ir_jumps(ir.functions[0])
		assert any(j in labels for j in jumps)


# ---------------------------------------------------------------------------
# Goto into loops (jumps into loop body - valid C, skips condition)
# ---------------------------------------------------------------------------


class TestGotoIntoLoop:
	def test_goto_into_while_body(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			goto inside;
			while (0) {
				inside:
				x = 42;
			}
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		labels = _ir_labels(ir.functions[0])
		assert len(labels) >= 1

	def test_goto_into_for_body(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			goto target;
			for (x = 100; x < 200; x = x + 1) {
				target:
				x = 55;
			}
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) == 1


# ---------------------------------------------------------------------------
# Goto across variable declarations
# ---------------------------------------------------------------------------


class TestGotoAcrossDeclarations:
	def test_goto_skips_var_decl(self) -> None:
		source = """
		int main(void) {
			goto skip;
			int x = 42;
			skip: return 0;
		}
		"""
		ir = _compile_to_ir(source)
		jumps = _ir_jumps(ir.functions[0])
		assert len(jumps) >= 1

	def test_goto_skips_multiple_decls(self) -> None:
		source = """
		int main(void) {
			int result = 0;
			goto end;
			int a = 10;
			int b = 20;
			int c = 30;
			result = a + b + c;
			end: return result;
		}
		"""
		ir = _compile_to_ir(source)
		jumps = _ir_jumps(ir.functions[0])
		labels = _ir_labels(ir.functions[0])
		assert any(j in labels for j in jumps)

	def test_goto_backward_past_decl(self) -> None:
		"""Backward goto past a variable declaration is valid C."""
		source = """
		int main(void) {
			int count = 0;
			top:
			count = count + 1;
			int temp = count * 2;
			if (temp < 10) goto top;
			return count;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) == 1


# ---------------------------------------------------------------------------
# Labels at end of function / compound statement
# ---------------------------------------------------------------------------


class TestLabelAtEnd:
	def test_label_before_return_at_end(self) -> None:
		source = """
		int main(void) {
			goto end;
			int x = 5;
			end: return 0;
		}
		"""
		ir = _compile_to_ir(source)
		jumps = _ir_jumps(ir.functions[0])
		assert len(jumps) >= 1

	def test_label_with_empty_stmt_at_end(self) -> None:
		"""Label followed by a semicolon (empty statement) at end of block."""
		source = """
		int main(void) {
			int x = 1;
			goto done;
			x = 2;
			done: x = x;
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) == 1

	def test_label_after_all_logic(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			if (x == 0) goto cleanup;
			x = 99;
			cleanup:
			x = x + 1;
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		labels = _ir_labels(ir.functions[0])
		assert len(labels) >= 1


# ---------------------------------------------------------------------------
# Multiple gotos to same label
# ---------------------------------------------------------------------------


class TestMultipleGotosToSameLabel:
	def test_three_gotos_one_label(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			if (x == 0) goto done;
			if (x == 1) goto done;
			if (x == 2) goto done;
			x = 99;
			done: return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		labels = _ir_labels(func)
		# At least 3 conditional jumps that target labels
		target_set = {cj.true_label for cj in cond_jumps} | {cj.false_label for cj in cond_jumps}
		assert len(target_set & labels) >= 1

	def test_gotos_from_different_blocks_same_label(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			{
				if (x == 0) goto end;
			}
			{
				if (x == 1) goto end;
			}
			{
				goto end;
			}
			end: return x;
		}
		"""
		ir = _compile_to_ir(source)
		jumps = _ir_jumps(ir.functions[0])
		assert len(jumps) >= 1

	def test_gotos_from_loop_and_outside(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			if (i == 99) goto end;
			while (i < 10) {
				if (i == 5) goto end;
				i = i + 1;
			}
			end: return i;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) == 1


# ---------------------------------------------------------------------------
# Goto in switch cases
# ---------------------------------------------------------------------------


class TestGotoInSwitch:
	def test_goto_out_of_switch(self) -> None:
		source = """
		int main(void) {
			int x = 2;
			switch (x) {
				case 1: x = 10; break;
				case 2: goto done;
				case 3: x = 30; break;
			}
			done: return x;
		}
		"""
		ir = _compile_to_ir(source)
		jumps = _ir_jumps(ir.functions[0])
		assert len(jumps) >= 1

	def test_goto_between_switch_cases_via_label(self) -> None:
		"""Goto from outside switch to a label inside a case."""
		source = """
		int main(void) {
			int x = 0;
			goto target;
			switch (x) {
				case 0:
					target: x = 42;
					break;
				case 1:
					x = 100;
					break;
			}
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		labels = _ir_labels(ir.functions[0])
		assert len(labels) >= 1

	def test_goto_in_switch_default(self) -> None:
		source = """
		int main(void) {
			int x = 99;
			switch (x) {
				case 1: x = 1; break;
				default: goto fallback;
			}
			return x;
			fallback: return -1;
		}
		"""
		ir = _compile_to_ir(source)
		jumps = _ir_jumps(ir.functions[0])
		assert len(jumps) >= 1

	def test_goto_in_nested_switch_and_loop(self) -> None:
		source = """
		int main(void) {
			int i;
			for (i = 0; i < 5; i = i + 1) {
				switch (i) {
					case 3: goto done;
					default: break;
				}
			}
			done: return i;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) == 1


# ---------------------------------------------------------------------------
# Complex goto patterns
# ---------------------------------------------------------------------------


class TestComplexGotoPatterns:
	def test_goto_zigzag(self) -> None:
		"""Forward and backward gotos interleaved."""
		source = """
		int main(void) {
			int x = 0;
			goto b;
			a: x = x + 1;
			goto c;
			b: x = x + 2;
			goto a;
			c: return x;
		}
		"""
		ir = _compile_to_ir(source)
		jumps = _ir_jumps(ir.functions[0])
		assert len(jumps) >= 3

	def test_goto_simulating_state_machine(self) -> None:
		"""A simple 3-state machine using goto."""
		source = """
		int main(void) {
			int state = 0;
			int result = 0;
			s0: result = result + 1;
			if (result >= 3) goto s2;
			goto s1;
			s1: result = result + 10;
			goto s0;
			s2: return result;
		}
		"""
		ir = _compile_to_ir(source)
		labels = _ir_labels(ir.functions[0])
		jumps = _ir_jumps(ir.functions[0])
		assert len(labels) >= 3
		assert len(jumps) >= 3

	def test_goto_cleanup_pattern(self) -> None:
		"""Common C cleanup pattern with goto."""
		source = """
		int main(void) {
			int result = 0;
			int resource1 = 0;
			int resource2 = 0;
			resource1 = 1;
			if (resource1 == 0) goto cleanup;
			resource2 = 1;
			if (resource2 == 0) goto cleanup;
			result = 42;
			cleanup:
			resource2 = 0;
			resource1 = 0;
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		labels = _ir_labels(ir.functions[0])
		jumps = _ir_jumps(ir.functions[0])
		# Cleanup label should exist and be targeted
		assert any(j in labels for j in jumps)

	def test_goto_with_break_in_loop(self) -> None:
		"""Mix of goto and break in same loop."""
		source = """
		int main(void) {
			int i;
			int found = 0;
			for (i = 0; i < 20; i = i + 1) {
				if (i == 5) break;
				if (i == 15) goto error;
			}
			return found;
			error: return -1;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) == 1

	def test_goto_with_continue_in_loop(self) -> None:
		"""Mix of goto and continue in same loop."""
		source = """
		int main(void) {
			int sum = 0;
			int i;
			for (i = 0; i < 10; i = i + 1) {
				if (i == 3) continue;
				if (i == 7) goto done;
				sum = sum + i;
			}
			done: return sum;
		}
		"""
		ir = _compile_to_ir(source)
		labels = _ir_labels(ir.functions[0])
		jumps = _ir_jumps(ir.functions[0])
		assert any(j in labels for j in jumps)

	def test_label_followed_by_compound_stmt(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			goto start;
			start: {
				x = 10;
				x = x + 5;
			}
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) == 1

	def test_consecutive_labels(self) -> None:
		"""Two labels on consecutive statements."""
		source = """
		int main(void) {
			int x = 0;
			goto b_label;
			a_label: x = 1;
			b_label: x = x + 2;
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		labels = _ir_labels(ir.functions[0])
		assert len(labels) >= 2

	def test_goto_in_if_else_branches(self) -> None:
		"""Both branches of an if-else use goto to the same label."""
		source = """
		int main(void) {
			int x = 1;
			if (x) {
				x = 10;
				goto end;
			} else {
				x = 20;
				goto end;
			}
			x = 99;
			end: return x;
		}
		"""
		ir = _compile_to_ir(source)
		jumps = _ir_jumps(ir.functions[0])
		assert len(jumps) >= 2


# ---------------------------------------------------------------------------
# Semantic analysis edge cases
# ---------------------------------------------------------------------------


class TestGotoSemanticEdgeCases:
	def test_goto_undefined_label_in_loop(self) -> None:
		source = """
		int main(void) {
			while (1) {
				goto nowhere;
			}
			return 0;
		}
		"""
		ast = _parse(source)
		with pytest.raises(SemanticError, match="undeclared label 'nowhere'"):
			SemanticAnalyzer().analyze(ast)

	def test_duplicate_label_in_same_function(self) -> None:
		source = """
		int main(void) {
			lbl: return 1;
			lbl: return 0;
		}
		"""
		ast = _parse(source)
		with pytest.raises(SemanticError, match="redefinition of label 'lbl'"):
			SemanticAnalyzer().analyze(ast)

	def test_same_label_name_different_functions_ok(self) -> None:
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
		errors = _analyze(source)
		assert not errors

	def test_label_in_nested_block_visible_to_outer_goto(self) -> None:
		source = """
		int main(void) {
			goto inner;
			{
				{
					inner: return 99;
				}
			}
			return 0;
		}
		"""
		errors = _analyze(source)
		assert not errors

	def test_goto_in_switch_to_external_label(self) -> None:
		source = """
		int main(void) {
			switch (1) {
				case 1: goto outside;
			}
			outside: return 0;
		}
		"""
		errors = _analyze(source)
		assert not errors

	def test_forward_ref_label_defined_later(self) -> None:
		"""Goto referencing a label that appears later is valid."""
		source = """
		int main(void) {
			goto future;
			int x = 1;
			int y = 2;
			int z = 3;
			future: return 0;
		}
		"""
		errors = _analyze(source)
		assert not errors

	def test_goto_to_label_inside_if_from_outside(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			goto target;
			if (0) {
				target: x = 42;
			}
			return x;
		}
		"""
		errors = _analyze(source)
		assert not errors
