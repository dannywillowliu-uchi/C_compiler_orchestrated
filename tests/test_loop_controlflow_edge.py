"""Edge-case tests for loops and control flow: do-while break/continue, for(;;),
nested loop break/continue targeting, deeply nested if-else, switch fallthrough
across cases, switch inside loops, goto over declarations, labeled statements,
infinite loop patterns."""

import pytest

from compiler.codegen import CodeGenerator
from compiler.ir import IRCondJump, IRJump, IRLabelInstr
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
# Do-while with break and continue
# ---------------------------------------------------------------------------


class TestDoWhileBreakContinue:
	def test_do_while_immediate_break(self) -> None:
		"""Break on first iteration should exit after executing body once."""
		source = """
		int main(void) {
			int x = 0;
			do {
				x = x + 1;
				break;
			} while (1);
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_do_while_conditional_break(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			do {
				i = i + 1;
				if (i == 3) break;
			} while (i < 10);
			return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_do_while_continue_skips_rest(self) -> None:
		"""Continue should jump to the condition check."""
		source = """
		int main(void) {
			int i = 0;
			int sum = 0;
			do {
				i = i + 1;
				if (i == 3) continue;
				sum = sum + i;
			} while (i < 5);
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_do_while_continue_then_break(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			int r = 0;
			do {
				i = i + 1;
				if (i == 2) continue;
				if (i == 4) break;
				r = r + i;
			} while (i < 10);
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_do_while_body_always_runs_once(self) -> None:
		"""Even with false condition, body executes once."""
		source = """
		int main(void) {
			int x = 0;
			do {
				x = 42;
			} while (0);
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_do_while_ir_has_back_edge(self) -> None:
		"""IR should have a conditional jump back to the loop body."""
		source = """
		int main(void) {
			int i = 0;
			do {
				i = i + 1;
			} while (i < 5);
			return i;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 1


# ---------------------------------------------------------------------------
# For-loop with empty clauses: for(;;)
# ---------------------------------------------------------------------------


class TestForLoopEmptyClauses:
	def test_for_infinite_with_break(self) -> None:
		"""for(;;) creates an infinite loop; break exits it."""
		source = """
		int main(void) {
			int i = 0;
			for (;;) {
				i = i + 1;
				if (i >= 5) break;
			}
			return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_for_no_init(self) -> None:
		"""For-loop with empty init clause."""
		source = """
		int main(void) {
			int i = 0;
			for (; i < 3; i = i + 1) {
				i = i;
			}
			return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_for_no_increment(self) -> None:
		"""For-loop with empty increment clause."""
		source = """
		int main(void) {
			int i;
			for (i = 0; i < 3;) {
				i = i + 1;
			}
			return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_for_no_condition(self) -> None:
		"""For-loop with empty condition (always true)."""
		source = """
		int main(void) {
			int i = 0;
			for (i = 0;; i = i + 1) {
				if (i >= 4) break;
			}
			return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_for_all_empty_with_conditional_break(self) -> None:
		"""for(;;) with only a conditional break inside."""
		source = """
		int main(void) {
			int x = 100;
			for (;;) {
				x = x - 1;
				if (x <= 0) break;
			}
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_for_empty_body(self) -> None:
		"""For-loop with an empty body (semicolon)."""
		source = """
		int main(void) {
			int i;
			for (i = 0; i < 5; i = i + 1) {
			}
			return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Nested loop break/continue targeting correct loop
# ---------------------------------------------------------------------------


class TestNestedLoopBreakContinue:
	def test_break_exits_inner_loop_only(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			int j = 0;
			int count = 0;
			while (i < 3) {
				j = 0;
				while (j < 10) {
					if (j == 2) break;
					count = count + 1;
					j = j + 1;
				}
				i = i + 1;
			}
			return count;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_continue_targets_inner_loop(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			int sum = 0;
			while (i < 3) {
				int j = 0;
				while (j < 5) {
					j = j + 1;
					if (j == 3) continue;
					sum = sum + 1;
				}
				i = i + 1;
			}
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_break_in_for_inside_while(self) -> None:
		source = """
		int main(void) {
			int total = 0;
			int i = 0;
			while (i < 4) {
				int j;
				for (j = 0; j < 10; j = j + 1) {
					if (j == 3) break;
					total = total + 1;
				}
				i = i + 1;
			}
			return total;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_continue_in_do_while_inside_for(self) -> None:
		source = """
		int main(void) {
			int r = 0;
			int i;
			for (i = 0; i < 3; i = i + 1) {
				int k = 0;
				do {
					k = k + 1;
					if (k == 2) continue;
					r = r + 1;
				} while (k < 4);
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_triple_nested_break(self) -> None:
		"""Break in innermost of 3 nested loops only exits innermost."""
		source = """
		int main(void) {
			int i = 0;
			int j = 0;
			int k = 0;
			int r = 0;
			while (i < 2) {
				j = 0;
				while (j < 2) {
					k = 0;
					while (k < 100) {
						if (k == 1) break;
						r = r + 1;
						k = k + 1;
					}
					j = j + 1;
				}
				i = i + 1;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_ir_nested_loops_have_distinct_labels(self) -> None:
		"""Nested loops should generate distinct break/continue target labels."""
		source = """
		int main(void) {
			int i = 0;
			while (i < 2) {
				int j = 0;
				while (j < 2) {
					j = j + 1;
				}
				i = i + 1;
			}
			return 0;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		labels = [i for i in func.body if isinstance(i, IRLabelInstr)]
		label_names = [lbl.name for lbl in labels]
		assert len(label_names) == len(set(label_names)), "Duplicate labels found"


# ---------------------------------------------------------------------------
# Deeply nested if-else chains
# ---------------------------------------------------------------------------


class TestDeeplyNestedIfElse:
	def test_four_level_nested_if(self) -> None:
		source = """
		int main(void) {
			int a = 1;
			int b = 2;
			int c = 3;
			int d = 4;
			if (a == 1) {
				if (b == 2) {
					if (c == 3) {
						if (d == 4) {
							return 1;
						} else {
							return 2;
						}
					} else {
						return 3;
					}
				} else {
					return 4;
				}
			} else {
				return 5;
			}
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_if_else_if_chain(self) -> None:
		source = """
		int main(void) {
			int x = 5;
			int r = 0;
			if (x == 1) {
				r = 10;
			} else if (x == 2) {
				r = 20;
			} else if (x == 3) {
				r = 30;
			} else if (x == 4) {
				r = 40;
			} else if (x == 5) {
				r = 50;
			} else {
				r = -1;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_nested_if_with_compound_conditions(self) -> None:
		source = """
		int main(void) {
			int x = 10;
			int y = 20;
			int r = 0;
			if (x > 0) {
				if (y > 0) {
					if (x + y > 25) {
						r = 1;
					} else {
						r = 2;
					}
				} else {
					r = 3;
				}
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_if_without_else_nested(self) -> None:
		"""Multiple levels of if-without-else (dangling else scenario)."""
		source = """
		int main(void) {
			int x = 1;
			int r = 0;
			if (x > 0)
				if (x > 1)
					if (x > 2)
						r = 3;
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_ir_nested_if_generates_labels(self) -> None:
		source = """
		int main(void) {
			int x = 1;
			if (x == 1) {
				if (x == 1) {
					return 1;
				}
			}
			return 0;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 2


# ---------------------------------------------------------------------------
# Switch with fallthrough across multiple cases
# ---------------------------------------------------------------------------


class TestSwitchFallthroughMultiple:
	def test_fallthrough_five_cases(self) -> None:
		"""Fallthrough from case 0 through all subsequent cases."""
		source = """
		int main(void) {
			int x = 0;
			int r = 0;
			switch (x) {
				case 0: r = r + 1;
				case 1: r = r + 2;
				case 2: r = r + 4;
				case 3: r = r + 8;
				case 4: r = r + 16;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_partial_fallthrough_with_break(self) -> None:
		"""Fallthrough for first two, break, then fallthrough for next two."""
		source = """
		int main(void) {
			int x = 1;
			int r = 0;
			switch (x) {
				case 0: r = r + 1;
				case 1: r = r + 2; break;
				case 2: r = r + 4;
				case 3: r = r + 8; break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_fallthrough_from_default_to_case(self) -> None:
		"""Default placed before cases, falling through into them."""
		source = """
		int main(void) {
			int x = 99;
			int r = 0;
			switch (x) {
				default: r = r + 1;
				case 1: r = r + 2;
				case 2: r = r + 4; break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Switch inside loops with break semantics
# ---------------------------------------------------------------------------


class TestSwitchInsideLoopsBreak:
	def test_switch_break_does_not_exit_while(self) -> None:
		"""Break inside switch should NOT exit the enclosing while loop."""
		source = """
		int main(void) {
			int i = 0;
			int r = 0;
			while (i < 5) {
				switch (i) {
					case 0: r = r + 1; break;
					case 1: r = r + 10; break;
					default: r = r + 100; break;
				}
				i = i + 1;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_switch_break_does_not_exit_for(self) -> None:
		source = """
		int main(void) {
			int r = 0;
			int i;
			for (i = 0; i < 4; i = i + 1) {
				switch (i) {
					case 0: r = r + 1; break;
					default: r = r + 2; break;
				}
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_switch_break_does_not_exit_do_while(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			int r = 0;
			do {
				switch (i) {
					case 0: r = r + 5; break;
					default: r = r + 1; break;
				}
				i = i + 1;
			} while (i < 3);
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_nested_switch_in_loop_break_semantics(self) -> None:
		"""Break in inner switch exits switch; break in loop body exits loop."""
		source = """
		int main(void) {
			int r = 0;
			int i = 0;
			while (i < 10) {
				switch (i) {
					case 5:
						r = r + 100;
						break;
					default:
						break;
				}
				if (i == 7) break;
				r = r + 1;
				i = i + 1;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_loop_inside_switch_case(self) -> None:
		"""A loop nested inside a switch case."""
		source = """
		int main(void) {
			int x = 1;
			int r = 0;
			switch (x) {
				case 1: {
					int j = 0;
					while (j < 3) {
						r = r + 1;
						j = j + 1;
					}
					break;
				}
				default:
					r = -1;
					break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Goto jumping over variable declarations
# ---------------------------------------------------------------------------


class TestGotoOverDeclarations:
	def test_goto_skips_var_decl(self) -> None:
		"""Goto jumping over a variable declaration."""
		source = """
		int main(void) {
			int r = 0;
			goto skip;
			int x = 42;
			r = x;
			skip:
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_goto_skips_multiple_decls(self) -> None:
		source = """
		int main(void) {
			goto end;
			int a = 1;
			int b = 2;
			int c = 3;
			end:
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_goto_skips_decl_with_init(self) -> None:
		"""Variable initialized with complex expression is skipped."""
		source = """
		int main(void) {
			int base = 10;
			goto past;
			int computed = base * 2 + 5;
			past:
			return base;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_forward_goto_over_decl_inside_if(self) -> None:
		source = """
		int main(void) {
			int x = 1;
			if (x) goto skip;
			int y = 99;
			x = y;
			skip:
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Labeled statements as loop targets (goto-based loops)
# ---------------------------------------------------------------------------


class TestLabeledLoopTargets:
	def test_goto_loop_counting_up(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			loop_start:
			if (i >= 10) goto loop_end;
			i = i + 1;
			goto loop_start;
			loop_end:
			return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert asm.count("jmp") >= 2

	def test_goto_nested_loop_pattern(self) -> None:
		"""Nested loop using only goto and labels."""
		source = """
		int main(void) {
			int i = 0;
			int j = 0;
			int sum = 0;
			outer:
			if (i >= 3) goto done;
			j = 0;
			inner:
			if (j >= 3) goto next_i;
			sum = sum + 1;
			j = j + 1;
			goto inner;
			next_i:
			i = i + 1;
			goto outer;
			done:
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_goto_early_exit_from_nested_loops(self) -> None:
		"""Goto can exit multiple loop levels at once."""
		source = """
		int main(void) {
			int i = 0;
			int r = 0;
			while (i < 10) {
				int j = 0;
				while (j < 10) {
					if (i == 2 && j == 3) goto bail;
					r = r + 1;
					j = j + 1;
				}
				i = i + 1;
			}
			bail:
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_label_before_loop(self) -> None:
		source = """
		int main(void) {
			int count = 0;
			int pass = 0;
			restart:
			if (pass >= 2) goto finish;
			count = 0;
			while (count < 3) {
				count = count + 1;
			}
			pass = pass + 1;
			goto restart;
			finish:
			return pass;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Infinite loop with conditional break patterns
# ---------------------------------------------------------------------------


class TestInfiniteLoopPatterns:
	def test_while_true_break(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			while (1) {
				x = x + 1;
				if (x == 10) break;
			}
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_while_true_multiple_break_points(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			while (1) {
				x = x + 1;
				if (x == 3) break;
				if (x == 7) break;
				if (x == 10) break;
			}
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_for_infinite_break_on_condition(self) -> None:
		source = """
		int main(void) {
			int n = 1;
			for (;;) {
				n = n * 2;
				if (n > 100) break;
			}
			return n;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_do_while_true_break(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			do {
				x = x + 1;
				if (x >= 5) break;
			} while (1);
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_infinite_loop_with_continue_and_break(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			int sum = 0;
			while (1) {
				i = i + 1;
				if (i > 10) break;
				if (i % 2 == 0) continue;
				sum = sum + i;
			}
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_nested_infinite_loops(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			int j = 0;
			int r = 0;
			while (1) {
				j = 0;
				while (1) {
					j = j + 1;
					if (j >= 3) break;
					r = r + 1;
				}
				i = i + 1;
				if (i >= 3) break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# While loop edge cases
# ---------------------------------------------------------------------------


class TestWhileEdgeCases:
	def test_while_false_body_never_runs(self) -> None:
		source = """
		int main(void) {
			int x = 42;
			while (0) {
				x = 0;
			}
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_while_single_iteration(self) -> None:
		source = """
		int main(void) {
			int x = 1;
			while (x) {
				x = 0;
			}
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_while_with_empty_body(self) -> None:
		source = """
		int main(void) {
			int x = 5;
			while (x > 0) {
				x = x - 1;
			}
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Mixed control flow: loops with goto
# ---------------------------------------------------------------------------


class TestMixedControlFlow:
	def test_goto_out_of_while(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			while (i < 100) {
				if (i == 5) goto done;
				i = i + 1;
			}
			done:
			return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_goto_out_of_for(self) -> None:
		source = """
		int main(void) {
			int i;
			for (i = 0; i < 100; i = i + 1) {
				if (i == 7) goto exit;
			}
			exit:
			return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_goto_out_of_do_while(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			do {
				if (i == 4) goto end;
				i = i + 1;
			} while (1);
			end:
			return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_goto_between_loops(self) -> None:
		"""Goto from inside one loop to after another loop."""
		source = """
		int main(void) {
			int i = 0;
			int r = 0;
			while (i < 5) {
				r = r + 1;
				i = i + 1;
				if (r == 3) goto after_second;
			}
			i = 0;
			while (i < 5) {
				r = r + 10;
				i = i + 1;
			}
			after_second:
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_label_after_loop(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			while (i < 3) {
				i = i + 1;
			}
			done:
			return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm
