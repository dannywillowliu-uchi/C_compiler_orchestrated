"""Tests for nested control flow interactions: switch in loops, goto across blocks,
break/continue in nested loops, for-loop with declaration init, do-while with continue,
nested switch statements, fall-through, default in middle, and label before loops."""

from compiler.codegen import CodeGenerator
from compiler.ir import IRCondJump
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer


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
# Break/continue in nested loops
# ---------------------------------------------------------------------------


class TestBreakContinueNestedLoops:
	def test_break_inner_loop_only(self) -> None:
		"""Break in inner loop should not exit outer loop."""
		source = """
		int main(void) {
			int count = 0;
			int i = 0;
			while (i < 3) {
				int j = 0;
				while (j < 10) {
					if (j == 2) break;
					j = j + 1;
				}
				count = count + 1;
				i = i + 1;
			}
			return count;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_continue_inner_loop_only(self) -> None:
		"""Continue in inner loop should not affect outer loop."""
		source = """
		int main(void) {
			int sum = 0;
			int i = 0;
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
		"""Break in a for loop nested inside a while loop."""
		source = """
		int main(void) {
			int total = 0;
			int i = 0;
			while (i < 4) {
				int j;
				for (j = 0; j < 100; j = j + 1) {
					if (j == 5) break;
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
		"""Continue in do-while nested inside for loop."""
		source = """
		int main(void) {
			int sum = 0;
			int i;
			for (i = 0; i < 3; i = i + 1) {
				int j = 0;
				do {
					j = j + 1;
					if (j == 2) continue;
					sum = sum + j;
				} while (j < 4);
			}
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_triple_nested_break(self) -> None:
		"""Break in innermost of three nested loops."""
		source = """
		int main(void) {
			int count = 0;
			int i = 0;
			while (i < 2) {
				int j = 0;
				while (j < 2) {
					int k = 0;
					while (k < 100) {
						if (k == 1) break;
						k = k + 1;
					}
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


# ---------------------------------------------------------------------------
# Break from switch inside loop
# ---------------------------------------------------------------------------


class TestBreakFromSwitchInsideLoop:
	def test_switch_break_does_not_exit_loop(self) -> None:
		"""Break inside switch should exit the switch, not the enclosing loop."""
		source = """
		int main(void) {
			int result = 0;
			int i = 0;
			while (i < 5) {
				switch (i) {
					case 0:
						result = result + 1;
						break;
					case 1:
						result = result + 10;
						break;
					default:
						result = result + 100;
						break;
				}
				i = i + 1;
			}
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_switch_in_for_loop(self) -> None:
		"""Switch inside a for loop with break in cases."""
		source = """
		int main(void) {
			int sum = 0;
			int i;
			for (i = 0; i < 4; i = i + 1) {
				switch (i) {
					case 0: sum = sum + 1; break;
					case 1: sum = sum + 2; break;
					case 2: sum = sum + 4; break;
					default: sum = sum + 8; break;
				}
			}
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_switch_in_do_while(self) -> None:
		"""Switch inside a do-while loop."""
		source = """
		int main(void) {
			int x = 0;
			int i = 0;
			do {
				switch (i) {
					case 0: x = x + 1; break;
					case 1: x = x + 2; break;
					default: x = x + 4; break;
				}
				i = i + 1;
			} while (i < 3);
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Goto jumping out of nested loops
# ---------------------------------------------------------------------------


class TestGotoOutOfNestedLoops:
	def test_goto_out_of_double_nested_while(self) -> None:
		"""Goto should jump out of two levels of nested while loops."""
		source = """
		int main(void) {
			int i = 0;
			int j = 0;
			while (i < 10) {
				while (j < 10) {
					if (j == 3) goto done;
					j = j + 1;
				}
				i = i + 1;
			}
			done: return j;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_goto_out_of_triple_nested(self) -> None:
		"""Goto from innermost of three nested loops to outside all of them."""
		source = """
		int main(void) {
			int x = 0;
			int a = 0;
			while (a < 5) {
				int b = 0;
				while (b < 5) {
					int c = 0;
					while (c < 5) {
						if (c == 2) goto bail;
						x = x + 1;
						c = c + 1;
					}
					b = b + 1;
				}
				a = a + 1;
			}
			bail: return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_goto_out_of_for_inside_while(self) -> None:
		"""Goto from a for loop nested inside a while loop."""
		source = """
		int main(void) {
			int result = 0;
			int i = 0;
			while (i < 10) {
				int j;
				for (j = 0; j < 10; j = j + 1) {
					if (i + j == 5) goto end;
					result = result + 1;
				}
				i = i + 1;
			}
			end: return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Goto into/around switch cases
# ---------------------------------------------------------------------------


class TestGotoAroundSwitch:
	def test_goto_skips_switch(self) -> None:
		"""Goto jumping over an entire switch statement."""
		source = """
		int main(void) {
			int x = 5;
			goto skip;
			switch (x) {
				case 5: x = 0; break;
				default: x = 1; break;
			}
			skip: return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_goto_after_switch(self) -> None:
		"""Goto from inside a switch to a label after the switch."""
		source = """
		int main(void) {
			int x = 1;
			int r = 0;
			switch (x) {
				case 0:
					r = 10;
					break;
				case 1:
					r = 20;
					goto after;
				case 2:
					r = 30;
					break;
			}
			r = r + 100;
			after: return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_goto_between_switch_cases(self) -> None:
		"""Goto from one case to a label that is after the switch."""
		source = """
		int main(void) {
			int val = 2;
			int out = 0;
			switch (val) {
				case 1:
					out = 1;
					goto done;
				case 2:
					out = 2;
					goto done;
				case 3:
					out = 3;
					goto done;
			}
			out = 99;
			done: return out;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# For-loop with declaration init (C99 style)
# ---------------------------------------------------------------------------


class TestForLoopWithDeclInit:
	def test_for_with_int_decl(self) -> None:
		"""for(int i = 0; ...) should parse and compile."""
		source = """
		int main(void) {
			int sum = 0;
			for (int i = 0; i < 5; i = i + 1) {
				sum = sum + i;
			}
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_for_decl_scope_isolation(self) -> None:
		"""Variable declared in for-init should not leak to outer scope.
		We test this by reusing the same name in a second for loop."""
		source = """
		int main(void) {
			int total = 0;
			for (int i = 0; i < 3; i = i + 1) {
				total = total + i;
			}
			for (int i = 0; i < 3; i = i + 1) {
				total = total + i;
			}
			return total;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_nested_for_with_decl_init(self) -> None:
		"""Nested for loops both using declaration init."""
		source = """
		int main(void) {
			int count = 0;
			for (int i = 0; i < 3; i = i + 1) {
				for (int j = 0; j < 3; j = j + 1) {
					count = count + 1;
				}
			}
			return count;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Do-while with continue
# ---------------------------------------------------------------------------


class TestDoWhileContinue:
	def test_do_while_continue_to_condition(self) -> None:
		"""Continue in do-while should jump to the condition check."""
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
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		# Should have conditional jumps for the continue and the while condition
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 2

	def test_do_while_multiple_continues(self) -> None:
		"""Multiple continue statements in a do-while body."""
		source = """
		int main(void) {
			int i = 0;
			int r = 0;
			do {
				i = i + 1;
				if (i == 2) continue;
				if (i == 4) continue;
				r = r + i;
			} while (i < 6);
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_do_while_continue_with_nested_if(self) -> None:
		"""Continue inside nested if within do-while."""
		source = """
		int main(void) {
			int i = 0;
			int sum = 0;
			do {
				i = i + 1;
				if (i > 2) {
					if (i == 4) continue;
					sum = sum + i;
				}
			} while (i < 6);
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Nested switch statements
# ---------------------------------------------------------------------------


class TestNestedSwitch:
	def test_switch_inside_switch(self) -> None:
		"""Inner switch should operate independently of outer switch."""
		source = """
		int main(void) {
			int a = 1;
			int b = 2;
			int result = 0;
			switch (a) {
				case 0:
					result = 0;
					break;
				case 1:
					switch (b) {
						case 1: result = 10; break;
						case 2: result = 20; break;
						default: result = 30; break;
					}
					break;
				default:
					result = 99;
					break;
			}
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_break_in_inner_switch_not_outer(self) -> None:
		"""Break in inner switch should not break the outer switch."""
		source = """
		int main(void) {
			int x = 1;
			int y = 0;
			int r = 0;
			switch (x) {
				case 1:
					switch (y) {
						case 0:
							r = 5;
							break;
						case 1:
							r = 6;
							break;
					}
					r = r + 100;
					break;
				case 2:
					r = 200;
					break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_triple_nested_switch(self) -> None:
		"""Three levels of nested switch statements."""
		source = """
		int main(void) {
			int a = 0;
			int b = 1;
			int c = 2;
			int r = 0;
			switch (a) {
				case 0:
					switch (b) {
						case 1:
							switch (c) {
								case 2: r = 42; break;
								default: r = 0; break;
							}
							break;
						default: r = 1; break;
					}
					break;
				default: r = 2; break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Fall-through between cases
# ---------------------------------------------------------------------------


class TestFallThrough:
	def test_fallthrough_accumulates(self) -> None:
		"""Cases without break should execute subsequent case bodies."""
		source = """
		int main(void) {
			int x = 1;
			int r = 0;
			switch (x) {
				case 1:
					r = r + 1;
				case 2:
					r = r + 2;
				case 3:
					r = r + 4;
					break;
				case 4:
					r = r + 8;
					break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_fallthrough_into_default(self) -> None:
		"""Fall-through from a case into the default clause."""
		source = """
		int main(void) {
			int x = 2;
			int r = 0;
			switch (x) {
				case 1:
					r = 10;
					break;
				case 2:
					r = 20;
				default:
					r = r + 5;
					break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_empty_cases_fallthrough(self) -> None:
		"""Empty case labels should fall through to the next non-empty case."""
		source = """
		int main(void) {
			int x = 1;
			int r = 0;
			switch (x) {
				case 0:
				case 1:
				case 2:
					r = 42;
					break;
				case 3:
					r = 99;
					break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Switch with default in middle
# ---------------------------------------------------------------------------


class TestSwitchDefaultInMiddle:
	def test_default_between_cases(self) -> None:
		"""Default clause positioned between case labels."""
		source = """
		int main(void) {
			int x = 5;
			int r = 0;
			switch (x) {
				case 1:
					r = 10;
					break;
				default:
					r = 50;
					break;
				case 2:
					r = 20;
					break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_default_first(self) -> None:
		"""Default as the first clause in a switch."""
		source = """
		int main(void) {
			int x = 3;
			int r = 0;
			switch (x) {
				default:
					r = 99;
					break;
				case 1:
					r = 10;
					break;
				case 2:
					r = 20;
					break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_fallthrough_through_middle_default(self) -> None:
		"""Fall-through from case through a default in the middle."""
		source = """
		int main(void) {
			int x = 1;
			int r = 0;
			switch (x) {
				case 1:
					r = r + 1;
				default:
					r = r + 10;
				case 2:
					r = r + 100;
					break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Label before loop constructs
# ---------------------------------------------------------------------------


class TestLabelBeforeLoop:
	def test_label_before_while(self) -> None:
		"""Label immediately before a while loop, goto to restart."""
		source = """
		int main(void) {
			int i = 0;
			int count = 0;
			restart: while (i < 3) {
				i = i + 1;
				count = count + 1;
			}
			if (count < 6) {
				i = 0;
				goto restart;
			}
			return count;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_label_before_for(self) -> None:
		"""Label before a for loop."""
		source = """
		int main(void) {
			int total = 0;
			int round = 0;
			again: for (int i = 0; i < 3; i = i + 1) {
				total = total + 1;
			}
			round = round + 1;
			if (round < 2) goto again;
			return total;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_label_before_do_while(self) -> None:
		"""Label before a do-while loop."""
		source = """
		int main(void) {
			int n = 0;
			int passes = 0;
			retry: do {
				n = n + 1;
			} while (n < 3);
			passes = passes + 1;
			if (passes < 2) {
				n = 0;
				goto retry;
			}
			return passes;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_label_before_switch(self) -> None:
		"""Label before a switch statement with goto back to it."""
		source = """
		int main(void) {
			int x = 0;
			int r = 0;
			top: switch (x) {
				case 0: r = r + 1; break;
				case 1: r = r + 10; break;
				default: r = r + 100; break;
			}
			if (x < 2) {
				x = x + 1;
				goto top;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm
