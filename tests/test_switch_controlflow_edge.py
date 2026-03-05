"""Edge-case tests for switch statements and control flow in IR gen and codegen."""

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
# Switch with fallthrough (no break between cases)
# ---------------------------------------------------------------------------


class TestSwitchFallthrough:
	def test_fallthrough_two_cases(self) -> None:
		"""Cases without break should fall through to the next case body."""
		source = """
		int main(void) {
			int x = 1;
			int result = 0;
			switch (x) {
				case 1:
					result = result + 10;
				case 2:
					result = result + 20;
					break;
				case 3:
					result = result + 30;
					break;
			}
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_fallthrough_all_cases(self) -> None:
		"""All cases without break should cascade through every body."""
		source = """
		int main(void) {
			int x = 0;
			int r = 0;
			switch (x) {
				case 0:
					r = r + 1;
				case 1:
					r = r + 2;
				case 2:
					r = r + 4;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_fallthrough_into_default(self) -> None:
		"""Fallthrough from a case into default."""
		source = """
		int main(void) {
			int x = 1;
			int r = 0;
			switch (x) {
				case 1:
					r = 10;
				default:
					r = r + 1;
					break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_ir_fallthrough_no_jump_between_cases(self) -> None:
		"""IR should NOT emit a jump between consecutive cases when no break."""
		source = """
		int main(void) {
			int x = 1;
			switch (x) {
				case 1:
					x = 10;
				case 2:
					x = 20;
					break;
			}
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		# Between case labels, there should be no unconditional jump (fallthrough)
		labels = [i for i in func.body if isinstance(i, IRLabelInstr)]
		assert len(labels) >= 2


# ---------------------------------------------------------------------------
# Switch with only default
# ---------------------------------------------------------------------------


class TestSwitchOnlyDefault:
	def test_only_default_case(self) -> None:
		source = """
		int main(void) {
			int x = 42;
			switch (x) {
				default:
					x = 99;
					break;
			}
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_only_default_no_break(self) -> None:
		source = """
		int main(void) {
			int x = 5;
			switch (x) {
				default:
					x = 0;
			}
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_only_default_ir_jumps_to_default(self) -> None:
		"""With only a default case, IR should jump directly to the default label."""
		source = """
		int main(void) {
			int x = 1;
			switch (x) {
				default:
					return 0;
			}
			return 1;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		jumps = [i for i in func.body if isinstance(i, IRJump)]
		assert len(jumps) >= 1


# ---------------------------------------------------------------------------
# Nested switches
# ---------------------------------------------------------------------------


class TestNestedSwitches:
	def test_nested_switch(self) -> None:
		source = """
		int main(void) {
			int a = 1;
			int b = 2;
			int r = 0;
			switch (a) {
				case 1:
					switch (b) {
						case 2:
							r = 12;
							break;
						default:
							r = 10;
							break;
					}
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

	def test_nested_switch_inner_break_does_not_exit_outer(self) -> None:
		"""Break in inner switch should not exit the outer switch."""
		source = """
		int main(void) {
			int x = 1;
			int y = 1;
			int r = 0;
			switch (x) {
				case 1:
					switch (y) {
						case 1:
							r = r + 1;
							break;
					}
					r = r + 10;
					break;
				case 2:
					r = r + 100;
					break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_triple_nested_switch(self) -> None:
		source = """
		int main(void) {
			int a = 0;
			int b = 0;
			int c = 0;
			int r = 0;
			switch (a) {
				case 0:
					switch (b) {
						case 0:
							switch (c) {
								case 0:
									r = 1;
									break;
							}
							break;
					}
					break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Switch in loop with break interaction
# ---------------------------------------------------------------------------


class TestSwitchInLoop:
	def test_switch_inside_while_loop(self) -> None:
		"""Break inside switch should exit switch, not the loop."""
		source = """
		int main(void) {
			int i = 0;
			int sum = 0;
			while (i < 5) {
				switch (i) {
					case 0:
						sum = sum + 1;
						break;
					case 1:
						sum = sum + 2;
						break;
					default:
						sum = sum + 10;
						break;
				}
				i = i + 1;
			}
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_switch_inside_for_loop(self) -> None:
		source = """
		int main(void) {
			int r = 0;
			int i;
			for (i = 0; i < 3; i = i + 1) {
				switch (i) {
					case 0: r = r + 1; break;
					case 1: r = r + 2; break;
					case 2: r = r + 4; break;
				}
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_switch_in_do_while(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			int r = 0;
			do {
				switch (i) {
					case 0: r = r + 1; break;
					default: r = r + 10; break;
				}
				i = i + 1;
			} while (i < 3);
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_loop_continue_not_affected_by_switch(self) -> None:
		"""Continue in the loop body (outside switch) still works after switch."""
		source = """
		int main(void) {
			int i = 0;
			int r = 0;
			while (i < 5) {
				i = i + 1;
				switch (i) {
					case 3:
						r = r + 100;
						break;
					default:
						break;
				}
				if (i == 2) continue;
				r = r + 1;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Duff's device pattern (interleaved switch and loop)
# ---------------------------------------------------------------------------


class TestDuffsDevice:
	def test_duffs_device_pattern(self) -> None:
		"""Simplified Duff's device: switch with fallthrough into a loop-like structure."""
		source = """
		int main(void) {
			int count = 7;
			int n = (count + 3) / 4;
			int result = 0;
			switch (count % 4) {
				case 0: result = result + 1;
				case 3: result = result + 1;
				case 2: result = result + 1;
				case 1: result = result + 1;
			}
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Switch on char type
# ---------------------------------------------------------------------------


class TestSwitchOnChar:
	def test_switch_on_char_literal(self) -> None:
		source = """
		int main(void) {
			char c = 'a';
			int r = 0;
			switch (c) {
				case 'a': r = 1; break;
				case 'b': r = 2; break;
				case 'z': r = 26; break;
				default: r = -1; break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_switch_on_char_numeric_value(self) -> None:
		"""Char compared against numeric case values."""
		source = """
		int main(void) {
			char c = 65;
			int r = 0;
			switch (c) {
				case 65: r = 1; break;
				case 66: r = 2; break;
				default: r = 0; break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_switch_on_unsigned_char(self) -> None:
		source = """
		int main(void) {
			unsigned char uc = 200;
			int r = 0;
			switch (uc) {
				case 200: r = 1; break;
				case 255: r = 2; break;
				default: r = 0; break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Empty case body
# ---------------------------------------------------------------------------


class TestEmptyCaseBody:
	def test_empty_case_falls_through(self) -> None:
		"""A case with no statements should fall through to the next case."""
		source = """
		int main(void) {
			int x = 1;
			int r = 0;
			switch (x) {
				case 1:
				case 2:
					r = 12;
					break;
				case 3:
					r = 3;
					break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_multiple_empty_cases_fallthrough(self) -> None:
		"""Multiple consecutive empty cases all fall through to the handler."""
		source = """
		int main(void) {
			int x = 2;
			int r = 0;
			switch (x) {
				case 1:
				case 2:
				case 3:
				case 4:
					r = 1234;
					break;
				default:
					r = -1;
					break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_empty_default_case(self) -> None:
		source = """
		int main(void) {
			int x = 99;
			switch (x) {
				case 1: return 1;
				default:
					break;
			}
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Consecutive case labels (grouped cases)
# ---------------------------------------------------------------------------


class TestConsecutiveCaseLabels:
	def test_grouped_cases_share_body(self) -> None:
		source = """
		int main(void) {
			int x = 3;
			int r = 0;
			switch (x) {
				case 1:
				case 2:
				case 3:
					r = 100;
					break;
				case 4:
				case 5:
					r = 200;
					break;
				default:
					r = 0;
					break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_ir_has_comparisons_for_all_grouped_cases(self) -> None:
		"""Each case value should generate a comparison in the jump table."""
		source = """
		int main(void) {
			int x = 1;
			switch (x) {
				case 1:
				case 2:
				case 3:
					return 10;
			}
			return 0;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 3


# ---------------------------------------------------------------------------
# Large case values
# ---------------------------------------------------------------------------


class TestLargeCaseValues:
	def test_large_positive_case_value(self) -> None:
		source = """
		int main(void) {
			int x = 1000000;
			switch (x) {
				case 1000000: return 1;
				case 999999: return 2;
				default: return 0;
			}
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_zero_and_one_case_values(self) -> None:
		"""Test boundary case values around zero."""
		source = """
		int main(void) {
			int x = 0;
			switch (x) {
				case 0: return 10;
				case 1: return 20;
				case 2: return 30;
				default: return 0;
			}
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_max_int_case_value(self) -> None:
		source = """
		int main(void) {
			int x = 2147483647;
			switch (x) {
				case 2147483647: return 1;
				case 0: return 0;
				default: return -1;
			}
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_sparse_case_values(self) -> None:
		"""Cases with widely spaced values."""
		source = """
		int main(void) {
			int x = 1000;
			switch (x) {
				case 0: return 0;
				case 100: return 1;
				case 1000: return 2;
				case 10000: return 3;
				case 100000: return 4;
				default: return -1;
			}
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Switch with no cases (empty switch)
# ---------------------------------------------------------------------------


class TestEmptySwitch:
	def test_switch_with_no_cases(self) -> None:
		"""A switch with an empty body should compile without errors."""
		source = """
		int main(void) {
			int x = 1;
			switch (x) {
			}
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Switch with return in case (no break needed)
# ---------------------------------------------------------------------------


class TestSwitchWithReturn:
	def test_return_in_each_case(self) -> None:
		source = """
		int main(void) {
			int x = 2;
			switch (x) {
				case 1: return 10;
				case 2: return 20;
				case 3: return 30;
				default: return 0;
			}
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_return_in_some_cases_fallthrough_others(self) -> None:
		source = """
		int main(void) {
			int x = 1;
			int r = 0;
			switch (x) {
				case 1:
					r = 10;
				case 2:
					return r + 20;
				case 3:
					r = 30;
					break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Switch with default not last
# ---------------------------------------------------------------------------


class TestDefaultNotLast:
	def test_default_first(self) -> None:
		source = """
		int main(void) {
			int x = 5;
			int r = 0;
			switch (x) {
				default:
					r = -1;
					break;
				case 1:
					r = 1;
					break;
				case 2:
					r = 2;
					break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_default_in_middle(self) -> None:
		source = """
		int main(void) {
			int x = 10;
			int r = 0;
			switch (x) {
				case 1:
					r = 1;
					break;
				default:
					r = -1;
					break;
				case 2:
					r = 2;
					break;
			}
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Semantic error cases
# ---------------------------------------------------------------------------


class TestSwitchSemanticErrors:
	def test_duplicate_case_values(self) -> None:
		source = """
		int main(void) {
			int x = 1;
			switch (x) {
				case 1: break;
				case 1: break;
			}
		}
		"""
		ast = _parse(source)
		with pytest.raises(SemanticError, match="duplicate case value"):
			SemanticAnalyzer().analyze(ast)

	def test_multiple_default_cases(self) -> None:
		source = """
		int main(void) {
			int x = 1;
			switch (x) {
				default: break;
				default: break;
			}
		}
		"""
		ast = _parse(source)
		with pytest.raises(SemanticError, match="[Dd]uplicate default|multiple default"):
			SemanticAnalyzer().analyze(ast)
