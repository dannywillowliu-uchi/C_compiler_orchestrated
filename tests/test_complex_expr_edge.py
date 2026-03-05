"""Edge-case tests for comma operator, ternary, and complex expression interactions.

Covers: comma operator in various contexts (function args, for loops, assignments),
nested ternary expressions, ternary with side effects, comma in ternary branches,
short-circuit evaluation with side effects in &&/||.
"""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRBinOp,
	IRCall,
	IRCondJump,
	IRReturn,
	IRStore,
)
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
# Comma operator in various contexts
# ---------------------------------------------------------------------------


class TestCommaOperatorContexts:
	def test_comma_in_return(self) -> None:
		"""return (a, b) should return b."""
		source = """
		int main(void) {
			return (1, 42);
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		ret = [i for i in func.body if isinstance(i, IRReturn)]
		assert len(ret) >= 1

	def test_comma_chain_three(self) -> None:
		"""(a, b, c) evaluates all and returns c."""
		source = """
		int main(void) {
			int x = 0;
			int y = 0;
			int z;
			z = (x = 1, y = 2, x + y);
			return z;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_comma_chain_five(self) -> None:
		"""Five-element comma chain should compile."""
		source = """
		int main(void) {
			int a = 0;
			int b = 0;
			int c = 0;
			int d = 0;
			int result;
			result = (a = 1, b = 2, c = 3, d = 4, a + b + c + d);
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_comma_in_for_all_parts(self) -> None:
		"""Comma in both init and update of for loop."""
		source = """
		int main(void) {
			int i;
			int j;
			int k;
			int sum = 0;
			for (i = 0, j = 10, k = 100; i < 3; i = i + 1, j = j - 1, k = k - 10) {
				sum = sum + i + j + k;
			}
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_comma_in_while_condition(self) -> None:
		"""Comma expression as a while condition: while((x++, x < 5))."""
		source = """
		int main(void) {
			int x = 0;
			int count = 0;
			while ((x = x + 1, x < 5)) {
				count = count + 1;
			}
			return count;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_comma_in_if_condition(self) -> None:
		"""Comma expression as if condition."""
		source = """
		int main(void) {
			int x = 0;
			if ((x = 5, x > 3)) {
				return 1;
			}
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_comma_discards_left_value(self) -> None:
		"""Left side evaluated for side effects, value discarded."""
		source = """
		int g;
		int main(void) {
			int x;
			x = (g = 99, 42);
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		stores = [i for i in func.body if isinstance(i, IRStore)]
		assert len(stores) >= 1, "Side effect on left of comma should produce store"

	def test_comma_with_function_call_side_effect(self) -> None:
		"""Comma left side is a function call (side effect)."""
		source = """
		int counter;
		void inc(void) { counter = counter + 1; }
		int main(void) {
			int x;
			counter = 0;
			x = (inc(), 10);
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		main_func = [f for f in ir.functions if f.name == "main"][0]
		calls = [i for i in main_func.body if isinstance(i, IRCall)]
		assert len(calls) >= 1, "Comma left side function call should be emitted"

	def test_comma_nested_in_assignment(self) -> None:
		"""x = (y = 1, z = 2, y + z) should assign 3 to x."""
		source = """
		int main(void) {
			int x;
			int y;
			int z;
			x = (y = 1, z = 2, y + z);
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Nested ternary expressions (deep and complex)
# ---------------------------------------------------------------------------


class TestNestedTernaryEdge:
	def test_deeply_nested_ternary_four_levels(self) -> None:
		"""Four levels of nesting: a ? b ? c ? d ? 1 : 2 : 3 : 4 : 5."""
		source = """
		int main(void) {
			int a = 1;
			int b = 1;
			int c = 1;
			int d = 1;
			return a ? b ? c ? d ? 1 : 2 : 3 : 4 : 5;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 4

	def test_ternary_both_branches_nested(self) -> None:
		"""a ? (b ? 1 : 2) : (c ? 3 : 4) -- nested in both branches."""
		source = """
		int main(void) {
			int a = 0;
			int b = 1;
			int c = 0;
			return a ? (b ? 1 : 2) : (c ? 3 : 4);
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 3, "Ternary in both branches needs >= 3 cond jumps"

	def test_ternary_with_comparison_condition(self) -> None:
		"""Ternary condition is a comparison expression."""
		source = """
		int main(void) {
			int x = 10;
			int y = 20;
			return (x > y) ? x : y;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == ">"]
		assert len(binops) >= 1

	def test_ternary_with_logical_condition(self) -> None:
		"""Ternary condition is a logical expression: (a && b) ? 1 : 0."""
		source = """
		int main(void) {
			int a = 1;
			int b = 1;
			return (a && b) ? 1 : 0;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 2, "Logical && in ternary condition produces extra cond jumps"

	def test_ternary_as_function_argument(self) -> None:
		"""Ternary used as a function argument."""
		source = """
		int identity(int x) { return x; }
		int main(void) {
			int a = 1;
			return identity(a ? 10 : 20);
		}
		"""
		ir = _compile_to_ir(source)
		main_func = [f for f in ir.functions if f.name == "main"][0]
		calls = [i for i in main_func.body if isinstance(i, IRCall)]
		assert len(calls) >= 1

	def test_ternary_as_array_index(self) -> None:
		"""Ternary used as an array subscript."""
		source = """
		int main(void) {
			int arr[3];
			arr[0] = 10;
			arr[1] = 20;
			arr[2] = 30;
			int i = 1;
			return arr[i ? 2 : 0];
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_ternary_with_assignment_in_branches(self) -> None:
		"""Side-effecting assignments inside ternary branches."""
		source = """
		int main(void) {
			int x = 0;
			int y = 0;
			int cond = 1;
			int result = cond ? (x = 10) : (y = 20);
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_ternary_with_unary_ops(self) -> None:
		"""Ternary branches contain unary operators."""
		source = """
		int main(void) {
			int a = 5;
			int b = 3;
			return (a > b) ? -a : ~b;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_ternary_result_in_arithmetic(self) -> None:
		"""Ternary result used in further arithmetic."""
		source = """
		int main(void) {
			int x = 1;
			int y = (x ? 10 : 5) + (x ? 3 : 7);
			return y;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 2, "Two ternaries in expression need >= 2 cond jumps"

	def test_ternary_zero_condition(self) -> None:
		"""Ternary with literal 0 as condition always takes false branch."""
		source = """
		int main(void) {
			return 0 ? 99 : 42;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_ternary_nonzero_condition(self) -> None:
		"""Ternary with non-zero literal condition always takes true branch."""
		source = """
		int main(void) {
			return 5 ? 42 : 99;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Comma in ternary branches
# ---------------------------------------------------------------------------


class TestCommaInTernary:
	def test_comma_in_true_branch(self) -> None:
		"""a ? (b, c) : d -- comma in true branch."""
		source = """
		int main(void) {
			int x = 1;
			int a = 0;
			int result = x ? (a = 5, a + 1) : 0;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_comma_in_false_branch(self) -> None:
		"""a ? b : (c, d) -- comma in false branch."""
		source = """
		int main(void) {
			int x = 0;
			int a = 0;
			int result = x ? 99 : (a = 10, a * 2);
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_comma_in_both_ternary_branches(self) -> None:
		"""Comma expressions in both branches of ternary."""
		source = """
		int main(void) {
			int x = 1;
			int a = 0;
			int b = 0;
			int result = x ? (a = 1, a + 10) : (b = 2, b + 20);
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_comma_in_ternary_condition(self) -> None:
		"""(a, b) ? c : d -- comma in condition."""
		source = """
		int main(void) {
			int x = 0;
			int result = (x = 1, x) ? 42 : 0;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_nested_ternary_with_comma_everywhere(self) -> None:
		"""Nested ternary with comma in condition and branches."""
		source = """
		int main(void) {
			int a = 0;
			int b = 0;
			int result = (a = 1, a) ? (b = 10, b) : (b = 20, b ? 30 : 40);
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Short-circuit evaluation with side effects in && and ||
# ---------------------------------------------------------------------------


class TestShortCircuitSideEffects:
	def test_and_left_false_skips_right_call(self) -> None:
		"""0 && f() should not call f (short-circuit). IR should have cond jump before call."""
		source = """
		int dummy(void) { return 1; }
		int main(void) {
			int result = 0 && dummy();
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		main_func = [f for f in ir.functions if f.name == "main"][0]
		cond_jumps = [i for i in main_func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 1

	def test_or_left_true_skips_right_call(self) -> None:
		"""1 || f() should not call f (short-circuit)."""
		source = """
		int dummy(void) { return 1; }
		int main(void) {
			int result = 1 || dummy();
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		main_func = [f for f in ir.functions if f.name == "main"][0]
		cond_jumps = [i for i in main_func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 1

	def test_and_with_assignment_side_effect(self) -> None:
		"""a && (b = 5) -- b should only be assigned if a is true."""
		source = """
		int main(void) {
			int a = 1;
			int b = 0;
			int result = a && (b = 5);
			return b;
		}
		"""
		ir = _compile_to_ir(source)
		main_func = ir.functions[0]
		cond_jumps = [i for i in main_func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 1

	def test_or_with_assignment_side_effect(self) -> None:
		"""a || (b = 5) -- b should only be assigned if a is false."""
		source = """
		int main(void) {
			int a = 0;
			int b = 0;
			int result = a || (b = 5);
			return b;
		}
		"""
		ir = _compile_to_ir(source)
		main_func = ir.functions[0]
		cond_jumps = [i for i in main_func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 1

	def test_chained_and_with_side_effects(self) -> None:
		"""a && b && c with variable conditions."""
		source = """
		int main(void) {
			int x = 1;
			int y = 2;
			int z = 3;
			int result = (x > 0) && (y > 0) && (z > 0);
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 2

	def test_chained_or_with_side_effects(self) -> None:
		"""a || b || c with variable conditions."""
		source = """
		int main(void) {
			int x = 0;
			int y = 0;
			int z = 1;
			int result = (x > 0) || (y > 0) || (z > 0);
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 2

	def test_mixed_and_or_with_side_effects(self) -> None:
		"""(a && b) || (c && d) -- complex short-circuit."""
		source = """
		int main(void) {
			int a = 1;
			int b = 0;
			int c = 1;
			int d = 1;
			int result = (a && b) || (c && d);
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 3

	def test_not_in_short_circuit(self) -> None:
		"""!a && b -- logical NOT combined with short-circuit."""
		source = """
		int main(void) {
			int a = 0;
			int b = 1;
			int result = !a && b;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_circuit_in_if_condition(self) -> None:
		"""Short-circuit && in if statement condition."""
		source = """
		int main(void) {
			int x = 5;
			int y = 10;
			if (x > 0 && y > 0) {
				return 1;
			}
			return 0;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 2

	def test_short_circuit_in_while_condition(self) -> None:
		"""Short-circuit || in while loop condition."""
		source = """
		int main(void) {
			int x = 0;
			int count = 0;
			while (x < 3 || count < 1) {
				x = x + 1;
				count = count + 1;
			}
			return count;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_short_circuit_in_for_condition(self) -> None:
		"""Short-circuit && in for loop condition."""
		source = """
		int main(void) {
			int sum = 0;
			int i;
			for (i = 0; i < 10 && sum < 20; i = i + 1) {
				sum = sum + i;
			}
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Complex expression interactions (ternary + comma + short-circuit combined)
# ---------------------------------------------------------------------------


class TestComplexExprInteractions:
	def test_ternary_condition_with_and(self) -> None:
		"""(a && b) ? x : y -- short-circuit in ternary condition."""
		source = """
		int main(void) {
			int a = 1;
			int b = 1;
			return (a && b) ? 42 : 0;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 2

	def test_ternary_condition_with_or(self) -> None:
		"""(a || b) ? x : y."""
		source = """
		int main(void) {
			int a = 0;
			int b = 1;
			return (a || b) ? 42 : 0;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 2

	def test_short_circuit_and_in_ternary_branches(self) -> None:
		"""Ternary branches themselves contain && expressions."""
		source = """
		int main(void) {
			int x = 1;
			int a = 1;
			int b = 1;
			int result = x ? (a && b) : 0;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_comma_and_short_circuit_combined(self) -> None:
		"""(a = 1, a) && (b = 2, b) -- comma inside &&."""
		source = """
		int main(void) {
			int a = 0;
			int b = 0;
			int result = (a = 1, a) && (b = 2, b);
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_all_three_combined(self) -> None:
		"""Comma + ternary + short-circuit all in one expression."""
		source = """
		int main(void) {
			int a = 0;
			int b = 0;
			int c = 0;
			int result = (a = 1, a) ? ((b = 2, b) && (c = 3, c)) : 0;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_ternary_in_comma_expression(self) -> None:
		"""(x ? 1 : 2, y ? 3 : 4) -- ternary inside comma."""
		source = """
		int main(void) {
			int x = 1;
			int y = 0;
			int result = (x ? 1 : 2, y ? 3 : 4);
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 2, "Two ternaries in comma need >= 2 cond jumps"

	def test_ternary_in_for_loop_update(self) -> None:
		"""Ternary inside for-loop update expression."""
		source = """
		int main(void) {
			int i;
			int sum = 0;
			for (i = 0; i < 6; i = (i < 3) ? i + 1 : i + 2) {
				sum = sum + 1;
			}
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_nested_function_calls_with_ternary_args(self) -> None:
		"""Function calls with ternary expressions as arguments."""
		source = """
		int add(int a, int b) { return a + b; }
		int main(void) {
			int x = 1;
			int y = 0;
			return add(x ? 10 : 5, y ? 20 : 15);
		}
		"""
		ir = _compile_to_ir(source)
		main_func = [f for f in ir.functions if f.name == "main"][0]
		calls = [i for i in main_func.body if isinstance(i, IRCall)]
		assert len(calls) >= 1
		cond_jumps = [i for i in main_func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 2

	def test_compound_assignment_with_ternary_rhs(self) -> None:
		"""x += (cond ? a : b)."""
		source = """
		int main(void) {
			int x = 10;
			int cond = 1;
			x += cond ? 5 : 15;
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_ternary_in_switch_case(self) -> None:
		"""Ternary inside switch case body."""
		source = """
		int main(void) {
			int x = 2;
			int result = 0;
			switch (x) {
				case 1:
					result = 0 ? 10 : 20;
					break;
				case 2:
					result = 1 ? 30 : 40;
					break;
				default:
					result = 50;
					break;
			}
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_comma_in_do_while_condition(self) -> None:
		"""Comma expression in do-while condition."""
		source = """
		int main(void) {
			int x = 0;
			int count = 0;
			do {
				count = count + 1;
				x = x + 1;
			} while ((x = x, x < 3));
			return count;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_deeply_nested_logical_chain(self) -> None:
		"""(a && b && c) || (d && e) -- complex short-circuit tree."""
		source = """
		int main(void) {
			int a = 1;
			int b = 1;
			int c = 0;
			int d = 1;
			int e = 1;
			int result = (a && b && c) || (d && e);
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 4

	def test_ternary_with_postfix_ops(self) -> None:
		"""Ternary with postfix increment/decrement in branches."""
		source = """
		int main(void) {
			int a = 5;
			int b = 10;
			int cond = 1;
			int result = cond ? a++ : b--;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_logical_not_with_ternary(self) -> None:
		"""!cond ? a : b -- negated ternary condition."""
		source = """
		int main(void) {
			int cond = 0;
			return !cond ? 42 : 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_multiple_ternaries_in_sequence(self) -> None:
		"""Multiple independent ternary statements."""
		source = """
		int main(void) {
			int a = 1;
			int b = 0;
			int x = a ? 10 : 20;
			int y = b ? 30 : 40;
			int z = (x > y) ? x : y;
			return z;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 3
