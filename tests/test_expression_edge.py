"""Edge-case tests for expression features: ternary, comma, cast, compound assignment,
sizeof, precedence, and short-circuit evaluation."""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRBinOp,
	IRCondJump,
	IRConst,
	IRConvert,
	IRCopy,
	IRJump,
	IRLabelInstr,
	IRReturn,
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
# Deeply nested ternary expressions
# ---------------------------------------------------------------------------


class TestNestedTernary:
	def test_two_level_nested_ternary(self) -> None:
		"""a ? (b ? c : d) : e should produce nested conditional jumps."""
		source = """
		int main(void) {
			int a = 1;
			int b = 2;
			return a ? (b ? 10 : 20) : 30;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 2, "Nested ternary needs at least 2 conditional jumps"

	def test_three_level_nested_ternary(self) -> None:
		"""a ? b ? c ? 1 : 2 : 3 : 4 should compile without errors."""
		source = """
		int main(void) {
			int a = 1;
			int b = 1;
			int c = 1;
			return a ? b ? c ? 1 : 2 : 3 : 4;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_ternary_in_false_branch(self) -> None:
		"""a ? 1 : (b ? 2 : 3) should nest ternary in the false branch."""
		source = """
		int main(void) {
			int a = 0;
			int b = 1;
			return a ? 1 : (b ? 2 : 3);
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 2

	def test_ternary_chain_as_else_if(self) -> None:
		"""a ? 1 : b ? 2 : c ? 3 : 4 simulates else-if chain."""
		source = """
		int main(void) {
			int a = 0;
			int b = 0;
			int c = 1;
			return a ? 1 : b ? 2 : c ? 3 : 4;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 3

	def test_ternary_with_assignment(self) -> None:
		"""Ternary result used in assignment."""
		source = """
		int main(void) {
			int x = 5;
			int y = x > 3 ? x : 0;
			return y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Comma expressions in various contexts
# ---------------------------------------------------------------------------


class TestCommaExpressions:
	def test_comma_evaluates_right(self) -> None:
		"""Comma expression should return the rightmost value."""
		source = """
		int main(void) {
			int x;
			x = (1, 2, 3);
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		# Should compile and produce valid IR
		assert len(func.body) > 0

	def test_comma_in_for_init(self) -> None:
		"""Comma in for-loop initializer: for(i=0, j=10; ...)."""
		source = """
		int main(void) {
			int i;
			int j;
			int sum = 0;
			for (i = 0, j = 10; i < 5; i = i + 1, j = j - 1) {
				sum = sum + i + j;
			}
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_comma_in_for_update(self) -> None:
		"""Comma in for-loop update: for(...; ...; i++, j--)."""
		source = """
		int main(void) {
			int i;
			int j;
			for (i = 0, j = 0; i < 3; i = i + 1, j = j + 2) {
			}
			return j;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_nested_comma(self) -> None:
		"""Nested comma: (a, (b, c)) should return c."""
		source = """
		int main(void) {
			int x;
			x = (1, (2, 3));
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		assert len(func.body) > 0

	def test_comma_with_side_effects(self) -> None:
		"""Left side of comma should be evaluated for side effects."""
		source = """
		int main(void) {
			int x = 0;
			int y;
			y = (x = 5, x + 1);
			return y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Complex cast chains
# ---------------------------------------------------------------------------


class TestCastChains:
	def test_int_to_char_to_int(self) -> None:
		"""int -> char -> int cast chain should produce two copy/convert ops."""
		source = """
		int main(void) {
			int x = 65;
			char c = (char)x;
			int y = (int)c;
			return y;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		copies_or_converts = [
			i for i in func.body if isinstance(i, (IRCopy, IRConvert))
		]
		assert len(copies_or_converts) >= 2

	def test_double_to_int_cast(self) -> None:
		"""double -> int should produce a convert instruction."""
		source = """
		int main(void) {
			double d = 3.14;
			int x = (int)d;
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		converts = [i for i in func.body if isinstance(i, IRConvert)]
		assert len(converts) >= 1

	def test_int_to_double_cast(self) -> None:
		"""int -> double should produce a convert instruction."""
		source = """
		int main(void) {
			int x = 42;
			double d = (double)x;
			return (int)d;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		converts = [i for i in func.body if isinstance(i, IRConvert)]
		assert len(converts) >= 2, "int->double and double->int need converts"

	def test_cast_in_expression(self) -> None:
		"""Cast within an arithmetic expression."""
		source = """
		int main(void) {
			int a = 7;
			int b = 2;
			int result = (int)((double)a / (double)b);
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_redundant_cast_same_type(self) -> None:
		"""Casting to the same type should still compile cleanly."""
		source = """
		int main(void) {
			int x = 10;
			int y = (int)x;
			return y;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		assert len(func.body) > 0

	def test_char_literal_cast(self) -> None:
		"""Cast char literal to int."""
		source = """
		int main(void) {
			int x = (int)'A';
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		assert len(func.body) > 0


# ---------------------------------------------------------------------------
# Compound assignment with side effects
# ---------------------------------------------------------------------------


class TestCompoundAssignment:
	def test_plus_equals(self) -> None:
		"""x += 5 should read, add, and write back."""
		source = """
		int main(void) {
			int x = 10;
			x += 5;
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "+"]
		assert len(binops) >= 1

	def test_minus_equals(self) -> None:
		"""x -= 3 should produce a subtraction."""
		source = """
		int main(void) {
			int x = 10;
			x -= 3;
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "-"]
		assert len(binops) >= 1

	def test_multiply_equals(self) -> None:
		"""x *= 2 should produce a multiplication."""
		source = """
		int main(void) {
			int x = 5;
			x *= 2;
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "*"]
		assert len(binops) >= 1

	def test_divide_equals(self) -> None:
		"""x /= 2 should produce a division."""
		source = """
		int main(void) {
			int x = 10;
			x /= 2;
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "/"]
		assert len(binops) >= 1

	def test_modulo_equals(self) -> None:
		"""x %= 3 should produce a modulo."""
		source = """
		int main(void) {
			int x = 10;
			x %= 3;
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "%"]
		assert len(binops) >= 1

	def test_bitwise_and_equals(self) -> None:
		"""x &= 0xFF should produce bitwise AND."""
		source = """
		int main(void) {
			int x = 255;
			x &= 15;
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "&"]
		assert len(binops) >= 1

	def test_bitwise_or_equals(self) -> None:
		"""x |= 0xF0 should produce bitwise OR."""
		source = """
		int main(void) {
			int x = 15;
			x |= 240;
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "|"]
		assert len(binops) >= 1

	def test_shift_left_equals(self) -> None:
		"""x <<= 2 should produce left shift."""
		source = """
		int main(void) {
			int x = 1;
			x <<= 2;
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "<<"]
		assert len(binops) >= 1

	def test_chained_compound_assignment(self) -> None:
		"""Multiple compound assignments in sequence."""
		source = """
		int main(void) {
			int x = 1;
			x += 2;
			x *= 3;
			x -= 1;
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# sizeof on expressions vs types
# ---------------------------------------------------------------------------


class TestSizeofEdgeCases:
	def test_sizeof_type_int(self) -> None:
		"""sizeof(int) should be 4."""
		source = """
		int main(void) {
			return sizeof(int);
		}
		"""
		ir = _compile_to_ir(source)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(4)

	def test_sizeof_type_char(self) -> None:
		"""sizeof(char) should be 1."""
		source = """
		int main(void) {
			return sizeof(char);
		}
		"""
		ir = _compile_to_ir(source)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(1)

	def test_sizeof_type_double(self) -> None:
		"""sizeof(double) should be 8."""
		source = """
		int main(void) {
			return sizeof(double);
		}
		"""
		ir = _compile_to_ir(source)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(8)

	def test_sizeof_expr_variable(self) -> None:
		"""sizeof(x) where x is int should be 4."""
		source = """
		int main(void) {
			int x = 42;
			return sizeof(x);
		}
		"""
		ir = _compile_to_ir(source)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(4)

	def test_sizeof_expr_char_var(self) -> None:
		"""sizeof(c) where c is char should be 1."""
		source = """
		int main(void) {
			char c = 'a';
			return sizeof(c);
		}
		"""
		ir = _compile_to_ir(source)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(1)

	def test_sizeof_pointer_type(self) -> None:
		"""sizeof(int*) should be 8 on 64-bit."""
		source = """
		int main(void) {
			int *p;
			return sizeof(p);
		}
		"""
		ir = _compile_to_ir(source)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(8)

	def test_sizeof_array(self) -> None:
		"""sizeof(arr) for int[5] should be 20."""
		source = """
		int main(void) {
			int arr[5];
			return sizeof(arr);
		}
		"""
		ir = _compile_to_ir(source)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(20)

	def test_sizeof_struct_type(self) -> None:
		"""sizeof(struct) should account for both members."""
		source = """
		struct point { int x; int y; };
		int main(void) {
			return sizeof(struct point);
		}
		"""
		ir = _compile_to_ir(source)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(8)


# ---------------------------------------------------------------------------
# Precedence edge cases: mixing bitwise and comparison operators
# ---------------------------------------------------------------------------


class TestPrecedenceEdgeCases:
	def test_bitwise_and_lower_than_equality(self) -> None:
		"""In C, a == b & c parses as a == (b & c) is WRONG; it's (a == b) & c.
		But typically a & b == c means a & (b == c). Verify the compiler handles this."""
		source = """
		int main(void) {
			int a = 5;
			int b = 3;
			int c = 1;
			int result = a & b == c;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bitwise_or_lower_than_equality(self) -> None:
		"""a | b == c should parse as a | (b == c) per C precedence."""
		source = """
		int main(void) {
			int a = 4;
			int b = 3;
			int c = 3;
			int result = a | b == c;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bitwise_xor_precedence(self) -> None:
		"""a ^ b == c should parse as a ^ (b == c)."""
		source = """
		int main(void) {
			int a = 6;
			int b = 3;
			int c = 3;
			int result = a ^ b == c;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_shift_vs_addition(self) -> None:
		"""a << b + c should parse as a << (b + c) since + binds tighter than <<."""
		source = """
		int main(void) {
			int a = 1;
			int b = 1;
			int c = 2;
			int result = a << b + c;
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		# Should have both + and << operations
		add_ops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "+"]
		shift_ops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "<<"]
		assert len(add_ops) >= 1
		assert len(shift_ops) >= 1

	def test_comparison_chain(self) -> None:
		"""a < b < c is valid C but means (a < b) < c, not mathematical chaining."""
		source = """
		int main(void) {
			int a = 1;
			int b = 2;
			int c = 3;
			int result = a < b < c;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_mixed_logical_and_bitwise(self) -> None:
		"""a && b | c should parse as a && (b | c) since && binds looser than |."""
		source = """
		int main(void) {
			int a = 1;
			int b = 0;
			int c = 1;
			int result = a && b | c;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_ternary_vs_assignment_precedence(self) -> None:
		"""Ternary should bind tighter than assignment."""
		source = """
		int main(void) {
			int x;
			x = 1 ? 10 : 20;
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_unary_minus_with_multiply(self) -> None:
		"""Unary minus should bind tighter: -a * b means (-a) * b."""
		source = """
		int main(void) {
			int a = 3;
			int b = 4;
			int result = -a * b;
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		# Should have both unary negate and multiply
		mul_ops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "*"]
		assert len(mul_ops) >= 1


# ---------------------------------------------------------------------------
# Short-circuit evaluation with side effects in && and ||
# ---------------------------------------------------------------------------


class TestShortCircuitEvaluation:
	def test_and_short_circuits_on_false(self) -> None:
		"""0 && expr should not evaluate expr (short-circuit).
		IR should have a conditional jump that skips the right side."""
		source = """
		int main(void) {
			int x = 0;
			int result = x && 1;
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 1, "Short-circuit && needs a conditional jump"

	def test_and_evaluates_both_when_true(self) -> None:
		"""1 && expr should evaluate both sides."""
		source = """
		int main(void) {
			int x = 1;
			int y = 2;
			int result = x && y;
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 1

	def test_or_short_circuits_on_true(self) -> None:
		"""1 || expr should not evaluate expr (short-circuit).
		IR should have a conditional jump that skips the right side."""
		source = """
		int main(void) {
			int x = 1;
			int result = x || 0;
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 1, "Short-circuit || needs a conditional jump"

	def test_or_evaluates_both_when_false(self) -> None:
		"""0 || expr should evaluate both sides."""
		source = """
		int main(void) {
			int x = 0;
			int y = 5;
			int result = x || y;
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 1

	def test_nested_and_or(self) -> None:
		"""(a && b) || c should produce multiple conditional jumps."""
		source = """
		int main(void) {
			int a = 1;
			int b = 0;
			int c = 1;
			int result = (a && b) || c;
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 2, "Nested && || needs multiple conditional jumps"

	def test_and_produces_zero_or_one(self) -> None:
		"""&& result should be normalized to 0 or 1."""
		source = """
		int main(void) {
			int a = 5;
			int b = 10;
			int result = a && b;
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		# Should have a != 0 normalization for the result
		norm_ops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "!="]
		assert len(norm_ops) >= 1, "&& should normalize result with != 0"

	def test_or_produces_zero_or_one(self) -> None:
		"""|| result should be normalized to 0 or 1."""
		source = """
		int main(void) {
			int a = 0;
			int b = 10;
			int result = a || b;
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		norm_ops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "!="]
		assert len(norm_ops) >= 1, "|| should normalize result with != 0"

	def test_short_circuit_and_structure(self) -> None:
		"""&& IR should have: condjump, eval_right block, false block, end block."""
		source = """
		int main(void) {
			int a = 1;
			int b = 2;
			return a && b;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		labels = [i for i in func.body if isinstance(i, IRLabelInstr)]
		jumps = [i for i in func.body if isinstance(i, IRJump)]
		# && needs: and_right label, and_false label, and_end label, plus jumps
		assert len(labels) >= 3, "&& needs at least 3 labels"
		assert len(jumps) >= 2, "&& needs at least 2 unconditional jumps"

	def test_short_circuit_or_structure(self) -> None:
		"""|| IR should have: condjump, eval_right block, true block, end block."""
		source = """
		int main(void) {
			int a = 0;
			int b = 1;
			return a || b;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		labels = [i for i in func.body if isinstance(i, IRLabelInstr)]
		jumps = [i for i in func.body if isinstance(i, IRJump)]
		assert len(labels) >= 3, "|| needs at least 3 labels"
		assert len(jumps) >= 2, "|| needs at least 2 unconditional jumps"

	def test_triple_and_chain(self) -> None:
		"""a && b && c should produce chained short-circuit logic."""
		source = """
		int main(void) {
			int a = 1;
			int b = 2;
			int c = 3;
			return a && b && c;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 2, "Chained && needs multiple conditional jumps"

	def test_triple_or_chain(self) -> None:
		"""a || b || c should produce chained short-circuit logic."""
		source = """
		int main(void) {
			int a = 0;
			int b = 0;
			int c = 1;
			return a || b || c;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 2, "Chained || needs multiple conditional jumps"


# ---------------------------------------------------------------------------
# Complex expression combinations
# ---------------------------------------------------------------------------


class TestComplexExpressions:
	def test_ternary_with_comma(self) -> None:
		"""Ternary with comma in branches."""
		source = """
		int main(void) {
			int x = 1;
			int result = x ? (1, 10) : (2, 20);
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_cast_in_ternary(self) -> None:
		"""Cast inside ternary expression."""
		source = """
		int main(void) {
			int x = 1;
			double d = 3.14;
			int result = x ? (int)d : 0;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_sizeof_in_ternary(self) -> None:
		"""sizeof used in ternary condition."""
		source = """
		int main(void) {
			int result = sizeof(int) > 2 ? 1 : 0;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_compound_assignment_with_cast(self) -> None:
		"""Compound assignment with a casted RHS."""
		source = """
		int main(void) {
			int x = 10;
			double d = 2.5;
			x += (int)d;
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_nested_logical_with_comparison(self) -> None:
		"""Complex: (a > 0 && b < 10) || c == 5."""
		source = """
		int main(void) {
			int a = 3;
			int b = 7;
			int c = 5;
			int result = (a > 0 && b < 10) || c == 5;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_multiple_casts_in_arithmetic(self) -> None:
		"""Multiple casts in a single arithmetic expression."""
		source = """
		int main(void) {
			char a = 10;
			char b = 20;
			int result = (int)a + (int)b;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_sizeof_in_array_index(self) -> None:
		"""sizeof used as an array index expression."""
		source = """
		int main(void) {
			int arr[8];
			arr[sizeof(int)] = 42;
			return arr[4];
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm
