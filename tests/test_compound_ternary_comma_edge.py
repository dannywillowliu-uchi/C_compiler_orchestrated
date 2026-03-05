"""Edge-case tests for compound literals, comma expressions, and ternary interactions.

Covers cross-feature interactions: compound literals in ternary branches,
comma expressions as function arguments, nested ternary expressions,
compound literals with designated initializers, ternary with side effects,
comma expression as for-loop increment, and compound literal assigned to pointer.
"""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRAlloc,
	IRBinOp,
	IRCall,
	IRCondJump,
	IRLoad,
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


def _get_func(ir, name: str):
	matches = [f for f in ir.functions if f.name == name]
	assert matches, f"Function '{name}' not found in IR"
	return matches[0]


# =============================================================================
# Compound literals in ternary branches
# =============================================================================


class TestCompoundLiteralInTernary:
	def test_struct_compound_literal_in_true_branch(self) -> None:
		"""Compound literal struct in true branch of ternary."""
		source = """
		struct Point { int x; int y; };
		int main(void) {
			int cond = 1;
			struct Point *p = cond ? &(struct Point){10, 20} : &(struct Point){30, 40};
			return p->x;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert len(allocs) >= 2, "Both branches should allocate compound literals"
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 1

	def test_struct_compound_literal_in_false_branch(self) -> None:
		"""When condition is false, false-branch compound literal is used."""
		source = """
		struct Pair { int a; int b; };
		int main(void) {
			int cond = 0;
			struct Pair *p = cond ? &(struct Pair){1, 2} : &(struct Pair){3, 4};
			return p->b;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_array_compound_literal_in_ternary(self) -> None:
		"""Array compound literal in ternary branches."""
		source = """
		int main(void) {
			int cond = 1;
			int *arr = cond ? (int[]){10, 20, 30} : (int[]){40, 50, 60};
			return arr[0];
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert len(allocs) >= 2

	def test_scalar_compound_literal_in_ternary(self) -> None:
		"""Scalar compound literal in ternary."""
		source = """
		int main(void) {
			int cond = 1;
			int *p = cond ? &(int){42} : &(int){99};
			return *p;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		stores = [i for i in func.body if isinstance(i, IRStore)]
		assert len(stores) >= 2, "Both compound literals should store values"


# =============================================================================
# Comma expressions as function arguments
# =============================================================================


class TestCommaAsArgument:
	def test_comma_expr_in_single_arg(self) -> None:
		"""f((a, b)) -- comma expression as sole argument, passes b."""
		source = """
		int identity(int x) { return x; }
		int main(void) {
			return identity((1, 42));
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		calls = [i for i in func.body if isinstance(i, IRCall)]
		assert len(calls) >= 1

	def test_comma_with_side_effect_in_arg(self) -> None:
		"""f((x = 5, x + 1)) -- side effect then value."""
		source = """
		int identity(int x) { return x; }
		int main(void) {
			int x = 0;
			return identity((x = 5, x + 1));
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_comma_exprs_in_multiple_args(self) -> None:
		"""f((a, b), (c, d)) -- comma expressions in multiple argument positions."""
		source = """
		int add(int a, int b) { return a + b; }
		int main(void) {
			return add((1, 10), (2, 20));
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		calls = [i for i in func.body if isinstance(i, IRCall)]
		assert len(calls) >= 1

	def test_comma_with_function_call_in_left(self) -> None:
		"""f((g(), value)) -- function call as side effect in comma."""
		source = """
		int counter;
		void inc(void) { counter = counter + 1; }
		int identity(int x) { return x; }
		int main(void) {
			counter = 0;
			return identity((inc(), 42));
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		calls = [i for i in func.body if isinstance(i, IRCall)]
		assert len(calls) >= 2, "Both inc() and identity() should be called"


# =============================================================================
# Nested ternary expressions (deep nesting edge cases)
# =============================================================================


class TestNestedTernaryEdge:
	def test_five_level_nested_ternary(self) -> None:
		"""Five levels deep ternary nesting."""
		source = """
		int main(void) {
			int a = 1, b = 1, c = 0, d = 1, e = 0;
			return a ? (b ? (c ? 1 : (d ? 2 : (e ? 3 : 4))) : 5) : 6;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 5

	def test_ternary_chain_right_associative(self) -> None:
		"""a ? 1 : b ? 2 : c ? 3 : 4 -- right-associative chain."""
		source = """
		int main(void) {
			int a = 0, b = 0, c = 1;
			return a ? 1 : b ? 2 : c ? 3 : 4;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 3

	def test_ternary_in_both_branches_of_ternary(self) -> None:
		"""Nested ternary in both true and false branches."""
		source = """
		int main(void) {
			int x = 0, y = 1, z = 0;
			return x ? (y ? 10 : 20) : (z ? 30 : 40);
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 3

	def test_ternary_with_arithmetic_in_condition(self) -> None:
		"""Ternary where condition is a complex arithmetic expression."""
		source = """
		int main(void) {
			int x = 3, y = 7;
			return (x * 2 - y + 1) ? 100 : 200;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		binops = [i for i in func.body if isinstance(i, IRBinOp)]
		assert len(binops) >= 2


# =============================================================================
# Compound literals with designated initializers
# =============================================================================


class TestCompoundLiteralDesignatedInit:
	def test_struct_designated_init_compound_literal(self) -> None:
		"""Compound literal with designated initializers."""
		source = """
		struct Rect { int x; int y; int w; int h; };
		int main(void) {
			struct Rect *r = &(struct Rect){ .x = 1, .y = 2, .w = 100, .h = 200 };
			return r->w;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert len(allocs) >= 1
		stores = [i for i in func.body if isinstance(i, IRStore)]
		assert len(stores) >= 4, "Four designated fields should produce four stores"

	def test_partial_designated_init(self) -> None:
		"""Compound literal with only some fields designated (rest zero-filled)."""
		source = """
		struct Vec3 { int x; int y; int z; };
		int main(void) {
			struct Vec3 *v = &(struct Vec3){ .z = 99 };
			return v->z;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert len(allocs) >= 1

	def test_designated_init_out_of_order(self) -> None:
		"""Designated initializers in non-declaration order."""
		source = """
		struct ABC { int a; int b; int c; };
		int main(void) {
			struct ABC *p = &(struct ABC){ .c = 30, .a = 10, .b = 20 };
			return p->a;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		stores = [i for i in func.body if isinstance(i, IRStore)]
		assert len(stores) >= 3

	def test_designated_init_in_ternary(self) -> None:
		"""Designated init compound literal in ternary branch."""
		source = """
		struct Pt { int x; int y; };
		int main(void) {
			int flag = 1;
			struct Pt *p = flag ? &(struct Pt){ .x = 5, .y = 10 } : &(struct Pt){ .y = 99, .x = 88 };
			return p->x;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert len(allocs) >= 2
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 1


# =============================================================================
# Ternary with side effects (++/-- in branches)
# =============================================================================


class TestTernarySideEffects:
	def test_postfix_increment_in_true_branch(self) -> None:
		"""cond ? x++ : y -- only x is incremented when cond is true."""
		source = """
		int main(void) {
			int x = 5, y = 10;
			int cond = 1;
			int result = cond ? x++ : y;
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_postfix_decrement_in_false_branch(self) -> None:
		"""cond ? x : y-- -- only y is decremented when cond is false."""
		source = """
		int main(void) {
			int x = 5, y = 10;
			int cond = 0;
			int result = cond ? x : y--;
			return y;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_both_branches_have_side_effects(self) -> None:
		"""cond ? x++ : y++ -- only one branch's side effect fires."""
		source = """
		int main(void) {
			int x = 0, y = 0;
			int cond = 1;
			int result = cond ? x++ : y++;
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 1

	def test_prefix_increment_in_ternary(self) -> None:
		"""cond ? ++x : --y -- prefix ops in ternary branches."""
		source = """
		int main(void) {
			int x = 5, y = 10;
			int cond = 1;
			int result = cond ? ++x : --y;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_nested_ternary_with_increments(self) -> None:
		"""Nested ternary with increments at each level."""
		source = """
		int main(void) {
			int a = 0, b = 0, c = 0;
			int x = 1, y = 0;
			int result = x ? (y ? a++ : b++) : c++;
			return b;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 2

	def test_assignment_side_effect_in_ternary(self) -> None:
		"""Assignment expressions as ternary branches."""
		source = """
		int main(void) {
			int x = 0, y = 0;
			int cond = 1;
			int result = cond ? (x = 42) : (y = 99);
			return x;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 1


# =============================================================================
# Comma expression as for-loop increment
# =============================================================================


class TestCommaInForIncrement:
	def test_two_var_increment(self) -> None:
		"""for (...; ...; i++, j--) -- two variables updated in increment."""
		source = """
		int main(void) {
			int i, j;
			int sum = 0;
			for (i = 0, j = 10; i < 5; i = i + 1, j = j - 1) {
				sum = sum + i + j;
			}
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_three_var_increment(self) -> None:
		"""Three variables updated in for-loop increment via comma."""
		source = """
		int main(void) {
			int i, j, k;
			int count = 0;
			for (i = 0, j = 0, k = 0; i < 3; i = i + 1, j = j + 2, k = k + 3) {
				count = count + 1;
			}
			return k;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_comma_with_function_call_in_increment(self) -> None:
		"""for (...; ...; inc(), i++) -- function call in increment."""
		source = """
		int counter;
		void inc(void) { counter = counter + 1; }
		int main(void) {
			int i;
			counter = 0;
			for (i = 0; i < 3; inc(), i = i + 1) {
			}
			return counter;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		calls = [i for i in func.body if isinstance(i, IRCall)]
		assert len(calls) >= 1

	def test_comma_increment_with_ternary(self) -> None:
		"""Ternary expression inside comma increment of for-loop."""
		source = """
		int main(void) {
			int i, step;
			int sum = 0;
			for (i = 0, step = 1; i < 10; step = (i < 5) ? 1 : 2, i = i + step) {
				sum = sum + 1;
			}
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# =============================================================================
# Compound literal assigned to pointer
# =============================================================================


class TestCompoundLiteralPointer:
	def test_struct_compound_literal_to_pointer(self) -> None:
		"""int *p = (int[]){1, 2, 3}; -- compound literal decays to pointer."""
		source = """
		int main(void) {
			int *p = (int[]){1, 2, 3};
			return p[1];
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert len(allocs) >= 1
		stores = [i for i in func.body if isinstance(i, IRStore)]
		assert len(stores) >= 3, "Three array elements should produce three stores"

	def test_struct_pointer_from_compound_literal(self) -> None:
		"""struct S *p = &(struct S){...}; -- address of compound literal."""
		source = """
		struct S { int a; int b; };
		int main(void) {
			struct S *p = &(struct S){10, 20};
			return p->a;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert len(allocs) >= 1

	def test_compound_literal_pointer_member_access(self) -> None:
		"""Access multiple members through compound literal pointer."""
		source = """
		struct Vec { int x; int y; int z; };
		int main(void) {
			struct Vec *v = &(struct Vec){1, 2, 3};
			return v->x + v->y + v->z;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		loads = [i for i in func.body if isinstance(i, IRLoad)]
		assert len(loads) >= 3, "Three member accesses produce three loads"

	def test_compound_literal_pointer_in_function_call(self) -> None:
		"""Passing compound literal pointer as function argument."""
		source = """
		struct Pair { int x; int y; };
		int sum_pair(struct Pair *p) { return p->x + p->y; }
		int main(void) {
			return sum_pair(&(struct Pair){10, 20});
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		calls = [i for i in func.body if isinstance(i, IRCall)]
		assert len(calls) >= 1
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert len(allocs) >= 1


# =============================================================================
# Cross-feature interactions: compound literal + comma + ternary combined
# =============================================================================


class TestCrossFeatureInteractions:
	def test_compound_literal_with_comma_init(self) -> None:
		"""Comma expression as value in compound literal initializer."""
		source = """
		struct Pt { int x; int y; };
		int main(void) {
			int a = 5;
			struct Pt *p = &(struct Pt){ (a = 10, a), (a = 20, a) };
			return p->x;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert len(allocs) >= 1

	def test_ternary_selecting_compound_literal_members(self) -> None:
		"""Use ternary to select which member to return from compound literal."""
		source = """
		struct Pair { int a; int b; };
		int main(void) {
			int sel = 1;
			struct Pair *p = &(struct Pair){10, 20};
			return sel ? p->a : p->b;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 1

	def test_comma_in_ternary_with_compound_literal(self) -> None:
		"""Comma expression inside ternary that uses compound literal."""
		source = """
		struct S { int val; };
		int main(void) {
			int flag = 1;
			int x = 0;
			struct S *p = flag ? (x = 1, &(struct S){x + 10}) : &(struct S){99};
			return p->val;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_compound_literal_in_comma_expression(self) -> None:
		"""Compound literal as left side of comma (evaluated for side effects)."""
		source = """
		int main(void) {
			int result = ((int){42}, 100);
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		ret = [i for i in func.body if isinstance(i, IRReturn)]
		assert len(ret) >= 1

	def test_nested_ternary_with_compound_literals_and_comma(self) -> None:
		"""Deeply nested combination of all three features."""
		source = """
		struct P { int x; int y; };
		int main(void) {
			int a = 1, b = 0;
			struct P *p = a
				? (b ? &(struct P){1, 2} : &(struct P){3, 4})
				: &(struct P){5, 6};
			return p->x;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 2
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert len(allocs) >= 3, "Three compound literals need three allocations"

	def test_for_loop_comma_increment_with_compound_literal_body(self) -> None:
		"""For loop with comma increment and compound literal usage in body."""
		source = """
		struct Pt { int x; int y; };
		int sum_pt(struct Pt *p) { return p->x + p->y; }
		int main(void) {
			int i, total;
			total = 0;
			for (i = 0; i < 3; i = i + 1, total = total + 1) {
				total = total + sum_pt(&(struct Pt){i, i * 2});
			}
			return total;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_ternary_with_postfix_and_compound_literal(self) -> None:
		"""Ternary with postfix increment and compound literal in same expression."""
		source = """
		struct Val { int n; };
		int main(void) {
			int x = 0;
			struct Val *v = (x++ ? &(struct Val){10} : &(struct Val){20});
			return v->n;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert len(allocs) >= 2
		cond_jumps = [i for i in func.body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 1

	def test_union_compound_literal_in_ternary(self) -> None:
		"""Union compound literal in ternary branches."""
		source = """
		union Data { int i; char c; };
		int main(void) {
			int flag = 1;
			union Data *d = flag ? &(union Data){42} : &(union Data){99};
			return d->i;
		}
		"""
		ir = _compile_to_ir(source)
		func = _get_func(ir, "main")
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		assert len(allocs) >= 2
