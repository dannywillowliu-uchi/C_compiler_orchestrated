"""Integration tests: full pipeline with optimizer enabled produces correct assembly."""

from compiler.__main__ import compile_source
from compiler.codegen import CodeGenerator
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.optimizer import IROptimizer
from compiler.parser import Parser
from compiler.preprocessor import Preprocessor
from compiler.semantic import SemanticAnalyzer


def _compile(source: str) -> str:
	"""Compile C source with optimizations enabled."""
	return compile_source(source, optimize=True)


def _compile_no_opt(source: str) -> str:
	"""Compile C source without optimizations."""
	return compile_source(source, optimize=False)


def _compile_ir_optimized(source: str) -> str:
	"""Compile with IR optimizer but without peephole, for cleaner IR-level checks."""
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	ast = Parser(tokens).parse()
	errors = SemanticAnalyzer().analyze(ast)
	assert not errors, f"Semantic errors: {errors}"
	ir = IRGenerator().generate(ast)
	ir = IROptimizer().optimize(ir)
	return CodeGenerator().generate(ir)


# -- Constant Folding Integration --


class TestConstantFoldingIntegration:
	"""Verify constant folding produces correct values through the full pipeline."""

	def test_simple_constant_arithmetic(self) -> None:
		source = """
		int main() {
			return 3 + 4;
		}
		"""
		asm = _compile(source)
		assert "main:" in asm
		assert "ret" in asm
		# 3 + 4 should be folded to 7; the assembly should contain $7
		assert "$7" in asm

	def test_nested_constant_folding(self) -> None:
		source = """
		int main() {
			return (2 * 3) + (10 - 4);
		}
		"""
		asm = _compile(source)
		# 2*3=6, 10-4=6, 6+6=12
		assert "$12" in asm

	def test_constant_fold_with_variable(self) -> None:
		source = """
		int compute(int x) {
			return x + (5 * 6);
		}
		"""
		asm = _compile(source)
		# 5 * 6 should fold to 30
		assert "$30" in asm
		# x is still a parameter, so there should be register/stack usage
		assert "compute:" in asm
		assert "ret" in asm

	def test_constant_comparison_folding(self) -> None:
		source = """
		int main() {
			int x;
			if (3 > 2) {
				x = 100;
			} else {
				x = 200;
			}
			return x;
		}
		"""
		asm = _compile(source)
		# 3 > 2 folds to 1 (true), should still compile correctly
		assert "main:" in asm
		assert "$100" in asm
		assert "ret" in asm

	def test_constant_fold_division(self) -> None:
		source = """
		int main() {
			return 100 / 5;
		}
		"""
		asm = _compile(source)
		# 100 / 5 = 20
		assert "$20" in asm
		# No division instruction should remain
		assert "idivq" not in asm

	def test_constant_fold_modulo(self) -> None:
		source = """
		int main() {
			return 17 % 5;
		}
		"""
		asm = _compile(source)
		# 17 % 5 = 2
		assert "$2" in asm

	def test_constant_fold_bitwise(self) -> None:
		source = """
		int main() {
			return (0xFF & 0x0F) | 0x30;
		}
		"""
		asm = _compile(source)
		# 0xFF & 0x0F = 0x0F = 15, 15 | 0x30 = 0x3F = 63
		assert "$63" in asm

	def test_constant_fold_shift(self) -> None:
		source = """
		int main() {
			return 1 << 10;
		}
		"""
		asm = _compile(source)
		# 1 << 10 = 1024
		assert "$1024" in asm


# -- Dead Code Elimination Integration --


class TestDeadCodeEliminationIntegration:
	"""Verify dead code elimination doesn't break program logic."""

	def test_unreachable_after_return(self) -> None:
		source = """
		int foo() {
			return 42;
			int x = 10;
			return x;
		}
		"""
		asm = _compile(source)
		assert "foo:" in asm
		assert "$42" in asm
		assert "ret" in asm

	def test_dead_variable_eliminated(self) -> None:
		source = """
		int main() {
			int unused = 99;
			int result = 42;
			return result;
		}
		"""
		asm = _compile(source)
		assert "main:" in asm
		assert "$42" in asm
		assert "ret" in asm

	def test_used_variable_preserved(self) -> None:
		source = """
		int main() {
			int a = 10;
			int b = 20;
			int c = a + b;
			return c;
		}
		"""
		asm = _compile(source)
		assert "main:" in asm
		# 10 + 20 should be folded to 30
		assert "$30" in asm
		assert "ret" in asm

	def test_dead_code_in_branch(self) -> None:
		source = """
		int pick(int x) {
			if (x > 0) {
				return 1;
			}
			return 0;
		}
		"""
		asm = _compile(source)
		assert "pick:" in asm
		assert "$1" in asm
		# movq $0 is now optimized to xorq %reg, %reg by peephole
		assert "$0" in asm or "xorq" in asm
		# Should have conditional logic (cmpq $0 may become testq)
		assert "cmpq" in asm or "testq" in asm
		assert "ret" in asm

	def test_loop_body_preserved(self) -> None:
		source = """
		int sum_to_n(int n) {
			int total = 0;
			int i;
			for (i = 1; i <= n; i++) {
				total = total + i;
			}
			return total;
		}
		"""
		asm = _compile(source)
		assert "sum_to_n:" in asm
		# Loop structure must be preserved
		assert "jmp" in asm
		assert "addq" in asm
		assert "ret" in asm

	def test_multi_return_paths(self) -> None:
		source = """
		int classify(int x) {
			if (x > 0) {
				return 1;
			}
			if (x < 0) {
				return -1;
			}
			return 0;
		}
		"""
		asm = _compile(source)
		assert "classify:" in asm
		ret_count = asm.count("ret")
		assert ret_count >= 3


# -- Copy Propagation Integration --


class TestCopyPropagationIntegration:
	"""Verify copy propagation doesn't lose side effects or break correctness."""

	def test_simple_copy_propagation(self) -> None:
		source = """
		int main() {
			int a = 5;
			int b = a;
			return b;
		}
		"""
		asm = _compile(source)
		assert "main:" in asm
		assert "$5" in asm
		assert "ret" in asm

	def test_copy_chain(self) -> None:
		source = """
		int main() {
			int a = 42;
			int b = a;
			int c = b;
			int d = c;
			return d;
		}
		"""
		asm = _compile(source)
		assert "main:" in asm
		# All copies should propagate back to 42
		assert "$42" in asm
		assert "ret" in asm

	def test_copy_propagation_with_arithmetic(self) -> None:
		source = """
		int main() {
			int x = 10;
			int y = x;
			int z = y + 5;
			return z;
		}
		"""
		asm = _compile(source)
		assert "main:" in asm
		# x=10, y=x=10, z = 10+5 = 15 (constant folded)
		assert "$15" in asm

	def test_copy_propagation_preserves_function_call(self) -> None:
		source = """
		int compute(int x) {
			return x * 2;
		}
		int main() {
			int a = 7;
			int b = a;
			return compute(b);
		}
		"""
		asm = _compile(source)
		assert "compute:" in asm
		assert "main:" in asm
		assert "call compute" in asm
		assert "$7" in asm

	def test_copy_propagation_with_store_side_effect(self) -> None:
		source = """
		int main() {
			int arr[2];
			int val = 42;
			int copy_val = val;
			arr[0] = copy_val;
			return arr[0];
		}
		"""
		asm = _compile(source)
		assert "main:" in asm
		assert "$42" in asm
		assert "ret" in asm

	def test_copy_propagation_across_branches(self) -> None:
		source = """
		int test(int flag) {
			int result;
			if (flag) {
				result = 10;
			} else {
				result = 20;
			}
			int output = result;
			return output;
		}
		"""
		asm = _compile(source)
		assert "test:" in asm
		assert "$10" in asm
		assert "$20" in asm
		assert "ret" in asm


# -- Strength Reduction Integration --


class TestStrengthReductionIntegration:
	"""Verify strength reduction produces equivalent results."""

	def test_multiply_by_power_of_two(self) -> None:
		source = """
		int double_it(int x) {
			return x * 2;
		}
		"""
		asm = _compile(source)
		assert "double_it:" in asm
		# x * 2 should become x << 1
		assert "salq" in asm
		# No multiply instruction
		assert "imulq" not in asm

	def test_multiply_by_four(self) -> None:
		source = """
		int quad(int x) {
			return x * 4;
		}
		"""
		asm = _compile(source)
		assert "quad:" in asm
		# x * 4 should become x << 2
		assert "salq" in asm
		assert "imulq" not in asm

	def test_multiply_by_eight(self) -> None:
		source = """
		int oct(int x) {
			return x * 8;
		}
		"""
		asm = _compile(source)
		assert "oct:" in asm
		# x * 8 should become x << 3
		assert "salq" in asm
		assert "imulq" not in asm

	def test_multiply_by_one_eliminated(self) -> None:
		source = """
		int identity(int x) {
			return x * 1;
		}
		"""
		asm = _compile(source)
		assert "identity:" in asm
		# x * 1 = x, no multiply needed
		assert "imulq" not in asm

	def test_multiply_by_zero(self) -> None:
		source = """
		int zero(int x) {
			return x * 0;
		}
		"""
		asm = _compile(source)
		assert "zero:" in asm
		# movq $0 is now optimized to xorq %reg, %reg by peephole
		assert "$0" in asm or "xorq" in asm
		assert "imulq" not in asm

	def test_add_zero_eliminated(self) -> None:
		source = """
		int same(int x) {
			return x + 0;
		}
		"""
		asm = _compile(source)
		assert "same:" in asm
		# x + 0 = x, should be a copy
		assert "addq" not in asm or "$0" not in asm

	def test_subtract_zero_eliminated(self) -> None:
		source = """
		int same(int x) {
			return x - 0;
		}
		"""
		asm = _compile(source)
		assert "same:" in asm

	def test_non_power_of_two_multiply_preserved(self) -> None:
		source = """
		int triple(int x) {
			return x * 3;
		}
		"""
		asm = _compile(source)
		assert "triple:" in asm
		# 3 is not a power of 2, so imulq should remain
		assert "imulq" in asm

	def test_strength_reduction_in_loop(self) -> None:
		source = """
		int scale_sum(int n) {
			int total = 0;
			int i;
			for (i = 0; i < n; i++) {
				total = total + i * 4;
			}
			return total;
		}
		"""
		asm = _compile(source)
		assert "scale_sum:" in asm
		# i * 4 should be strength-reduced to i << 2
		assert "salq" in asm
		assert "ret" in asm


# -- Loop Optimization Integration --


class TestLoopOptimizationIntegration:
	"""Verify loop optimizations don't break loop semantics."""

	def test_simple_while_loop(self) -> None:
		source = """
		int count(int n) {
			int i = 0;
			while (i < n) {
				i = i + 1;
			}
			return i;
		}
		"""
		asm = _compile(source)
		assert "count:" in asm
		assert "jmp" in asm
		assert "cmpq" in asm
		assert "ret" in asm

	def test_for_loop_preserved(self) -> None:
		source = """
		int sum(int n) {
			int total = 0;
			int i;
			for (i = 0; i < n; i++) {
				total = total + i;
			}
			return total;
		}
		"""
		asm = _compile(source)
		assert "sum:" in asm
		assert "addq" in asm
		assert "jmp" in asm
		assert "ret" in asm

	def test_nested_loops(self) -> None:
		source = """
		int nested(int n) {
			int total = 0;
			int i;
			int j;
			for (i = 0; i < n; i++) {
				for (j = 0; j < n; j++) {
					total = total + 1;
				}
			}
			return total;
		}
		"""
		asm = _compile(source)
		assert "nested:" in asm
		# Should have multiple jump/loop structures
		jmp_count = asm.count("jmp")
		assert jmp_count >= 2
		assert "ret" in asm

	def test_do_while_loop(self) -> None:
		source = """
		int count_down(int n) {
			int x = n;
			do {
				x = x - 1;
			} while (x > 0);
			return x;
		}
		"""
		asm = _compile(source)
		assert "count_down:" in asm
		assert "subq" in asm or "$1" in asm
		assert "ret" in asm


# -- Optimizer Correctness: Both Paths Produce Valid Assembly --


class TestOptimizerCorrectnessComparison:
	"""Ensure optimized and unoptimized paths both produce structurally valid assembly."""

	def test_both_paths_have_function_labels(self) -> None:
		source = """
		int add(int a, int b) {
			return a + b;
		}
		int main() {
			return add(3, 4);
		}
		"""
		opt_asm = _compile(source)
		noopt_asm = _compile_no_opt(source)
		for asm in (opt_asm, noopt_asm):
			assert "add:" in asm
			assert "main:" in asm
			assert ".globl add" in asm
			assert ".globl main" in asm

	def test_both_paths_have_prologue_epilogue(self) -> None:
		source = """
		int main() {
			return 42;
		}
		"""
		opt_asm = _compile(source)
		noopt_asm = _compile_no_opt(source)
		for asm in (opt_asm, noopt_asm):
			assert "pushq %rbp" in asm
			assert "movq %rsp, %rbp" in asm
			assert "popq %rbp" in asm
			assert "ret" in asm

	def test_recursive_function_optimized(self) -> None:
		source = """
		int factorial(int n) {
			if (n <= 1) {
				return 1;
			}
			return n * factorial(n - 1);
		}
		int main() {
			return factorial(5);
		}
		"""
		asm = _compile(source)
		assert "factorial:" in asm
		assert "call factorial" in asm
		assert "imulq" in asm
		assert "$5" in asm
		assert "ret" in asm

	def test_pointer_operations_preserved(self) -> None:
		source = """
		int deref(int *p) {
			return *p;
		}
		int main() {
			int x = 99;
			return deref(&x);
		}
		"""
		asm = _compile(source)
		assert "deref:" in asm
		assert "main:" in asm
		assert "call deref" in asm
		assert "ret" in asm

	def test_array_operations_preserved(self) -> None:
		source = """
		int main() {
			int arr[3];
			arr[0] = 10;
			arr[1] = 20;
			arr[2] = 30;
			return arr[1];
		}
		"""
		asm = _compile(source)
		assert "main:" in asm
		assert "$10" in asm
		assert "$20" in asm
		assert "$30" in asm
		assert "ret" in asm


# -- Mixed Optimization Interactions --


class TestMixedOptimizations:
	"""Test interactions between multiple optimizer passes."""

	def test_fold_then_propagate(self) -> None:
		source = """
		int main() {
			int a = 3 + 4;
			int b = a;
			return b;
		}
		"""
		asm = _compile(source)
		# 3+4 folds to 7, then a=7 propagates through b
		assert "$7" in asm

	def test_fold_and_strength_reduce(self) -> None:
		source = """
		int scale(int x) {
			int factor = 2;
			return x * factor;
		}
		"""
		asm = _compile(source)
		assert "scale:" in asm
		# factor=2 propagated, then x*2 -> x<<1
		assert "salq" in asm
		assert "imulq" not in asm

	def test_complex_expression_optimization(self) -> None:
		source = """
		int compute(int x) {
			int a = x * 2;
			int b = x * 2;
			return a + b;
		}
		"""
		asm = _compile_ir_optimized(source)
		assert "compute:" in asm
		# CSE should eliminate the duplicate x*2 (now x<<1 after strength reduction)
		# Count shift instructions - should only have one
		sal_count = asm.count("salq")
		assert sal_count <= 1
		assert "ret" in asm

	def test_constant_in_loop_hoisted_and_folded(self) -> None:
		source = """
		int compute(int n) {
			int result = 0;
			int i;
			for (i = 0; i < n; i++) {
				result = result + (2 + 3);
			}
			return result;
		}
		"""
		asm = _compile(source)
		assert "compute:" in asm
		# 2+3=5 should be constant folded
		assert "$5" in asm
		assert "ret" in asm

	def test_switch_with_optimizable_body(self) -> None:
		source = """
		int dispatch(int cmd) {
			int result;
			switch (cmd) {
				case 0:
					result = 10 + 20;
					break;
				case 1:
					result = 3 * 4;
					break;
				default:
					result = 0;
					break;
			}
			return result;
		}
		"""
		asm = _compile(source)
		assert "dispatch:" in asm
		# 10+20=30, 3*4=12
		assert "$30" in asm
		assert "$12" in asm
		assert "ret" in asm

	def test_ternary_with_constants(self) -> None:
		source = """
		int pick(int flag) {
			return flag ? (5 + 5) : (3 * 3);
		}
		"""
		asm = _compile(source)
		assert "pick:" in asm
		# 5+5=10, 3*3=9
		assert "$10" in asm
		assert "$9" in asm
		assert "ret" in asm

	def test_multiple_functions_all_optimized(self) -> None:
		source = """
		int double_val(int x) {
			return x * 2;
		}
		int add_constants() {
			return 10 + 20 + 30;
		}
		int main() {
			return double_val(5) + add_constants();
		}
		"""
		asm = _compile(source)
		assert "double_val:" in asm
		assert "add_constants:" in asm
		assert "main:" in asm
		# double_val: x*2 -> x<<1
		assert "salq" in asm
		# add_constants: 10+20+30 = 60
		assert "$60" in asm

	def test_struct_member_access_with_optimizer(self) -> None:
		source = """
		struct Point {
			int x;
			int y;
		};
		int sum_point() {
			struct Point p;
			p.x = 10;
			p.y = 20;
			return p.x + p.y;
		}
		int main() {
			return sum_point();
		}
		"""
		asm = _compile(source)
		assert "sum_point:" in asm
		assert "main:" in asm
		assert "addq" in asm
		assert "ret" in asm

	def test_enum_with_optimizer(self) -> None:
		source = """
		enum Flags { A = 1, B = 2, C = 4 };
		int combine() {
			return A + B + C;
		}
		int main() {
			return combine();
		}
		"""
		asm = _compile(source)
		assert "combine:" in asm
		# 1 + 2 + 4 = 7
		assert "$7" in asm
