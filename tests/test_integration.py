"""End-to-end integration tests for the full compiler pipeline.

Each test passes C source through:
Preprocessor -> Lexer -> Parser -> SemanticAnalyzer -> IRGenerator -> CodeGenerator
and asserts the assembly output contains expected patterns.
"""

from compiler.codegen import CodeGenerator
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.preprocessor import Preprocessor
from compiler.semantic import SemanticAnalyzer


def compile_source(source: str) -> str:
	"""Run C source through the full compiler pipeline, returning assembly."""
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	ast = Parser(tokens).parse()
	errors = SemanticAnalyzer().analyze(ast)
	assert not errors, f"Semantic errors: {errors}"
	ir = IRGenerator().generate(ast)
	return CodeGenerator().generate(ir)


class TestReturnConstant:
	"""(1) Programs that return a constant value."""

	def test_return_zero(self) -> None:
		asm = compile_source("int main() { return 0; }")
		assert "main:" in asm
		assert ".globl main" in asm
		assert "ret" in asm
		assert "$0" in asm

	def test_return_literal(self) -> None:
		asm = compile_source("int main() { return 42; }")
		assert "main:" in asm
		assert "$42" in asm
		assert "ret" in asm

	def test_return_negative(self) -> None:
		source = """
		int main() {
			return -1;
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm
		# Constant-folded: -1 loaded directly as $-1 (no negq needed)
		assert "$-1" in asm or "negq" in asm
		assert "ret" in asm


class TestArithmeticExpressions:
	"""(2) Programs with arithmetic expressions."""

	def test_addition(self) -> None:
		asm = compile_source("int main() { return 2 + 3; }")
		assert "addq" in asm
		assert "ret" in asm

	def test_subtraction(self) -> None:
		asm = compile_source("int main() { return 10 - 4; }")
		assert "subq" in asm
		assert "ret" in asm

	def test_multiplication(self) -> None:
		asm = compile_source("int main() { return 3 * 7; }")
		assert "imulq" in asm
		assert "ret" in asm

	def test_division(self) -> None:
		asm = compile_source("int main() { return 10 / 2; }")
		assert "idivq" in asm
		assert "ret" in asm

	def test_modulo(self) -> None:
		asm = compile_source("int main() { return 10 % 3; }")
		assert "idivq" in asm
		assert "ret" in asm

	def test_compound_expression(self) -> None:
		asm = compile_source("int main() { return 2 + 3 * 4; }")
		assert "imulq" in asm
		assert "addq" in asm
		assert "ret" in asm

	def test_comparison(self) -> None:
		asm = compile_source("int main() { return 5 > 3; }")
		assert "cmpq" in asm
		assert "setg" in asm
		assert "ret" in asm


class TestFunctionWithParameters:
	"""(3) Functions with parameters."""

	def test_single_parameter(self) -> None:
		source = """
		int identity(int x) {
			return x;
		}
		"""
		asm = compile_source(source)
		assert "identity:" in asm
		assert ".globl identity" in asm
		# First param passed in %rdi
		assert "%rdi" in asm
		assert "ret" in asm

	def test_two_parameters_add(self) -> None:
		source = """
		int add(int a, int b) {
			return a + b;
		}
		"""
		asm = compile_source(source)
		assert "add:" in asm
		assert "%rdi" in asm
		assert "%rsi" in asm
		assert "addq" in asm
		assert "ret" in asm

	def test_three_parameters(self) -> None:
		source = """
		int sum3(int a, int b, int c) {
			return a + b + c;
		}
		"""
		asm = compile_source(source)
		assert "sum3:" in asm
		assert "%rdi" in asm
		assert "%rsi" in asm
		assert "%rdx" in asm
		assert "ret" in asm


class TestIfElseBranching:
	"""(4) If/else branching."""

	def test_if_only(self) -> None:
		source = """
		int abs_val(int x) {
			if (x < 0) {
				x = -x;
			}
			return x;
		}
		"""
		asm = compile_source(source)
		assert "abs_val:" in asm
		assert "cmpq" in asm
		assert "jne" in asm or "jmp" in asm
		assert "ret" in asm

	def test_if_else(self) -> None:
		source = """
		int max(int a, int b) {
			if (a > b) {
				return a;
			} else {
				return b;
			}
		}
		"""
		asm = compile_source(source)
		assert "max:" in asm
		assert "cmpq" in asm
		# Should have conditional and unconditional jumps
		assert "jne" in asm
		assert "jmp" in asm
		# Two return paths
		assert asm.count("ret") >= 2

	def test_nested_if(self) -> None:
		source = """
		int clamp(int x, int lo, int hi) {
			if (x < lo) {
				return lo;
			} else {
				if (x > hi) {
					return hi;
				} else {
					return x;
				}
			}
		}
		"""
		asm = compile_source(source)
		assert "clamp:" in asm
		# Multiple branches
		assert asm.count("cmpq") >= 2
		assert asm.count("ret") >= 3


class TestWhileLoop:
	"""(5) While loop."""

	def test_simple_while(self) -> None:
		source = """
		int countdown(int n) {
			while (n > 0) {
				n = n - 1;
			}
			return n;
		}
		"""
		asm = compile_source(source)
		assert "countdown:" in asm
		assert "cmpq" in asm
		# Loop needs backward jump
		assert "jmp" in asm
		assert "subq" in asm
		assert "ret" in asm

	def test_while_accumulator(self) -> None:
		source = """
		int sum_to(int n) {
			int s;
			s = 0;
			while (n > 0) {
				s = s + n;
				n = n - 1;
			}
			return s;
		}
		"""
		asm = compile_source(source)
		assert "sum_to:" in asm
		assert "addq" in asm
		assert "subq" in asm
		assert "jmp" in asm
		assert "ret" in asm


class TestForLoop:
	"""(6) For loop."""

	def test_simple_for(self) -> None:
		source = """
		int sum_n(int n) {
			int s;
			int i;
			s = 0;
			for (i = 1; i <= n; i = i + 1) {
				s = s + i;
			}
			return s;
		}
		"""
		asm = compile_source(source)
		assert "sum_n:" in asm
		assert "cmpq" in asm
		assert "addq" in asm
		assert "jmp" in asm
		assert "ret" in asm

	def test_for_with_multiplication(self) -> None:
		source = """
		int factorial(int n) {
			int result;
			int i;
			result = 1;
			for (i = 2; i <= n; i = i + 1) {
				result = result * i;
			}
			return result;
		}
		"""
		asm = compile_source(source)
		assert "factorial:" in asm
		assert "imulq" in asm
		assert "jmp" in asm
		assert "ret" in asm


class TestPointerDereference:
	"""(7) Pointer dereference."""

	def test_pointer_deref_read(self) -> None:
		source = """
		int deref(int *p) {
			return *p;
		}
		"""
		asm = compile_source(source)
		assert "deref:" in asm
		assert "(%r" in asm
		assert "ret" in asm

	def test_pointer_deref_write(self) -> None:
		source = """
		void set_val(int *p, int v) {
			*p = v;
		}
		"""
		asm = compile_source(source)
		assert "set_val:" in asm
		assert "%rdi" in asm
		assert "%rsi" in asm

	def test_pointer_param_passthrough(self) -> None:
		"""Pointer parameters can be received and returned without dereferencing."""
		source = """
		int *identity_ptr(int *p) {
			return p;
		}
		"""
		asm = compile_source(source)
		assert "identity_ptr:" in asm
		assert "%rdi" in asm
		assert "ret" in asm


class TestArrayAccess:
	"""(8) Array access."""

	def test_array_local(self) -> None:
		source = """
		int first() {
			int arr[3];
			arr[0] = 10;
			return arr[0];
		}
		"""
		asm = compile_source(source)
		assert "first:" in asm
		assert "subq" in asm  # Stack allocation for array
		assert "ret" in asm

	def test_array_indexing(self) -> None:
		source = """
		int second() {
			int arr[5];
			arr[0] = 1;
			arr[1] = 2;
			return arr[1];
		}
		"""
		asm = compile_source(source)
		assert "second:" in asm
		assert "ret" in asm


class TestNestedFunctionCalls:
	"""(9) Nested function calls."""

	def test_call_in_return(self) -> None:
		source = """
		int double_val(int x) {
			return x + x;
		}
		int quad(int x) {
			return double_val(double_val(x));
		}
		"""
		asm = compile_source(source)
		assert "double_val:" in asm
		assert "quad:" in asm
		# quad should call double_val twice
		lines = [line.strip() for line in asm.splitlines()]
		call_count = sum(1 for line in lines if line == "call double_val")
		assert call_count == 2

	def test_call_in_expression(self) -> None:
		source = """
		int inc(int x) {
			return x + 1;
		}
		int add_two(int x) {
			return inc(inc(x));
		}
		"""
		asm = compile_source(source)
		assert "inc:" in asm
		assert "add_two:" in asm
		assert "call inc" in asm


class TestMultipleFunctions:
	"""(10) Multiple functions calling each other."""

	def test_two_functions(self) -> None:
		source = """
		int square(int x) {
			return x * x;
		}
		int sum_of_squares(int a, int b) {
			return square(a) + square(b);
		}
		"""
		asm = compile_source(source)
		assert "square:" in asm
		assert "sum_of_squares:" in asm
		assert ".globl square" in asm
		assert ".globl sum_of_squares" in asm
		assert "call square" in asm
		assert "imulq" in asm

	def test_chain_of_calls(self) -> None:
		source = """
		int add1(int x) {
			return x + 1;
		}
		int add2(int x) {
			return add1(add1(x));
		}
		int add4(int x) {
			return add2(add2(x));
		}
		"""
		asm = compile_source(source)
		assert "add1:" in asm
		assert "add2:" in asm
		assert "add4:" in asm
		assert "call add1" in asm
		assert "call add2" in asm

	def test_multiple_functions_complete_program(self) -> None:
		source = """
		int multiply(int a, int b) {
			return a * b;
		}
		int add(int a, int b) {
			return a + b;
		}
		int compute(int x, int y) {
			int prod;
			int sum;
			prod = multiply(x, y);
			sum = add(x, y);
			return add(prod, sum);
		}
		"""
		asm = compile_source(source)
		assert "multiply:" in asm
		# Note: 'add:' also matches 'addq', so check for the label specifically
		assert "\nadd:" in asm or asm.startswith("add:")
		assert "compute:" in asm
		assert "call multiply" in asm
		assert "call add" in asm
		assert "imulq" in asm
		assert "addq" in asm


class TestPipelineStructure:
	"""Verify general assembly structure properties."""

	def test_section_header(self) -> None:
		asm = compile_source("int main() { return 0; }")
		assert asm.startswith(".section .text\n")

	def test_function_prologue(self) -> None:
		asm = compile_source("int main() { return 0; }")
		assert "pushq %rbp" in asm
		assert "movq %rsp, %rbp" in asm

	def test_function_epilogue(self) -> None:
		asm = compile_source("int main() { return 0; }")
		assert "movq %rbp, %rsp" in asm
		assert "popq %rbp" in asm
		assert "ret" in asm

	def test_void_function(self) -> None:
		source = """
		void noop(int x) {
			x = x + 1;
		}
		"""
		asm = compile_source(source)
		assert "noop:" in asm

	def test_local_variable(self) -> None:
		source = """
		int main() {
			int x;
			x = 5;
			return x;
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm
		assert "$5" in asm
		assert "ret" in asm


class TestLogicalShortCircuit:
	"""(11) Logical && and || with short-circuit evaluation."""

	def test_and_both_true(self) -> None:
		source = """
		int f() {
			return 1 && 2;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "jne" in asm
		assert "ret" in asm

	def test_and_left_false(self) -> None:
		source = """
		int f() {
			return 0 && 1;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "jne" in asm
		assert "$0" in asm

	def test_or_left_true(self) -> None:
		source = """
		int f() {
			return 1 || 0;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "jne" in asm

	def test_or_both_false(self) -> None:
		source = """
		int f() {
			return 0 || 0;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "$0" in asm

	def test_and_in_if(self) -> None:
		source = """
		int f(int a, int b) {
			if (a > 0 && b > 0) {
				return 1;
			}
			return 0;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "cmpq" in asm
		assert "ret" in asm

	def test_or_in_if(self) -> None:
		source = """
		int f(int a, int b) {
			if (a > 0 || b > 0) {
				return 1;
			}
			return 0;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "cmpq" in asm
		assert "ret" in asm


class TestPointerReadWrite:
	"""(12) Pointer read and write through dereference."""

	def test_address_of_and_deref(self) -> None:
		source = """
		int f() {
			int x;
			int *p;
			x = 42;
			p = &x;
			return *p;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "movl (%rax), %eax" in asm
		assert "ret" in asm

	def test_write_through_pointer(self) -> None:
		source = """
		void set(int *p, int v) {
			*p = v;
		}
		"""
		asm = compile_source(source)
		assert "set:" in asm
		assert "(%rax)" in asm

	def test_pointer_read_param(self) -> None:
		source = """
		int deref(int *p) {
			return *p;
		}
		"""
		asm = compile_source(source)
		assert "deref:" in asm
		assert "movl (%rax), %eax" in asm
		assert "ret" in asm


class TestPrefixIncrementDecrement:
	"""(13) Prefix ++ and -- operators."""

	def test_prefix_increment(self) -> None:
		source = """
		int f() {
			int x;
			x = 5;
			++x;
			return x;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "addq" in asm
		assert "ret" in asm

	def test_prefix_decrement(self) -> None:
		source = """
		int f() {
			int x;
			x = 10;
			--x;
			return x;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "subq" in asm
		assert "ret" in asm

	def test_prefix_increment_in_loop(self) -> None:
		source = """
		int f() {
			int i;
			int s;
			s = 0;
			for (i = 0; i < 5; ++i) {
				s = s + i;
			}
			return s;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "addq" in asm
		assert "jmp" in asm
		assert "ret" in asm


class TestCompoundAssignment:
	"""(14) Compound assignment operators: +=, -=, *=, /=, %=."""

	def test_plus_assign(self) -> None:
		source = """
		int f() {
			int x;
			x = 10;
			x += 5;
			return x;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "addq" in asm
		assert "ret" in asm

	def test_minus_assign(self) -> None:
		source = """
		int f() {
			int x;
			x = 10;
			x -= 3;
			return x;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "subq" in asm
		assert "ret" in asm

	def test_star_assign(self) -> None:
		source = """
		int f() {
			int x;
			x = 4;
			x *= 3;
			return x;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "imulq" in asm
		assert "ret" in asm

	def test_slash_assign(self) -> None:
		source = """
		int f() {
			int x;
			x = 20;
			x /= 4;
			return x;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "idivq" in asm
		assert "ret" in asm

	def test_percent_assign(self) -> None:
		source = """
		int f() {
			int x;
			x = 17;
			x %= 5;
			return x;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "idivq" in asm
		# Modulo result comes from %rdx
		assert "movq %rdx, %rax" in asm
		assert "ret" in asm

	def test_compound_assign_in_loop(self) -> None:
		source = """
		int f() {
			int sum;
			int i;
			sum = 0;
			for (i = 1; i <= 5; i = i + 1) {
				sum += i;
			}
			return sum;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "addq" in asm
		assert "jmp" in asm
		assert "ret" in asm


class TestSwitchCaseIntegration:
	"""(15) Switch/case with multiple cases, default, fallthrough, and break."""

	def test_switch_with_cases_and_default(self) -> None:
		source = """
		int f(int x) {
			switch (x) {
				case 1: return 10;
				case 2: return 20;
				default: return 0;
			}
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		# Case comparisons generate == checks
		assert "sete" in asm
		# Conditional jumps for case matching
		assert "jne" in asm
		# Multiple return paths
		assert asm.count("ret") >= 3

	def test_switch_with_break(self) -> None:
		source = """
		int f(int x) {
			int result;
			result = 0;
			switch (x) {
				case 1:
					result = 10;
					break;
				case 2:
					result = 20;
					break;
				default:
					result = 99;
					break;
			}
			return result;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "sete" in asm
		# break generates jumps to switch_end
		assert asm.count("jmp") >= 3
		assert "ret" in asm

	def test_switch_without_default(self) -> None:
		source = """
		int f(int x) {
			switch (x) {
				case 0: return 1;
				case 1: return 2;
			}
			return -1;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "sete" in asm
		assert "jne" in asm
		assert "ret" in asm

	def test_switch_fallthrough(self) -> None:
		"""Cases without break fall through to the next case body."""
		source = """
		int f(int x) {
			int r;
			r = 0;
			switch (x) {
				case 1:
					r = r + 10;
				case 2:
					r = r + 20;
					break;
				default:
					r = 99;
					break;
			}
			return r;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		# Case labels present
		assert "case" in asm.lower() or "sete" in asm
		assert "ret" in asm


class TestTernaryExpression:
	"""(16) Ternary conditional expressions."""

	def test_ternary_in_return(self) -> None:
		source = "int f(int x) { return x > 0 ? 1 : -1; }"
		asm = compile_source(source)
		assert "f:" in asm
		# Condition comparison
		assert "cmpq" in asm
		# Conditional jump for ternary
		assert "jne" in asm
		# Both branches present: -1 may be constant-folded
		assert "$1" in asm
		assert "$-1" in asm or "negq" in asm
		assert "ret" in asm

	def test_ternary_in_assignment(self) -> None:
		source = """
		int f(int a, int b) {
			int max;
			max = a > b ? a : b;
			return max;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "cmpq" in asm
		assert "jne" in asm
		assert "jmp" in asm
		assert "ret" in asm

	def test_nested_ternary(self) -> None:
		source = """
		int classify(int x) {
			return x > 0 ? 1 : (x < 0 ? -1 : 0);
		}
		"""
		asm = compile_source(source)
		assert "classify:" in asm
		# Multiple ternary branches produce multiple cmpq and jne
		assert asm.count("jne") >= 2
		assert "ret" in asm

	def test_ternary_with_expressions(self) -> None:
		source = """
		int f(int a, int b) {
			return a > b ? a + b : a - b;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "addq" in asm
		assert "subq" in asm
		assert "ret" in asm


class TestStructMemberAccess:
	"""(17) Struct definition, dot access, and member offset computation."""

	def test_struct_first_field(self) -> None:
		source = """
		struct Point {
			int x;
			int y;
		};
		int f() {
			struct Point p;
			return p.x;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		# First field offset is 0
		assert "$0" in asm
		# Load from computed address (INT width)
		assert "movl (%rax), %eax" in asm
		assert "ret" in asm

	def test_struct_second_field_offset(self) -> None:
		source = """
		struct Point {
			int x;
			int y;
		};
		int f() {
			struct Point p;
			return p.y;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		# Second field: offset 4 (int x is 4 bytes)
		assert "$4" in asm
		assert "addq" in asm
		assert "movl (%rax), %eax" in asm
		assert "ret" in asm

	def test_struct_arrow_access(self) -> None:
		source = """
		struct Pair {
			int a;
			int b;
		};
		int f(struct Pair *p) {
			return p->a;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		# Arrow access on first field, offset 0
		assert "$0" in asm
		assert "movl (%rax), %eax" in asm
		assert "ret" in asm

	def test_struct_arrow_second_field(self) -> None:
		source = """
		struct Pair {
			int a;
			int b;
		};
		int f(struct Pair *p) {
			return p->b;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		# Second field offset = 4
		assert "$4" in asm
		assert "addq" in asm
		assert "movl (%rax), %eax" in asm
		assert "ret" in asm

	def test_struct_allocation(self) -> None:
		source = """
		struct RGB {
			int r;
			int g;
			int b;
		};
		int f() {
			struct RGB c;
			return c.r;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		# Struct allocated on stack via subq
		assert "subq" in asm
		assert "ret" in asm


class TestPostfixOperators:
	"""(18) Postfix ++ and -- with old-value-returned semantics."""

	def test_postfix_increment_returns_old(self) -> None:
		source = """
		int f() {
			int x;
			int y;
			x = 5;
			y = x++;
			return y;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		# Increment generates addq with $1
		assert "addq" in asm
		assert "$1" in asm
		assert "ret" in asm

	def test_postfix_decrement(self) -> None:
		source = """
		int f() {
			int x;
			x = 10;
			x--;
			return x;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "subq" in asm
		assert "$1" in asm
		assert "ret" in asm

	def test_postfix_in_loop(self) -> None:
		source = """
		int f() {
			int i;
			int s;
			s = 0;
			for (i = 0; i < 5; i++) {
				s = s + i;
			}
			return s;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "addq" in asm
		assert "jmp" in asm
		assert "ret" in asm

	def test_postfix_on_array_element(self) -> None:
		source = """
		int f() {
			int arr[3];
			arr[0] = 10;
			arr[0]++;
			return arr[0];
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		# Array element postfix: load, increment, store
		assert "addq" in asm
		assert "ret" in asm

	def test_postfix_increment_and_decrement(self) -> None:
		source = """
		int f() {
			int a;
			int b;
			a = 5;
			b = 10;
			a++;
			b--;
			return a + b;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "addq" in asm
		assert "subq" in asm
		assert "ret" in asm


class TestBitwiseOperators:
	"""(19) Bitwise operators: &, |, ^, <<, >>."""

	def test_bitwise_and(self) -> None:
		source = "int f() { return 0xFF & 0x0F; }"
		asm = compile_source(source)
		assert "f:" in asm
		assert "andq" in asm
		assert "ret" in asm

	def test_bitwise_or(self) -> None:
		source = "int f() { return 0xF0 | 0x0F; }"
		asm = compile_source(source)
		assert "f:" in asm
		assert "orq" in asm
		assert "ret" in asm

	def test_bitwise_xor(self) -> None:
		source = "int f() { return 0xFF ^ 0x0F; }"
		asm = compile_source(source)
		assert "f:" in asm
		assert "xorq" in asm
		assert "ret" in asm

	def test_left_shift(self) -> None:
		source = "int f() { return 1 << 4; }"
		asm = compile_source(source)
		assert "f:" in asm
		assert "salq" in asm
		assert "ret" in asm

	def test_right_shift(self) -> None:
		source = "int f() { return 128 >> 3; }"
		asm = compile_source(source)
		assert "f:" in asm
		assert "sarq" in asm
		assert "ret" in asm

	def test_bitwise_in_expression(self) -> None:
		source = """
		int f(int a, int b) {
			return (a & b) | (a ^ b);
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "andq" in asm
		assert "orq" in asm
		assert "xorq" in asm
		assert "ret" in asm

	def test_shift_in_expression(self) -> None:
		source = """
		int f(int x) {
			return (x << 2) + (x >> 1);
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "salq" in asm
		assert "sarq" in asm
		assert "addq" in asm
		assert "ret" in asm


class TestNestedControlFlow:
	"""(20) Nested control flow: loops inside loops, if inside switch, etc."""

	def test_for_inside_while(self) -> None:
		source = """
		int f() {
			int total;
			int i;
			int j;
			total = 0;
			i = 0;
			while (i < 3) {
				for (j = 0; j < 3; j = j + 1) {
					total = total + 1;
				}
				i = i + 1;
			}
			return total;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		# Multiple loop constructs produce multiple cmpq and jmp
		assert asm.count("cmpq") >= 2
		assert asm.count("jmp") >= 2
		assert "addq" in asm
		assert "ret" in asm

	def test_if_inside_for(self) -> None:
		source = """
		int count_positive(int a, int b, int c) {
			int arr[3];
			int count;
			int i;
			arr[0] = a;
			arr[1] = b;
			arr[2] = c;
			count = 0;
			for (i = 0; i < 3; i = i + 1) {
				if (arr[i] > 0) {
					count = count + 1;
				}
			}
			return count;
		}
		"""
		asm = compile_source(source)
		assert "count_positive:" in asm
		assert "cmpq" in asm
		assert "jne" in asm
		assert "jmp" in asm
		assert "ret" in asm

	def test_nested_if_else_chain(self) -> None:
		source = """
		int classify(int x) {
			if (x > 100) {
				return 3;
			} else {
				if (x > 10) {
					return 2;
				} else {
					if (x > 0) {
						return 1;
					} else {
						return 0;
					}
				}
			}
		}
		"""
		asm = compile_source(source)
		assert "classify:" in asm
		assert asm.count("cmpq") >= 3
		assert asm.count("ret") >= 4
		assert "jne" in asm

	def test_while_with_break_and_continue(self) -> None:
		source = """
		int f() {
			int i;
			int sum;
			i = 0;
			sum = 0;
			while (i < 100) {
				i = i + 1;
				if (i > 10) {
					break;
				}
				sum = sum + i;
			}
			return sum;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		# break generates jmp to loop end
		assert asm.count("jmp") >= 2
		assert "cmpq" in asm
		assert "addq" in asm
		assert "ret" in asm


class TestDoWhileLoop:
	"""(21) Do-while loops with basic usage, break, and continue."""

	def test_basic_do_while(self) -> None:
		source = """
		int f() {
			int x;
			x = 0;
			do {
				x = x + 1;
			} while (x < 5);
			return x;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		# Body label appears before condition check
		assert "do_body" in asm
		assert "do_cond" in asm
		# Condition check jumps back to body
		assert "jne" in asm
		assert "addq" in asm
		assert "ret" in asm

	def test_do_while_single_iteration(self) -> None:
		"""Do-while always executes body at least once."""
		source = """
		int f() {
			int x;
			x = 100;
			do {
				x = x + 1;
			} while (x < 5);
			return x;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "do_body" in asm
		assert "do_cond" in asm
		assert "addq" in asm
		assert "ret" in asm

	def test_do_while_with_break(self) -> None:
		source = """
		int f() {
			int x;
			x = 0;
			do {
				x = x + 1;
				if (x > 3) {
					break;
				}
			} while (x < 10);
			return x;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "do_body" in asm
		# break jumps to do_end
		assert "do_end" in asm
		assert asm.count("jmp") >= 2
		assert "ret" in asm

	def test_do_while_accumulator(self) -> None:
		source = """
		int f(int n) {
			int sum;
			sum = 0;
			do {
				sum = sum + n;
				n = n - 1;
			} while (n > 0);
			return sum;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "addq" in asm
		assert "subq" in asm
		assert "do_body" in asm
		assert "do_cond" in asm
		assert "ret" in asm
