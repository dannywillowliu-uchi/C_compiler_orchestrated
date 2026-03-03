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
		assert "negq" in asm
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
		assert "movq (%rax), %rax" in asm
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
		assert "movq (%rax), %rax" in asm
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
