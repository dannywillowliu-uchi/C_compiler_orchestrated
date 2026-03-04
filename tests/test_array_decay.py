"""Tests for array-to-pointer decay, array parameter semantics, and pointer arithmetic type rules."""

import pytest

from compiler.ast_nodes import FunctionDecl, ParamDecl
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer, SemanticError


def parse(source: str):
	return Parser.from_source(source).parse()


def parse_and_analyze(source: str):
	program = parse(source)
	analyzer = SemanticAnalyzer()
	analyzer.analyze(program)
	return program, analyzer


# ---------------------------------------------------------------------------
# Array parameter syntax: int a[] -> int *a
# ---------------------------------------------------------------------------


class TestArrayParameterSyntax:
	def test_array_param_becomes_pointer(self):
		"""void foo(int a[]) should be parsed as void foo(int *a)."""
		prog = parse("void foo(int a[]) { int x; x = a[0]; }")
		func = prog.declarations[0]
		assert isinstance(func, FunctionDecl)
		param = func.params[0]
		assert isinstance(param, ParamDecl)
		assert param.type_spec.base_type == "int"
		assert param.type_spec.pointer_count == 1

	def test_array_param_with_size_becomes_pointer(self):
		"""void foo(int a[10]) should be parsed as void foo(int *a)."""
		prog = parse("void foo(int a[10]) { int x; x = a[0]; }")
		func = prog.declarations[0]
		param = func.params[0]
		assert isinstance(param, ParamDecl)
		assert param.type_spec.base_type == "int"
		assert param.type_spec.pointer_count == 1

	def test_char_array_param(self):
		"""void foo(char s[]) -> void foo(char *s)."""
		prog = parse("void foo(char s[]) { int x; x = s[0]; }")
		func = prog.declarations[0]
		param = func.params[0]
		assert param.type_spec.base_type == "char"
		assert param.type_spec.pointer_count == 1

	def test_pointer_array_param(self):
		"""void foo(int *a[]) -> void foo(int **a)."""
		prog = parse("void foo(int *a[]) { int *x; x = a[0]; }")
		func = prog.declarations[0]
		param = func.params[0]
		assert param.type_spec.base_type == "int"
		assert param.type_spec.pointer_count == 2

	def test_array_param_semantic_pass(self):
		"""Array parameter functions should pass semantic analysis."""
		source = """
		int foo(int a[]) {
			return a[0];
		}
		int main() {
			int arr[5];
			int x;
			x = foo(arr);
			return 0;
		}
		"""
		parse_and_analyze(source)

	def test_multiple_params_with_array(self):
		"""Multiple params where one is array."""
		prog = parse("int foo(int n, int a[], int *p) { return n; }")
		func = prog.declarations[0]
		assert func.params[0].type_spec.pointer_count == 0
		assert func.params[1].type_spec.pointer_count == 1
		assert func.params[2].type_spec.pointer_count == 1


# ---------------------------------------------------------------------------
# Array-to-pointer decay in expression contexts
# ---------------------------------------------------------------------------


class TestArrayDecay:
	def test_array_decays_to_pointer_in_assignment(self):
		"""Assigning an array to a pointer should pass semantic analysis."""
		source = """
		int main() {
			int arr[5];
			int *p;
			p = arr;
			return 0;
		}
		"""
		parse_and_analyze(source)

	def test_array_decays_in_function_arg(self):
		"""Passing an array where a pointer is expected should work."""
		source = """
		int take_ptr(int *p) { return p[0]; }
		int main() {
			int arr[10];
			int x;
			x = take_ptr(arr);
			return 0;
		}
		"""
		parse_and_analyze(source)

	def test_array_decays_in_arithmetic(self):
		"""Array + int should be valid (pointer arithmetic)."""
		source = """
		int main() {
			int arr[5];
			int *p;
			p = arr + 2;
			return 0;
		}
		"""
		parse_and_analyze(source)

	def test_array_decays_in_comparison(self):
		"""Array can be compared with pointer."""
		source = """
		int main() {
			int arr[5];
			int *p;
			p = arr;
			int same;
			same = (arr == p);
			return same;
		}
		"""
		parse_and_analyze(source)

	def test_array_decay_char_array(self):
		"""char array decays to char*."""
		source = """
		int take_str(char *s) { return s[0]; }
		int main() {
			char msg[20];
			int x;
			x = take_str(msg);
			return 0;
		}
		"""
		parse_and_analyze(source)

	def test_sizeof_inhibits_decay(self):
		"""sizeof(array) should not cause decay (semantic analysis should not error)."""
		source = """
		int main() {
			int arr[5];
			int size;
			size = sizeof(arr);
			return size;
		}
		"""
		parse_and_analyze(source)

	def test_addressof_inhibits_decay(self):
		"""&array should not cause decay and should produce a pointer type."""
		source = """
		int main() {
			int arr[5];
			int *p;
			p = &arr;
			return 0;
		}
		"""
		parse_and_analyze(source)

	def test_array_subscript_still_works(self):
		"""Array subscript should still work with decay."""
		source = """
		int main() {
			int arr[5];
			arr[0] = 42;
			int val;
			val = arr[2];
			return val;
		}
		"""
		parse_and_analyze(source)

	def test_array_in_ternary(self):
		"""Array in ternary expression should decay."""
		source = """
		int main() {
			int a[5];
			int b[5];
			int *p;
			p = 1 ? a : b;
			return 0;
		}
		"""
		parse_and_analyze(source)


# ---------------------------------------------------------------------------
# Pointer arithmetic type rules
# ---------------------------------------------------------------------------


class TestPointerArithmetic:
	def test_pointer_plus_int(self):
		"""ptr + int should be valid and produce pointer type."""
		source = """
		int main() {
			int arr[10];
			int *p;
			p = arr;
			int *q;
			q = p + 3;
			return 0;
		}
		"""
		parse_and_analyze(source)

	def test_int_plus_pointer(self):
		"""int + ptr should be valid and produce pointer type."""
		source = """
		int main() {
			int *p;
			int *q;
			q = 2 + p;
			return 0;
		}
		"""
		parse_and_analyze(source)

	def test_pointer_minus_int(self):
		"""ptr - int should be valid and produce pointer type."""
		source = """
		int main() {
			int *p;
			int *q;
			q = p - 1;
			return 0;
		}
		"""
		parse_and_analyze(source)

	def test_pointer_minus_pointer(self):
		"""ptr - ptr should be valid and produce integer type."""
		source = """
		int main() {
			int arr[10];
			int *p;
			int *q;
			p = arr;
			q = arr + 5;
			int diff;
			diff = q - p;
			return diff;
		}
		"""
		parse_and_analyze(source)

	def test_pointer_plus_pointer_error(self):
		"""ptr + ptr should produce a semantic error."""
		source = """
		int main() {
			int *p;
			int *q;
			int *r;
			r = p + q;
			return 0;
		}
		"""
		with pytest.raises(SemanticError, match="addition of two pointers"):
			parse_and_analyze(source)

	def test_int_minus_pointer_error(self):
		"""int - ptr should produce a semantic error."""
		source = """
		int main() {
			int *p;
			int result;
			result = 5 - p;
			return 0;
		}
		"""
		with pytest.raises(SemanticError, match="subtraction of pointer from integer"):
			parse_and_analyze(source)

	def test_pointer_multiply_error(self):
		"""ptr * int should produce a semantic error."""
		source = """
		int main() {
			int *p;
			int *q;
			q = p * 2;
			return 0;
		}
		"""
		with pytest.raises(SemanticError, match="pointer type not allowed"):
			parse_and_analyze(source)

	def test_pointer_divide_error(self):
		"""ptr / int should produce a semantic error."""
		source = """
		int main() {
			int *p;
			int result;
			result = p / 4;
			return 0;
		}
		"""
		with pytest.raises(SemanticError, match="pointer type not allowed"):
			parse_and_analyze(source)

	def test_pointer_modulo_error(self):
		"""ptr % int should produce a semantic error."""
		source = """
		int main() {
			int *p;
			int result;
			result = p % 4;
			return 0;
		}
		"""
		with pytest.raises(SemanticError, match="pointer type not allowed"):
			parse_and_analyze(source)

	def test_pointer_bitwise_error(self):
		"""ptr & int should produce a semantic error."""
		source = """
		int main() {
			int *p;
			int result;
			result = p & 0xFF;
			return 0;
		}
		"""
		with pytest.raises(SemanticError, match="pointer type not allowed"):
			parse_and_analyze(source)

	def test_pointer_comparison_ok(self):
		"""Pointer comparisons should be allowed."""
		source = """
		int main() {
			int *p;
			int *q;
			int result;
			result = (p < q);
			result = (p == q);
			result = (p != q);
			result = (p >= q);
			return result;
		}
		"""
		parse_and_analyze(source)


# ---------------------------------------------------------------------------
# Combined scenarios
# ---------------------------------------------------------------------------


class TestArrayDecayCombined:
	def test_array_to_function_and_back(self):
		"""Pass array to function expecting pointer, use pointer arithmetic."""
		source = """
		int sum(int *arr, int n) {
			int total;
			total = 0;
			int i;
			for (i = 0; i < n; i = i + 1) {
				total = total + arr[i];
			}
			return total;
		}
		int main() {
			int data[5];
			data[0] = 1;
			data[1] = 2;
			data[2] = 3;
			data[3] = 4;
			data[4] = 5;
			int result;
			result = sum(data, 5);
			return result;
		}
		"""
		parse_and_analyze(source)

	def test_array_param_function_call(self):
		"""Function declared with array param syntax called with array arg."""
		source = """
		int first(int a[]) {
			return a[0];
		}
		int main() {
			int nums[3];
			nums[0] = 42;
			int val;
			val = first(nums);
			return val;
		}
		"""
		parse_and_analyze(source)

	def test_pointer_arithmetic_with_decayed_array(self):
		"""Use pointer arithmetic on a decayed array."""
		source = """
		int main() {
			int arr[5];
			int *end;
			end = arr + 5;
			int *begin;
			begin = arr;
			int len;
			len = end - begin;
			return len;
		}
		"""
		parse_and_analyze(source)

	def test_array_in_return_as_pointer(self):
		"""Return an array name (which decays to pointer) from a function."""
		source = """
		int *get_data() {
			int data[10];
			return data;
		}
		int main() {
			int *p;
			p = get_data();
			return 0;
		}
		"""
		parse_and_analyze(source)

	def test_string_literal_as_char_pointer(self):
		"""String literals are already char*, should work with char* params."""
		source = """
		int take_str(char *s) { return s[0]; }
		int main() {
			int x;
			x = take_str("hello");
			return 0;
		}
		"""
		parse_and_analyze(source)

	def test_prototype_with_array_param(self):
		"""Function prototype with array parameter."""
		source = """
		int process(int data[], int len);
		int process(int data[], int len) {
			return data[0];
		}
		int main() {
			int arr[5];
			int x;
			x = process(arr, 5);
			return 0;
		}
		"""
		parse_and_analyze(source)

	def test_ir_generation_array_decay(self):
		"""Array decay should work through IR generation without errors."""
		from compiler.ir_gen import IRGenerator

		source = """
		int take(int *p) { return p[0]; }
		int main() {
			int arr[5];
			int x;
			x = take(arr);
			int *p;
			p = arr;
			p = arr + 1;
			return 0;
		}
		"""
		program = parse(source)
		analyzer = SemanticAnalyzer()
		analyzer.analyze(program)
		gen = IRGenerator()
		ir_prog = gen.generate(program)
		assert len(ir_prog.functions) == 2
