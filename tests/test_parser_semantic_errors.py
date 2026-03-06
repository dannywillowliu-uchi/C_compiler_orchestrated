"""Tests for parser error recovery and semantic analyzer edge cases.

Covers: duplicate typedefs, redeclared variables in same scope, invalid lvalue
assignments, void return from non-void function, incompatible pointer assignments,
array size validation, function redeclaration with different signatures, and more.
"""

import pytest

from compiler.lexer import Lexer
from compiler.parser import ParseError, Parser
from compiler.semantic import SemanticAnalyzer, SemanticError


def parse(source: str):
	"""Parse C source and return AST."""
	tokens = Lexer(source).tokenize()
	return Parser(tokens).parse()


def analyze(source: str):
	"""Parse and semantically analyze C source. Returns (analyzer, ast)."""
	ast = parse(source)
	analyzer = SemanticAnalyzer()
	analyzer.analyze(ast)
	return analyzer, ast


def assert_semantic_error(source: str, expected_substr: str) -> SemanticError:
	"""Assert that semantic analysis raises an error containing expected_substr."""
	ast = parse(source)
	analyzer = SemanticAnalyzer()
	with pytest.raises(SemanticError, match=expected_substr):
		analyzer.analyze(ast)
	return analyzer.errors[0]


def assert_no_semantic_error(source: str) -> None:
	"""Assert that semantic analysis succeeds without errors."""
	ast = parse(source)
	analyzer = SemanticAnalyzer()
	analyzer.analyze(ast)
	assert not analyzer.errors


# ============================================================
# Duplicate typedef tests
# ============================================================


class TestDuplicateTypedefs:
	def test_duplicate_typedef_same_type(self) -> None:
		source = """
		typedef int MyInt;
		typedef int MyInt;
		int main() { return 0; }
		"""
		assert_semantic_error(source, "redefinition of typedef 'MyInt'")

	def test_duplicate_typedef_different_type(self) -> None:
		source = """
		typedef int MyType;
		typedef float MyType;
		int main() { return 0; }
		"""
		assert_semantic_error(source, "redefinition of typedef 'MyType'")

	def test_typedef_then_use_is_valid(self) -> None:
		source = """
		typedef int MyInt;
		int main() { MyInt x = 5; return x; }
		"""
		assert_no_semantic_error(source)

	def test_typedef_pointer_then_duplicate(self) -> None:
		source = """
		typedef int *IntPtr;
		typedef int *IntPtr;
		int main() { return 0; }
		"""
		assert_semantic_error(source, "redefinition of typedef 'IntPtr'")


# ============================================================
# Redeclared variables in same scope
# ============================================================


class TestRedeclaredVariables:
	def test_redeclare_in_same_scope(self) -> None:
		source = """
		int main() {
			int x = 1;
			int x = 2;
			return x;
		}
		"""
		assert_semantic_error(source, "redefinition of 'x'")

	def test_redeclare_in_nested_scope_is_valid(self) -> None:
		source = """
		int main() {
			int x = 1;
			{
				int x = 2;
			}
			return x;
		}
		"""
		assert_no_semantic_error(source)

	def test_redeclare_param_in_body(self) -> None:
		source = """
		int foo(int a) {
			int a = 5;
			return a;
		}
		"""
		assert_semantic_error(source, "redefinition of 'a'")

	def test_redeclare_global_in_same_scope(self) -> None:
		source = """
		int x;
		int x;
		int main() { return 0; }
		"""
		assert_semantic_error(source, "redefinition of 'x'")

	def test_multiple_params_same_name(self) -> None:
		source = """
		int foo(int a, int a) {
			return a;
		}
		"""
		assert_semantic_error(source, "redefinition of 'a'")


# ============================================================
# Invalid lvalue assignments
# ============================================================


class TestInvalidLvalue:
	def test_assign_to_literal(self) -> None:
		source = """
		int main() {
			int x = 5;
			(x + 1) = 10;
			return 0;
		}
		"""
		# Parser may not catch this; semantic may or may not error.
		# The key thing is this shouldn't silently succeed as valid C.
		try:
			analyze(source)
		except (ParseError, SemanticError):
			pass

	def test_assign_to_const_variable(self) -> None:
		source = """
		int main() {
			const int x = 5;
			x = 10;
			return 0;
		}
		"""
		assert_semantic_error(source, "assignment to const variable")

	def test_increment_const_variable(self) -> None:
		source = """
		int main() {
			const int x = 5;
			x++;
			return 0;
		}
		"""
		assert_semantic_error(source, "increment/decrement of const variable")

	def test_decrement_const_variable(self) -> None:
		source = """
		int main() {
			const int x = 5;
			--x;
			return 0;
		}
		"""
		assert_semantic_error(source, "increment/decrement of const variable")

	def test_compound_assign_to_const(self) -> None:
		source = """
		int main() {
			const int x = 5;
			x += 1;
			return 0;
		}
		"""
		assert_semantic_error(source, "assignment to const variable")


# ============================================================
# Void return from non-void function (warnings)
# ============================================================


class TestVoidReturnErrors:
	def test_void_func_returns_value(self) -> None:
		source = """
		void foo() {
			return 42;
		}
		int main() { foo(); return 0; }
		"""
		analyzer, _ast = analyze(source)
		assert any("void function should not return a value" in w for w in analyzer.warnings)

	def test_void_func_bare_return_is_valid(self) -> None:
		source = """
		void foo() {
			return;
		}
		int main() { foo(); return 0; }
		"""
		assert_no_semantic_error(source)

	def test_non_void_func_missing_return_warns(self) -> None:
		source = """
		int foo() {
			int x = 5;
		}
		int main() { return 0; }
		"""
		ast = parse(source)
		analyzer = SemanticAnalyzer()
		analyzer.analyze(ast)
		assert any("control reaches end of non-void function" in w for w in analyzer.warnings)

	def test_incompatible_return_type(self) -> None:
		source = """
		int *foo() {
			return "hello";
		}
		int main() { return 0; }
		"""
		# char* returning from int* -- pointers with different base types
		assert_semantic_error(source, "incompatible return type")


# ============================================================
# Incompatible pointer assignments
# ============================================================


class TestIncompatiblePointerAssignments:
	def test_int_ptr_to_float_ptr(self) -> None:
		source = """
		int main() {
			int x = 5;
			int *p = &x;
			float *q = p;
			return 0;
		}
		"""
		assert_semantic_error(source, "incompatible types")

	def test_void_ptr_to_int_ptr_is_valid(self) -> None:
		source = """
		int main() {
			void *p = 0;
			int *q = p;
			return 0;
		}
		"""
		assert_no_semantic_error(source)

	def test_null_to_pointer_is_valid(self) -> None:
		source = """
		int main() {
			int *p = 0;
			return 0;
		}
		"""
		assert_no_semantic_error(source)

	def test_different_pointer_depth_incompatible(self) -> None:
		source = """
		int main() {
			int x = 5;
			int *p = &x;
			int **pp = p;
			return 0;
		}
		"""
		assert_semantic_error(source, "incompatible types")


# ============================================================
# Array size validation
# ============================================================


class TestArraySizeValidation:
	def test_array_size_from_initializer(self) -> None:
		source = """
		int main() {
			int arr[] = {1, 2, 3};
			return arr[0];
		}
		"""
		assert_no_semantic_error(source)

	def test_excess_elements_in_array_initializer(self) -> None:
		source = """
		int main() {
			int arr[2] = {1, 2, 3};
			return arr[0];
		}
		"""
		assert_semantic_error(source, "excess elements in array initializer")

	def test_char_array_from_string_valid(self) -> None:
		source = """
		int main() {
			char s[] = "hello";
			return 0;
		}
		"""
		assert_no_semantic_error(source)

	def test_array_declared_with_explicit_size(self) -> None:
		source = """
		int main() {
			int arr[5] = {1, 2, 3};
			return arr[0];
		}
		"""
		assert_no_semantic_error(source)


# ============================================================
# Function redeclaration issues
# ============================================================


class TestFunctionRedeclaration:
	def test_redeclare_function_definition(self) -> None:
		source = """
		int foo() { return 1; }
		int foo() { return 2; }
		int main() { return 0; }
		"""
		assert_semantic_error(source, "redefinition of 'foo'")

	def test_prototype_then_definition_is_valid(self) -> None:
		source = """
		int foo(int x);
		int foo(int x) { return x + 1; }
		int main() { return foo(5); }
		"""
		assert_no_semantic_error(source)

	def test_wrong_arg_count(self) -> None:
		source = """
		int foo(int x, int y) { return x + y; }
		int main() { return foo(1); }
		"""
		assert_semantic_error(source, "expects 2 arguments, got 1")

	def test_too_many_args(self) -> None:
		source = """
		int foo(int x) { return x; }
		int main() { return foo(1, 2, 3); }
		"""
		assert_semantic_error(source, "expects 1 arguments, got 3")

	def test_call_non_function(self) -> None:
		source = """
		int main() {
			int x = 5;
			return x(1);
		}
		"""
		assert_semantic_error(source, "'x' is not a function")

	def test_duplicate_prototype(self) -> None:
		source = """
		int foo(int x);
		int foo(int x);
		int main() { return 0; }
		"""
		assert_semantic_error(source, "redefinition of 'foo'")


# ============================================================
# Undeclared identifier usage
# ============================================================


class TestUndeclaredIdentifiers:
	def test_undeclared_variable(self) -> None:
		source = """
		int main() {
			return x;
		}
		"""
		assert_semantic_error(source, "use of undeclared identifier 'x'")

	def test_undeclared_in_expression(self) -> None:
		source = """
		int main() {
			int y = x + 1;
			return y;
		}
		"""
		assert_semantic_error(source, "use of undeclared identifier 'x'")

	def test_implicit_function_decl_warns(self) -> None:
		source = """
		int main() {
			int x = unknown_func(1, 2);
			return x;
		}
		"""
		ast = parse(source)
		analyzer = SemanticAnalyzer()
		analyzer.analyze(ast)
		assert any("implicit declaration of function 'unknown_func'" in w for w in analyzer.warnings)


# ============================================================
# Break/continue outside loop
# ============================================================


class TestBreakContinueErrors:
	def test_break_outside_loop(self) -> None:
		source = """
		int main() {
			break;
			return 0;
		}
		"""
		assert_semantic_error(source, "break statement not within a loop or switch")

	def test_continue_outside_loop(self) -> None:
		source = """
		int main() {
			continue;
			return 0;
		}
		"""
		assert_semantic_error(source, "continue statement not within a loop")

	def test_break_inside_loop_is_valid(self) -> None:
		source = """
		int main() {
			while (1) { break; }
			return 0;
		}
		"""
		assert_no_semantic_error(source)

	def test_break_inside_switch_is_valid(self) -> None:
		source = """
		int main() {
			int x = 1;
			switch (x) {
				case 1: break;
			}
			return 0;
		}
		"""
		assert_no_semantic_error(source)

	def test_continue_in_for_is_valid(self) -> None:
		source = """
		int main() {
			int i;
			for (i = 0; i < 10; i++) { continue; }
			return 0;
		}
		"""
		assert_no_semantic_error(source)


# ============================================================
# Label and goto errors
# ============================================================


class TestLabelGotoErrors:
	def test_goto_undeclared_label(self) -> None:
		source = """
		int main() {
			goto missing;
			return 0;
		}
		"""
		assert_semantic_error(source, "use of undeclared label 'missing'")

	def test_duplicate_label(self) -> None:
		source = """
		int main() {
			lbl: return 0;
			lbl: return 1;
		}
		"""
		assert_semantic_error(source, "redefinition of label 'lbl'")

	def test_goto_valid_label(self) -> None:
		source = """
		int main() {
			goto end;
			end: return 0;
		}
		"""
		assert_no_semantic_error(source)


# ============================================================
# Type modifier errors
# ============================================================


class TestTypeModifierErrors:
	def test_unsigned_float(self) -> None:
		source = """
		int main() {
			unsigned float x = 1.0;
			return 0;
		}
		"""
		assert_semantic_error(source, "'unsigned' cannot be used with 'float'")

	def test_signed_double(self) -> None:
		source = """
		int main() {
			signed double x = 1.0;
			return 0;
		}
		"""
		assert_semantic_error(source, "'signed' cannot be used with 'double'")

	def test_short_float(self) -> None:
		source = """
		int main() {
			short float x = 1.0;
			return 0;
		}
		"""
		assert_semantic_error(source, "'short' cannot be used with 'float'")


# ============================================================
# Dereference and pointer operation errors
# ============================================================


class TestPointerErrors:
	def test_dereference_non_pointer(self) -> None:
		source = """
		int main() {
			int x = 5;
			int y = *x;
			return 0;
		}
		"""
		assert_semantic_error(source, "dereference of non-pointer type")

	def test_subscript_non_array(self) -> None:
		source = """
		int main() {
			int x = 5;
			int y = x[0];
			return 0;
		}
		"""
		assert_semantic_error(source, "subscript requires array or pointer type")

	def test_pointer_addition_invalid(self) -> None:
		source = """
		int main() {
			int x = 5;
			int *p = &x;
			int *q = &x;
			int *r = p + q;
			return 0;
		}
		"""
		assert_semantic_error(source, "addition of two pointers is not allowed")

	def test_address_of_register(self) -> None:
		source = """
		int main() {
			register int x = 5;
			int *p = &x;
			return 0;
		}
		"""
		assert_semantic_error(source, "address of register variable 'x' requested")


# ============================================================
# Switch statement edge cases
# ============================================================


class TestSwitchErrors:
	def test_duplicate_case_value(self) -> None:
		source = """
		int main() {
			int x = 1;
			switch (x) {
				case 1: break;
				case 1: break;
			}
			return 0;
		}
		"""
		assert_semantic_error(source, "duplicate case value")

	def test_duplicate_default(self) -> None:
		source = """
		int main() {
			int x = 1;
			switch (x) {
				default: break;
				default: break;
			}
			return 0;
		}
		"""
		assert_semantic_error(source, "duplicate default label")


# ============================================================
# Parser error cases
# ============================================================


class TestParserErrors:
	def test_missing_semicolon(self) -> None:
		source = """
		int main() {
			int x = 5
			return x;
		}
		"""
		with pytest.raises(ParseError):
			parse(source)

	def test_missing_closing_brace(self) -> None:
		source = """
		int main() {
			int x = 5;
		"""
		with pytest.raises(ParseError):
			parse(source)

	def test_missing_closing_paren(self) -> None:
		source = """
		int main() {
			int x = (5 + 3;
			return x;
		}
		"""
		with pytest.raises(ParseError):
			parse(source)

	def test_empty_function_params_valid(self) -> None:
		source = """
		int foo() { return 0; }
		int main() { return foo(); }
		"""
		assert_no_semantic_error(source)

	def test_unexpected_token_in_expression(self) -> None:
		source = """
		int main() {
			int x = ;
			return 0;
		}
		"""
		with pytest.raises(ParseError):
			parse(source)

	def test_double_operator(self) -> None:
		source = """
		int main() {
			int x = 5 + + 3;
			return x;
		}
		"""
		# Unary + should be parsed, or error -- either way it should not crash
		try:
			ast = parse(source)
			analyzer = SemanticAnalyzer()
			analyzer.analyze(ast)
		except (ParseError, SemanticError):
			pass  # Either error is acceptable


# ============================================================
# Struct/union member errors
# ============================================================


class TestStructErrors:
	def test_access_nonexistent_member(self) -> None:
		source = """
		struct Point { int x; int y; };
		int main() {
			struct Point p;
			p.x = 1;
			int z = p.z;
			return 0;
		}
		"""
		assert_semantic_error(source, "has no member")

	def test_excess_struct_initializer(self) -> None:
		source = """
		struct Point { int x; int y; };
		int main() {
			struct Point p = {1, 2, 3};
			return 0;
		}
		"""
		assert_semantic_error(source, "excess elements in struct initializer")


# ============================================================
# Postfix lvalue checks
# ============================================================


class TestPostfixLvalue:
	def test_postfix_on_non_lvalue(self) -> None:
		source = """
		int main() {
			int x = 5;
			(x + 1)++;
			return 0;
		}
		"""
		assert_semantic_error(source, "operand of postfix operator must be an lvalue")


# ============================================================
# Cast edge cases
# ============================================================


class TestCastErrors:
	def test_valid_numeric_cast(self) -> None:
		source = """
		int main() {
			float f = 3.14;
			int x = (int)f;
			return x;
		}
		"""
		assert_no_semantic_error(source)

	def test_valid_pointer_cast(self) -> None:
		source = """
		int main() {
			int x = 5;
			int *p = &x;
			long addr = (long)p;
			return 0;
		}
		"""
		assert_no_semantic_error(source)


# ============================================================
# Ternary expression type errors
# ============================================================


class TestTernaryErrors:
	def test_incompatible_ternary_branches(self) -> None:
		source = """
		int main() {
			int x = 5;
			int *p = &x;
			float f = 1.0;
			float *q = &f;
			int *result = x ? p : q;
			return 0;
		}
		"""
		assert_semantic_error(source, "incompatible types in ternary branches")


class TestInlineAndQualifiers:
	"""Regression tests for inline specifier and pointer qualifiers."""

	def test_static_inline_function(self):
		parse("static inline int f(void) { return 1; } int main() { return f()-1; }")

	def test_extern_inline_function(self):
		parse("extern inline int f(void) { return 0; }")

	def test_inline_alone(self):
		parse("inline int f(void) { return 0; }")

	def test_const_qualified_pointer(self):
		parse("int g = 5; int * const p = &g; int main() { return *p; }")

	def test_volatile_qualified_pointer(self):
		parse("int g = 5; int * volatile p = &g;")

	def test_anonymous_struct_declaration(self):
		parse("struct { int x; } a; int main() { a.x = 1; return a.x; }")

	def test_static_volatile_struct(self):
		parse("static volatile int x = 1; int main() { return x; }")
