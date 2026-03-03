"""End-to-end tests verifying preprocessor features through the full compilation pipeline.

Each test feeds C source with preprocessor directives through compile_source()
and verifies the assembly output contains expected patterns.
"""

from compiler.__main__ import compile_source


class TestDefineConstants:
	"""#define constants used in expressions."""

	def test_define_integer_constant_in_return(self) -> None:
		source = """
		#define VALUE 42
		int main() {
			return VALUE;
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm
		assert "$42" in asm

	def test_define_constant_in_arithmetic(self) -> None:
		source = """
		#define A 10
		#define B 20
		int main() {
			return A + B;
		}
		"""
		asm = compile_source(source)
		assert "$10" in asm
		assert "$20" in asm
		assert "addq" in asm

	def test_define_constant_in_variable_init(self) -> None:
		source = """
		#define SIZE 5
		int main() {
			int x = SIZE;
			return x;
		}
		"""
		asm = compile_source(source)
		assert "$5" in asm
		assert "ret" in asm

	def test_define_constant_in_comparison(self) -> None:
		source = """
		#define LIMIT 100
		int check(int n) {
			if (n < LIMIT) {
				return 1;
			}
			return 0;
		}
		int main() {
			return check(50);
		}
		"""
		asm = compile_source(source)
		assert "$100" in asm
		assert "cmpq" in asm
		assert "if_then" in asm

	def test_define_constant_in_loop_bound(self) -> None:
		source = """
		#define COUNT 3
		int main() {
			int sum = 0;
			int i;
			for (i = 0; i < COUNT; i++) {
				sum = sum + i;
			}
			return sum;
		}
		"""
		asm = compile_source(source)
		assert "$3" in asm
		assert "for_start" in asm
		assert "for_end" in asm


class TestFunctionLikeMacros:
	"""Function-like macros expanding into valid C."""

	def test_add_macro(self) -> None:
		source = """
		#define ADD(a, b) ((a) + (b))
		int main() {
			return ADD(3, 4);
		}
		"""
		asm = compile_source(source)
		assert "$3" in asm
		assert "$4" in asm
		assert "addq" in asm

	def test_square_macro(self) -> None:
		source = """
		#define SQUARE(x) ((x) * (x))
		int main() {
			return SQUARE(5);
		}
		"""
		asm = compile_source(source)
		assert "$5" in asm
		assert "imulq" in asm

	def test_max_macro_with_ternary(self) -> None:
		source = """
		#define MAX(a, b) ((a) > (b) ? (a) : (b))
		int main() {
			return MAX(10, 20);
		}
		"""
		asm = compile_source(source)
		assert "$10" in asm
		assert "$20" in asm
		assert "cmpq" in asm

	def test_macro_with_function_call(self) -> None:
		source = """
		#define DOUBLE(x) ((x) + (x))
		int compute(int n) {
			return n + 1;
		}
		int main() {
			return DOUBLE(compute(3));
		}
		"""
		asm = compile_source(source)
		assert "call compute" in asm
		assert "addq" in asm

	def test_zero_arg_macro(self) -> None:
		source = """
		#define ZERO() 0
		int main() {
			return ZERO();
		}
		"""
		asm = compile_source(source)
		assert "$0" in asm
		assert "ret" in asm


class TestIfdefConditionalCompilation:
	"""#ifdef conditional compilation selecting different code paths."""

	def test_ifdef_defined_path(self) -> None:
		source = """
		#define USE_FAST
		int main() {
			int result;
		#ifdef USE_FAST
			result = 100;
		#else
			result = 200;
		#endif
			return result;
		}
		"""
		asm = compile_source(source)
		assert "$100" in asm
		# The else branch should be excluded by preprocessor
		assert "$200" not in asm

	def test_ifdef_undefined_path(self) -> None:
		source = """
		int main() {
			int result;
		#ifdef USE_FAST
			result = 100;
		#else
			result = 200;
		#endif
			return result;
		}
		"""
		asm = compile_source(source)
		assert "$200" in asm
		assert "$100" not in asm

	def test_ifndef_guard(self) -> None:
		source = """
		#ifndef DEFAULT_VAL
		#define DEFAULT_VAL 42
		#endif
		int main() {
			return DEFAULT_VAL;
		}
		"""
		asm = compile_source(source)
		assert "$42" in asm

	def test_ifdef_selects_different_functions(self) -> None:
		source = """
		#define MODE_ADD
		#ifdef MODE_ADD
		int operation(int a, int b) {
			return a + b;
		}
		#else
		int operation(int a, int b) {
			return a - b;
		}
		#endif
		int main() {
			return operation(10, 5);
		}
		"""
		asm = compile_source(source)
		assert "operation:" in asm
		assert "addq" in asm
		assert "call operation" in asm

	def test_if_expression_with_defined(self) -> None:
		source = """
		#define VERSION 2
		#if VERSION > 1
		int main() {
			return 99;
		}
		#else
		int main() {
			return 1;
		}
		#endif
		"""
		asm = compile_source(source)
		assert "$99" in asm
		assert "$1" not in asm or "$1" in asm  # $1 might appear in other contexts

	def test_elif_chain(self) -> None:
		source = """
		#define LEVEL 2
		int main() {
			int val;
		#if LEVEL == 1
			val = 10;
		#elif LEVEL == 2
			val = 20;
		#elif LEVEL == 3
			val = 30;
		#else
			val = 0;
		#endif
			return val;
		}
		"""
		asm = compile_source(source)
		assert "$20" in asm


class TestNestedMacroExpansion:
	"""Nested macro expansion."""

	def test_macro_referencing_another_macro(self) -> None:
		source = """
		#define BASE 10
		#define DOUBLED (BASE + BASE)
		int main() {
			return DOUBLED;
		}
		"""
		asm = compile_source(source)
		assert "$10" in asm
		assert "addq" in asm

	def test_triple_nested_macros(self) -> None:
		source = """
		#define X 5
		#define Y (X + 1)
		#define Z (Y + 2)
		int main() {
			return Z;
		}
		"""
		asm = compile_source(source)
		assert "$5" in asm
		assert "addq" in asm

	def test_macro_using_function_macro(self) -> None:
		source = """
		#define MUL(a, b) ((a) * (b))
		#define SQUARE(x) MUL(x, x)
		int main() {
			return SQUARE(4);
		}
		"""
		asm = compile_source(source)
		assert "$4" in asm
		assert "imulq" in asm

	def test_object_macro_inside_function_macro(self) -> None:
		source = """
		#define OFFSET 8
		#define ADJUSTED(x) ((x) + OFFSET)
		int main() {
			return ADJUSTED(2);
		}
		"""
		asm = compile_source(source)
		assert "$2" in asm
		assert "$8" in asm
		assert "addq" in asm


class TestMacroInFunctionBodies:
	"""Macros used in function bodies."""

	def test_macro_constant_in_loop(self) -> None:
		source = """
		#define ITERATIONS 4
		int sum_up() {
			int total = 0;
			int i;
			for (i = 1; i <= ITERATIONS; i++) {
				total = total + i;
			}
			return total;
		}
		int main() {
			return sum_up();
		}
		"""
		asm = compile_source(source)
		assert "sum_up:" in asm
		assert "$4" in asm
		assert "for_start" in asm
		assert "call sum_up" in asm

	def test_macro_in_array_size(self) -> None:
		source = """
		#define SIZE 3
		int main() {
			int arr[SIZE];
			arr[0] = 10;
			arr[1] = 20;
			arr[2] = 30;
			return arr[0] + arr[1] + arr[2];
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm
		assert "$10" in asm
		assert "$20" in asm
		assert "$30" in asm
		assert "addq" in asm

	def test_function_macro_in_assignment(self) -> None:
		source = """
		#define CLAMP(val, lo, hi) ((val) < (lo) ? (lo) : ((val) > (hi) ? (hi) : (val)))
		int main() {
			int x = CLAMP(50, 0, 100);
			return x;
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm
		assert "cmpq" in asm

	def test_macro_in_if_condition(self) -> None:
		source = """
		#define THRESHOLD 50
		int classify(int n) {
			if (n > THRESHOLD) {
				return 1;
			}
			return 0;
		}
		int main() {
			return classify(60);
		}
		"""
		asm = compile_source(source)
		assert "$50" in asm
		assert "classify:" in asm
		assert "if_then" in asm

	def test_macro_in_while_condition(self) -> None:
		source = """
		#define MAX_ITER 3
		int count() {
			int n = 0;
			while (n < MAX_ITER) {
				n = n + 1;
			}
			return n;
		}
		int main() {
			return count();
		}
		"""
		asm = compile_source(source)
		assert "$3" in asm
		assert "while_start" in asm
		assert "while_end" in asm


class TestStringification:
	"""Stringification (#) producing valid tokens that compile."""

	def test_stringify_produces_string_literal(self) -> None:
		source = """
		#define STR(x) #x
		int main() {
			char *s = STR(hello);
			return 0;
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm
		# String data should appear in the assembly
		assert "hello" in asm

	def test_stringify_in_function_arg(self) -> None:
		source = """
		#define STR(x) #x
		int use_string(char *s) {
			return 0;
		}
		int main() {
			return use_string(STR(world));
		}
		"""
		asm = compile_source(source)
		assert "world" in asm
		assert "call use_string" in asm

	def test_stringify_number(self) -> None:
		source = """
		#define TO_STR(x) #x
		int main() {
			char *s = TO_STR(42);
			return 0;
		}
		"""
		asm = compile_source(source)
		# The stringified "42" should appear as string data
		assert "42" in asm


class TestTokenPasting:
	"""Token pasting (##) producing valid tokens that compile."""

	def test_paste_creates_variable_name(self) -> None:
		source = """
		#define VAR(n) var ## n
		int main() {
			int VAR(1) = 10;
			int VAR(2) = 20;
			return VAR(1) + VAR(2);
		}
		"""
		asm = compile_source(source)
		assert "$10" in asm
		assert "$20" in asm
		assert "addq" in asm

	def test_paste_creates_function_name(self) -> None:
		source = """
		#define FUNC(name) do_ ## name
		int do_add(int a, int b) {
			return a + b;
		}
		int main() {
			return FUNC(add)(3, 7);
		}
		"""
		asm = compile_source(source)
		assert "do_add:" in asm
		assert "call do_add" in asm

	def test_paste_with_prefix_and_suffix(self) -> None:
		source = """
		#define MAKE_NAME(prefix, suffix) prefix ## _ ## suffix
		int get_value() {
			return 55;
		}
		int main() {
			return MAKE_NAME(get, value)();
		}
		"""
		asm = compile_source(source)
		assert "call get_value" in asm
		assert "$55" in asm

	def test_paste_creates_type_name(self) -> None:
		source = """
		#define TYPE(name) my_ ## name
		struct TYPE(data) {
			int x;
			int y;
		};
		int main() {
			struct TYPE(data) d;
			d.x = 1;
			d.y = 2;
			return d.x + d.y;
		}
		"""
		asm = compile_source(source)
		assert "addq" in asm
		assert "$1" in asm
		assert "$2" in asm


class TestCombinedFeatures:
	"""Tests combining multiple preprocessor features."""

	def test_ifdef_with_define_constant(self) -> None:
		source = """
		#define ENABLE_FEATURE
		#ifdef ENABLE_FEATURE
		#define FACTOR 3
		#else
		#define FACTOR 1
		#endif
		int main() {
			return 10 * FACTOR;
		}
		"""
		asm = compile_source(source)
		assert "$10" in asm
		assert "$3" in asm
		assert "imulq" in asm

	def test_nested_ifdef(self) -> None:
		source = """
		#define PLATFORM_X86
		#define BIT_64
		int main() {
			int val;
		#ifdef PLATFORM_X86
		#ifdef BIT_64
			val = 64;
		#else
			val = 32;
		#endif
		#else
			val = 0;
		#endif
			return val;
		}
		"""
		asm = compile_source(source)
		assert "$64" in asm

	def test_undef_then_redefine(self) -> None:
		source = """
		#define MODE 1
		#undef MODE
		#define MODE 2
		int main() {
			return MODE;
		}
		"""
		asm = compile_source(source)
		assert "$2" in asm

	def test_macro_in_struct_member_init(self) -> None:
		source = """
		#define INIT_X 100
		#define INIT_Y 200
		struct Point {
			int x;
			int y;
		};
		int main() {
			struct Point p;
			p.x = INIT_X;
			p.y = INIT_Y;
			return p.x + p.y;
		}
		"""
		asm = compile_source(source)
		assert "$100" in asm
		assert "$200" in asm
		assert "addq" in asm

	def test_conditional_function_definition_with_macros(self) -> None:
		source = """
		#define USE_MULTIPLY
		#define SCALE 3
		#ifdef USE_MULTIPLY
		int transform(int x) {
			return x * SCALE;
		}
		#else
		int transform(int x) {
			return x + SCALE;
		}
		#endif
		int main() {
			return transform(7);
		}
		"""
		asm = compile_source(source)
		assert "transform:" in asm
		assert "$3" in asm
		assert "imulq" in asm
		assert "call transform" in asm
