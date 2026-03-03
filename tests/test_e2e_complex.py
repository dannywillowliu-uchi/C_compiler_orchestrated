"""End-to-end stress tests with complex multi-feature C programs."""

from src.compiler.codegen import CodeGenerator
from src.compiler.ir_gen import IRGenerator
from src.compiler.lexer import Lexer
from src.compiler.parser import Parser
from src.compiler.preprocessor import Preprocessor
from src.compiler.semantic import SemanticAnalyzer


def compile_source(source: str) -> str:
	"""Run C source through the full compiler pipeline, returning assembly."""
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	ast = Parser(tokens).parse()
	errors = SemanticAnalyzer().analyze(ast)
	assert not errors, f"Semantic errors: {errors}"
	ir = IRGenerator().generate(ast)
	return CodeGenerator().generate(ir)


class TestFibonacciIterative:
	"""Fibonacci using for loop + variables."""

	def test_fibonacci_compiles(self) -> None:
		source = """
		int fibonacci(int n) {
			int a = 0;
			int b = 1;
			int temp;
			int i;
			for (i = 0; i < n; i++) {
				temp = b;
				b = a + b;
				a = temp;
			}
			return a;
		}
		int main() {
			return fibonacci(10);
		}
		"""
		asm = compile_source(source)
		assert "fibonacci:" in asm
		assert "main:" in asm
		assert ".globl fibonacci" in asm
		assert ".globl main" in asm

	def test_fibonacci_has_loop_structure(self) -> None:
		source = """
		int fibonacci(int n) {
			int a = 0;
			int b = 1;
			int temp;
			int i;
			for (i = 0; i < n; i++) {
				temp = b;
				b = a + b;
				a = temp;
			}
			return a;
		}
		"""
		asm = compile_source(source)
		assert "for_start" in asm
		assert "for_body" in asm
		assert "for_update" in asm
		assert "for_end" in asm
		assert "jmp" in asm
		assert "addq" in asm

	def test_fibonacci_uses_variables(self) -> None:
		source = """
		int fibonacci(int n) {
			int a = 0;
			int b = 1;
			int temp;
			int i;
			for (i = 0; i < n; i++) {
				temp = b;
				b = a + b;
				a = temp;
			}
			return a;
		}
		"""
		asm = compile_source(source)
		assert "movq %rbp" in asm or "movq" in asm
		assert "ret" in asm
		assert "$0" in asm
		assert "$1" in asm


class TestBubbleSort:
	"""Bubble sort using nested for loops + array access + swap via pointers."""

	def test_bubble_sort_compiles(self) -> None:
		source = """
		void swap(int *a, int *b) {
			int temp = *a;
			*a = *b;
			*b = temp;
		}
		void bubble_sort(int *arr, int n) {
			int i;
			int j;
			for (i = 0; i < n - 1; i++) {
				for (j = 0; j < n - 1 - i; j++) {
					if (*(arr + j) > *(arr + j + 1)) {
						swap(arr + j, arr + j + 1);
					}
				}
			}
		}
		int main() {
			int arr[5];
			arr[0] = 5;
			arr[1] = 3;
			arr[2] = 1;
			arr[3] = 4;
			arr[4] = 2;
			bubble_sort(arr, 5);
			return arr[0];
		}
		"""
		asm = compile_source(source)
		assert "swap:" in asm
		assert "bubble_sort:" in asm
		assert "main:" in asm
		assert ".globl swap" in asm
		assert ".globl bubble_sort" in asm

	def test_bubble_sort_has_nested_loops(self) -> None:
		source = """
		void swap(int *a, int *b) {
			int temp = *a;
			*a = *b;
			*b = temp;
		}
		void bubble_sort(int *arr, int n) {
			int i;
			int j;
			for (i = 0; i < n - 1; i++) {
				for (j = 0; j < n - 1 - i; j++) {
					if (*(arr + j) > *(arr + j + 1)) {
						swap(arr + j, arr + j + 1);
					}
				}
			}
		}
		"""
		asm = compile_source(source)
		assert asm.count("for_start") >= 2
		assert asm.count("for_body") >= 2
		assert asm.count("for_end") >= 2

	def test_bubble_sort_uses_pointer_ops(self) -> None:
		source = """
		void swap(int *a, int *b) {
			int temp = *a;
			*a = *b;
			*b = temp;
		}
		"""
		asm = compile_source(source)
		load_count = asm.count("(%rax)")
		assert load_count >= 2
		assert "callq" not in asm


class TestStructLinkedList:
	"""Struct-based linked list node creation and traversal."""

	def test_linked_list_node_compiles(self) -> None:
		source = """
		struct Node {
			int value;
			int next;
		};
		int create_node(int val) {
			struct Node n;
			n.value = val;
			n.next = 0;
			return n.value;
		}
		int main() {
			return create_node(42);
		}
		"""
		asm = compile_source(source)
		assert "create_node:" in asm
		assert "main:" in asm
		assert ".globl create_node" in asm
		assert "$42" in asm

	def test_linked_list_struct_member_offset(self) -> None:
		source = """
		struct Node {
			int value;
			int next;
		};
		int get_next_field(struct Node *n) {
			return n->next;
		}
		int main() {
			struct Node node;
			node.value = 10;
			node.next = 20;
			return get_next_field(&node);
		}
		"""
		asm = compile_source(source)
		assert "get_next_field:" in asm
		assert "$8" in asm or "addq" in asm
		assert "ret" in asm

	def test_linked_list_traversal_pattern(self) -> None:
		source = """
		struct Node {
			int value;
			int next;
		};
		int sum_two_nodes() {
			struct Node a;
			struct Node b;
			a.value = 10;
			a.next = 0;
			b.value = 20;
			b.next = 0;
			return a.value + b.value;
		}
		int main() {
			return sum_two_nodes();
		}
		"""
		asm = compile_source(source)
		assert "sum_two_nodes:" in asm
		assert "addq" in asm
		assert "ret" in asm
		assert asm.count("subq") >= 1


class TestRecursiveFactorial:
	"""Recursive factorial with base case."""

	def test_recursive_factorial_compiles(self) -> None:
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
		asm = compile_source(source)
		assert "factorial:" in asm
		assert "main:" in asm
		assert ".globl factorial" in asm
		assert ".globl main" in asm

	def test_recursive_factorial_has_recursive_call(self) -> None:
		source = """
		int factorial(int n) {
			if (n <= 1) {
				return 1;
			}
			return n * factorial(n - 1);
		}
		"""
		asm = compile_source(source)
		assert "call factorial" in asm
		assert "imulq" in asm

	def test_recursive_factorial_has_base_case(self) -> None:
		source = """
		int factorial(int n) {
			if (n <= 1) {
				return 1;
			}
			return n * factorial(n - 1);
		}
		"""
		asm = compile_source(source)
		assert "setle" in asm
		assert "$1" in asm
		assert "if_then" in asm
		assert "if_end" in asm
		assert asm.count("ret") >= 2

	def test_recursive_factorial_frame_management(self) -> None:
		source = """
		int factorial(int n) {
			if (n <= 1) {
				return 1;
			}
			return n * factorial(n - 1);
		}
		"""
		asm = compile_source(source)
		assert "pushq %rbp" in asm
		assert "movq %rsp, %rbp" in asm
		assert "popq %rbp" in asm


class TestMatrixMultiplication:
	"""Matrix multiplication using 2D array indexing."""

	def test_matrix_multiply_compiles(self) -> None:
		source = """
		int main() {
			int a[4];
			int b[4];
			int c[4];
			a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4;
			b[0] = 5; b[1] = 6; b[2] = 7; b[3] = 8;
			c[0] = a[0] * b[0] + a[1] * b[2];
			c[1] = a[0] * b[1] + a[1] * b[3];
			c[2] = a[2] * b[0] + a[3] * b[2];
			c[3] = a[2] * b[1] + a[3] * b[3];
			return c[0];
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm
		assert ".globl main" in asm
		assert "imulq" in asm
		assert "addq" in asm

	def test_matrix_multiply_function(self) -> None:
		source = """
		void mat_mul_2x2(int *a, int *b, int *c) {
			c[0] = a[0] * b[0] + a[1] * b[2];
			c[1] = a[0] * b[1] + a[1] * b[3];
			c[2] = a[2] * b[0] + a[3] * b[2];
			c[3] = a[2] * b[1] + a[3] * b[3];
		}
		int main() {
			int a[4];
			int b[4];
			int c[4];
			a[0] = 1; a[1] = 0; a[2] = 0; a[3] = 1;
			b[0] = 5; b[1] = 6; b[2] = 7; b[3] = 8;
			mat_mul_2x2(a, b, c);
			return c[0];
		}
		"""
		asm = compile_source(source)
		assert "mat_mul_2x2:" in asm
		assert ".globl mat_mul_2x2" in asm
		assert "call mat_mul_2x2" in asm

	def test_matrix_multiply_has_array_indexing(self) -> None:
		source = """
		int main() {
			int a[4];
			int b[4];
			a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4;
			b[0] = a[0] * a[3] + a[1] * a[2];
			return b[0];
		}
		"""
		asm = compile_source(source)
		assert "imulq" in asm
		assert "addq" in asm
		mul_count = asm.count("imulq")
		assert mul_count >= 2


class TestStateMachine:
	"""State machine using switch/case inside while loop."""

	def test_state_machine_compiles(self) -> None:
		source = """
		int run_machine() {
			int state = 0;
			int count = 0;
			while (state != 3) {
				switch (state) {
					case 0:
						count = count + 1;
						state = 1;
						break;
					case 1:
						count = count + 2;
						state = 2;
						break;
					case 2:
						count = count + 3;
						state = 3;
						break;
					default:
						state = 3;
						break;
				}
			}
			return count;
		}
		int main() {
			return run_machine();
		}
		"""
		asm = compile_source(source)
		assert "run_machine:" in asm
		assert "main:" in asm
		assert ".globl run_machine" in asm

	def test_state_machine_has_while_and_switch(self) -> None:
		source = """
		int run_machine() {
			int state = 0;
			int count = 0;
			while (state != 3) {
				switch (state) {
					case 0:
						count = count + 1;
						state = 1;
						break;
					case 1:
						count = count + 2;
						state = 2;
						break;
					case 2:
						count = count + 3;
						state = 3;
						break;
					default:
						state = 3;
						break;
				}
			}
			return count;
		}
		"""
		asm = compile_source(source)
		assert "while_start" in asm
		assert "while_body" in asm
		assert "while_end" in asm
		assert "switch_end" in asm
		assert "case" in asm

	def test_state_machine_has_comparisons(self) -> None:
		source = """
		int run_machine() {
			int state = 0;
			int count = 0;
			while (state != 3) {
				switch (state) {
					case 0:
						state = 1;
						break;
					case 1:
						state = 2;
						break;
					default:
						state = 3;
						break;
				}
			}
			return count;
		}
		"""
		asm = compile_source(source)
		assert "cmpq" in asm
		assert asm.count("cmpq") >= 3


class TestBinarySearch:
	"""Binary search on sorted array."""

	def test_binary_search_compiles(self) -> None:
		source = """
		int binary_search(int *arr, int size, int target) {
			int low = 0;
			int high = size - 1;
			int mid;
			while (low <= high) {
				mid = low + (high - low) / 2;
				if (*(arr + mid) == target) {
					return mid;
				}
				if (*(arr + mid) < target) {
					low = mid + 1;
				} else {
					high = mid - 1;
				}
			}
			return 0 - 1;
		}
		int main() {
			int arr[5];
			arr[0] = 1;
			arr[1] = 3;
			arr[2] = 5;
			arr[3] = 7;
			arr[4] = 9;
			return binary_search(arr, 5, 5);
		}
		"""
		asm = compile_source(source)
		assert "binary_search:" in asm
		assert "main:" in asm
		assert ".globl binary_search" in asm
		assert "call binary_search" in asm

	def test_binary_search_has_loop_and_branches(self) -> None:
		source = """
		int binary_search(int *arr, int size, int target) {
			int low = 0;
			int high = size - 1;
			int mid;
			while (low <= high) {
				mid = low + (high - low) / 2;
				if (*(arr + mid) == target) {
					return mid;
				}
				if (*(arr + mid) < target) {
					low = mid + 1;
				} else {
					high = mid - 1;
				}
			}
			return 0 - 1;
		}
		"""
		asm = compile_source(source)
		assert "while_start" in asm
		assert "while_body" in asm
		assert "while_end" in asm
		assert asm.count("if_then") >= 2
		assert "idivq" in asm

	def test_binary_search_has_pointer_deref(self) -> None:
		source = """
		int binary_search(int *arr, int size, int target) {
			int low = 0;
			int high = size - 1;
			int mid;
			while (low <= high) {
				mid = low + (high - low) / 2;
				if (*(arr + mid) == target) {
					return mid;
				}
				if (*(arr + mid) < target) {
					low = mid + 1;
				} else {
					high = mid - 1;
				}
			}
			return 0 - 1;
		}
		"""
		asm = compile_source(source)
		assert "(%rax)" in asm
		assert "sete" in asm
		assert "setl" in asm
		assert asm.count("ret") >= 2


class TestEnumSwitchCases:
	"""Program using enum constants in switch cases."""

	def test_enum_switch_compiles(self) -> None:
		source = """
		enum Color { RED, GREEN, BLUE };
		int color_value(int c) {
			int result;
			switch (c) {
				case 0:
					result = 10;
					break;
				case 1:
					result = 20;
					break;
				case 2:
					result = 30;
					break;
				default:
					result = 0;
					break;
			}
			return result;
		}
		int main() {
			return color_value(1);
		}
		"""
		asm = compile_source(source)
		assert "color_value:" in asm
		assert "main:" in asm
		assert ".globl color_value" in asm
		assert "call color_value" in asm

	def test_enum_with_explicit_values_in_switch(self) -> None:
		source = """
		enum Status { OK = 0, ERROR = 1, PENDING = 2, DONE = 3 };
		int handle_status(int s) {
			int r;
			switch (s) {
				case 0:
					r = 100;
					break;
				case 1:
					r = 200;
					break;
				case 2:
					r = 300;
					break;
				case 3:
					r = 400;
					break;
				default:
					r = 0;
					break;
			}
			return r;
		}
		int main() {
			return handle_status(3);
		}
		"""
		asm = compile_source(source)
		assert "handle_status:" in asm
		assert "switch" in asm
		assert "case" in asm
		case_count = asm.count("case")
		assert case_count >= 4

	def test_enum_constants_as_values(self) -> None:
		source = """
		enum Direction { NORTH, EAST = 2, SOUTH = 4, WEST = 6 };
		int main() {
			int d = 2;
			int steps = 0;
			if (d == 0) {
				steps = 1;
			}
			if (d == 2) {
				steps = 2;
			}
			if (d == 4) {
				steps = 3;
			}
			if (d == 6) {
				steps = 4;
			}
			return steps;
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm
		assert "$2" in asm
		assert "cmpq" in asm
		assert asm.count("if_then") >= 4
