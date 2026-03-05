"""End-to-end compilation tests for multi-feature C programs.

Each test compiles a small but complete C program exercising multiple features
together through the full pipeline (preprocess -> lex -> parse -> semantic ->
IR -> optimize -> regalloc -> codegen -> peephole), then verifies the generated
assembly is syntactically valid.
"""

import re

from compiler.__main__ import compile_source
from compiler.preprocessor import Preprocessor
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer
from compiler.ir_gen import IRGenerator
from compiler.codegen import CodeGenerator
from compiler.optimizer import IROptimizer
from compiler.regalloc import allocate_registers
from compiler.peephole import PeepholeOptimizer


def _full_pipeline(source: str, optimize: bool = False) -> str:
	"""Run source through the complete pipeline."""
	return compile_source(source, optimize=optimize)


def _full_pipeline_with_preprocess(source: str, optimize: bool = False) -> str:
	"""Run source through the complete pipeline including preprocessor."""
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	ast = Parser(tokens).parse()
	errors = SemanticAnalyzer().analyze(ast)
	assert not errors, f"Semantic errors: {errors}"
	ir_program = IRGenerator().generate(ast)
	if optimize:
		ir_program = IROptimizer().optimize(ir_program)
	regalloc_maps = allocate_registers(ir_program) if optimize else None
	assembly = CodeGenerator(regalloc_maps=regalloc_maps).generate(ir_program)
	if optimize:
		assembly = PeepholeOptimizer().optimize(assembly)
	return assembly


def _validate_asm(asm: str) -> None:
	"""Validate that assembly output has basic syntactic structure."""
	lines = [ln.strip() for ln in asm.splitlines() if ln.strip() and not ln.strip().startswith("#")]
	assert len(lines) > 0, "Assembly output is empty"
	# Must have at least one function label
	has_label = any(re.match(r"^\w+:$", ln) for ln in lines)
	assert has_label, "No function labels found in assembly"
	# Must have a ret instruction
	assert any("ret" in ln for ln in lines), "No ret instruction found"
	# Check no obviously broken instructions (bare register names, etc.)
	for line in lines:
		# Skip directives and labels
		if line.startswith(".") or line.endswith(":"):
			continue
		# An instruction line should start with a known prefix or be a valid mnemonic
		# Just ensure lines aren't empty after stripping comments
		stripped = line.split("#")[0].strip()
		if stripped:
			assert not stripped.startswith("INVALID"), f"Invalid instruction: {stripped}"


class TestStructBitfieldsWithPointers:
	"""Structs with bitfields combined with pointer operations."""

	def test_bitfield_struct_with_address_of(self) -> None:
		source = """
		struct Flags {
			unsigned int read : 1;
			unsigned int write : 1;
			unsigned int exec : 1;
		};
		int main(void) {
			struct Flags f;
			int *p = (int*)&f;
			*p = 7;
			return 0;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "main:" in asm

	def test_bitfield_struct_with_address_of_optimized(self) -> None:
		source = """
		struct Flags {
			unsigned int read : 1;
			unsigned int write : 1;
			unsigned int exec : 1;
		};
		int main(void) {
			struct Flags f;
			int *p = (int*)&f;
			*p = 7;
			return 0;
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		_validate_asm(asm)
		assert "main:" in asm

	def test_struct_pointer_with_bitfield_and_regular_members(self) -> None:
		source = """
		struct Packet {
			unsigned int version : 4;
			unsigned int type : 4;
			int payload;
			int checksum;
		};
		int compute_check(int payload) {
			return payload * 31;
		}
		int main(void) {
			struct Packet pkt;
			pkt.payload = 42;
			pkt.checksum = compute_check(pkt.payload);
			int *data = &pkt.payload;
			return *data + pkt.checksum;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "compute_check:" in asm
		assert "main:" in asm
		assert "call" in asm

	def test_bitfield_struct_array_access(self) -> None:
		"""Bitfield struct used alongside array indexing."""
		source = """
		struct Config {
			unsigned int enabled : 1;
			unsigned int priority : 3;
			int value;
		};
		int main(void) {
			int values[4];
			values[0] = 10;
			values[1] = 20;
			values[2] = 30;
			values[3] = 40;
			struct Config cfg;
			int idx = 2;
			return values[idx];
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "main:" in asm

	def test_nested_struct_with_bitfield_member(self) -> None:
		source = """
		struct Status {
			unsigned int active : 1;
			unsigned int error : 1;
		};
		struct Device {
			struct Status status;
			int id;
		};
		int main(void) {
			struct Device dev;
			dev.id = 42;
			int *pid = &dev.id;
			return *pid;
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		_validate_asm(asm)
		assert "main:" in asm


class TestVariadicFunctionsCallingEachOther:
	"""Variadic functions that invoke other variadic or regular functions."""

	def test_variadic_calls_regular_function(self) -> None:
		source = """
		#include <stdarg.h>
		int add(int a, int b) { return a + b; }
		int sum_with_add(int count, ...) {
			va_list args;
			va_start(args, count);
			int total = 0;
			int i = 0;
			while (i < count) {
				int val = va_arg(args, int);
				total = add(total, val);
				i = i + 1;
			}
			va_end(args);
			return total;
		}
		int main(void) {
			return sum_with_add(3, 10, 20, 30);
		}
		"""
		asm = _full_pipeline_with_preprocess(source)
		_validate_asm(asm)
		assert "add:" in asm
		assert "sum_with_add:" in asm
		assert "main:" in asm

	def test_variadic_calls_regular_function_optimized(self) -> None:
		source = """
		#include <stdarg.h>
		int add(int a, int b) { return a + b; }
		int sum_with_add(int count, ...) {
			va_list args;
			va_start(args, count);
			int total = 0;
			int i = 0;
			while (i < count) {
				int val = va_arg(args, int);
				total = add(total, val);
				i = i + 1;
			}
			va_end(args);
			return total;
		}
		int main(void) {
			return sum_with_add(3, 10, 20, 30);
		}
		"""
		asm = _full_pipeline_with_preprocess(source, optimize=True)
		_validate_asm(asm)
		assert "sum_with_add:" in asm
		assert "main:" in asm

	def test_two_variadic_functions(self) -> None:
		source = """
		#include <stdarg.h>
		int sum_ints(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int s = 0;
			for (int i = 0; i < n; i = i + 1) {
				s = s + va_arg(ap, int);
			}
			va_end(ap);
			return s;
		}
		int max_int(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int m = va_arg(ap, int);
			for (int i = 1; i < n; i = i + 1) {
				int v = va_arg(ap, int);
				m = (v > m) ? v : m;
			}
			va_end(ap);
			return m;
		}
		int main(void) {
			int s = sum_ints(3, 1, 2, 3);
			int m = max_int(3, 1, 2, 3);
			return s + m;
		}
		"""
		asm = _full_pipeline_with_preprocess(source)
		_validate_asm(asm)
		assert "sum_ints:" in asm
		assert "max_int:" in asm
		assert "main:" in asm

	def test_variadic_with_struct_param(self) -> None:
		source = """
		#include <stdarg.h>
		struct Pair { int a; int b; };
		int weighted_sum(int weight, int count, ...) {
			va_list ap;
			va_start(ap, count);
			int total = 0;
			for (int i = 0; i < count; i = i + 1) {
				total = total + va_arg(ap, int) * weight;
			}
			va_end(ap);
			return total;
		}
		int main(void) {
			struct Pair p;
			p.a = 5;
			p.b = 10;
			return weighted_sum(2, 2, p.a, p.b);
		}
		"""
		asm = _full_pipeline_with_preprocess(source)
		_validate_asm(asm)
		assert "weighted_sum:" in asm
		assert "main:" in asm
		assert "call" in asm


class TestGotoAcrossNestedLoops:
	"""Goto statements that jump across nested loop boundaries."""

	def test_goto_out_of_double_nested_for(self) -> None:
		source = """
		int main(void) {
			int total = 0;
			for (int i = 0; i < 10; i = i + 1) {
				for (int j = 0; j < 10; j = j + 1) {
					if (i * 10 + j == 25) goto done;
					total = total + 1;
				}
			}
			done:
			return total;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "jmp" in asm

	def test_goto_out_of_double_nested_for_optimized(self) -> None:
		source = """
		int main(void) {
			int total = 0;
			for (int i = 0; i < 10; i = i + 1) {
				for (int j = 0; j < 10; j = j + 1) {
					if (i * 10 + j == 25) goto done;
					total = total + 1;
				}
			}
			done:
			return total;
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		_validate_asm(asm)
		assert "main:" in asm

	def test_goto_across_while_and_for(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			int count = 0;
			while (x < 100) {
				for (int i = 0; i < 5; i = i + 1) {
					x = x + 1;
					count = count + 1;
					if (x >= 15) goto escape;
				}
			}
			escape:
			return count;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "jmp" in asm
		assert "main:" in asm

	def test_goto_backward_into_loop_region(self) -> None:
		"""Goto that jumps backward to a label before a loop."""
		source = """
		int main(void) {
			int x = 0;
			int iterations = 0;
			start:
			if (iterations >= 3) goto end;
			for (int i = 0; i < 2; i = i + 1) {
				x = x + 1;
			}
			iterations = iterations + 1;
			goto start;
			end:
			return x;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "jmp" in asm

	def test_goto_across_do_while_and_while(self) -> None:
		source = """
		int main(void) {
			int result = 0;
			int phase = 0;
			do {
				int i = 0;
				while (i < 3) {
					if (phase == 1 && i == 2) goto finished;
					result = result + 1;
					i = i + 1;
				}
				phase = phase + 1;
			} while (phase < 5);
			finished:
			return result;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "main:" in asm

	def test_goto_from_nested_switch_in_loop(self) -> None:
		source = """
		int main(void) {
			int state = 0;
			int counter = 0;
			for (int round = 0; round < 10; round = round + 1) {
				switch (state) {
					case 0:
						state = 1;
						break;
					case 1:
						state = 2;
						break;
					case 2:
						goto bail;
				}
				counter = counter + 1;
			}
			bail:
			return counter;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "jmp" in asm
		assert "main:" in asm


class TestFunctionPointersInArrays:
	"""Function pointers stored in arrays and dispatched."""

	def test_function_pointer_dispatch(self) -> None:
		"""Function pointers assigned and called."""
		source = """
		int add_one(int x) { return x + 1; }
		int double_it(int x) { return x * 2; }
		int negate(int x) { return 0 - x; }
		int main(void) {
			int (*op)(int);
			op = double_it;
			int result = op(10);
			op = negate;
			result = result + op(3);
			return result;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "add_one:" in asm
		assert "double_it:" in asm
		assert "negate:" in asm
		assert "main:" in asm

	def test_function_pointer_in_loop(self) -> None:
		source = """
		int square(int x) { return x * x; }
		int identity(int x) { return x; }
		int main(void) {
			int (*f)(int);
			int sum = 0;
			for (int i = 0; i < 4; i = i + 1) {
				f = (i % 2 == 0) ? square : identity;
				sum = sum + f(i);
			}
			return sum;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "square:" in asm
		assert "identity:" in asm
		assert "main:" in asm

	def test_function_pointer_as_parameter(self) -> None:
		source = """
		int apply(int (*fn)(int), int val) {
			return fn(val);
		}
		int triple(int x) { return x * 3; }
		int main(void) {
			return apply(triple, 7);
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "apply:" in asm
		assert "triple:" in asm
		assert "main:" in asm
		assert "call" in asm


class TestEnumSwitchWithFallthrough:
	"""Enum-based switch statements with deliberate fallthrough."""

	def test_enum_switch_fallthrough(self) -> None:
		source = """
		enum Level { LOW, MEDIUM, HIGH, CRITICAL };
		int score(int level) {
			int s = 0;
			switch (level) {
				case CRITICAL:
					s = s + 100;
				case HIGH:
					s = s + 50;
				case MEDIUM:
					s = s + 20;
				case LOW:
					s = s + 10;
					break;
			}
			return s;
		}
		int main(void) {
			int a = score(LOW);
			int b = score(HIGH);
			return a + b;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "score:" in asm
		assert "main:" in asm
		assert "call" in asm

	def test_enum_switch_fallthrough_optimized(self) -> None:
		source = """
		enum Level { LOW, MEDIUM, HIGH, CRITICAL };
		int score(int level) {
			int s = 0;
			switch (level) {
				case CRITICAL:
					s = s + 100;
				case HIGH:
					s = s + 50;
				case MEDIUM:
					s = s + 20;
				case LOW:
					s = s + 10;
					break;
			}
			return s;
		}
		int main(void) {
			return score(HIGH);
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		_validate_asm(asm)
		assert "score:" in asm
		assert "main:" in asm

	def test_enum_switch_partial_fallthrough_with_default(self) -> None:
		source = """
		enum Op { ADD, SUB, MUL, DIV, MOD };
		int compute(int op, int a, int b) {
			int result = 0;
			switch (op) {
				case ADD:
					result = a + b;
					break;
				case SUB:
					result = a - b;
					break;
				case MUL:
				case DIV:
					result = a * b;
					break;
				default:
					result = -1;
					break;
			}
			return result;
		}
		int main(void) {
			int r1 = compute(ADD, 3, 4);
			int r2 = compute(MUL, 3, 4);
			int r3 = compute(MOD, 3, 4);
			return r1 + r2 + r3;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "compute:" in asm
		assert "main:" in asm

	def test_enum_switch_all_fallthrough(self) -> None:
		"""All cases fall through to accumulate a result."""
		source = """
		enum Phase { INIT, LOAD, RUN, DONE };
		int count_remaining(int phase) {
			int n = 0;
			switch (phase) {
				case INIT: n = n + 1;
				case LOAD: n = n + 1;
				case RUN:  n = n + 1;
				case DONE: n = n + 1;
			}
			return n;
		}
		int main(void) {
			return count_remaining(INIT) + count_remaining(RUN);
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "count_remaining:" in asm
		assert "main:" in asm


class TestComplexMultiFeaturePrograms:
	"""Programs combining three or more features to test deep integration."""

	def test_struct_enum_switch_function_call(self) -> None:
		"""Struct + enum + switch + function calls."""
		source = """
		enum Shape { CIRCLE, SQUARE, TRIANGLE };
		struct Figure {
			int shape;
			int size;
		};
		int area_approx(struct Figure fig) {
			switch (fig.shape) {
				case CIRCLE:   return fig.size * fig.size * 3;
				case SQUARE:   return fig.size * fig.size;
				case TRIANGLE: return fig.size * fig.size / 2;
			}
			return 0;
		}
		int main(void) {
			struct Figure f;
			f.shape = SQUARE;
			f.size = 5;
			return area_approx(f);
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "area_approx:" in asm
		assert "main:" in asm
		assert "call" in asm

	def test_typedef_struct_with_ternary_and_loop(self) -> None:
		source = """
		struct Vec2 { int x; int y; };
		typedef struct Vec2 Vec2;
		int manhattan(Vec2 a, Vec2 b) {
			int dx = a.x - b.x;
			int dy = a.y - b.y;
			dx = (dx < 0) ? (0 - dx) : dx;
			dy = (dy < 0) ? (0 - dy) : dy;
			return dx + dy;
		}
		int main(void) {
			Vec2 points[3];
			points[0].x = 0; points[0].y = 0;
			points[1].x = 3; points[1].y = 4;
			points[2].x = 6; points[2].y = 8;
			int total = 0;
			for (int i = 0; i < 2; i = i + 1) {
				total = total + manhattan(points[i], points[i + 1]);
			}
			return total;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "manhattan:" in asm
		assert "main:" in asm

	def test_nested_loops_goto_enum_compound_assign(self) -> None:
		"""Nested loops + goto + enum + compound assignment."""
		source = """
		enum Dir { UP, DOWN, LEFT, RIGHT };
		int main(void) {
			int pos_x = 0;
			int pos_y = 0;
			int dir = RIGHT;
			int steps = 0;
			for (int i = 0; i < 4; i = i + 1) {
				for (int j = 0; j < 3; j = j + 1) {
					switch (dir) {
						case RIGHT: pos_x += 1; break;
						case UP:    pos_y += 1; break;
						case LEFT:  pos_x -= 1; break;
						case DOWN:  pos_y -= 1; break;
					}
					steps += 1;
					if (steps >= 8) goto stop;
				}
				dir = (dir + 1) % 4;
			}
			stop:
			return pos_x + pos_y;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "main:" in asm
		assert "jmp" in asm

	def test_variadic_with_enum_and_switch(self) -> None:
		source = """
		#include <stdarg.h>
		enum Action { PRINT, ADD, NOOP };
		int dispatch(int action, int count, ...) {
			va_list ap;
			va_start(ap, count);
			int result = 0;
			switch (action) {
				case ADD:
					for (int i = 0; i < count; i = i + 1) {
						result = result + va_arg(ap, int);
					}
					break;
				case PRINT:
					result = va_arg(ap, int);
					break;
				default:
					result = -1;
					break;
			}
			va_end(ap);
			return result;
		}
		int main(void) {
			int a = dispatch(ADD, 3, 10, 20, 30);
			int b = dispatch(PRINT, 1, 42);
			return a + b;
		}
		"""
		asm = _full_pipeline_with_preprocess(source)
		_validate_asm(asm)
		assert "dispatch:" in asm
		assert "main:" in asm

	def test_union_cast_sizeof_ternary(self) -> None:
		"""Union + cast + sizeof + ternary expression."""
		source = """
		union Value {
			int ival;
			char cval;
		};
		int main(void) {
			union Value v;
			v.ival = 256;
			int sz = (int)sizeof(union Value);
			int result = (sz >= 4) ? v.ival : (int)v.cval;
			return result;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "main:" in asm

	def test_do_while_switch_postfix_compound_assign(self) -> None:
		"""do-while + switch + postfix ops + compound assignment."""
		source = """
		int main(void) {
			int state = 0;
			int acc = 0;
			int iter = 0;
			do {
				switch (state) {
					case 0:
						acc += 10;
						state = 1;
						break;
					case 1:
						acc *= 2;
						state = 2;
						break;
					case 2:
						acc -= 5;
						state = 3;
						break;
					default:
						break;
				}
				iter++;
			} while (state < 3);
			return acc + iter;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "main:" in asm

	def test_struct_array_for_loop_function(self) -> None:
		"""Struct array + for loop + function call + member access."""
		source = """
		struct Entry { int key; int value; };
		int lookup(struct Entry *entries, int count, int key) {
			for (int i = 0; i < count; i = i + 1) {
				if (entries[i].key == key) {
					return entries[i].value;
				}
			}
			return -1;
		}
		int main(void) {
			struct Entry table[3];
			table[0].key = 1; table[0].value = 100;
			table[1].key = 2; table[1].value = 200;
			table[2].key = 3; table[2].value = 300;
			return lookup(table, 3, 2);
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "lookup:" in asm
		assert "main:" in asm
		assert "call" in asm

	def test_complex_program_optimized(self) -> None:
		"""Multi-feature program through the full optimized pipeline."""
		source = """
		enum Mode { FAST, SLOW, AUTO };
		struct Config {
			int mode;
			int threshold;
		};
		int process(struct Config cfg, int input) {
			int result = input;
			switch (cfg.mode) {
				case FAST:
					result = result * 2;
					break;
				case SLOW:
					for (int i = 0; i < cfg.threshold; i = i + 1) {
						result = result + 1;
					}
					break;
				case AUTO:
					result = (input > cfg.threshold) ? input * 2 : input + cfg.threshold;
					break;
			}
			return result;
		}
		int main(void) {
			struct Config c;
			c.mode = AUTO;
			c.threshold = 10;
			int r1 = process(c, 5);
			c.mode = FAST;
			int r2 = process(c, 3);
			return r1 + r2;
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		_validate_asm(asm)
		assert "process:" in asm
		assert "main:" in asm


class TestBothOptimizationPaths:
	"""Run the same programs through both optimized and unoptimized pipelines."""

	def test_loop_with_ternary_both_paths(self) -> None:
		source = """
		int abs_val(int x) {
			return (x < 0) ? (0 - x) : x;
		}
		int main(void) {
			int sum = 0;
			for (int i = -5; i <= 5; i = i + 1) {
				sum = sum + abs_val(i);
			}
			return sum;
		}
		"""
		asm_unopt = _full_pipeline(source, optimize=False)
		asm_opt = _full_pipeline(source, optimize=True)
		_validate_asm(asm_unopt)
		_validate_asm(asm_opt)
		assert "abs_val:" in asm_unopt
		assert "abs_val:" in asm_opt
		assert "main:" in asm_unopt
		assert "main:" in asm_opt

	def test_struct_switch_both_paths(self) -> None:
		source = """
		struct Msg { int type; int data; };
		int handle(struct Msg m) {
			switch (m.type) {
				case 0: return m.data;
				case 1: return m.data * 2;
				case 2: return m.data + 100;
				default: return -1;
			}
		}
		int main(void) {
			struct Msg m;
			m.type = 1;
			m.data = 21;
			return handle(m);
		}
		"""
		asm_unopt = _full_pipeline(source, optimize=False)
		asm_opt = _full_pipeline(source, optimize=True)
		_validate_asm(asm_unopt)
		_validate_asm(asm_opt)
		assert "handle:" in asm_unopt
		assert "handle:" in asm_opt

	def test_variadic_with_loop_both_paths(self) -> None:
		source = """
		#include <stdarg.h>
		int product(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int p = 1;
			for (int i = 0; i < n; i = i + 1) {
				p = p * va_arg(ap, int);
			}
			va_end(ap);
			return p;
		}
		int main(void) {
			return product(3, 2, 3, 4);
		}
		"""
		asm_unopt = _full_pipeline_with_preprocess(source, optimize=False)
		asm_opt = _full_pipeline_with_preprocess(source, optimize=True)
		_validate_asm(asm_unopt)
		_validate_asm(asm_opt)
		assert "product:" in asm_unopt
		assert "product:" in asm_opt

	def test_nested_control_flow_both_paths(self) -> None:
		"""Multiple nested control structures compiled both ways."""
		source = """
		int classify(int x) {
			if (x < 0) {
				return -1;
			}
			int result = 0;
			for (int i = 0; i < x; i = i + 1) {
				if (i % 3 == 0) {
					result += i;
				} else if (i % 3 == 1) {
					result -= 1;
				} else {
					continue;
				}
			}
			return result;
		}
		int main(void) {
			int a = classify(10);
			int b = classify(-5);
			return a + b;
		}
		"""
		asm_unopt = _full_pipeline(source, optimize=False)
		asm_opt = _full_pipeline(source, optimize=True)
		_validate_asm(asm_unopt)
		_validate_asm(asm_opt)
		assert "classify:" in asm_unopt
		assert "classify:" in asm_opt


class TestEdgeCaseMultiFeature:
	"""Edge cases that combine features in unusual ways."""

	def test_empty_switch_in_loop_with_goto(self) -> None:
		source = """
		int main(void) {
			int x = 5;
			for (int i = 0; i < 3; i = i + 1) {
				switch (x) {
					default: break;
				}
			}
			goto end;
			end:
			return x;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "main:" in asm

	def test_sizeof_in_ternary_with_cast(self) -> None:
		source = """
		int main(void) {
			int sz = (int)sizeof(long);
			int result = (sz > 4) ? 64 : 32;
			return result;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "main:" in asm

	def test_struct_with_all_modifier_types(self) -> None:
		source = """
		struct AllTypes {
			short s;
			long l;
			unsigned int u;
			signed int si;
			char c;
		};
		int main(void) {
			struct AllTypes t;
			t.s = 1;
			t.l = 100;
			t.u = 42;
			t.si = -5;
			t.c = 65;
			return (int)t.s + (int)t.l + (int)t.u + t.si + (int)t.c;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "main:" in asm

	def test_recursive_function_with_ternary(self) -> None:
		source = """
		int factorial(int n) {
			return (n <= 1) ? 1 : n * factorial(n - 1);
		}
		int main(void) {
			return factorial(5);
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "factorial:" in asm
		assert "main:" in asm
		assert "call" in asm

	def test_recursive_function_with_ternary_optimized(self) -> None:
		source = """
		int factorial(int n) {
			return (n <= 1) ? 1 : n * factorial(n - 1);
		}
		int main(void) {
			return factorial(5);
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		_validate_asm(asm)
		assert "factorial:" in asm
		assert "main:" in asm

	def test_many_local_variables_with_operations(self) -> None:
		"""Stress register allocator with many live variables."""
		source = """
		int main(void) {
			int a = 1;
			int b = 2;
			int c = 3;
			int d = 4;
			int e = 5;
			int f = 6;
			int g = 7;
			int h = 8;
			int sum = a + b + c + d + e + f + g + h;
			return sum;
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		_validate_asm(asm)
		assert "main:" in asm

	def test_deeply_nested_if_else_chain(self) -> None:
		source = """
		int classify(int x) {
			if (x > 100) {
				return 5;
			} else if (x > 50) {
				return 4;
			} else if (x > 20) {
				return 3;
			} else if (x > 10) {
				return 2;
			} else if (x > 0) {
				return 1;
			} else {
				return 0;
			}
		}
		int main(void) {
			return classify(75) + classify(5) + classify(-1);
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		_validate_asm(asm)
		assert "classify:" in asm
		assert "main:" in asm

	def test_pointer_arithmetic_with_array_and_loop(self) -> None:
		source = """
		int sum_array(int *arr, int len) {
			int total = 0;
			for (int i = 0; i < len; i = i + 1) {
				total = total + arr[i];
			}
			return total;
		}
		int main(void) {
			int data[5];
			data[0] = 1;
			data[1] = 2;
			data[2] = 3;
			data[3] = 4;
			data[4] = 5;
			return sum_array(data, 5);
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "sum_array:" in asm
		assert "main:" in asm
		assert "call" in asm

	def test_compound_literal_with_struct(self) -> None:
		source = """
		struct Point { int x; int y; };
		int main(void) {
			struct Point p = (struct Point){ 10, 20 };
			return p.x + p.y;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "main:" in asm
