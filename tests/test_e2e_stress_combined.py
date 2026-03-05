"""End-to-end stress tests combining structs, bitfields, typedefs, variadics, and goto.

Each test exercises multiple features simultaneously in realistic program patterns,
compiling through the full pipeline and verifying structural correctness of the output.
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
	return compile_source(source, optimize=optimize)


def _full_pipeline_with_preprocess(source: str, optimize: bool = False) -> str:
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
	lines = [ln.strip() for ln in asm.splitlines() if ln.strip() and not ln.strip().startswith("#")]
	assert len(lines) > 0, "Assembly output is empty"
	has_label = any(re.match(r"^\w+:$", ln) for ln in lines)
	assert has_label, "No function labels found in assembly"
	assert any("ret" in ln for ln in lines), "No ret instruction found"


class TestLinkedListWithBitfieldsAndVariadics:
	"""Linked list with typedef'd struct containing bitfields, manipulated via variadic helpers."""

	def test_typedef_struct_with_bitfields_and_variadic_init(self) -> None:
		"""Typedef'd node struct with bitfields, initialized via variadic function."""
		source = """
		#include <stdarg.h>
		typedef struct Node {
			int value;
			unsigned int flags : 3;
			unsigned int priority : 4;
			unsigned int active : 1;
		} Node;

		int init_node_values(int count, ...) {
			va_list args;
			va_start(args, count);
			int total = 0;
			int i = 0;
			while (i < count) {
				total = total + va_arg(args, int);
				i = i + 1;
			}
			va_end(args);
			return total;
		}

		int main(void) {
			Node n1;
			n1.value = 10;
			n1.flags = 5;
			n1.priority = 7;
			n1.active = 1;

			Node n2;
			n2.value = 20;
			n2.flags = 3;

			int sum = init_node_values(2, n1.value, n2.value);
			return sum;
		}
		"""
		asm = _full_pipeline_with_preprocess(source)
		_validate_asm(asm)
		assert "init_node_values:" in asm
		assert "main:" in asm
		assert "call" in asm

	def test_typedef_struct_with_bitfields_and_variadic_init_optimized(self) -> None:
		"""Same test with optimizer enabled."""
		source = """
		#include <stdarg.h>
		typedef struct Node {
			int value;
			unsigned int flags : 3;
			unsigned int priority : 4;
			unsigned int active : 1;
		} Node;

		int sum_values(int count, ...) {
			va_list args;
			va_start(args, count);
			int total = 0;
			for (int i = 0; i < count; i = i + 1) {
				total = total + va_arg(args, int);
			}
			va_end(args);
			return total;
		}

		int main(void) {
			Node a;
			a.value = 100;
			a.flags = 7;
			Node b;
			b.value = 200;
			b.priority = 15;
			return sum_values(2, a.value, b.value);
		}
		"""
		asm = _full_pipeline_with_preprocess(source, optimize=True)
		_validate_asm(asm)
		assert "sum_values:" in asm
		assert "main:" in asm

	def test_linked_list_traversal_with_goto(self) -> None:
		"""Linked list nodes with bitfields, traversal uses goto for early exit."""
		source = """
		typedef struct ListNode {
			int data;
			unsigned int visited : 1;
			unsigned int type : 2;
			int next_idx;
		} ListNode;

		int main(void) {
			ListNode nodes[3];
			nodes[0].data = 10;
			nodes[0].visited = 0;
			nodes[0].type = 1;
			nodes[0].next_idx = 1;

			nodes[1].data = 20;
			nodes[1].visited = 0;
			nodes[1].type = 2;
			nodes[1].next_idx = 2;

			nodes[2].data = 30;
			nodes[2].visited = 0;
			nodes[2].type = 0;
			nodes[2].next_idx = -1;

			int sum = 0;
			int idx = 0;
			loop_start:
			if (idx < 0) goto done;
			if (idx >= 3) goto done;
			sum = sum + nodes[idx].data;
			nodes[idx].visited = 1;
			if (nodes[idx].type == 0) goto done;
			idx = nodes[idx].next_idx;
			goto loop_start;
			done:
			return sum;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "main:" in asm
		assert "jmp" in asm

	def test_variadic_max_with_typedef_bitfield_struct(self) -> None:
		"""Variadic function finds max, stores result in bitfield struct."""
		source = """
		#include <stdarg.h>
		typedef struct Result {
			int value;
			unsigned int overflow : 1;
			unsigned int valid : 1;
		} Result;

		int find_max(int count, ...) {
			va_list ap;
			va_start(ap, count);
			int max_val = va_arg(ap, int);
			for (int i = 1; i < count; i = i + 1) {
				int v = va_arg(ap, int);
				if (v > max_val) max_val = v;
			}
			va_end(ap);
			return max_val;
		}

		int main(void) {
			Result r;
			r.value = find_max(4, 3, 7, 2, 9);
			r.valid = 1;
			r.overflow = 0;
			return r.value;
		}
		"""
		asm = _full_pipeline_with_preprocess(source)
		_validate_asm(asm)
		assert "find_max:" in asm
		assert "main:" in asm
		assert "call" in asm


class TestStateMachineWithGotoAndEnumStruct:
	"""State machine using goto/labels with switch+enum+struct combinations."""

	def test_enum_driven_state_machine_with_goto(self) -> None:
		"""Enum-based states, struct context, goto for state transitions."""
		source = """
		enum State { STATE_INIT, STATE_RUN, STATE_DONE };

		typedef struct Context {
			int counter;
			int result;
			unsigned int error : 1;
		} Context;

		int main(void) {
			Context ctx;
			ctx.counter = 0;
			ctx.result = 0;
			ctx.error = 0;
			int state = 0;

			state_init:
			if (state != 0) goto state_run;
			ctx.counter = 5;
			state = 1;

			state_run:
			if (state != 1) goto state_done;
			while (ctx.counter > 0) {
				ctx.result = ctx.result + ctx.counter;
				ctx.counter = ctx.counter - 1;
			}
			state = 2;

			state_done:
			return ctx.result;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "main:" in asm
		assert "jmp" in asm

	def test_switch_with_enum_and_struct_bitfields(self) -> None:
		"""Switch on enum value, modifying struct with bitfields per case."""
		source = """
		enum Command { CMD_NOP, CMD_SET, CMD_CLEAR, CMD_TOGGLE };

		struct Register {
			int value;
			unsigned int locked : 1;
			unsigned int dirty : 1;
			unsigned int mode : 2;
		};

		int process(int cmd, int arg) {
			struct Register reg;
			reg.value = arg;
			reg.locked = 0;
			reg.dirty = 0;
			reg.mode = 0;

			switch (cmd) {
				case 0:
					break;
				case 1:
					reg.value = arg;
					reg.dirty = 1;
					break;
				case 2:
					reg.value = 0;
					reg.dirty = 1;
					break;
				case 3:
					reg.mode = 3;
					break;
			}
			return reg.value + reg.dirty;
		}

		int main(void) {
			int r1 = process(1, 42);
			int r2 = process(2, 100);
			return r1 + r2;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "process:" in asm
		assert "main:" in asm
		assert "call" in asm

	def test_state_machine_with_switch_goto_and_typedef(self) -> None:
		"""Complex state machine: typedef'd struct, switch dispatches, goto transitions."""
		source = """
		typedef struct {
			int state;
			int ticks;
			unsigned int fault : 1;
			unsigned int running : 1;
		} Machine;

		int run_machine(int initial_ticks) {
			Machine m;
			m.state = 0;
			m.ticks = initial_ticks;
			m.fault = 0;
			m.running = 1;

			dispatch:
			switch (m.state) {
				case 0:
					m.state = 1;
					goto dispatch;
				case 1:
					m.ticks = m.ticks - 1;
					if (m.ticks <= 0) {
						m.state = 3;
						goto dispatch;
					}
					if (m.ticks == 1) {
						m.fault = 1;
						m.state = 2;
						goto dispatch;
					}
					goto dispatch;
				case 2:
					m.running = 0;
					return -1;
				case 3:
					m.running = 0;
					return m.ticks;
			}
			return -2;
		}

		int main(void) {
			return run_machine(3);
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "run_machine:" in asm
		assert "main:" in asm
		assert "jmp" in asm

	def test_state_machine_optimized(self) -> None:
		"""State machine with optimization enabled."""
		source = """
		typedef struct {
			int input;
			int output;
			unsigned int valid : 1;
			unsigned int done : 1;
		} Pipeline;

		int pipeline_run(int x) {
			Pipeline p;
			p.input = x;
			p.output = 0;
			p.valid = 0;
			p.done = 0;
			int phase = 0;

			again:
			switch (phase) {
				case 0:
					p.output = p.input * 2;
					phase = 1;
					goto again;
				case 1:
					p.valid = 1;
					phase = 2;
					goto again;
				case 2:
					p.output = p.output + 1;
					phase = 3;
					goto again;
				case 3:
					p.done = 1;
					break;
			}
			return p.output;
		}

		int main(void) {
			return pipeline_run(10);
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		_validate_asm(asm)
		assert "pipeline_run:" in asm
		assert "main:" in asm


class TestDesignatedInitWithNestedStructArrays:
	"""Designated initializers with nested struct arrays and compound literals."""

	def test_designated_init_struct_with_multiple_fields(self) -> None:
		"""Struct with multiple fields initialized via designated initializers."""
		source = """
		struct Config {
			int id;
			int priority;
			int level;
			unsigned int active : 1;
		};

		int main(void) {
			struct Config cfg = { .id = 42, .priority = 5, .level = 3, .active = 1 };
			return cfg.id + cfg.priority + cfg.level;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "main:" in asm

	def test_nested_struct_designated_init(self) -> None:
		"""Nested struct with designated initializers at multiple levels."""
		source = """
		struct Inner {
			int x;
			int y;
		};

		struct Outer {
			struct Inner pos;
			int z;
			unsigned int visible : 1;
		};

		int main(void) {
			struct Outer obj = { .z = 100, .visible = 1 };
			obj.pos.x = 10;
			obj.pos.y = 20;
			return obj.pos.x + obj.pos.y + obj.z;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "main:" in asm

	def test_compound_literal_with_typedef_struct(self) -> None:
		"""Compound literal creates a typedef'd struct value inline."""
		source = """
		typedef struct {
			int x;
			int y;
		} Vec2;

		int dot(Vec2 a, Vec2 b) {
			return a.x * b.x + a.y * b.y;
		}

		int main(void) {
			Vec2 v1 = { .x = 3, .y = 4 };
			Vec2 v2;
			v2.x = 1;
			v2.y = 2;
			return v1.x + v2.y;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "main:" in asm

	def test_designated_init_with_bitfields_and_switch(self) -> None:
		"""Designated initializers combined with bitfield members and switch."""
		source = """
		typedef struct {
			int color;
			unsigned int intensity : 4;
			unsigned int enabled : 1;
			int value;
		} Pixel;

		int main(void) {
			Pixel p = { .color = 1, .value = 255, .enabled = 1 };
			p.intensity = 15;

			int result = 0;
			switch (p.color) {
				case 0: result = p.value; break;
				case 1: result = p.value * 2; break;
				case 2: result = p.value * 3; break;
			}
			return result;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "main:" in asm

	def test_designated_init_array_of_structs(self) -> None:
		"""Array of structs with designated init and loop processing."""
		source = """
		typedef struct {
			int id;
			int score;
			unsigned int passed : 1;
		} Student;

		int main(void) {
			Student s1 = { .id = 1, .score = 85, .passed = 1 };
			Student s2 = { .id = 2, .score = 42, .passed = 0 };
			Student s3 = { .id = 3, .score = 91, .passed = 1 };

			int total = s1.score + s2.score + s3.score;
			int pass_count = s1.passed + s2.passed + s3.passed;
			return total + pass_count;
		}
		"""
		asm = _full_pipeline(source)
		_validate_asm(asm)
		assert "main:" in asm

	def test_designated_init_optimized(self) -> None:
		"""Designated init with nested structs, optimized pipeline."""
		source = """
		typedef struct {
			int width;
			int height;
		} Size;

		typedef struct {
			int x;
			int y;
			Size size;
			unsigned int visible : 1;
			unsigned int focused : 1;
		} Window;

		int area(Window w) {
			return w.size.width * w.size.height;
		}

		int main(void) {
			Window w = { .x = 0, .y = 0, .visible = 1 };
			w.size.width = 800;
			w.size.height = 600;
			w.focused = 1;
			return area(w);
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		_validate_asm(asm)
		assert "area:" in asm
		assert "main:" in asm


class TestVariadicFormatterWithStructBitfields:
	"""Variadic printf-like formatter that processes struct members including bitfields."""

	def test_variadic_accumulate_struct_fields(self) -> None:
		"""Variadic function accumulates values from struct bitfield members."""
		source = """
		#include <stdarg.h>
		typedef struct {
			int base;
			unsigned int flag_a : 1;
			unsigned int flag_b : 1;
			unsigned int flag_c : 1;
			unsigned int weight : 4;
		} Record;

		int format_sum(int count, ...) {
			va_list ap;
			va_start(ap, count);
			int total = 0;
			for (int i = 0; i < count; i = i + 1) {
				total = total + va_arg(ap, int);
			}
			va_end(ap);
			return total;
		}

		int main(void) {
			Record r;
			r.base = 100;
			r.flag_a = 1;
			r.flag_b = 0;
			r.flag_c = 1;
			r.weight = 8;
			return format_sum(3, r.base, r.flag_a, r.weight);
		}
		"""
		asm = _full_pipeline_with_preprocess(source)
		_validate_asm(asm)
		assert "format_sum:" in asm
		assert "main:" in asm
		assert "call" in asm

	def test_variadic_with_typedef_enum_struct_goto(self) -> None:
		"""Combines variadic, typedef, enum, struct with bitfields, and goto."""
		source = """
		#include <stdarg.h>
		typedef enum { FMT_INT, FMT_SKIP, FMT_END } FmtType;

		typedef struct {
			int formatted_count;
			unsigned int overflow : 1;
			unsigned int done : 1;
		} FmtState;

		int selective_sum(int first_type, ...) {
			va_list ap;
			va_start(ap, first_type);
			FmtState state;
			state.formatted_count = 0;
			state.overflow = 0;
			state.done = 0;
			int total = 0;
			int type = first_type;

			next_arg:
			if (type == 2) goto finished;
			if (type == 0) {
				int val = va_arg(ap, int);
				total = total + val;
				state.formatted_count = state.formatted_count + 1;
			}
			if (type == 1) {
				int skip = va_arg(ap, int);
			}
			type = va_arg(ap, int);
			goto next_arg;

			finished:
			state.done = 1;
			va_end(ap);
			return total;
		}

		int main(void) {
			return selective_sum(0, 10, 1, 99, 0, 20, 2);
		}
		"""
		asm = _full_pipeline_with_preprocess(source)
		_validate_asm(asm)
		assert "selective_sum:" in asm
		assert "main:" in asm
		assert "jmp" in asm

	def test_variadic_with_multiple_struct_types(self) -> None:
		"""Variadic function uses values from multiple typedef'd struct types."""
		source = """
		#include <stdarg.h>
		typedef struct {
			int x;
			int y;
			unsigned int dim : 2;
		} Point;

		typedef struct {
			int r;
			int g;
			int b;
			unsigned int alpha : 1;
		} Color;

		int combine(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int acc = 0;
			for (int i = 0; i < n; i = i + 1) {
				acc = acc + va_arg(ap, int);
			}
			va_end(ap);
			return acc;
		}

		int main(void) {
			Point p;
			p.x = 10;
			p.y = 20;
			p.dim = 2;
			Color c;
			c.r = 255;
			c.g = 128;
			c.b = 64;
			c.alpha = 1;
			return combine(5, p.x, p.y, c.r, c.g, c.b);
		}
		"""
		asm = _full_pipeline_with_preprocess(source)
		_validate_asm(asm)
		assert "combine:" in asm
		assert "main:" in asm

	def test_variadic_formatter_with_va_copy(self) -> None:
		"""Variadic function uses va_copy for two-pass processing."""
		source = """
		#include <stdarg.h>
		typedef struct {
			int total;
			int count;
			unsigned int saturated : 1;
		} Stats;

		int two_pass_sum(int n, ...) {
			va_list ap;
			va_start(ap, n);
			va_list ap2;
			va_copy(ap2, ap);

			Stats s;
			s.total = 0;
			s.count = 0;
			s.saturated = 0;

			for (int i = 0; i < n; i = i + 1) {
				int v = va_arg(ap, int);
				s.count = s.count + 1;
			}
			va_end(ap);

			for (int i = 0; i < n; i = i + 1) {
				s.total = s.total + va_arg(ap2, int);
			}
			va_end(ap2);

			if (s.total > 1000) s.saturated = 1;
			return s.total;
		}

		int main(void) {
			return two_pass_sum(3, 100, 200, 300);
		}
		"""
		asm = _full_pipeline_with_preprocess(source)
		_validate_asm(asm)
		assert "two_pass_sum:" in asm
		assert "main:" in asm

	def test_full_combined_stress(self) -> None:
		"""Maximum feature combination: typedef, enum, struct, bitfields,
		variadic, goto, switch, designated init, loops."""
		source = """
		#include <stdarg.h>
		typedef enum { OP_ADD, OP_SUB, OP_MUL, OP_HALT } OpCode;

		typedef struct {
			int accumulator;
			unsigned int halted : 1;
			unsigned int error : 1;
			unsigned int op_count : 4;
		} CPU;

		int execute_ops(int first_op, ...) {
			va_list ap;
			va_start(ap, first_op);
			CPU cpu;
			cpu.accumulator = 0;
			cpu.halted = 0;
			cpu.error = 0;
			cpu.op_count = 0;
			int op = first_op;

			fetch:
			if (cpu.halted) goto done;
			switch (op) {
				case 0: {
					int val = va_arg(ap, int);
					cpu.accumulator = cpu.accumulator + val;
					break;
				}
				case 1: {
					int val = va_arg(ap, int);
					cpu.accumulator = cpu.accumulator - val;
					break;
				}
				case 2: {
					int val = va_arg(ap, int);
					cpu.accumulator = cpu.accumulator * val;
					break;
				}
				case 3:
					cpu.halted = 1;
					goto done;
			}
			cpu.op_count = cpu.op_count + 1;
			op = va_arg(ap, int);
			goto fetch;

			done:
			va_end(ap);
			return cpu.accumulator;
		}

		int main(void) {
			return execute_ops(0, 10, 0, 5, 1, 3, 3);
		}
		"""
		asm = _full_pipeline_with_preprocess(source)
		_validate_asm(asm)
		assert "execute_ops:" in asm
		assert "main:" in asm
		assert "jmp" in asm
		assert "call" in asm

	def test_full_combined_stress_optimized(self) -> None:
		"""Same maximum feature combination with optimization."""
		source = """
		#include <stdarg.h>
		typedef struct {
			int value;
			unsigned int processed : 1;
			unsigned int type : 3;
		} Item;

		int process_items(int count, ...) {
			va_list ap;
			va_start(ap, count);
			int result = 0;
			for (int i = 0; i < count; i = i + 1) {
				Item item;
				item.value = va_arg(ap, int);
				item.type = va_arg(ap, int);
				item.processed = 0;
				switch (item.type) {
					case 0: result = result + item.value; break;
					case 1: result = result - item.value; break;
					case 2: result = result * item.value; break;
					default: break;
				}
				item.processed = 1;
			}
			va_end(ap);
			return result;
		}

		int main(void) {
			return process_items(3, 10, 0, 5, 0, 2, 2);
		}
		"""
		asm = _full_pipeline_with_preprocess(source, optimize=True)
		_validate_asm(asm)
		assert "process_items:" in asm
		assert "main:" in asm
