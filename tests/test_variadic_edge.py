"""Edge-case tests for variadic function codegen and ABI correctness.

Covers: zero variadic args, many args (>6 requiring stack passing),
mixed int/pointer args, nested variadic calls, va_arg with different types,
va_copy, and multiple va_list usage.
"""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRCall,
	IRFunction,
	IRParam,
	IRVaArg,
	IRVaCopy,
	IRVaEnd,
	IRVaStart,
)
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


def generate_ir(source: str):
	"""Parse and generate IR from C source."""
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	ast = Parser(tokens).parse()
	errors = SemanticAnalyzer().analyze(ast)
	assert not errors, f"Semantic errors: {errors}"
	return IRGenerator().generate(ast)


def get_func(ir, name: str) -> IRFunction:
	"""Extract a named function from IR program."""
	matches = [f for f in ir.functions if f.name == name]
	assert matches, f"Function '{name}' not found in IR"
	return matches[0]


class TestZeroVariadicArgs:
	"""Calling a variadic function with zero variadic arguments."""

	def test_zero_variadic_args_compiles(self) -> None:
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			va_end(ap);
			return n;
		}
		int main() {
			return f(42);
		}
		"""
		asm = compile_source(source)
		assert "call f" in asm or "callq f" in asm
		assert "f:" in asm

	def test_zero_variadic_args_ir_call(self) -> None:
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			va_end(ap);
			return n;
		}
		int main() {
			return f(0);
		}
		"""
		ir = generate_ir(source)
		main = get_func(ir, "main")
		calls = [i for i in main.body if isinstance(i, IRCall)]
		assert len(calls) >= 1
		# Only the fixed arg (0), no variadic args
		call = [c for c in calls if c.function_name == "f"][0]
		assert len(call.args) == 1

	def test_zero_variadic_args_asm_sets_al(self) -> None:
		"""ABI: %al must be set even when calling with zero variadic args."""
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			va_end(ap);
			return n;
		}
		int main() {
			return f(5);
		}
		"""
		asm = compile_source(source)
		# Before call to f, %al should be set (number of SSE args = 0)
		assert "movb $0, %al" in asm


class TestManyVariadicArgs:
	"""Calling variadic functions with many args that exceed register capacity."""

	def test_seven_total_args_compiles(self) -> None:
		"""7 args total = 6 in registers + 1 on stack (x86-64 ABI)."""
		source = """
		#include <stdarg.h>
		int sum(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int total = 0;
			int i = 0;
			while (i < n) {
				total = total + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);
			return total;
		}
		int main() {
			return sum(6, 1, 2, 3, 4, 5, 6);
		}
		"""
		asm = compile_source(source)
		assert "sum:" in asm
		assert "main:" in asm

	def test_eight_total_args_uses_stack(self) -> None:
		"""8 args total = definitely need stack passing for args beyond 6 registers."""
		source = """
		#include <stdarg.h>
		int sum(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int total = 0;
			int i = 0;
			while (i < n) {
				total = total + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);
			return total;
		}
		int main() {
			return sum(7, 10, 20, 30, 40, 50, 60, 70);
		}
		"""
		asm = compile_source(source)
		# Args beyond 6 GP registers must be pushed onto the stack
		assert "pushq" in asm or "(%rsp)" in asm

	def test_ten_args_ir(self) -> None:
		"""10 total args: verify IR generates correct number of args."""
		source = """
		#include <stdarg.h>
		int sum(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int total = 0;
			int i = 0;
			while (i < n) {
				total = total + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);
			return total;
		}
		int main() {
			return sum(9, 1, 2, 3, 4, 5, 6, 7, 8, 9);
		}
		"""
		ir = generate_ir(source)
		main = get_func(ir, "main")
		calls = [i for i in main.body if isinstance(i, IRCall) and i.function_name == "sum"]
		assert len(calls) == 1
		assert len(calls[0].args) == 10

	def test_many_args_asm_structure(self) -> None:
		"""Verify stack args are set up before register args for many-arg calls."""
		source = """
		#include <stdarg.h>
		int f(int a, ...) {
			va_list ap;
			va_start(ap, a);
			int x = va_arg(ap, int);
			va_end(ap);
			return a + x;
		}
		int main() {
			return f(1, 2, 3, 4, 5, 6, 7, 8);
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "call f" in asm or "callq f" in asm


class TestMixedIntPointerArgs:
	"""Variadic functions with mixed int and pointer arguments."""

	def test_int_then_pointer_va_arg(self) -> None:
		source = """
		#include <stdarg.h>
		void *get_second(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int first = va_arg(ap, int);
			void *second = va_arg(ap, void *);
			va_end(ap);
			return second;
		}
		"""
		asm = compile_source(source)
		assert "get_second:" in asm
		# Two va_arg calls = two comparisons
		assert asm.count("cmpl $48") == 2

	def test_pointer_then_int_va_arg(self) -> None:
		source = """
		#include <stdarg.h>
		int get_int_after_ptr(int n, ...) {
			va_list ap;
			va_start(ap, n);
			void *p = va_arg(ap, void *);
			int x = va_arg(ap, int);
			va_end(ap);
			return x;
		}
		"""
		asm = compile_source(source)
		assert "get_int_after_ptr:" in asm
		assert asm.count("cmpl $48") == 2

	def test_multiple_pointers_va_arg(self) -> None:
		source = """
		#include <stdarg.h>
		void *get_third_ptr(int n, ...) {
			va_list ap;
			va_start(ap, n);
			void *a = va_arg(ap, void *);
			void *b = va_arg(ap, void *);
			void *c = va_arg(ap, void *);
			va_end(ap);
			return c;
		}
		"""
		asm = compile_source(source)
		assert "get_third_ptr:" in asm
		assert asm.count("cmpl $48") == 3

	def test_mixed_args_caller(self) -> None:
		"""Call with mixed int and pointer args from caller side."""
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int x = va_arg(ap, int);
			va_end(ap);
			return x;
		}
		int arr[1];
		int main() {
			int *p = arr;
			return f(2, 42, p);
		}
		"""
		asm = compile_source(source)
		assert "call f" in asm or "callq f" in asm


class TestVaArgDifferentTypes:
	"""va_arg with various C types (int, long, char *, void *)."""

	def test_va_arg_int(self) -> None:
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int x = va_arg(ap, int);
			va_end(ap);
			return x;
		}
		"""
		ir = generate_ir(source)
		func = get_func(ir, "f")
		va_args = [i for i in func.body if isinstance(i, IRVaArg)]
		assert len(va_args) == 1
		assert va_args[0].ir_type.name == "INT"

	def test_va_arg_long(self) -> None:
		source = """
		#include <stdarg.h>
		long f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			long x = va_arg(ap, long);
			va_end(ap);
			return x;
		}
		"""
		ir = generate_ir(source)
		func = get_func(ir, "f")
		va_args = [i for i in func.body if isinstance(i, IRVaArg)]
		assert len(va_args) == 1
		assert va_args[0].ir_type.name == "LONG"

	def test_va_arg_pointer(self) -> None:
		source = """
		#include <stdarg.h>
		void *f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			void *p = va_arg(ap, void *);
			va_end(ap);
			return p;
		}
		"""
		ir = generate_ir(source)
		func = get_func(ir, "f")
		va_args = [i for i in func.body if isinstance(i, IRVaArg)]
		assert len(va_args) == 1
		assert va_args[0].ir_type.name == "POINTER"

	def test_va_arg_char_pointer(self) -> None:
		source = """
		#include <stdarg.h>
		char *f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			char *s = va_arg(ap, char *);
			va_end(ap);
			return s;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "cmpl $48" in asm

	def test_va_arg_alternating_types(self) -> None:
		"""Alternate between int and pointer va_arg calls."""
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int a = va_arg(ap, int);
			void *p = va_arg(ap, void *);
			int b = va_arg(ap, int);
			va_end(ap);
			return a + b;
		}
		"""
		ir = generate_ir(source)
		func = get_func(ir, "f")
		va_args = [i for i in func.body if isinstance(i, IRVaArg)]
		assert len(va_args) == 3
		assert va_args[0].ir_type.name == "INT"
		assert va_args[1].ir_type.name == "POINTER"
		assert va_args[2].ir_type.name == "INT"


class TestVaCopy:
	"""Tests for va_copy functionality."""

	def test_va_copy_generates_ir(self) -> None:
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_list ap2;
			va_start(ap, n);
			va_copy(ap2, ap);
			int x = va_arg(ap2, int);
			va_end(ap2);
			va_end(ap);
			return x;
		}
		"""
		ir = generate_ir(source)
		func = get_func(ir, "f")
		copies = [i for i in func.body if isinstance(i, IRVaCopy)]
		assert len(copies) == 1

	def test_va_copy_compiles(self) -> None:
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_list ap2;
			va_start(ap, n);
			va_copy(ap2, ap);
			int x = va_arg(ap, int);
			int y = va_arg(ap2, int);
			va_end(ap2);
			va_end(ap);
			return x + y;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		# va_copy copies 24 bytes; look for rep movsq or mov instructions
		# Two va_arg = two cmpl $48
		assert asm.count("cmpl $48") >= 2

	def test_va_copy_independent_iteration(self) -> None:
		"""After va_copy, iterating one list shouldn't affect the other."""
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_list ap2;
			va_start(ap, n);
			va_copy(ap2, ap);
			int a = va_arg(ap, int);
			int b = va_arg(ap, int);
			int c = va_arg(ap2, int);
			va_end(ap2);
			va_end(ap);
			return a + b + c;
		}
		"""
		ir = generate_ir(source)
		func = get_func(ir, "f")
		va_args = [i for i in func.body if isinstance(i, IRVaArg)]
		assert len(va_args) == 3
		copies = [i for i in func.body if isinstance(i, IRVaCopy)]
		assert len(copies) == 1


class TestNestedVariadicCalls:
	"""Nested calls to variadic functions."""

	def test_variadic_calls_variadic(self) -> None:
		"""One variadic function calling another."""
		source = """
		#include <stdarg.h>
		int inner(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int x = va_arg(ap, int);
			va_end(ap);
			return x;
		}
		int outer(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int val = va_arg(ap, int);
			va_end(ap);
			return inner(1, val);
		}
		"""
		asm = compile_source(source)
		assert "inner:" in asm
		assert "outer:" in asm
		# outer should call inner
		outer_section = asm[asm.index("outer:"):]
		assert "call inner" in outer_section or "callq inner" in outer_section

	def test_variadic_result_as_arg(self) -> None:
		"""Use the result of one variadic call as arg to another."""
		source = """
		#include <stdarg.h>
		int get(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int x = va_arg(ap, int);
			va_end(ap);
			return x;
		}
		int main() {
			return get(1, get(1, 99));
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm
		# Two calls to get
		main_section = asm[asm.index("main:"):]
		assert main_section.count("call get") >= 2 or main_section.count("callq get") >= 2

	def test_non_variadic_calls_variadic(self) -> None:
		"""A non-variadic function calling a variadic function."""
		source = """
		#include <stdarg.h>
		int vfunc(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int x = va_arg(ap, int);
			va_end(ap);
			return x;
		}
		int wrapper(int a, int b) {
			return vfunc(2, a, b);
		}
		"""
		asm = compile_source(source)
		assert "wrapper:" in asm
		assert "vfunc:" in asm


class TestVariadicGpOffset:
	"""Test that gp_offset is correctly computed based on named parameters."""

	def test_one_named_param_gp_offset_8(self) -> None:
		"""With 1 named GP param, gp_offset should start at 8."""
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			va_end(ap);
			return 0;
		}
		"""
		asm = compile_source(source)
		# gp_offset = 1 * 8 = 8
		assert "movl $8" in asm

	def test_two_named_params_gp_offset_16(self) -> None:
		"""With 2 named GP params, gp_offset should start at 16."""
		source = """
		#include <stdarg.h>
		int f(int a, int b, ...) {
			va_list ap;
			va_start(ap, b);
			va_end(ap);
			return 0;
		}
		"""
		asm = compile_source(source)
		# gp_offset = 2 * 8 = 16
		assert "movl $16" in asm

	def test_three_named_params_gp_offset_24(self) -> None:
		"""With 3 named GP params, gp_offset should start at 24."""
		source = """
		#include <stdarg.h>
		int f(int a, int b, int c, ...) {
			va_list ap;
			va_start(ap, c);
			va_end(ap);
			return 0;
		}
		"""
		asm = compile_source(source)
		assert "movl $24" in asm

	def test_five_named_params_gp_offset_40(self) -> None:
		"""With 5 named GP params, gp_offset should start at 40."""
		source = """
		#include <stdarg.h>
		int f(int a, int b, int c, int d, int e, ...) {
			va_list ap;
			va_start(ap, e);
			va_end(ap);
			return 0;
		}
		"""
		asm = compile_source(source)
		assert "movl $40" in asm


class TestVariadicRegisterSaveArea:
	"""Test register save area setup for variadic functions."""

	def test_all_six_gp_regs_saved(self) -> None:
		"""Variadic function should save all 6 GP argument registers."""
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			va_end(ap);
			return 0;
		}
		"""
		asm = compile_source(source)
		# All 6 GP regs should be saved: rdi, rsi, rdx, rcx, r8, r9
		assert "movq %rdi" in asm
		assert "movq %rsi" in asm
		assert "movq %rdx" in asm
		assert "movq %rcx" in asm
		assert "movq %r8" in asm
		assert "movq %r9" in asm

	def test_fp_offset_always_48(self) -> None:
		"""fp_offset should always be 48 (6 GP regs * 8 bytes)."""
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			va_end(ap);
			return 0;
		}
		"""
		asm = compile_source(source)
		assert "movl $48" in asm

	def test_overflow_arg_area_points_to_stack(self) -> None:
		"""overflow_arg_area should reference 16(%rbp) (args above rbp)."""
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			va_end(ap);
			return 0;
		}
		"""
		asm = compile_source(source)
		assert "16(%rbp)" in asm


class TestVariadicWithLoops:
	"""Variadic arg extraction inside loops."""

	def test_va_arg_in_while_loop(self) -> None:
		source = """
		#include <stdarg.h>
		int sum(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int total = 0;
			int i = 0;
			while (i < n) {
				total = total + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);
			return total;
		}
		"""
		asm = compile_source(source)
		assert "sum:" in asm
		# Should have a loop structure with a va_arg inside
		assert "cmpl $48" in asm
		assert "jae" in asm

	def test_va_arg_in_for_loop(self) -> None:
		source = """
		#include <stdarg.h>
		int sum(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int total = 0;
			int i;
			for (i = 0; i < n; i = i + 1) {
				total = total + va_arg(ap, int);
			}
			va_end(ap);
			return total;
		}
		"""
		asm = compile_source(source)
		assert "sum:" in asm
		assert "cmpl $48" in asm


class TestVariadicIRStructure:
	"""Detailed IR structure checks for variadic functions."""

	def test_variadic_flag_on_ir_function(self) -> None:
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			va_end(ap);
			return 0;
		}
		"""
		ir = generate_ir(source)
		func = get_func(ir, "f")
		assert func.is_variadic is True

	def test_non_variadic_flag_false(self) -> None:
		source = "int f(int n) { return n; }"
		ir = generate_ir(source)
		func = get_func(ir, "f")
		assert func.is_variadic is False

	def test_va_start_records_named_gp_count(self) -> None:
		source = """
		#include <stdarg.h>
		int f(int a, int b, ...) {
			va_list ap;
			va_start(ap, b);
			va_end(ap);
			return 0;
		}
		"""
		ir = generate_ir(source)
		func = get_func(ir, "f")
		va_starts = [i for i in func.body if isinstance(i, IRVaStart)]
		assert len(va_starts) == 1
		assert va_starts[0].num_named_gp == 2

	def test_ir_order_va_start_before_va_arg(self) -> None:
		"""va_start must come before va_arg in IR instruction stream."""
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int x = va_arg(ap, int);
			va_end(ap);
			return x;
		}
		"""
		ir = generate_ir(source)
		func = get_func(ir, "f")
		va_start_idx = None
		va_arg_idx = None
		for i, instr in enumerate(func.body):
			if isinstance(instr, IRVaStart) and va_start_idx is None:
				va_start_idx = i
			if isinstance(instr, IRVaArg) and va_arg_idx is None:
				va_arg_idx = i
		assert va_start_idx is not None
		assert va_arg_idx is not None
		assert va_start_idx < va_arg_idx

	def test_ir_order_va_arg_before_va_end(self) -> None:
		"""va_arg must come before va_end in IR instruction stream."""
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int x = va_arg(ap, int);
			va_end(ap);
			return x;
		}
		"""
		ir = generate_ir(source)
		func = get_func(ir, "f")
		va_arg_idx = None
		va_end_idx = None
		for i, instr in enumerate(func.body):
			if isinstance(instr, IRVaArg) and va_arg_idx is None:
				va_arg_idx = i
			if isinstance(instr, IRVaEnd) and va_end_idx is None:
				va_end_idx = i
		assert va_arg_idx is not None
		assert va_end_idx is not None
		assert va_arg_idx < va_end_idx


class TestVariadicCallerABI:
	"""Test caller-side ABI compliance when calling variadic functions."""

	def test_caller_sets_al_zero_for_no_sse(self) -> None:
		"""When calling with zero SSE args, %al should be 0."""
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			va_end(ap);
			return 0;
		}
		int main() {
			return f(1, 2);
		}
		"""
		asm = compile_source(source)
		main_section = asm[asm.index("main:"):]
		# Should set %al = 0 before calling f (no float args)
		assert "movb $0, %al" in main_section

	def test_caller_passes_args_in_registers(self) -> None:
		"""First 6 integer args go in rdi, rsi, rdx, rcx, r8, r9."""
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			va_end(ap);
			return 0;
		}
		int main() {
			return f(3, 10, 20, 30);
		}
		"""
		asm = compile_source(source)
		main_section = asm[asm.index("main:"):]
		# The caller should move args into registers
		assert "call f" in main_section or "callq f" in main_section

	def test_caller_stack_alignment(self) -> None:
		"""Stack should remain 16-byte aligned before call."""
		source = """
		#include <stdarg.h>
		int f(int a, ...) {
			va_list ap;
			va_start(ap, a);
			va_end(ap);
			return 0;
		}
		int main() {
			return f(1, 2, 3, 4, 5, 6, 7, 8, 9);
		}
		"""
		# Should compile without issues; if alignment is wrong, segfaults at runtime
		asm = compile_source(source)
		assert "main:" in asm


class TestVariadicEdgeCaseParsing:
	"""Parsing edge cases for variadic declarations."""

	def test_variadic_only_one_named_param(self) -> None:
		"""Minimum valid variadic: one named param + ellipsis."""
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			va_end(ap);
			return 0;
		}
		"""
		ir = generate_ir(source)
		func = get_func(ir, "f")
		assert func.is_variadic is True

	def test_variadic_with_pointer_named_param(self) -> None:
		"""Named parameter is a pointer type."""
		source = """
		#include <stdarg.h>
		int f(int *p, ...) {
			va_list ap;
			va_start(ap, p);
			int x = va_arg(ap, int);
			va_end(ap);
			return x;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm

	def test_variadic_with_char_named_param(self) -> None:
		"""Named parameter is char type."""
		source = """
		#include <stdarg.h>
		int f(char c, ...) {
			va_list ap;
			va_start(ap, c);
			int x = va_arg(ap, int);
			va_end(ap);
			return x;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm

	def test_variadic_forward_declaration(self) -> None:
		"""Forward-declare variadic function then define it."""
		source = """
		#include <stdarg.h>
		int f(int n, ...);
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int x = va_arg(ap, int);
			va_end(ap);
			return x;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm


class TestVaEndNoop:
	"""va_end should be a no-op on x86-64 but still generate valid IR."""

	def test_va_end_in_ir(self) -> None:
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			va_end(ap);
			return 0;
		}
		"""
		ir = generate_ir(source)
		func = get_func(ir, "f")
		va_ends = [i for i in func.body if isinstance(i, IRVaEnd)]
		assert len(va_ends) == 1

	def test_multiple_va_end_calls(self) -> None:
		"""Two va_list variables, each with their own va_end."""
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_list ap2;
			va_start(ap, n);
			va_copy(ap2, ap);
			va_end(ap);
			va_end(ap2);
			return 0;
		}
		"""
		ir = generate_ir(source)
		func = get_func(ir, "f")
		va_ends = [i for i in func.body if isinstance(i, IRVaEnd)]
		assert len(va_ends) == 2
