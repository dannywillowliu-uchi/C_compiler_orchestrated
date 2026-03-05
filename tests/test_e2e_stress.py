"""End-to-end stress tests combining multiple compiler features.

Each test exercises feature interactions that could expose bugs when
features are composed: variadic + structs, goto + switch/loops,
bitfields + type modifiers, nested struct + pointer arithmetic,
compound expressions + side effects, do-while + break/continue in switch.
"""

from compiler.__main__ import compile_source
from compiler.preprocessor import Preprocessor
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer
from compiler.ir_gen import IRGenerator
from compiler.codegen import CodeGenerator


def _full_pipeline(source: str, optimize: bool = False) -> str:
	return compile_source(source, optimize=optimize)


def _full_pipeline_with_preprocess(source: str, optimize: bool = False) -> str:
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	ast = Parser(tokens).parse()
	errors = SemanticAnalyzer().analyze(ast)
	assert not errors, f"Semantic errors: {errors}"
	ir = IRGenerator().generate(ast)
	return CodeGenerator().generate(ir)


class TestVariadicWithStructs:
	"""Variadic functions that accept or process struct-related values."""

	def test_variadic_function_with_int_args_and_struct_decl(self) -> None:
		source = """
		#include <stdarg.h>
		struct Point { int x; int y; };
		int sum_ints(int count, ...) {
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
			struct Point p;
			p.x = 10;
			p.y = 20;
			int result = sum_ints(2, p.x, p.y);
			return result;
		}
		"""
		asm = _full_pipeline_with_preprocess(source)
		assert "sum_ints:" in asm
		assert "main:" in asm
		assert "ret" in asm
		assert "call" in asm

	def test_variadic_with_struct_local_and_member_access(self) -> None:
		source = """
		#include <stdarg.h>
		struct Pair { int a; int b; };
		int first_arg(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int val = va_arg(ap, int);
			va_end(ap);
			return val;
		}
		int main(void) {
			struct Pair p;
			p.a = 42;
			p.b = 99;
			return first_arg(1, p.a);
		}
		"""
		asm = _full_pipeline_with_preprocess(source)
		assert "first_arg:" in asm
		assert "main:" in asm
		assert "call" in asm

	def test_variadic_sum_with_loop_and_struct(self) -> None:
		source = """
		#include <stdarg.h>
		struct Counter { int count; };
		int sum_n(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int s = 0;
			for (int i = 0; i < n; i = i + 1) {
				s = s + va_arg(ap, int);
			}
			va_end(ap);
			return s;
		}
		int main(void) {
			struct Counter c;
			c.count = 3;
			return sum_n(c.count, 1, 2, 3);
		}
		"""
		asm = _full_pipeline_with_preprocess(source)
		assert "sum_n:" in asm
		assert "main:" in asm


class TestGotoWithSwitchAndLoops:
	"""Goto interacting with switch statements and loop constructs."""

	def test_goto_out_of_switch(self) -> None:
		source = """
		int main(void) {
			int x = 2;
			int result = 0;
			switch (x) {
				case 1: result = 10; break;
				case 2: goto done;
				case 3: result = 30; break;
			}
			result = 99;
			done: return result;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "jmp" in asm
		assert "ret" in asm

	def test_goto_skips_loop_body(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			int sum = 0;
			goto skip_loop;
			while (i < 10) {
				sum = sum + i;
				i = i + 1;
			}
			skip_loop: return sum;
		}
		"""
		asm = _full_pipeline(source)
		assert "jmp" in asm
		assert "main:" in asm

	def test_goto_within_for_loop(self) -> None:
		source = """
		int main(void) {
			int total = 0;
			for (int i = 0; i < 5; i = i + 1) {
				if (i == 3) goto end_loop;
				total = total + i;
			}
			end_loop: return total;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "jmp" in asm

	def test_goto_backward_with_switch(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			int count = 0;
			retry:
			count = count + 1;
			switch (x) {
				case 0:
					x = 1;
					goto retry;
				case 1:
					return count;
			}
			return -1;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "jmp" in asm

	def test_goto_between_switch_cases_and_loop(self) -> None:
		source = """
		int main(void) {
			int state = 0;
			int result = 0;
			switch (state) {
				case 0:
					result = 1;
					goto after;
				case 1:
					result = 2;
					break;
			}
			after:
			for (int i = 0; i < 3; i = i + 1) {
				result = result + 1;
			}
			return result;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm


class TestBitfieldsWithTypeModifiers:
	"""Bitfield structs combined with signed/unsigned/short/long modifiers."""

	def test_unsigned_int_bitfield_in_function(self) -> None:
		source = """
		struct Flags {
			unsigned int read : 1;
			unsigned int write : 1;
			unsigned int exec : 1;
		};
		int main(void) {
			struct Flags f;
			return 0;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_signed_and_unsigned_bitfield_mix(self) -> None:
		source = """
		struct Mixed {
			signed int val : 7;
			unsigned int flags : 3;
			int plain : 4;
		};
		int main(void) {
			struct Mixed m;
			return 0;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm

	def test_short_bitfield_with_regular_member(self) -> None:
		source = """
		struct Record {
			short tag : 4;
			int value;
			unsigned short extra : 8;
		};
		int main(void) {
			struct Record r;
			return 0;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_bitfield_struct_with_long_modifier(self) -> None:
		source = """
		struct Wide {
			long big : 16;
			unsigned int small : 2;
		};
		int main(void) {
			struct Wide w;
			return 0;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm

	def test_multiple_bitfield_structs_in_function(self) -> None:
		source = """
		struct A { unsigned int x : 3; };
		struct B { signed int y : 5; unsigned int z : 2; };
		int main(void) {
			struct A a;
			struct B b;
			return 0;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm


class TestNestedStructAccessWithPointerArithmetic:
	"""Nested struct member access combined with pointer operations."""

	def test_nested_struct_member_access(self) -> None:
		source = """
		struct Inner { int val; };
		struct Outer { struct Inner inner; int extra; };
		int main(void) {
			struct Outer o;
			o.inner.val = 42;
			o.extra = 10;
			return o.inner.val + o.extra;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_struct_with_array_and_member_access(self) -> None:
		source = """
		struct Vec { int data; int len; };
		int main(void) {
			struct Vec v;
			v.data = 100;
			v.len = 5;
			int result = v.data + v.len;
			return result;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "addq" in asm

	def test_struct_pointer_deref_and_member(self) -> None:
		source = """
		struct Node { int value; int next; };
		int main(void) {
			struct Node n;
			n.value = 7;
			n.next = 0;
			int *p = &n.value;
			return *p;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_deeply_nested_structs(self) -> None:
		source = """
		struct A { int x; };
		struct B { struct A a; int y; };
		struct C { struct B b; int z; };
		int main(void) {
			struct C c;
			c.b.a.x = 1;
			c.b.y = 2;
			c.z = 3;
			return c.b.a.x + c.b.y + c.z;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_struct_with_sizeof(self) -> None:
		source = """
		struct Pair { int first; int second; };
		int main(void) {
			int sz = sizeof(struct Pair);
			return sz;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm


class TestCompoundExpressionsWithSideEffects:
	"""Compound assignments, postfix/prefix ops, comma expressions, ternary."""

	def test_compound_assignment_chain(self) -> None:
		source = """
		int main(void) {
			int x = 10;
			x += 5;
			x -= 3;
			x *= 2;
			return x;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_postfix_increment_in_expression(self) -> None:
		source = """
		int main(void) {
			int a = 5;
			int b = a++;
			return a + b;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_comma_expression_evaluates_all(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			int y = 0;
			x = (y = 5, y + 10);
			return x;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm

	def test_ternary_with_compound_assignment(self) -> None:
		source = """
		int main(void) {
			int x = 5;
			int y = (x > 3) ? x * 2 : x + 1;
			y += 1;
			return y;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_nested_ternary_expressions(self) -> None:
		source = """
		int main(void) {
			int a = 3;
			int b = 5;
			int result = (a > b) ? a : (b > 10) ? b : a + b;
			return result;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm

	def test_postfix_in_array_subscript(self) -> None:
		source = """
		int main(void) {
			int arr[4];
			arr[0] = 10;
			arr[1] = 20;
			arr[2] = 30;
			arr[3] = 40;
			int i = 1;
			int val = arr[i++];
			return val + i;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_compound_assignment_with_function_call(self) -> None:
		source = """
		int add(int a, int b) { return a + b; }
		int main(void) {
			int x = 10;
			x += add(3, 4);
			return x;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "call" in asm


class TestDoWhileWithBreakContinueInSwitch:
	"""do-while loops containing switch with break/continue interactions."""

	def test_do_while_with_switch_break(self) -> None:
		source = """
		int main(void) {
			int x = 0;
			int count = 0;
			do {
				switch (x) {
					case 0:
						x = 1;
						break;
					case 1:
						x = 2;
						break;
					default:
						x = 3;
						break;
				}
				count = count + 1;
			} while (x < 3);
			return count;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_do_while_with_continue(self) -> None:
		source = """
		int main(void) {
			int i = 0;
			int sum = 0;
			do {
				i = i + 1;
				if (i == 3) continue;
				sum = sum + i;
			} while (i < 5);
			return sum;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "jmp" in asm

	def test_do_while_with_nested_for_and_break(self) -> None:
		source = """
		int main(void) {
			int outer = 0;
			int total = 0;
			do {
				for (int j = 0; j < 3; j = j + 1) {
					if (j == 2) break;
					total = total + 1;
				}
				outer = outer + 1;
			} while (outer < 2);
			return total;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_switch_in_while_in_do_while(self) -> None:
		source = """
		int main(void) {
			int mode = 0;
			int result = 0;
			do {
				int i = 0;
				while (i < 2) {
					switch (mode) {
						case 0: result = result + 1; break;
						case 1: result = result + 10; break;
					}
					i = i + 1;
				}
				mode = mode + 1;
			} while (mode < 2);
			return result;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm


class TestComplexFeatureCombinations:
	"""Stress tests combining three or more features at once."""

	def test_struct_with_goto_and_switch(self) -> None:
		source = """
		struct State { int phase; int value; };
		int main(void) {
			struct State s;
			s.phase = 0;
			s.value = 0;
			restart:
			switch (s.phase) {
				case 0:
					s.value = 10;
					s.phase = 1;
					goto restart;
				case 1:
					s.value = s.value + 20;
					s.phase = 2;
					goto restart;
				case 2:
					return s.value;
			}
			return -1;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "jmp" in asm
		assert "ret" in asm

	def test_enum_with_switch_and_struct(self) -> None:
		source = """
		enum Color { RED, GREEN, BLUE };
		struct Pixel { int color; int brightness; };
		int main(void) {
			struct Pixel px;
			px.color = GREEN;
			px.brightness = 100;
			int result = 0;
			switch (px.color) {
				case RED: result = 1; break;
				case GREEN: result = 2; break;
				case BLUE: result = 3; break;
			}
			return result + px.brightness;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_for_loop_with_ternary_and_compound_assign(self) -> None:
		source = """
		int main(void) {
			int sum = 0;
			for (int i = 0; i < 10; i = i + 1) {
				int add = (i % 2 == 0) ? i : 0;
				sum += add;
			}
			return sum;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_nested_loops_with_break_continue_and_goto(self) -> None:
		source = """
		int main(void) {
			int total = 0;
			for (int i = 0; i < 5; i = i + 1) {
				if (i == 4) goto done;
				for (int j = 0; j < 3; j = j + 1) {
					if (j == 1) continue;
					if (j == 2) break;
					total = total + 1;
				}
			}
			done: return total;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "jmp" in asm

	def test_typedef_with_struct_and_function(self) -> None:
		source = """
		struct Point { int x; int y; };
		typedef struct Point Point;
		int distance_sq(Point a, Point b) {
			int dx = a.x - b.x;
			int dy = a.y - b.y;
			return dx * dx + dy * dy;
		}
		int main(void) {
			Point p1;
			p1.x = 3;
			p1.y = 4;
			Point p2;
			p2.x = 0;
			p2.y = 0;
			return distance_sq(p1, p2);
		}
		"""
		asm = _full_pipeline(source)
		assert "distance_sq:" in asm
		assert "main:" in asm
		assert "call" in asm

	def test_cast_with_sizeof_and_pointer(self) -> None:
		source = """
		int main(void) {
			int x = 42;
			int *p = &x;
			long sz = (long)sizeof(int);
			int val = *p;
			return val + (int)sz;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_while_with_postfix_and_array(self) -> None:
		source = """
		int main(void) {
			int arr[5];
			int i = 0;
			while (i < 5) {
				arr[i] = i * i;
				i++;
			}
			return arr[3];
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_union_with_struct_member(self) -> None:
		source = """
		struct Header { int tag; int size; };
		union Data { int ival; char cval; };
		int main(void) {
			struct Header h;
			h.tag = 1;
			h.size = 4;
			union Data d;
			d.ival = 255;
			return h.tag + d.ival;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm


class TestOptimizerStress:
	"""Tests that exercise the optimizer with complex feature combinations."""

	def test_optimized_loop_with_struct(self) -> None:
		source = """
		struct Acc { int total; };
		int main(void) {
			struct Acc a;
			a.total = 0;
			for (int i = 0; i < 10; i = i + 1) {
				a.total = a.total + i;
			}
			return a.total;
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		assert "main:" in asm
		assert "ret" in asm

	def test_optimized_ternary_chain(self) -> None:
		source = """
		int main(void) {
			int x = 5;
			int y = (x > 10) ? 100 : (x > 3) ? 50 : 0;
			return y;
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		assert "main:" in asm
		assert "ret" in asm

	def test_optimized_switch_with_many_cases(self) -> None:
		source = """
		int classify(int x) {
			switch (x) {
				case 0: return 0;
				case 1: return 10;
				case 2: return 20;
				case 3: return 30;
				case 4: return 40;
				case 5: return 50;
				default: return -1;
			}
		}
		int main(void) {
			return classify(3);
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		assert "classify:" in asm
		assert "main:" in asm
		assert "call" in asm

	def test_optimized_nested_struct_access(self) -> None:
		source = """
		struct Inner { int x; };
		struct Outer { struct Inner i; int y; };
		int main(void) {
			struct Outer o;
			o.i.x = 5;
			o.y = 10;
			return o.i.x + o.y;
		}
		"""
		asm = _full_pipeline(source, optimize=True)
		assert "main:" in asm
		assert "ret" in asm
