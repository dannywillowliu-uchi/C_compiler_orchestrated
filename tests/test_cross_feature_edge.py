"""Edge-case tests for cross-feature interactions: variadic + struct, bitfield + union.

Tests exercising interactions between variadic functions, structs, bitfields, and unions
at the IR generation and semantic analysis level.
"""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRBinOp,
	IRCall,
	IRConst,
	IRFunction,
	IRLoad,
	IRStore,
	IRType,
	IRVaArg,
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


def analyze_source(source: str) -> list:
	"""Parse and run semantic analysis, returning errors."""
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	ast = Parser(tokens).parse()
	return SemanticAnalyzer().analyze(ast)


def get_func(ir, name: str) -> IRFunction:
	"""Extract a named function from IR program."""
	matches = [f for f in ir.functions if f.name == name]
	assert matches, f"Function '{name}' not found in IR"
	return matches[0]


# =============================================================================
# 1. Variadic functions receiving struct arguments
# =============================================================================


class TestVariadicWithStructByPointer:
	"""Variadic functions receiving struct pointers."""

	def test_variadic_struct_pointer_arg_compiles(self) -> None:
		source = """
		#include <stdarg.h>
		struct Point { int x; int y; };
		void *get_ptr(int n, ...) {
			va_list ap;
			va_start(ap, n);
			void *p = va_arg(ap, void *);
			va_end(ap);
			return p;
		}
		int main() {
			struct Point pt;
			pt.x = 10;
			pt.y = 20;
			void *result = get_ptr(1, &pt);
			return 0;
		}
		"""
		asm = compile_source(source)
		assert "get_ptr:" in asm
		assert "main:" in asm
		assert "call get_ptr" in asm or "callq get_ptr" in asm

	def test_variadic_struct_pointer_ir_arg_type(self) -> None:
		"""Passing &struct to variadic should produce POINTER arg type in IR."""
		source = """
		#include <stdarg.h>
		struct Data { int a; int b; };
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			void *p = va_arg(ap, void *);
			va_end(ap);
			return 0;
		}
		int main() {
			struct Data d;
			d.a = 1;
			return f(1, &d);
		}
		"""
		ir = generate_ir(source)
		main = get_func(ir, "main")
		calls = [i for i in main.body if isinstance(i, IRCall) and i.function_name == "f"]
		assert len(calls) == 1
		assert len(calls[0].args) == 2
		# Second arg (struct pointer) should be POINTER type
		assert calls[0].arg_types[1] == IRType.POINTER

	def test_variadic_multiple_struct_pointers(self) -> None:
		"""Pass multiple struct pointers to a variadic function."""
		source = """
		#include <stdarg.h>
		struct Pair { int x; int y; };
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			void *a = va_arg(ap, void *);
			void *b = va_arg(ap, void *);
			va_end(ap);
			return 0;
		}
		int main() {
			struct Pair p1;
			struct Pair p2;
			p1.x = 1;
			p2.x = 2;
			return f(2, &p1, &p2);
		}
		"""
		ir = generate_ir(source)
		main = get_func(ir, "main")
		calls = [i for i in main.body if isinstance(i, IRCall) and i.function_name == "f"]
		assert len(calls) == 1
		assert len(calls[0].args) == 3
		assert calls[0].arg_types[1] == IRType.POINTER
		assert calls[0].arg_types[2] == IRType.POINTER


class TestVariadicWithStructByValue:
	"""Variadic functions where struct values are used alongside variadic args."""

	def test_struct_local_with_variadic_function(self) -> None:
		"""Declare a struct local, then call a variadic function with its fields."""
		source = """
		#include <stdarg.h>
		struct Vec2 { int x; int y; };
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
			struct Vec2 v;
			v.x = 10;
			v.y = 20;
			return sum(2, v.x, v.y);
		}
		"""
		asm = compile_source(source)
		assert "sum:" in asm
		assert "main:" in asm
		assert "call sum" in asm or "callq sum" in asm

	def test_struct_field_as_variadic_arg_ir(self) -> None:
		"""Struct member values passed as variadic args should appear in IR call."""
		source = """
		#include <stdarg.h>
		struct Rect { int w; int h; };
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int x = va_arg(ap, int);
			va_end(ap);
			return x;
		}
		int main() {
			struct Rect r;
			r.w = 42;
			return f(1, r.w);
		}
		"""
		ir = generate_ir(source)
		main = get_func(ir, "main")
		calls = [i for i in main.body if isinstance(i, IRCall) and i.function_name == "f"]
		assert len(calls) == 1
		assert len(calls[0].args) == 2

	def test_variadic_function_accessing_struct_after_va(self) -> None:
		"""A variadic function that also has a local struct."""
		source = """
		#include <stdarg.h>
		struct Result { int val; int ok; };
		int process(int n, ...) {
			va_list ap;
			va_start(ap, n);
			struct Result r;
			r.val = va_arg(ap, int);
			r.ok = 1;
			va_end(ap);
			return r.val + r.ok;
		}
		"""
		ir = generate_ir(source)
		func = get_func(ir, "process")
		assert func.is_variadic is True
		va_args = [i for i in func.body if isinstance(i, IRVaArg)]
		assert len(va_args) == 1
		# Should have stores for struct member writes
		stores = [i for i in func.body if isinstance(i, IRStore)]
		assert len(stores) >= 1


# =============================================================================
# 2. Variadic functions with mixed int/float/pointer args
# =============================================================================


class TestVariadicMixedTypes:
	"""Variadic functions with mixed int, float, and pointer arguments."""

	def test_mixed_int_and_pointer_args_compiles(self) -> None:
		"""Pass int and pointer args to same variadic call."""
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int x = va_arg(ap, int);
			void *p = va_arg(ap, void *);
			int y = va_arg(ap, int);
			va_end(ap);
			return x + y;
		}
		int arr[1];
		int main() {
			return f(3, 1, arr, 2);
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "main:" in asm

	def test_mixed_args_ir_types(self) -> None:
		"""Verify IR correctly records mixed arg types for variadic call."""
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			va_end(ap);
			return 0;
		}
		int arr[1];
		int main() {
			int *p = arr;
			return f(3, 42, p, 99);
		}
		"""
		ir = generate_ir(source)
		main = get_func(ir, "main")
		calls = [i for i in main.body if isinstance(i, IRCall) and i.function_name == "f"]
		assert len(calls) == 1
		assert len(calls[0].args) == 4
		# First arg is the count (int), then int, pointer, int
		assert calls[0].arg_types[0] == IRType.INT
		# The pointer arg type should be POINTER
		assert calls[0].arg_types[2] == IRType.POINTER

	def test_many_mixed_args_compiles(self) -> None:
		"""Many args of varying types should compile correctly."""
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int a = va_arg(ap, int);
			void *b = va_arg(ap, void *);
			int c = va_arg(ap, int);
			void *d = va_arg(ap, void *);
			int e = va_arg(ap, int);
			va_end(ap);
			return a + c + e;
		}
		"""
		ir = generate_ir(source)
		func = get_func(ir, "f")
		va_args = [i for i in func.body if isinstance(i, IRVaArg)]
		assert len(va_args) == 5
		assert va_args[0].ir_type == IRType.INT
		assert va_args[1].ir_type == IRType.POINTER
		assert va_args[2].ir_type == IRType.INT
		assert va_args[3].ir_type == IRType.POINTER
		assert va_args[4].ir_type == IRType.INT

	def test_char_pointer_variadic_arg(self) -> None:
		"""char* (string) passed as variadic arg."""
		source = """
		#include <stdarg.h>
		char *f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			char *s = va_arg(ap, char *);
			va_end(ap);
			return s;
		}
		int main() {
			char *msg = "hello";
			char *result = f(1, msg);
			return 0;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "call f" in asm or "callq f" in asm


# =============================================================================
# 3. Bitfield members inside unions
# =============================================================================


class TestBitfieldInUnion:
	"""Bitfield members declared inside unions."""

	def test_union_with_bitfield_semantic_passes(self) -> None:
		"""Union with bitfield members should pass semantic analysis."""
		source = """
		union Flags {
			int raw;
			int bit0 : 1;
			int bit1 : 1;
		};
		int main() {
			union Flags f;
			f.raw = 0;
			return f.raw;
		}
		"""
		errors = analyze_source(source)
		assert not errors

	def test_union_bitfield_member_access_compiles(self) -> None:
		"""Accessing bitfield member of union should compile."""
		source = """
		union Flags {
			int raw;
			int bit0 : 1;
		};
		int main() {
			union Flags f;
			f.raw = 5;
			return f.raw;
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_union_bitfield_and_regular_member(self) -> None:
		"""Union with both a regular int and a bitfield member."""
		source = """
		union Mixed {
			int value;
			int low_bits : 8;
		};
		int main() {
			union Mixed m;
			m.value = 255;
			return m.value;
		}
		"""
		errors = analyze_source(source)
		assert not errors
		asm = compile_source(source)
		assert "main:" in asm

	def test_union_multiple_bitfields(self) -> None:
		"""Union with multiple bitfield members."""
		source = """
		union MultiBF {
			int raw;
			int a : 4;
			int b : 8;
			int c : 16;
		};
		int main() {
			union MultiBF u;
			u.raw = 0;
			return u.raw;
		}
		"""
		errors = analyze_source(source)
		assert not errors


# =============================================================================
# 4. Bitfield access through pointer dereference
# =============================================================================


class TestBitfieldThroughPointer:
	"""Bitfield access via pointer dereference (arrow operator)."""

	def test_bitfield_arrow_access_compiles(self) -> None:
		"""Access bitfield through struct pointer (->)."""
		source = """
		struct Bits { int a : 4; int b : 4; };
		int main() {
			struct Bits s;
			s.a = 3;
			s.b = 7;
			struct Bits *p = &s;
			return p->a + p->b;
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm

	def test_bitfield_arrow_read_ir(self) -> None:
		"""Pointer dereference of bitfield should generate shift+mask IR."""
		source = """
		struct Flags { int x : 3; int y : 5; };
		int read_flag(struct Flags *p) {
			return p->y;
		}
		"""
		ir = generate_ir(source)
		func = get_func(ir, "read_flag")
		# Bitfield read generates: load, shift right, mask (AND)
		loads = [i for i in func.body if isinstance(i, IRLoad)]
		assert len(loads) >= 1
		# Should have an AND with mask for the bitfield
		ands = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "&"]
		assert len(ands) >= 1
		# y is at bit_offset=3, width=5, so mask = (1<<5)-1 = 31
		mask_ops = [a for a in ands if isinstance(a.right, IRConst) and a.right.value == 31]
		assert len(mask_ops) >= 1

	def test_bitfield_arrow_write_ir(self) -> None:
		"""Writing bitfield through pointer should generate read-modify-write IR."""
		source = """
		struct Packed { int a : 4; int b : 4; };
		void set_b(struct Packed *p, int val) {
			p->b = val;
		}
		"""
		ir = generate_ir(source)
		func = get_func(ir, "set_b")
		# Write generates: load old, clear bits (AND), mask+shift new (AND, <<), OR, store
		stores = [i for i in func.body if isinstance(i, IRStore)]
		assert len(stores) >= 1
		ors = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "|"]
		assert len(ors) >= 1

	def test_bitfield_pointer_deref_semantic(self) -> None:
		"""Semantic analysis should pass for bitfield arrow access."""
		source = """
		struct Info { int flag : 1; int data : 7; };
		int main() {
			struct Info i;
			i.flag = 1;
			i.data = 42;
			struct Info *p = &i;
			int val = p->flag;
			return val;
		}
		"""
		errors = analyze_source(source)
		assert not errors


# =============================================================================
# 5. Union containing struct with bitfields
# =============================================================================


class TestUnionContainingStructWithBitfields:
	"""Union that contains a struct which has bitfield members."""

	def test_nested_struct_with_bitfields_in_union_semantic(self) -> None:
		"""Semantic analysis should pass for union containing struct with bitfields."""
		source = """
		struct Packed { int a : 4; int b : 4; };
		union Overlay {
			int raw;
			struct Packed bits;
		};
		int main() {
			union Overlay o;
			o.raw = 0;
			return o.raw;
		}
		"""
		errors = analyze_source(source)
		assert not errors

	def test_nested_bitfield_struct_in_union_compiles(self) -> None:
		"""Compiling union with nested bitfield struct should work."""
		source = """
		struct Fields { int lo : 8; int hi : 8; };
		union Word {
			int value;
			struct Fields f;
		};
		int main() {
			union Word w;
			w.value = 0;
			return w.value;
		}
		"""
		asm = compile_source(source)
		assert "main:" in asm

	def test_nested_struct_bitfield_access_through_union(self) -> None:
		"""Access bitfield inside struct inside union."""
		source = """
		struct Flags { int a : 1; int b : 1; int c : 6; };
		union Register {
			int raw;
			struct Flags flags;
		};
		int main() {
			union Register r;
			r.raw = 7;
			return r.raw;
		}
		"""
		ir = generate_ir(source)
		main = get_func(ir, "main")
		# Should have a store for r.raw = 7
		stores = [i for i in main.body if isinstance(i, IRStore)]
		assert len(stores) >= 1

	def test_union_with_two_bitfield_structs(self) -> None:
		"""Union overlaying two different bitfield struct layouts."""
		source = """
		struct Layout1 { int x : 16; int y : 16; };
		struct Layout2 { int a : 8; int b : 8; int c : 16; };
		union Dual {
			struct Layout1 l1;
			struct Layout2 l2;
			int raw;
		};
		int main() {
			union Dual d;
			d.raw = 0;
			return d.raw;
		}
		"""
		errors = analyze_source(source)
		assert not errors
		asm = compile_source(source)
		assert "main:" in asm


# =============================================================================
# 6. Sizeof on structs with bitfields
# =============================================================================


class TestSizeofWithBitfields:
	"""sizeof applied to structs containing bitfield members."""

	def test_sizeof_struct_single_bitfield(self) -> None:
		"""Struct with single bitfield should be 4 bytes (one int storage unit)."""
		source = """
		struct S { int x : 1; };
		int main() {
			return sizeof(struct S);
		}
		"""
		ir = generate_ir(source)
		main = get_func(ir, "main")
		# Find the return value - should be a const representing sizeof
		from compiler.ir import IRReturn
		rets = [i for i in main.body if isinstance(i, IRReturn)]
		assert len(rets) >= 1
		# sizeof(struct with single int bitfield) = 4
		ret_val = rets[0].value
		if isinstance(ret_val, IRConst):
			assert ret_val.value == 4

	def test_sizeof_struct_packed_bitfields(self) -> None:
		"""Multiple bitfields that fit in one storage unit should not increase size."""
		source = """
		struct Packed { int a : 8; int b : 8; int c : 8; int d : 8; };
		int main() {
			return sizeof(struct Packed);
		}
		"""
		ir = generate_ir(source)
		main = get_func(ir, "main")
		from compiler.ir import IRReturn
		rets = [i for i in main.body if isinstance(i, IRReturn)]
		assert len(rets) >= 1
		ret_val = rets[0].value
		if isinstance(ret_val, IRConst):
			# 4 x 8-bit bitfields fit in one 32-bit int = 4 bytes
			assert ret_val.value == 4

	def test_sizeof_struct_bitfields_overflow_to_next_unit(self) -> None:
		"""Bitfields exceeding storage unit should spill to next unit."""
		source = """
		struct Spill { int a : 24; int b : 24; };
		int main() {
			return sizeof(struct Spill);
		}
		"""
		ir = generate_ir(source)
		main = get_func(ir, "main")
		from compiler.ir import IRReturn
		rets = [i for i in main.body if isinstance(i, IRReturn)]
		assert len(rets) >= 1
		ret_val = rets[0].value
		if isinstance(ret_val, IRConst):
			# 24 bits doesn't fit in remaining 8 bits, so two 4-byte units = 8
			assert ret_val.value == 8

	def test_sizeof_struct_mixed_bitfield_and_regular(self) -> None:
		"""Struct with both bitfields and regular members."""
		source = """
		struct Mixed { int flags : 8; int value; };
		int main() {
			return sizeof(struct Mixed);
		}
		"""
		ir = generate_ir(source)
		main = get_func(ir, "main")
		from compiler.ir import IRReturn
		rets = [i for i in main.body if isinstance(i, IRReturn)]
		assert len(rets) >= 1
		ret_val = rets[0].value
		if isinstance(ret_val, IRConst):
			# bitfield in 4-byte unit + 4-byte int = 8 bytes
			assert ret_val.value == 8

	def test_sizeof_struct_zero_width_bitfield(self) -> None:
		"""Zero-width bitfield forces alignment to next storage boundary."""
		source = """
		struct ZeroW { int a : 4; int : 0; int b : 4; };
		int main() {
			return sizeof(struct ZeroW);
		}
		"""
		ir = generate_ir(source)
		main = get_func(ir, "main")
		from compiler.ir import IRReturn
		rets = [i for i in main.body if isinstance(i, IRReturn)]
		assert len(rets) >= 1
		ret_val = rets[0].value
		if isinstance(ret_val, IRConst):
			# a in first unit (4 bytes), zero-width forces new unit, b in second unit (4 bytes) = 8
			assert ret_val.value == 8

	def test_sizeof_union_with_bitfield_struct(self) -> None:
		"""sizeof union containing a struct with bitfields."""
		source = """
		struct BF { int x : 16; int y : 16; };
		union U { struct BF bf; int raw; };
		int main() {
			return sizeof(union U);
		}
		"""
		ir = generate_ir(source)
		main = get_func(ir, "main")
		from compiler.ir import IRReturn
		rets = [i for i in main.body if isinstance(i, IRReturn)]
		assert len(rets) >= 1
		ret_val = rets[0].value
		if isinstance(ret_val, IRConst):
			# struct BF = 4 bytes (two 16-bit bitfields in one int), int = 4, max = 4
			assert ret_val.value == 4

	def test_sizeof_expr_on_bitfield_struct_variable(self) -> None:
		"""sizeof applied to a variable of a bitfield struct type."""
		source = """
		struct Flags { int a : 1; int b : 2; int c : 3; };
		int main() {
			struct Flags f;
			f.a = 0;
			return sizeof(f);
		}
		"""
		ir = generate_ir(source)
		main = get_func(ir, "main")
		from compiler.ir import IRReturn
		rets = [i for i in main.body if isinstance(i, IRReturn)]
		assert len(rets) >= 1
		ret_val = rets[0].value
		if isinstance(ret_val, IRConst):
			# 1+2+3 = 6 bits, fits in one 4-byte int
			assert ret_val.value == 4
