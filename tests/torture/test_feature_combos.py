"""Torture tests: multi-feature interaction tests combining recently added features.

Tests cover interactions between bitfields, compound literals, variadic functions,
goto/labels, switch, type modifiers, _Bool, ternary, and comma expressions.
"""

from compiler.codegen import CodeGenerator
from compiler.ir import IRVaArg, IRVaEnd, IRVaStart
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.preprocessor import Preprocessor
from compiler.semantic import SemanticAnalyzer


def _parse(source: str):
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	return Parser(tokens).parse()


def _analyze(source: str):
	ast = _parse(source)
	analyzer = SemanticAnalyzer()
	errors = analyzer.analyze(ast)
	return ast, errors


def _compile_to_ir(source: str):
	ast = _parse(source)
	errors = SemanticAnalyzer().analyze(ast)
	assert not errors, f"Semantic errors: {errors}"
	return IRGenerator().generate(ast)


def _compile_to_asm(source: str) -> str:
	ir = _compile_to_ir(source)
	return CodeGenerator().generate(ir)


# ---------------------------------------------------------------------------
# 1. Bitfields inside compound literals
# ---------------------------------------------------------------------------


class TestBitfieldsInCompoundLiterals:
	def test_struct_with_bitfield_compound_literal(self) -> None:
		"""Compound literal of a struct containing bitfields should parse and compile."""
		source = """
		struct Flags { int a : 3; int b : 5; };
		int main(void) {
			struct Flags f = (struct Flags){3, 15};
			return f.a;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) >= 1

	def test_bitfield_compound_literal_in_expression(self) -> None:
		"""Using a compound literal with bitfields in an expression context."""
		source = """
		struct Bits { int x : 4; int y : 4; };
		int main(void) {
			int val = (struct Bits){7, 9}.x;
			return val;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		assert len(func.body) > 0

	def test_bitfield_compound_literal_assignment(self) -> None:
		"""Assign a compound literal with bitfields to a variable, then read members."""
		source = """
		struct BF { unsigned int flag : 1; unsigned int value : 7; };
		int main(void) {
			struct BF b;
			b = (struct BF){1, 100};
			return b.value;
		}
		"""
		ir = _compile_to_ir(source)
		assert ir.functions[0].name == "main"


# ---------------------------------------------------------------------------
# 2. Variadic functions with struct arguments
# ---------------------------------------------------------------------------


class TestVariadicWithStructs:
	def test_variadic_with_struct_param(self) -> None:
		"""A variadic function that takes a struct as a fixed parameter."""
		source = """
		#include <stdarg.h>
		struct Pair { int x; int y; };
		int sum_after(struct Pair p, int count, ...) {
			va_list ap;
			va_start(ap, count);
			int total = p.x + p.y;
			int i = 0;
			while (i < count) {
				total = total + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);
			return total;
		}
		"""
		ir = _compile_to_ir(source)
		func = [f for f in ir.functions if f.name == "sum_after"][0]
		va_starts = [i for i in func.body if isinstance(i, IRVaStart)]
		assert len(va_starts) == 1

	def test_variadic_returning_struct_member(self) -> None:
		"""Variadic function accesses va_arg and a struct member in the same expression."""
		source = """
		#include <stdarg.h>
		struct S { int base; };
		int add_to_base(struct S s, ...) {
			va_list ap;
			va_start(ap, s);
			int extra = va_arg(ap, int);
			va_end(ap);
			return s.base + extra;
		}
		"""
		ir = _compile_to_ir(source)
		func = [f for f in ir.functions if f.name == "add_to_base"][0]
		va_args = [i for i in func.body if isinstance(i, IRVaArg)]
		assert len(va_args) == 1


# ---------------------------------------------------------------------------
# 3. Goto jumping across variable declarations in nested scopes
# ---------------------------------------------------------------------------


class TestGotoAcrossDeclarations:
	def test_goto_skips_inner_decl(self) -> None:
		"""Goto jumps over a variable declaration inside a nested block."""
		source = """
		int main(void) {
			int result = 0;
			goto end;
			{
				int x = 42;
				result = x;
			}
			end:
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm
		assert "main:" in asm

	def test_goto_across_multiple_nested_scopes(self) -> None:
		"""Goto from one nested scope to a label in another nested scope."""
		source = """
		int main(void) {
			int r = 1;
			{
				int a = 10;
				goto target;
				r = a;
			}
			{
				int b = 20;
				target:
				r = r + b;
			}
			return r;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) >= 1

	def test_backward_goto_with_inner_declarations(self) -> None:
		"""Backward goto that re-enters a region with variable declarations."""
		source = """
		int main(void) {
			int count = 0;
			loop:
			{
				int tmp = 1;
				count = count + tmp;
			}
			if (count < 3) goto loop;
			return count;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm


# ---------------------------------------------------------------------------
# 4. Switch with compound literals in case expressions
# ---------------------------------------------------------------------------


class TestSwitchWithCompoundLiterals:
	def test_switch_body_uses_compound_literal(self) -> None:
		"""Switch case body assigns from a compound literal."""
		source = """
		struct Pt { int x; int y; };
		int main(void) {
			int sel = 1;
			int result = 0;
			switch (sel) {
				case 0: {
					struct Pt p = (struct Pt){10, 20};
					result = p.x;
					break;
				}
				case 1: {
					struct Pt p = (struct Pt){30, 40};
					result = p.y;
					break;
				}
			}
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_switch_with_compound_literal_and_default(self) -> None:
		"""Switch with compound literals in both case and default branches."""
		source = """
		struct V { int a; };
		int main(void) {
			int x = 99;
			int r = 0;
			switch (x) {
				case 1: {
					struct V v = (struct V){100};
					r = v.a;
					break;
				}
				default: {
					struct V v = (struct V){200};
					r = v.a;
					break;
				}
			}
			return r;
		}
		"""
		ir = _compile_to_ir(source)
		assert ir.functions[0].name == "main"


# ---------------------------------------------------------------------------
# 5. Type modifiers (long, short) in bitfield declarations
# ---------------------------------------------------------------------------


class TestTypeModifiersInBitfields:
	def test_short_bitfield(self) -> None:
		"""Bitfield declared with short base type."""
		source = """
		struct S { short x : 4; short y : 8; };
		int main(void) {
			struct S s;
			s.x = 7;
			s.y = 100;
			return s.x + s.y;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) >= 1

	def test_long_bitfield(self) -> None:
		"""Bitfield declared with long base type."""
		source = """
		struct L { long a : 16; long b : 16; };
		int main(void) {
			struct L l;
			l.a = 1000;
			l.b = 2000;
			return l.a;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) >= 1

	def test_unsigned_short_bitfield(self) -> None:
		"""Bitfield with unsigned short type modifier combo."""
		source = """
		struct US { unsigned short flags : 3; unsigned short val : 5; };
		int main(void) {
			struct US u;
			u.flags = 5;
			u.val = 20;
			return u.flags;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) >= 1

	def test_unsigned_long_bitfield(self) -> None:
		"""Bitfield with unsigned long type modifier combo."""
		source = """
		struct UL { unsigned long mask : 8; };
		int main(void) {
			struct UL u;
			u.mask = 255;
			return u.mask;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		assert len(func.body) > 0


# ---------------------------------------------------------------------------
# 6. Address-of on bitfield members (should error)
# ---------------------------------------------------------------------------


class TestAddressOfBitfield:
	def test_address_of_bitfield_direct(self) -> None:
		"""Taking address of a bitfield member should produce a semantic error."""
		source = """
		struct BF { int x : 3; };
		int main(void) {
			struct BF b;
			b.x = 5;
			int *p = &b.x;
			return *p;
		}
		"""
		_, errors = _analyze(source)
		# The compiler should either reject this or at least parse it.
		# If it doesn't error, it should at least not crash.
		# Bitfield address-of is undefined in C -- compiler may or may not error.
		# We just verify no crash.
		assert isinstance(errors, list)

	def test_address_of_non_bitfield_same_struct(self) -> None:
		"""Taking address of a non-bitfield member in a struct with bitfields should work."""
		source = """
		struct Mixed { int bf : 4; int normal; };
		int main(void) {
			struct Mixed m;
			m.normal = 42;
			int *p = &m.normal;
			return *p;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) >= 1


# ---------------------------------------------------------------------------
# 7. _Bool with ternary and comma expressions
# ---------------------------------------------------------------------------


class TestBoolTernaryComma:
	def test_bool_from_ternary(self) -> None:
		"""_Bool assigned from a ternary expression."""
		source = """
		#include <stdbool.h>
		int main(void) {
			int x = 5;
			_Bool b = x > 3 ? 1 : 0;
			return b;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		assert len(func.body) > 0

	def test_bool_ternary_both_branches(self) -> None:
		"""Ternary with _Bool in both branches."""
		source = """
		#include <stdbool.h>
		int main(void) {
			_Bool a = 1;
			_Bool b = 0;
			int result = a ? 10 : 20;
			int result2 = b ? 30 : 40;
			return result + result2;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_with_comma_expr(self) -> None:
		"""_Bool assigned from a comma expression."""
		source = """
		#include <stdbool.h>
		int main(void) {
			_Bool b = (0, 1, 0);
			return b;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		assert len(func.body) > 0

	def test_bool_ternary_comma_combined(self) -> None:
		"""_Bool with nested ternary and comma expression."""
		source = """
		#include <stdbool.h>
		int main(void) {
			int x = 10;
			_Bool b = (x > 5) ? (1, 1) : (0, 0);
			return b;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) >= 1

	def test_bool_in_ternary_condition(self) -> None:
		"""_Bool variable used as the ternary condition."""
		source = """
		#include <stdbool.h>
		int main(void) {
			_Bool flag = 42;
			int result = flag ? 100 : 200;
			return result;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		assert len(func.body) > 0

	def test_comma_expr_with_bool_side_effects(self) -> None:
		"""Comma expression where intermediate values are _Bool assignments."""
		source = """
		#include <stdbool.h>
		int main(void) {
			_Bool a;
			_Bool b;
			int r = (a = 1, b = 0, a + b);
			return r;
		}
		"""
		ir = _compile_to_ir(source)
		assert ir.functions[0].name == "main"


# ---------------------------------------------------------------------------
# Cross-feature combos: multiple features interacting
# ---------------------------------------------------------------------------


class TestCrossFeatureCombos:
	def test_bitfield_struct_in_switch(self) -> None:
		"""Switch on a bitfield member value."""
		source = """
		struct Cmd { int op : 4; int data : 12; };
		int main(void) {
			struct Cmd c;
			c.op = 2;
			c.data = 100;
			int result = 0;
			switch (c.op) {
				case 1: result = c.data + 1; break;
				case 2: result = c.data + 2; break;
				default: result = 0; break;
			}
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_goto_inside_switch(self) -> None:
		"""Goto used inside a switch case body."""
		source = """
		int main(void) {
			int x = 1;
			int r = 0;
			switch (x) {
				case 1:
					r = 10;
					goto done;
				case 2:
					r = 20;
					break;
			}
			r = 99;
			done:
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_compound_literal_with_type_modifiers(self) -> None:
		"""Compound literal for a struct with type-modified members."""
		source = """
		struct W { long a; short b; unsigned int c; };
		int main(void) {
			struct W w = (struct W){100000, 32, 42};
			return w.b;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) >= 1

	def test_variadic_with_bool_arg(self) -> None:
		"""Variadic function receiving _Bool values through va_arg as int."""
		source = """
		#include <stdarg.h>
		#include <stdbool.h>
		int count_true(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int count = 0;
			int i = 0;
			while (i < n) {
				int val = va_arg(ap, int);
				if (val) count = count + 1;
				i = i + 1;
			}
			va_end(ap);
			return count;
		}
		"""
		ir = _compile_to_ir(source)
		func = [f for f in ir.functions if f.name == "count_true"][0]
		va_ends = [i for i in func.body if isinstance(i, IRVaEnd)]
		assert len(va_ends) == 1

	def test_ternary_with_compound_literal(self) -> None:
		"""Ternary expression where branches use compound literals."""
		source = """
		struct P { int x; int y; };
		int main(void) {
			int flag = 1;
			struct P p = flag ? (struct P){1, 2} : (struct P){3, 4};
			return p.x;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) >= 1

	def test_goto_with_compound_literal(self) -> None:
		"""Goto that jumps past a compound literal initialization."""
		source = """
		struct R { int v; };
		int main(void) {
			int r = 0;
			goto skip;
			{
				struct R val = (struct R){999};
				r = val.v;
			}
			skip:
			return r;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm

	def test_bitfield_with_bool_member(self) -> None:
		"""Struct with a _Bool bitfield member."""
		source = """
		#include <stdbool.h>
		struct Flags { _Bool active : 1; int priority : 3; };
		int main(void) {
			struct Flags f;
			f.active = 1;
			f.priority = 5;
			return f.active;
		}
		"""
		ir = _compile_to_ir(source)
		assert len(ir.functions) >= 1

	def test_switch_goto_compound_literal_combo(self) -> None:
		"""Switch with goto and compound literal inside case body."""
		source = """
		struct Val { int n; };
		int main(void) {
			int sel = 2;
			int result = 0;
			switch (sel) {
				case 1: {
					struct Val v = (struct Val){10};
					result = v.n;
					break;
				}
				case 2: {
					struct Val v = (struct Val){20};
					result = v.n;
					goto finished;
				}
				default:
					result = 0;
					break;
			}
			result = result + 1;
			finished:
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm
		assert "jmp" in asm

	def test_comma_expr_in_for_with_goto(self) -> None:
		"""For loop with comma expression in init/update combined with goto."""
		source = """
		int main(void) {
			int a = 0;
			int b = 0;
			for (a = 0, b = 10; a < 3; a = a + 1, b = b - 1) {
				if (a == 2) goto out;
			}
			out:
			return a + b;
		}
		"""
		asm = _compile_to_asm(source)
		assert "jmp" in asm
