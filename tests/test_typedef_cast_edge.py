"""Edge-case tests for typedef chains, pointer typedefs, struct typedefs, cast interactions,
type modifier casts, implicit promotions with typedefs, and sizeof on typedef'd types."""

import pytest

from compiler.ast_nodes import CastExpr
from compiler.parser import Parser, ParseError
from compiler.semantic import SemanticAnalyzer, SemanticError


def _parse(source: str):
	return Parser.from_source(source).parse()


def _analyze(source: str):
	prog = _parse(source)
	analyzer = SemanticAnalyzer()
	errors = analyzer.analyze(prog)
	return prog, analyzer, errors


def _analyze_ok(source: str):
	prog, analyzer, errors = _analyze(source)
	assert not errors, f"Unexpected semantic errors: {errors}"
	return prog, analyzer


# ---------------------------------------------------------------------------
# Typedef chains
# ---------------------------------------------------------------------------


class TestTypedefChains:
	def test_chain_two_levels(self):
		src = """
		typedef int myint;
		typedef myint myint2;
		int foo() { myint2 x = 10; return x; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert analyzer._typedef_types["myint"].base_type == "int"
		assert analyzer._typedef_types["myint2"].base_type == "int"

	def test_chain_four_levels(self):
		src = """
		typedef int A;
		typedef A B;
		typedef B C;
		typedef C D;
		int foo() { D x = 1; return x; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert analyzer._typedef_types["D"].base_type == "int"

	def test_chain_preserves_pointer_count(self):
		"""typedef int *ip; typedef ip ip2; => ip2 is int* (pointer_count=1)."""
		src = """
		typedef int *ip;
		typedef ip ip2;
		int foo() { int v = 0; ip2 p = &v; return *p; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert analyzer._typedef_types["ip2"].base_type == "int"
		assert analyzer._typedef_types["ip2"].pointer_count == 1

	def test_chain_adds_pointer_layers(self):
		"""typedef int *ip; typedef ip *ipp; => ipp is int** (pointer_count=2)."""
		src = """
		typedef int *ip;
		typedef ip *ipp;
		int foo() { int v = 0; int *p = &v; ipp pp = &p; return 0; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert analyzer._typedef_types["ipp"].base_type == "int"
		assert analyzer._typedef_types["ipp"].pointer_count == 2

	def test_chain_used_in_function_params(self):
		src = """
		typedef int myint;
		typedef myint myint2;
		myint2 add(myint2 a, myint2 b) { return a + b; }
		"""
		_analyze_ok(src)

	def test_chain_used_in_return_type(self):
		src = """
		typedef int myint;
		typedef myint myint2;
		myint2 identity(myint2 x) { return x; }
		"""
		_analyze_ok(src)


# ---------------------------------------------------------------------------
# Typedef with pointers
# ---------------------------------------------------------------------------


class TestTypedefPointers:
	def test_typedef_int_ptr(self):
		src = """
		typedef int *intptr;
		int foo() { int x = 42; intptr p = &x; return *p; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert analyzer._typedef_types["intptr"].pointer_count == 1
		assert analyzer._typedef_types["intptr"].base_type == "int"

	def test_typedef_char_ptr(self):
		src = """
		typedef char *cstr;
		int foo() { cstr s = "hello"; return 0; }
		"""
		_analyze_ok(src)

	def test_typedef_void_ptr(self):
		src = """
		typedef void *voidptr;
		int foo() { int x = 1; voidptr p = &x; return 0; }
		"""
		_analyze_ok(src)

	def test_pointer_to_typedef_type(self):
		"""MyInt *p where MyInt is typedef'd int => p is int*."""
		src = """
		typedef int MyInt;
		int foo() { int v = 5; MyInt *p = &v; return *p; }
		"""
		_analyze_ok(src)

	def test_double_pointer_typedef(self):
		src = """
		typedef int **intpp;
		int foo() { int x = 1; int *p = &x; intpp pp = &p; return 0; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert analyzer._typedef_types["intpp"].pointer_count == 2


# ---------------------------------------------------------------------------
# Typedef of struct types
# ---------------------------------------------------------------------------


class TestTypedefStruct:
	def test_typedef_inline_struct(self):
		src = """
		typedef struct { int x; int y; } Point;
		int foo() { Point p; p.x = 1; p.y = 2; return p.x + p.y; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert analyzer._typedef_types["Point"].base_type == "struct Point"

	def test_typedef_named_struct(self):
		src = """
		typedef struct Vec { int x; int y; } Vec;
		int foo() { Vec v; v.x = 3; v.y = 4; return v.x; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert analyzer._typedef_types["Vec"].base_type == "struct Vec"

	def test_typedef_struct_pointer(self):
		src = """
		typedef struct Node { int val; } Node;
		int foo() { Node n; n.val = 10; Node *p = &n; return p->val; }
		"""
		_analyze_ok(src)

	def test_typedef_struct_in_another_struct(self):
		src = """
		typedef struct { int x; int y; } Point;
		struct Line { Point start; Point end; };
		int foo() {
			struct Line l;
			l.start.x = 0;
			l.end.x = 10;
			return l.end.x;
		}
		"""
		_analyze_ok(src)

	def test_typedef_existing_struct_ref(self):
		src = """
		struct Pair { int a; int b; };
		typedef struct Pair Pair;
		int foo() { Pair p; p.a = 1; p.b = 2; return p.a; }
		"""
		_analyze_ok(src)

	def test_typedef_union_inline(self):
		src = """
		typedef union { int i; float f; } Number;
		int foo() { Number n; n.i = 42; return n.i; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert "Number" in analyzer._typedef_types

	def test_typedef_enum_inline(self):
		src = """
		typedef enum { LOW, MED, HIGH } Priority;
		int foo() { int p = LOW; return p; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert "Priority" in analyzer._typedef_types


# ---------------------------------------------------------------------------
# Casts between typedef'd types
# ---------------------------------------------------------------------------


class TestCastBetweenTypedefs:
	def test_cast_typedef_to_typedef(self):
		"""Cast from one typedef'd int to another typedef'd int."""
		src = """
		typedef int TypeA;
		typedef int TypeB;
		int foo() { TypeA a = 10; TypeB b = (TypeB)a; return b; }
		"""
		_analyze_ok(src)

	def test_cast_typedef_to_base_type(self):
		src = """
		typedef int MyInt;
		int foo() { MyInt x = 5; int y = (int)x; return y; }
		"""
		_analyze_ok(src)

	def test_cast_base_to_typedef(self):
		src = """
		typedef int MyInt;
		int foo() { int x = 5; MyInt y = (MyInt)x; return y; }
		"""
		_analyze_ok(src)

	def test_cast_char_to_typedef_int(self):
		src = """
		typedef int MyInt;
		int foo() { char c = 'z'; MyInt x = (MyInt)c; return x; }
		"""
		_analyze_ok(src)

	def test_cast_typedef_chain_to_base(self):
		"""Cast from a chained typedef back to the base type."""
		src = """
		typedef int A;
		typedef A B;
		int foo() { B x = 7; int y = (int)x; return y; }
		"""
		_analyze_ok(src)

	def test_cast_between_pointer_typedefs(self):
		src = """
		typedef int *IntPtr;
		typedef char *CharPtr;
		int foo() { int x = 1; IntPtr p = &x; CharPtr q = (CharPtr)p; return 0; }
		"""
		_analyze_ok(src)

	def test_cast_typedef_pointer_to_int(self):
		src = """
		typedef int *IntPtr;
		int foo() { int x = 1; IntPtr p = &x; int v = (int)p; return v; }
		"""
		_analyze_ok(src)

	def test_cast_int_to_typedef_pointer(self):
		src = """
		typedef int *IntPtr;
		int foo() { int x = 42; IntPtr p = (IntPtr)x; return 0; }
		"""
		_analyze_ok(src)

	def test_cast_expr_ast_node(self):
		"""Verify that a cast to a typedef'd type produces a CastExpr in the AST."""
		src = """
		typedef int MyInt;
		int foo() { char c = 1; MyInt x = (MyInt)c; return x; }
		"""
		prog = _parse(src)
		func = prog.declarations[1]
		body = func.body.statements
		# Second statement: MyInt x = (MyInt)c
		var_decl = body[1]
		assert isinstance(var_decl.initializer, CastExpr)


# ---------------------------------------------------------------------------
# Casts involving type modifiers (long to short, signed to unsigned)
# ---------------------------------------------------------------------------


class TestCastTypeModifiers:
	def test_cast_int_to_long(self):
		src = """
		int foo() { int x = 1; long y = (long)x; return y; }
		"""
		_analyze_ok(src)

	def test_cast_long_to_short(self):
		src = """
		int foo() { long x = 100000; short y = (short)x; return y; }
		"""
		_analyze_ok(src)

	def test_cast_int_to_char(self):
		src = """
		int foo() { int x = 65; char c = (char)x; return c; }
		"""
		_analyze_ok(src)

	def test_cast_char_to_int(self):
		src = """
		int foo() { char c = 'A'; int x = (int)c; return x; }
		"""
		_analyze_ok(src)

	def test_cast_float_to_int(self):
		src = """
		int foo() { float f = 3.14; int x = (int)f; return x; }
		"""
		_analyze_ok(src)

	def test_cast_int_to_float(self):
		src = """
		int foo() { int x = 42; float f = (float)x; return 0; }
		"""
		_analyze_ok(src)

	def test_cast_double_to_int(self):
		src = """
		int foo() { double d = 2.718; int x = (int)d; return x; }
		"""
		_analyze_ok(src)

	def test_cast_int_to_double(self):
		src = """
		int foo() { int x = 10; double d = (double)x; return 0; }
		"""
		_analyze_ok(src)

	def test_cast_long_long_to_int(self):
		src = """
		int foo() { long long x = 999; int y = (int)x; return y; }
		"""
		_analyze_ok(src)

	def test_cast_short_to_long_long(self):
		src = """
		int foo() { short s = 5; long long x = (long long)s; return 0; }
		"""
		_analyze_ok(src)

	def test_cast_typedef_with_modifier(self):
		"""Cast involving a typedef'd type and a width-modified type."""
		src = """
		typedef int MyInt;
		int foo() { MyInt x = 10; short s = (short)x; return s; }
		"""
		_analyze_ok(src)


# ---------------------------------------------------------------------------
# Implicit integer promotions in arithmetic with typedefs
# ---------------------------------------------------------------------------


class TestImplicitPromotionsTypedefs:
	def test_typedef_int_arithmetic(self):
		src = """
		typedef int MyInt;
		int foo() { MyInt a = 3; MyInt b = 4; MyInt c = a + b; return c; }
		"""
		_analyze_ok(src)

	def test_typedef_char_arithmetic(self):
		"""char arithmetic should work (C promotes to int implicitly)."""
		src = """
		typedef char MyChar;
		int foo() { MyChar a = 10; MyChar b = 20; int c = a + b; return c; }
		"""
		_analyze_ok(src)

	def test_typedef_mixed_arithmetic(self):
		"""Mixing typedef'd type with base type in expressions."""
		src = """
		typedef int MyInt;
		int foo() { MyInt a = 5; int b = 10; int c = a * b; return c; }
		"""
		_analyze_ok(src)

	def test_typedef_in_comparison(self):
		src = """
		typedef int MyInt;
		int foo() { MyInt a = 5; MyInt b = 10; int c = a < b; return c; }
		"""
		_analyze_ok(src)

	def test_typedef_in_ternary(self):
		src = """
		typedef int MyInt;
		int foo() { MyInt x = 1; int r = x ? 100 : 200; return r; }
		"""
		_analyze_ok(src)

	def test_typedef_chain_in_arithmetic(self):
		src = """
		typedef int A;
		typedef A B;
		int foo() { B x = 3; B y = 4; int z = x + y * 2; return z; }
		"""
		_analyze_ok(src)

	def test_typedef_unary_ops(self):
		src = """
		typedef int MyInt;
		int foo() { MyInt x = 5; MyInt y = -x; MyInt z = ~x; return y + z; }
		"""
		_analyze_ok(src)

	def test_typedef_postfix_ops(self):
		src = """
		typedef int MyInt;
		int foo() { MyInt x = 0; x++; x--; return x; }
		"""
		_analyze_ok(src)

	def test_typedef_compound_assignment(self):
		src = """
		typedef int MyInt;
		int foo() { MyInt x = 10; x += 5; x -= 2; x *= 3; return x; }
		"""
		_analyze_ok(src)


# ---------------------------------------------------------------------------
# sizeof on typedef'd types
# ---------------------------------------------------------------------------


class TestSizeofTypedef:
	def test_sizeof_typedef_int(self):
		src = """
		typedef int MyInt;
		int foo() { int s = sizeof(MyInt); return s; }
		"""
		_analyze_ok(src)

	def test_sizeof_typedef_char(self):
		src = """
		typedef char MyChar;
		int foo() { int s = sizeof(MyChar); return s; }
		"""
		_analyze_ok(src)

	def test_sizeof_typedef_pointer(self):
		src = """
		typedef int *IntPtr;
		int foo() { int s = sizeof(IntPtr); return s; }
		"""
		_analyze_ok(src)

	def test_sizeof_typedef_chain(self):
		src = """
		typedef int A;
		typedef A B;
		typedef B C;
		int foo() { int s = sizeof(C); return s; }
		"""
		_analyze_ok(src)

	def test_sizeof_typedef_struct(self):
		src = """
		typedef struct { int x; int y; } Point;
		int foo() { int s = sizeof(Point); return s; }
		"""
		_analyze_ok(src)

	def test_sizeof_typedef_in_expression(self):
		src = """
		typedef int MyInt;
		int foo() { int x = sizeof(MyInt) + sizeof(char); return x; }
		"""
		_analyze_ok(src)

	def test_sizeof_typedef_var(self):
		"""sizeof applied to a variable of a typedef'd type."""
		src = """
		typedef int MyInt;
		int foo() { MyInt x = 5; int s = sizeof(x); return s; }
		"""
		_analyze_ok(src)


# ---------------------------------------------------------------------------
# Error cases for typedef and cast interactions
# ---------------------------------------------------------------------------


class TestTypedefCastErrors:
	def test_redefinition_of_typedef(self):
		src = """
		typedef int MyInt;
		typedef char MyInt;
		int foo() { return 0; }
		"""
		with pytest.raises(SemanticError, match="redefinition of typedef"):
			_analyze_ok(src)

	def test_cast_struct_to_int_invalid(self):
		"""Cannot cast a struct value to int."""
		src = """
		struct S { int x; };
		int foo() { struct S s; s.x = 1; int x = (int)s; return x; }
		"""
		with pytest.raises(SemanticError, match="invalid cast"):
			_analyze_ok(src)

	def test_typedef_used_before_definition(self):
		"""Using a typedef name before it is defined should fail at parse time."""
		src = """
		int foo() { MyType x = 5; return x; }
		typedef int MyType;
		"""
		with pytest.raises(ParseError):
			_parse(src)


# ---------------------------------------------------------------------------
# Typedef in local scope
# ---------------------------------------------------------------------------


class TestTypedefLocalScope:
	def test_typedef_inside_function(self):
		src = """
		int foo() {
			typedef int LocalInt;
			LocalInt x = 42;
			return x;
		}
		"""
		_analyze_ok(src)

	def test_typedef_in_nested_block(self):
		src = """
		int foo() {
			int r = 0;
			{
				typedef int BlockInt;
				BlockInt x = 10;
				r = x;
			}
			return r;
		}
		"""
		_analyze_ok(src)

	def test_multiple_local_typedefs(self):
		src = """
		int foo() {
			typedef int A;
			typedef char B;
			A x = 10;
			B y = 'z';
			return x;
		}
		"""
		_analyze_ok(src)
