"""Tests for designated initializer and compound literal interaction edge cases."""

from compiler.ast_nodes import (
	CompoundLiteral,
	DesignatedInit,
	InitializerList,
	UnaryOp,
)
from compiler.ir import IRAlloc, IRStore
from compiler.ir_gen import IRGenerator
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer


def _parse(source: str):
	return Parser.from_source(source).parse()


def _analyze(source: str):
	program = _parse(source)
	analyzer = SemanticAnalyzer()
	errors = analyzer.analyze(program)
	return program, errors


def _generate_ir(source: str):
	program = _parse(source)
	analyzer = SemanticAnalyzer()
	analyzer.analyze(program)
	gen = IRGenerator()
	return gen.generate(program)


class TestStructDesignatedInitParsing:
	"""Test parsing of struct designated initializers (.field = val)."""

	def test_single_field_designated(self):
		src = """
		struct S { int a; int b; };
		int main(void) {
			struct S s = {.a = 10};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[1]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl.initializer, InitializerList)
		elem = var_decl.initializer.elements[0]
		assert isinstance(elem, DesignatedInit)
		assert elem.field_name == "a"
		assert elem.index is None

	def test_multiple_field_designated(self):
		src = """
		struct S { int a; int b; int c; };
		int main(void) {
			struct S s = {.c = 3, .a = 1, .b = 2};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[1]
		var_decl = func.body.statements[0]
		init_list = var_decl.initializer
		assert isinstance(init_list, InitializerList)
		assert len(init_list.elements) == 3
		assert all(isinstance(e, DesignatedInit) for e in init_list.elements)
		assert init_list.elements[0].field_name == "c"
		assert init_list.elements[1].field_name == "a"
		assert init_list.elements[2].field_name == "b"

	def test_designated_with_trailing_comma(self):
		src = """
		struct S { int x; int y; };
		int main(void) {
			struct S s = {.x = 1, .y = 2,};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[1]
		var_decl = func.body.statements[0]
		init_list = var_decl.initializer
		assert len(init_list.elements) == 2


class TestArrayDesignatedInitParsing:
	"""Test parsing of array designated initializers ([idx] = val)."""

	def test_single_index_designated(self):
		src = """
		int main(void) {
			int arr[5] = {[2] = 42};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[0]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl.initializer, InitializerList)
		elem = var_decl.initializer.elements[0]
		assert isinstance(elem, DesignatedInit)
		assert elem.field_name is None
		assert elem.index is not None

	def test_multiple_index_designated(self):
		src = """
		int main(void) {
			int arr[10] = {[0] = 10, [5] = 50, [9] = 90};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[0]
		var_decl = func.body.statements[0]
		init_list = var_decl.initializer
		assert len(init_list.elements) == 3
		assert all(isinstance(e, DesignatedInit) for e in init_list.elements)
		assert all(e.index is not None for e in init_list.elements)

	def test_mixed_designated_and_positional(self):
		src = """
		int main(void) {
			int arr[5] = {1, [3] = 30, 2};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[0]
		var_decl = func.body.statements[0]
		init_list = var_decl.initializer
		assert len(init_list.elements) == 3
		assert not isinstance(init_list.elements[0], DesignatedInit)
		assert isinstance(init_list.elements[1], DesignatedInit)
		assert not isinstance(init_list.elements[2], DesignatedInit)


class TestNestedDesignatedInit:
	"""Test nested designated initializers."""

	def test_nested_struct_designated(self):
		src = """
		struct Inner { int a; int b; };
		struct Outer { struct Inner inner; int c; };
		int main(void) {
			struct Outer o = {.inner = {.a = 1, .b = 2}, .c = 3};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[2]
		var_decl = func.body.statements[0]
		init_list = var_decl.initializer
		assert isinstance(init_list, InitializerList)
		outer_inner = init_list.elements[0]
		assert isinstance(outer_inner, DesignatedInit)
		assert outer_inner.field_name == "inner"
		# The value should be an InitializerList with designated inits
		assert isinstance(outer_inner.value, InitializerList)
		assert len(outer_inner.value.elements) == 2
		assert isinstance(outer_inner.value.elements[0], DesignatedInit)
		assert outer_inner.value.elements[0].field_name == "a"

	def test_struct_with_array_member_designated(self):
		src = """
		struct S { int arr[3]; int x; };
		int main(void) {
			struct S s = {.arr = {1, 2, 3}, .x = 10};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[1]
		var_decl = func.body.statements[0]
		init_list = var_decl.initializer
		arr_init = init_list.elements[0]
		assert isinstance(arr_init, DesignatedInit)
		assert arr_init.field_name == "arr"
		assert isinstance(arr_init.value, InitializerList)
		assert len(arr_init.value.elements) == 3

	def test_array_of_structs_designated(self):
		src = """
		struct Point { int x; int y; };
		int main(void) {
			struct Point pts[2] = {[0] = {.x = 1, .y = 2}, [1] = {.x = 3, .y = 4}};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[1]
		var_decl = func.body.statements[0]
		init_list = var_decl.initializer
		assert len(init_list.elements) == 2
		for elem in init_list.elements:
			assert isinstance(elem, DesignatedInit)
			assert elem.index is not None
			assert isinstance(elem.value, InitializerList)


class TestPartialDesignatedInit:
	"""Test partial initialization with designated fields."""

	def test_partial_struct_init_semantic(self):
		"""Only some fields designated - should parse and analyze without errors."""
		src = """
		struct S { int a; int b; int c; int d; };
		int main(void) {
			struct S s = {.b = 20};
			return 0;
		}
		"""
		_, errors = _analyze(src)
		assert len(errors) == 0

	def test_partial_array_init_semantic(self):
		"""Sparse array initialization - should have no errors."""
		src = """
		int main(void) {
			int arr[100] = {[50] = 500, [99] = 990};
			return 0;
		}
		"""
		_, errors = _analyze(src)
		assert len(errors) == 0

	def test_empty_init_list(self):
		"""Empty initializer list should parse."""
		src = """
		struct S { int a; int b; };
		int main(void) {
			struct S s = {};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[1]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl.initializer, InitializerList)
		assert len(var_decl.initializer.elements) == 0


class TestCompoundLiteralAsArg:
	"""Test compound literals as function arguments."""

	def test_struct_compound_literal_arg(self):
		src = """
		struct Point { int x; int y; };
		int sum(struct Point *p) { return p->x + p->y; }
		int main(void) {
			int r = sum(&(struct Point){10, 20});
			return r;
		}
		"""
		_, errors = _analyze(src)
		assert len(errors) == 0

	def test_struct_compound_literal_arg_ir(self):
		src = """
		struct Point { int x; int y; };
		int sum(struct Point *p) { return p->x + p->y; }
		int main(void) {
			int r = sum(&(struct Point){10, 20});
			return r;
		}
		"""
		ir_prog = _generate_ir(src)
		main_fn = [f for f in ir_prog.functions if f.name == "main"][0]
		allocs = [i for i in main_fn.body if isinstance(i, IRAlloc)]
		# Should allocate for r and the compound literal
		assert len(allocs) >= 2

	def test_array_compound_literal_arg(self):
		src = """
		int first(int *arr) { return arr[0]; }
		int main(void) {
			int r = first((int[]){100, 200, 300});
			return r;
		}
		"""
		_, errors = _analyze(src)
		assert len(errors) == 0


class TestCompoundLiteralWithDesignatedInit:
	"""Test compound literals with designated initializers."""

	def test_compound_literal_designated_struct(self):
		src = """
		struct Rect { int w; int h; };
		int main(void) {
			struct Rect r = (struct Rect){.h = 20, .w = 10};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[1]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl.initializer, CompoundLiteral)
		init_list = var_decl.initializer.init_list
		assert len(init_list.elements) == 2
		assert isinstance(init_list.elements[0], DesignatedInit)
		assert init_list.elements[0].field_name == "h"
		assert isinstance(init_list.elements[1], DesignatedInit)
		assert init_list.elements[1].field_name == "w"

	def test_compound_literal_designated_semantic(self):
		src = """
		struct Rect { int w; int h; };
		int main(void) {
			struct Rect r = (struct Rect){.h = 20, .w = 10};
			return 0;
		}
		"""
		_, errors = _analyze(src)
		assert len(errors) == 0

	def test_compound_literal_designated_ir(self):
		src = """
		struct Rect { int w; int h; };
		int main(void) {
			struct Rect r = (struct Rect){.h = 20, .w = 10};
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		main_fn = [f for f in ir_prog.functions if f.name == "main"][0]
		stores = [i for i in main_fn.body if isinstance(i, IRStore)]
		assert len(stores) >= 2

	def test_compound_literal_array_designated(self):
		src = """
		int main(void) {
			int *p = (int[]){[0] = 10, [2] = 30};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[0]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl.initializer, CompoundLiteral)
		init_list = var_decl.initializer.init_list
		assert len(init_list.elements) == 2
		assert all(isinstance(e, DesignatedInit) for e in init_list.elements)

	def test_nested_compound_literal_designated(self):
		"""Compound literal with nested designated initializers in sub-structs."""
		src = """
		struct Inner { int a; int b; };
		struct Outer { struct Inner i; int c; };
		int main(void) {
			struct Outer o = (struct Outer){.i = {.a = 1, .b = 2}, .c = 3};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[2]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl.initializer, CompoundLiteral)
		init_list = var_decl.initializer.init_list
		assert len(init_list.elements) == 2
		inner_init = init_list.elements[0]
		assert isinstance(inner_init, DesignatedInit)
		assert inner_init.field_name == "i"
		assert isinstance(inner_init.value, InitializerList)


class TestCompoundLiteralAddressOf:
	"""Test taking the address of compound literals."""

	def test_address_of_struct_compound_literal(self):
		src = """
		struct S { int x; int y; };
		int main(void) {
			struct S *p = &(struct S){1, 2};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[1]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl.initializer, UnaryOp)
		assert var_decl.initializer.op == "&"
		assert isinstance(var_decl.initializer.operand, CompoundLiteral)

	def test_address_of_compound_literal_semantic(self):
		src = """
		struct S { int x; int y; };
		int main(void) {
			struct S *p = &(struct S){1, 2};
			return 0;
		}
		"""
		_, errors = _analyze(src)
		assert len(errors) == 0

	def test_address_of_compound_literal_ir(self):
		src = """
		struct S { int x; int y; };
		int main(void) {
			struct S *p = &(struct S){1, 2};
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		main_fn = [f for f in ir_prog.functions if f.name == "main"][0]
		allocs = [i for i in main_fn.body if isinstance(i, IRAlloc)]
		# At least one for p, one for the compound literal
		assert len(allocs) >= 2

	def test_address_of_designated_compound_literal(self):
		src = """
		struct S { int x; int y; };
		int main(void) {
			struct S *p = &(struct S){.y = 99, .x = 42};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[1]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl.initializer, UnaryOp)
		assert isinstance(var_decl.initializer.operand, CompoundLiteral)
		init_list = var_decl.initializer.operand.init_list
		assert isinstance(init_list.elements[0], DesignatedInit)

	def test_address_of_array_compound_literal(self):
		src = """
		int main(void) {
			int *p = (int[]){10, 20, 30};
			int *q = &p[0];
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[0]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl.initializer, CompoundLiteral)


class TestDesignatedInitWithTypedef:
	"""Test designated initializers and compound literals with typedef'd types."""

	def test_typedef_struct_designated_init(self):
		src = """
		struct _Point { int x; int y; };
		typedef struct _Point Point;
		int main(void) {
			Point p = {.x = 10, .y = 20};
			return 0;
		}
		"""
		_, errors = _analyze(src)
		assert len(errors) == 0

	def test_typedef_struct_compound_literal(self):
		src = """
		struct _Point { int x; int y; };
		typedef struct _Point Point;
		int main(void) {
			Point p = (Point){1, 2};
			return 0;
		}
		"""
		program, errors = _analyze(src)
		assert len(errors) == 0
		func = program.declarations[2]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl.initializer, CompoundLiteral)

	def test_typedef_compound_literal_with_designated(self):
		src = """
		struct _Rect { int w; int h; };
		typedef struct _Rect Rect;
		int main(void) {
			Rect r = (Rect){.w = 100, .h = 200};
			return 0;
		}
		"""
		_, errors = _analyze(src)
		assert len(errors) == 0

	def test_typedef_compound_literal_address_of(self):
		src = """
		struct _S { int a; int b; };
		typedef struct _S S;
		int main(void) {
			S *p = &(S){.a = 1, .b = 2};
			return 0;
		}
		"""
		_, errors = _analyze(src)
		assert len(errors) == 0

	def test_typedef_nested_struct_designated(self):
		src = """
		struct _Inner { int x; };
		typedef struct _Inner Inner;
		struct _Outer { Inner i; int y; };
		typedef struct _Outer Outer;
		int main(void) {
			Outer o = {.i = {.x = 5}, .y = 10};
			return 0;
		}
		"""
		_, errors = _analyze(src)
		assert len(errors) == 0

	def test_typedef_compound_literal_as_arg(self):
		src = """
		struct _Point { int x; int y; };
		typedef struct _Point Point;
		int use(Point *p) { return p->x; }
		int main(void) {
			int r = use(&(Point){.x = 7, .y = 8});
			return r;
		}
		"""
		_, errors = _analyze(src)
		assert len(errors) == 0

	def test_typedef_compound_literal_ir(self):
		src = """
		struct _Point { int x; int y; };
		typedef struct _Point Point;
		int main(void) {
			Point p = (Point){.x = 10, .y = 20};
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		main_fn = [f for f in ir_prog.functions if f.name == "main"][0]
		stores = [i for i in main_fn.body if isinstance(i, IRStore)]
		assert len(stores) >= 2
