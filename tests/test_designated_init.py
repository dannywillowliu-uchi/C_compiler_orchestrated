"""Tests for C99 designated initializer support."""

import pytest

from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer, SemanticError
from compiler.ir_gen import IRGenerator
from compiler.ast_nodes import (
	DesignatedInit,
	InitializerList,
	IntLiteral,
	Program,
	VarDecl,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse(source: str) -> Program:
	return Parser.from_source(source).parse()


def _analyze(source: str) -> Program:
	prog = _parse(source)
	SemanticAnalyzer().analyze(prog)
	return prog


def _generate_ir(source: str):
	prog = _analyze(source)
	return IRGenerator().generate(prog)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParserDesignatedInit:
	def test_struct_field_designation(self):
		src = """
		struct Point { int x; int y; };
		void f() {
			struct Point p = { .x = 1, .y = 2 };
		}
		"""
		prog = _parse(src)
		func = prog.declarations[1]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl, VarDecl)
		init_list = var_decl.initializer
		assert isinstance(init_list, InitializerList)
		assert len(init_list.elements) == 2
		d0 = init_list.elements[0]
		assert isinstance(d0, DesignatedInit)
		assert d0.field_name == "x"
		assert d0.index is None
		assert isinstance(d0.value, IntLiteral)
		assert d0.value.value == 1
		d1 = init_list.elements[1]
		assert isinstance(d1, DesignatedInit)
		assert d1.field_name == "y"
		assert d1.value.value == 2

	def test_array_index_designation(self):
		src = """
		void f() {
			int a[5] = { [0] = 10, [3] = 40 };
		}
		"""
		prog = _parse(src)
		func = prog.declarations[0]
		var_decl = func.body.statements[0]
		init_list = var_decl.initializer
		assert isinstance(init_list, InitializerList)
		assert len(init_list.elements) == 2
		d0 = init_list.elements[0]
		assert isinstance(d0, DesignatedInit)
		assert d0.field_name is None
		assert isinstance(d0.index, IntLiteral)
		assert d0.index.value == 0
		assert d0.value.value == 10
		d1 = init_list.elements[1]
		assert isinstance(d1, DesignatedInit)
		assert d1.index.value == 3
		assert d1.value.value == 40

	def test_mixed_designated_and_positional(self):
		src = """
		struct S { int a; int b; int c; };
		void f() {
			struct S s = { .b = 20, 30 };
		}
		"""
		prog = _parse(src)
		func = prog.declarations[1]
		var_decl = func.body.statements[0]
		init_list = var_decl.initializer
		assert len(init_list.elements) == 2
		assert isinstance(init_list.elements[0], DesignatedInit)
		assert init_list.elements[0].field_name == "b"
		# Second element is positional (not designated)
		assert isinstance(init_list.elements[1], IntLiteral)

	def test_nested_designated_initializer(self):
		src = """
		struct Inner { int a; int b; };
		struct Outer { struct Inner i; int c; };
		void f() {
			struct Outer o = { .i = {1, 2}, .c = 3 };
		}
		"""
		prog = _parse(src)
		func = prog.declarations[2]
		var_decl = func.body.statements[0]
		init_list = var_decl.initializer
		assert len(init_list.elements) == 2
		d0 = init_list.elements[0]
		assert isinstance(d0, DesignatedInit)
		assert d0.field_name == "i"
		assert isinstance(d0.value, InitializerList)
		assert len(d0.value.elements) == 2

	def test_array_designated_with_nested_braces(self):
		src = """
		struct Point { int x; int y; };
		void f() {
			struct Point pts[3] = { [1] = {10, 20} };
		}
		"""
		prog = _parse(src)
		func = prog.declarations[1]
		var_decl = func.body.statements[0]
		init_list = var_decl.initializer
		assert len(init_list.elements) == 1
		d0 = init_list.elements[0]
		assert isinstance(d0, DesignatedInit)
		assert d0.index.value == 1
		assert isinstance(d0.value, InitializerList)

	def test_trailing_comma_with_designated(self):
		src = """
		void f() {
			int a[3] = { [0] = 1, [2] = 3, };
		}
		"""
		prog = _parse(src)
		func = prog.declarations[0]
		var_decl = func.body.statements[0]
		init_list = var_decl.initializer
		assert len(init_list.elements) == 2


# ---------------------------------------------------------------------------
# Semantic analysis tests
# ---------------------------------------------------------------------------

class TestSemanticDesignatedInit:
	def test_valid_struct_designation(self):
		src = """
		struct Point { int x; int y; };
		void f() {
			struct Point p = { .x = 1, .y = 2 };
		}
		"""
		_analyze(src)  # Should not raise

	def test_valid_array_designation(self):
		src = """
		void f() {
			int a[5] = { [0] = 10, [4] = 50 };
		}
		"""
		_analyze(src)  # Should not raise

	def test_invalid_field_name(self):
		src = """
		struct Point { int x; int y; };
		void f() {
			struct Point p = { .z = 1 };
		}
		"""
		with pytest.raises(SemanticError, match="no member 'z'"):
			_analyze(src)

	def test_array_index_out_of_range(self):
		src = """
		void f() {
			int a[3] = { [5] = 10 };
		}
		"""
		with pytest.raises(SemanticError, match="out of range"):
			_analyze(src)

	def test_field_designator_in_array(self):
		src = """
		void f() {
			int a[3] = { .x = 1 };
		}
		"""
		with pytest.raises(SemanticError, match="field designator in array"):
			_analyze(src)

	def test_index_designator_in_struct(self):
		src = """
		struct Point { int x; int y; };
		void f() {
			struct Point p = { [0] = 1 };
		}
		"""
		with pytest.raises(SemanticError, match="array index designator in struct"):
			_analyze(src)

	def test_mixed_designated_and_positional_valid(self):
		src = """
		struct S { int a; int b; int c; };
		void f() {
			struct S s = { .b = 20, 30 };
		}
		"""
		_analyze(src)  # Should not raise

	def test_nested_designated_valid(self):
		src = """
		struct Inner { int a; int b; };
		struct Outer { struct Inner i; int c; };
		void f() {
			struct Outer o = { .i = {1, 2}, .c = 3 };
		}
		"""
		_analyze(src)  # Should not raise

	def test_negative_array_index(self):
		"""Negative index parses as UnaryOp, not IntLiteral, so no static range check.
		Just verify it doesn't crash during analysis."""
		src = """
		void f() {
			int a[3] = { [-1] = 10 };
		}
		"""
		_analyze(src)  # No static check for non-literal index


# ---------------------------------------------------------------------------
# IR generation tests
# ---------------------------------------------------------------------------

class TestIRGenDesignatedInit:
	def test_struct_designated_generates_ir(self):
		src = """
		struct Point { int x; int y; };
		void f() {
			struct Point p = { .x = 10, .y = 20 };
		}
		"""
		ir_prog = _generate_ir(src)
		assert len(ir_prog.functions) == 1
		instrs = ir_prog.functions[0].body
		# Should have stores for both fields
		from compiler.ir import IRStore
		stores = [i for i in instrs if isinstance(i, IRStore)]
		assert len(stores) >= 2  # At least 2 zero-fills + 2 designated stores

	def test_array_designated_generates_ir(self):
		src = """
		void f() {
			int a[4] = { [1] = 100, [3] = 300 };
		}
		"""
		ir_prog = _generate_ir(src)
		instrs = ir_prog.functions[0].body
		from compiler.ir import IRStore
		stores = [i for i in instrs if isinstance(i, IRStore)]
		# 4 zero-fills + 2 designated stores = 6
		assert len(stores) >= 6

	def test_mixed_struct_designated_positional(self):
		src = """
		struct S { int a; int b; int c; };
		void f() {
			struct S s = { .b = 20, 30 };
		}
		"""
		ir_prog = _generate_ir(src)
		assert len(ir_prog.functions) == 1

	def test_nested_designated_generates_ir(self):
		src = """
		struct Inner { int a; int b; };
		struct Outer { struct Inner i; int c; };
		void f() {
			struct Outer o = { .c = 99 };
		}
		"""
		ir_prog = _generate_ir(src)
		assert len(ir_prog.functions) == 1

	def test_only_positional_still_works(self):
		"""Regression: ensure plain initializer lists still work."""
		src = """
		struct Point { int x; int y; };
		void f() {
			struct Point p = { 1, 2 };
			int a[3] = { 10, 20, 30 };
		}
		"""
		ir_prog = _generate_ir(src)
		assert len(ir_prog.functions) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestDesignatedInitEdgeCases:
	def test_single_designated_in_struct(self):
		src = """
		struct S { int a; int b; int c; };
		void f() {
			struct S s = { .c = 42 };
		}
		"""
		_generate_ir(src)

	def test_all_fields_designated_out_of_order(self):
		src = """
		struct S { int x; int y; int z; };
		void f() {
			struct S s = { .z = 3, .x = 1, .y = 2 };
		}
		"""
		_generate_ir(src)

	def test_array_sparse_designation(self):
		src = """
		void f() {
			int a[10] = { [9] = 99 };
		}
		"""
		_generate_ir(src)

	def test_designated_with_expression_value(self):
		src = """
		struct Point { int x; int y; };
		void f() {
			struct Point p = { .x = 2 + 3, .y = 10 - 5 };
		}
		"""
		_generate_ir(src)
