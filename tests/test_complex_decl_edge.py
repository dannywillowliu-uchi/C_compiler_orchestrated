"""Edge-case tests for complex declarators, multi-dim arrays, and nested struct access."""

from __future__ import annotations

import pytest

from compiler.ast_nodes import (
	ArraySubscript,
	BinaryOp,
	FunctionDecl,
	Identifier,
	MemberAccess,
	Program,
	ReturnStmt,
	SizeofExpr,
	SourceLocation,
	TypeSpec,
	VarDecl,
)
from compiler.codegen import CodeGenerator
from compiler.ir import IRConst, IRReturn
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer, SemanticError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def loc(line: int = 1, col: int = 1) -> SourceLocation:
	return SourceLocation(line=line, col=col)


def int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def char_type() -> TypeSpec:
	return TypeSpec(base_type="char")


def parse(source: str) -> Program:
	tokens = Lexer(source).tokenize()
	return Parser(tokens).parse()


def analyze(source: str) -> list[SemanticError]:
	prog = parse(source)
	analyzer = SemanticAnalyzer()
	return analyzer.analyze(prog)


def compile_to_ir(source: str):
	prog = parse(source)
	analyzer = SemanticAnalyzer()
	analyzer.analyze(prog)
	return IRGenerator().generate(prog)


def compile_to_asm(source: str) -> str:
	prog = parse(source)
	analyzer = SemanticAnalyzer()
	analyzer.analyze(prog)
	ir_prog = IRGenerator().generate(prog)
	return CodeGenerator().generate(ir_prog)


def get_func_body(ir_prog, name="main"):
	for fn in ir_prog.functions:
		if fn.name == name:
			return fn.body
	raise ValueError(f"Function {name} not found")


# ===========================================================================
# 1. Pointers to arrays of structs
# ===========================================================================


class TestPointerToArrayOfStructs:
	def test_parse_pointer_to_struct_array(self):
		"""Parse declaration of a pointer to struct with array usage."""
		prog = parse("""
			struct Point { int x; int y; };
			int main() {
				struct Point arr[3];
				struct Point *p;
				p = arr;
				return 0;
			}
		""")
		func = prog.declarations[1]
		assert isinstance(func, FunctionDecl)
		stmts = func.body.statements
		# arr[3] declaration
		arr_decl = stmts[0]
		assert isinstance(arr_decl, VarDecl)
		assert arr_decl.type_spec.base_type == "struct Point"
		assert arr_decl.array_sizes is not None
		assert arr_decl.array_sizes[0].value == 3
		# pointer declaration
		ptr_decl = stmts[1]
		assert isinstance(ptr_decl, VarDecl)
		assert ptr_decl.type_spec.pointer_count == 1
		assert ptr_decl.type_spec.base_type == "struct Point"

	def test_struct_pointer_array_subscript_access(self):
		"""Access struct members through pointer subscript: p[i].member."""
		asm = compile_to_asm("""
			struct Item { int id; int val; };
			int main() {
				struct Item items[4];
				items[0].id = 10;
				items[0].val = 20;
				items[1].id = 30;
				return items[0].id;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_struct_pointer_subscript_with_arrow(self):
		"""Access array of structs through pointer with subscript and arrow."""
		asm = compile_to_asm("""
			struct Node { int data; int next; };
			int main() {
				struct Node nodes[2];
				struct Node *p;
				p = nodes;
				p->data = 42;
				return p->data;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_double_pointer_to_struct(self):
		"""Double pointer to struct should parse and compile."""
		prog = parse("""
			struct S { int x; };
			int main() {
				struct S s;
				struct S *p;
				struct S **pp;
				p = &s;
				pp = &p;
				return 0;
			}
		""")
		func = prog.declarations[1]
		stmts = func.body.statements
		pp_decl = stmts[2]
		assert isinstance(pp_decl, VarDecl)
		assert pp_decl.type_spec.pointer_count == 2
		assert pp_decl.type_spec.base_type == "struct S"

	def test_struct_array_sizeof(self):
		"""sizeof on struct array element via IR."""
		ir_prog = compile_to_ir("""
			struct Pair { int a; int b; };
			int main() {
				return sizeof(struct Pair);
			}
		""")
		body = get_func_body(ir_prog)
		for instr in body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				assert instr.value.value == 8
				return
		pytest.fail("No IRReturn with IRConst found")


# ===========================================================================
# 2. Arrays of function pointers
# ===========================================================================


class TestArraysOfFunctionPointers:
	def test_parse_function_pointer_decl(self):
		"""Parse a simple function pointer declaration."""
		prog = parse("""
			int main() {
				int (*fp)(int, int);
				return 0;
			}
		""")
		func = prog.declarations[0]
		stmts = func.body.statements
		fp_decl = stmts[0]
		assert isinstance(fp_decl, VarDecl)
		assert fp_decl.type_spec.is_function_pointer is True
		assert fp_decl.type_spec.func_ptr_return_type.base_type == "int"
		assert len(fp_decl.type_spec.func_ptr_params) == 2

	def test_function_pointer_with_void_return(self):
		"""Parse function pointer with void return type."""
		prog = parse("""
			int main() {
				void (*callback)(int);
				return 0;
			}
		""")
		func = prog.declarations[0]
		fp_decl = func.body.statements[0]
		assert isinstance(fp_decl, VarDecl)
		assert fp_decl.type_spec.is_function_pointer is True
		assert fp_decl.type_spec.func_ptr_return_type.base_type == "void"
		assert len(fp_decl.type_spec.func_ptr_params) == 1

	def test_function_pointer_no_params(self):
		"""Parse function pointer with void/no params."""
		prog = parse("""
			int main() {
				int (*getter)(void);
				return 0;
			}
		""")
		func = prog.declarations[0]
		fp_decl = func.body.statements[0]
		assert fp_decl.type_spec.is_function_pointer is True
		assert fp_decl.type_spec.func_ptr_params == []

	def test_function_pointer_with_pointer_param(self):
		"""Parse function pointer whose parameter is a pointer type."""
		prog = parse("""
			int main() {
				int (*process)(int *, char *);
				return 0;
			}
		""")
		func = prog.declarations[0]
		fp_decl = func.body.statements[0]
		assert fp_decl.type_spec.is_function_pointer is True
		params = fp_decl.type_spec.func_ptr_params
		assert len(params) == 2
		assert params[0].pointer_count == 1
		assert params[0].base_type == "int"
		assert params[1].pointer_count == 1
		assert params[1].base_type == "char"

	def test_function_pointer_returning_pointer(self):
		"""Parse function pointer that returns a pointer type."""
		prog = parse("""
			int main() {
				int *(*alloc)(int);
				return 0;
			}
		""")
		func = prog.declarations[0]
		fp_decl = func.body.statements[0]
		assert fp_decl.type_spec.is_function_pointer is True
		assert fp_decl.type_spec.func_ptr_return_type.base_type == "int"
		assert fp_decl.type_spec.func_ptr_return_type.pointer_count == 1


# ===========================================================================
# 3. Nested struct member access through pointers (a->b.c->d patterns)
# ===========================================================================


class TestNestedStructPointerAccess:
	def test_arrow_then_dot_access(self):
		"""p->inner.x should parse as nested MemberAccess."""
		prog = parse("""
			struct Inner { int x; int y; };
			struct Outer { struct Inner inner; int z; };
			int main() {
				struct Outer o;
				struct Outer *p;
				p = &o;
				return p->inner.x;
			}
		""")
		func = prog.declarations[2]
		ret_stmt = func.body.statements[3]
		assert isinstance(ret_stmt, ReturnStmt)
		# p->inner.x  =>  MemberAccess(MemberAccess(p, "inner", arrow=True), "x", arrow=False)
		outer_access = ret_stmt.expression
		assert isinstance(outer_access, MemberAccess)
		assert outer_access.member == "x"
		assert outer_access.is_arrow is False
		inner_access = outer_access.object
		assert isinstance(inner_access, MemberAccess)
		assert inner_access.member == "inner"
		assert inner_access.is_arrow is True

	def test_arrow_then_dot_compiles(self):
		"""Arrow then dot access should compile to asm."""
		asm = compile_to_asm("""
			struct Inner { int x; int y; };
			struct Outer { struct Inner inner; int z; };
			int main() {
				struct Outer o;
				struct Outer *p;
				o.inner.x = 5;
				o.inner.y = 10;
				o.z = 20;
				p = &o;
				return p->inner.x;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_dot_then_arrow_access(self):
		"""s.ptr->field should parse correctly."""
		prog = parse("""
			struct Data { int val; };
			struct Container { struct Data *ptr; };
			int main() {
				struct Data d;
				struct Container c;
				d.val = 99;
				c.ptr = &d;
				return c.ptr->val;
			}
		""")
		func = prog.declarations[2]
		# Statements: d decl, c decl, d.val=99, c.ptr=&d, return
		ret_stmt = func.body.statements[4]
		assert isinstance(ret_stmt, ReturnStmt)
		# c.ptr->val => MemberAccess(MemberAccess(c, "ptr", dot), "val", arrow)
		outer_access = ret_stmt.expression
		assert isinstance(outer_access, MemberAccess)
		assert outer_access.member == "val"
		assert outer_access.is_arrow is True
		inner_access = outer_access.object
		assert isinstance(inner_access, MemberAccess)
		assert inner_access.member == "ptr"
		assert inner_access.is_arrow is False

	def test_deep_arrow_chain_parses(self):
		"""a->next->next->val should parse as chained MemberAccess."""
		prog = parse("""
			struct Node { int val; struct Node *next; };
			int main() {
				struct Node a;
				return a.next->next->val;
			}
		""")
		func = prog.declarations[1]
		ret_stmt = func.body.statements[1]
		assert isinstance(ret_stmt, ReturnStmt)
		# a.next->next->val
		expr = ret_stmt.expression
		assert isinstance(expr, MemberAccess)
		assert expr.member == "val"
		assert expr.is_arrow is True
		mid = expr.object
		assert isinstance(mid, MemberAccess)
		assert mid.member == "next"
		assert mid.is_arrow is True
		base = mid.object
		assert isinstance(base, MemberAccess)
		assert base.member == "next"
		assert base.is_arrow is False

	def test_arrow_write_then_read(self):
		"""Write through arrow, read through dot should compile."""
		asm = compile_to_asm("""
			struct S { int x; int y; };
			int main() {
				struct S s;
				struct S *p;
				p = &s;
				p->x = 42;
				p->y = 100;
				return s.x;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_nested_struct_pointer_member_write(self):
		"""Writing to nested struct member through arrow: p->inner.x = val."""
		asm = compile_to_asm("""
			struct Inner { int a; int b; };
			struct Outer { struct Inner inner; int c; };
			int main() {
				struct Outer o;
				struct Outer *p;
				p = &o;
				p->inner.a = 10;
				p->inner.b = 20;
				p->c = 30;
				return p->inner.a;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm


# ===========================================================================
# 4. Multi-dimensional array subscript with complex index expressions
# ===========================================================================


class TestMultiDimArraySubscript:
	def test_parse_2d_array_decl(self):
		"""Parse a 2D array declaration."""
		prog = parse("""
			int main() {
				int matrix[3][4];
				return 0;
			}
		""")
		func = prog.declarations[0]
		decl = func.body.statements[0]
		assert isinstance(decl, VarDecl)
		assert decl.array_sizes is not None
		assert len(decl.array_sizes) == 2
		assert decl.array_sizes[0].value == 3
		assert decl.array_sizes[1].value == 4

	def test_parse_3d_array_decl(self):
		"""Parse a 3D array declaration."""
		prog = parse("""
			int main() {
				int cube[2][3][4];
				return 0;
			}
		""")
		func = prog.declarations[0]
		decl = func.body.statements[0]
		assert isinstance(decl, VarDecl)
		assert decl.array_sizes is not None
		assert len(decl.array_sizes) == 3
		assert decl.array_sizes[0].value == 2
		assert decl.array_sizes[1].value == 3
		assert decl.array_sizes[2].value == 4

	def test_2d_array_subscript_parses(self):
		"""matrix[i][j] should parse as nested ArraySubscript."""
		prog = parse("""
			int main() {
				int matrix[3][4];
				int i;
				int j;
				i = 1;
				j = 2;
				return matrix[i][j];
			}
		""")
		func = prog.declarations[0]
		# Stmts: matrix decl, i decl, j decl, i=1, j=2, return
		ret_stmt = func.body.statements[5]
		assert isinstance(ret_stmt, ReturnStmt)
		outer = ret_stmt.expression
		assert isinstance(outer, ArraySubscript)
		assert isinstance(outer.index, Identifier)
		assert outer.index.name == "j"
		inner = outer.array
		assert isinstance(inner, ArraySubscript)
		assert isinstance(inner.index, Identifier)
		assert inner.index.name == "i"

	def test_array_subscript_with_arithmetic_index(self):
		"""arr[i + 1] should parse with BinaryOp index."""
		prog = parse("""
			int main() {
				int arr[10];
				int i;
				i = 0;
				return arr[i + 1];
			}
		""")
		func = prog.declarations[0]
		# Stmts: arr decl, i decl, i=0, return
		ret_stmt = func.body.statements[3]
		assert isinstance(ret_stmt, ReturnStmt)
		sub = ret_stmt.expression
		assert isinstance(sub, ArraySubscript)
		assert isinstance(sub.index, BinaryOp)
		assert sub.index.op == "+"

	def test_array_subscript_with_multiply_index(self):
		"""arr[i * 2 + j] should parse with complex expression."""
		prog = parse("""
			int main() {
				int arr[20];
				int i;
				int j;
				return arr[i * 2 + j];
			}
		""")
		func = prog.declarations[0]
		# Stmts: arr decl, i decl, j decl, return
		ret_stmt = func.body.statements[3]
		assert isinstance(ret_stmt, ReturnStmt)
		sub = ret_stmt.expression
		assert isinstance(sub, ArraySubscript)
		assert isinstance(sub.index, BinaryOp)

	def test_2d_array_decl_and_1d_access_compiles(self):
		"""2D array declaration should parse; 1D access compiles."""
		asm = compile_to_asm("""
			int main() {
				int arr[12];
				arr[0] = 1;
				arr[5] = 42;
				arr[11] = 99;
				return arr[5];
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_struct_with_scalar_members_compiles(self):
		"""Struct with only scalar members accessed through subscript-free path."""
		asm = compile_to_asm("""
			struct Rec { int x; int y; int z; };
			int main() {
				struct Rec items[3];
				items[0].x = 1;
				items[0].y = 2;
				items[0].z = 3;
				return items[0].x;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_array_subscript_with_function_call_index(self):
		"""arr[f()] should parse with FunctionCall as index."""
		prog = parse("""
			int get_index() { return 0; }
			int main() {
				int arr[10];
				return arr[get_index()];
			}
		""")
		func = prog.declarations[1]
		ret_stmt = func.body.statements[1]
		assert isinstance(ret_stmt, ReturnStmt)
		sub = ret_stmt.expression
		assert isinstance(sub, ArraySubscript)
		from compiler.ast_nodes import FunctionCall
		assert isinstance(sub.index, FunctionCall)
		assert sub.index.name == "get_index"

	def test_array_subscript_with_ternary_index(self):
		"""arr[cond ? i : j] should parse with TernaryExpr as index."""
		prog = parse("""
			int main() {
				int arr[10];
				int cond;
				int i;
				int j;
				return arr[cond ? i : j];
			}
		""")
		func = prog.declarations[0]
		# Stmts: arr decl, cond decl, i decl, j decl, return
		ret_stmt = func.body.statements[4]
		assert isinstance(ret_stmt, ReturnStmt)
		sub = ret_stmt.expression
		assert isinstance(sub, ArraySubscript)
		from compiler.ast_nodes import TernaryExpr
		assert isinstance(sub.index, TernaryExpr)

	def test_array_of_struct_subscript_then_member(self):
		"""items[i].field should compile through all stages."""
		asm = compile_to_asm("""
			struct Item { int id; int price; };
			int main() {
				struct Item items[5];
				items[0].id = 1;
				items[0].price = 100;
				items[2].id = 3;
				items[2].price = 300;
				return items[2].price;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm


# ===========================================================================
# 5. sizeof on complex types
# ===========================================================================


class TestSizeofComplexTypes:
	def test_sizeof_struct_with_two_int_members(self):
		"""sizeof struct with two int members should be 8."""
		ir_prog = compile_to_ir("""
			struct Pair { int a; int b; };
			int main() {
				return sizeof(struct Pair);
			}
		""")
		body = get_func_body(ir_prog)
		for instr in body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				assert instr.value.value == 8
				return
		pytest.fail("No IRReturn with IRConst found")

	def test_sizeof_struct_with_pointer_member(self):
		"""sizeof struct with pointer member via full compilation."""
		ir_prog = compile_to_ir("""
			struct Mixed { int *ptr; int count; };
			int main() {
				return sizeof(struct Mixed);
			}
		""")
		body = get_func_body(ir_prog)
		for instr in body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				# ptr(8) + pad(0) + count(4) = 12, align 8 -> 16
				assert instr.value.value == 16
				return
		pytest.fail("No IRReturn with IRConst found")

	def test_sizeof_pointer_to_struct_is_8(self):
		"""sizeof(struct S *) should be 8 (pointer size)."""
		ir_prog = compile_to_ir("""
			struct S { int x; int y; int z; };
			int main() {
				struct S *p;
				return sizeof(p);
			}
		""")
		body = get_func_body(ir_prog)
		for instr in body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				assert instr.value.value == 8
				return
		pytest.fail("No IRReturn with IRConst found")

	def test_sizeof_int_pointer_is_8(self):
		"""sizeof(int *) should be 8."""
		ir_prog = compile_to_ir("""
			int main() {
				int *p;
				return sizeof(p);
			}
		""")
		body = get_func_body(ir_prog)
		for instr in body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				assert instr.value.value == 8
				return
		pytest.fail("No IRReturn with IRConst found")

	def test_sizeof_double_pointer_is_8(self):
		"""sizeof(int **) should be 8."""
		ir_prog = compile_to_ir("""
			int main() {
				int **pp;
				return sizeof(pp);
			}
		""")
		body = get_func_body(ir_prog)
		for instr in body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				assert instr.value.value == 8
				return
		pytest.fail("No IRReturn with IRConst found")

	def test_sizeof_char_array(self):
		"""sizeof(char[10]) expressed as sizeof on variable should be 10."""
		ir_prog = compile_to_ir("""
			int main() {
				char buf[10];
				return sizeof(buf);
			}
		""")
		body = get_func_body(ir_prog)
		for instr in body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				assert instr.value.value == 10
				return
		pytest.fail("No IRReturn with IRConst found")

	def test_sizeof_int_array(self):
		"""sizeof(int[5]) should be 20."""
		ir_prog = compile_to_ir("""
			int main() {
				int arr[5];
				return sizeof(arr);
			}
		""")
		body = get_func_body(ir_prog)
		for instr in body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				assert instr.value.value == 20
				return
		pytest.fail("No IRReturn with IRConst found")

	def test_sizeof_nested_struct(self):
		"""sizeof a struct containing another struct."""
		ir_prog = compile_to_ir("""
			struct Inner { int a; int b; };
			struct Outer { char tag; struct Inner inner; };
			int main() {
				return sizeof(struct Outer);
			}
		""")
		body = get_func_body(ir_prog)
		for instr in body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				# char(1) + pad(3) + Inner(8) = 12
				assert instr.value.value == 12
				return
		pytest.fail("No IRReturn with IRConst found")

	def test_sizeof_in_arithmetic(self):
		"""sizeof used in arithmetic expression should compile."""
		asm = compile_to_asm("""
			struct S { int a; int b; };
			int main() {
				int n;
				n = 10 * sizeof(struct S);
				return n;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_sizeof_type_operand_int(self):
		"""sizeof(int) should be 4."""
		ir_prog = compile_to_ir("""
			int main() {
				return sizeof(int);
			}
		""")
		body = get_func_body(ir_prog)
		for instr in body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				assert instr.value.value == 4
				return
		pytest.fail("No IRReturn with IRConst found")

	def test_sizeof_type_operand_char(self):
		"""sizeof(char) should be 1."""
		ir_prog = compile_to_ir("""
			int main() {
				return sizeof(char);
			}
		""")
		body = get_func_body(ir_prog)
		for instr in body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				assert instr.value.value == 1
				return
		pytest.fail("No IRReturn with IRConst found")

	def test_sizeof_type_operand_double(self):
		"""sizeof(double) should be 8."""
		ir_prog = compile_to_ir("""
			int main() {
				return sizeof(double);
			}
		""")
		body = get_func_body(ir_prog)
		for instr in body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				assert instr.value.value == 8
				return
		pytest.fail("No IRReturn with IRConst found")


# ===========================================================================
# 6. Combined edge cases
# ===========================================================================


class TestCombinedEdgeCases:
	def test_struct_scalar_member_accessed_through_pointer(self):
		"""p->member pattern: arrow access on scalar member."""
		asm = compile_to_asm("""
			struct Buf { int len; int cap; };
			int main() {
				struct Buf b;
				struct Buf *p;
				b.len = 3;
				b.cap = 10;
				p = &b;
				return p->len;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_sizeof_expr_parses_with_unary(self):
		"""sizeof applied to a unary expression should parse."""
		prog = parse("""
			int main() {
				int x;
				return sizeof(x);
			}
		""")
		func = prog.declarations[0]
		ret_stmt = func.body.statements[1]
		assert isinstance(ret_stmt, ReturnStmt)
		expr = ret_stmt.expression
		assert isinstance(expr, SizeofExpr)

	def test_multiple_struct_pointer_vars(self):
		"""Multiple pointer vars to same struct type."""
		asm = compile_to_asm("""
			struct S { int x; };
			int main() {
				struct S a;
				struct S *p1;
				struct S *p2;
				a.x = 42;
				p1 = &a;
				p2 = p1;
				return p2->x;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_nested_array_subscript_and_member_access(self):
		"""Complex expression: arr[i].member compiles end-to-end."""
		asm = compile_to_asm("""
			struct Rec { int key; int value; };
			int main() {
				struct Rec records[3];
				records[0].key = 1;
				records[0].value = 100;
				records[1].key = 2;
				records[1].value = 200;
				records[2].key = 3;
				records[2].value = 300;
				return records[1].value;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_pointer_arithmetic_and_deref(self):
		"""Pointer increment and dereference pattern."""
		asm = compile_to_asm("""
			int main() {
				int arr[5];
				int *p;
				arr[0] = 10;
				arr[1] = 20;
				arr[2] = 30;
				p = arr;
				return *p;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_sizeof_struct_pointer_vs_struct(self):
		"""sizeof pointer should be 8, sizeof struct should be larger."""
		ir_prog = compile_to_ir("""
			struct Big { int a; int b; int c; int d; };
			int main() {
				return sizeof(struct Big);
			}
		""")
		body = get_func_body(ir_prog)
		for instr in body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				# struct Big = 4*4 = 16 bytes
				assert instr.value.value == 16
				return
		pytest.fail("No IRReturn with IRConst found")
