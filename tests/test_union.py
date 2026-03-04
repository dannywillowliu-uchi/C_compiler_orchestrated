"""Comprehensive tests for union type support across the compiler pipeline."""

import pytest

from compiler.ast_nodes import (
	MemberAccess,
	UnionDecl,
	VarDecl,
)
from compiler.codegen import CodeGenerator
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer, SemanticError


def parse(source: str):
	tokens = Lexer(source).tokenize()
	return Parser(tokens).parse()


def analyze(source: str):
	prog = parse(source)
	analyzer = SemanticAnalyzer()
	analyzer.analyze(prog)
	return prog, analyzer


def compile_to_asm(source: str) -> str:
	prog = parse(source)
	analyzer = SemanticAnalyzer()
	analyzer.analyze(prog)
	ir_prog = IRGenerator().generate(prog)
	return CodeGenerator().generate(ir_prog)


# ---------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------


class TestUnionParsing:
	def test_union_declaration(self):
		prog = parse("union Data { int i; char c; };")
		assert len(prog.declarations) == 1
		decl = prog.declarations[0]
		assert isinstance(decl, UnionDecl)
		assert decl.name == "Data"
		assert len(decl.members) == 2
		assert decl.members[0].name == "i"
		assert decl.members[0].type_spec.base_type == "int"
		assert decl.members[1].name == "c"
		assert decl.members[1].type_spec.base_type == "char"

	def test_union_variable_declaration(self):
		prog = parse("union Data { int i; char c; }; int main() { union Data d; return 0; }")
		func = prog.declarations[1]
		stmts = func.body.statements
		assert isinstance(stmts[0], VarDecl)
		assert stmts[0].type_spec.base_type == "union Data"

	def test_union_pointer_variable(self):
		prog = parse("union Data { int i; char c; }; int main() { union Data *p; return 0; }")
		func = prog.declarations[1]
		stmts = func.body.statements
		assert isinstance(stmts[0], VarDecl)
		assert stmts[0].type_spec.base_type == "union Data"
		assert stmts[0].type_spec.pointer_count == 1

	def test_union_member_access_dot(self):
		prog = parse("""
			union Data { int i; char c; };
			int main() {
				union Data d;
				d.i = 42;
				return d.i;
			}
		""")
		func = prog.declarations[1]
		stmts = func.body.statements
		# d.i = 42 is an ExprStmt wrapping an Assignment with MemberAccess target
		assign = stmts[1].expression
		assert isinstance(assign.target, MemberAccess)
		assert assign.target.member == "i"
		assert assign.target.is_arrow is False

	def test_union_member_access_arrow(self):
		prog = parse("""
			union Data { int i; char c; };
			int main() {
				union Data d;
				union Data *p;
				p = &d;
				p->i = 10;
				return p->i;
			}
		""")
		func = prog.declarations[1]
		stmts = func.body.statements
		# p->i = 10
		assign = stmts[3].expression
		assert isinstance(assign.target, MemberAccess)
		assert assign.target.member == "i"
		assert assign.target.is_arrow is True

	def test_typedef_union(self):
		prog = parse("""
			typedef union { int i; char c; } Data;
			int main() {
				Data d;
				return 0;
			}
		""")
		assert len(prog.declarations) == 2

	def test_typedef_union_named(self):
		prog = parse("""
			typedef union MyUnion { int x; double y; } MyU;
			int main() {
				MyU u;
				return 0;
			}
		""")
		assert len(prog.declarations) == 2

	def test_union_with_multiple_types(self):
		prog = parse("""
			union Value {
				int i;
				float f;
				char c;
				double d;
			};
		""")
		decl = prog.declarations[0]
		assert isinstance(decl, UnionDecl)
		assert len(decl.members) == 4

	def test_union_in_function_scope(self):
		prog = parse("""
			int main() {
				union Local { int a; char b; };
				union Local x;
				return 0;
			}
		""")
		func = prog.declarations[0]
		stmts = func.body.statements
		assert isinstance(stmts[0], UnionDecl)


# ---------------------------------------------------------------
# Semantic analysis tests
# ---------------------------------------------------------------


class TestUnionSemantics:
	def test_union_member_access_valid(self):
		analyze("""
			union Data { int i; char c; };
			int main() {
				union Data d;
				d.i = 42;
				return d.i;
			}
		""")

	def test_union_member_access_arrow_valid(self):
		analyze("""
			union Data { int i; char c; };
			int main() {
				union Data d;
				union Data *p;
				p = &d;
				p->i = 10;
				return p->i;
			}
		""")

	def test_union_invalid_member(self):
		with pytest.raises(SemanticError, match="no member named 'z'"):
			analyze("""
				union Data { int i; char c; };
				int main() {
					union Data d;
					d.z = 1;
					return 0;
				}
			""")

	def test_union_dot_on_pointer_error(self):
		with pytest.raises(SemanticError, match="non-pointer"):
			analyze("""
				union Data { int i; char c; };
				int main() {
					union Data *p;
					p.i = 1;
					return 0;
				}
			""")

	def test_union_arrow_on_non_pointer_error(self):
		with pytest.raises(SemanticError, match="pointer"):
			analyze("""
				union Data { int i; char c; };
				int main() {
					union Data d;
					d->i = 1;
					return 0;
				}
			""")

	def test_union_duplicate_member(self):
		with pytest.raises(SemanticError, match="duplicate member"):
			analyze("""
				union Bad { int x; int x; };
				int main() { return 0; }
			""")

	def test_union_redefinition(self):
		with pytest.raises(SemanticError, match="redefinition of union"):
			analyze("""
				union Data { int i; };
				union Data { char c; };
				int main() { return 0; }
			""")

	def test_typedef_union_semantics(self):
		analyze("""
			typedef union { int i; float f; } Value;
			int main() {
				Value v;
				v.i = 5;
				return v.i;
			}
		""")


# ---------------------------------------------------------------
# IR generation tests
# ---------------------------------------------------------------


class TestUnionIRGen:
	def test_union_sizeof(self):
		"""sizeof(union) should be max member size."""
		prog = parse("""
			union Data { int i; double d; char c; };
			int main() {
				return sizeof(union Data);
			}
		""")
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		# Should have generated IR; the sizeof should resolve to 8 (double)
		assert len(ir_prog.functions) == 1
		# Check the return value in IR - find the IRReturn with a const of 8
		from compiler.ir import IRConst, IRReturn
		found_sizeof = False
		for instr in ir_prog.functions[0].body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				assert instr.value.value == 8
				found_sizeof = True
		assert found_sizeof, "Expected sizeof(union Data) == 8 in IR"

	def test_union_sizeof_int_char(self):
		"""sizeof(union) with int and char should be 4 (int size)."""
		prog = parse("""
			union Small { int i; char c; };
			int main() {
				return sizeof(union Small);
			}
		""")
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		from compiler.ir import IRConst, IRReturn
		for instr in ir_prog.functions[0].body:
			if isinstance(instr, IRReturn) and isinstance(instr.value, IRConst):
				assert instr.value.value == 4

	def test_union_member_write_read(self):
		"""Writing to a union member and reading it back should produce valid IR."""
		prog = parse("""
			union Data { int i; char c; };
			int main() {
				union Data d;
				d.i = 42;
				return d.i;
			}
		""")
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		assert len(ir_prog.functions) == 1
		# Just verify it generates without error
		assert len(ir_prog.functions[0].body) > 0

	def test_union_arrow_member_access(self):
		"""Arrow member access on union pointer should generate valid IR."""
		prog = parse("""
			union Data { int i; char c; };
			int main() {
				union Data d;
				union Data *p;
				p = &d;
				p->i = 99;
				return p->i;
			}
		""")
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		assert len(ir_prog.functions) == 1


# ---------------------------------------------------------------
# Codegen tests
# ---------------------------------------------------------------


class TestUnionCodegen:
	def test_union_compiles_to_asm(self):
		asm = compile_to_asm("""
			union Data { int i; char c; };
			int main() {
				union Data d;
				d.i = 42;
				return d.i;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_union_pointer_compiles_to_asm(self):
		asm = compile_to_asm("""
			union Data { int i; char c; };
			int main() {
				union Data d;
				union Data *p;
				p = &d;
				p->i = 100;
				return p->i;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_union_sizeof_compiles(self):
		asm = compile_to_asm("""
			union Data { int i; double d; };
			int main() {
				return sizeof(union Data);
			}
		""")
		assert "main:" in asm

	def test_nested_union_in_struct(self):
		"""A struct containing a union member should compile."""
		asm = compile_to_asm("""
			union Value { int i; float f; };
			struct Container {
				int tag;
				union Value val;
			};
			int main() {
				struct Container c;
				c.tag = 1;
				return c.tag;
			}
		""")
		assert "main:" in asm

	def test_union_assignment(self):
		asm = compile_to_asm("""
			union Data { int i; char c; };
			int main() {
				union Data a;
				union Data b;
				a.i = 5;
				b.i = a.i;
				return b.i;
			}
		""")
		assert "main:" in asm

	def test_typedef_union_codegen(self):
		asm = compile_to_asm("""
			typedef union { int i; char c; } Data;
			int main() {
				Data d;
				d.i = 7;
				return d.i;
			}
		""")
		assert "main:" in asm
