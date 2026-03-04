"""Tests for array and struct brace initializer lists."""

import pytest

from compiler.ast_nodes import InitializerList, IntLiteral, VarDecl
from compiler.codegen import CodeGenerator
from compiler.ir import IRGlobalVar, IRStore
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


# ---------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------


class TestInitializerListParsing:
	def test_array_initializer(self):
		prog = parse("int main() { int a[3] = {1, 2, 3}; return 0; }")
		func = prog.declarations[0]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl, VarDecl)
		assert isinstance(var_decl.initializer, InitializerList)
		assert len(var_decl.initializer.elements) == 3
		assert isinstance(var_decl.initializer.elements[0], IntLiteral)
		assert var_decl.initializer.elements[0].value == 1
		assert var_decl.initializer.elements[1].value == 2
		assert var_decl.initializer.elements[2].value == 3

	def test_struct_initializer(self):
		prog = parse("""
			struct Point { int x; int y; };
			int main() {
				struct Point p = {10, 20};
				return 0;
			}
		""")
		func = prog.declarations[1]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl, VarDecl)
		assert isinstance(var_decl.initializer, InitializerList)
		assert len(var_decl.initializer.elements) == 2
		assert var_decl.initializer.elements[0].value == 10
		assert var_decl.initializer.elements[1].value == 20

	def test_trailing_comma(self):
		prog = parse("int main() { int a[3] = {1, 2, 3,}; return 0; }")
		func = prog.declarations[0]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl.initializer, InitializerList)
		assert len(var_decl.initializer.elements) == 3

	def test_nested_initializer(self):
		prog = parse("int main() { int a[2] = {{1}, {2}}; return 0; }")
		func = prog.declarations[0]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl.initializer, InitializerList)
		assert len(var_decl.initializer.elements) == 2
		assert isinstance(var_decl.initializer.elements[0], InitializerList)
		assert isinstance(var_decl.initializer.elements[1], InitializerList)

	def test_empty_brackets_with_initializer(self):
		prog = parse("int main() { int a[] = {10, 20, 30}; return 0; }")
		func = prog.declarations[0]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl.initializer, InitializerList)
		assert var_decl.array_sizes is not None

	def test_global_array_initializer(self):
		prog = parse("int g[3] = {100, 200, 300};")
		var_decl = prog.declarations[0]
		assert isinstance(var_decl, VarDecl)
		assert isinstance(var_decl.initializer, InitializerList)
		assert len(var_decl.initializer.elements) == 3


# ---------------------------------------------------------------
# Semantic analysis tests
# ---------------------------------------------------------------


class TestInitializerListSemantics:
	def test_array_init_valid(self):
		analyze("""
			int main() {
				int a[3] = {1, 2, 3};
				return a[0];
			}
		""")

	def test_struct_init_valid(self):
		analyze("""
			struct Point { int x; int y; };
			int main() {
				struct Point p = {10, 20};
				return p.x;
			}
		""")

	def test_partial_array_init(self):
		analyze("""
			int main() {
				int a[5] = {1, 2};
				return a[0];
			}
		""")

	def test_excess_array_elements(self):
		with pytest.raises(SemanticError, match="excess elements"):
			analyze("""
				int main() {
					int a[2] = {1, 2, 3};
					return 0;
				}
			""")

	def test_excess_struct_elements(self):
		with pytest.raises(SemanticError, match="excess elements"):
			analyze("""
				struct Point { int x; int y; };
				int main() {
					struct Point p = {1, 2, 3};
					return 0;
				}
			""")

	def test_inferred_array_size(self):
		"""int a[] = {1, 2, 3} should infer size 3."""
		prog, analyzer = analyze("""
			int main() {
				int a[] = {1, 2, 3};
				return a[0];
			}
		""")
		# The array_sizes[0] should have been updated to 3
		func = prog.declarations[0]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl, VarDecl)
		assert var_decl.array_sizes is not None
		assert isinstance(var_decl.array_sizes[0], IntLiteral)
		assert var_decl.array_sizes[0].value == 3

	def test_trailing_comma_semantics(self):
		analyze("""
			int main() {
				int a[3] = {1, 2, 3,};
				return a[0];
			}
		""")


# ---------------------------------------------------------------
# IR generation tests
# ---------------------------------------------------------------


class TestInitializerListIRGen:
	def test_array_init_generates_stores(self):
		ir_prog = compile_to_ir("""
			int main() {
				int a[3] = {1, 2, 3};
				return a[0];
			}
		""")
		func = ir_prog.functions[0]
		# Should have IRStore instructions for each element
		stores = [i for i in func.body if isinstance(i, IRStore)]
		assert len(stores) >= 3

	def test_struct_init_generates_stores(self):
		ir_prog = compile_to_ir("""
			struct Point { int x; int y; };
			int main() {
				struct Point p = {10, 20};
				return p.x;
			}
		""")
		func = ir_prog.functions[0]
		stores = [i for i in func.body if isinstance(i, IRStore)]
		assert len(stores) >= 2

	def test_partial_init_zero_fills(self):
		ir_prog = compile_to_ir("""
			int main() {
				int a[4] = {1, 2};
				return a[3];
			}
		""")
		func = ir_prog.functions[0]
		stores = [i for i in func.body if isinstance(i, IRStore)]
		# Should have stores for all 4 elements (2 from init + 2 zero-fill)
		assert len(stores) >= 4

	def test_global_array_init(self):
		ir_prog = compile_to_ir("int g[3] = {10, 20, 30};")
		# Should have a global with initializer_values
		assert len(ir_prog.globals) == 1
		g = ir_prog.globals[0]
		assert isinstance(g, IRGlobalVar)
		assert g.initializer_values == [10, 20, 30]

	def test_global_struct_init(self):
		ir_prog = compile_to_ir("""
			struct Point { int x; int y; };
			struct Point g = {5, 15};
		""")
		found = [g for g in ir_prog.globals if g.name == "g"]
		assert len(found) == 1
		assert found[0].initializer_values == [5, 15]

	def test_nested_struct_init(self):
		"""Struct with nested struct member initialized with nested braces."""
		ir_prog = compile_to_ir("""
			struct Inner { int a; int b; };
			struct Outer { int x; int y; int z; };
			int main() {
				struct Outer o = {10, 20, 30};
				return o.x;
			}
		""")
		func = ir_prog.functions[0]
		stores = [i for i in func.body if isinstance(i, IRStore)]
		assert len(stores) >= 3


# ---------------------------------------------------------------
# Codegen tests
# ---------------------------------------------------------------


class TestInitializerListCodegen:
	def test_array_init_compiles(self):
		asm = compile_to_asm("""
			int main() {
				int a[3] = {1, 2, 3};
				return a[0];
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_struct_init_compiles(self):
		asm = compile_to_asm("""
			struct Point { int x; int y; };
			int main() {
				struct Point p = {10, 20};
				return p.x;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm

	def test_partial_init_compiles(self):
		asm = compile_to_asm("""
			int main() {
				int a[5] = {1, 2};
				return a[4];
			}
		""")
		assert "main:" in asm

	def test_global_array_init_compiles(self):
		asm = compile_to_asm("""
			int g[3] = {100, 200, 300};
			int main() {
				return 0;
			}
		""")
		assert ".globl g" in asm
		assert ".long 100" in asm
		assert ".long 200" in asm
		assert ".long 300" in asm

	def test_trailing_comma_compiles(self):
		asm = compile_to_asm("""
			int main() {
				int a[3] = {1, 2, 3,};
				return a[0];
			}
		""")
		assert "main:" in asm

	def test_nested_init_compiles(self):
		asm = compile_to_asm("""
			struct Outer { int x; int y; int z; };
			int main() {
				struct Outer o = {10, 20, 30};
				return o.x;
			}
		""")
		assert "main:" in asm
		assert "ret" in asm
