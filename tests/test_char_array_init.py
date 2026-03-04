"""Tests for char array initialization from string literals."""

from compiler.ast_nodes import IntLiteral, StringLiteral, VarDecl
from compiler.codegen import CodeGenerator
from compiler.ir import IRAlloc, IRGlobalVar, IRStore, IRType
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer


def parse(source: str):
	tokens = Lexer(source).tokenize()
	return Parser(tokens).parse()


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


class TestCharArrayParsing:
	def test_char_array_string_literal_init(self):
		prog = parse('int main() { char s[] = "hello"; return 0; }')
		func = prog.declarations[0]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl, VarDecl)
		assert var_decl.type_spec.base_type == "char"
		assert var_decl.array_sizes is not None
		assert isinstance(var_decl.initializer, StringLiteral)
		assert var_decl.initializer.value == "hello"

	def test_char_array_explicit_size(self):
		prog = parse('int main() { char s[10] = "hi"; return 0; }')
		func = prog.declarations[0]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl, VarDecl)
		assert var_decl.array_sizes is not None
		assert isinstance(var_decl.array_sizes[0], IntLiteral)
		assert var_decl.array_sizes[0].value == 10
		assert isinstance(var_decl.initializer, StringLiteral)


# ---------------------------------------------------------------
# Semantic analysis tests
# ---------------------------------------------------------------


class TestCharArraySemantics:
	def test_auto_sized_char_array(self):
		"""char s[] = "hello" should infer size 6 (5 chars + null)."""
		prog = parse('int main() { char s[] = "hello"; return 0; }')
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)
		func = prog.declarations[0]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl, VarDecl)
		assert var_decl.array_sizes is not None
		assert isinstance(var_decl.array_sizes[0], IntLiteral)
		assert var_decl.array_sizes[0].value == 6

	def test_explicit_size_char_array(self):
		"""char s[10] = "hi" should pass semantic analysis."""
		prog = parse('int main() { char s[10] = "hi"; return 0; }')
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)

	def test_exact_fit_char_array(self):
		"""char s[3] = "ab" should pass (2 chars + null = 3)."""
		prog = parse('int main() { char s[3] = "ab"; return 0; }')
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)

	def test_empty_string_char_array(self):
		"""char s[] = "" should infer size 1 (just null terminator)."""
		prog = parse('int main() { char s[] = ""; return 0; }')
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)
		func = prog.declarations[0]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl, VarDecl)
		assert var_decl.array_sizes[0].value == 1


# ---------------------------------------------------------------
# IR generation tests - local char arrays
# ---------------------------------------------------------------


class TestCharArrayIRGen:
	def test_auto_sized_generates_alloc_and_stores(self):
		"""char s[] = "hello" should alloc 6 bytes and emit 6 stores."""
		ir_prog = compile_to_ir('int main() { char s[] = "hello"; return 0; }')
		func = ir_prog.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		stores = [i for i in func.body if isinstance(i, IRStore)]
		assert any(a.size == 6 for a in allocs)
		assert len(stores) == 6
		# Verify store types are CHAR
		for s in stores:
			assert s.ir_type == IRType.CHAR

	def test_explicit_size_zero_padded(self):
		"""char s[10] = "hi" should alloc 10 bytes and emit 10 stores."""
		ir_prog = compile_to_ir('int main() { char s[10] = "hi"; return 0; }')
		func = ir_prog.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		stores = [i for i in func.body if isinstance(i, IRStore)]
		assert any(a.size == 10 for a in allocs)
		assert len(stores) == 10
		# First 3 stores: 'h', 'i', '\0'; remaining 7 stores: 0
		store_values = [s.value.value for s in stores]
		assert store_values[0] == ord("h")
		assert store_values[1] == ord("i")
		assert store_values[2] == 0  # null terminator
		for v in store_values[3:]:
			assert v == 0  # zero-padded

	def test_exact_fit(self):
		"""char s[3] = "ab" should alloc 3 bytes and emit 3 stores."""
		ir_prog = compile_to_ir('int main() { char s[3] = "ab"; return 0; }')
		func = ir_prog.functions[0]
		stores = [i for i in func.body if isinstance(i, IRStore)]
		assert len(stores) == 3
		store_values = [s.value.value for s in stores]
		assert store_values == [ord("a"), ord("b"), 0]

	def test_empty_string(self):
		"""char s[] = "" should alloc 1 byte and emit 1 store (null)."""
		ir_prog = compile_to_ir('int main() { char s[] = ""; return 0; }')
		func = ir_prog.functions[0]
		allocs = [i for i in func.body if isinstance(i, IRAlloc)]
		stores = [i for i in func.body if isinstance(i, IRStore)]
		assert any(a.size == 1 for a in allocs)
		assert len(stores) == 1
		assert stores[0].value.value == 0


# ---------------------------------------------------------------
# IR generation tests - global char arrays
# ---------------------------------------------------------------


class TestGlobalCharArrayIRGen:
	def test_global_auto_sized(self):
		"""Global char s[] = "hello" should produce IRGlobalVar with byte values."""
		ir_prog = compile_to_ir("""
			char s[] = "hello";
			int main() { return 0; }
		""")
		globals_named_s = [g for g in ir_prog.globals if g.name == "s"]
		assert len(globals_named_s) == 1
		g = globals_named_s[0]
		assert isinstance(g, IRGlobalVar)
		assert g.ir_type == IRType.CHAR
		assert g.total_size == 6
		expected = [ord("h"), ord("e"), ord("l"), ord("l"), ord("o"), 0]
		assert g.initializer_values == expected

	def test_global_explicit_size_zero_padded(self):
		"""Global char s[10] = "hi" should produce 10-byte IRGlobalVar."""
		ir_prog = compile_to_ir("""
			char s[10] = "hi";
			int main() { return 0; }
		""")
		globals_named_s = [g for g in ir_prog.globals if g.name == "s"]
		assert len(globals_named_s) == 1
		g = globals_named_s[0]
		assert g.ir_type == IRType.CHAR
		assert g.total_size == 10
		expected = [ord("h"), ord("i"), 0, 0, 0, 0, 0, 0, 0, 0]
		assert g.initializer_values == expected

	def test_global_exact_fit(self):
		"""Global char s[3] = "ab" should produce 3-byte IRGlobalVar."""
		ir_prog = compile_to_ir("""
			char s[3] = "ab";
			int main() { return 0; }
		""")
		globals_named_s = [g for g in ir_prog.globals if g.name == "s"]
		assert len(globals_named_s) == 1
		g = globals_named_s[0]
		assert g.initializer_values == [ord("a"), ord("b"), 0]

	def test_global_empty_string(self):
		"""Global char s[] = "" should produce 1-byte IRGlobalVar with just null."""
		ir_prog = compile_to_ir("""
			char s[] = "";
			int main() { return 0; }
		""")
		globals_named_s = [g for g in ir_prog.globals if g.name == "s"]
		assert len(globals_named_s) == 1
		g = globals_named_s[0]
		assert g.total_size == 1
		assert g.initializer_values == [0]


# ---------------------------------------------------------------
# Codegen tests
# ---------------------------------------------------------------


class TestCharArrayCodegen:
	def test_local_char_array_compiles(self):
		asm = compile_to_asm('int main() { char s[] = "hello"; return 0; }')
		assert "main:" in asm
		assert "ret" in asm

	def test_global_char_array_compiles(self):
		asm = compile_to_asm("""
			char g[] = "test";
			int main() { return 0; }
		""")
		assert ".globl g" in asm
		# Should emit .byte directives for each character
		assert ".byte" in asm

	def test_explicit_size_compiles(self):
		asm = compile_to_asm('int main() { char s[10] = "hi"; return 0; }')
		assert "main:" in asm
		assert "ret" in asm
