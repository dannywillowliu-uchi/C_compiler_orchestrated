"""Tests for cast expressions and global variable IR generation."""

from compiler.ast_nodes import CastExpr, VarDecl
from compiler.codegen import CodeGenerator
from compiler.ir import IRGlobalVar, IRType
from compiler.ir_gen import IRGenerator
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer


# ---------------------------------------------------------------------------
# Cast expression parsing
# ---------------------------------------------------------------------------


class TestCastParsing:
	def test_cast_int_to_char(self) -> None:
		src = "int main() { char c = (char)65; return 0; }"
		prog = Parser.from_source(src).parse()
		func = prog.declarations[0]
		body = func.body.statements
		var_decl = body[0]
		assert isinstance(var_decl, VarDecl)
		assert isinstance(var_decl.initializer, CastExpr)
		assert var_decl.initializer.target_type.base_type == "char"

	def test_cast_pointer_types(self) -> None:
		src = "int main() { int* p; char* q = (char*)p; return 0; }"
		prog = Parser.from_source(src).parse()
		func = prog.declarations[0]
		body = func.body.statements
		var_decl = body[1]
		assert isinstance(var_decl, VarDecl)
		assert isinstance(var_decl.initializer, CastExpr)
		assert var_decl.initializer.target_type.base_type == "char"
		assert var_decl.initializer.target_type.pointer_count == 1

	def test_cast_in_expression(self) -> None:
		src = "int main() { int x = (int)3 + 1; return x; }"
		prog = Parser.from_source(src).parse()
		func = prog.declarations[0]
		body = func.body.statements
		var_decl = body[0]
		assert isinstance(var_decl, VarDecl)
		# The initializer is a BinaryOp with a CastExpr on the left
		from compiler.ast_nodes import BinaryOp
		assert isinstance(var_decl.initializer, BinaryOp)
		assert isinstance(var_decl.initializer.left, CastExpr)

	def test_cast_void_pointer(self) -> None:
		src = "int main() { int x; void* p = (void*)&x; return 0; }"
		prog = Parser.from_source(src).parse()
		func = prog.declarations[0]
		body = func.body.statements
		var_decl = body[1]
		assert isinstance(var_decl, VarDecl)
		assert isinstance(var_decl.initializer, CastExpr)
		assert var_decl.initializer.target_type.base_type == "void"
		assert var_decl.initializer.target_type.pointer_count == 1


# ---------------------------------------------------------------------------
# Cast semantic analysis
# ---------------------------------------------------------------------------


class TestCastSemantic:
	def test_numeric_to_numeric_allowed(self) -> None:
		src = "int main() { int x = (int)'a'; return x; }"
		prog = Parser.from_source(src).parse()
		errors = SemanticAnalyzer().analyze(prog)
		assert len(errors) == 0

	def test_pointer_to_pointer_allowed(self) -> None:
		src = "int main() { int* p; char* q = (char*)p; return 0; }"
		prog = Parser.from_source(src).parse()
		errors = SemanticAnalyzer().analyze(prog)
		assert len(errors) == 0

	def test_numeric_to_pointer_allowed(self) -> None:
		src = "int main() { int* p = (int*)0; return 0; }"
		prog = Parser.from_source(src).parse()
		errors = SemanticAnalyzer().analyze(prog)
		assert len(errors) == 0


# ---------------------------------------------------------------------------
# Cast IR generation
# ---------------------------------------------------------------------------


class TestCastIRGen:
	def test_cast_generates_copy(self) -> None:
		src = "int main() { int x = (int)'a'; return x; }"
		prog = Parser.from_source(src).parse()
		SemanticAnalyzer().analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		assert len(ir_prog.functions) == 1
		# The IR should contain a copy for the cast
		from compiler.ir import IRCopy
		copies = [i for i in ir_prog.functions[0].body if isinstance(i, IRCopy)]
		assert len(copies) >= 1


# ---------------------------------------------------------------------------
# Global variable parsing
# ---------------------------------------------------------------------------


class TestGlobalVarParsing:
	def test_global_int_with_initializer(self) -> None:
		src = "int x = 42; int main() { return x; }"
		prog = Parser.from_source(src).parse()
		assert len(prog.declarations) == 2
		var_decl = prog.declarations[0]
		assert isinstance(var_decl, VarDecl)
		assert var_decl.name == "x"
		assert var_decl.type_spec.base_type == "int"

	def test_global_without_initializer(self) -> None:
		src = "int g; int main() { return 0; }"
		prog = Parser.from_source(src).parse()
		var_decl = prog.declarations[0]
		assert isinstance(var_decl, VarDecl)
		assert var_decl.name == "g"
		assert var_decl.initializer is None


# ---------------------------------------------------------------------------
# Global variable IR generation
# ---------------------------------------------------------------------------


class TestGlobalVarIR:
	def test_global_emits_ir_global_var(self) -> None:
		src = "int x = 42; int main() { return x; }"
		prog = Parser.from_source(src).parse()
		SemanticAnalyzer().analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		assert len(ir_prog.globals) == 1
		g = ir_prog.globals[0]
		assert isinstance(g, IRGlobalVar)
		assert g.name == "x"
		assert g.initializer == 42
		assert g.ir_type == IRType.INT

	def test_global_without_initializer_ir(self) -> None:
		src = "int g; int main() { return 0; }"
		prog = Parser.from_source(src).parse()
		SemanticAnalyzer().analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		assert len(ir_prog.globals) == 1
		g = ir_prog.globals[0]
		assert g.name == "g"
		assert g.initializer is None

	def test_global_read_from_function(self) -> None:
		src = "int x = 10; int main() { return x; }"
		prog = Parser.from_source(src).parse()
		SemanticAnalyzer().analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		# The function body should contain an IRLoad from the global
		from compiler.ir import IRLoad, IRGlobalRef
		loads = [
			i for i in ir_prog.functions[0].body
			if isinstance(i, IRLoad) and isinstance(i.address, IRGlobalRef)
		]
		assert len(loads) >= 1
		assert loads[0].address.name == "x"


# ---------------------------------------------------------------------------
# Global variable codegen
# ---------------------------------------------------------------------------


class TestGlobalVarCodegen:
	def test_initialized_global_in_data_section(self) -> None:
		src = "int x = 42; int main() { return x; }"
		prog = Parser.from_source(src).parse()
		SemanticAnalyzer().analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		asm = CodeGenerator().generate(ir_prog)
		assert ".section .data" in asm
		assert ".globl x" in asm
		assert ".quad 42" in asm

	def test_uninitialized_global_in_bss_section(self) -> None:
		src = "int g; int main() { return 0; }"
		prog = Parser.from_source(src).parse()
		SemanticAnalyzer().analyze(prog)
		ir_prog = IRGenerator().generate(prog)
		asm = CodeGenerator().generate(ir_prog)
		assert ".section .bss" in asm
		assert ".globl g" in asm
		assert ".zero 8" in asm
