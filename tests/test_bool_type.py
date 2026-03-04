"""Tests for _Bool type and stdbool.h support across the compiler pipeline."""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRBinOp,
	IRConst,
	IRFunction,
	IRGlobalVar,
	IRLoad,
	IRProgram,
	IRReturn,
	IRStore,
	IRTemp,
	IRType,
	ir_type_asm_suffix,
	ir_type_byte_width,
	ir_type_is_integer,
)
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.preprocessor import Preprocessor
from compiler.semantic import SemanticAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compile_to_ast(source: str):
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	return Parser(tokens).parse()


def _compile_to_ir(source: str) -> IRProgram:
	ast = _compile_to_ast(source)
	errors = SemanticAnalyzer().analyze(ast)
	assert not errors, f"Semantic errors: {errors}"
	return IRGenerator().generate(ast)


def _compile_to_asm(source: str) -> str:
	ir_program = _compile_to_ir(source)
	return CodeGenerator().generate(ir_program)


def _gen(func: IRFunction, globals_list: list[IRGlobalVar] | None = None) -> str:
	return CodeGenerator().generate(IRProgram([func], globals_list or [], []))


# ---------------------------------------------------------------------------
# Token / Lexer tests
# ---------------------------------------------------------------------------


class TestBoolLexer:
	def test_bool_keyword_tokenized(self) -> None:
		from compiler.tokens import TokenType
		tokens = Lexer("_Bool").tokenize()
		assert tokens[0].type == TokenType.BOOL
		assert tokens[0].value == "_Bool"

	def test_bool_in_declaration(self) -> None:
		from compiler.tokens import TokenType
		tokens = Lexer("_Bool x;").tokenize()
		assert tokens[0].type == TokenType.BOOL
		assert tokens[1].type == TokenType.IDENTIFIER
		assert tokens[1].value == "x"


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestBoolParser:
	def test_parse_bool_variable(self) -> None:
		ast = _compile_to_ast("int main(void) { _Bool b = 1; return b; }")
		func = ast.declarations[0]
		body = func.body.statements
		var_decl = body[0]
		assert var_decl.type_spec.base_type == "_Bool"

	def test_parse_bool_parameter(self) -> None:
		ast = _compile_to_ast("int f(_Bool b) { return b; }")
		func = ast.declarations[0]
		assert func.params[0].type_spec.base_type == "_Bool"

	def test_parse_bool_pointer(self) -> None:
		ast = _compile_to_ast("int main(void) { _Bool *p; return 0; }")
		func = ast.declarations[0]
		var_decl = func.body.statements[0]
		assert var_decl.type_spec.base_type == "_Bool"
		assert var_decl.type_spec.pointer_count == 1

	def test_parse_bool_cast(self) -> None:
		ast = _compile_to_ast("int main(void) { int x = 5; _Bool b = (_Bool)x; return b; }")
		assert ast is not None

	def test_parse_sizeof_bool(self) -> None:
		ast = _compile_to_ast("int main(void) { return sizeof(_Bool); }")
		assert ast is not None


# ---------------------------------------------------------------------------
# Semantic tests
# ---------------------------------------------------------------------------


class TestBoolSemantic:
	def test_bool_is_numeric(self) -> None:
		source = "int main(void) { _Bool b = 1; int x = b + 1; return x; }"
		ast = _compile_to_ast(source)
		errors = SemanticAnalyzer().analyze(ast)
		assert not errors

	def test_bool_in_condition(self) -> None:
		source = "int main(void) { _Bool b = 1; if (b) return 1; return 0; }"
		ast = _compile_to_ast(source)
		errors = SemanticAnalyzer().analyze(ast)
		assert not errors

	def test_bool_assignment_from_int(self) -> None:
		source = "int main(void) { _Bool b; b = 42; return b; }"
		ast = _compile_to_ast(source)
		errors = SemanticAnalyzer().analyze(ast)
		assert not errors

	def test_bool_comparison(self) -> None:
		source = "int main(void) { _Bool a = 1; _Bool b = 0; return a == b; }"
		ast = _compile_to_ast(source)
		errors = SemanticAnalyzer().analyze(ast)
		assert not errors


# ---------------------------------------------------------------------------
# IR type tests
# ---------------------------------------------------------------------------


class TestIRTypeBool:
	def test_bool_exists(self) -> None:
		assert IRType.BOOL is not None
		assert IRType.BOOL.name == "BOOL"

	def test_bool_byte_width(self) -> None:
		assert ir_type_byte_width(IRType.BOOL) == 1

	def test_bool_is_integer(self) -> None:
		assert ir_type_is_integer(IRType.BOOL) is True

	def test_bool_asm_suffix(self) -> None:
		assert ir_type_asm_suffix(IRType.BOOL) == "b"


# ---------------------------------------------------------------------------
# IR generation tests
# ---------------------------------------------------------------------------


class TestBoolIRGen:
	def test_bool_var_decl_generates_alloc(self) -> None:
		source = "int main(void) { _Bool b = 1; return b; }"
		ir = _compile_to_ir(source)
		main_func = ir.functions[0]
		# Should have an alloc of size 1
		from compiler.ir import IRAlloc
		allocs = [i for i in main_func.body if isinstance(i, IRAlloc)]
		assert any(a.size == 1 for a in allocs)

	def test_bool_normalization_nonzero_to_one(self) -> None:
		"""Assigning non-zero value to _Bool should produce a != 0 comparison."""
		source = "int main(void) { _Bool b = 42; return b; }"
		ir = _compile_to_ir(source)
		main_func = ir.functions[0]
		# Should contain a != 0 comparison for normalization
		binops = [i for i in main_func.body if isinstance(i, IRBinOp) and i.op == "!="]
		assert len(binops) >= 1

	def test_bool_normalization_on_assignment(self) -> None:
		"""Assigning to _Bool variable should normalize."""
		source = "int main(void) { _Bool b; b = 100; return b; }"
		ir = _compile_to_ir(source)
		main_func = ir.functions[0]
		binops = [i for i in main_func.body if isinstance(i, IRBinOp) and i.op == "!="]
		assert len(binops) >= 1

	def test_sizeof_bool_is_1(self) -> None:
		source = "int main(void) { return sizeof(_Bool); }"
		ir = _compile_to_ir(source)
		main_func = ir.functions[0]
		# The return should include IRConst(1)
		rets = [i for i in main_func.body if isinstance(i, IRReturn)]
		assert any(isinstance(r.value, IRConst) and r.value.value == 1 for r in rets)


# ---------------------------------------------------------------------------
# Codegen tests
# ---------------------------------------------------------------------------


class TestBoolCodegen:
	def test_load_bool_uses_movzbl(self) -> None:
		body = [
			IRLoad(IRTemp("t1"), IRTemp("t0"), ir_type=IRType.BOOL),
			IRReturn(IRTemp("t1")),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.POINTER])
		asm = _gen(func)
		assert "movzbl (%rax), %eax" in asm

	def test_store_bool_uses_movb(self) -> None:
		body = [
			IRStore(address=IRTemp("t0"), value=IRConst(1), ir_type=IRType.BOOL),
			IRReturn(IRConst(0)),
		]
		func = IRFunction("f", [IRTemp("t0")], body, IRType.INT, [IRType.POINTER])
		asm = _gen(func)
		assert "movb %cl, (%rax)" in asm

	def test_global_bool_bss_one_byte(self) -> None:
		glob = IRGlobalVar("flag", IRType.BOOL)
		func = IRFunction("f", [], [IRReturn(IRConst(0))], IRType.INT)
		asm = _gen(func, [glob])
		assert ".zero 1" in asm

	def test_global_bool_initialized(self) -> None:
		glob = IRGlobalVar("flag", IRType.BOOL, initializer=1)
		func = IRFunction("f", [], [IRReturn(IRConst(0))], IRType.INT)
		asm = _gen(func, [glob])
		assert ".byte 1" in asm


# ---------------------------------------------------------------------------
# Preprocessor / stdbool.h tests
# ---------------------------------------------------------------------------


class TestStdboolH:
	def test_stdbool_defines_bool(self) -> None:
		pp = Preprocessor()
		result = pp.process('#include <stdbool.h>\nbool x;')
		assert "_Bool" in result

	def test_stdbool_defines_true(self) -> None:
		pp = Preprocessor()
		result = pp.process('#include <stdbool.h>\nint x = true;')
		assert "1" in result

	def test_stdbool_defines_false(self) -> None:
		pp = Preprocessor()
		result = pp.process('#include <stdbool.h>\nint x = false;')
		assert "0" in result

	def test_stdbool_bool_true_false_defined(self) -> None:
		pp = Preprocessor()
		result = pp.process('#include <stdbool.h>\nint x = __bool_true_false_are_defined;')
		assert "1" in result

	def test_stdbool_quoted_include(self) -> None:
		pp = Preprocessor()
		result = pp.process('#include "stdbool.h"\nbool y;')
		assert "_Bool" in result

	def test_stdbool_include_guard(self) -> None:
		"""Including stdbool.h twice should not cause errors."""
		pp = Preprocessor()
		source = '#include <stdbool.h>\n#include <stdbool.h>\nbool x;'
		result = pp.process(source)
		assert "_Bool" in result


# ---------------------------------------------------------------------------
# Pipeline integration tests (lexer -> parser -> semantic -> IR -> asm)
# ---------------------------------------------------------------------------


class TestBoolPipeline:
	def test_bool_variable_declaration_pipeline(self) -> None:
		source = "int main(void) { _Bool b = 1; return b; }"
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_with_stdbool_pipeline(self) -> None:
		source = '#include <stdbool.h>\nint main(void) { bool b = true; return b; }'
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_false_with_stdbool(self) -> None:
		source = '#include <stdbool.h>\nint main(void) { bool b = false; return b; }'
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_normalization_in_pipeline(self) -> None:
		"""Non-zero values should be normalized to 1 when stored to _Bool."""
		source = "int main(void) { _Bool b = 42; return b; }"
		ir = _compile_to_ir(source)
		main_func = ir.functions[0]
		# Should contain normalization (!=0 comparison)
		binops = [i for i in main_func.body if isinstance(i, IRBinOp) and i.op == "!="]
		assert len(binops) >= 1

	def test_bool_in_if_condition(self) -> None:
		source = """
		int main(void) {
			_Bool flag = 1;
			if (flag) return 10;
			return 20;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_arithmetic(self) -> None:
		source = """
		int main(void) {
			_Bool a = 1;
			_Bool b = 0;
			return a + b;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_global_variable(self) -> None:
		source = """
		_Bool flag;
		int main(void) {
			flag = 1;
			return flag;
		}
		"""
		asm = _compile_to_asm(source)
		assert "flag:" in asm

	def test_bool_as_function_parameter(self) -> None:
		source = """
		int check(_Bool b) {
			if (b) return 1;
			return 0;
		}
		int main(void) {
			return check(1);
		}
		"""
		asm = _compile_to_asm(source)
		assert "check:" in asm

	def test_bool_sizeof(self) -> None:
		source = "int main(void) { return sizeof(_Bool); }"
		ir = _compile_to_ir(source)
		main_func = ir.functions[0]
		rets = [i for i in main_func.body if isinstance(i, IRReturn)]
		assert any(isinstance(r.value, IRConst) and r.value.value == 1 for r in rets)

	def test_bool_pointer(self) -> None:
		source = """
		int main(void) {
			_Bool b = 1;
			_Bool *p = &b;
			return *p;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_stdbool_full_program(self) -> None:
		source = """
		#include <stdbool.h>
		bool is_positive(int x) {
			return x > 0;
		}
		int main(void) {
			bool result = is_positive(42);
			if (result) return 1;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "is_positive:" in asm
		assert "main:" in asm

	def test_bool_negation(self) -> None:
		source = """
		int main(void) {
			_Bool b = 1;
			_Bool nb = !b;
			return nb;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_logical_ops(self) -> None:
		source = """
		int main(void) {
			_Bool a = 1;
			_Bool b = 0;
			_Bool c = a && b;
			_Bool d = a || b;
			return c + d;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_comparison_result(self) -> None:
		source = """
		int main(void) {
			int x = 5;
			_Bool lt = x < 10;
			_Bool gt = x > 10;
			return lt + gt;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm
