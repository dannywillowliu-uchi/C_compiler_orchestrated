"""Tests for variadic function support (va_start, va_arg, va_end).

Tests cover parsing, semantic analysis, IR generation, and codegen
for variadic functions using the stdarg.h builtin header.
"""

from compiler.codegen import CodeGenerator
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.preprocessor import Preprocessor
from compiler.semantic import SemanticAnalyzer
from compiler.ast_nodes import (
	VaStartExpr,
	VaArgExpr,
	VaEndExpr,
)
from compiler.ir import IRVaStart, IRVaArg, IRVaEnd


def compile_source(source: str) -> str:
	"""Run C source through the full compiler pipeline, returning assembly."""
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	ast = Parser(tokens).parse()
	errors = SemanticAnalyzer().analyze(ast)
	assert not errors, f"Semantic errors: {errors}"
	ir = IRGenerator().generate(ast)
	return CodeGenerator().generate(ir)


def parse_source(source: str):
	"""Parse C source and return the AST."""
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	return Parser(tokens).parse()


def generate_ir(source: str):
	"""Parse and generate IR from C source."""
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	ast = Parser(tokens).parse()
	errors = SemanticAnalyzer().analyze(ast)
	assert not errors, f"Semantic errors: {errors}"
	return IRGenerator().generate(ast)


# -- Preprocessor tests --


class TestStdargPreprocessor:
	"""Test that stdarg.h is properly handled by the preprocessor."""

	def test_stdarg_include(self) -> None:
		source = '#include <stdarg.h>\nint main() { return 0; }'
		preprocessed = Preprocessor().process(source)
		assert "va_list" in preprocessed

	def test_va_list_typedef(self) -> None:
		source = '#include <stdarg.h>\nint main() { return 0; }'
		preprocessed = Preprocessor().process(source)
		assert "typedef" in preprocessed
		assert "va_list" in preprocessed


# -- Parser tests --


class TestVariadicParsing:
	"""Test parsing of variadic function declarations and va_* builtins."""

	def test_parse_variadic_function_decl(self) -> None:
		source = "int sum(int count, ...) { return 0; }"
		ast = parse_source(source)
		func = ast.declarations[0]
		assert func.is_variadic is True
		assert len(func.params) == 1

	def test_parse_va_start(self) -> None:
		source = '#include <stdarg.h>\nvoid f(int n, ...) { va_list ap; va_start(ap, n); va_end(ap); }'
		ast = parse_source(source)
		func = [d for d in ast.declarations if hasattr(d, "body") and d.body is not None][0]
		body = func.body.statements
		found_va_start = False
		for stmt in body:
			if hasattr(stmt, "expression") and isinstance(stmt.expression, VaStartExpr):
				found_va_start = True
				assert stmt.expression.last_param == "n"
		assert found_va_start, "VaStartExpr not found in parsed AST"

	def test_parse_va_arg(self) -> None:
		source = '#include <stdarg.h>\nint f(int n, ...) { va_list ap; va_start(ap, n); int x = va_arg(ap, int); va_end(ap); return x; }'
		ast = parse_source(source)
		func = [d for d in ast.declarations if hasattr(d, "body") and d.body is not None][0]
		body = func.body.statements
		found_va_arg = False
		for stmt in body:
			if hasattr(stmt, "initializer") and isinstance(stmt.initializer, VaArgExpr):
				found_va_arg = True
				assert stmt.initializer.arg_type.base_type == "int"
		assert found_va_arg, "VaArgExpr not found in parsed AST"

	def test_parse_va_end(self) -> None:
		source = '#include <stdarg.h>\nvoid f(int n, ...) { va_list ap; va_start(ap, n); va_end(ap); }'
		ast = parse_source(source)
		func = [d for d in ast.declarations if hasattr(d, "body") and d.body is not None][0]
		body = func.body.statements
		found_va_end = False
		for stmt in body:
			if hasattr(stmt, "expression") and isinstance(stmt.expression, VaEndExpr):
				found_va_end = True
		assert found_va_end, "VaEndExpr not found in parsed AST"


# -- Semantic analysis tests --


class TestVariadicSemantic:
	"""Test semantic analysis of variadic functions."""

	def test_semantic_passes_for_variadic(self) -> None:
		source = '#include <stdarg.h>\nint sum(int n, ...) { va_list ap; va_start(ap, n); int x = va_arg(ap, int); va_end(ap); return x; }'
		preprocessed = Preprocessor().process(source)
		tokens = Lexer(preprocessed).tokenize()
		ast = Parser(tokens).parse()
		errors = SemanticAnalyzer().analyze(ast)
		assert not errors

	def test_variadic_call_min_args(self) -> None:
		source = '#include <stdarg.h>\nint sum(int n, ...) { va_list ap; va_start(ap, n); va_end(ap); return 0; }\nint main() { return sum(3, 1, 2, 3); }'
		preprocessed = Preprocessor().process(source)
		tokens = Lexer(preprocessed).tokenize()
		ast = Parser(tokens).parse()
		errors = SemanticAnalyzer().analyze(ast)
		assert not errors


# -- IR generation tests --


class TestVariadicIR:
	"""Test IR generation for variadic functions."""

	def test_ir_variadic_function_flag(self) -> None:
		source = '#include <stdarg.h>\nint sum(int n, ...) { va_list ap; va_start(ap, n); va_end(ap); return 0; }'
		ir = generate_ir(source)
		func = [f for f in ir.functions if f.name == "sum"][0]
		assert func.is_variadic is True

	def test_ir_contains_va_start(self) -> None:
		source = '#include <stdarg.h>\nint sum(int n, ...) { va_list ap; va_start(ap, n); va_end(ap); return 0; }'
		ir = generate_ir(source)
		func = [f for f in ir.functions if f.name == "sum"][0]
		va_starts = [i for i in func.body if isinstance(i, IRVaStart)]
		assert len(va_starts) == 1

	def test_ir_contains_va_arg(self) -> None:
		source = '#include <stdarg.h>\nint sum(int n, ...) { va_list ap; va_start(ap, n); int x = va_arg(ap, int); va_end(ap); return x; }'
		ir = generate_ir(source)
		func = [f for f in ir.functions if f.name == "sum"][0]
		va_args = [i for i in func.body if isinstance(i, IRVaArg)]
		assert len(va_args) == 1

	def test_ir_contains_va_end(self) -> None:
		source = '#include <stdarg.h>\nint sum(int n, ...) { va_list ap; va_start(ap, n); va_end(ap); return 0; }'
		ir = generate_ir(source)
		func = [f for f in ir.functions if f.name == "sum"][0]
		va_ends = [i for i in func.body if isinstance(i, IRVaEnd)]
		assert len(va_ends) == 1


# -- Codegen tests --


class TestVariadicCodegen:
	"""Test x86-64 code generation for variadic functions."""

	def test_variadic_function_compiles(self) -> None:
		source = """
		#include <stdarg.h>
		int sum(int count, ...) {
			va_list ap;
			va_start(ap, count);
			int total = 0;
			int i = 0;
			while (i < count) {
				total = total + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);
			return total;
		}
		"""
		asm = compile_source(source)
		assert "sum:" in asm
		assert ".globl sum" in asm

	def test_variadic_saves_registers(self) -> None:
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			va_end(ap);
			return 0;
		}
		"""
		asm = compile_source(source)
		# Variadic function should save GP registers to register save area
		assert "movq %rdi" in asm
		assert "movq %rsi" in asm

	def test_va_start_initializes_struct(self) -> None:
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			va_end(ap);
			return 0;
		}
		"""
		asm = compile_source(source)
		# va_start should set gp_offset (1 named GP param = 8)
		assert "movl $8" in asm
		# va_start should set fp_offset = 48
		assert "movl $48" in asm
		# va_start should set overflow_arg_area
		assert "16(%rbp)" in asm

	def test_va_arg_generates_branch(self) -> None:
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int x = va_arg(ap, int);
			va_end(ap);
			return x;
		}
		"""
		asm = compile_source(source)
		# va_arg should generate a comparison against 48 and a branch
		assert "cmpl $48" in asm
		assert "jae" in asm

	def test_va_arg_with_long_type(self) -> None:
		source = """
		#include <stdarg.h>
		long f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			long x = va_arg(ap, long);
			va_end(ap);
			return x;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "cmpl $48" in asm

	def test_multiple_va_arg_calls(self) -> None:
		source = """
		#include <stdarg.h>
		int sum3(int n, ...) {
			va_list ap;
			va_start(ap, n);
			int a = va_arg(ap, int);
			int b = va_arg(ap, int);
			int c = va_arg(ap, int);
			va_end(ap);
			return a + b + c;
		}
		"""
		asm = compile_source(source)
		assert "sum3:" in asm
		# Should have 3 va_arg branches
		assert asm.count("cmpl $48") == 3

	def test_variadic_with_caller(self) -> None:
		source = """
		#include <stdarg.h>
		int sum(int count, ...) {
			va_list ap;
			va_start(ap, count);
			int total = 0;
			int i = 0;
			while (i < count) {
				total = total + va_arg(ap, int);
				i = i + 1;
			}
			va_end(ap);
			return total;
		}
		int main() {
			return sum(3, 10, 20, 30);
		}
		"""
		asm = compile_source(source)
		assert "sum:" in asm
		assert "main:" in asm
		assert "call sum" in asm

	def test_variadic_pointer_arg(self) -> None:
		source = """
		#include <stdarg.h>
		void *get_ptr(int n, ...) {
			va_list ap;
			va_start(ap, n);
			void *p = va_arg(ap, void *);
			va_end(ap);
			return p;
		}
		"""
		asm = compile_source(source)
		assert "get_ptr:" in asm
		assert "cmpl $48" in asm

	def test_variadic_no_extra_args(self) -> None:
		source = """
		#include <stdarg.h>
		int f(int n, ...) {
			va_list ap;
			va_start(ap, n);
			va_end(ap);
			return n;
		}
		"""
		asm = compile_source(source)
		assert "f:" in asm
		assert "ret" in asm
