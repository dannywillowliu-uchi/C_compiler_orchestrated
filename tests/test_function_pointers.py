"""Tests for function pointer declarations and indirect calls across the compiler pipeline."""

import pytest

from compiler.ast_nodes import TypedefDecl, VarDecl
from compiler.codegen import CodeGenerator
from compiler.ir import IRCall
from compiler.ir_gen import IRGenerator
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer, SemanticError


def parse(source: str):
	return Parser.from_source(source).parse()


def parse_and_analyze(source: str):
	program = parse(source)
	analyzer = SemanticAnalyzer()
	analyzer.analyze(program)
	return program, analyzer


def compile_to_ir(source: str):
	program = parse(source)
	analyzer = SemanticAnalyzer()
	analyzer.analyze(program)
	ir_gen = IRGenerator()
	return ir_gen.generate(program)


def compile_to_asm(source: str) -> str:
	ir_prog = compile_to_ir(source)
	codegen = CodeGenerator()
	return codegen.generate(ir_prog)


# ---------------------------------------------------------------------------
# Parsing: function pointer declarations
# ---------------------------------------------------------------------------

class TestParsingFuncPtrDecl:
	def test_parse_local_func_ptr(self):
		src = """
		int add(int a, int b) { return a + b; }
		int main() {
			int (*fp)(int, int) = add;
			return 0;
		}
		"""
		prog = parse(src)
		# main function should have a VarDecl for fp
		main_fn = prog.declarations[1]
		assert main_fn.name == "main"
		fp_decl = main_fn.body.statements[0]
		assert isinstance(fp_decl, VarDecl)
		assert fp_decl.name == "fp"
		assert fp_decl.type_spec.is_function_pointer is True
		assert fp_decl.type_spec.func_ptr_return_type.base_type == "int"
		assert len(fp_decl.type_spec.func_ptr_params) == 2

	def test_parse_global_func_ptr(self):
		src = """
		int add(int a, int b) { return a + b; }
		int (*gfp)(int, int);
		int main() { return 0; }
		"""
		prog = parse(src)
		gfp_decl = prog.declarations[1]
		assert isinstance(gfp_decl, VarDecl)
		assert gfp_decl.name == "gfp"
		assert gfp_decl.type_spec.is_function_pointer is True

	def test_parse_func_ptr_no_params(self):
		src = """
		int get_val() { return 42; }
		int main() {
			int (*fp)() = get_val;
			return 0;
		}
		"""
		prog = parse(src)
		main_fn = prog.declarations[1]
		fp_decl = main_fn.body.statements[0]
		assert fp_decl.type_spec.is_function_pointer is True
		assert len(fp_decl.type_spec.func_ptr_params) == 0

	def test_parse_func_ptr_void_return(self):
		src = """
		void do_nothing() { return; }
		int main() {
			void (*fp)() = do_nothing;
			return 0;
		}
		"""
		prog = parse(src)
		main_fn = prog.declarations[1]
		fp_decl = main_fn.body.statements[0]
		assert fp_decl.type_spec.func_ptr_return_type.base_type == "void"

	def test_parse_func_ptr_param(self):
		"""Function pointer as a function parameter."""
		src = """
		int apply(int (*op)(int, int), int a, int b) {
			return op(a, b);
		}
		int add(int a, int b) { return a + b; }
		int main() { return apply(add, 3, 4); }
		"""
		prog = parse(src)
		apply_fn = prog.declarations[0]
		assert apply_fn.name == "apply"
		first_param = apply_fn.params[0]
		assert first_param.name == "op"
		assert first_param.type_spec.is_function_pointer is True


# ---------------------------------------------------------------------------
# Parsing: typedef function pointers
# ---------------------------------------------------------------------------

class TestParsingTypedefFuncPtr:
	def test_typedef_func_ptr(self):
		src = """
		typedef int (*BinOp)(int, int);
		int add(int a, int b) { return a + b; }
		int main() {
			BinOp fp = add;
			return 0;
		}
		"""
		prog = parse(src)
		td = prog.declarations[0]
		assert isinstance(td, TypedefDecl)
		assert td.name == "BinOp"
		assert td.type_spec.is_function_pointer is True
		assert td.type_spec.func_ptr_return_type.base_type == "int"
		assert len(td.type_spec.func_ptr_params) == 2

	def test_typedef_func_ptr_use_as_var(self):
		src = """
		typedef int (*BinOp)(int, int);
		int add(int a, int b) { return a + b; }
		int main() {
			BinOp fp = add;
			return fp(1, 2);
		}
		"""
		prog = parse(src)
		# Should parse without errors
		assert len(prog.declarations) == 3


# ---------------------------------------------------------------------------
# Semantic analysis
# ---------------------------------------------------------------------------

class TestSemanticFuncPtr:
	def test_func_ptr_init_with_func_name(self):
		src = """
		int add(int a, int b) { return a + b; }
		int main() {
			int (*fp)(int, int) = add;
			return 0;
		}
		"""
		parse_and_analyze(src)

	def test_func_ptr_init_with_address_of(self):
		src = """
		int add(int a, int b) { return a + b; }
		int main() {
			int (*fp)(int, int) = &add;
			return 0;
		}
		"""
		parse_and_analyze(src)

	def test_func_ptr_assignment(self):
		src = """
		int add(int a, int b) { return a + b; }
		int sub(int a, int b) { return a - b; }
		int main() {
			int (*fp)(int, int) = add;
			fp = sub;
			return 0;
		}
		"""
		parse_and_analyze(src)

	def test_func_ptr_indirect_call(self):
		src = """
		int add(int a, int b) { return a + b; }
		int main() {
			int (*fp)(int, int) = add;
			int result = fp(3, 4);
			return result;
		}
		"""
		parse_and_analyze(src)

	def test_func_ptr_wrong_arg_count(self):
		src = """
		int add(int a, int b) { return a + b; }
		int main() {
			int (*fp)(int, int) = add;
			int result = fp(3);
			return result;
		}
		"""
		with pytest.raises(SemanticError, match="expects 2 arguments, got 1"):
			parse_and_analyze(src)

	def test_func_ptr_as_parameter(self):
		src = """
		int apply(int (*op)(int, int), int a, int b) {
			return op(a, b);
		}
		int add(int a, int b) { return a + b; }
		int main() {
			return apply(add, 3, 4);
		}
		"""
		parse_and_analyze(src)

	def test_typedef_func_ptr_semantic(self):
		src = """
		typedef int (*BinOp)(int, int);
		int add(int a, int b) { return a + b; }
		int main() {
			BinOp fp = add;
			int result = fp(10, 20);
			return result;
		}
		"""
		parse_and_analyze(src)

	def test_func_ptr_init_undeclared(self):
		src = """
		int main() {
			int (*fp)(int, int) = nonexistent;
			return 0;
		}
		"""
		with pytest.raises(SemanticError, match="undeclared"):
			parse_and_analyze(src)

	def test_func_ptr_init_non_function(self):
		src = """
		int main() {
			int x = 5;
			int (*fp)(int, int) = x;
			return 0;
		}
		"""
		with pytest.raises(SemanticError, match="not a function"):
			parse_and_analyze(src)


# ---------------------------------------------------------------------------
# IR generation
# ---------------------------------------------------------------------------

class TestIRGenFuncPtr:
	def test_indirect_call_ir(self):
		src = """
		int add(int a, int b) { return a + b; }
		int main() {
			int (*fp)(int, int) = add;
			int result = fp(3, 4);
			return result;
		}
		"""
		ir_prog = compile_to_ir(src)
		main_func = [f for f in ir_prog.functions if f.name == "main"][0]
		# Find the indirect IRCall
		indirect_calls = [i for i in main_func.body if isinstance(i, IRCall) and i.indirect]
		assert len(indirect_calls) == 1
		call = indirect_calls[0]
		assert call.indirect is True
		assert call.func_value is not None

	def test_direct_call_not_indirect(self):
		src = """
		int add(int a, int b) { return a + b; }
		int main() { return add(1, 2); }
		"""
		ir_prog = compile_to_ir(src)
		main_func = [f for f in ir_prog.functions if f.name == "main"][0]
		calls = [i for i in main_func.body if isinstance(i, IRCall)]
		assert len(calls) == 1
		assert calls[0].indirect is False

	def test_func_ptr_reassignment_ir(self):
		src = """
		int add(int a, int b) { return a + b; }
		int sub(int a, int b) { return a - b; }
		int main() {
			int (*fp)(int, int) = add;
			fp = sub;
			return fp(10, 5);
		}
		"""
		ir_prog = compile_to_ir(src)
		main_func = [f for f in ir_prog.functions if f.name == "main"][0]
		indirect_calls = [i for i in main_func.body if isinstance(i, IRCall) and i.indirect]
		assert len(indirect_calls) == 1

	def test_func_ptr_param_ir(self):
		src = """
		int apply(int (*op)(int, int), int a, int b) {
			return op(a, b);
		}
		int add(int a, int b) { return a + b; }
		int main() { return apply(add, 3, 4); }
		"""
		ir_prog = compile_to_ir(src)
		apply_func = [f for f in ir_prog.functions if f.name == "apply"][0]
		indirect_calls = [i for i in apply_func.body if isinstance(i, IRCall) and i.indirect]
		assert len(indirect_calls) == 1


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

class TestCodegenFuncPtr:
	def test_indirect_call_asm(self):
		src = """
		int add(int a, int b) { return a + b; }
		int main() {
			int (*fp)(int, int) = add;
			int result = fp(3, 4);
			return result;
		}
		"""
		asm = compile_to_asm(src)
		assert "call *%r11" in asm

	def test_direct_call_no_indirect(self):
		src = """
		int add(int a, int b) { return a + b; }
		int main() { return add(1, 2); }
		"""
		asm = compile_to_asm(src)
		assert "call add" in asm

	def test_func_ptr_param_codegen(self):
		src = """
		int apply(int (*op)(int, int), int a, int b) {
			return op(a, b);
		}
		int add(int a, int b) { return a + b; }
		int main() { return apply(add, 3, 4); }
		"""
		asm = compile_to_asm(src)
		assert "call *%r11" in asm

	def test_address_of_func_in_asm(self):
		src = """
		int add(int a, int b) { return a + b; }
		int main() {
			int (*fp)(int, int) = &add;
			return fp(1, 2);
		}
		"""
		asm = compile_to_asm(src)
		assert "call *%r11" in asm
		assert "leaq add(%rip)" in asm

	def test_reassign_func_ptr_codegen(self):
		src = """
		int add(int a, int b) { return a + b; }
		int sub(int a, int b) { return a - b; }
		int main() {
			int (*fp)(int, int) = add;
			fp = sub;
			return fp(10, 5);
		}
		"""
		asm = compile_to_asm(src)
		assert "call *%r11" in asm
		# Should have leaq for both add and sub
		assert "leaq add(%rip)" in asm
		assert "leaq sub(%rip)" in asm

	def test_typedef_func_ptr_codegen(self):
		src = """
		typedef int (*BinOp)(int, int);
		int add(int a, int b) { return a + b; }
		int main() {
			BinOp fp = add;
			return fp(5, 3);
		}
		"""
		asm = compile_to_asm(src)
		assert "call *%r11" in asm


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

class TestFullPipeline:
	def test_basic_func_ptr_pipeline(self):
		"""Basic function pointer through entire pipeline."""
		src = """
		int add(int a, int b) { return a + b; }
		int main() {
			int (*fp)(int, int) = add;
			return fp(3, 4);
		}
		"""
		asm = compile_to_asm(src)
		assert ".globl main" in asm
		assert ".globl add" in asm
		assert "call *%r11" in asm

	def test_func_ptr_with_multiple_calls(self):
		src = """
		int add(int a, int b) { return a + b; }
		int mul(int a, int b) { return a * b; }
		int main() {
			int (*fp)(int, int) = add;
			int r1 = fp(2, 3);
			fp = mul;
			int r2 = fp(4, 5);
			return r1 + r2;
		}
		"""
		asm = compile_to_asm(src)
		assert asm.count("call *%r11") == 2

	def test_func_ptr_passed_as_arg(self):
		src = """
		int apply(int (*f)(int, int), int x, int y) {
			return f(x, y);
		}
		int add(int a, int b) { return a + b; }
		int main() {
			return apply(add, 10, 20);
		}
		"""
		asm = compile_to_asm(src)
		assert "call *%r11" in asm
		assert "call apply" in asm

	def test_func_ptr_void_return(self):
		src = """
		int g;
		void set_g(int val) { g = val; }
		int main() {
			void (*fp)(int) = set_g;
			fp(42);
			return g;
		}
		"""
		asm = compile_to_asm(src)
		assert "call *%r11" in asm

	def test_func_ptr_typedef_full(self):
		src = """
		typedef int (*MathOp)(int, int);
		int add(int a, int b) { return a + b; }
		int sub(int a, int b) { return a - b; }
		int compute(MathOp op, int x, int y) {
			return op(x, y);
		}
		int main() {
			MathOp op = add;
			int r1 = compute(op, 5, 3);
			op = sub;
			int r2 = compute(op, 10, 4);
			return r1 + r2;
		}
		"""
		asm = compile_to_asm(src)
		assert "call *%r11" in asm
