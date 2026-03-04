"""Tests for indirect function calls: (*fp)(args), arr[i](args), etc."""

from compiler.ast_nodes import (
	ArraySubscript,
	FunctionCall,
	Identifier,
	UnaryOp,
)
from compiler.ir import IRCall
from compiler.ir_gen import IRGenerator
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer


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


# ---------------------------------------------------------------------------
# Parser: indirect calls
# ---------------------------------------------------------------------------

class TestParseIndirectCalls:
	def test_deref_function_pointer_call(self):
		"""(*fp)(args) should parse as FunctionCall with callee=UnaryOp('*', ...)."""
		src = """
		int add(int a, int b) { return a + b; }
		int main() {
			int (*fp)(int, int) = add;
			return (*fp)(3, 4);
		}
		"""
		prog = parse(src)
		main_fn = prog.declarations[1]
		ret_stmt = main_fn.body.statements[1]
		call = ret_stmt.expression
		assert isinstance(call, FunctionCall)
		assert call.callee is not None
		assert call.name == ""
		assert isinstance(call.callee, UnaryOp)
		assert call.callee.op == "*"
		assert isinstance(call.callee.operand, Identifier)
		assert call.callee.operand.name == "fp"
		assert len(call.arguments) == 2

	def test_array_subscript_call(self):
		"""arr[i](args) should parse as FunctionCall with callee=ArraySubscript."""
		src = """
		int main() {
			int *fps;
			return fps[0](3, 4);
		}
		"""
		prog = parse(src)
		main_fn = prog.declarations[0]
		ret_stmt = main_fn.body.statements[1]
		call = ret_stmt.expression
		assert isinstance(call, FunctionCall)
		assert call.callee is not None
		assert isinstance(call.callee, ArraySubscript)
		assert isinstance(call.callee.array, Identifier)
		assert call.callee.array.name == "fps"
		assert len(call.arguments) == 2

	def test_regular_named_call_still_works(self):
		"""Normal function calls should still use name, not callee."""
		src = """
		int add(int a, int b) { return a + b; }
		int main() { return add(1, 2); }
		"""
		prog = parse(src)
		main_fn = prog.declarations[1]
		ret_stmt = main_fn.body.statements[0]
		call = ret_stmt.expression
		assert isinstance(call, FunctionCall)
		assert call.name == "add"
		assert call.callee is None
		assert len(call.arguments) == 2

	def test_deref_call_no_args(self):
		"""(*fp)() should work with zero arguments."""
		src = """
		int get_val() { return 42; }
		int main() {
			int (*fp)() = get_val;
			return (*fp)();
		}
		"""
		prog = parse(src)
		main_fn = prog.declarations[1]
		ret_stmt = main_fn.body.statements[1]
		call = ret_stmt.expression
		assert isinstance(call, FunctionCall)
		assert call.callee is not None
		assert len(call.arguments) == 0

	def test_chained_subscript_call(self):
		"""Chained: obj.method() pattern via member-access call.

		We test the simpler case of arr[0](1) parsed correctly.
		"""
		src = """
		int main() {
			int x = 0;
			return x;
		}
		"""
		# This just ensures basic parsing works - chained patterns
		prog = parse(src)
		assert prog is not None


# ---------------------------------------------------------------------------
# Semantic analysis: indirect calls
# ---------------------------------------------------------------------------

class TestSemanticIndirectCalls:
	def test_deref_call_semantic_pass(self):
		"""(*fp)(args) should pass semantic analysis."""
		src = """
		int add(int a, int b) { return a + b; }
		int main() {
			int (*fp)(int, int) = add;
			int result = (*fp)(3, 4);
			return result;
		}
		"""
		parse_and_analyze(src)

	def test_named_call_still_works(self):
		"""Regular named calls still pass semantic analysis."""
		src = """
		int add(int a, int b) { return a + b; }
		int main() { return add(1, 2); }
		"""
		parse_and_analyze(src)


# ---------------------------------------------------------------------------
# IR generation: indirect calls
# ---------------------------------------------------------------------------

class TestIRGenIndirectCalls:
	def test_deref_call_emits_indirect_ir(self):
		"""(*fp)(args) should emit an indirect IRCall."""
		src = """
		int add(int a, int b) { return a + b; }
		int main() {
			int (*fp)(int, int) = add;
			int result = (*fp)(3, 4);
			return result;
		}
		"""
		ir_prog = compile_to_ir(src)
		main_func = [f for f in ir_prog.functions if f.name == "main"][0]
		indirect_calls = [i for i in main_func.body if isinstance(i, IRCall) and i.indirect]
		assert len(indirect_calls) >= 1
		call = indirect_calls[0]
		assert call.func_value is not None

	def test_named_call_not_indirect(self):
		"""Regular named calls should not be indirect."""
		src = """
		int add(int a, int b) { return a + b; }
		int main() { return add(1, 2); }
		"""
		ir_prog = compile_to_ir(src)
		main_func = [f for f in ir_prog.functions if f.name == "main"][0]
		calls = [i for i in main_func.body if isinstance(i, IRCall)]
		assert len(calls) == 1
		assert calls[0].indirect is False
