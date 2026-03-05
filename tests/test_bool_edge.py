"""Edge-case tests for _Bool: truncation semantics, conditions, arithmetic, stdbool.h macros."""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRBinOp,
	IRConst,
	IRReturn,
	IRType,
	ir_type_byte_width,
)
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.preprocessor import Preprocessor
from compiler.semantic import SemanticAnalyzer


def _parse(source: str):
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	return Parser(tokens).parse()


def _compile_to_ir(source: str):
	ast = _parse(source)
	errors = SemanticAnalyzer().analyze(ast)
	assert not errors, f"Semantic errors: {errors}"
	return IRGenerator().generate(ast)


def _compile_to_asm(source: str) -> str:
	ir = _compile_to_ir(source)
	return CodeGenerator().generate(ir)


# ---------------------------------------------------------------------------
# _Bool truncation semantics (any non-zero -> 1)
# ---------------------------------------------------------------------------


class TestBoolTruncation:
	def test_nonzero_int_to_bool_normalizes(self) -> None:
		"""Assigning 42 to _Bool should generate != 0 normalization."""
		source = "int main(void) { _Bool b = 42; return b; }"
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "!="]
		assert len(binops) >= 1

	def test_negative_to_bool_normalizes(self) -> None:
		"""Assigning -1 to _Bool should normalize to 1."""
		source = "int main(void) { _Bool b = -1; return b; }"
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "!="]
		assert len(binops) >= 1

	def test_zero_to_bool(self) -> None:
		"""Assigning 0 to _Bool should still compile (result is 0)."""
		source = "int main(void) { _Bool b = 0; return b; }"
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_one_to_bool(self) -> None:
		source = "int main(void) { _Bool b = 1; return b; }"
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_large_value_to_bool_normalizes(self) -> None:
		source = "int main(void) { _Bool b = 999999; return b; }"
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "!="]
		assert len(binops) >= 1

	def test_bool_reassignment_normalizes(self) -> None:
		"""Reassigning to _Bool should also normalize."""
		source = """
		int main(void) {
			_Bool b = 0;
			b = 100;
			return b;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "!="]
		assert len(binops) >= 1

	def test_bool_from_expression(self) -> None:
		"""Result of an expression assigned to _Bool should normalize."""
		source = """
		int main(void) {
			int x = 5;
			_Bool b = x + 3;
			return b;
		}
		"""
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "!="]
		assert len(binops) >= 1


# ---------------------------------------------------------------------------
# Bool in conditions
# ---------------------------------------------------------------------------


class TestBoolInConditions:
	def test_bool_in_if(self) -> None:
		source = """
		int main(void) {
			_Bool flag = 1;
			if (flag) return 10;
			return 20;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_in_while(self) -> None:
		source = """
		int main(void) {
			_Bool running = 1;
			int count = 0;
			while (running) {
				count = count + 1;
				if (count >= 5) running = 0;
			}
			return count;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_in_for_condition(self) -> None:
		source = """
		int main(void) {
			_Bool go = 1;
			int i;
			for (i = 0; go; i = i + 1) {
				if (i >= 3) go = 0;
			}
			return i;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_in_ternary(self) -> None:
		source = """
		int main(void) {
			_Bool b = 1;
			int x = b ? 10 : 20;
			return x;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_negation_in_condition(self) -> None:
		source = """
		int main(void) {
			_Bool b = 0;
			if (!b) return 1;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_from_comparison_in_condition(self) -> None:
		source = """
		int main(void) {
			int x = 5;
			_Bool gt = x > 3;
			if (gt) return 1;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Bool arithmetic
# ---------------------------------------------------------------------------


class TestBoolArithmetic:
	def test_bool_addition(self) -> None:
		source = """
		int main(void) {
			_Bool a = 1;
			_Bool b = 1;
			int sum = a + b;
			return sum;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_subtraction(self) -> None:
		source = """
		int main(void) {
			_Bool a = 1;
			_Bool b = 0;
			int diff = a - b;
			return diff;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_multiply(self) -> None:
		source = """
		int main(void) {
			_Bool a = 1;
			_Bool b = 0;
			int prod = a * b;
			return prod;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_in_larger_expression(self) -> None:
		source = """
		int main(void) {
			_Bool a = 1;
			_Bool b = 1;
			_Bool c = 0;
			int result = a + b + c + 10;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_and_int_mixed(self) -> None:
		source = """
		int main(void) {
			_Bool flag = 1;
			int x = 5;
			int result = x * flag;
			return result;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_logical_and(self) -> None:
		source = """
		int main(void) {
			_Bool a = 1;
			_Bool b = 0;
			_Bool c = a && b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_logical_or(self) -> None:
		source = """
		int main(void) {
			_Bool a = 0;
			_Bool b = 1;
			_Bool c = a || b;
			return c;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_bool_comparison_equality(self) -> None:
		source = """
		int main(void) {
			_Bool a = 1;
			_Bool b = 1;
			_Bool eq = a == b;
			return eq;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# stdbool.h macros
# ---------------------------------------------------------------------------


class TestStdboolMacros:
	def test_bool_macro_expands(self) -> None:
		pp = Preprocessor()
		result = pp.process("#include <stdbool.h>\nbool x;")
		assert "_Bool" in result

	def test_true_expands_to_1(self) -> None:
		pp = Preprocessor()
		result = pp.process("#include <stdbool.h>\nint x = true;")
		assert "1" in result

	def test_false_expands_to_0(self) -> None:
		pp = Preprocessor()
		result = pp.process("#include <stdbool.h>\nint x = false;")
		assert "0" in result

	def test_bool_true_false_are_defined(self) -> None:
		pp = Preprocessor()
		result = pp.process("#include <stdbool.h>\nint x = __bool_true_false_are_defined;")
		assert "1" in result

	def test_stdbool_full_program_compiles(self) -> None:
		source = """
		#include <stdbool.h>
		int main(void) {
			bool flag = true;
			if (flag) return 1;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_stdbool_false_return(self) -> None:
		source = """
		#include <stdbool.h>
		int main(void) {
			bool b = false;
			return b;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_stdbool_bool_parameter(self) -> None:
		source = """
		#include <stdbool.h>
		int check(bool b) {
			if (b) return 1;
			return 0;
		}
		int main(void) {
			return check(true);
		}
		"""
		asm = _compile_to_asm(source)
		assert "check:" in asm
		assert "main:" in asm

	def test_stdbool_double_include(self) -> None:
		"""Including stdbool.h twice should not cause issues."""
		source = """
		#include <stdbool.h>
		#include <stdbool.h>
		int main(void) {
			bool b = true;
			return b;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm

	def test_stdbool_bool_in_struct(self) -> None:
		source = """
		#include <stdbool.h>
		struct Flags {
			bool active;
			bool visible;
		};
		int main(void) {
			struct Flags f;
			f.active = true;
			f.visible = false;
			if (f.active) return 1;
			return 0;
		}
		"""
		asm = _compile_to_asm(source)
		assert "main:" in asm


# ---------------------------------------------------------------------------
# Bool sizeof
# ---------------------------------------------------------------------------


class TestBoolSizeof:
	def test_sizeof_bool_is_1(self) -> None:
		source = "int main(void) { return sizeof(_Bool); }"
		ir = _compile_to_ir(source)
		func = ir.functions[0]
		rets = [i for i in func.body if isinstance(i, IRReturn)]
		assert any(isinstance(r.value, IRConst) and r.value.value == 1 for r in rets)

	def test_ir_type_bool_width_is_1(self) -> None:
		assert ir_type_byte_width(IRType.BOOL) == 1
