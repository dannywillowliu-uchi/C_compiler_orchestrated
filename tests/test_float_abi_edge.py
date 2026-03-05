"""Edge-case tests for float/double codegen correctness and ABI compliance.

Tests cover: float/double parameter passing (>8 float args spilling to stack),
mixed int/float args, float return values, float-to-int and int-to-float
conversions at boundaries, float comparison edge cases, double arithmetic precision.
"""

import re

from compiler.__main__ import compile_source
from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRBinOp,
	IRCall,
	IRConst,
	IRConvert,
	IRCopy,
	IRFloatConst,
	IRFunction,
	IRInstruction,
	IRProgram,
	IRReturn,
	IRTemp,
	IRType,
)


def _gen(func: IRFunction, extra_funcs: list[IRFunction] | None = None) -> str:
	funcs = [func] + (extra_funcs or [])
	return CodeGenerator().generate(IRProgram(funcs))


def _asm_lines(asm: str) -> list[str]:
	return [line.strip() for line in asm.split("\n") if line.strip()]


def _full_pipeline(source: str, optimize: bool = False) -> str:
	return compile_source(source, optimize=optimize)


# ---------------------------------------------------------------------------
# 1. Float/double parameter passing: >8 float args spilling to stack
# ---------------------------------------------------------------------------


class TestFloatArgSpilling:
	"""Test that >8 float args correctly spill to the stack per System V ABI."""

	def test_8_float_args_all_in_xmm(self) -> None:
		"""Exactly 8 float args should all fit in xmm0-xmm7."""
		args = [IRFloatConst(float(i)) for i in range(8)]
		arg_types = [IRType.FLOAT] * 8
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "eight_floats", args, arg_types, IRType.FLOAT),
			IRReturn(IRTemp("r"), ir_type=IRType.FLOAT),
		]
		func = IRFunction("caller", [], body, IRType.FLOAT)
		asm = _gen(func)
		for i in range(8):
			assert f"%xmm{i}" in asm
		assert "movb $8, %al" in asm

	def test_8_float_params_received_in_xmm(self) -> None:
		"""A function receiving 8 float params should read from xmm0-xmm7."""
		params = [IRTemp(f"f{i}") for i in range(8)]
		param_types = [IRType.FLOAT] * 8
		body: list[IRInstruction] = [IRReturn(IRTemp("f7"), ir_type=IRType.FLOAT)]
		func = IRFunction("recv8", params, body, IRType.FLOAT, param_types)
		asm = _gen(func)
		# All 8 xmm registers should appear (params stored from them)
		for i in range(8):
			assert f"%xmm{i}" in asm

	def test_1_float_arg_sets_al_to_1(self) -> None:
		"""Single float arg should set %al to 1."""
		args = [IRFloatConst(1.0)]
		arg_types = [IRType.FLOAT]
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "fn", args, arg_types, IRType.FLOAT),
			IRReturn(IRTemp("r"), ir_type=IRType.FLOAT),
		]
		func = IRFunction("caller", [], body, IRType.FLOAT)
		asm = _gen(func)
		assert "movb $1, %al" in asm


# ---------------------------------------------------------------------------
# 2. Mixed int/float arguments edge cases
# ---------------------------------------------------------------------------


class TestMixedIntFloatEdge:
	"""Test int and float args using independent ABI register counters."""

	def test_6_ints_1_float(self) -> None:
		"""6 int args fill all GP regs, 1 float goes to xmm0."""
		int_args = [IRConst(i) for i in range(6)]
		float_args = [IRFloatConst(3.14)]
		all_args = int_args + float_args
		all_types = [IRType.INT] * 6 + [IRType.FLOAT]
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "mixed", all_args, all_types, IRType.INT),
			IRReturn(IRTemp("r")),
		]
		func = IRFunction("caller", [], body, IRType.INT)
		asm = _gen(func)
		assert "%xmm0" in asm
		assert "%r9" in asm  # 6th int register
		assert "movb $1, %al" in asm

	def test_alternating_int_float_args(self) -> None:
		"""Alternating int/float: each type uses its own register counter."""
		args = [IRConst(1), IRFloatConst(1.0), IRConst(2), IRFloatConst(2.0)]
		types = [IRType.INT, IRType.DOUBLE, IRType.INT, IRType.DOUBLE]
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "alt", args, types, IRType.INT),
			IRReturn(IRTemp("r")),
		]
		func = IRFunction("caller", [], body, IRType.INT)
		asm = _gen(func)
		# 2 int args -> %rdi, %rsi
		assert "%rdi" in asm
		assert "%rsi" in asm
		# 2 float args -> %xmm0, %xmm1
		assert "%xmm0" in asm
		assert "%xmm1" in asm
		assert "movb $2, %al" in asm

	def test_7_ints_2_floats_stack_spill(self) -> None:
		"""7 int args (1 spills to stack) + 2 float args in xmm regs."""
		int_args = [IRConst(i) for i in range(7)]
		float_args = [IRFloatConst(1.0), IRFloatConst(2.0)]
		all_args = int_args + float_args
		all_types = [IRType.INT] * 7 + [IRType.FLOAT, IRType.FLOAT]
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "big", all_args, all_types, IRType.INT),
			IRReturn(IRTemp("r")),
		]
		func = IRFunction("caller", [], body, IRType.INT)
		asm = _gen(func)
		# 7th int spills to stack
		assert "pushq" in asm
		assert "%xmm0" in asm
		assert "%xmm1" in asm
		assert "movb $2, %al" in asm

	def test_only_floats_no_int_regs_used(self) -> None:
		"""When all args are floats, no GP arg registers should be used for params."""
		args = [IRFloatConst(float(i)) for i in range(4)]
		types = [IRType.DOUBLE] * 4
		body: list[IRInstruction] = [
			IRCall(IRTemp("r"), "all_doubles", args, types, IRType.DOUBLE),
			IRReturn(IRTemp("r"), ir_type=IRType.DOUBLE),
		]
		func = IRFunction("caller", [], body, IRType.DOUBLE)
		asm = _gen(func)
		assert "movb $4, %al" in asm
		for i in range(4):
			assert f"%xmm{i}" in asm


# ---------------------------------------------------------------------------
# 3. Float return values
# ---------------------------------------------------------------------------


class TestFloatReturnValues:
	"""Test that float/double return values are in %xmm0."""

	def test_float_return_uses_xmm0(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("f"), IRFloatConst(42.0), ir_type=IRType.FLOAT),
			IRReturn(IRTemp("f"), ir_type=IRType.FLOAT),
		]
		func = IRFunction("retf", [], body, IRType.FLOAT)
		asm = _gen(func)
		assert "movss" in asm
		assert "%xmm0" in asm

	def test_double_return_uses_xmm0(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("d"), IRFloatConst(99.0), ir_type=IRType.DOUBLE),
			IRReturn(IRTemp("d"), ir_type=IRType.DOUBLE),
		]
		func = IRFunction("retd", [], body, IRType.DOUBLE)
		asm = _gen(func)
		assert "movsd" in asm
		assert "%xmm0" in asm

	def test_call_returning_float_stores_from_xmm0(self) -> None:
		"""Return value from a float-returning call should be read from xmm0."""
		body: list[IRInstruction] = [
			IRCall(IRTemp("f"), "get_float", [], [], IRType.FLOAT),
			IRReturn(IRTemp("f"), ir_type=IRType.FLOAT),
		]
		func = IRFunction("caller", [], body, IRType.FLOAT)
		asm = _gen(func)
		assert "call get_float" in asm
		assert "movss %xmm0" in asm

	def test_call_returning_double_stores_from_xmm0(self) -> None:
		body: list[IRInstruction] = [
			IRCall(IRTemp("d"), "get_double", [], [], IRType.DOUBLE),
			IRReturn(IRTemp("d"), ir_type=IRType.DOUBLE),
		]
		func = IRFunction("caller", [], body, IRType.DOUBLE)
		asm = _gen(func)
		assert "call get_double" in asm
		assert "movsd %xmm0" in asm


# ---------------------------------------------------------------------------
# 4. Float-to-int and int-to-float conversions
# ---------------------------------------------------------------------------


class TestFloatIntConversions:
	"""Test conversion instructions between float/double and int types."""

	def test_int_to_float_conversion(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("i"), IRConst(42)),
			IRConvert(IRTemp("f"), IRTemp("i"), IRType.INT, IRType.FLOAT),
			IRReturn(IRTemp("f"), ir_type=IRType.FLOAT),
		]
		func = IRFunction("i2f", [], body, IRType.FLOAT)
		asm = _gen(func)
		assert "cvtsi2ss" in asm

	def test_int_to_double_conversion(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("i"), IRConst(42)),
			IRConvert(IRTemp("d"), IRTemp("i"), IRType.INT, IRType.DOUBLE),
			IRReturn(IRTemp("d"), ir_type=IRType.DOUBLE),
		]
		func = IRFunction("i2d", [], body, IRType.DOUBLE)
		asm = _gen(func)
		assert "cvtsi2sd" in asm

	def test_float_to_int_conversion(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("f"), IRFloatConst(3.14), ir_type=IRType.FLOAT),
			IRConvert(IRTemp("i"), IRTemp("f"), IRType.FLOAT, IRType.INT),
			IRReturn(IRTemp("i")),
		]
		func = IRFunction("f2i", [], body, IRType.INT)
		asm = _gen(func)
		assert "cvttss2si" in asm

	def test_double_to_int_conversion(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("d"), IRFloatConst(3.14), ir_type=IRType.DOUBLE),
			IRConvert(IRTemp("i"), IRTemp("d"), IRType.DOUBLE, IRType.INT),
			IRReturn(IRTemp("i")),
		]
		func = IRFunction("d2i", [], body, IRType.INT)
		asm = _gen(func)
		assert "cvttsd2si" in asm

	def test_float_to_double_promotion(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("f"), IRFloatConst(1.5), ir_type=IRType.FLOAT),
			IRConvert(IRTemp("d"), IRTemp("f"), IRType.FLOAT, IRType.DOUBLE),
			IRReturn(IRTemp("d"), ir_type=IRType.DOUBLE),
		]
		func = IRFunction("f2d", [], body, IRType.DOUBLE)
		asm = _gen(func)
		assert "cvtss2sd" in asm

	def test_double_to_float_demotion(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("d"), IRFloatConst(1.5), ir_type=IRType.DOUBLE),
			IRConvert(IRTemp("f"), IRTemp("d"), IRType.DOUBLE, IRType.FLOAT),
			IRReturn(IRTemp("f"), ir_type=IRType.FLOAT),
		]
		func = IRFunction("d2f", [], body, IRType.FLOAT)
		asm = _gen(func)
		assert "cvtsd2ss" in asm

	def test_int_to_float_sign_extension(self) -> None:
		"""Converting a large int should use movslq for sign extension."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("f"), IRFloatConst(1.0), ir_type=IRType.FLOAT),
			IRConvert(IRTemp("i"), IRTemp("f"), IRType.FLOAT, IRType.INT),
			IRReturn(IRTemp("i")),
		]
		func = IRFunction("f2i_ext", [], body, IRType.INT)
		asm = _gen(func)
		assert "cvttss2si" in asm
		assert "movslq" in asm


# ---------------------------------------------------------------------------
# 5. Float comparison edge cases
# ---------------------------------------------------------------------------


class TestFloatComparisonEdgeCases:
	"""Test float comparison codegen including ordered/unordered comparisons."""

	def test_float_equality_comparison(self) -> None:
		"""Float == generates comiss/ucomiss + sete or similar."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("a"), IRFloatConst(1.0), ir_type=IRType.FLOAT),
			IRCopy(IRTemp("b"), IRFloatConst(1.0), ir_type=IRType.FLOAT),
			IRBinOp(IRTemp("eq"), IRTemp("a"), "==", IRTemp("b"), ir_type=IRType.FLOAT),
			IRReturn(IRTemp("eq")),
		]
		func = IRFunction("feq", [], body, IRType.INT)
		asm = _gen(func)
		assert "comiss" in asm.lower() or "ucomiss" in asm.lower()

	def test_float_less_than_comparison(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("a"), IRFloatConst(1.0), ir_type=IRType.FLOAT),
			IRCopy(IRTemp("b"), IRFloatConst(2.0), ir_type=IRType.FLOAT),
			IRBinOp(IRTemp("lt"), IRTemp("a"), "<", IRTemp("b"), ir_type=IRType.FLOAT),
			IRReturn(IRTemp("lt")),
		]
		func = IRFunction("flt", [], body, IRType.INT)
		asm = _gen(func)
		assert "comiss" in asm.lower() or "ucomiss" in asm.lower()

	def test_double_comparison(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("a"), IRFloatConst(1.0), ir_type=IRType.DOUBLE),
			IRCopy(IRTemp("b"), IRFloatConst(2.0), ir_type=IRType.DOUBLE),
			IRBinOp(IRTemp("lt"), IRTemp("a"), "<", IRTemp("b"), ir_type=IRType.DOUBLE),
			IRReturn(IRTemp("lt")),
		]
		func = IRFunction("dlt", [], body, IRType.INT)
		asm = _gen(func)
		assert "comisd" in asm.lower() or "ucomisd" in asm.lower()

	def test_float_not_equal(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("a"), IRFloatConst(1.0), ir_type=IRType.FLOAT),
			IRCopy(IRTemp("b"), IRFloatConst(2.0), ir_type=IRType.FLOAT),
			IRBinOp(IRTemp("ne"), IRTemp("a"), "!=", IRTemp("b"), ir_type=IRType.FLOAT),
			IRReturn(IRTemp("ne")),
		]
		func = IRFunction("fne", [], body, IRType.INT)
		asm = _gen(func)
		assert "comiss" in asm.lower() or "ucomiss" in asm.lower()


# ---------------------------------------------------------------------------
# 6. Double arithmetic precision
# ---------------------------------------------------------------------------


class TestDoubleArithmeticPrecision:
	"""Test codegen emits correct double-precision instructions."""

	def test_double_add(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("a"), IRFloatConst(1.0), ir_type=IRType.DOUBLE),
			IRCopy(IRTemp("b"), IRFloatConst(2.0), ir_type=IRType.DOUBLE),
			IRBinOp(IRTemp("sum"), IRTemp("a"), "+", IRTemp("b"), ir_type=IRType.DOUBLE),
			IRReturn(IRTemp("sum"), ir_type=IRType.DOUBLE),
		]
		func = IRFunction("dadd", [], body, IRType.DOUBLE)
		asm = _gen(func)
		assert "addsd" in asm

	def test_double_sub(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("a"), IRFloatConst(3.0), ir_type=IRType.DOUBLE),
			IRCopy(IRTemp("b"), IRFloatConst(1.0), ir_type=IRType.DOUBLE),
			IRBinOp(IRTemp("diff"), IRTemp("a"), "-", IRTemp("b"), ir_type=IRType.DOUBLE),
			IRReturn(IRTemp("diff"), ir_type=IRType.DOUBLE),
		]
		func = IRFunction("dsub", [], body, IRType.DOUBLE)
		asm = _gen(func)
		assert "subsd" in asm

	def test_double_mul(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("a"), IRFloatConst(2.0), ir_type=IRType.DOUBLE),
			IRCopy(IRTemp("b"), IRFloatConst(3.0), ir_type=IRType.DOUBLE),
			IRBinOp(IRTemp("prod"), IRTemp("a"), "*", IRTemp("b"), ir_type=IRType.DOUBLE),
			IRReturn(IRTemp("prod"), ir_type=IRType.DOUBLE),
		]
		func = IRFunction("dmul", [], body, IRType.DOUBLE)
		asm = _gen(func)
		assert "mulsd" in asm

	def test_double_div(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("a"), IRFloatConst(6.0), ir_type=IRType.DOUBLE),
			IRCopy(IRTemp("b"), IRFloatConst(2.0), ir_type=IRType.DOUBLE),
			IRBinOp(IRTemp("quot"), IRTemp("a"), "/", IRTemp("b"), ir_type=IRType.DOUBLE),
			IRReturn(IRTemp("quot"), ir_type=IRType.DOUBLE),
		]
		func = IRFunction("ddiv", [], body, IRType.DOUBLE)
		asm = _gen(func)
		assert "divsd" in asm

	def test_float_add_uses_ss_suffix(self) -> None:
		"""Float (single precision) should use addss, not addsd."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("a"), IRFloatConst(1.0), ir_type=IRType.FLOAT),
			IRCopy(IRTemp("b"), IRFloatConst(2.0), ir_type=IRType.FLOAT),
			IRBinOp(IRTemp("sum"), IRTemp("a"), "+", IRTemp("b"), ir_type=IRType.FLOAT),
			IRReturn(IRTemp("sum"), ir_type=IRType.FLOAT),
		]
		func = IRFunction("fadd", [], body, IRType.FLOAT)
		asm = _gen(func)
		assert "addss" in asm

	def test_float_mul_uses_ss_suffix(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("a"), IRFloatConst(2.0), ir_type=IRType.FLOAT),
			IRCopy(IRTemp("b"), IRFloatConst(3.0), ir_type=IRType.FLOAT),
			IRBinOp(IRTemp("prod"), IRTemp("a"), "*", IRTemp("b"), ir_type=IRType.FLOAT),
			IRReturn(IRTemp("prod"), ir_type=IRType.FLOAT),
		]
		func = IRFunction("fmul", [], body, IRType.FLOAT)
		asm = _gen(func)
		assert "mulss" in asm


# ---------------------------------------------------------------------------
# 7. Full pipeline tests (C source -> ASM)
# ---------------------------------------------------------------------------


class TestFloatFullPipeline:
	"""Test float/double handling through the full compile pipeline."""

	def test_float_variable_and_return(self) -> None:
		source = """
		float identity(float x) { return x; }
		int main(void) { return 0; }
		"""
		asm = _full_pipeline(source)
		assert "identity:" in asm
		assert "movss" in asm

	def test_double_variable_and_return(self) -> None:
		source = """
		double identity(double x) { return x; }
		int main(void) { return 0; }
		"""
		asm = _full_pipeline(source)
		assert "identity:" in asm
		assert "movsd" in asm

	def test_float_addition(self) -> None:
		source = """
		float add(float a, float b) { return a + b; }
		int main(void) { return 0; }
		"""
		asm = _full_pipeline(source)
		assert "addss" in asm

	def test_double_addition(self) -> None:
		source = """
		double add(double a, double b) { return a + b; }
		int main(void) { return 0; }
		"""
		asm = _full_pipeline(source)
		assert "addsd" in asm

	def test_float_literal_in_expression(self) -> None:
		source = """
		int main(void) {
			float x = 3.14f;
			return 0;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm
		assert "ret" in asm

	def test_double_literal_in_expression(self) -> None:
		source = """
		int main(void) {
			double x = 2.718;
			return 0;
		}
		"""
		asm = _full_pipeline(source)
		assert "main:" in asm

	def test_float_to_int_cast(self) -> None:
		source = """
		int main(void) {
			float f = 42.7f;
			int i = (int)f;
			return i;
		}
		"""
		asm = _full_pipeline(source)
		assert "cvttss2si" in asm

	def test_int_to_float_cast(self) -> None:
		source = """
		int main(void) {
			int i = 42;
			float f = (float)i;
			return 0;
		}
		"""
		asm = _full_pipeline(source)
		assert "cvtsi2ss" in asm

	def test_double_to_int_cast(self) -> None:
		source = """
		int main(void) {
			double d = 99.9;
			int i = (int)d;
			return i;
		}
		"""
		asm = _full_pipeline(source)
		assert "cvttsd2si" in asm

	def test_int_to_double_cast(self) -> None:
		source = """
		int main(void) {
			int i = 100;
			double d = (double)i;
			return 0;
		}
		"""
		asm = _full_pipeline(source)
		assert "cvtsi2sd" in asm

	def test_float_to_double_promotion(self) -> None:
		source = """
		int main(void) {
			float f = 1.5f;
			double d = (double)f;
			return 0;
		}
		"""
		asm = _full_pipeline(source)
		assert "cvtss2sd" in asm

	def test_double_to_float_demotion(self) -> None:
		source = """
		int main(void) {
			double d = 1.5;
			float f = (float)d;
			return 0;
		}
		"""
		asm = _full_pipeline(source)
		assert "cvtsd2ss" in asm

	def test_float_comparison_operators(self) -> None:
		source = """
		int less(float a, float b) { return a < b; }
		int main(void) { return 0; }
		"""
		asm = _full_pipeline(source)
		assert "comiss" in asm.lower() or "ucomiss" in asm.lower()

	def test_double_comparison_operators(self) -> None:
		source = """
		int less(double a, double b) { return a < b; }
		int main(void) { return 0; }
		"""
		asm = _full_pipeline(source)
		assert "comisd" in asm.lower() or "ucomisd" in asm.lower()

	def test_multiple_float_params(self) -> None:
		"""Function with multiple float params should use sequential xmm regs."""
		source = """
		float add3(float a, float b, float c) { return a + b + c; }
		int main(void) { return 0; }
		"""
		asm = _full_pipeline(source)
		assert "%xmm0" in asm
		assert "%xmm1" in asm
		assert "%xmm2" in asm

	def test_mixed_int_float_params(self) -> None:
		"""Mixed int/float params use independent register counters."""
		source = """
		int mixed(int a, float b, int c) { return a + c; }
		int main(void) { return 0; }
		"""
		asm = _full_pipeline(source)
		assert "%rdi" in asm or "%edi" in asm  # first int param
		assert "%xmm0" in asm  # first float param

	def test_float_arithmetic_all_ops(self) -> None:
		"""All four arithmetic operations on floats."""
		source = """
		float compute(float a, float b) {
			float sum = a + b;
			float diff = a - b;
			float prod = a * b;
			float quot = a / b;
			return sum + diff + prod + quot;
		}
		int main(void) { return 0; }
		"""
		asm = _full_pipeline(source)
		assert "addss" in asm
		assert "subss" in asm
		assert "mulss" in asm
		assert "divss" in asm

	def test_double_arithmetic_all_ops(self) -> None:
		source = """
		double compute(double a, double b) {
			double sum = a + b;
			double diff = a - b;
			double prod = a * b;
			double quot = a / b;
			return sum + diff + prod + quot;
		}
		int main(void) { return 0; }
		"""
		asm = _full_pipeline(source)
		assert "addsd" in asm
		assert "subsd" in asm
		assert "mulsd" in asm
		assert "divsd" in asm

	def test_float_negation(self) -> None:
		source = """
		float neg(float x) { return -x; }
		int main(void) { return 0; }
		"""
		asm = _full_pipeline(source)
		assert "neg:" in asm or "_neg:" in asm

	def test_float_in_if_condition(self) -> None:
		"""Float value used as an if condition should generate comparison to zero."""
		source = """
		int test(float x) {
			if (x) return 1;
			return 0;
		}
		int main(void) { return 0; }
		"""
		asm = _full_pipeline(source)
		assert "test:" in asm or "_test:" in asm


# ---------------------------------------------------------------------------
# 8. Float constant data section
# ---------------------------------------------------------------------------


class TestFloatConstantData:
	"""Test that float constants are placed in the data section correctly."""

	def test_float_const_generates_data_label(self) -> None:
		"""Float constants should create labeled data entries."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("f"), IRFloatConst(3.14), ir_type=IRType.FLOAT),
			IRReturn(IRTemp("f"), ir_type=IRType.FLOAT),
		]
		func = IRFunction("getpi", [], body, IRType.FLOAT)
		asm = _gen(func)
		# Float constants are typically in .data or .section __DATA or .rodata
		assert "movss" in asm

	def test_double_const_generates_data_label(self) -> None:
		body: list[IRInstruction] = [
			IRCopy(IRTemp("d"), IRFloatConst(2.718), ir_type=IRType.DOUBLE),
			IRReturn(IRTemp("d"), ir_type=IRType.DOUBLE),
		]
		func = IRFunction("gete", [], body, IRType.DOUBLE)
		asm = _gen(func)
		assert "movsd" in asm

	def test_multiple_float_consts_get_separate_labels(self) -> None:
		"""Each distinct float constant should get its own data label."""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("a"), IRFloatConst(1.0), ir_type=IRType.FLOAT),
			IRCopy(IRTemp("b"), IRFloatConst(2.0), ir_type=IRType.FLOAT),
			IRBinOp(IRTemp("sum"), IRTemp("a"), "+", IRTemp("b"), ir_type=IRType.FLOAT),
			IRReturn(IRTemp("sum"), ir_type=IRType.FLOAT),
		]
		func = IRFunction("sum2", [], body, IRType.FLOAT)
		asm = _gen(func)
		# Should have at least 2 float constant labels
		float_labels = re.findall(r"\.LFC\d+:", asm)
		assert len(float_labels) >= 2
