"""Tests for floating-point IR types and x86-64 SSE code generation."""

from compiler.__main__ import compile_source
from compiler.ir import (
	IRBinOp,
	IRConvert,
	IRCopy,
	IRFloatConst,
	IRFunction,
	IRProgram,
	IRReturn,
	IRTemp,
	IRType,
	IRUnaryOp,
)
from compiler.ir_gen import IRGenerator
from compiler.parser import Parser


def _compile(source: str) -> str:
	"""Compile C source through the full pipeline and return assembly."""
	return compile_source(source)


def _ir_gen(source: str) -> IRProgram:
	"""Parse and generate IR from C source."""
	ast = Parser.from_source(source).parse()
	return IRGenerator().generate(ast)


# ---------------------------------------------------------------------------
# IR type additions
# ---------------------------------------------------------------------------


class TestIRFloatTypes:
	def test_float_type_exists(self) -> None:
		assert IRType.FLOAT is not None
		assert IRType.DOUBLE is not None

	def test_float_const_value(self) -> None:
		fc = IRFloatConst(value=3.14, ir_type=IRType.FLOAT)
		assert fc.value == 3.14
		assert fc.ir_type == IRType.FLOAT
		assert str(fc) == "3.14"

	def test_double_const_value(self) -> None:
		fc = IRFloatConst(value=2.718, ir_type=IRType.DOUBLE)
		assert fc.value == 2.718
		assert fc.ir_type == IRType.DOUBLE

	def test_binop_with_float_type(self) -> None:
		dest = IRTemp("t0")
		left = IRFloatConst(1.0)
		right = IRFloatConst(2.0)
		instr = IRBinOp(dest=dest, left=left, op="+", right=right, ir_type=IRType.FLOAT)
		assert instr.ir_type == IRType.FLOAT

	def test_unaryop_with_float_type(self) -> None:
		dest = IRTemp("t0")
		operand = IRFloatConst(1.0)
		instr = IRUnaryOp(dest=dest, op="-", operand=operand, ir_type=IRType.FLOAT)
		assert instr.ir_type == IRType.FLOAT

	def test_copy_with_float_type(self) -> None:
		dest = IRTemp("t0")
		source = IRTemp("t1")
		instr = IRCopy(dest=dest, source=source, ir_type=IRType.FLOAT)
		assert instr.ir_type == IRType.FLOAT

	def test_convert_instruction(self) -> None:
		dest = IRTemp("t0")
		source = IRTemp("t1")
		instr = IRConvert(dest=dest, source=source, from_type=IRType.INT, to_type=IRType.FLOAT)
		assert instr.from_type == IRType.INT
		assert instr.to_type == IRType.FLOAT
		assert "INT->FLOAT" in str(instr)

	def test_function_param_types(self) -> None:
		func = IRFunction(
			name="foo", params=[IRTemp("t0")], body=[], return_type=IRType.FLOAT,
			param_types=[IRType.FLOAT],
		)
		assert func.param_types == [IRType.FLOAT]
		assert func.return_type == IRType.FLOAT


# ---------------------------------------------------------------------------
# Float variable declaration and assignment
# ---------------------------------------------------------------------------


class TestFloatVarDecl:
	def test_float_var_decl(self) -> None:
		asm = _compile("float add(float a, float b) { float c = a; return c; }")
		assert "movss" in asm

	def test_float_var_assignment(self) -> None:
		asm = _compile("""
			float f(void) {
				float x = 1.5f;
				x = 2.5f;
				return x;
			}
		""")
		assert "movss" in asm

	def test_double_var_decl(self) -> None:
		asm = _compile("double f(void) { double x = 1.5; return x; }")
		assert "movsd" in asm

	def test_float_literal_ir_gen(self) -> None:
		ir = _ir_gen("float f(void) { return 3.14f; }")
		func = ir.functions[0]
		assert func.return_type == IRType.FLOAT
		# Should have a float constant in the IR
		has_float = any(
			isinstance(instr, IRReturn) and isinstance(instr.value, IRFloatConst)
			for instr in func.body
		)
		assert has_float

	def test_double_literal_ir_gen(self) -> None:
		ir = _ir_gen("double f(void) { return 2.718; }")
		func = ir.functions[0]
		assert func.return_type == IRType.DOUBLE


# ---------------------------------------------------------------------------
# Float arithmetic operations
# ---------------------------------------------------------------------------


class TestFloatArithmetic:
	def test_float_addition(self) -> None:
		asm = _compile("float f(float a, float b) { return a + b; }")
		assert "addss" in asm

	def test_float_subtraction(self) -> None:
		asm = _compile("float f(float a, float b) { return a - b; }")
		assert "subss" in asm

	def test_float_multiplication(self) -> None:
		asm = _compile("float f(float a, float b) { return a * b; }")
		assert "mulss" in asm

	def test_float_division(self) -> None:
		asm = _compile("float f(float a, float b) { return a / b; }")
		assert "divss" in asm

	def test_double_addition(self) -> None:
		asm = _compile("double f(double a, double b) { return a + b; }")
		assert "addsd" in asm

	def test_double_subtraction(self) -> None:
		asm = _compile("double f(double a, double b) { return a - b; }")
		assert "subsd" in asm

	def test_double_multiplication(self) -> None:
		asm = _compile("double f(double a, double b) { return a * b; }")
		assert "mulsd" in asm

	def test_double_division(self) -> None:
		asm = _compile("double f(double a, double b) { return a / b; }")
		assert "divsd" in asm

	def test_float_negate(self) -> None:
		asm = _compile("float f(float x) { return -x; }")
		assert "subss" in asm

	def test_float_complex_expr(self) -> None:
		asm = _compile("float f(float a, float b, float c) { return a + b * c; }")
		assert "mulss" in asm
		assert "addss" in asm

	def test_float_ir_binop_type(self) -> None:
		ir = _ir_gen("float f(float a, float b) { return a + b; }")
		func = ir.functions[0]
		binops = [i for i in func.body if isinstance(i, IRBinOp)]
		assert len(binops) > 0
		assert binops[0].ir_type == IRType.FLOAT


# ---------------------------------------------------------------------------
# Float function parameters and return
# ---------------------------------------------------------------------------


class TestFloatFunctionParams:
	def test_float_param_uses_xmm(self) -> None:
		asm = _compile("float f(float x) { return x; }")
		# Float param should be moved from xmm0 to stack
		assert "movss" in asm
		assert "%xmm0" in asm

	def test_float_return_uses_xmm0(self) -> None:
		asm = _compile("float f(void) { return 1.0f; }")
		# Return should load into xmm0
		assert "movss" in asm
		assert "%xmm0" in asm

	def test_double_param_uses_xmm(self) -> None:
		asm = _compile("double f(double x) { return x; }")
		assert "movsd" in asm
		assert "%xmm0" in asm

	def test_double_return_uses_xmm0(self) -> None:
		asm = _compile("double f(void) { return 1.0; }")
		assert "movsd" in asm
		assert "%xmm0" in asm

	def test_multiple_float_params(self) -> None:
		asm = _compile("float f(float a, float b, float c) { return a + b + c; }")
		assert "movss" in asm
		# Should see multiple xmm register references
		assert "%xmm0" in asm

	def test_mixed_int_float_params(self) -> None:
		asm = _compile("float f(int a, float b) { return b; }")
		# Integer param goes to rdi, float param goes to xmm0
		assert "movss" in asm
		assert "%xmm0" in asm

	def test_float_function_ir_param_types(self) -> None:
		ir = _ir_gen("float f(float a, int b, double c) { return a; }")
		func = ir.functions[0]
		assert func.param_types[0] == IRType.FLOAT
		assert func.param_types[1] == IRType.INT
		assert func.param_types[2] == IRType.DOUBLE
		assert func.return_type == IRType.FLOAT


# ---------------------------------------------------------------------------
# Float-to-int and int-to-float casts
# ---------------------------------------------------------------------------


class TestFloatIntConversion:
	def test_int_to_float_cast(self) -> None:
		asm = _compile("float f(int x) { return (float)x; }")
		assert "cvtsi2ss" in asm

	def test_int_to_double_cast(self) -> None:
		asm = _compile("double f(int x) { return (double)x; }")
		assert "cvtsi2sd" in asm

	def test_float_to_int_cast(self) -> None:
		asm = _compile("int f(float x) { return (int)x; }")
		assert "cvttss2si" in asm

	def test_double_to_int_cast(self) -> None:
		asm = _compile("int f(double x) { return (int)x; }")
		assert "cvttsd2si" in asm

	def test_float_to_double_cast(self) -> None:
		asm = _compile("double f(float x) { return (double)x; }")
		assert "cvtss2sd" in asm

	def test_double_to_float_cast(self) -> None:
		asm = _compile("float f(double x) { return (float)x; }")
		assert "cvtsd2ss" in asm

	def test_convert_ir_instruction(self) -> None:
		ir = _ir_gen("float f(int x) { return (float)x; }")
		func = ir.functions[0]
		converts = [i for i in func.body if isinstance(i, IRConvert)]
		assert len(converts) > 0
		assert converts[0].from_type == IRType.INT
		assert converts[0].to_type == IRType.FLOAT

	def test_implicit_int_to_float_assignment(self) -> None:
		asm = _compile("""
			float f(void) {
				float x = 5;
				return x;
			}
		""")
		assert "cvtsi2ss" in asm


# ---------------------------------------------------------------------------
# Double type support
# ---------------------------------------------------------------------------


class TestDoubleSupport:
	def test_double_arithmetic(self) -> None:
		asm = _compile("double f(double a, double b) { return (a + b) * (a - b); }")
		assert "addsd" in asm
		assert "subsd" in asm
		assert "mulsd" in asm

	def test_double_var_and_return(self) -> None:
		asm = _compile("""
			double f(void) {
				double pi = 3.14159;
				double r = 2.0;
				return pi * r * r;
			}
		""")
		assert "mulsd" in asm
		assert "movsd" in asm

	def test_double_comparison(self) -> None:
		asm = _compile("""
			int f(double a, double b) {
				if (a < b) {
					return 1;
				}
				return 0;
			}
		""")
		assert "ucomisd" in asm

	def test_double_negate(self) -> None:
		asm = _compile("double f(double x) { return -x; }")
		assert "subsd" in asm


# ---------------------------------------------------------------------------
# Mixed int/float expressions
# ---------------------------------------------------------------------------


class TestMixedExpressions:
	def test_float_comparison(self) -> None:
		asm = _compile("""
			int f(float a, float b) {
				if (a > b) {
					return 1;
				}
				return 0;
			}
		""")
		assert "ucomiss" in asm

	def test_float_constant_pool(self) -> None:
		asm = _compile("float f(void) { return 3.14f; }")
		# Float constants need to be loaded from .rodata
		assert ".rodata" in asm
		assert ".long" in asm or ".quad" in asm

	def test_double_constant_pool(self) -> None:
		asm = _compile("double f(void) { return 2.718281828; }")
		assert ".rodata" in asm
		assert ".quad" in asm

	def test_mixed_int_float_arithmetic(self) -> None:
		asm = _compile("""
			float f(int a) {
				float b = 2.5f;
				return (float)a + b;
			}
		""")
		assert "cvtsi2ss" in asm
		assert "addss" in asm

	def test_full_pipeline_float_function(self) -> None:
		asm = _compile("""
			float add_floats(float a, float b) {
				return a + b;
			}
		""")
		assert ".globl add_floats" in asm
		assert "pushq %rbp" in asm
		assert "movss" in asm
		assert "addss" in asm
		assert "ret" in asm

	def test_full_pipeline_double_function(self) -> None:
		asm = _compile("""
			double multiply(double x, double y) {
				return x * y;
			}
		""")
		assert ".globl multiply" in asm
		assert "movsd" in asm
		assert "mulsd" in asm
		assert "ret" in asm
