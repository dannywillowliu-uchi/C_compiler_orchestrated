"""Tests for unsigned integer arithmetic in IR and codegen."""

from compiler.__main__ import compile_source
from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRBinOp,
	IRConst,
	IRFunction,
	IRProgram,
	IRReturn,
	IRTemp,
	IRType,
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


def _generate(program: IRProgram) -> str:
	"""Generate assembly from an IR program."""
	return CodeGenerator().generate(program)


# ---------------------------------------------------------------------------
# IR-level: is_unsigned flag on IRBinOp
# ---------------------------------------------------------------------------


class TestIRBinOpUnsigned:
	def test_default_is_signed(self) -> None:
		instr = IRBinOp(dest=IRTemp("t0"), left=IRConst(1), op="+", right=IRConst(2))
		assert instr.is_unsigned is False

	def test_is_unsigned_flag(self) -> None:
		instr = IRBinOp(
			dest=IRTemp("t0"), left=IRConst(1), op="/", right=IRConst(2),
			is_unsigned=True,
		)
		assert instr.is_unsigned is True


# ---------------------------------------------------------------------------
# IR generation: signedness propagation from TypeSpec
# ---------------------------------------------------------------------------


class TestIRGenUnsigned:
	def test_unsigned_division_sets_flag(self) -> None:
		source = "int main() { unsigned int a = 10; unsigned int b = 3; return a / b; }"
		program = _ir_gen(source)
		func = program.functions[0]
		div_ops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "/"]
		assert len(div_ops) == 1
		assert div_ops[0].is_unsigned is True

	def test_signed_division_not_flagged(self) -> None:
		source = "int main() { int a = 10; int b = 3; return a / b; }"
		program = _ir_gen(source)
		func = program.functions[0]
		div_ops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "/"]
		assert len(div_ops) == 1
		assert div_ops[0].is_unsigned is False

	def test_unsigned_modulo_sets_flag(self) -> None:
		source = "int main() { unsigned int a = 10; unsigned int b = 3; return a % b; }"
		program = _ir_gen(source)
		func = program.functions[0]
		mod_ops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "%"]
		assert len(mod_ops) == 1
		assert mod_ops[0].is_unsigned is True

	def test_unsigned_comparison_sets_flag(self) -> None:
		source = "int main() { unsigned int a = 5; unsigned int b = 10; return a < b; }"
		program = _ir_gen(source)
		func = program.functions[0]
		cmp_ops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "<"]
		assert len(cmp_ops) == 1
		assert cmp_ops[0].is_unsigned is True

	def test_unsigned_right_shift_sets_flag(self) -> None:
		source = "int main() { unsigned int x = 16; return x >> 2; }"
		program = _ir_gen(source)
		func = program.functions[0]
		shift_ops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == ">>"]
		assert len(shift_ops) == 1
		assert shift_ops[0].is_unsigned is True

	def test_mixed_signed_unsigned_promotes_to_unsigned(self) -> None:
		source = "int main() { unsigned int a = 10; int b = 3; return a / b; }"
		program = _ir_gen(source)
		func = program.functions[0]
		div_ops = [i for i in func.body if isinstance(i, IRBinOp) and i.op == "/"]
		assert len(div_ops) == 1
		assert div_ops[0].is_unsigned is True


# ---------------------------------------------------------------------------
# Codegen: unsigned instructions emitted
# ---------------------------------------------------------------------------


class TestUnsignedCodegen:
	def test_unsigned_div_emits_divq(self) -> None:
		"""Unsigned division should use xorq %rdx,%rdx + divq, not cqto + idivq."""
		instr = IRBinOp(
			dest=IRTemp("t0"), left=IRConst(10), op="/", right=IRConst(3),
			is_unsigned=True,
		)
		func = IRFunction(
			name="test", params=[], body=[instr, IRReturn(value=IRTemp("t0"))],
			return_type=IRType.INT,
		)
		asm = _generate(IRProgram(functions=[func]))
		assert "divq" in asm
		assert "xorq %rdx, %rdx" in asm
		assert "idivq" not in asm
		assert "cqto" not in asm

	def test_signed_div_emits_idivq(self) -> None:
		"""Signed division should use cqto + idivq."""
		instr = IRBinOp(
			dest=IRTemp("t0"), left=IRConst(10), op="/", right=IRConst(3),
			is_unsigned=False,
		)
		func = IRFunction(
			name="test", params=[], body=[instr, IRReturn(value=IRTemp("t0"))],
			return_type=IRType.INT,
		)
		asm = _generate(IRProgram(functions=[func]))
		assert "idivq" in asm
		assert "cqto" in asm

	def test_unsigned_mod_emits_divq(self) -> None:
		"""Unsigned modulo should use xorq %rdx,%rdx + divq."""
		instr = IRBinOp(
			dest=IRTemp("t0"), left=IRConst(10), op="%", right=IRConst(3),
			is_unsigned=True,
		)
		func = IRFunction(
			name="test", params=[], body=[instr, IRReturn(value=IRTemp("t0"))],
			return_type=IRType.INT,
		)
		asm = _generate(IRProgram(functions=[func]))
		assert "divq" in asm
		assert "xorq %rdx, %rdx" in asm
		assert "idivq" not in asm

	def test_unsigned_less_than_emits_setb(self) -> None:
		"""Unsigned < should use setb, not setl."""
		instr = IRBinOp(
			dest=IRTemp("t0"), left=IRConst(1), op="<", right=IRConst(2),
			is_unsigned=True,
		)
		func = IRFunction(
			name="test", params=[], body=[instr, IRReturn(value=IRTemp("t0"))],
			return_type=IRType.INT,
		)
		asm = _generate(IRProgram(functions=[func]))
		assert "setb " in asm
		assert "setl " not in asm

	def test_unsigned_greater_than_emits_seta(self) -> None:
		"""Unsigned > should use seta, not setg."""
		instr = IRBinOp(
			dest=IRTemp("t0"), left=IRConst(2), op=">", right=IRConst(1),
			is_unsigned=True,
		)
		func = IRFunction(
			name="test", params=[], body=[instr, IRReturn(value=IRTemp("t0"))],
			return_type=IRType.INT,
		)
		asm = _generate(IRProgram(functions=[func]))
		assert "seta " in asm
		assert "setg " not in asm

	def test_unsigned_le_emits_setbe(self) -> None:
		"""Unsigned <= should use setbe, not setle."""
		instr = IRBinOp(
			dest=IRTemp("t0"), left=IRConst(1), op="<=", right=IRConst(2),
			is_unsigned=True,
		)
		func = IRFunction(
			name="test", params=[], body=[instr, IRReturn(value=IRTemp("t0"))],
			return_type=IRType.INT,
		)
		asm = _generate(IRProgram(functions=[func]))
		assert "setbe " in asm
		assert "setle " not in asm

	def test_unsigned_ge_emits_setae(self) -> None:
		"""Unsigned >= should use setae, not setge."""
		instr = IRBinOp(
			dest=IRTemp("t0"), left=IRConst(1), op=">=", right=IRConst(2),
			is_unsigned=True,
		)
		func = IRFunction(
			name="test", params=[], body=[instr, IRReturn(value=IRTemp("t0"))],
			return_type=IRType.INT,
		)
		asm = _generate(IRProgram(functions=[func]))
		assert "setae " in asm
		assert "setge " not in asm

	def test_unsigned_right_shift_emits_shrq(self) -> None:
		"""Unsigned >> should use shrq (logical), not sarq (arithmetic)."""
		instr = IRBinOp(
			dest=IRTemp("t0"), left=IRConst(16), op=">>", right=IRConst(2),
			is_unsigned=True,
		)
		func = IRFunction(
			name="test", params=[], body=[instr, IRReturn(value=IRTemp("t0"))],
			return_type=IRType.INT,
		)
		asm = _generate(IRProgram(functions=[func]))
		assert "shrq" in asm
		assert "sarq" not in asm

	def test_signed_right_shift_emits_sarq(self) -> None:
		"""Signed >> should still use sarq."""
		instr = IRBinOp(
			dest=IRTemp("t0"), left=IRConst(16), op=">>", right=IRConst(2),
			is_unsigned=False,
		)
		func = IRFunction(
			name="test", params=[], body=[instr, IRReturn(value=IRTemp("t0"))],
			return_type=IRType.INT,
		)
		asm = _generate(IRProgram(functions=[func]))
		assert "sarq" in asm
		assert "shrq" not in asm

	def test_unsigned_eq_ne_same_as_signed(self) -> None:
		"""== and != are the same for signed and unsigned."""
		for op, setcc in [("==", "sete"), ("!=", "setne")]:
			instr = IRBinOp(
				dest=IRTemp("t0"), left=IRConst(1), op=op, right=IRConst(2),
				is_unsigned=True,
			)
			func = IRFunction(
				name="test", params=[], body=[instr, IRReturn(value=IRTemp("t0"))],
				return_type=IRType.INT,
			)
			asm = _generate(IRProgram(functions=[func]))
			assert setcc in asm


# ---------------------------------------------------------------------------
# Full pipeline: C source -> assembly with unsigned ops
# ---------------------------------------------------------------------------


class TestUnsignedPipeline:
	def test_unsigned_division_pipeline(self) -> None:
		"""Full pipeline: unsigned int division should emit divq."""
		source = "int main() { unsigned int a = 100; unsigned int b = 7; return a / b; }"
		asm = _compile(source)
		assert "divq" in asm
		assert "idivq" not in asm

	def test_unsigned_comparison_pipeline(self) -> None:
		"""Full pipeline: unsigned int comparison should emit setb."""
		source = "int main() { unsigned int a = 5; unsigned int b = 10; return a < b; }"
		asm = _compile(source)
		assert "setb " in asm

	def test_unsigned_right_shift_pipeline(self) -> None:
		"""Full pipeline: unsigned int right shift should emit shrq."""
		source = "int main() { unsigned int x = 255; return x >> 4; }"
		asm = _compile(source)
		assert "shrq" in asm
