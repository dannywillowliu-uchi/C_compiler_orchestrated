"""Tests for IR type extensions: SHORT, LONG, and width helper functions."""

import pytest

from compiler.ir import (
	IRType,
	ir_type_asm_suffix,
	ir_type_byte_width,
	ir_type_is_integer,
)


class TestIRTypeEnumExtensions:
	def test_short_exists(self) -> None:
		assert IRType.SHORT is not None
		assert IRType.SHORT.name == "SHORT"

	def test_long_exists(self) -> None:
		assert IRType.LONG is not None
		assert IRType.LONG.name == "LONG"

	def test_all_members_distinct(self) -> None:
		values = [m.value for m in IRType]
		assert len(values) == len(set(values))


class TestIRTypeByteWidth:
	@pytest.mark.parametrize(
		"ir_type,expected",
		[
			(IRType.CHAR, 1),
			(IRType.SHORT, 2),
			(IRType.INT, 4),
			(IRType.LONG, 8),
			(IRType.POINTER, 8),
			(IRType.FLOAT, 4),
			(IRType.DOUBLE, 8),
			(IRType.VOID, 0),
		],
	)
	def test_byte_widths(self, ir_type: IRType, expected: int) -> None:
		assert ir_type_byte_width(ir_type) == expected


class TestIRTypeIsInteger:
	@pytest.mark.parametrize("ir_type", [IRType.CHAR, IRType.SHORT, IRType.INT, IRType.LONG])
	def test_integer_types(self, ir_type: IRType) -> None:
		assert ir_type_is_integer(ir_type) is True

	@pytest.mark.parametrize("ir_type", [IRType.VOID, IRType.POINTER, IRType.FLOAT, IRType.DOUBLE])
	def test_non_integer_types(self, ir_type: IRType) -> None:
		assert ir_type_is_integer(ir_type) is False


class TestIRTypeAsmSuffix:
	@pytest.mark.parametrize(
		"ir_type,expected",
		[
			(IRType.CHAR, "b"),
			(IRType.SHORT, "w"),
			(IRType.INT, "l"),
			(IRType.LONG, "q"),
			(IRType.POINTER, "q"),
		],
	)
	def test_asm_suffixes(self, ir_type: IRType, expected: str) -> None:
		assert ir_type_asm_suffix(ir_type) == expected

	def test_unsupported_type_raises(self) -> None:
		with pytest.raises(KeyError):
			ir_type_asm_suffix(IRType.VOID)

		with pytest.raises(KeyError):
			ir_type_asm_suffix(IRType.FLOAT)
