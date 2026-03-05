"""Edge-case tests for IR type system and instruction validation."""

import pytest

from compiler.ir import (
	IRAddrOf,
	IRAlloc,
	IRBinOp,
	IRCall,
	IRCondJump,
	IRConst,
	IRConvert,
	IRCopy,
	IRFloatConst,
	IRFunction,
	IRGlobalRef,
	IRGlobalVar,
	IRInstruction,
	IRJump,
	IRLabel,
	IRLabelInstr,
	IRLoad,
	IRParam,
	IRProgram,
	IRReturn,
	IRStore,
	IRStringData,
	IRTemp,
	IRType,
	IRUnaryOp,
	IRValue,
	ir_type_asm_suffix,
	ir_type_byte_width,
	ir_type_is_integer,
)


class TestIRTypeByteWidth:
	"""Test ir_type_byte_width for all IR types."""

	@pytest.mark.parametrize(
		"ir_type, expected",
		[
			(IRType.BOOL, 1),
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
	def test_byte_width(self, ir_type: IRType, expected: int) -> None:
		assert ir_type_byte_width(ir_type) == expected

	def test_all_types_covered(self) -> None:
		"""Every IRType member has a defined byte width."""
		for t in IRType:
			result = ir_type_byte_width(t)
			assert isinstance(result, int)
			assert result >= 0


class TestIRTypeIsInteger:
	"""Test ir_type_is_integer boundary cases."""

	@pytest.mark.parametrize("ir_type", [IRType.BOOL, IRType.CHAR, IRType.SHORT, IRType.INT, IRType.LONG])
	def test_integer_types(self, ir_type: IRType) -> None:
		assert ir_type_is_integer(ir_type) is True

	@pytest.mark.parametrize("ir_type", [IRType.VOID, IRType.POINTER, IRType.FLOAT, IRType.DOUBLE])
	def test_non_integer_types(self, ir_type: IRType) -> None:
		assert ir_type_is_integer(ir_type) is False

	def test_all_types_return_bool(self) -> None:
		for t in IRType:
			assert isinstance(ir_type_is_integer(t), bool)


class TestIRTypeAsmSuffix:
	"""Test ir_type_asm_suffix for each supported type."""

	@pytest.mark.parametrize(
		"ir_type, expected",
		[
			(IRType.BOOL, "b"),
			(IRType.CHAR, "b"),
			(IRType.SHORT, "w"),
			(IRType.INT, "l"),
			(IRType.LONG, "q"),
			(IRType.POINTER, "q"),
		],
	)
	def test_asm_suffix(self, ir_type: IRType, expected: str) -> None:
		assert ir_type_asm_suffix(ir_type) == expected

	@pytest.mark.parametrize("ir_type", [IRType.FLOAT, IRType.DOUBLE, IRType.VOID])
	def test_unsupported_types_raise(self, ir_type: IRType) -> None:
		with pytest.raises(KeyError):
			ir_type_asm_suffix(ir_type)


class TestIRValues:
	"""Test IR value types and their string representations."""

	def test_irconst_default_type(self) -> None:
		c = IRConst(value=42)
		assert c.ir_type == IRType.INT
		assert c.is_unsigned is False
		assert str(c) == "42"

	def test_irconst_unsigned(self) -> None:
		c = IRConst(value=255, ir_type=IRType.CHAR, is_unsigned=True)
		assert c.is_unsigned is True
		assert c.ir_type == IRType.CHAR

	def test_irconst_negative(self) -> None:
		c = IRConst(value=-1)
		assert str(c) == "-1"

	def test_irconst_zero(self) -> None:
		c = IRConst(value=0)
		assert str(c) == "0"

	def test_irfloatconst_default_type(self) -> None:
		f = IRFloatConst(value=3.14)
		assert f.ir_type == IRType.FLOAT
		assert str(f) == "3.14"

	def test_irfloatconst_double(self) -> None:
		f = IRFloatConst(value=2.718, ir_type=IRType.DOUBLE)
		assert f.ir_type == IRType.DOUBLE

	def test_irfloatconst_zero(self) -> None:
		f = IRFloatConst(value=0.0)
		assert str(f) == "0.0"

	def test_irtemp_str(self) -> None:
		t = IRTemp(name="t0")
		assert str(t) == "t0"

	def test_irlabel_str(self) -> None:
		lbl = IRLabel(name="L_entry")
		assert str(lbl) == "L_entry"

	def test_irglobalref_str(self) -> None:
		g = IRGlobalRef(name="my_global")
		assert str(g) == "&my_global"

	def test_irvalue_base_str_raises(self) -> None:
		v = IRValue()
		with pytest.raises(NotImplementedError):
			str(v)

	def test_frozen_values(self) -> None:
		"""IR values are frozen dataclasses (immutable)."""
		c = IRConst(value=10)
		with pytest.raises(AttributeError):
			c.value = 20  # type: ignore[misc]

		t = IRTemp(name="t0")
		with pytest.raises(AttributeError):
			t.name = "t1"  # type: ignore[misc]


class TestIRConvert:
	"""Test IRConvert between all type pairs."""

	ALL_TYPES = list(IRType)

	def test_convert_str_format(self) -> None:
		conv = IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.INT, to_type=IRType.FLOAT)
		assert str(conv) == "t1 = convert t0 INT->FLOAT"

	def test_convert_same_type(self) -> None:
		conv = IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.INT, to_type=IRType.INT)
		assert "INT->INT" in str(conv)

	@pytest.mark.parametrize("from_type", ALL_TYPES)
	def test_convert_from_each_type(self, from_type: IRType) -> None:
		"""IRConvert can be constructed from any type."""
		conv = IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=from_type, to_type=IRType.INT)
		assert conv.from_type == from_type

	@pytest.mark.parametrize("to_type", ALL_TYPES)
	def test_convert_to_each_type(self, to_type: IRType) -> None:
		"""IRConvert can be constructed to any type."""
		conv = IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.INT, to_type=to_type)
		assert conv.to_type == to_type

	def test_convert_defaults(self) -> None:
		conv = IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"))
		assert conv.from_type == IRType.INT
		assert conv.to_type == IRType.FLOAT

	def test_convert_with_const_source(self) -> None:
		conv = IRConvert(dest=IRTemp("t1"), source=IRConst(value=42), from_type=IRType.INT, to_type=IRType.DOUBLE)
		assert "42" in str(conv)

	def test_convert_int_to_pointer(self) -> None:
		conv = IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.INT, to_type=IRType.POINTER)
		assert "INT->POINTER" in str(conv)

	def test_convert_pointer_to_int(self) -> None:
		conv = IRConvert(dest=IRTemp("t1"), source=IRTemp("t0"), from_type=IRType.POINTER, to_type=IRType.LONG)
		assert "POINTER->LONG" in str(conv)


class TestIRBinOp:
	"""Test IRBinOp with different type combinations."""

	def test_binop_str(self) -> None:
		b = IRBinOp(dest=IRTemp("t2"), left=IRTemp("t0"), op="+", right=IRTemp("t1"))
		assert str(b) == "t2 = t0 + t1"

	def test_binop_with_const(self) -> None:
		b = IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="*", right=IRConst(value=2))
		assert str(b) == "t1 = t0 * 2"

	def test_binop_default_type(self) -> None:
		b = IRBinOp(dest=IRTemp("t0"), left=IRConst(0), op="+", right=IRConst(1))
		assert b.ir_type == IRType.INT
		assert b.is_unsigned is False

	def test_binop_long_type(self) -> None:
		b = IRBinOp(dest=IRTemp("t0"), left=IRTemp("t1"), op="+", right=IRTemp("t2"), ir_type=IRType.LONG)
		assert b.ir_type == IRType.LONG

	def test_binop_unsigned(self) -> None:
		b = IRBinOp(
			dest=IRTemp("t0"), left=IRTemp("t1"), op="/", right=IRTemp("t2"),
			ir_type=IRType.INT, is_unsigned=True,
		)
		assert b.is_unsigned is True

	def test_binop_float_type(self) -> None:
		b = IRBinOp(dest=IRTemp("t0"), left=IRTemp("t1"), op="+", right=IRTemp("t2"), ir_type=IRType.FLOAT)
		assert b.ir_type == IRType.FLOAT

	def test_binop_double_type(self) -> None:
		b = IRBinOp(dest=IRTemp("t0"), left=IRTemp("t1"), op="-", right=IRTemp("t2"), ir_type=IRType.DOUBLE)
		assert b.ir_type == IRType.DOUBLE

	def test_binop_pointer_arithmetic(self) -> None:
		b = IRBinOp(dest=IRTemp("t0"), left=IRTemp("ptr"), op="+", right=IRConst(8), ir_type=IRType.POINTER)
		assert b.ir_type == IRType.POINTER

	def test_binop_all_arithmetic_ops(self) -> None:
		for op in ["+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^"]:
			b = IRBinOp(dest=IRTemp("t0"), left=IRTemp("t1"), op=op, right=IRTemp("t2"))
			assert op in str(b)

	def test_binop_comparison_ops(self) -> None:
		for op in ["==", "!=", "<", ">", "<=", ">="]:
			b = IRBinOp(dest=IRTemp("t0"), left=IRTemp("t1"), op=op, right=IRTemp("t2"))
			assert op in str(b)


class TestIRUnaryOp:
	"""Test IRUnaryOp with different type combinations."""

	def test_unaryop_str(self) -> None:
		u = IRUnaryOp(dest=IRTemp("t1"), op="-", operand=IRTemp("t0"))
		assert str(u) == "t1 = - t0"

	def test_unaryop_bitwise_not(self) -> None:
		u = IRUnaryOp(dest=IRTemp("t1"), op="~", operand=IRTemp("t0"))
		assert str(u) == "t1 = ~ t0"

	def test_unaryop_logical_not(self) -> None:
		u = IRUnaryOp(dest=IRTemp("t1"), op="!", operand=IRTemp("t0"))
		assert str(u) == "t1 = ! t0"

	def test_unaryop_default_type(self) -> None:
		u = IRUnaryOp(dest=IRTemp("t1"), op="-", operand=IRTemp("t0"))
		assert u.ir_type == IRType.INT

	def test_unaryop_long_type(self) -> None:
		u = IRUnaryOp(dest=IRTemp("t1"), op="-", operand=IRTemp("t0"), ir_type=IRType.LONG)
		assert u.ir_type == IRType.LONG

	def test_unaryop_with_const(self) -> None:
		u = IRUnaryOp(dest=IRTemp("t1"), op="-", operand=IRConst(value=5))
		assert "5" in str(u)

	def test_unaryop_float(self) -> None:
		u = IRUnaryOp(dest=IRTemp("t1"), op="-", operand=IRTemp("t0"), ir_type=IRType.FLOAT)
		assert u.ir_type == IRType.FLOAT


class TestIRStoreLoad:
	"""Test IRStore/IRLoad with pointer types."""

	def test_load_str(self) -> None:
		ld = IRLoad(dest=IRTemp("t1"), address=IRTemp("ptr"))
		assert str(ld) == "t1 = *ptr"

	def test_store_str(self) -> None:
		st = IRStore(address=IRTemp("ptr"), value=IRConst(42))
		assert str(st) == "*ptr = 42"

	def test_load_default_type(self) -> None:
		ld = IRLoad(dest=IRTemp("t0"), address=IRTemp("ptr"))
		assert ld.ir_type == IRType.INT

	def test_store_default_type(self) -> None:
		st = IRStore(address=IRTemp("ptr"), value=IRTemp("t0"))
		assert st.ir_type == IRType.INT

	def test_load_char_type(self) -> None:
		ld = IRLoad(dest=IRTemp("t0"), address=IRTemp("ptr"), ir_type=IRType.CHAR)
		assert ld.ir_type == IRType.CHAR

	def test_store_long_type(self) -> None:
		st = IRStore(address=IRTemp("ptr"), value=IRTemp("t0"), ir_type=IRType.LONG)
		assert st.ir_type == IRType.LONG

	def test_load_pointer_type(self) -> None:
		ld = IRLoad(dest=IRTemp("t0"), address=IRTemp("ptr"), ir_type=IRType.POINTER)
		assert ld.ir_type == IRType.POINTER

	def test_store_float_type(self) -> None:
		st = IRStore(address=IRTemp("ptr"), value=IRFloatConst(3.14), ir_type=IRType.FLOAT)
		assert st.ir_type == IRType.FLOAT

	def test_load_from_global_ref(self) -> None:
		ld = IRLoad(dest=IRTemp("t0"), address=IRGlobalRef("my_var"))
		assert str(ld) == "t0 = *&my_var"

	def test_store_to_global_ref(self) -> None:
		st = IRStore(address=IRGlobalRef("my_var"), value=IRConst(100))
		assert str(st) == "*&my_var = 100"


class TestIRAddrOf:
	"""Test IRAddrOf semantics."""

	def test_addrof_str(self) -> None:
		a = IRAddrOf(dest=IRTemp("t1"), source=IRTemp("local_var"))
		assert str(a) == "t1 = &local_var"

	def test_addrof_fields(self) -> None:
		a = IRAddrOf(dest=IRTemp("t1"), source=IRTemp("x"))
		assert a.dest == IRTemp("t1")
		assert a.source == IRTemp("x")


class TestIRCall:
	"""Test IRCall with varying argument counts."""

	def test_call_no_args(self) -> None:
		c = IRCall(dest=IRTemp("t0"), function_name="foo")
		assert str(c) == "t0 = call foo()"

	def test_call_one_arg(self) -> None:
		c = IRCall(dest=IRTemp("t0"), function_name="bar", args=[IRConst(1)])
		assert str(c) == "t0 = call bar(1)"

	def test_call_many_args(self) -> None:
		args = [IRTemp(f"t{i}") for i in range(10)]
		c = IRCall(dest=IRTemp("result"), function_name="multi", args=args)
		result = str(c)
		assert "result = call multi(" in result
		for i in range(10):
			assert f"t{i}" in result

	def test_call_void_return(self) -> None:
		c = IRCall(dest=None, function_name="print_stuff", args=[IRConst(42)])
		assert str(c) == "call print_stuff(42)"

	def test_call_with_arg_types(self) -> None:
		c = IRCall(
			dest=IRTemp("t0"),
			function_name="typed_fn",
			args=[IRTemp("t1"), IRTemp("t2")],
			arg_types=[IRType.INT, IRType.FLOAT],
			return_type=IRType.DOUBLE,
		)
		assert c.arg_types == [IRType.INT, IRType.FLOAT]
		assert c.return_type == IRType.DOUBLE

	def test_call_default_return_type(self) -> None:
		c = IRCall(dest=IRTemp("t0"), function_name="fn")
		assert c.return_type == IRType.INT

	def test_call_indirect(self) -> None:
		c = IRCall(
			dest=IRTemp("t0"), function_name="", indirect=True,
			func_value=IRTemp("fptr"), args=[IRConst(1)],
		)
		assert str(c) == "t0 = call *fptr(1)"

	def test_call_indirect_no_func_value(self) -> None:
		c = IRCall(dest=IRTemp("t0"), function_name="fallback", indirect=True, func_value=None)
		assert str(c) == "t0 = call fallback()"

	def test_call_mixed_arg_types(self) -> None:
		args: list[IRValue] = [IRConst(1), IRFloatConst(2.5), IRTemp("t0"), IRGlobalRef("g")]
		c = IRCall(dest=IRTemp("r"), function_name="mixed", args=args)
		s = str(c)
		assert "1" in s
		assert "2.5" in s
		assert "t0" in s
		assert "&g" in s


class TestIROtherInstructions:
	"""Test remaining instruction types."""

	def test_ircopy_str(self) -> None:
		c = IRCopy(dest=IRTemp("t1"), source=IRTemp("t0"))
		assert str(c) == "t1 = t0"

	def test_ircopy_default_type(self) -> None:
		c = IRCopy(dest=IRTemp("t1"), source=IRConst(0))
		assert c.ir_type == IRType.INT

	def test_irlabelinstr_str(self) -> None:
		lbl = IRLabelInstr(name="L0")
		assert str(lbl) == "L0:"

	def test_irjump_str(self) -> None:
		j = IRJump(target="L_end")
		assert str(j) == "jump L_end"

	def test_ircondjump_str(self) -> None:
		cj = IRCondJump(condition=IRTemp("t0"), true_label="L_true", false_label="L_false")
		assert str(cj) == "if t0 goto L_true else goto L_false"

	def test_irreturn_with_value(self) -> None:
		r = IRReturn(value=IRConst(0))
		assert str(r) == "return 0"

	def test_irreturn_void(self) -> None:
		r = IRReturn()
		assert str(r) == "return"
		assert r.value is None

	def test_irreturn_default_type(self) -> None:
		r = IRReturn(value=IRConst(1))
		assert r.ir_type == IRType.INT

	def test_irparam_str(self) -> None:
		p = IRParam(value=IRTemp("t0"))
		assert str(p) == "param t0"

	def test_iralloc_str(self) -> None:
		a = IRAlloc(dest=IRTemp("t0"), size=16)
		assert str(a) == "t0 = alloc 16"

	def test_irinstruction_base_str_raises(self) -> None:
		instr = IRInstruction()
		with pytest.raises(NotImplementedError):
			str(instr)


class TestIRProgramStructure:
	"""Test IRFunction, IRGlobalVar, IRStringData, IRProgram."""

	def test_irfunction_str(self) -> None:
		fn = IRFunction(
			name="main",
			params=[IRTemp("arg0")],
			body=[IRReturn(value=IRConst(0))],
			return_type=IRType.INT,
			param_types=[IRType.INT],
		)
		s = str(fn)
		assert "function main(arg0) -> INT" in s
		assert "return 0" in s

	def test_irfunction_no_params(self) -> None:
		fn = IRFunction(name="nop", params=[], body=[], return_type=IRType.VOID)
		assert "function nop() -> VOID" in str(fn)

	def test_irfunction_prototype(self) -> None:
		fn = IRFunction(name="ext", params=[], body=[], return_type=IRType.INT, is_prototype=True)
		assert fn.is_prototype is True

	def test_irfunction_static(self) -> None:
		fn = IRFunction(name="helper", params=[], body=[], return_type=IRType.INT, storage_class="static")
		assert fn.storage_class == "static"

	def test_irglobalvar_str_with_init(self) -> None:
		g = IRGlobalVar(name="x", ir_type=IRType.INT, initializer=42)
		assert str(g) == "global INT x = 42"

	def test_irglobalvar_str_no_init(self) -> None:
		g = IRGlobalVar(name="y", ir_type=IRType.LONG)
		assert str(g) == "global LONG y"

	def test_irglobalvar_str_with_values(self) -> None:
		g = IRGlobalVar(name="arr", ir_type=IRType.INT, initializer_values=[1, 2, 3], total_size=12)
		assert str(g) == "global INT arr = {1, 2, 3}"

	def test_irglobalvar_static(self) -> None:
		g = IRGlobalVar(name="s", ir_type=IRType.INT, storage_class="static")
		assert g.storage_class == "static"

	def test_irglobalvar_float_initializer(self) -> None:
		g = IRGlobalVar(name="f", ir_type=IRType.FLOAT, float_initializer=1.5)
		assert g.float_initializer == 1.5

	def test_irglobalvar_string_label(self) -> None:
		g = IRGlobalVar(name="msg", ir_type=IRType.POINTER, string_label=".LC0")
		assert g.string_label == ".LC0"

	def test_irstringdata_str(self) -> None:
		s = IRStringData(label=".LC0", value="hello")
		assert str(s) == '.LC0: .string "hello"'

	def test_irprogram_str_empty(self) -> None:
		p = IRProgram()
		assert str(p) == ""

	def test_irprogram_str_full(self) -> None:
		p = IRProgram(
			functions=[IRFunction(name="main", params=[], body=[IRReturn(value=IRConst(0))], return_type=IRType.INT)],
			globals=[IRGlobalVar(name="x", ir_type=IRType.INT, initializer=0)],
			string_data=[IRStringData(label=".LC0", value="hi")],
		)
		s = str(p)
		assert "global INT x = 0" in s
		assert '.LC0: .string "hi"' in s
		assert "function main" in s
