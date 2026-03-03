"""Tests for IR definitions: construction and string representation."""

from compiler.ir import (
	IRAlloc,
	IRBinOp,
	IRCall,
	IRCondJump,
	IRConst,
	IRCopy,
	IRFunction,
	IRInstruction,
	IRJump,
	IRLabel,
	IRLabelInstr,
	IRLoad,
	IRParam,
	IRProgram,
	IRReturn,
	IRStore,
	IRTemp,
	IRType,
	IRUnaryOp,
	IRValue,
)


# ---------------------------------------------------------------------------
# IRType
# ---------------------------------------------------------------------------

class TestIRType:
	def test_enum_members(self) -> None:
		assert IRType.INT is not None
		assert IRType.CHAR is not None
		assert IRType.VOID is not None
		assert IRType.POINTER is not None

	def test_enum_names(self) -> None:
		assert IRType.INT.name == "INT"
		assert IRType.CHAR.name == "CHAR"
		assert IRType.VOID.name == "VOID"
		assert IRType.POINTER.name == "POINTER"


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

class TestIRConst:
	def test_construction(self) -> None:
		c = IRConst(42)
		assert c.value == 42

	def test_str(self) -> None:
		assert str(IRConst(0)) == "0"
		assert str(IRConst(-1)) == "-1"
		assert str(IRConst(999)) == "999"

	def test_is_irvalue(self) -> None:
		assert isinstance(IRConst(1), IRValue)

	def test_frozen(self) -> None:
		c = IRConst(10)
		try:
			c.value = 20  # type: ignore[misc]
			assert False, "Should be frozen"
		except AttributeError:
			pass

	def test_equality(self) -> None:
		assert IRConst(5) == IRConst(5)
		assert IRConst(5) != IRConst(6)


class TestIRTemp:
	def test_construction(self) -> None:
		t = IRTemp("t0")
		assert t.name == "t0"

	def test_str(self) -> None:
		assert str(IRTemp("t1")) == "t1"
		assert str(IRTemp("result")) == "result"

	def test_is_irvalue(self) -> None:
		assert isinstance(IRTemp("x"), IRValue)

	def test_equality(self) -> None:
		assert IRTemp("t0") == IRTemp("t0")
		assert IRTemp("t0") != IRTemp("t1")


class TestIRLabel:
	def test_construction(self) -> None:
		lbl = IRLabel("L0")
		assert lbl.name == "L0"

	def test_str(self) -> None:
		assert str(IRLabel("loop_start")) == "loop_start"

	def test_is_irvalue(self) -> None:
		assert isinstance(IRLabel("L0"), IRValue)


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------

class TestIRBinOp:
	def test_construction(self) -> None:
		op = IRBinOp(IRTemp("t0"), IRTemp("a"), "+", IRTemp("b"))
		assert op.dest == IRTemp("t0")
		assert op.left == IRTemp("a")
		assert op.op == "+"
		assert op.right == IRTemp("b")

	def test_str(self) -> None:
		op = IRBinOp(IRTemp("t0"), IRTemp("a"), "+", IRConst(1))
		assert str(op) == "t0 = a + 1"

	def test_str_multiply(self) -> None:
		op = IRBinOp(IRTemp("t1"), IRConst(3), "*", IRTemp("x"))
		assert str(op) == "t1 = 3 * x"

	def test_is_instruction(self) -> None:
		assert isinstance(IRBinOp(IRTemp("t0"), IRConst(1), "+", IRConst(2)), IRInstruction)


class TestIRUnaryOp:
	def test_construction(self) -> None:
		op = IRUnaryOp(IRTemp("t0"), "-", IRTemp("x"))
		assert op.dest == IRTemp("t0")
		assert op.op == "-"
		assert op.operand == IRTemp("x")

	def test_str(self) -> None:
		assert str(IRUnaryOp(IRTemp("t0"), "-", IRConst(5))) == "t0 = - 5"
		assert str(IRUnaryOp(IRTemp("t1"), "!", IRTemp("flag"))) == "t1 = ! flag"


class TestIRCopy:
	def test_construction(self) -> None:
		cp = IRCopy(IRTemp("t0"), IRConst(42))
		assert cp.dest == IRTemp("t0")
		assert cp.source == IRConst(42)

	def test_str(self) -> None:
		assert str(IRCopy(IRTemp("x"), IRTemp("y"))) == "x = y"
		assert str(IRCopy(IRTemp("t0"), IRConst(0))) == "t0 = 0"


class TestIRLoad:
	def test_construction(self) -> None:
		ld = IRLoad(IRTemp("t0"), IRTemp("ptr"))
		assert ld.dest == IRTemp("t0")
		assert ld.address == IRTemp("ptr")

	def test_str(self) -> None:
		assert str(IRLoad(IRTemp("val"), IRTemp("addr"))) == "val = *addr"


class TestIRStore:
	def test_construction(self) -> None:
		st = IRStore(IRTemp("ptr"), IRConst(10))
		assert st.address == IRTemp("ptr")
		assert st.value == IRConst(10)

	def test_str(self) -> None:
		assert str(IRStore(IRTemp("p"), IRTemp("v"))) == "*p = v"


class TestIRLabelInstr:
	def test_construction(self) -> None:
		lbl = IRLabelInstr("L0")
		assert lbl.name == "L0"

	def test_str(self) -> None:
		assert str(IRLabelInstr("loop_start")) == "loop_start:"

	def test_is_instruction(self) -> None:
		assert isinstance(IRLabelInstr("L0"), IRInstruction)


class TestIRJump:
	def test_construction(self) -> None:
		j = IRJump("L1")
		assert j.target == "L1"

	def test_str(self) -> None:
		assert str(IRJump("end")) == "jump end"


class TestIRCondJump:
	def test_construction(self) -> None:
		cj = IRCondJump(IRTemp("cond"), "L_true", "L_false")
		assert cj.condition == IRTemp("cond")
		assert cj.true_label == "L_true"
		assert cj.false_label == "L_false"

	def test_str(self) -> None:
		cj = IRCondJump(IRTemp("t0"), "then", "else")
		assert str(cj) == "if t0 goto then else goto else"


class TestIRCall:
	def test_construction_with_dest(self) -> None:
		call = IRCall(IRTemp("ret"), "printf", [IRTemp("fmt"), IRConst(42)])
		assert call.dest == IRTemp("ret")
		assert call.function_name == "printf"
		assert call.args == [IRTemp("fmt"), IRConst(42)]

	def test_construction_no_dest(self) -> None:
		call = IRCall(None, "exit", [IRConst(0)])
		assert call.dest is None

	def test_str_with_dest(self) -> None:
		call = IRCall(IRTemp("r"), "foo", [IRTemp("a"), IRTemp("b")])
		assert str(call) == "r = call foo(a, b)"

	def test_str_no_dest(self) -> None:
		call = IRCall(None, "bar", [])
		assert str(call) == "call bar()"

	def test_str_no_args(self) -> None:
		call = IRCall(IRTemp("t0"), "getchar", [])
		assert str(call) == "t0 = call getchar()"


class TestIRReturn:
	def test_construction_with_value(self) -> None:
		ret = IRReturn(IRConst(0))
		assert ret.value == IRConst(0)

	def test_construction_no_value(self) -> None:
		ret = IRReturn()
		assert ret.value is None

	def test_str_with_value(self) -> None:
		assert str(IRReturn(IRTemp("result"))) == "return result"

	def test_str_no_value(self) -> None:
		assert str(IRReturn()) == "return"


class TestIRParam:
	def test_construction(self) -> None:
		p = IRParam(IRTemp("arg0"))
		assert p.value == IRTemp("arg0")

	def test_str(self) -> None:
		assert str(IRParam(IRConst(5))) == "param 5"
		assert str(IRParam(IRTemp("x"))) == "param x"


class TestIRAlloc:
	def test_construction(self) -> None:
		a = IRAlloc(IRTemp("arr"), 40)
		assert a.dest == IRTemp("arr")
		assert a.size == 40

	def test_str(self) -> None:
		assert str(IRAlloc(IRTemp("buf"), 256)) == "buf = alloc 256"


# ---------------------------------------------------------------------------
# Program structure
# ---------------------------------------------------------------------------

class TestIRFunction:
	def test_construction(self) -> None:
		func = IRFunction(
			name="main",
			params=[],
			body=[IRReturn(IRConst(0))],
			return_type=IRType.INT,
		)
		assert func.name == "main"
		assert func.params == []
		assert len(func.body) == 1
		assert func.return_type == IRType.INT

	def test_str_simple(self) -> None:
		func = IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)
		text = str(func)
		assert "function main() -> INT" in text
		assert "  return 0" in text

	def test_str_with_params(self) -> None:
		func = IRFunction(
			"add",
			[IRTemp("a"), IRTemp("b")],
			[
				IRBinOp(IRTemp("t0"), IRTemp("a"), "+", IRTemp("b")),
				IRReturn(IRTemp("t0")),
			],
			IRType.INT,
		)
		text = str(func)
		assert "function add(a, b) -> INT" in text
		assert "  t0 = a + b" in text
		assert "  return t0" in text

	def test_str_void(self) -> None:
		func = IRFunction("noop", [], [IRReturn()], IRType.VOID)
		assert "-> VOID" in str(func)


class TestIRProgram:
	def test_construction_empty(self) -> None:
		prog = IRProgram()
		assert prog.functions == []

	def test_construction_with_functions(self) -> None:
		f1 = IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)
		f2 = IRFunction("helper", [], [IRReturn()], IRType.VOID)
		prog = IRProgram([f1, f2])
		assert len(prog.functions) == 2

	def test_str_multiple_functions(self) -> None:
		f1 = IRFunction("main", [], [IRReturn(IRConst(0))], IRType.INT)
		f2 = IRFunction("noop", [], [IRReturn()], IRType.VOID)
		prog = IRProgram([f1, f2])
		text = str(prog)
		assert "function main" in text
		assert "function noop" in text
		# Functions separated by blank line
		assert "\n\n" in text

	def test_str_empty_program(self) -> None:
		assert str(IRProgram()) == ""


# ---------------------------------------------------------------------------
# Integration: build a small program
# ---------------------------------------------------------------------------

class TestIntegration:
	def test_add_function_program(self) -> None:
		"""Build an IR program for: int add(int a, int b) { return a + b; }"""
		body = [
			IRBinOp(IRTemp("t0"), IRTemp("a"), "+", IRTemp("b")),
			IRReturn(IRTemp("t0")),
		]
		func = IRFunction("add", [IRTemp("a"), IRTemp("b")], body, IRType.INT)
		prog = IRProgram([func])

		text = str(prog)
		assert "function add(a, b) -> INT" in text
		assert "t0 = a + b" in text
		assert "return t0" in text

	def test_loop_with_labels(self) -> None:
		"""Build IR for a simple loop: while (i < 10) { i = i + 1; }"""
		body: list[IRInstruction] = [
			IRCopy(IRTemp("i"), IRConst(0)),
			IRLabelInstr("loop"),
			IRBinOp(IRTemp("t0"), IRTemp("i"), "<", IRConst(10)),
			IRCondJump(IRTemp("t0"), "body", "end"),
			IRLabelInstr("body"),
			IRBinOp(IRTemp("i"), IRTemp("i"), "+", IRConst(1)),
			IRJump("loop"),
			IRLabelInstr("end"),
			IRReturn(IRTemp("i")),
		]
		func = IRFunction("count", [], body, IRType.INT)
		text = str(func)
		assert "loop:" in text
		assert "if t0 goto body else goto end" in text
		assert "jump loop" in text
		assert "end:" in text

	def test_function_call_with_params(self) -> None:
		"""Build IR for: x = add(1, 2);"""
		body: list[IRInstruction] = [
			IRParam(IRConst(1)),
			IRParam(IRConst(2)),
			IRCall(IRTemp("x"), "add", [IRConst(1), IRConst(2)]),
			IRReturn(IRTemp("x")),
		]
		func = IRFunction("main", [], body, IRType.INT)
		text = str(func)
		assert "param 1" in text
		assert "param 2" in text
		assert "x = call add(1, 2)" in text

	def test_pointer_operations(self) -> None:
		"""Build IR for pointer load/store."""
		body: list[IRInstruction] = [
			IRAlloc(IRTemp("p"), 4),
			IRStore(IRTemp("p"), IRConst(42)),
			IRLoad(IRTemp("val"), IRTemp("p")),
			IRReturn(IRTemp("val")),
		]
		func = IRFunction("ptr_test", [], body, IRType.INT)
		text = str(func)
		assert "p = alloc 4" in text
		assert "*p = 42" in text
		assert "val = *p" in text
