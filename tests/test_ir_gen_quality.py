"""Tests that the IR generator produces efficient IR with minimal redundant temporaries."""

from compiler.ast_nodes import (
	Assignment,
	BinaryOp,
	CompoundAssignment,
	CompoundStmt,
	ExprStmt,
	FunctionDecl,
	Identifier,
	IfStmt,
	IntLiteral,
	ParamDecl,
	Program,
	ReturnStmt,
	TypeSpec,
	UnaryOp,
	VarDecl,
	WhileStmt,
)
from compiler.ir import (
	IRAlloc,
	IRBinOp,
	IRConst,
	IRCopy,
	IRReturn,
	IRTemp,
	IRUnaryOp,
)
from compiler.ir_gen import IRGenerator


def _int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def _void_type() -> TypeSpec:
	return TypeSpec(base_type="void")


def _make_program(*decls) -> Program:
	return Program(declarations=list(decls))


def _count_type(body: list, cls: type) -> int:
	return sum(1 for i in body if isinstance(i, cls))


def _count_temps(body: list) -> int:
	"""Count distinct IRTemp names used as destinations in the body."""
	temps = set()
	for instr in body:
		if hasattr(instr, "dest") and isinstance(instr.dest, IRTemp):
			temps.add(instr.dest.name)
	return len(temps)


# ------------------------------------------------------------------
# Constant assignment: int x = 5; should not produce intermediate copies
# ------------------------------------------------------------------

class TestConstantAssignment:
	def test_var_init_from_literal_no_extra_copy(self) -> None:
		"""int f() { int x = 5; return x; }
		Should produce: alloc x, copy 5->x, return x (no intermediate temp for 5)
		"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=5)),
						ReturnStmt(expression=Identifier(name="x")),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# alloc + copy(5->x) + return(x) = 3 instructions
		assert len(body) == 3
		assert isinstance(body[0], IRAlloc)
		assert isinstance(body[1], IRCopy)
		assert body[1].source == IRConst(5)
		# Return should use x's temp directly, no intermediate copy
		assert isinstance(body[2], IRReturn)
		assert body[2].value == body[0].dest

	def test_reassign_from_literal(self) -> None:
		"""int f() { int x = 0; x = 42; return x; }
		Assignment from literal should be direct IRCopy(x, 42).
		"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=0)),
						ExprStmt(expression=Assignment(
							target=Identifier(name="x"),
							value=IntLiteral(value=42),
						)),
						ReturnStmt(expression=Identifier(name="x")),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# alloc + copy(0->x) + copy(42->x) + return(x) = 4 instructions
		assert len(body) == 4
		copies = [i for i in body if isinstance(i, IRCopy)]
		assert copies[1].source == IRConst(42)


# ------------------------------------------------------------------
# Variable-to-variable assignment: x = y should not create intermediate temp
# ------------------------------------------------------------------

class TestVarToVarAssignment:
	def test_assign_local_to_local(self) -> None:
		"""int f() { int x = 1; int y = 0; y = x; return y; }
		y = x should be a single IRCopy(y, x), not IRCopy(tmp, x) + IRCopy(y, tmp).
		"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=1)),
						VarDecl(type_spec=_int_type(), name="y", initializer=IntLiteral(value=0)),
						ExprStmt(expression=Assignment(
							target=Identifier(name="y"),
							value=Identifier(name="x"),
						)),
						ReturnStmt(expression=Identifier(name="y")),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		x_temp = body[0].dest  # alloc for x
		y_temp = body[2].dest  # alloc for y
		# Find the assignment y = x
		assign_copies = [i for i in body if isinstance(i, IRCopy) and i.dest == y_temp and i.source == x_temp]
		# Should have a direct copy from x to y (no intermediate temp)
		assert len(assign_copies) == 1

	def test_init_from_param(self) -> None:
		"""int f(int a) { int x = a; return x; }
		Initialization from param should be direct copy, no intermediate.
		"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[ParamDecl(type_spec=_int_type(), name="a")],
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="x", initializer=Identifier(name="a")),
						ReturnStmt(expression=Identifier(name="x")),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		fn = ir.functions[0]
		body = fn.body
		param_temp = fn.params[0]
		# alloc for x, copy(a->x), return(x) = 3 instructions
		assert len(body) == 3
		assert isinstance(body[1], IRCopy)
		assert body[1].source == param_temp


# ------------------------------------------------------------------
# Binary ops with simple operands should not create intermediate copies
# ------------------------------------------------------------------

class TestBinaryOpTemps:
	def test_add_two_params(self) -> None:
		"""int f(int a, int b) { return a + b; }
		Should be: binop(t2, t0, +, t1) + return(t2). No copies of params.
		"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[
					ParamDecl(type_spec=_int_type(), name="a"),
					ParamDecl(type_spec=_int_type(), name="b"),
				],
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=BinaryOp(
								left=Identifier(name="a"),
								op="+",
								right=Identifier(name="b"),
							)
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		fn = ir.functions[0]
		body = fn.body
		# binop + return = 2 instructions, no copies
		assert len(body) == 2
		assert isinstance(body[0], IRBinOp)
		assert body[0].left == fn.params[0]
		assert body[0].right == fn.params[1]
		assert _count_type(body, IRCopy) == 0

	def test_add_param_and_literal(self) -> None:
		"""int f(int x) { return x + 1; }
		Should use param directly in binop.
		"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[ParamDecl(type_spec=_int_type(), name="x")],
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=BinaryOp(
								left=Identifier(name="x"),
								op="+",
								right=IntLiteral(value=1),
							)
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		fn = ir.functions[0]
		body = fn.body
		assert len(body) == 2
		assert isinstance(body[0], IRBinOp)
		assert body[0].left == fn.params[0]
		assert body[0].right == IRConst(1)

	def test_chained_add(self) -> None:
		"""int f(int a, int b, int c) { return a + b + c; }
		Two binops, no intermediate copies of params.
		"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[
					ParamDecl(type_spec=_int_type(), name="a"),
					ParamDecl(type_spec=_int_type(), name="b"),
					ParamDecl(type_spec=_int_type(), name="c"),
				],
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=BinaryOp(
								left=BinaryOp(
									left=Identifier(name="a"),
									op="+",
									right=Identifier(name="b"),
								),
								op="+",
								right=Identifier(name="c"),
							)
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		fn = ir.functions[0]
		body = fn.body
		# binop(a+b) + binop(result+c) + return = 3 instructions
		assert len(body) == 3
		assert _count_type(body, IRBinOp) == 2
		assert _count_type(body, IRCopy) == 0


# ------------------------------------------------------------------
# Unary ops: constant folding for simple cases
# ------------------------------------------------------------------

class TestUnaryConstantFolding:
	def test_negate_literal(self) -> None:
		"""return -5 should constant-fold to IRConst(-5), no IRUnaryOp."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=UnaryOp(op="-", operand=IntLiteral(value=5))
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		assert len(body) == 1
		assert isinstance(body[0], IRReturn)
		assert body[0].value == IRConst(-5)

	def test_bitwise_not_literal(self) -> None:
		"""return ~0 should constant-fold to IRConst(-1)."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=UnaryOp(op="~", operand=IntLiteral(value=0))
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		assert len(body) == 1
		assert isinstance(body[0], IRReturn)
		assert body[0].value == IRConst(-1)

	def test_logical_not_literal(self) -> None:
		"""return !0 should constant-fold to IRConst(1)."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=UnaryOp(op="!", operand=IntLiteral(value=0))
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		assert len(body) == 1
		assert isinstance(body[0], IRReturn)
		assert body[0].value == IRConst(1)

	def test_negate_param_no_copy(self) -> None:
		"""int f(int x) { return -x; }
		Unary op should use param directly, no intermediate copy.
		"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[ParamDecl(type_spec=_int_type(), name="x")],
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=UnaryOp(op="-", operand=Identifier(name="x"))
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		fn = ir.functions[0]
		body = fn.body
		# unary_op + return = 2 instructions
		assert len(body) == 2
		assert isinstance(body[0], IRUnaryOp)
		assert body[0].operand == fn.params[0]
		assert _count_type(body, IRCopy) == 0


# ------------------------------------------------------------------
# Return statement: should use simple values directly
# ------------------------------------------------------------------

class TestReturnOptimization:
	def test_return_literal(self) -> None:
		"""int f() { return 42; } -- 1 instruction only."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[ReturnStmt(expression=IntLiteral(value=42))]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		assert len(body) == 1
		assert isinstance(body[0], IRReturn)
		assert body[0].value == IRConst(42)

	def test_return_param(self) -> None:
		"""int f(int x) { return x; } -- uses param directly."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[ParamDecl(type_spec=_int_type(), name="x")],
				body=CompoundStmt(
					statements=[ReturnStmt(expression=Identifier(name="x"))]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		fn = ir.functions[0]
		body = fn.body
		assert len(body) == 1
		assert isinstance(body[0], IRReturn)
		assert body[0].value == fn.params[0]


# ------------------------------------------------------------------
# Compound assignment should use simple RHS directly
# ------------------------------------------------------------------

class TestCompoundAssignmentQuality:
	def test_plus_equals_literal(self) -> None:
		"""int f(int x) { int y = 0; y += x; return y; }
		The RHS 'x' should be the param temp directly, no intermediate copy.
		"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[ParamDecl(type_spec=_int_type(), name="x")],
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="y", initializer=IntLiteral(value=0)),
						ExprStmt(expression=CompoundAssignment(
							target=Identifier(name="y"),
							op="+=",
							value=Identifier(name="x"),
						)),
						ReturnStmt(expression=Identifier(name="y")),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		fn = ir.functions[0]
		body = fn.body
		binops = [i for i in body if isinstance(i, IRBinOp)]
		assert len(binops) == 1
		# The RHS of y += x should use x's param temp directly
		assert binops[0].right == fn.params[0]


# ------------------------------------------------------------------
# Condition expressions in control flow should use simple values directly
# ------------------------------------------------------------------

class TestConditionOptimization:
	def test_if_param_condition(self) -> None:
		"""int f(int x) { if (x) return 1; return 0; }
		Condition should use param directly without copy.
		"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[ParamDecl(type_spec=_int_type(), name="x")],
				body=CompoundStmt(
					statements=[
						IfStmt(
							condition=Identifier(name="x"),
							then_branch=ReturnStmt(expression=IntLiteral(value=1)),
							else_branch=None,
						),
						ReturnStmt(expression=IntLiteral(value=0)),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# No IRCopy should be emitted just for the condition
		assert _count_type(body, IRCopy) == 0

	def test_while_param_condition(self) -> None:
		"""void f(int n) { while (n) { n = n - 1; } }
		Initial condition check should use param directly.
		"""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				params=[ParamDecl(type_spec=_int_type(), name="n")],
				body=CompoundStmt(
					statements=[
						WhileStmt(
							condition=Identifier(name="n"),
							body=CompoundStmt(statements=[
								ExprStmt(expression=Assignment(
									target=Identifier(name="n"),
									value=BinaryOp(
										left=Identifier(name="n"),
										op="-",
										right=IntLiteral(value=1),
									),
								)),
							]),
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# The while condition should not generate a copy just to test the param
		copies = [i for i in body if isinstance(i, IRCopy)]
		# Only the assignment n = n - 1 should produce a copy (result -> n)
		assert len(copies) == 1


# ------------------------------------------------------------------
# Temp count comparisons: overall program efficiency
# ------------------------------------------------------------------

class TestOverallTempReduction:
	def test_simple_program_temp_count(self) -> None:
		"""int f(int a, int b) { int c = a + b; return c; }
		Should need exactly 2 temps: a(param), b(param), c(alloc), result(binop).
		"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[
					ParamDecl(type_spec=_int_type(), name="a"),
					ParamDecl(type_spec=_int_type(), name="b"),
				],
				body=CompoundStmt(
					statements=[
						VarDecl(
							type_spec=_int_type(),
							name="c",
							initializer=BinaryOp(
								left=Identifier(name="a"),
								op="+",
								right=Identifier(name="b"),
							),
						),
						ReturnStmt(expression=Identifier(name="c")),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# alloc c, binop(a+b)->tmp, copy(tmp->c), return(c) = 4 instructions
		assert len(body) == 4
		# Only 2 dest temps: c (from alloc) and the binop result
		assert _count_temps(body) == 2

	def test_no_intermediate_for_identity_copy(self) -> None:
		"""int f(int x) { int y = x; int z = y; return z; }
		Each init should be a single direct copy, no intermediates.
		"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[ParamDecl(type_spec=_int_type(), name="x")],
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="y", initializer=Identifier(name="x")),
						VarDecl(type_spec=_int_type(), name="z", initializer=Identifier(name="y")),
						ReturnStmt(expression=Identifier(name="z")),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		fn = ir.functions[0]
		body = fn.body
		# alloc y, copy(x->y), alloc z, copy(y->z), return(z)
		assert len(body) == 5
		copies = [i for i in body if isinstance(i, IRCopy)]
		assert len(copies) == 2
		# First copy: x (param) -> y
		assert copies[0].source == fn.params[0]
		# Second copy: y -> z
		assert copies[1].source == copies[0].dest
