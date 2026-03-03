"""Tests for the AST-to-IR lowering pass."""

from compiler.ast_nodes import (
	Assignment,
	BinaryOp,
	CompoundStmt,
	ExprStmt,
	ForStmt,
	FunctionCall,
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
	IRCall,
	IRCondJump,
	IRConst,
	IRCopy,
	IRJump,
	IRLabelInstr,
	IRParam,
	IRReturn,
	IRType,
	IRUnaryOp,
)
from compiler.ir_gen import IRGenerator


def _int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def _void_type() -> TypeSpec:
	return TypeSpec(base_type="void")


def _make_program(*decls) -> Program:
	return Program(declarations=list(decls))


# ------------------------------------------------------------------
# Basic function lowering
# ------------------------------------------------------------------


class TestFunctionDecl:
	def test_empty_function(self) -> None:
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="main",
				params=[],
				body=CompoundStmt(statements=[]),
			)
		)
		ir = IRGenerator().generate(prog)
		assert len(ir.functions) == 1
		fn = ir.functions[0]
		assert fn.name == "main"
		assert fn.params == []
		assert fn.return_type == IRType.VOID
		assert fn.body == []

	def test_function_with_params(self) -> None:
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="add",
				params=[
					ParamDecl(type_spec=_int_type(), name="a"),
					ParamDecl(type_spec=_int_type(), name="b"),
				],
				body=CompoundStmt(statements=[]),
			)
		)
		ir = IRGenerator().generate(prog)
		fn = ir.functions[0]
		assert len(fn.params) == 2
		assert fn.return_type == IRType.INT

	def test_multiple_functions(self) -> None:
		prog = _make_program(
			FunctionDecl(return_type=_void_type(), name="f1", body=CompoundStmt()),
			FunctionDecl(return_type=_void_type(), name="f2", body=CompoundStmt()),
		)
		ir = IRGenerator().generate(prog)
		assert len(ir.functions) == 2
		assert ir.functions[0].name == "f1"
		assert ir.functions[1].name == "f2"


# ------------------------------------------------------------------
# Return statements
# ------------------------------------------------------------------


class TestReturn:
	def test_return_literal(self) -> None:
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[ReturnStmt(expression=IntLiteral(value=42))]),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		assert len(body) == 1
		assert isinstance(body[0], IRReturn)
		assert body[0].value == IRConst(42)

	def test_return_expression(self) -> None:
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=BinaryOp(
								left=IntLiteral(value=1), op="+", right=IntLiteral(value=2)
							)
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		assert isinstance(body[0], IRBinOp)
		assert body[0].op == "+"
		assert body[0].left == IRConst(1)
		assert body[0].right == IRConst(2)
		assert isinstance(body[1], IRReturn)
		assert body[1].value == body[0].dest


# ------------------------------------------------------------------
# Variable declarations
# ------------------------------------------------------------------


class TestVarDecl:
	def test_var_without_init(self) -> None:
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[VarDecl(type_spec=_int_type(), name="x")]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		assert len(body) == 1
		assert isinstance(body[0], IRAlloc)
		assert body[0].size == 4

	def test_var_with_init(self) -> None:
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(
							type_spec=_int_type(),
							name="x",
							initializer=IntLiteral(value=10),
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		assert len(body) == 2
		assert isinstance(body[0], IRAlloc)
		assert isinstance(body[1], IRCopy)
		assert body[1].dest == body[0].dest
		assert body[1].source == IRConst(10)

	def test_pointer_var(self) -> None:
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(
							type_spec=TypeSpec(base_type="int", pointer_count=1),
							name="p",
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		alloc = ir.functions[0].body[0]
		assert isinstance(alloc, IRAlloc)
		assert alloc.size == 8


# ------------------------------------------------------------------
# Expressions
# ------------------------------------------------------------------


class TestExpressions:
	def test_binary_op(self) -> None:
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=BinaryOp(
								left=IntLiteral(value=3), op="*", right=IntLiteral(value=4)
							)
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		assert isinstance(body[0], IRBinOp)
		assert body[0].op == "*"

	def test_unary_op(self) -> None:
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
		assert isinstance(body[0], IRUnaryOp)
		assert body[0].op == "-"
		assert body[0].operand == IRConst(5)

	def test_identifier_lookup(self) -> None:
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
		assert isinstance(body[0], IRCopy)
		assert body[0].source == fn.params[0]

	def test_nested_binary(self) -> None:
		# (1 + 2) * 3
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=BinaryOp(
								left=BinaryOp(
									left=IntLiteral(value=1),
									op="+",
									right=IntLiteral(value=2),
								),
								op="*",
								right=IntLiteral(value=3),
							)
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		assert isinstance(body[0], IRBinOp)
		assert body[0].op == "+"
		assert isinstance(body[1], IRBinOp)
		assert body[1].op == "*"
		assert body[1].left == body[0].dest


# ------------------------------------------------------------------
# Assignment
# ------------------------------------------------------------------


class TestAssignment:
	def test_assign_to_var(self) -> None:
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="x"),
						ExprStmt(
							expression=Assignment(
								target=Identifier(name="x"),
								value=IntLiteral(value=7),
							)
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		alloc = body[0]
		assert isinstance(alloc, IRAlloc)
		copy = body[1]
		assert isinstance(copy, IRCopy)
		assert copy.dest == alloc.dest
		assert copy.source == IRConst(7)


# ------------------------------------------------------------------
# Function calls
# ------------------------------------------------------------------


class TestFunctionCall:
	def test_simple_call(self) -> None:
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ExprStmt(
							expression=FunctionCall(
								name="foo", arguments=[IntLiteral(value=1)]
							)
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		assert isinstance(body[0], IRParam)
		assert body[0].value == IRConst(1)
		assert isinstance(body[1], IRCall)
		assert body[1].function_name == "foo"

	def test_call_multiple_args(self) -> None:
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=FunctionCall(
								name="add",
								arguments=[IntLiteral(value=1), IntLiteral(value=2)],
							)
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		assert isinstance(body[0], IRParam)
		assert isinstance(body[1], IRParam)
		assert isinstance(body[2], IRCall)
		assert isinstance(body[3], IRReturn)


# ------------------------------------------------------------------
# Control flow
# ------------------------------------------------------------------


class TestIfStmt:
	def test_if_without_else(self) -> None:
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						IfStmt(
							condition=IntLiteral(value=1),
							then_branch=CompoundStmt(
								statements=[
									ReturnStmt(expression=IntLiteral(value=0))
								]
							),
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		assert isinstance(body[0], IRCondJump)
		assert isinstance(body[1], IRLabelInstr)
		assert body[1].name == body[0].true_label
		# then body
		assert isinstance(body[2], IRReturn)
		# end label
		assert isinstance(body[3], IRLabelInstr)
		assert body[3].name == body[0].false_label

	def test_if_with_else(self) -> None:
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						IfStmt(
							condition=IntLiteral(value=1),
							then_branch=CompoundStmt(
								statements=[
									ReturnStmt(expression=IntLiteral(value=1))
								]
							),
							else_branch=CompoundStmt(
								statements=[
									ReturnStmt(expression=IntLiteral(value=0))
								]
							),
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		cond_jump = body[0]
		assert isinstance(cond_jump, IRCondJump)
		# then label
		assert isinstance(body[1], IRLabelInstr)
		assert body[1].name == cond_jump.true_label
		# then body + jump to end
		assert isinstance(body[2], IRReturn)
		assert isinstance(body[3], IRJump)
		# else label
		assert isinstance(body[4], IRLabelInstr)
		assert body[4].name == cond_jump.false_label
		# else body
		assert isinstance(body[5], IRReturn)
		# end label
		assert isinstance(body[6], IRLabelInstr)
		assert body[6].name == body[3].target


class TestWhileStmt:
	def test_while_loop(self) -> None:
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						WhileStmt(
							condition=IntLiteral(value=1),
							body=CompoundStmt(statements=[]),
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# start label
		assert isinstance(body[0], IRLabelInstr)
		start_label = body[0].name
		# cond jump
		assert isinstance(body[1], IRCondJump)
		# body label
		assert isinstance(body[2], IRLabelInstr)
		assert body[2].name == body[1].true_label
		# jump back to start
		assert isinstance(body[3], IRJump)
		assert body[3].target == start_label
		# end label
		assert isinstance(body[4], IRLabelInstr)
		assert body[4].name == body[1].false_label


class TestForStmt:
	def test_for_loop(self) -> None:
		# for (i = 0; i < 10; i = i + 1) {}
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="i", initializer=IntLiteral(value=0)),
						ForStmt(
							init=ExprStmt(
								expression=Assignment(
									target=Identifier(name="i"),
									value=IntLiteral(value=0),
								)
							),
							condition=BinaryOp(
								left=Identifier(name="i"),
								op="<",
								right=IntLiteral(value=10),
							),
							update=ExprStmt(
								expression=Assignment(
									target=Identifier(name="i"),
									value=BinaryOp(
										left=Identifier(name="i"),
										op="+",
										right=IntLiteral(value=1),
									),
								)
							),
							body=CompoundStmt(statements=[]),
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# Should contain alloc, copy (var init), then copy (for init assignment),
		# start label, condition evaluation, cond jump, body label,
		# update, jump back, end label
		labels = [i for i in body if isinstance(i, IRLabelInstr)]
		jumps = [i for i in body if isinstance(i, IRJump)]
		cond_jumps = [i for i in body if isinstance(i, IRCondJump)]
		assert len(labels) == 3  # start, body, end
		assert len(jumps) == 1
		assert len(cond_jumps) == 1

	def test_for_no_condition(self) -> None:
		# for (;;) {}  -- infinite loop
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ForStmt(
							body=CompoundStmt(statements=[]),
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		cond_jumps = [i for i in body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) == 0
		jumps = [i for i in body if isinstance(i, IRJump)]
		assert len(jumps) == 1


# ------------------------------------------------------------------
# Integration: full function lowering
# ------------------------------------------------------------------


class TestIntegration:
	def test_return_param_plus_literal(self) -> None:
		"""int f(int x) { return x + 1; }"""
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
		assert fn.name == "f"
		assert fn.return_type == IRType.INT
		# copy from param, binop, return
		assert len(fn.body) == 3
		assert isinstance(fn.body[0], IRCopy)
		assert isinstance(fn.body[1], IRBinOp)
		assert isinstance(fn.body[2], IRReturn)
		assert fn.body[1].left == fn.body[0].dest
		assert fn.body[1].right == IRConst(1)
		assert fn.body[2].value == fn.body[1].dest

	def test_var_decl_and_assign_then_return(self) -> None:
		"""int f() { int x = 5; x = x + 1; return x; }"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=5)),
						ExprStmt(
							expression=Assignment(
								target=Identifier(name="x"),
								value=BinaryOp(
									left=Identifier(name="x"),
									op="+",
									right=IntLiteral(value=1),
								),
							)
						),
						ReturnStmt(expression=Identifier(name="x")),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		fn = ir.functions[0]
		assert fn.name == "f"
		# alloc, copy(5), copy(x->t), binop(+1), copy(result->x), copy(x->t), return
		allocs = [i for i in fn.body if isinstance(i, IRAlloc)]
		assert len(allocs) == 1
		returns = [i for i in fn.body if isinstance(i, IRReturn)]
		assert len(returns) == 1
