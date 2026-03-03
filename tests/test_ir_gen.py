"""Tests for the AST-to-IR lowering pass."""

from compiler.ast_nodes import (
	ArraySubscript,
	Assignment,
	BinaryOp,
	BreakStmt,
	CaseClause,
	CompoundAssignment,
	CompoundStmt,
	ContinueStmt,
	DoWhileStmt,
	ExprStmt,
	ForStmt,
	FunctionCall,
	FunctionDecl,
	Identifier,
	IfStmt,
	IntLiteral,
	MemberAccess,
	ParamDecl,
	PostfixExpr,
	Program,
	ReturnStmt,
	SizeofExpr,
	StructDecl,
	StructMember,
	SwitchStmt,
	TernaryExpr,
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
	IRLoad,
	IRParam,
	IRReturn,
	IRStore,
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
		assert len(labels) == 4  # start, body, update, end
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


class TestArrayIR:
	def test_array_decl_alloc_size(self) -> None:
		"""int arr[10]; should allocate 40 bytes (10 * 4)."""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(
							type_spec=_int_type(),
							name="arr",
							array_sizes=[IntLiteral(value=10)],
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		assert len(body) == 1
		assert isinstance(body[0], IRAlloc)
		assert body[0].size == 40

	def test_array_subscript_read(self) -> None:
		"""int arr[10]; return arr[2]; -- generates pointer arithmetic + load."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(
							type_spec=_int_type(),
							name="arr",
							array_sizes=[IntLiteral(value=10)],
						),
						ReturnStmt(expression=ArraySubscript(
							array=Identifier(name="arr"),
							index=IntLiteral(value=2),
						)),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# alloc, copy(arr base), mul(index*4), add(base+offset), load, return
		loads = [i for i in body if isinstance(i, IRLoad)]
		assert len(loads) == 1
		binops = [i for i in body if isinstance(i, IRBinOp)]
		# At least a multiply (index * size) and an add (base + offset)
		assert len(binops) >= 2
		ops = [b.op for b in binops]
		assert "*" in ops
		assert "+" in ops

	def test_array_subscript_write(self) -> None:
		"""int arr[10]; arr[0] = 42; -- generates store to computed address."""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(
							type_spec=_int_type(),
							name="arr",
							array_sizes=[IntLiteral(value=10)],
						),
						ExprStmt(expression=Assignment(
							target=ArraySubscript(
								array=Identifier(name="arr"),
								index=IntLiteral(value=0),
							),
							value=IntLiteral(value=42),
						)),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		stores = [i for i in body if isinstance(i, IRStore)]
		assert len(stores) == 1
		assert stores[0].value == IRConst(42)

	def test_char_array_alloc(self) -> None:
		"""char buf[256]; should allocate 256 bytes."""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(
							type_spec=TypeSpec(base_type="char"),
							name="buf",
							array_sizes=[IntLiteral(value=256)],
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		alloc = ir.functions[0].body[0]
		assert isinstance(alloc, IRAlloc)
		assert alloc.size == 256


class TestDoWhileStmt:
	def test_do_while_basic(self) -> None:
		"""do { } while (1); -- body executes before condition check."""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						DoWhileStmt(
							body=CompoundStmt(statements=[]),
							condition=IntLiteral(value=1),
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# body label, cond label, cond_jump, end label
		assert isinstance(body[0], IRLabelInstr)
		body_label = body[0].name
		assert isinstance(body[1], IRLabelInstr)  # cond label
		assert isinstance(body[2], IRCondJump)
		assert body[2].true_label == body_label
		assert isinstance(body[3], IRLabelInstr)
		assert body[3].name == body[2].false_label

	def test_do_while_with_body(self) -> None:
		"""do { return 42; } while (0);"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						DoWhileStmt(
							body=CompoundStmt(
								statements=[ReturnStmt(expression=IntLiteral(value=42))]
							),
							condition=IntLiteral(value=0),
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# body_label, return, cond_label, cond_jump, end_label
		assert isinstance(body[0], IRLabelInstr)
		assert isinstance(body[1], IRReturn)
		assert isinstance(body[2], IRLabelInstr)
		assert isinstance(body[3], IRCondJump)
		assert isinstance(body[4], IRLabelInstr)


class TestBreakStmt:
	def test_break_in_while(self) -> None:
		"""while (1) { break; }"""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						WhileStmt(
							condition=IntLiteral(value=1),
							body=CompoundStmt(
								statements=[BreakStmt()]
							),
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# start_label, cond_jump, body_label, jump(break->end), jump(back->start), end_label
		end_label = [i for i in body if isinstance(i, IRLabelInstr)][-1]
		break_jump = [i for i in body if isinstance(i, IRJump)][0]
		assert break_jump.target == end_label.name

	def test_break_in_for(self) -> None:
		"""for (;;) { break; }"""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ForStmt(
							body=CompoundStmt(
								statements=[BreakStmt()]
							),
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		end_label = [i for i in body if isinstance(i, IRLabelInstr)][-1]
		break_jump = [i for i in body if isinstance(i, IRJump)][0]
		assert break_jump.target == end_label.name

	def test_break_in_do_while(self) -> None:
		"""do { break; } while (1);"""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						DoWhileStmt(
							body=CompoundStmt(
								statements=[BreakStmt()]
							),
							condition=IntLiteral(value=1),
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		end_label = [i for i in body if isinstance(i, IRLabelInstr)][-1]
		break_jump = [i for i in body if isinstance(i, IRJump)][0]
		assert break_jump.target == end_label.name


class TestContinueStmt:
	def test_continue_in_while(self) -> None:
		"""while (1) { continue; }"""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						WhileStmt(
							condition=IntLiteral(value=1),
							body=CompoundStmt(
								statements=[ContinueStmt()]
							),
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# continue in while jumps to loop_start (condition re-check)
		start_label = body[0]
		assert isinstance(start_label, IRLabelInstr)
		continue_jump = [i for i in body if isinstance(i, IRJump)][0]
		assert continue_jump.target == start_label.name

	def test_continue_in_for(self) -> None:
		"""for (;;) { continue; } -- continue jumps to update label."""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ForStmt(
							body=CompoundStmt(
								statements=[ContinueStmt()]
							),
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# start_label, body_label, jump(continue->update), update_label, jump(->start), end_label
		labels = [i for i in body if isinstance(i, IRLabelInstr)]
		jumps = [i for i in body if isinstance(i, IRJump)]
		# continue jump should target the update label (3rd label)
		assert jumps[0].target == labels[2].name

	def test_continue_in_do_while(self) -> None:
		"""do { continue; } while (1); -- continue jumps to condition."""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						DoWhileStmt(
							body=CompoundStmt(
								statements=[ContinueStmt()]
							),
							condition=IntLiteral(value=1),
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# body_label, jump(continue->cond), cond_label, cond_jump, end_label
		labels = [i for i in body if isinstance(i, IRLabelInstr)]
		jumps = [i for i in body if isinstance(i, IRJump)]
		# continue jumps to cond label (2nd label)
		assert jumps[0].target == labels[1].name


class TestCompoundAssignment:
	def test_compound_assign_identifier(self) -> None:
		"""int x = 5; x += 3;"""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=5)),
						ExprStmt(
							expression=CompoundAssignment(
								target=Identifier(name="x"),
								op="+",
								value=IntLiteral(value=3),
							)
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# alloc, copy(5), copy(x->t), binop(t+3), copy(result->x)
		binops = [i for i in body if isinstance(i, IRBinOp)]
		assert len(binops) == 1
		assert binops[0].op == "+"
		assert binops[0].right == IRConst(3)

	def test_compound_assign_subtract(self) -> None:
		"""int x = 10; x -= 2;"""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=10)),
						ExprStmt(
							expression=CompoundAssignment(
								target=Identifier(name="x"),
								op="-",
								value=IntLiteral(value=2),
							)
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		binops = [i for i in body if isinstance(i, IRBinOp)]
		assert len(binops) == 1
		assert binops[0].op == "-"

	def test_compound_assign_multiply(self) -> None:
		"""int x = 3; x *= 4;"""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=3)),
						ExprStmt(
							expression=CompoundAssignment(
								target=Identifier(name="x"),
								op="*",
								value=IntLiteral(value=4),
							)
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		binops = [i for i in body if isinstance(i, IRBinOp)]
		assert len(binops) == 1
		assert binops[0].op == "*"

	def test_compound_assign_array_subscript(self) -> None:
		"""int arr[10]; arr[2] += 5;"""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(
							type_spec=_int_type(),
							name="arr",
							array_sizes=[IntLiteral(value=10)],
						),
						ExprStmt(
							expression=CompoundAssignment(
								target=ArraySubscript(
									array=Identifier(name="arr"),
									index=IntLiteral(value=2),
								),
								op="+",
								value=IntLiteral(value=5),
							)
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		loads = [i for i in body if isinstance(i, IRLoad)]
		stores = [i for i in body if isinstance(i, IRStore)]
		assert len(loads) == 1  # load current value
		assert len(stores) == 1  # store updated value
		binops = [i for i in body if isinstance(i, IRBinOp)]
		# There should be binops for addr computation (mul, add) twice + the actual += op
		add_ops = [b for b in binops if b.op == "+"]
		assert any(b.right == IRConst(5) for b in add_ops)


class TestShortCircuit:
	def test_logical_and(self) -> None:
		"""1 && 2 should emit conditional jumps, not IRBinOp with '&&'."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=BinaryOp(
								left=IntLiteral(value=1), op="&&", right=IntLiteral(value=2)
							)
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# Should NOT contain IRBinOp with '&&'
		binops = [i for i in body if isinstance(i, IRBinOp) and i.op == "&&"]
		assert len(binops) == 0
		# Should contain conditional jumps
		cond_jumps = [i for i in body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 1
		labels = [i for i in body if isinstance(i, IRLabelInstr)]
		assert len(labels) >= 2

	def test_logical_or(self) -> None:
		"""1 || 0 should emit conditional jumps, not IRBinOp with '||'."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=BinaryOp(
								left=IntLiteral(value=1), op="||", right=IntLiteral(value=0)
							)
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		binops = [i for i in body if isinstance(i, IRBinOp) and i.op == "||"]
		assert len(binops) == 0
		cond_jumps = [i for i in body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 1


class TestPointerOps:
	def test_deref_emits_load(self) -> None:
		"""*p should emit IRLoad, not IRUnaryOp."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[ParamDecl(type_spec=TypeSpec(base_type="int", pointer_count=1), name="p")],
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=UnaryOp(op="*", operand=Identifier(name="p"))
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		loads = [i for i in body if isinstance(i, IRLoad)]
		assert len(loads) == 1
		unary_stars = [i for i in body if isinstance(i, IRUnaryOp) and i.op == "*"]
		assert len(unary_stars) == 0

	def test_address_of_returns_local(self) -> None:
		"""&x should return the temp for x directly (its stack address)."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="x"),
						ReturnStmt(
							expression=UnaryOp(op="&", operand=Identifier(name="x"))
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# Should not emit IRUnaryOp with '&'
		unary_amps = [i for i in body if isinstance(i, IRUnaryOp) and i.op == "&"]
		assert len(unary_amps) == 0

	def test_deref_write_emits_store(self) -> None:
		"""*p = 42 should emit IRStore."""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				params=[ParamDecl(type_spec=TypeSpec(base_type="int", pointer_count=1), name="p")],
				body=CompoundStmt(
					statements=[
						ExprStmt(
							expression=Assignment(
								target=UnaryOp(op="*", operand=Identifier(name="p")),
								value=IntLiteral(value=42),
							)
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		stores = [i for i in body if isinstance(i, IRStore)]
		assert len(stores) == 1
		assert stores[0].value == IRConst(42)


class TestPrefixIncDec:
	def test_prefix_increment(self) -> None:
		"""++x should emit load + add 1 + store, not IRUnaryOp."""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=5)),
						ExprStmt(
							expression=UnaryOp(op="++", operand=Identifier(name="x"))
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# Should not have IRUnaryOp with '++'
		unary_inc = [i for i in body if isinstance(i, IRUnaryOp) and i.op == "++"]
		assert len(unary_inc) == 0
		# Should have a binop with '+'
		binops = [i for i in body if isinstance(i, IRBinOp)]
		assert any(b.op == "+" and b.right == IRConst(1) for b in binops)

	def test_prefix_decrement(self) -> None:
		"""--x should emit load + sub 1 + store, not IRUnaryOp."""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=10)),
						ExprStmt(
							expression=UnaryOp(op="--", operand=Identifier(name="x"))
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		unary_dec = [i for i in body if isinstance(i, IRUnaryOp) and i.op == "--"]
		assert len(unary_dec) == 0
		binops = [i for i in body if isinstance(i, IRBinOp)]
		assert any(b.op == "-" and b.right == IRConst(1) for b in binops)


# ------------------------------------------------------------------
# Switch statement
# ------------------------------------------------------------------


class TestSwitchStmt:
	def test_switch_single_case(self) -> None:
		"""switch (x) { case 1: return 10; }"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[ParamDecl(type_spec=_int_type(), name="x")],
				body=CompoundStmt(
					statements=[
						SwitchStmt(
							expression=Identifier(name="x"),
							cases=[
								CaseClause(
									value=IntLiteral(value=1),
									statements=[ReturnStmt(expression=IntLiteral(value=10))],
								),
							],
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# Should have a comparison == and a cond jump
		cond_jumps = [i for i in body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) >= 1
		# Should have a case body with return 10
		returns = [i for i in body if isinstance(i, IRReturn)]
		assert len(returns) == 1
		assert returns[0].value == IRConst(10)

	def test_switch_with_default(self) -> None:
		"""switch (x) { case 1: return 10; default: return 0; }"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[ParamDecl(type_spec=_int_type(), name="x")],
				body=CompoundStmt(
					statements=[
						SwitchStmt(
							expression=Identifier(name="x"),
							cases=[
								CaseClause(
									value=IntLiteral(value=1),
									statements=[ReturnStmt(expression=IntLiteral(value=10))],
								),
								CaseClause(
									value=None,
									statements=[ReturnStmt(expression=IntLiteral(value=0))],
								),
							],
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		returns = [i for i in body if isinstance(i, IRReturn)]
		assert len(returns) == 2
		# Default case should produce a jump (not fall off to end)
		jumps = [i for i in body if isinstance(i, IRJump)]
		# There should be a jump to default label (not end_label)
		labels = [i for i in body if isinstance(i, IRLabelInstr)]
		label_names = [lbl.name for lbl in labels]
		# The jump after case checks should target a case label, not switch_end
		assert any(j.target in label_names for j in jumps)

	def test_switch_with_break(self) -> None:
		"""switch (x) { case 1: return 1; break; case 2: return 2; break; }"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[ParamDecl(type_spec=_int_type(), name="x")],
				body=CompoundStmt(
					statements=[
						SwitchStmt(
							expression=Identifier(name="x"),
							cases=[
								CaseClause(
									value=IntLiteral(value=1),
									statements=[
										ReturnStmt(expression=IntLiteral(value=1)),
										BreakStmt(),
									],
								),
								CaseClause(
									value=IntLiteral(value=2),
									statements=[
										ReturnStmt(expression=IntLiteral(value=2)),
										BreakStmt(),
									],
								),
							],
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# Break should generate jumps to the switch end label
		end_label = [i for i in body if isinstance(i, IRLabelInstr)][-1]
		break_jumps = [i for i in body if isinstance(i, IRJump) and i.target == end_label.name]
		assert len(break_jumps) >= 2

	def test_switch_fallthrough(self) -> None:
		"""switch (x) { case 1: case 2: return 99; } -- fallthrough from case 1 to case 2."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[ParamDecl(type_spec=_int_type(), name="x")],
				body=CompoundStmt(
					statements=[
						SwitchStmt(
							expression=Identifier(name="x"),
							cases=[
								CaseClause(
									value=IntLiteral(value=1),
									statements=[],  # fallthrough
								),
								CaseClause(
									value=IntLiteral(value=2),
									statements=[ReturnStmt(expression=IntLiteral(value=99))],
								),
							],
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# Both case body labels should exist (not counting case_check labels)
		case_body_labels = [
			i for i in body
			if isinstance(i, IRLabelInstr) and i.name.startswith("case") and "check" not in i.name
		]
		assert len(case_body_labels) == 2
		returns = [i for i in body if isinstance(i, IRReturn)]
		assert len(returns) == 1
		assert returns[0].value == IRConst(99)

	def test_switch_multiple_cases(self) -> None:
		"""switch (x) { case 0: return 0; case 1: return 1; case 2: return 2; }"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[ParamDecl(type_spec=_int_type(), name="x")],
				body=CompoundStmt(
					statements=[
						SwitchStmt(
							expression=Identifier(name="x"),
							cases=[
								CaseClause(
									value=IntLiteral(value=0),
									statements=[ReturnStmt(expression=IntLiteral(value=0))],
								),
								CaseClause(
									value=IntLiteral(value=1),
									statements=[ReturnStmt(expression=IntLiteral(value=1))],
								),
								CaseClause(
									value=IntLiteral(value=2),
									statements=[ReturnStmt(expression=IntLiteral(value=2))],
								),
							],
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# Should have 3 comparisons
		eq_binops = [i for i in body if isinstance(i, IRBinOp) and i.op == "=="]
		assert len(eq_binops) == 3
		cond_jumps = [i for i in body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) == 3


# ------------------------------------------------------------------
# Ternary expression
# ------------------------------------------------------------------


class TestTernaryExpr:
	def test_simple_ternary(self) -> None:
		"""return x ? 1 : 0;"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[ParamDecl(type_spec=_int_type(), name="x")],
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=TernaryExpr(
								condition=Identifier(name="x"),
								true_expr=IntLiteral(value=1),
								false_expr=IntLiteral(value=0),
							)
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		cond_jumps = [i for i in body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) == 1
		# Should have two copy instructions to the result temp
		copies = [i for i in body if isinstance(i, IRCopy)]
		# At least 3 copies: one for identifier load, two for true/false branch results
		assert len(copies) >= 3
		# Should have a return
		returns = [i for i in body if isinstance(i, IRReturn)]
		assert len(returns) == 1

	def test_ternary_with_expressions(self) -> None:
		"""return x ? x + 1 : x - 1;"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[ParamDecl(type_spec=_int_type(), name="x")],
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=TernaryExpr(
								condition=Identifier(name="x"),
								true_expr=BinaryOp(
									left=Identifier(name="x"), op="+", right=IntLiteral(value=1)
								),
								false_expr=BinaryOp(
									left=Identifier(name="x"), op="-", right=IntLiteral(value=1)
								),
							)
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		binops = [i for i in body if isinstance(i, IRBinOp)]
		ops = [b.op for b in binops]
		assert "+" in ops
		assert "-" in ops

	def test_nested_ternary(self) -> None:
		"""return x ? (y ? 1 : 2) : 3;"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[
					ParamDecl(type_spec=_int_type(), name="x"),
					ParamDecl(type_spec=_int_type(), name="y"),
				],
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=TernaryExpr(
								condition=Identifier(name="x"),
								true_expr=TernaryExpr(
									condition=Identifier(name="y"),
									true_expr=IntLiteral(value=1),
									false_expr=IntLiteral(value=2),
								),
								false_expr=IntLiteral(value=3),
							)
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# Two ternary expansions = 2 cond jumps
		cond_jumps = [i for i in body if isinstance(i, IRCondJump)]
		assert len(cond_jumps) == 2

	def test_ternary_result_used_in_return(self) -> None:
		"""Ternary result temp is what gets returned."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=TernaryExpr(
								condition=IntLiteral(value=1),
								true_expr=IntLiteral(value=42),
								false_expr=IntLiteral(value=0),
							)
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		ret = [i for i in body if isinstance(i, IRReturn)][0]
		# The return value should be the result temp of the ternary
		assert ret.value is not None


# ------------------------------------------------------------------
# Sizeof expression
# ------------------------------------------------------------------


class TestSizeofExpr:
	def test_sizeof_int(self) -> None:
		"""sizeof(int) == 4"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=SizeofExpr(type_operand=TypeSpec(base_type="int"))
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		ret = body[0]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(4)

	def test_sizeof_char(self) -> None:
		"""sizeof(char) == 1"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=SizeofExpr(type_operand=TypeSpec(base_type="char"))
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		ret = body[0]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(1)

	def test_sizeof_pointer(self) -> None:
		"""sizeof(int*) == 8"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=SizeofExpr(
								type_operand=TypeSpec(base_type="int", pointer_count=1)
							)
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		ret = body[0]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(8)

	def test_sizeof_char_pointer(self) -> None:
		"""sizeof(char*) == 8"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=SizeofExpr(
								type_operand=TypeSpec(base_type="char", pointer_count=1)
							)
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		ret = body[0]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(8)

	def test_sizeof_expr_defaults_to_int_size(self) -> None:
		"""sizeof(expr) defaults to 4."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=SizeofExpr(operand=IntLiteral(value=42))
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		ret = body[0]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(4)

	def test_sizeof_struct(self) -> None:
		"""sizeof(struct point) with two int members == 8."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						StructDecl(
							name="point",
							members=[
								StructMember(type_spec=TypeSpec(base_type="int"), name="x"),
								StructMember(type_spec=TypeSpec(base_type="int"), name="y"),
							],
						),
						ReturnStmt(
							expression=SizeofExpr(
								type_operand=TypeSpec(base_type="point")
							)
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		ret = body[0]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(8)

	def test_sizeof_in_expression(self) -> None:
		"""return sizeof(int) + 1; -- sizeof folds to constant."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						ReturnStmt(
							expression=BinaryOp(
								left=SizeofExpr(type_operand=TypeSpec(base_type="int")),
								op="+",
								right=IntLiteral(value=1),
							)
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		binop = body[0]
		assert isinstance(binop, IRBinOp)
		assert binop.left == IRConst(4)
		assert binop.right == IRConst(1)


# ------------------------------------------------------------------
# Postfix increment/decrement
# ------------------------------------------------------------------


class TestPostfixExpr:
	def test_postfix_increment_returns_old_value(self) -> None:
		"""int x = 5; return x++; -- returns old value (5)."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=5)),
						ReturnStmt(
							expression=PostfixExpr(
								operand=Identifier(name="x"), op="++"
							)
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# Should NOT have IRUnaryOp with '++'
		unary_inc = [i for i in body if isinstance(i, IRUnaryOp) and i.op == "++"]
		assert len(unary_inc) == 0
		# Should have a binop with '+'
		binops = [i for i in body if isinstance(i, IRBinOp)]
		assert any(b.op == "+" and b.right == IRConst(1) for b in binops)
		# The return should use the OLD value (copy before increment)
		ret = [i for i in body if isinstance(i, IRReturn)][0]
		# The returned temp should be the copy made before the binop
		copies = [i for i in body if isinstance(i, IRCopy)]
		old_copy = copies[1]  # first copy is init, second is old value save
		assert ret.value == old_copy.dest

	def test_postfix_decrement(self) -> None:
		"""int x = 10; x--;"""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=10)),
						ExprStmt(
							expression=PostfixExpr(
								operand=Identifier(name="x"), op="--"
							)
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		binops = [i for i in body if isinstance(i, IRBinOp)]
		assert any(b.op == "-" and b.right == IRConst(1) for b in binops)

	def test_postfix_in_expression(self) -> None:
		"""int x = 5; return x++ + 1; -- x++ returns 5, so result is 6."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=5)),
						ReturnStmt(
							expression=BinaryOp(
								left=PostfixExpr(operand=Identifier(name="x"), op="++"),
								op="+",
								right=IntLiteral(value=1),
							)
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# The outer binop uses the OLD value of x
		ret = [i for i in body if isinstance(i, IRReturn)][0]
		assert ret.value is not None
		# There should be an addition with literal 1 from the outer expression
		outer_adds = [
			i for i in body
			if isinstance(i, IRBinOp) and i.op == "+" and i.right == IRConst(1)
		]
		# Two additions: one from postfix (x + 1), one from outer (old + 1)
		assert len(outer_adds) == 2

	def test_postfix_on_array_subscript(self) -> None:
		"""int arr[10]; arr[0]++;"""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(
							type_spec=_int_type(),
							name="arr",
							array_sizes=[IntLiteral(value=10)],
						),
						ExprStmt(
							expression=PostfixExpr(
								operand=ArraySubscript(
									array=Identifier(name="arr"),
									index=IntLiteral(value=0),
								),
								op="++",
							)
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		loads = [i for i in body if isinstance(i, IRLoad)]
		stores = [i for i in body if isinstance(i, IRStore)]
		assert len(loads) == 1
		assert len(stores) == 1
		binops = [i for i in body if isinstance(i, IRBinOp) and i.op == "+" and i.right == IRConst(1)]
		assert len(binops) >= 1

	def test_postfix_in_for_update(self) -> None:
		"""for (int i = 0; i < 10; i++) {} -- postfix in update position."""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						VarDecl(type_spec=_int_type(), name="i", initializer=IntLiteral(value=0)),
						ForStmt(
							condition=BinaryOp(
								left=Identifier(name="i"),
								op="<",
								right=IntLiteral(value=10),
							),
							update=ExprStmt(
								expression=PostfixExpr(
									operand=Identifier(name="i"), op="++"
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
		# The for loop should contain the postfix increment in the update section
		binops = [i for i in body if isinstance(i, IRBinOp)]
		assert any(b.op == "+" and b.right == IRConst(1) for b in binops)


# ------------------------------------------------------------------
# Struct declarations and member access
# ------------------------------------------------------------------


class TestStructDecl:
	def test_struct_decl_no_op(self) -> None:
		"""struct point { int x; int y; }; -- no IR instructions emitted."""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						StructDecl(
							name="point",
							members=[
								StructMember(type_spec=TypeSpec(base_type="int"), name="x"),
								StructMember(type_spec=TypeSpec(base_type="int"), name="y"),
							],
						)
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		assert len(body) == 0


class TestMemberAccess:
	def test_dot_access_first_field(self) -> None:
		"""struct point p; p.x -- offset 0."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						StructDecl(
							name="point",
							members=[
								StructMember(type_spec=TypeSpec(base_type="int"), name="x"),
								StructMember(type_spec=TypeSpec(base_type="int"), name="y"),
							],
						),
						VarDecl(type_spec=TypeSpec(base_type="point"), name="p"),
						ReturnStmt(
							expression=MemberAccess(
								object=Identifier(name="p"),
								member="x",
								is_arrow=False,
							)
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# Should have: alloc(p), copy(p), binop(p+0), load, return
		binops = [i for i in body if isinstance(i, IRBinOp)]
		assert any(b.op == "+" and b.right == IRConst(0) for b in binops)
		loads = [i for i in body if isinstance(i, IRLoad)]
		assert len(loads) == 1

	def test_dot_access_second_field(self) -> None:
		"""struct point p; p.y -- offset 4 (after int x)."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						StructDecl(
							name="point",
							members=[
								StructMember(type_spec=TypeSpec(base_type="int"), name="x"),
								StructMember(type_spec=TypeSpec(base_type="int"), name="y"),
							],
						),
						VarDecl(type_spec=TypeSpec(base_type="point"), name="p"),
						ReturnStmt(
							expression=MemberAccess(
								object=Identifier(name="p"),
								member="y",
								is_arrow=False,
							)
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		binops = [i for i in body if isinstance(i, IRBinOp)]
		assert any(b.op == "+" and b.right == IRConst(4) for b in binops)

	def test_arrow_access(self) -> None:
		"""struct point *p; p->x -- arrow access."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[
					ParamDecl(type_spec=TypeSpec(base_type="point", pointer_count=1), name="p")
				],
				body=CompoundStmt(
					statements=[
						StructDecl(
							name="point",
							members=[
								StructMember(type_spec=TypeSpec(base_type="int"), name="x"),
								StructMember(type_spec=TypeSpec(base_type="int"), name="y"),
							],
						),
						ReturnStmt(
							expression=MemberAccess(
								object=Identifier(name="p"),
								member="x",
								is_arrow=True,
							)
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		# Should have binop for offset computation and a load
		loads = [i for i in body if isinstance(i, IRLoad)]
		assert len(loads) == 1

	def test_member_access_with_char_field(self) -> None:
		"""struct s { int a; char b; }; s.b -- offset 4."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						StructDecl(
							name="s",
							members=[
								StructMember(type_spec=TypeSpec(base_type="int"), name="a"),
								StructMember(type_spec=TypeSpec(base_type="char"), name="b"),
							],
						),
						VarDecl(type_spec=TypeSpec(base_type="s"), name="val"),
						ReturnStmt(
							expression=MemberAccess(
								object=Identifier(name="val"),
								member="b",
								is_arrow=False,
							)
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		binops = [i for i in body if isinstance(i, IRBinOp)]
		# Offset for b: int(4) + 0 for b = 4
		assert any(b.op == "+" and b.right == IRConst(4) for b in binops)


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
