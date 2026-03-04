"""Tests for compound assignment and postfix on MemberAccess and pointer-deref targets."""

from compiler.ast_nodes import (
	Assignment,
	CompoundAssignment,
	CompoundStmt,
	ExprStmt,
	FunctionDecl,
	Identifier,
	IntLiteral,
	MemberAccess,
	ParamDecl,
	PostfixExpr,
	Program,
	ReturnStmt,
	StructDecl,
	StructMember,
	TypeSpec,
	UnaryOp,
	VarDecl,
)
from compiler.ir import (
	IRBinOp,
	IRLoad,
	IRStore,
)
from compiler.ir_gen import IRGenerator


def _int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def _void_type() -> TypeSpec:
	return TypeSpec(base_type="void")


def _ptr_int_type() -> TypeSpec:
	return TypeSpec(base_type="int", pointer_count=1)


def _struct_type(name: str) -> TypeSpec:
	return TypeSpec(base_type=f"struct {name}")


def _make_program(*decls) -> Program:
	return Program(declarations=list(decls))


def _struct_point() -> StructDecl:
	"""struct Point { int x; int y; };"""
	return StructDecl(
		name="Point",
		members=[
			StructMember(name="x", type_spec=_int_type()),
			StructMember(name="y", type_spec=_int_type()),
		],
	)


def _struct_triple() -> StructDecl:
	"""struct Triple { int a; int b; int c; };"""
	return StructDecl(
		name="Triple",
		members=[
			StructMember(name="a", type_spec=_int_type()),
			StructMember(name="b", type_spec=_int_type()),
			StructMember(name="c", type_spec=_int_type()),
		],
	)


def _get_instrs(prog: Program) -> list:
	gen = IRGenerator()
	ir_prog = gen.generate(prog)
	assert len(ir_prog.functions) > 0
	return ir_prog.functions[0].body


# ------------------------------------------------------------------
# Compound assignment on struct members: s.x += 1, s.y -= 2, etc.
# ------------------------------------------------------------------


class TestCompoundAssignMember:
	def test_member_plus_equals(self) -> None:
		"""s.x += 1 should load from member addr, add, store back."""
		prog = _make_program(
			_struct_point(),
			FunctionDecl(
				return_type=_void_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="s", type_spec=_struct_type("Point")),
					ExprStmt(expression=CompoundAssignment(
						target=MemberAccess(object=Identifier(name="s"), member="x"),
						op="+=",
						value=IntLiteral(value=1),
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		stores = [i for i in instrs if isinstance(i, IRStore)]
		binops = [i for i in instrs if isinstance(i, IRBinOp) and i.op == "+"]
		assert len(loads) >= 1, "Should load current member value"
		assert len(stores) >= 1, "Should store result back"
		assert len(binops) >= 1, "Should have an add operation"

	def test_member_minus_equals(self) -> None:
		"""s.y -= 2 should use subtraction."""
		prog = _make_program(
			_struct_point(),
			FunctionDecl(
				return_type=_void_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="s", type_spec=_struct_type("Point")),
					ExprStmt(expression=CompoundAssignment(
						target=MemberAccess(object=Identifier(name="s"), member="y"),
						op="-=",
						value=IntLiteral(value=2),
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		binops = [i for i in instrs if isinstance(i, IRBinOp) and i.op == "-"]
		stores = [i for i in instrs if isinstance(i, IRStore)]
		assert len(binops) >= 1
		assert len(stores) >= 1

	def test_member_mul_equals(self) -> None:
		"""s.x *= 3 should use multiplication."""
		prog = _make_program(
			_struct_point(),
			FunctionDecl(
				return_type=_void_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="s", type_spec=_struct_type("Point")),
					ExprStmt(expression=CompoundAssignment(
						target=MemberAccess(object=Identifier(name="s"), member="x"),
						op="*=",
						value=IntLiteral(value=3),
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		binops = [i for i in instrs if isinstance(i, IRBinOp) and i.op == "*"]
		assert len(binops) >= 1

	def test_member_div_equals(self) -> None:
		"""s.x /= 2."""
		prog = _make_program(
			_struct_point(),
			FunctionDecl(
				return_type=_void_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="s", type_spec=_struct_type("Point")),
					ExprStmt(expression=CompoundAssignment(
						target=MemberAccess(object=Identifier(name="s"), member="x"),
						op="/=",
						value=IntLiteral(value=2),
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		binops = [i for i in instrs if isinstance(i, IRBinOp) and i.op == "/"]
		assert len(binops) >= 1

	def test_member_mod_equals(self) -> None:
		"""s.x %= 5."""
		prog = _make_program(
			_struct_point(),
			FunctionDecl(
				return_type=_void_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="s", type_spec=_struct_type("Point")),
					ExprStmt(expression=CompoundAssignment(
						target=MemberAccess(object=Identifier(name="s"), member="x"),
						op="%=",
						value=IntLiteral(value=5),
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		binops = [i for i in instrs if isinstance(i, IRBinOp) and i.op == "%"]
		assert len(binops) >= 1

	def test_second_member_compound_assign(self) -> None:
		"""s.y += 10 -- second member has non-zero offset."""
		prog = _make_program(
			_struct_point(),
			FunctionDecl(
				return_type=_void_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="s", type_spec=_struct_type("Point")),
					ExprStmt(expression=CompoundAssignment(
						target=MemberAccess(object=Identifier(name="s"), member="y"),
						op="+=",
						value=IntLiteral(value=10),
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		stores = [i for i in instrs if isinstance(i, IRStore)]
		assert len(loads) >= 1
		assert len(stores) >= 1


# ------------------------------------------------------------------
# Compound assignment on pointer-deref targets: *p += 1
# ------------------------------------------------------------------


class TestCompoundAssignPointerDeref:
	def test_deref_plus_equals(self) -> None:
		"""*p += 1 should load via pointer, add, store back."""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="test",
				params=[ParamDecl(name="p", type_spec=_ptr_int_type())],
				body=CompoundStmt(statements=[
					ExprStmt(expression=CompoundAssignment(
						target=UnaryOp(op="*", operand=Identifier(name="p")),
						op="+=",
						value=IntLiteral(value=1),
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		stores = [i for i in instrs if isinstance(i, IRStore)]
		binops = [i for i in instrs if isinstance(i, IRBinOp) and i.op == "+"]
		assert len(loads) >= 1
		assert len(stores) >= 1
		assert len(binops) >= 1

	def test_deref_minus_equals(self) -> None:
		"""*p -= 5."""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="test",
				params=[ParamDecl(name="p", type_spec=_ptr_int_type())],
				body=CompoundStmt(statements=[
					ExprStmt(expression=CompoundAssignment(
						target=UnaryOp(op="*", operand=Identifier(name="p")),
						op="-=",
						value=IntLiteral(value=5),
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		binops = [i for i in instrs if isinstance(i, IRBinOp) and i.op == "-"]
		stores = [i for i in instrs if isinstance(i, IRStore)]
		assert len(binops) >= 1
		assert len(stores) >= 1

	def test_deref_mul_equals(self) -> None:
		"""*p *= 3."""
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="test",
				params=[ParamDecl(name="p", type_spec=_ptr_int_type())],
				body=CompoundStmt(statements=[
					ExprStmt(expression=CompoundAssignment(
						target=UnaryOp(op="*", operand=Identifier(name="p")),
						op="*=",
						value=IntLiteral(value=3),
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		binops = [i for i in instrs if isinstance(i, IRBinOp) and i.op == "*"]
		assert len(binops) >= 1


# ------------------------------------------------------------------
# Postfix increment/decrement on struct members: s.x++, s.y--
# ------------------------------------------------------------------


class TestPostfixMember:
	def test_member_postfix_increment(self) -> None:
		"""s.x++ should load old value, add 1, store new, return old."""
		prog = _make_program(
			_struct_point(),
			FunctionDecl(
				return_type=_int_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="s", type_spec=_struct_type("Point")),
					ReturnStmt(expression=PostfixExpr(
						operand=MemberAccess(object=Identifier(name="s"), member="x"),
						op="++",
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		stores = [i for i in instrs if isinstance(i, IRStore)]
		binops = [i for i in instrs if isinstance(i, IRBinOp) and i.op == "+"]
		assert len(loads) >= 1
		assert len(stores) >= 1
		assert len(binops) >= 1

	def test_member_postfix_decrement(self) -> None:
		"""s.y-- should load old value, subtract 1, store new, return old."""
		prog = _make_program(
			_struct_point(),
			FunctionDecl(
				return_type=_int_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="s", type_spec=_struct_type("Point")),
					ReturnStmt(expression=PostfixExpr(
						operand=MemberAccess(object=Identifier(name="s"), member="y"),
						op="--",
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		stores = [i for i in instrs if isinstance(i, IRStore)]
		binops = [i for i in instrs if isinstance(i, IRBinOp) and i.op == "-"]
		assert len(loads) >= 1
		assert len(stores) >= 1
		assert len(binops) >= 1

	def test_third_member_postfix(self) -> None:
		"""Triple.c++ -- third member has larger offset."""
		prog = _make_program(
			_struct_triple(),
			FunctionDecl(
				return_type=_int_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="t", type_spec=_struct_type("Triple")),
					ReturnStmt(expression=PostfixExpr(
						operand=MemberAccess(object=Identifier(name="t"), member="c"),
						op="++",
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		stores = [i for i in instrs if isinstance(i, IRStore)]
		assert len(loads) >= 1
		assert len(stores) >= 1


# ------------------------------------------------------------------
# Postfix increment/decrement on pointer-deref: (*p)++, (*p)--
# ------------------------------------------------------------------


class TestPostfixPointerDeref:
	def test_deref_postfix_increment(self) -> None:
		"""(*p)++ should load via pointer, add 1, store back, return old."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="test",
				params=[ParamDecl(name="p", type_spec=_ptr_int_type())],
				body=CompoundStmt(statements=[
					ReturnStmt(expression=PostfixExpr(
						operand=UnaryOp(op="*", operand=Identifier(name="p")),
						op="++",
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		stores = [i for i in instrs if isinstance(i, IRStore)]
		binops = [i for i in instrs if isinstance(i, IRBinOp) and i.op == "+"]
		assert len(loads) >= 1
		assert len(stores) >= 1
		assert len(binops) >= 1

	def test_deref_postfix_decrement(self) -> None:
		"""(*p)-- should load via pointer, subtract 1, store back, return old."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="test",
				params=[ParamDecl(name="p", type_spec=_ptr_int_type())],
				body=CompoundStmt(statements=[
					ReturnStmt(expression=PostfixExpr(
						operand=UnaryOp(op="*", operand=Identifier(name="p")),
						op="--",
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		stores = [i for i in instrs if isinstance(i, IRStore)]
		binops = [i for i in instrs if isinstance(i, IRBinOp) and i.op == "-"]
		assert len(loads) >= 1
		assert len(stores) >= 1
		assert len(binops) >= 1


# ------------------------------------------------------------------
# Nested / combined cases
# ------------------------------------------------------------------


class TestNestedCases:
	def test_multiple_member_compound_assigns(self) -> None:
		"""s.x += 1; s.y -= 2; both in the same function."""
		prog = _make_program(
			_struct_point(),
			FunctionDecl(
				return_type=_void_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="s", type_spec=_struct_type("Point")),
					ExprStmt(expression=CompoundAssignment(
						target=MemberAccess(object=Identifier(name="s"), member="x"),
						op="+=",
						value=IntLiteral(value=1),
					)),
					ExprStmt(expression=CompoundAssignment(
						target=MemberAccess(object=Identifier(name="s"), member="y"),
						op="-=",
						value=IntLiteral(value=2),
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		stores = [i for i in instrs if isinstance(i, IRStore)]
		assert len(loads) >= 2, "Should load both member values"
		assert len(stores) >= 2, "Should store both results"

	def test_member_assign_then_postfix(self) -> None:
		"""s.x = 5; s.x++; exercises both assignment and postfix."""
		prog = _make_program(
			_struct_point(),
			FunctionDecl(
				return_type=_int_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="s", type_spec=_struct_type("Point")),
					ExprStmt(expression=Assignment(
						target=MemberAccess(object=Identifier(name="s"), member="x"),
						value=IntLiteral(value=5),
					)),
					ReturnStmt(expression=PostfixExpr(
						operand=MemberAccess(object=Identifier(name="s"), member="x"),
						op="++",
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		# Assignment store + postfix store
		stores = [i for i in instrs if isinstance(i, IRStore)]
		assert len(stores) >= 2

	def test_deref_compound_then_postfix(self) -> None:
		"""*p += 10; (*p)++; exercises compound assign then postfix on pointer deref."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="test",
				params=[ParamDecl(name="p", type_spec=_ptr_int_type())],
				body=CompoundStmt(statements=[
					ExprStmt(expression=CompoundAssignment(
						target=UnaryOp(op="*", operand=Identifier(name="p")),
						op="+=",
						value=IntLiteral(value=10),
					)),
					ReturnStmt(expression=PostfixExpr(
						operand=UnaryOp(op="*", operand=Identifier(name="p")),
						op="++",
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		stores = [i for i in instrs if isinstance(i, IRStore)]
		assert len(loads) >= 2
		assert len(stores) >= 2
