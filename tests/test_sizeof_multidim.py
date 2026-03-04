"""Tests for sizeof(expr) type inference and multi-dimensional array indexing."""

from compiler.ast_nodes import (
	ArraySubscript,
	CastExpr,
	CharLiteral,
	CompoundStmt,
	ExprStmt,
	FunctionDecl,
	Identifier,
	IntLiteral,
	MemberAccess,
	ParamDecl,
	Program,
	ReturnStmt,
	SizeofExpr,
	StringLiteral,
	StructDecl,
	StructMember,
	TypeSpec,
	UnaryOp,
	VarDecl,
)
from compiler.ir import (
	IRBinOp,
	IRConst,
	IRReturn,
)
from compiler.ir_gen import IRGenerator


def _int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def _void_type() -> TypeSpec:
	return TypeSpec(base_type="void")


def _make_program(*decls) -> Program:
	return Program(declarations=list(decls))


def _gen(prog: Program):
	gen = IRGenerator()
	return gen.generate(prog)


# ------------------------------------------------------------------
# sizeof(expr) -- type inference
# ------------------------------------------------------------------


class TestSizeofExpr:
	def test_sizeof_char_var(self) -> None:
		"""sizeof(char_var) == 1"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					VarDecl(type_spec=TypeSpec(base_type="char"), name="c"),
					ReturnStmt(expression=SizeofExpr(operand=Identifier(name="c"))),
				]),
			)
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(1)

	def test_sizeof_short_var(self) -> None:
		"""sizeof(short_var) == 2"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=TypeSpec(base_type="int", width_modifier="short"),
						name="s",
					),
					ReturnStmt(expression=SizeofExpr(operand=Identifier(name="s"))),
				]),
			)
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(2)

	def test_sizeof_int_var(self) -> None:
		"""sizeof(int_var) == 4"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					VarDecl(type_spec=_int_type(), name="x"),
					ReturnStmt(expression=SizeofExpr(operand=Identifier(name="x"))),
				]),
			)
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(4)

	def test_sizeof_long_var(self) -> None:
		"""sizeof(long_var) == 8"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=TypeSpec(base_type="int", width_modifier="long"),
						name="l",
					),
					ReturnStmt(expression=SizeofExpr(operand=Identifier(name="l"))),
				]),
			)
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(8)

	def test_sizeof_pointer_var(self) -> None:
		"""sizeof(ptr) == 8"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=TypeSpec(base_type="int", pointer_count=1),
						name="p",
					),
					ReturnStmt(expression=SizeofExpr(operand=Identifier(name="p"))),
				]),
			)
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(8)

	def test_sizeof_struct_var(self) -> None:
		"""sizeof(struct_var) returns the struct's computed size."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					StructDecl(
						name="point",
						members=[
							StructMember(type_spec=_int_type(), name="x"),
							StructMember(type_spec=_int_type(), name="y"),
						],
					),
					VarDecl(
						type_spec=TypeSpec(base_type="struct point"),
						name="pt",
					),
					ReturnStmt(expression=SizeofExpr(operand=Identifier(name="pt"))),
				]),
			)
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(8)  # 2 ints = 8 bytes

	def test_sizeof_array_var(self) -> None:
		"""sizeof(int_array[10]) == 40"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=_int_type(),
						name="arr",
						array_sizes=[IntLiteral(value=10)],
					),
					ReturnStmt(expression=SizeofExpr(operand=Identifier(name="arr"))),
				]),
			)
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(40)  # 10 * 4

	def test_sizeof_char_literal(self) -> None:
		"""sizeof('a') == 1 (char type)"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					ReturnStmt(expression=SizeofExpr(operand=CharLiteral(value="a"))),
				]),
			)
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(1)

	def test_sizeof_string_literal(self) -> None:
		"""sizeof("hello") == 8 (char pointer)"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					ReturnStmt(expression=SizeofExpr(operand=StringLiteral(value="hello"))),
				]),
			)
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(8)

	def test_sizeof_deref_pointer(self) -> None:
		"""sizeof(*int_ptr) == 4"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=TypeSpec(base_type="int", pointer_count=1),
						name="p",
					),
					ReturnStmt(expression=SizeofExpr(
						operand=UnaryOp(op="*", operand=Identifier(name="p")),
					)),
				]),
			)
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(4)

	def test_sizeof_cast_expr(self) -> None:
		"""sizeof((char)x) == 1"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					VarDecl(type_spec=_int_type(), name="x"),
					ReturnStmt(expression=SizeofExpr(
						operand=CastExpr(
							target_type=TypeSpec(base_type="char"),
							operand=Identifier(name="x"),
						),
					)),
				]),
			)
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(1)

	def test_sizeof_struct_member(self) -> None:
		"""sizeof(pt.x) where x is int == 4"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					StructDecl(
						name="point",
						members=[
							StructMember(type_spec=_int_type(), name="x"),
							StructMember(type_spec=TypeSpec(base_type="char"), name="c"),
						],
					),
					VarDecl(
						type_spec=TypeSpec(base_type="struct point"),
						name="pt",
					),
					ReturnStmt(expression=SizeofExpr(
						operand=MemberAccess(
							object=Identifier(name="pt"),
							member="c",
						),
					)),
				]),
			)
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(1)  # char member


# ------------------------------------------------------------------
# Multi-dimensional array indexing
# ------------------------------------------------------------------


class TestMultiDimArray:
	def test_2d_array_strides(self) -> None:
		"""int a[3][4]; a[i][j] should compute base + i*16 + j*4"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[
					ParamDecl(type_spec=_int_type(), name="i"),
					ParamDecl(type_spec=_int_type(), name="j"),
				],
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=_int_type(),
						name="a",
						array_sizes=[IntLiteral(value=3), IntLiteral(value=4)],
					),
					ReturnStmt(expression=ArraySubscript(
						array=ArraySubscript(
							array=Identifier(name="a"),
							index=Identifier(name="i"),
						),
						index=Identifier(name="j"),
					)),
				]),
			)
		)
		ir = _gen(prog)
		body = ir.functions[0].body
		# Find the multiply instructions for stride computation
		mul_ops = [i for i in body if isinstance(i, IRBinOp) and i.op == "*"]
		# First multiply: i * 16 (stride for first dimension = 4 * 4)
		assert mul_ops[0].right == IRConst(16)
		# Second multiply: j * 4 (stride for second dimension = element size)
		assert mul_ops[1].right == IRConst(4)

	def test_3d_array_strides(self) -> None:
		"""int a[2][3][4]; a[i][j][k] -> base + i*48 + j*16 + k*4"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[
					ParamDecl(type_spec=_int_type(), name="i"),
					ParamDecl(type_spec=_int_type(), name="j"),
					ParamDecl(type_spec=_int_type(), name="k"),
				],
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=_int_type(),
						name="a",
						array_sizes=[IntLiteral(value=2), IntLiteral(value=3), IntLiteral(value=4)],
					),
					ReturnStmt(expression=ArraySubscript(
						array=ArraySubscript(
							array=ArraySubscript(
								array=Identifier(name="a"),
								index=Identifier(name="i"),
							),
							index=Identifier(name="j"),
						),
						index=Identifier(name="k"),
					)),
				]),
			)
		)
		ir = _gen(prog)
		body = ir.functions[0].body
		mul_ops = [i for i in body if isinstance(i, IRBinOp) and i.op == "*"]
		# i * 48 (3*4*4), j * 16 (4*4), k * 4
		assert mul_ops[0].right == IRConst(48)
		assert mul_ops[1].right == IRConst(16)
		assert mul_ops[2].right == IRConst(4)

	def test_2d_char_array_strides(self) -> None:
		"""char a[3][4]; a[i][j] -> base + i*4 + j*1"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				params=[
					ParamDecl(type_spec=_int_type(), name="i"),
					ParamDecl(type_spec=_int_type(), name="j"),
				],
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=TypeSpec(base_type="char"),
						name="a",
						array_sizes=[IntLiteral(value=3), IntLiteral(value=4)],
					),
					ReturnStmt(expression=ArraySubscript(
						array=ArraySubscript(
							array=Identifier(name="a"),
							index=Identifier(name="i"),
						),
						index=Identifier(name="j"),
					)),
				]),
			)
		)
		ir = _gen(prog)
		body = ir.functions[0].body
		mul_ops = [i for i in body if isinstance(i, IRBinOp) and i.op == "*"]
		# i * 4 (4*1), j * 1
		assert mul_ops[0].right == IRConst(4)
		assert mul_ops[1].right == IRConst(1)

	def test_2d_array_assignment(self) -> None:
		"""Assignment to a[i][j] should use correct strides."""
		from compiler.ast_nodes import Assignment
		prog = _make_program(
			FunctionDecl(
				return_type=_void_type(),
				name="f",
				params=[
					ParamDecl(type_spec=_int_type(), name="i"),
					ParamDecl(type_spec=_int_type(), name="j"),
				],
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=_int_type(),
						name="a",
						array_sizes=[IntLiteral(value=3), IntLiteral(value=4)],
					),
					ExprStmt(expression=Assignment(
						target=ArraySubscript(
							array=ArraySubscript(
								array=Identifier(name="a"),
								index=Identifier(name="i"),
							),
							index=Identifier(name="j"),
						),
						value=IntLiteral(value=42),
					)),
				]),
			)
		)
		ir = _gen(prog)
		body = ir.functions[0].body
		mul_ops = [i for i in body if isinstance(i, IRBinOp) and i.op == "*"]
		assert mul_ops[0].right == IRConst(16)
		assert mul_ops[1].right == IRConst(4)

	def test_sizeof_2d_array(self) -> None:
		"""sizeof(int a[3][4]) == 48"""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=_int_type(),
						name="a",
						array_sizes=[IntLiteral(value=3), IntLiteral(value=4)],
					),
					ReturnStmt(expression=SizeofExpr(operand=Identifier(name="a"))),
				]),
			)
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(48)  # 3 * 4 * 4
