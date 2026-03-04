"""Tests for sizeof edge cases: global variables, pointer deref, array subscript."""

from compiler.ast_nodes import (
	ArraySubscript,
	CompoundStmt,
	FunctionDecl,
	Identifier,
	IntLiteral,
	Program,
	ReturnStmt,
	SizeofExpr,
	StructDecl,
	StructMember,
	TypeSpec,
	UnaryOp,
	VarDecl,
)
from compiler.ir import IRConst, IRReturn
from compiler.ir_gen import IRGenerator


def _int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def _make_program(*decls) -> Program:
	return Program(declarations=list(decls))


def _gen(prog: Program):
	gen = IRGenerator()
	return gen.generate(prog)


class TestSizeofGlobalVars:
	def test_sizeof_global_struct_var(self) -> None:
		"""sizeof(global_struct_var) should return struct size, not 4."""
		prog = _make_program(
			StructDecl(
				name="point",
				members=[
					StructMember(type_spec=_int_type(), name="x"),
					StructMember(type_spec=_int_type(), name="y"),
				],
			),
			VarDecl(type_spec=TypeSpec(base_type="struct point"), name="pt"),
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					ReturnStmt(expression=SizeofExpr(operand=Identifier(name="pt"))),
				]),
			),
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(8)  # 2 ints = 8 bytes

	def test_sizeof_global_int_var(self) -> None:
		"""sizeof(global_int) should return 4."""
		prog = _make_program(
			VarDecl(type_spec=_int_type(), name="g"),
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					ReturnStmt(expression=SizeofExpr(operand=Identifier(name="g"))),
				]),
			),
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(4)

	def test_sizeof_global_pointer_var(self) -> None:
		"""sizeof(global_ptr) should return 8."""
		prog = _make_program(
			VarDecl(type_spec=TypeSpec(base_type="int", pointer_count=1), name="gp"),
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					ReturnStmt(expression=SizeofExpr(operand=Identifier(name="gp"))),
				]),
			),
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(8)

	def test_sizeof_global_array(self) -> None:
		"""sizeof(global_array) should return total array size."""
		prog = _make_program(
			VarDecl(
				type_spec=_int_type(),
				name="arr",
				array_sizes=[IntLiteral(value=5)],
			),
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					ReturnStmt(expression=SizeofExpr(operand=Identifier(name="arr"))),
				]),
			),
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(20)  # 5 * 4

	def test_sizeof_global_char_var(self) -> None:
		"""sizeof(global_char) should return 1."""
		prog = _make_program(
			VarDecl(type_spec=TypeSpec(base_type="char"), name="gc"),
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					ReturnStmt(expression=SizeofExpr(operand=Identifier(name="gc"))),
				]),
			),
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(1)


class TestSizeofPointerDeref:
	def test_sizeof_deref_struct_pointer(self) -> None:
		"""sizeof(*ptr) where ptr is struct pointer should return struct size."""
		prog = _make_program(
			StructDecl(
				name="pair",
				members=[
					StructMember(type_spec=_int_type(), name="a"),
					StructMember(type_spec=_int_type(), name="b"),
					StructMember(type_spec=_int_type(), name="c"),
				],
			),
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=TypeSpec(base_type="struct pair", pointer_count=1),
						name="p",
					),
					ReturnStmt(expression=SizeofExpr(
						operand=UnaryOp(op="*", operand=Identifier(name="p")),
					)),
				]),
			),
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(12)  # 3 ints = 12 bytes

	def test_sizeof_deref_global_struct_pointer(self) -> None:
		"""sizeof(*global_ptr) where global_ptr is struct pointer."""
		prog = _make_program(
			StructDecl(
				name="vec2",
				members=[
					StructMember(type_spec=TypeSpec(base_type="double"), name="x"),
					StructMember(type_spec=TypeSpec(base_type="double"), name="y"),
				],
			),
			VarDecl(
				type_spec=TypeSpec(base_type="struct vec2", pointer_count=1),
				name="gvec",
			),
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					ReturnStmt(expression=SizeofExpr(
						operand=UnaryOp(op="*", operand=Identifier(name="gvec")),
					)),
				]),
			),
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(16)  # 2 doubles = 16 bytes

	def test_sizeof_deref_int_pointer(self) -> None:
		"""sizeof(*int_ptr) should return 4."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=TypeSpec(base_type="int", pointer_count=1),
						name="ip",
					),
					ReturnStmt(expression=SizeofExpr(
						operand=UnaryOp(op="*", operand=Identifier(name="ip")),
					)),
				]),
			),
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(4)


class TestSizeofArraySubscript:
	def test_sizeof_array_element(self) -> None:
		"""sizeof(arr[0]) for int array should return 4."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=TypeSpec(base_type="int", pointer_count=1),
						name="arr",
					),
					ReturnStmt(expression=SizeofExpr(
						operand=ArraySubscript(
							array=Identifier(name="arr"),
							index=IntLiteral(value=0),
						),
					)),
				]),
			),
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(4)

	def test_sizeof_struct_pointer_subscript(self) -> None:
		"""sizeof(arr[0]) for struct pointer array should return struct size."""
		prog = _make_program(
			StructDecl(
				name="item",
				members=[
					StructMember(type_spec=_int_type(), name="id"),
					StructMember(type_spec=_int_type(), name="val"),
				],
			),
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=TypeSpec(base_type="struct item", pointer_count=1),
						name="items",
					),
					ReturnStmt(expression=SizeofExpr(
						operand=ArraySubscript(
							array=Identifier(name="items"),
							index=IntLiteral(value=0),
						),
					)),
				]),
			),
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(8)  # struct with 2 ints

	def test_sizeof_char_pointer_subscript(self) -> None:
		"""sizeof(str[0]) for char pointer should return 1."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(statements=[
					VarDecl(
						type_spec=TypeSpec(base_type="char", pointer_count=1),
						name="s",
					),
					ReturnStmt(expression=SizeofExpr(
						operand=ArraySubscript(
							array=Identifier(name="s"),
							index=IntLiteral(value=0),
						),
					)),
				]),
			),
		)
		ir = _gen(prog)
		ret = ir.functions[0].body[-1]
		assert isinstance(ret, IRReturn)
		assert ret.value == IRConst(1)
