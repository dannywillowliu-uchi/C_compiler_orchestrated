"""Torture test: struct element size in array subscript computation."""

from compiler.ast_nodes import (
	ArraySubscript,
	CompoundStmt,
	FunctionDecl,
	Identifier,
	IntLiteral,
	Program,
	ReturnStmt,
	StructDecl,
	StructMember,
	TypeSpec,
	VarDecl,
)
from compiler.ir import IRBinOp, IRConst
from compiler.ir_gen import IRGenerator


def _int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def _make_program(*decls) -> Program:
	return Program(declarations=list(decls))


def test_struct_in_array_element_size() -> None:
	"""struct S { int x; int y; }; struct S arr[5]; arr[1] should use stride 8."""
	prog = _make_program(
		StructDecl(
			name="S",
			members=[
				StructMember(name="x", type_spec=_int_type()),
				StructMember(name="y", type_spec=_int_type()),
			],
		),
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			body=CompoundStmt(
				statements=[
					VarDecl(
						type_spec=TypeSpec(base_type="struct S"),
						name="arr",
						array_sizes=[5],
					),
					ReturnStmt(expression=ArraySubscript(
						array=Identifier(name="arr"),
						index=IntLiteral(value=1),
					)),
				]
			),
		),
	)
	ir = IRGenerator().generate(prog)
	body = ir.functions[0].body
	mul_ops = [i for i in body if isinstance(i, IRBinOp) and i.op == "*"]
	assert len(mul_ops) >= 1
	stride_op = mul_ops[-1]
	assert isinstance(stride_op.right, IRConst)
	# struct S has 2 ints = 8 bytes
	assert stride_op.right.value == 8


def test_struct_in_array_padded_size() -> None:
	"""struct P { char c; int i; }; struct P arr[3]; arr[1] should use stride 8 (padded)."""
	prog = _make_program(
		StructDecl(
			name="P",
			members=[
				StructMember(name="c", type_spec=TypeSpec(base_type="char")),
				StructMember(name="i", type_spec=_int_type()),
			],
		),
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			body=CompoundStmt(
				statements=[
					VarDecl(
						type_spec=TypeSpec(base_type="struct P"),
						name="arr",
						array_sizes=[3],
					),
					ReturnStmt(expression=ArraySubscript(
						array=Identifier(name="arr"),
						index=IntLiteral(value=1),
					)),
				]
			),
		),
	)
	ir = IRGenerator().generate(prog)
	body = ir.functions[0].body
	mul_ops = [i for i in body if isinstance(i, IRBinOp) and i.op == "*"]
	assert len(mul_ops) >= 1
	stride_op = mul_ops[-1]
	assert isinstance(stride_op.right, IRConst)
	# struct P: char(1) + padding(3) + int(4) = 8 bytes
	assert stride_op.right.value == 8
