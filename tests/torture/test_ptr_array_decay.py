"""Torture test: pointer/array decay -- array name used as pointer in subscript."""

from compiler.ast_nodes import (
	ArraySubscript,
	CompoundStmt,
	FunctionDecl,
	Identifier,
	IntLiteral,
	Program,
	ReturnStmt,
	TypeSpec,
	VarDecl,
)
from compiler.ir import IRBinOp, IRConst
from compiler.ir_gen import IRGenerator


def _int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def _make_program(*decls) -> Program:
	return Program(declarations=list(decls))


def test_ptr_array_decay_int_array() -> None:
	"""int a[10]; a[2] should use stride 4."""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			body=CompoundStmt(
				statements=[
					VarDecl(
						type_spec=_int_type(),
						name="a",
						array_sizes=[10],
					),
					ReturnStmt(expression=ArraySubscript(
						array=Identifier(name="a"),
						index=IntLiteral(value=2),
					)),
				]
			),
		)
	)
	ir = IRGenerator().generate(prog)
	body = ir.functions[0].body
	mul_ops = [i for i in body if isinstance(i, IRBinOp) and i.op == "*"]
	assert len(mul_ops) >= 1
	stride_op = mul_ops[-1]
	assert isinstance(stride_op.right, IRConst)
	assert stride_op.right.value == 4


def test_ptr_array_decay_char_array() -> None:
	"""char b[20]; b[5] should use stride 1."""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			body=CompoundStmt(
				statements=[
					VarDecl(
						type_spec=TypeSpec(base_type="char"),
						name="b",
						array_sizes=[20],
					),
					ReturnStmt(expression=ArraySubscript(
						array=Identifier(name="b"),
						index=IntLiteral(value=5),
					)),
				]
			),
		)
	)
	ir = IRGenerator().generate(prog)
	body = ir.functions[0].body
	mul_ops = [i for i in body if isinstance(i, IRBinOp) and i.op == "*"]
	assert len(mul_ops) >= 1
	stride_op = mul_ops[-1]
	assert isinstance(stride_op.right, IRConst)
	assert stride_op.right.value == 1
