"""Torture test: pointer arithmetic through casts sets correct pointee size."""

from compiler.ast_nodes import (
	ArraySubscript,
	CompoundStmt,
	FunctionDecl,
	Identifier,
	IntLiteral,
	ParamDecl,
	Program,
	ReturnStmt,
	TypeSpec,
)
from compiler.ir import IRBinOp, IRConst
from compiler.ir_gen import IRGenerator


def _int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def _make_program(*decls) -> Program:
	return Program(declarations=list(decls))


def test_ptr_array_equiv_int_ptr_subscript() -> None:
	"""int *p; p[3] should use stride 4."""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			params=[ParamDecl(name="p", type_spec=TypeSpec(base_type="int", pointer_count=1))],
			body=CompoundStmt(
				statements=[
					ReturnStmt(expression=ArraySubscript(
						array=Identifier(name="p"),
						index=IntLiteral(value=3),
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


def test_ptr_array_equiv_char_ptr_subscript() -> None:
	"""char *s; s[1] should use stride 1."""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			params=[ParamDecl(name="s", type_spec=TypeSpec(base_type="char", pointer_count=1))],
			body=CompoundStmt(
				statements=[
					ReturnStmt(expression=ArraySubscript(
						array=Identifier(name="s"),
						index=IntLiteral(value=1),
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
