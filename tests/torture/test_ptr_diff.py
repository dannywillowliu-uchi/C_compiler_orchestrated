"""Torture test: pointer difference and pointer arithmetic scaling."""

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


def test_ptr_diff_long_ptr_subscript() -> None:
	"""long *lp; lp[1] should use stride 8."""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			params=[ParamDecl(name="lp", type_spec=TypeSpec(base_type="long", pointer_count=1))],
			body=CompoundStmt(
				statements=[
					ReturnStmt(expression=ArraySubscript(
						array=Identifier(name="lp"),
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
	assert stride_op.right.value == 8


def test_ptr_diff_short_ptr_subscript() -> None:
	"""short *sp; sp[2] should use stride 2."""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			params=[ParamDecl(name="sp", type_spec=TypeSpec(base_type="short", pointer_count=1))],
			body=CompoundStmt(
				statements=[
					ReturnStmt(expression=ArraySubscript(
						array=Identifier(name="sp"),
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
	assert stride_op.right.value == 2
