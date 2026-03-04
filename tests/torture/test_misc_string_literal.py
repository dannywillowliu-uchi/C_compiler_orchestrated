"""Torture test: string literal and char pointer subscript element sizing."""

from compiler.ast_nodes import (
	ArraySubscript,
	CompoundStmt,
	FunctionDecl,
	Identifier,
	IntLiteral,
	ParamDecl,
	Program,
	ReturnStmt,
	StringLiteral,
	TypeSpec,
	VarDecl,
)
from compiler.ir import IRBinOp, IRConst
from compiler.ir_gen import IRGenerator


def _int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def _make_program(*decls) -> Program:
	return Program(declarations=list(decls))


def test_misc_string_literal_char_ptr_subscript() -> None:
	"""char *s = "hello"; return s[1]; -- stride should be 1."""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			body=CompoundStmt(
				statements=[
					VarDecl(
						type_spec=TypeSpec(base_type="char", pointer_count=1),
						name="s",
						initializer=StringLiteral(value="hello"),
					),
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
	binops = [i for i in body if isinstance(i, IRBinOp)]
	mul_ops = [b for b in binops if b.op == "*"]
	assert len(mul_ops) >= 1
	stride_op = mul_ops[-1]
	assert isinstance(stride_op.right, IRConst)
	assert stride_op.right.value == 1


def test_misc_string_literal_int_ptr_subscript() -> None:
	"""int *p param; return p[2]; -- stride should be 4."""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			params=[ParamDecl(name="p", type_spec=TypeSpec(base_type="int", pointer_count=1))],
			body=CompoundStmt(
				statements=[
					ReturnStmt(expression=ArraySubscript(
						array=Identifier(name="p"),
						index=IntLiteral(value=2),
					)),
				]
			),
		)
	)
	ir = IRGenerator().generate(prog)
	body = ir.functions[0].body
	binops = [i for i in body if isinstance(i, IRBinOp)]
	mul_ops = [b for b in binops if b.op == "*"]
	assert len(mul_ops) >= 1
	stride_op = mul_ops[-1]
	assert isinstance(stride_op.right, IRConst)
	assert stride_op.right.value == 4


def test_misc_string_literal_long_ptr_subscript() -> None:
	"""long *lp param; return lp[1]; -- stride should be 8."""
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
	binops = [i for i in body if isinstance(i, IRBinOp)]
	mul_ops = [b for b in binops if b.op == "*"]
	assert len(mul_ops) >= 1
	stride_op = mul_ops[-1]
	assert isinstance(stride_op.right, IRConst)
	assert stride_op.right.value == 8
