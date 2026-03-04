"""Torture test: type cast truncation for integer narrowing."""

from compiler.ast_nodes import (
	CastExpr,
	CompoundStmt,
	FunctionDecl,
	IntLiteral,
	Program,
	ReturnStmt,
	TypeSpec,
)
from compiler.ir import IRConvert
from compiler.ir_gen import IRGenerator


def _int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def _make_program(*decls) -> Program:
	return Program(declarations=list(decls))


def test_arith_cast_int_to_char() -> None:
	"""(char)321 should truncate via IRConvert to CHAR type."""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			body=CompoundStmt(
				statements=[
					ReturnStmt(expression=CastExpr(
						target_type=TypeSpec(base_type="char"),
						operand=IntLiteral(value=321),
					)),
				]
			),
		)
	)
	ir = IRGenerator().generate(prog)
	body = ir.functions[0].body
	converts = [i for i in body if isinstance(i, IRConvert)]
	assert len(converts) >= 1
	conv = converts[0]
	assert conv.to_type.name == "CHAR"


def test_arith_cast_int_to_short() -> None:
	"""(short)70000 should truncate via IRConvert to SHORT type."""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			body=CompoundStmt(
				statements=[
					ReturnStmt(expression=CastExpr(
						target_type=TypeSpec(base_type="short"),
						operand=IntLiteral(value=70000),
					)),
				]
			),
		)
	)
	ir = IRGenerator().generate(prog)
	body = ir.functions[0].body
	converts = [i for i in body if isinstance(i, IRConvert)]
	assert len(converts) >= 1
	conv = converts[0]
	assert conv.to_type.name == "SHORT"


def test_arith_cast_int_to_bool() -> None:
	"""(_Bool)42 should truncate via IRConvert to BOOL type."""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			body=CompoundStmt(
				statements=[
					ReturnStmt(expression=CastExpr(
						target_type=TypeSpec(base_type="_Bool"),
						operand=IntLiteral(value=42),
					)),
				]
			),
		)
	)
	ir = IRGenerator().generate(prog)
	body = ir.functions[0].body
	converts = [i for i in body if isinstance(i, IRConvert)]
	assert len(converts) >= 1
	conv = converts[0]
	assert conv.to_type.name == "BOOL"
