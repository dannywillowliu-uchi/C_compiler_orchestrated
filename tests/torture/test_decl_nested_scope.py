"""Torture test: nested block scoping preserves outer variable bindings."""

from compiler.ast_nodes import (
	CompoundStmt,
	ForStmt,
	FunctionDecl,
	Identifier,
	IntLiteral,
	Program,
	ReturnStmt,
	TypeSpec,
	VarDecl,
)
from compiler.ir import IRAlloc, IRCopy, IRReturn
from compiler.ir_gen import IRGenerator


def _int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def _make_program(*decls) -> Program:
	return Program(declarations=list(decls))


def test_decl_nested_scope_basic() -> None:
	"""int x = 10; { int x = 20; } return x; -- returns outer x."""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			body=CompoundStmt(
				statements=[
					VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=10)),
					CompoundStmt(
						statements=[
							VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=20)),
						]
					),
					ReturnStmt(expression=Identifier(name="x")),
				]
			),
		)
	)
	ir = IRGenerator().generate(prog)
	fn = ir.functions[0]
	allocs = [i for i in fn.body if isinstance(i, IRAlloc)]
	assert len(allocs) == 2  # outer x, inner x
	outer_x = allocs[0].dest
	# The return value should come from a copy of outer_x, not inner_x
	copies_from_outer = [
		i for i in fn.body
		if isinstance(i, IRCopy) and i.source == outer_x
	]
	assert len(copies_from_outer) >= 1


def test_decl_nested_scope_multiple_levels() -> None:
	"""int x = 1; { int x = 2; { int x = 3; } } return x;"""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			body=CompoundStmt(
				statements=[
					VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=1)),
					CompoundStmt(
						statements=[
							VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=2)),
							CompoundStmt(
								statements=[
									VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=3)),
								]
							),
						]
					),
					ReturnStmt(expression=Identifier(name="x")),
				]
			),
		)
	)
	ir = IRGenerator().generate(prog)
	fn = ir.functions[0]
	allocs = [i for i in fn.body if isinstance(i, IRAlloc)]
	assert len(allocs) == 3  # three separate x allocations
	outer_x = allocs[0].dest
	# Return should use outer x
	copies_from_outer = [
		i for i in fn.body
		if isinstance(i, IRCopy) and i.source == outer_x
	]
	assert len(copies_from_outer) >= 1


def test_decl_nested_scope_different_vars() -> None:
	"""int x = 1; { int y = 2; } return x; -- y doesn't leak out."""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			body=CompoundStmt(
				statements=[
					VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=1)),
					CompoundStmt(
						statements=[
							VarDecl(type_spec=_int_type(), name="y", initializer=IntLiteral(value=2)),
						]
					),
					ReturnStmt(expression=Identifier(name="x")),
				]
			),
		)
	)
	ir = IRGenerator().generate(prog)
	fn = ir.functions[0]
	outer_x = [i for i in fn.body if isinstance(i, IRAlloc)][0].dest
	# x is referenced either via a copy or directly in the return
	copies_from_outer = [
		i for i in fn.body
		if isinstance(i, IRCopy) and i.source == outer_x
	]
	returns_outer = [
		i for i in fn.body
		if isinstance(i, IRReturn) and i.value == outer_x
	]
	assert len(copies_from_outer) >= 1 or len(returns_outer) >= 1


def test_decl_nested_scope_in_for() -> None:
	"""int x = 5; for (...) { int x = 0; } return x; -- outer x survives loop."""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			body=CompoundStmt(
				statements=[
					VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=5)),
					ForStmt(
						init=VarDecl(type_spec=_int_type(), name="i", initializer=IntLiteral(value=0)),
						body=CompoundStmt(
							statements=[
								VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=0)),
							]
						),
					),
					ReturnStmt(expression=Identifier(name="x")),
				]
			),
		)
	)
	ir = IRGenerator().generate(prog)
	fn = ir.functions[0]
	allocs = [i for i in fn.body if isinstance(i, IRAlloc)]
	# outer x, loop var i, inner x
	assert len(allocs) >= 3
	outer_x = allocs[0].dest
	copies_from_outer = [
		i for i in fn.body
		if isinstance(i, IRCopy) and i.source == outer_x
	]
	assert len(copies_from_outer) >= 1
