"""Torture test: basic pointer operations -- address-of, dereference, assignment through pointer."""

from compiler.ast_nodes import (
	Assignment,
	CompoundStmt,
	FunctionDecl,
	Identifier,
	IntLiteral,
	Program,
	ReturnStmt,
	TypeSpec,
	UnaryOp,
	VarDecl,
)
from compiler.ir import IRAlloc, IRLoad, IRStore
from compiler.ir_gen import IRGenerator


def _int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def _int_ptr_type() -> TypeSpec:
	return TypeSpec(base_type="int", pointer_count=1)


def _make_program(*decls) -> Program:
	return Program(declarations=list(decls))


def test_ptr_basic_address_taken_uses_store() -> None:
	"""int x = 42; int *p = &x; -- x should use IRStore for initialization."""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="main",
			body=CompoundStmt(
				statements=[
					VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=42)),
					VarDecl(
						type_spec=_int_ptr_type(),
						name="p",
						initializer=UnaryOp(op="&", operand=Identifier(name="x")),
					),
					ReturnStmt(expression=UnaryOp(op="*", operand=Identifier(name="p"))),
				]
			),
		)
	)
	ir = IRGenerator().generate(prog)
	body = ir.functions[0].body
	# x should be allocated with IRAlloc
	allocs = [i for i in body if isinstance(i, IRAlloc)]
	assert len(allocs) >= 1
	# x's initialization should use IRStore (not IRCopy) because x is address-taken
	stores = [i for i in body if isinstance(i, IRStore)]
	assert len(stores) >= 1, "address-taken var should use IRStore for initialization"
	# The dereference *p should produce an IRLoad
	loads = [i for i in body if isinstance(i, IRLoad)]
	assert len(loads) >= 1, "pointer dereference should produce IRLoad"


def test_ptr_basic_deref_reads_stored_value() -> None:
	"""int x = 10; int *p = &x; *p = 20; return x; -- assignment through pointer."""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="main",
			body=CompoundStmt(
				statements=[
					VarDecl(type_spec=_int_type(), name="x", initializer=IntLiteral(value=10)),
					VarDecl(
						type_spec=_int_ptr_type(),
						name="p",
						initializer=UnaryOp(op="&", operand=Identifier(name="x")),
					),
					Assignment(
						target=UnaryOp(op="*", operand=Identifier(name="p")),
						value=IntLiteral(value=20),
					),
					ReturnStmt(expression=Identifier(name="x")),
				]
			),
		)
	)
	ir = IRGenerator().generate(prog)
	body = ir.functions[0].body
	# Should have stores: one for x init (IRStore to alloc), one for *p = 20 (IRStore through pointer)
	stores = [i for i in body if isinstance(i, IRStore)]
	assert len(stores) >= 2, f"expected at least 2 stores, got {len(stores)}"
	# Reading x after *p = 20 should use IRLoad (address-taken var)
	loads = [i for i in body if isinstance(i, IRLoad)]
	assert len(loads) >= 1, "reading address-taken var should use IRLoad"
