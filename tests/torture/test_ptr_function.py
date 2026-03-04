"""Torture test: function pointer IR generation -- address capture and indirect call."""

from compiler.ast_nodes import (
	CompoundStmt,
	FunctionCall,
	FunctionDecl,
	Identifier,
	IntLiteral,
	Program,
	ReturnStmt,
	TypeSpec,
	VarDecl,
)
from compiler.ir import IRCall, IRCopy, IRGlobalRef
from compiler.ir_gen import IRGenerator


def _int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def _void_type() -> TypeSpec:
	return TypeSpec(base_type="void")


def _make_program(*decls) -> Program:
	return Program(declarations=list(decls))


def test_ptr_function_address_capture() -> None:
	"""Function pointer variable gets address via IRGlobalRef."""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="target",
			body=CompoundStmt(
				statements=[ReturnStmt(expression=IntLiteral(value=42))]
			),
		),
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			body=CompoundStmt(
				statements=[
					VarDecl(
						type_spec=TypeSpec(
							base_type="int",
							is_function_pointer=True,
							func_ptr_return_type=_int_type(),
							func_ptr_params=[],
						),
						name="fp",
						initializer=Identifier(name="target"),
					),
					ReturnStmt(expression=FunctionCall(
						name="fp",
						arguments=[],
					)),
				]
			),
		),
	)
	ir = IRGenerator().generate(prog)
	# Should have at least 2 functions
	assert len(ir.functions) == 2
	fn = ir.functions[1]  # "f"
	# Check that a GlobalRef to "target" appears in the body
	global_refs = [
		i for i in fn.body
		if isinstance(i, IRCopy) and isinstance(i.source, IRGlobalRef)
	]
	assert len(global_refs) >= 1
	assert any(g.source.name == "target" for g in global_refs)


def test_ptr_function_indirect_call() -> None:
	"""Indirect call through function pointer has indirect=True."""
	prog = _make_program(
		FunctionDecl(
			return_type=_int_type(),
			name="target",
			body=CompoundStmt(
				statements=[ReturnStmt(expression=IntLiteral(value=10))]
			),
		),
		FunctionDecl(
			return_type=_int_type(),
			name="f",
			body=CompoundStmt(
				statements=[
					VarDecl(
						type_spec=TypeSpec(
							base_type="int",
							is_function_pointer=True,
							func_ptr_return_type=_int_type(),
							func_ptr_params=[],
						),
						name="fp",
						initializer=Identifier(name="target"),
					),
					ReturnStmt(expression=FunctionCall(
						name="fp",
						arguments=[],
					)),
				]
			),
		),
	)
	ir = IRGenerator().generate(prog)
	fn = ir.functions[1]
	calls = [i for i in fn.body if isinstance(i, IRCall)]
	assert len(calls) >= 1
	indirect_calls = [c for c in calls if c.indirect]
	assert len(indirect_calls) >= 1
