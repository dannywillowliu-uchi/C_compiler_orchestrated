"""Torture test: pointer binding in multi-declarator declarations."""

from compiler.ast_nodes import VarDecl
from compiler.lexer import Lexer
from compiler.parser import Parser


def parse(source: str):
	tokens = Lexer(source).tokenize()
	return Parser(tokens).parse()


def test_decl_multi_ptr_local_star_binds_first_only():
	"""int *a, b; -- pointer binds only to a."""
	prog = parse("int f() { int *a, b; return 0; }")
	func = prog.declarations[0]
	stmts = func.body.statements
	a_decl = stmts[0]
	b_decl = stmts[1]
	assert isinstance(a_decl, VarDecl)
	assert isinstance(b_decl, VarDecl)
	assert a_decl.name == "a"
	assert a_decl.type_spec.pointer_count == 1
	assert b_decl.name == "b"
	assert b_decl.type_spec.pointer_count == 0


def test_decl_multi_ptr_global_star_binds_first_only():
	"""Global int *a, b; -- pointer binds only to a."""
	prog = parse("int *a, b;")
	assert len(prog.declarations) == 2
	a_decl = prog.declarations[0]
	b_decl = prog.declarations[1]
	assert isinstance(a_decl, VarDecl)
	assert isinstance(b_decl, VarDecl)
	assert a_decl.type_spec.pointer_count == 1
	assert b_decl.type_spec.pointer_count == 0


def test_decl_multi_ptr_double_pointer_first():
	"""int **a, b; -- double pointer binds only to a."""
	prog = parse("int **a, b;")
	a_decl = prog.declarations[0]
	b_decl = prog.declarations[1]
	assert a_decl.type_spec.pointer_count == 2
	assert b_decl.type_spec.pointer_count == 0


def test_decl_multi_ptr_independent_stars():
	"""int *a, *b; -- each declarator gets its own pointer."""
	prog = parse("int *a, *b;")
	a_decl = prog.declarations[0]
	b_decl = prog.declarations[1]
	assert a_decl.type_spec.pointer_count == 1
	assert b_decl.type_spec.pointer_count == 1


def test_decl_multi_ptr_mixed():
	"""int a, *b; -- only b is a pointer."""
	prog = parse("int a, *b;")
	a_decl = prog.declarations[0]
	b_decl = prog.declarations[1]
	assert a_decl.type_spec.pointer_count == 0
	assert b_decl.type_spec.pointer_count == 1
