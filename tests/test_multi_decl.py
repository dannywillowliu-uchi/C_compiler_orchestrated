"""Tests for multi-variable declaration lists in the parser."""

from compiler.parser import Parser
from compiler.ast_nodes import (
	CompoundStmt,
	ForStmt,
	FunctionDecl,
	IntLiteral,
	VarDecl,
)


class TestMultiDeclLocal:
	"""Test multi-variable declarations in local (function body) scope."""

	def _parse_func_body(self, source: str) -> CompoundStmt:
		prog = Parser.from_source(source).parse()
		func = prog.declarations[0]
		assert isinstance(func, FunctionDecl)
		assert func.body is not None
		return func.body

	def test_plain_vars(self):
		body = self._parse_func_body("int main() { int a, b; return 0; }")
		assert len(body.statements) == 3
		a, b = body.statements[0], body.statements[1]
		assert isinstance(a, VarDecl) and a.name == "a"
		assert isinstance(b, VarDecl) and b.name == "b"
		assert a.type_spec.base_type == "int" and a.type_spec.pointer_count == 0
		assert b.type_spec.base_type == "int" and b.type_spec.pointer_count == 0

	def test_three_vars(self):
		body = self._parse_func_body("int main() { int a, b, c; return 0; }")
		assert len(body.statements) == 4
		for i, name in enumerate(["a", "b", "c"]):
			decl = body.statements[i]
			assert isinstance(decl, VarDecl)
			assert decl.name == name
			assert decl.type_spec.base_type == "int"

	def test_mixed_pointers(self):
		body = self._parse_func_body("int main() { int *a, b; return 0; }")
		a, b = body.statements[0], body.statements[1]
		assert isinstance(a, VarDecl) and a.type_spec.pointer_count == 1
		assert isinstance(b, VarDecl) and b.type_spec.pointer_count == 0

	def test_both_pointers(self):
		body = self._parse_func_body("int main() { int *a, *b; return 0; }")
		a, b = body.statements[0], body.statements[1]
		assert isinstance(a, VarDecl) and a.type_spec.pointer_count == 1
		assert isinstance(b, VarDecl) and b.type_spec.pointer_count == 1

	def test_double_pointer(self):
		body = self._parse_func_body("int main() { int **a, *b, c; return 0; }")
		a, b, c = body.statements[0], body.statements[1], body.statements[2]
		assert isinstance(a, VarDecl) and a.type_spec.pointer_count == 2
		assert isinstance(b, VarDecl) and b.type_spec.pointer_count == 1
		assert isinstance(c, VarDecl) and c.type_spec.pointer_count == 0

	def test_with_initializers(self):
		body = self._parse_func_body("int main() { int a = 1, b = 2; return 0; }")
		a, b = body.statements[0], body.statements[1]
		assert isinstance(a, VarDecl)
		assert isinstance(a.initializer, IntLiteral) and a.initializer.value == 1
		assert isinstance(b, VarDecl)
		assert isinstance(b.initializer, IntLiteral) and b.initializer.value == 2

	def test_partial_initializers(self):
		body = self._parse_func_body("int main() { int a, b = 5; return 0; }")
		a, b = body.statements[0], body.statements[1]
		assert isinstance(a, VarDecl) and a.initializer is None
		assert isinstance(b, VarDecl)
		assert isinstance(b.initializer, IntLiteral) and b.initializer.value == 5

	def test_arrays(self):
		body = self._parse_func_body("int main() { int a[10], b; return 0; }")
		a, b = body.statements[0], body.statements[1]
		assert isinstance(a, VarDecl)
		assert a.array_sizes is not None and len(a.array_sizes) == 1
		assert isinstance(b, VarDecl)
		assert b.array_sizes is None

	def test_array_with_pointer(self):
		body = self._parse_func_body("int main() { int a[10], *b; return 0; }")
		a, b = body.statements[0], body.statements[1]
		assert isinstance(a, VarDecl)
		assert a.array_sizes is not None
		assert a.type_spec.pointer_count == 0
		assert isinstance(b, VarDecl)
		assert b.array_sizes is None
		assert b.type_spec.pointer_count == 1

	def test_combination(self):
		body = self._parse_func_body("int main() { int *a, b[5], c = 3; return 0; }")
		a, b, c = body.statements[0], body.statements[1], body.statements[2]
		assert isinstance(a, VarDecl) and a.type_spec.pointer_count == 1
		assert isinstance(b, VarDecl) and b.array_sizes is not None
		assert isinstance(c, VarDecl) and isinstance(c.initializer, IntLiteral) and c.initializer.value == 3

	def test_single_decl_still_works(self):
		body = self._parse_func_body("int main() { int x = 42; return 0; }")
		assert len(body.statements) == 2
		x = body.statements[0]
		assert isinstance(x, VarDecl) and x.name == "x"
		assert isinstance(x.initializer, IntLiteral) and x.initializer.value == 42

	def test_char_type(self):
		body = self._parse_func_body("int main() { char a, *b; return 0; }")
		a, b = body.statements[0], body.statements[1]
		assert isinstance(a, VarDecl) and a.type_spec.base_type == "char"
		assert isinstance(b, VarDecl) and b.type_spec.base_type == "char" and b.type_spec.pointer_count == 1


class TestMultiDeclGlobal:
	"""Test multi-variable declarations at global scope."""

	def test_global_plain_vars(self):
		prog = Parser.from_source("int a, b;").parse()
		assert len(prog.declarations) == 2
		assert all(isinstance(d, VarDecl) for d in prog.declarations)
		assert prog.declarations[0].name == "a"
		assert prog.declarations[1].name == "b"

	def test_global_mixed_pointers(self):
		prog = Parser.from_source("int *a, b;").parse()
		assert len(prog.declarations) == 2
		assert prog.declarations[0].type_spec.pointer_count == 1
		assert prog.declarations[1].type_spec.pointer_count == 0

	def test_global_with_initializers(self):
		prog = Parser.from_source("int a = 1, b = 2;").parse()
		assert len(prog.declarations) == 2
		assert isinstance(prog.declarations[0].initializer, IntLiteral)
		assert prog.declarations[0].initializer.value == 1
		assert isinstance(prog.declarations[1].initializer, IntLiteral)
		assert prog.declarations[1].initializer.value == 2

	def test_global_arrays(self):
		prog = Parser.from_source("int a[10], b[20];").parse()
		assert len(prog.declarations) == 2
		assert prog.declarations[0].array_sizes is not None
		assert prog.declarations[1].array_sizes is not None

	def test_global_single_still_works(self):
		prog = Parser.from_source("int x;").parse()
		assert len(prog.declarations) == 1
		assert isinstance(prog.declarations[0], VarDecl) and prog.declarations[0].name == "x"

	def test_global_with_function(self):
		prog = Parser.from_source("int a, b; int main() { return 0; }").parse()
		assert len(prog.declarations) == 3
		assert isinstance(prog.declarations[0], VarDecl) and prog.declarations[0].name == "a"
		assert isinstance(prog.declarations[1], VarDecl) and prog.declarations[1].name == "b"
		assert isinstance(prog.declarations[2], FunctionDecl)


class TestMultiDeclForLoop:
	"""Test multi-variable declarations in for-loop init clauses."""

	def _parse_for_stmt(self, source: str) -> ForStmt:
		prog = Parser.from_source(source).parse()
		func = prog.declarations[0]
		assert isinstance(func, FunctionDecl) and func.body is not None
		for_stmt = func.body.statements[0]
		assert isinstance(for_stmt, ForStmt)
		return for_stmt

	def test_for_multi_decl(self):
		for_stmt = self._parse_for_stmt(
			"int main() { for (int i = 0, j = 10; i < j; i++) { } return 0; }"
		)
		assert isinstance(for_stmt.init, list)
		assert len(for_stmt.init) == 2
		assert isinstance(for_stmt.init[0], VarDecl) and for_stmt.init[0].name == "i"
		assert isinstance(for_stmt.init[1], VarDecl) and for_stmt.init[1].name == "j"

	def test_for_single_decl_still_works(self):
		for_stmt = self._parse_for_stmt(
			"int main() { for (int i = 0; i < 10; i++) { } return 0; }"
		)
		assert isinstance(for_stmt.init, VarDecl)
		assert for_stmt.init.name == "i"

	def test_for_multi_decl_with_pointers(self):
		for_stmt = self._parse_for_stmt(
			"int main() { for (int *p = 0, i = 0; i < 10; i++) { } return 0; }"
		)
		assert isinstance(for_stmt.init, list)
		assert for_stmt.init[0].type_spec.pointer_count == 1
		assert for_stmt.init[1].type_spec.pointer_count == 0


class TestMultiDeclEdgeCases:
	"""Test edge cases and special types."""

	def test_unsigned_multi_decl(self):
		prog = Parser.from_source("int main() { unsigned int a, b; return 0; }").parse()
		func = prog.declarations[0]
		body = func.body
		a, b = body.statements[0], body.statements[1]
		assert isinstance(a, VarDecl) and a.type_spec.signedness == "unsigned"
		assert isinstance(b, VarDecl) and b.type_spec.signedness == "unsigned"

	def test_long_multi_decl(self):
		prog = Parser.from_source("int main() { long a, b; return 0; }").parse()
		func = prog.declarations[0]
		body = func.body
		a, b = body.statements[0], body.statements[1]
		assert isinstance(a, VarDecl) and a.type_spec.width_modifier == "long"
		assert isinstance(b, VarDecl) and b.type_spec.width_modifier == "long"

	def test_struct_multi_decl(self):
		source = "struct Foo { int x; }; int main() { struct Foo a, *b; return 0; }"
		prog = Parser.from_source(source).parse()
		func = prog.declarations[1]
		body = func.body
		a, b = body.statements[0], body.statements[1]
		assert isinstance(a, VarDecl) and a.type_spec.base_type == "struct Foo"
		assert isinstance(b, VarDecl) and b.type_spec.base_type == "struct Foo"
		assert a.type_spec.pointer_count == 0
		assert b.type_spec.pointer_count == 1

	def test_global_struct_multi_decl(self):
		source = "struct Foo { int x; }; struct Foo a, b;"
		prog = Parser.from_source(source).parse()
		assert len(prog.declarations) == 3
		assert isinstance(prog.declarations[1], VarDecl) and prog.declarations[1].name == "a"
		assert isinstance(prog.declarations[2], VarDecl) and prog.declarations[2].name == "b"
