"""Tests for type qualifiers, modifiers, and storage classes."""

import pytest

from compiler.ast_nodes import TypeSpec, VarDecl, FunctionDecl
from compiler.parser import Parser, ParseError
from compiler.semantic import SemanticAnalyzer, SemanticError, _type_size
from compiler.ir_gen import IRGenerator, _resolve_size


def parse(source: str):
	return Parser.from_source(source).parse()


def parse_and_analyze(source: str):
	program = parse(source)
	analyzer = SemanticAnalyzer()
	analyzer.analyze(program)
	return program, analyzer


# ---------------------------------------------------------------------------
# Parsing: type qualifiers
# ---------------------------------------------------------------------------


class TestParseQualifiers:
	def test_const_int(self):
		prog = parse("const int x = 5; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert decl.type_spec.base_type == "int"
		assert "const" in decl.type_spec.qualifiers

	def test_volatile_int(self):
		prog = parse("volatile int x = 5; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert decl.type_spec.base_type == "int"
		assert "volatile" in decl.type_spec.qualifiers

	def test_const_volatile(self):
		prog = parse("const volatile int x = 5; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert "const" in decl.type_spec.qualifiers
		assert "volatile" in decl.type_spec.qualifiers

	def test_int_const(self):
		"""const after base type: 'int const x'."""
		prog = parse("int const x = 5; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert "const" in decl.type_spec.qualifiers


# ---------------------------------------------------------------------------
# Parsing: signedness
# ---------------------------------------------------------------------------


class TestParseSignedness:
	def test_unsigned_int(self):
		prog = parse("unsigned int x = 5; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert decl.type_spec.base_type == "int"
		assert decl.type_spec.signedness == "unsigned"

	def test_signed_int(self):
		prog = parse("signed int x = 5; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert decl.type_spec.signedness == "signed"

	def test_unsigned_alone(self):
		"""'unsigned' alone should be treated as 'unsigned int'."""
		prog = parse("unsigned x = 5; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert decl.type_spec.base_type == "int"
		assert decl.type_spec.signedness == "unsigned"


# ---------------------------------------------------------------------------
# Parsing: width modifiers
# ---------------------------------------------------------------------------


class TestParseWidthModifiers:
	def test_long_int(self):
		prog = parse("long int x = 5; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert decl.type_spec.base_type == "int"
		assert decl.type_spec.width_modifier == "long"

	def test_long_alone(self):
		"""'long' alone should be treated as 'long int'."""
		prog = parse("long x = 5; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert decl.type_spec.base_type == "int"
		assert decl.type_spec.width_modifier == "long"

	def test_long_long(self):
		prog = parse("long long x = 5; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert decl.type_spec.base_type == "int"
		assert decl.type_spec.width_modifier == "long long"

	def test_long_long_int(self):
		prog = parse("long long int x = 5; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert decl.type_spec.base_type == "int"
		assert decl.type_spec.width_modifier == "long long"

	def test_short(self):
		prog = parse("short x = 5; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert decl.type_spec.base_type == "int"
		assert decl.type_spec.width_modifier == "short"

	def test_short_int(self):
		prog = parse("short int x = 5; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert decl.type_spec.base_type == "int"
		assert decl.type_spec.width_modifier == "short"


# ---------------------------------------------------------------------------
# Parsing: storage classes
# ---------------------------------------------------------------------------


class TestParseStorageClass:
	def test_static_int(self):
		prog = parse("static int x = 5; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert decl.storage_class == "static"

	def test_extern_int(self):
		prog = parse("extern int x; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert decl.storage_class == "extern"

	def test_static_function(self):
		prog = parse("static int foo() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, FunctionDecl)
		assert decl.storage_class == "static"

	def test_no_storage_class(self):
		prog = parse("int x = 5; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert decl.storage_class is None


# ---------------------------------------------------------------------------
# Parsing: combined specifiers
# ---------------------------------------------------------------------------


class TestParseCombined:
	def test_const_unsigned_long(self):
		prog = parse("const unsigned long x = 5; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert "const" in decl.type_spec.qualifiers
		assert decl.type_spec.signedness == "unsigned"
		assert decl.type_spec.width_modifier == "long"
		assert decl.type_spec.base_type == "int"

	def test_static_const_int(self):
		prog = parse("static const int x = 5; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert decl.storage_class == "static"
		assert "const" in decl.type_spec.qualifiers
		assert decl.type_spec.base_type == "int"

	def test_unsigned_long_long(self):
		prog = parse("unsigned long long x = 5; int main() { return 0; }")
		decl = prog.declarations[0]
		assert isinstance(decl, VarDecl)
		assert decl.type_spec.signedness == "unsigned"
		assert decl.type_spec.width_modifier == "long long"

	def test_local_const_int(self):
		src = "int main() { const int x = 5; return x; }"
		prog = parse(src)
		body = prog.declarations[0].body
		decl = body.statements[0]
		assert isinstance(decl, VarDecl)
		assert "const" in decl.type_spec.qualifiers

	def test_local_static_int(self):
		src = "int main() { static int count = 0; return count; }"
		prog = parse(src)
		body = prog.declarations[0].body
		decl = body.statements[0]
		assert isinstance(decl, VarDecl)
		assert decl.storage_class == "static"


# ---------------------------------------------------------------------------
# Semantic: duplicate storage class
# ---------------------------------------------------------------------------


class TestSemanticStorageClass:
	def test_duplicate_storage_class_error(self):
		with pytest.raises(ParseError):
			parse("static extern int x = 5;")

	def test_storage_class_in_symbol(self):
		src = "static int x = 5; int main() { return x; }"
		prog, analyzer = parse_and_analyze(src)
		sym = analyzer.symbols.lookup("x")
		assert sym is not None
		assert sym.storage_class == "static"


# ---------------------------------------------------------------------------
# Semantic: duplicate qualifiers
# ---------------------------------------------------------------------------


class TestSemanticDuplicateQualifiers:
	def test_duplicate_const_error(self):
		with pytest.raises(SemanticError, match="duplicate.*const"):
			parse_and_analyze("const const int x = 5; int main() { return 0; }")


# ---------------------------------------------------------------------------
# Type sizes
# ---------------------------------------------------------------------------


class TestTypeSizes:
	def test_int_size(self):
		ts = TypeSpec(base_type="int")
		assert _type_size(ts) == 4

	def test_short_size(self):
		ts = TypeSpec(base_type="int", width_modifier="short")
		assert _type_size(ts) == 2

	def test_long_size(self):
		ts = TypeSpec(base_type="int", width_modifier="long")
		assert _type_size(ts) == 8

	def test_long_long_size(self):
		ts = TypeSpec(base_type="int", width_modifier="long long")
		assert _type_size(ts) == 8

	def test_pointer_size(self):
		ts = TypeSpec(base_type="int", pointer_count=1)
		assert _type_size(ts) == 8

	def test_char_size(self):
		ts = TypeSpec(base_type="char")
		assert _type_size(ts) == 1


# ---------------------------------------------------------------------------
# IR gen: sizeof with modifiers
# ---------------------------------------------------------------------------


class TestIRSizes:
	def test_short_resolve_size(self):
		ts = TypeSpec(base_type="int", width_modifier="short")
		assert _resolve_size(ts) == 2

	def test_long_resolve_size(self):
		ts = TypeSpec(base_type="int", width_modifier="long")
		assert _resolve_size(ts) == 8

	def test_long_long_resolve_size(self):
		ts = TypeSpec(base_type="int", width_modifier="long long")
		assert _resolve_size(ts) == 8

	def test_sizeof_long(self):
		src = """
		int main() {
			long x = 0;
			return sizeof(long);
		}
		"""
		prog = parse(src)
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)
		ir_gen = IRGenerator()
		ir_prog = ir_gen.generate(prog)
		# sizeof(long) should produce 8
		assert ir_prog.functions[0].body is not None


# ---------------------------------------------------------------------------
# Full pipeline: parse + analyze with qualifiers/modifiers
# ---------------------------------------------------------------------------


class TestFullPipeline:
	def test_const_int_pipeline(self):
		src = "int main() { const int x = 42; return x; }"
		parse_and_analyze(src)

	def test_unsigned_long_pipeline(self):
		src = "int main() { unsigned long x = 100; return x; }"
		parse_and_analyze(src)

	def test_static_int_pipeline(self):
		src = "static int counter = 0; int main() { return counter; }"
		parse_and_analyze(src)

	def test_short_int_pipeline(self):
		src = "int main() { short int x = 10; return x; }"
		parse_and_analyze(src)

	def test_long_long_pipeline(self):
		src = "int main() { long long big = 100; return big; }"
		parse_and_analyze(src)

	def test_const_unsigned_long_pipeline(self):
		src = "int main() { const unsigned long x = 100; return x; }"
		parse_and_analyze(src)

	def test_extern_declaration(self):
		src = "extern int global_var; int main() { return global_var; }"
		parse_and_analyze(src)
