"""Tests for typedef support in parser and semantic analyzer."""

import pytest

from compiler.parser import Parser, ParseError
from compiler.semantic import SemanticAnalyzer, SemanticError


def parse(source: str):
	"""Parse source and return the AST program node."""
	return Parser.from_source(source).parse()


def parse_and_analyze(source: str):
	"""Parse source and run semantic analysis; return (program, analyzer)."""
	program = parse(source)
	analyzer = SemanticAnalyzer()
	analyzer.analyze(program)
	return program, analyzer


# ---------------------------------------------------------------------------
# Simple typedef
# ---------------------------------------------------------------------------

class TestSimpleTypedef:
	def test_typedef_int(self):
		src = "typedef int MyInt; MyInt foo() { MyInt x = 5; return x; }"
		prog, analyzer = parse_and_analyze(src)
		# The typedef should be registered
		assert "MyInt" in analyzer._typedef_types
		assert analyzer._typedef_types["MyInt"].base_type == "int"

	def test_typedef_char(self):
		src = "typedef char Byte; Byte foo() { Byte b = 'a'; return b; }"
		prog, analyzer = parse_and_analyze(src)
		assert "Byte" in analyzer._typedef_types
		assert analyzer._typedef_types["Byte"].base_type == "char"

	def test_typedef_void(self):
		src = "typedef void Void;"
		prog = parse(src)
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)
		assert "Void" in analyzer._typedef_types
		assert analyzer._typedef_types["Void"].base_type == "void"


# ---------------------------------------------------------------------------
# Pointer typedef
# ---------------------------------------------------------------------------

class TestPointerTypedef:
	def test_typedef_int_pointer(self):
		src = "typedef int* IntPtr; int foo() { int x = 5; IntPtr p = &x; return *p; }"
		prog, analyzer = parse_and_analyze(src)
		assert "IntPtr" in analyzer._typedef_types
		resolved = analyzer._typedef_types["IntPtr"]
		assert resolved.base_type == "int"
		assert resolved.pointer_count == 1

	def test_typedef_char_pointer(self):
		src = 'typedef char* String; int foo() { String s = "hello"; return 0; }'
		prog, analyzer = parse_and_analyze(src)
		assert analyzer._typedef_types["String"].base_type == "char"
		assert analyzer._typedef_types["String"].pointer_count == 1

	def test_typedef_double_pointer(self):
		src = "typedef int** IntPtrPtr; int foo() { int x = 1; int* p = &x; IntPtrPtr pp = &p; return 0; }"
		prog, analyzer = parse_and_analyze(src)
		assert analyzer._typedef_types["IntPtrPtr"].pointer_count == 2

	def test_pointer_to_typedef(self):
		"""typedef int MyInt; MyInt* should be int*."""
		src = "typedef int MyInt; int foo() { int x = 5; MyInt* p = &x; return *p; }"
		prog, analyzer = parse_and_analyze(src)
		assert "MyInt" in analyzer._typedef_types


# ---------------------------------------------------------------------------
# Struct typedef
# ---------------------------------------------------------------------------

class TestStructTypedef:
	def test_typedef_struct_inline(self):
		src = """
		typedef struct { int x; int y; } Point;
		int foo() {
			Point p;
			p.x = 1;
			p.y = 2;
			return p.x;
		}
		"""
		prog, analyzer = parse_and_analyze(src)
		assert "Point" in analyzer._typedef_types
		assert analyzer._typedef_types["Point"].base_type == "struct Point"

	def test_typedef_struct_named(self):
		src = """
		typedef struct Coord { int x; int y; } Coord;
		int foo() {
			Coord c;
			c.x = 10;
			return c.x;
		}
		"""
		prog, analyzer = parse_and_analyze(src)
		assert "Coord" in analyzer._typedef_types
		assert analyzer._typedef_types["Coord"].base_type == "struct Coord"

	def test_typedef_existing_struct(self):
		src = """
		struct Vec2 { int x; int y; };
		typedef struct Vec2 Vector;
		int foo() {
			Vector v;
			v.x = 3;
			return v.x;
		}
		"""
		prog, analyzer = parse_and_analyze(src)
		assert "Vector" in analyzer._typedef_types
		assert analyzer._typedef_types["Vector"].base_type == "struct Vec2"


# ---------------------------------------------------------------------------
# Typedef in variable declarations
# ---------------------------------------------------------------------------

class TestTypedefVarDecl:
	def test_typedef_local_var(self):
		src = "typedef int MyInt; int foo() { MyInt x = 42; return x; }"
		parse_and_analyze(src)

	def test_typedef_global_var(self):
		src = "typedef int MyInt; MyInt g = 10; int foo() { return g; }"
		parse_and_analyze(src)

	def test_typedef_var_with_init(self):
		src = "typedef int MyInt; int foo() { MyInt a = 1; MyInt b = a; return b; }"
		parse_and_analyze(src)


# ---------------------------------------------------------------------------
# Typedef in function parameters
# ---------------------------------------------------------------------------

class TestTypedefFuncParams:
	def test_typedef_param(self):
		src = "typedef int MyInt; int add(MyInt a, MyInt b) { return a + b; }"
		parse_and_analyze(src)

	def test_typedef_pointer_param(self):
		src = "typedef int* IntPtr; int deref(IntPtr p) { return *p; }"
		parse_and_analyze(src)


# ---------------------------------------------------------------------------
# Typedef in function return types
# ---------------------------------------------------------------------------

class TestTypedefReturnType:
	def test_typedef_return(self):
		src = "typedef int MyInt; MyInt foo() { return 42; }"
		parse_and_analyze(src)

	def test_typedef_pointer_return(self):
		src = """
		typedef int* IntPtr;
		IntPtr get_ptr() {
			int x = 5;
			return &x;
		}
		"""
		parse_and_analyze(src)


# ---------------------------------------------------------------------------
# Chained typedefs
# ---------------------------------------------------------------------------

class TestChainedTypedef:
	def test_chained_typedef(self):
		src = """
		typedef int MyInt;
		typedef MyInt MyInt2;
		int foo() { MyInt2 x = 42; return x; }
		"""
		prog, analyzer = parse_and_analyze(src)
		assert analyzer._typedef_types["MyInt"].base_type == "int"
		assert analyzer._typedef_types["MyInt2"].base_type == "int"

	def test_triple_chained(self):
		src = """
		typedef int A;
		typedef A B;
		typedef B C;
		int foo() { C x = 99; return x; }
		"""
		prog, analyzer = parse_and_analyze(src)
		assert analyzer._typedef_types["C"].base_type == "int"

	def test_chained_pointer_typedef(self):
		src = """
		typedef int* IntPtr;
		typedef IntPtr IntPtrAlias;
		int foo() { int x = 1; IntPtrAlias p = &x; return *p; }
		"""
		prog, analyzer = parse_and_analyze(src)
		assert analyzer._typedef_types["IntPtrAlias"].base_type == "int"
		assert analyzer._typedef_types["IntPtrAlias"].pointer_count == 1


# ---------------------------------------------------------------------------
# Typedef scope rules
# ---------------------------------------------------------------------------

class TestTypedefScope:
	def test_typedef_at_top_level(self):
		src = """
		typedef int MyInt;
		MyInt foo() { return 1; }
		MyInt bar() { return 2; }
		"""
		parse_and_analyze(src)

	def test_typedef_used_across_functions(self):
		src = """
		typedef int MyInt;
		MyInt add(MyInt a, MyInt b) { return a + b; }
		MyInt mul(MyInt a, MyInt b) { return a * b; }
		"""
		parse_and_analyze(src)

	def test_typedef_in_local_scope(self):
		"""Typedef inside function body is valid in C."""
		src = """
		int foo() {
			typedef int LocalInt;
			LocalInt x = 5;
			return x;
		}
		"""
		parse_and_analyze(src)


# ---------------------------------------------------------------------------
# Enum typedef
# ---------------------------------------------------------------------------

class TestEnumTypedef:
	def test_typedef_enum_inline(self):
		src = """
		typedef enum { RED, GREEN, BLUE } Color;
		int foo() { int c = RED; return c; }
		"""
		prog, analyzer = parse_and_analyze(src)
		assert "Color" in analyzer._typedef_types
		assert analyzer._typedef_types["Color"].base_type == "enum Color"

	def test_typedef_enum_named(self):
		src = """
		typedef enum Colors { RED, GREEN, BLUE } Color;
		int foo() { int c = GREEN; return c; }
		"""
		prog, analyzer = parse_and_analyze(src)
		assert "Color" in analyzer._typedef_types
		assert analyzer._typedef_types["Color"].base_type == "enum Colors"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestTypedefErrors:
	def test_redefinition_error(self):
		src = """
		typedef int MyInt;
		typedef char MyInt;
		int foo() { return 0; }
		"""
		with pytest.raises(SemanticError, match="redefinition of typedef 'MyInt'"):
			parse_and_analyze(src)

	def test_typedef_unknown_type_in_expr(self):
		"""Using an unknown name as type should fail at parse time."""
		src = "int foo() { UnknownType x = 5; return x; }"
		with pytest.raises(ParseError):
			parse(src)


# ---------------------------------------------------------------------------
# Typedef in cast and sizeof
# ---------------------------------------------------------------------------

class TestTypedefCastSizeof:
	def test_typedef_cast(self):
		src = """
		typedef int MyInt;
		int foo() {
			char c = 'a';
			MyInt x = (MyInt)c;
			return x;
		}
		"""
		parse_and_analyze(src)

	def test_typedef_sizeof(self):
		src = """
		typedef int MyInt;
		int foo() {
			int s = sizeof(MyInt);
			return s;
		}
		"""
		parse_and_analyze(src)


# ---------------------------------------------------------------------------
# Parser-level typedef AST node checks
# ---------------------------------------------------------------------------

class TestTypedefASTNode:
	def test_typedef_decl_node(self):
		from compiler.ast_nodes import TypedefDecl
		prog = parse("typedef int MyInt;")
		assert len(prog.declarations) == 1
		td = prog.declarations[0]
		assert isinstance(td, TypedefDecl)
		assert td.name == "MyInt"
		assert td.type_spec.base_type == "int"
		assert td.type_spec.pointer_count == 0

	def test_typedef_pointer_node(self):
		from compiler.ast_nodes import TypedefDecl
		prog = parse("typedef int* IntPtr;")
		td = prog.declarations[0]
		assert isinstance(td, TypedefDecl)
		assert td.name == "IntPtr"
		assert td.type_spec.base_type == "int"
		assert td.type_spec.pointer_count == 1

	def test_typedef_struct_node_has_struct_decl(self):
		from compiler.ast_nodes import TypedefDecl, StructDecl
		prog = parse("typedef struct { int x; } Point;")
		td = prog.declarations[0]
		assert isinstance(td, TypedefDecl)
		assert td.name == "Point"
		assert td.struct_decl is not None
		assert isinstance(td.struct_decl, StructDecl)
		assert len(td.struct_decl.members) == 1

	def test_typedef_enum_node_has_enum_decl(self):
		from compiler.ast_nodes import TypedefDecl, EnumDecl
		prog = parse("typedef enum { A, B, C } Letters;")
		td = prog.declarations[0]
		assert isinstance(td, TypedefDecl)
		assert td.name == "Letters"
		assert td.enum_decl is not None
		assert isinstance(td.enum_decl, EnumDecl)
		assert len(td.enum_decl.constants) == 3


# ---------------------------------------------------------------------------
# Typedef in for-loop init
# ---------------------------------------------------------------------------

class TestTypedefForLoop:
	def test_typedef_in_for_init(self):
		src = """
		typedef int MyInt;
		int foo() {
			int sum = 0;
			for (MyInt i = 0; i < 10; i++) {
				sum = sum + i;
			}
			return sum;
		}
		"""
		parse_and_analyze(src)
