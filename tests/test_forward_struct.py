"""Tests for struct and union forward declarations."""

import pytest

from compiler.ast_nodes import StructDecl, UnionDecl
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer, SemanticError


def parse(source: str):
	tokens = Lexer(source).tokenize()
	return Parser(tokens).parse()


def analyze(source: str):
	prog = parse(source)
	analyzer = SemanticAnalyzer()
	analyzer.analyze(prog)
	return prog, analyzer


# ---------------------------------------------------------------
# Parser: struct forward declarations
# ---------------------------------------------------------------

class TestParserForwardStruct:
	def test_basic_forward_struct(self):
		prog = parse("struct Node;")
		assert len(prog.declarations) == 1
		decl = prog.declarations[0]
		assert isinstance(decl, StructDecl)
		assert decl.name == "Node"
		assert decl.members == []

	def test_forward_struct_followed_by_definition(self):
		prog = parse("""
			struct Node;
			struct Node {
				int value;
				struct Node *next;
			};
		""")
		assert len(prog.declarations) == 2
		fwd = prog.declarations[0]
		full = prog.declarations[1]
		assert isinstance(fwd, StructDecl)
		assert fwd.members == []
		assert isinstance(full, StructDecl)
		assert len(full.members) == 2

	def test_basic_forward_union(self):
		prog = parse("union Data;")
		assert len(prog.declarations) == 1
		decl = prog.declarations[0]
		assert isinstance(decl, UnionDecl)
		assert decl.name == "Data"
		assert decl.members == []

	def test_forward_union_followed_by_definition(self):
		prog = parse("""
			union Data;
			union Data {
				int i;
				float f;
			};
		""")
		assert len(prog.declarations) == 2
		fwd = prog.declarations[0]
		full = prog.declarations[1]
		assert isinstance(fwd, UnionDecl)
		assert fwd.members == []
		assert isinstance(full, UnionDecl)
		assert len(full.members) == 2

	def test_multiple_forward_declarations(self):
		prog = parse("""
			struct A;
			struct B;
			struct A { int x; };
			struct B { int y; };
		""")
		assert len(prog.declarations) == 4


# ---------------------------------------------------------------
# Semantic: forward declarations
# ---------------------------------------------------------------

class TestSemanticForwardStruct:
	def test_forward_decl_then_full_definition(self):
		"""Forward declaration followed by full definition should succeed."""
		analyze("""
			struct Node;
			struct Node {
				int value;
			};
			int main() {
				struct Node n;
				n.value = 42;
				return n.value;
			}
		""")

	def test_forward_decl_pointer_before_definition(self):
		"""Forward-declared struct should be usable in pointer types."""
		analyze("""
			struct Node;
			int main() {
				struct Node *p;
				return 0;
			}
			struct Node {
				int value;
			};
		""")

	def test_self_referential_struct(self):
		"""A struct with a pointer to itself (linked list node)."""
		analyze("""
			struct Node;
			struct Node {
				int value;
				struct Node *next;
			};
			int main() {
				struct Node n;
				n.value = 1;
				return 0;
			}
		""")

	def test_mutual_struct_references(self):
		"""Two structs referencing each other via pointers."""
		analyze("""
			struct A;
			struct B;
			struct A {
				int x;
				struct B *b_ptr;
			};
			struct B {
				int y;
				struct A *a_ptr;
			};
			int main() {
				struct A a;
				struct B b;
				a.x = 1;
				b.y = 2;
				return 0;
			}
		""")

	def test_repeated_forward_declaration(self):
		"""Multiple forward declarations of the same struct should be allowed."""
		analyze("""
			struct Node;
			struct Node;
			struct Node {
				int value;
			};
			int main() {
				struct Node n;
				n.value = 10;
				return 0;
			}
		""")

	def test_duplicate_full_definitions_error(self):
		"""Two full definitions of the same struct should be an error."""
		with pytest.raises(SemanticError, match="redefinition of struct 'Node'"):
			analyze("""
				struct Node {
					int value;
				};
				struct Node {
					int other;
				};
			""")

	def test_forward_then_duplicate_full_definitions_error(self):
		"""Forward decl + full def + another full def should error."""
		with pytest.raises(SemanticError, match="redefinition of struct 'Node'"):
			analyze("""
				struct Node;
				struct Node {
					int value;
				};
				struct Node {
					int other;
				};
			""")

	def test_forward_union_then_full_definition(self):
		"""Forward declaration of union followed by full definition."""
		analyze("""
			union Data;
			union Data {
				int i;
				float f;
			};
			int main() {
				union Data d;
				d.i = 42;
				return d.i;
			}
		""")

	def test_duplicate_union_full_definitions_error(self):
		"""Two full definitions of the same union should be an error."""
		with pytest.raises(SemanticError, match="redefinition of union 'Data'"):
			analyze("""
				union Data {
					int i;
				};
				union Data {
					float f;
				};
			""")

	def test_repeated_forward_union(self):
		"""Multiple forward declarations of the same union should be allowed."""
		analyze("""
			union Data;
			union Data;
			union Data {
				int i;
				float f;
			};
			int main() {
				union Data d;
				d.i = 1;
				return 0;
			}
		""")
