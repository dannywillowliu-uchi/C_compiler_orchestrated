"""Torture tests for typedef alias resolution through the full pipeline."""

from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer


def _analyze(source: str) -> None:
	prog = Parser.from_source(source).parse()
	SemanticAnalyzer().analyze(prog)


class TestMiscTypedef:
	def test_typedef_struct_member_read(self) -> None:
		_analyze("""
			typedef int i32;
			struct Vec { i32 x; i32 y; };
			int main() {
				struct Vec v;
				v.x = 10;
				int result = v.x;
				return result;
			}
		""")

	def test_typedef_struct_member_write(self) -> None:
		_analyze("""
			typedef int i32;
			struct Pair { i32 a; i32 b; };
			int main() {
				struct Pair p;
				p.a = 1;
				p.b = 2;
				return p.a + p.b;
			}
		""")

	def test_typedef_struct_member_init(self) -> None:
		_analyze("""
			typedef int num;
			struct Box { num val; };
			int main() {
				struct Box b;
				b.val = 42;
				num x = b.val;
				return x;
			}
		""")

	def test_typedef_chain(self) -> None:
		_analyze("""
			typedef int i32;
			typedef i32 myint;
			int main() {
				myint x = 5;
				int y = x;
				return y;
			}
		""")

	def test_typedef_pointer(self) -> None:
		_analyze("""
			typedef int* intptr;
			int main() {
				int a = 5;
				intptr p = &a;
				return *p;
			}
		""")

	def test_typedef_in_function_params(self) -> None:
		_analyze("""
			typedef int i32;
			i32 add(i32 a, i32 b) { return a + b; }
			int main() {
				int result = add(1, 2);
				return result;
			}
		""")

	def test_typedef_in_return_type(self) -> None:
		_analyze("""
			typedef int i32;
			i32 identity(int x) { return x; }
			int main() {
				i32 y = identity(42);
				return y;
			}
		""")
