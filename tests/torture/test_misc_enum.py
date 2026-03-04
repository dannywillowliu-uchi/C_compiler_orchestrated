"""Torture tests for enum/int interoperability through the full pipeline."""

from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer


def _analyze(source: str) -> None:
	prog = Parser.from_source(source).parse()
	SemanticAnalyzer().analyze(prog)


class TestMiscEnum:
	def test_enum_var_init_from_constant(self) -> None:
		_analyze("""
			enum Color { RED, GREEN, BLUE };
			int main() { enum Color c = RED; return 0; }
		""")

	def test_enum_var_assigned_to_int(self) -> None:
		_analyze("""
			enum Color { RED, GREEN, BLUE };
			int main() {
				enum Color c = GREEN;
				int x = c;
				return x;
			}
		""")

	def test_int_assigned_to_enum(self) -> None:
		_analyze("""
			enum Color { RED, GREEN, BLUE };
			int main() {
				int x = 1;
				enum Color c = x;
				return 0;
			}
		""")

	def test_enum_in_arithmetic(self) -> None:
		_analyze("""
			enum Dir { UP, DOWN, LEFT, RIGHT };
			int main() {
				enum Dir d = UP;
				int result = d + 1;
				return result;
			}
		""")

	def test_enum_in_comparison(self) -> None:
		_analyze("""
			enum State { OFF, ON };
			int main() {
				enum State s = ON;
				if (s == 1) { return 1; }
				return 0;
			}
		""")

	def test_enum_assignment(self) -> None:
		_analyze("""
			enum Color { RED, GREEN, BLUE };
			int main() {
				enum Color c = RED;
				c = 2;
				return c;
			}
		""")

	def test_typedef_enum(self) -> None:
		_analyze("""
			typedef enum { A, B, C } Letter;
			int main() {
				Letter x = A;
				int y = x;
				return y;
			}
		""")

	def test_enum_switch(self) -> None:
		_analyze("""
			enum Color { RED, GREEN, BLUE };
			int main() {
				enum Color c = RED;
				switch (c) {
					case 0: return 0;
					case 1: return 1;
					default: return 2;
				}
			}
		""")
