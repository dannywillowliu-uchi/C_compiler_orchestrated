"""Tests for C99 compound literal support."""

from compiler.ast_nodes import CompoundLiteral
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer
from compiler.ir_gen import IRGenerator


def _parse(source: str):
	return Parser.from_source(source).parse()


def _analyze(source: str):
	program = _parse(source)
	analyzer = SemanticAnalyzer()
	errors = analyzer.analyze(program)
	return program, errors


def _generate_ir(source: str):
	program = _parse(source)
	analyzer = SemanticAnalyzer()
	analyzer.analyze(program)
	gen = IRGenerator()
	return gen.generate(program)


class TestCompoundLiteralParsing:
	"""Test that compound literals are correctly parsed into AST nodes."""

	def test_struct_compound_literal(self):
		src = """
		struct Point { int x; int y; };
		int main(void) {
			struct Point p = (struct Point){1, 2};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[1]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl.initializer, CompoundLiteral)
		assert var_decl.initializer.type_spec.base_type == "struct Point"
		assert len(var_decl.initializer.init_list.elements) == 2

	def test_int_array_compound_literal(self):
		src = """
		int main(void) {
			int *p = (int[]){10, 20, 30};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[0]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl.initializer, CompoundLiteral)

	def test_scalar_compound_literal(self):
		src = """
		int main(void) {
			int x = (int){42};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[0]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl.initializer, CompoundLiteral)
		assert var_decl.initializer.type_spec.base_type == "int"

	def test_designated_init_compound_literal(self):
		src = """
		struct Point { int x; int y; };
		int main(void) {
			struct Point p = (struct Point){.y = 5, .x = 3};
			return 0;
		}
		"""
		program = _parse(src)
		func = program.declarations[1]
		var_decl = func.body.statements[0]
		assert isinstance(var_decl.initializer, CompoundLiteral)
		assert len(var_decl.initializer.init_list.elements) == 2

	def test_compound_literal_in_expression(self):
		src = """
		struct Point { int x; int y; };
		int main(void) {
			int x = (struct Point){1, 2}.x;
			return 0;
		}
		"""
		# Compound literal used with member access - parsed via postfix
		program = _parse(src)
		func = program.declarations[1]
		var_decl = func.body.statements[0]
		# The initializer should be a MemberAccess of a CompoundLiteral
		from compiler.ast_nodes import MemberAccess
		assert isinstance(var_decl.initializer, MemberAccess)
		assert isinstance(var_decl.initializer.object, CompoundLiteral)


class TestCompoundLiteralSemantic:
	"""Test semantic analysis of compound literals."""

	def test_struct_compound_literal_no_errors(self):
		src = """
		struct Point { int x; int y; };
		int main(void) {
			struct Point p = (struct Point){1, 2};
			return 0;
		}
		"""
		_, errors = _analyze(src)
		assert len(errors) == 0

	def test_scalar_compound_literal_no_errors(self):
		src = """
		int main(void) {
			int x = (int){42};
			return 0;
		}
		"""
		_, errors = _analyze(src)
		assert len(errors) == 0

	def test_designated_compound_literal_no_errors(self):
		src = """
		struct Point { int x; int y; };
		int main(void) {
			struct Point p = (struct Point){.x = 1, .y = 2};
			return 0;
		}
		"""
		_, errors = _analyze(src)
		assert len(errors) == 0


class TestCompoundLiteralIRGen:
	"""Test IR generation for compound literals."""

	def test_struct_compound_literal_ir(self):
		src = """
		struct Point { int x; int y; };
		int main(void) {
			struct Point p = (struct Point){10, 20};
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		assert len(ir_prog.functions) > 0
		main_fn = ir_prog.functions[0]
		# Should have ALLOC instructions for both p and the compound literal
		from compiler.ir import IRAlloc, IRStore
		allocs = [i for i in main_fn.body if isinstance(i, IRAlloc)]
		stores = [i for i in main_fn.body if isinstance(i, IRStore)]
		assert len(allocs) >= 2  # one for p, one for compound literal
		assert len(stores) >= 2  # stores for x=10 and y=20

	def test_scalar_compound_literal_ir(self):
		src = """
		int main(void) {
			int x = (int){42};
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		main_fn = ir_prog.functions[0]
		from compiler.ir import IRAlloc
		allocs = [i for i in main_fn.body if isinstance(i, IRAlloc)]
		assert len(allocs) >= 2  # one for x, one for the compound literal

	def test_union_compound_literal_ir(self):
		src = """
		union Data { int i; char c; };
		int main(void) {
			union Data d = (union Data){42};
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		main_fn = ir_prog.functions[0]
		from compiler.ir import IRAlloc, IRStore
		allocs = [i for i in main_fn.body if isinstance(i, IRAlloc)]
		stores = [i for i in main_fn.body if isinstance(i, IRStore)]
		assert len(allocs) >= 2
		assert len(stores) >= 1

	def test_array_compound_literal_ir(self):
		src = """
		int main(void) {
			int *p = (int[]){1, 2, 3};
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		main_fn = ir_prog.functions[0]
		from compiler.ir import IRAlloc, IRStore
		allocs = [i for i in main_fn.body if isinstance(i, IRAlloc)]
		stores = [i for i in main_fn.body if isinstance(i, IRStore)]
		assert len(allocs) >= 2
		assert len(stores) >= 3  # stores for 1, 2, 3

	def test_designated_struct_compound_literal_ir(self):
		src = """
		struct Point { int x; int y; };
		int main(void) {
			struct Point p = (struct Point){.y = 99, .x = 42};
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		main_fn = ir_prog.functions[0]
		from compiler.ir import IRStore
		stores = [i for i in main_fn.body if isinstance(i, IRStore)]
		assert len(stores) >= 2

	def test_compound_literal_as_function_arg(self):
		src = """
		struct Point { int x; int y; };
		int use_point(struct Point *p) { return p->x; }
		int main(void) {
			int r = use_point(&(struct Point){5, 6});
			return r;
		}
		"""
		ir_prog = _generate_ir(src)
		assert len(ir_prog.functions) >= 2
