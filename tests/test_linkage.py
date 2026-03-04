"""Tests for static/extern linkage directives and ELF metadata in codegen."""

from compiler.codegen import CodeGenerator
from compiler.ir import (
	IRConst,
	IRFunction,
	IRGlobalVar,
	IRProgram,
	IRReturn,
	IRType,
)


def _generate(program: IRProgram) -> str:
	"""Helper to generate assembly from an IR program."""
	return CodeGenerator().generate(program)


# ------------------------------------------------------------------
# Function linkage
# ------------------------------------------------------------------


class TestFunctionLinkage:
	"""Tests for .globl, .type, and .size directives on functions."""

	def test_default_function_has_globl(self) -> None:
		"""A normal (non-static) function should have .globl."""
		func = IRFunction(
			name="foo", params=[], body=[IRReturn()],
			return_type=IRType.VOID,
		)
		asm = _generate(IRProgram(functions=[func]))
		assert ".globl foo" in asm

	def test_static_function_no_globl(self) -> None:
		"""A static function should NOT have .globl."""
		func = IRFunction(
			name="helper", params=[], body=[IRReturn()],
			return_type=IRType.VOID, storage_class="static",
		)
		asm = _generate(IRProgram(functions=[func]))
		assert ".globl helper" not in asm
		# But it should still have a label and body
		assert "helper:" in asm
		assert "ret" in asm

	def test_extern_function_no_body(self) -> None:
		"""An extern function declaration should not emit any body."""
		func = IRFunction(
			name="printf", params=[], body=[],
			return_type=IRType.INT, storage_class="extern", is_prototype=True,
		)
		asm = _generate(IRProgram(functions=[func]))
		assert "printf:" not in asm
		assert ".globl printf" not in asm

	def test_prototype_only_no_body(self) -> None:
		"""A prototype-only function (no body) should not emit code."""
		func = IRFunction(
			name="bar", params=[], body=[],
			return_type=IRType.INT, is_prototype=True,
		)
		asm = _generate(IRProgram(functions=[func]))
		assert "bar:" not in asm

	def test_function_type_annotation(self) -> None:
		"""Functions should have .type funcname, @function."""
		func = IRFunction(
			name="compute", params=[], body=[IRReturn(value=IRConst(42))],
			return_type=IRType.INT,
		)
		asm = _generate(IRProgram(functions=[func]))
		assert ".type compute, @function" in asm

	def test_function_size_directive(self) -> None:
		"""Functions should have .size funcname, .-funcname."""
		func = IRFunction(
			name="compute", params=[], body=[IRReturn(value=IRConst(42))],
			return_type=IRType.INT,
		)
		asm = _generate(IRProgram(functions=[func]))
		assert ".size compute, .-compute" in asm

	def test_static_function_has_type_and_size(self) -> None:
		"""Static functions still get .type and .size but no .globl."""
		func = IRFunction(
			name="internal", params=[], body=[IRReturn()],
			return_type=IRType.VOID, storage_class="static",
		)
		asm = _generate(IRProgram(functions=[func]))
		assert ".globl internal" not in asm
		assert ".type internal, @function" in asm
		assert ".size internal, .-internal" in asm

	def test_multiple_functions_mixed_linkage(self) -> None:
		"""Mix of static, extern, and default functions."""
		funcs = [
			IRFunction(
				name="public_fn", params=[], body=[IRReturn()],
				return_type=IRType.VOID,
			),
			IRFunction(
				name="static_fn", params=[], body=[IRReturn()],
				return_type=IRType.VOID, storage_class="static",
			),
			IRFunction(
				name="extern_fn", params=[], body=[],
				return_type=IRType.INT, storage_class="extern", is_prototype=True,
			),
		]
		asm = _generate(IRProgram(functions=funcs))
		# public_fn: .globl + body
		assert ".globl public_fn" in asm
		assert "public_fn:" in asm
		# static_fn: no .globl + body
		assert ".globl static_fn" not in asm
		assert "static_fn:" in asm
		# extern_fn: no body at all
		assert "extern_fn:" not in asm


# ------------------------------------------------------------------
# Global variable linkage
# ------------------------------------------------------------------


class TestGlobalVarLinkage:
	"""Tests for .globl directives on global variables."""

	def test_default_global_has_globl(self) -> None:
		"""A normal global variable should have .globl."""
		gvar = IRGlobalVar(name="counter", ir_type=IRType.INT, initializer=0)
		asm = _generate(IRProgram(globals=[gvar]))
		assert ".globl counter" in asm

	def test_static_global_no_globl(self) -> None:
		"""A static global variable should NOT have .globl."""
		gvar = IRGlobalVar(
			name="file_local", ir_type=IRType.INT, initializer=42,
			storage_class="static",
		)
		asm = _generate(IRProgram(globals=[gvar]))
		assert ".globl file_local" not in asm
		# But the label and data should still be emitted
		assert "file_local:" in asm
		assert ".long 42" in asm

	def test_extern_global_no_storage(self) -> None:
		"""An extern global should not allocate storage (no .bss entry)."""
		gvar = IRGlobalVar(
			name="ext_var", ir_type=IRType.INT,
			storage_class="extern",
		)
		asm = _generate(IRProgram(globals=[gvar]))
		assert "ext_var:" not in asm

	def test_static_uninitialized_global(self) -> None:
		"""A static uninitialized global goes to .bss without .globl."""
		gvar = IRGlobalVar(
			name="bss_local", ir_type=IRType.INT,
			storage_class="static",
		)
		asm = _generate(IRProgram(globals=[gvar]))
		assert ".globl bss_local" not in asm
		assert "bss_local:" in asm
		assert ".zero" in asm

	def test_mixed_global_linkage(self) -> None:
		"""Mix of static, extern, and default globals."""
		gvars = [
			IRGlobalVar(name="pub_var", ir_type=IRType.INT, initializer=10),
			IRGlobalVar(name="priv_var", ir_type=IRType.INT, initializer=20, storage_class="static"),
			IRGlobalVar(name="ext_var", ir_type=IRType.INT, storage_class="extern"),
		]
		asm = _generate(IRProgram(globals=gvars))
		assert ".globl pub_var" in asm
		assert ".globl priv_var" not in asm
		assert "ext_var:" not in asm


# ------------------------------------------------------------------
# End-to-end: parse -> ir_gen -> codegen
# ------------------------------------------------------------------


class TestLinkageEndToEnd:
	"""End-to-end tests from C source to assembly output."""

	def _compile(self, source: str) -> str:
		from compiler.ir_gen import IRGenerator
		from compiler.parser import Parser
		ast = Parser.from_source(source).parse()
		ir = IRGenerator().generate(ast)
		return CodeGenerator().generate(ir)

	def test_static_function_e2e(self) -> None:
		source = "static int helper(int x) { return x + 1; }"
		asm = self._compile(source)
		assert ".globl helper" not in asm
		assert ".type helper, @function" in asm
		assert ".size helper, .-helper" in asm
		assert "helper:" in asm

	def test_default_function_e2e(self) -> None:
		source = "int add(int a, int b) { return a + b; }"
		asm = self._compile(source)
		assert ".globl add" in asm
		assert ".type add, @function" in asm
		assert ".size add, .-add" in asm

	def test_extern_function_e2e(self) -> None:
		source = "extern int printf(int x);"
		asm = self._compile(source)
		assert "printf:" not in asm
		assert ".globl printf" not in asm

	def test_prototype_e2e(self) -> None:
		source = "int forward(int x);"
		asm = self._compile(source)
		assert "forward:" not in asm

	def test_static_global_e2e(self) -> None:
		source = "static int counter = 5;"
		asm = self._compile(source)
		assert ".globl counter" not in asm
		assert "counter:" in asm

	def test_extern_global_e2e(self) -> None:
		source = "extern int errno;"
		asm = self._compile(source)
		assert "errno:" not in asm

	def test_combined_linkage_e2e(self) -> None:
		source = """
		extern int external_fn(int x);
		static int file_count = 0;
		int public_func(int n) { return n * 2; }
		static int private_helper(int n) { return n + 1; }
		"""
		asm = self._compile(source)
		# extern function: no body
		assert "external_fn:" not in asm
		# static global: no .globl
		assert ".globl file_count" not in asm
		assert "file_count:" in asm
		# public function: .globl present
		assert ".globl public_func" in asm
		assert ".type public_func, @function" in asm
		# static function: no .globl, but body present
		assert ".globl private_helper" not in asm
		assert "private_helper:" in asm
		assert ".type private_helper, @function" in asm
