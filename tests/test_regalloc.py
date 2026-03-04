"""Tests for graph-coloring register allocator."""

from compiler.cfg import CFG
from compiler.ir import (
	IRBinOp,
	IRCall,
	IRCondJump,
	IRConst,
	IRCopy,
	IRFunction,
	IRJump,
	IRLabelInstr,
	IRProgram,
	IRReturn,
	IRTemp,
	IRType,
)
from compiler.liveness import LivenessAnalyzer
from compiler.regalloc import (
	ALLOCATABLE_REGS,
	CALLEE_SAVED_REGS,
	K,
	RegisterAllocator,
	allocate_registers,
)

from compiler.codegen import CodeGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t(name: str) -> IRTemp:
	return IRTemp(name)


def _c(val: int) -> IRConst:
	return IRConst(val)


def _simple_func(name: str, body: list, params: list | None = None) -> IRFunction:
	"""Create a simple IRFunction with INT return type."""
	return IRFunction(
		name=name,
		params=params or [],
		body=body,
		return_type=IRType.INT,
	)


# ---------------------------------------------------------------------------
# Tests: interference graph construction
# ---------------------------------------------------------------------------

class TestInterferenceGraphForRegalloc:
	def test_two_non_overlapping_temps(self) -> None:
		"""Sequential non-overlapping temps should not interfere."""
		body = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRCopy(dest=_t("b"), source=_t("a")),
			IRReturn(value=_t("b")),
		]
		cfg = CFG(body)
		analyzer = LivenessAnalyzer(cfg)
		ig = analyzer.interference_graph()
		# a and b don't interfere (copy coalescing)
		assert "b" not in ig.get("a", set())
		assert "a" not in ig.get("b", set())

	def test_two_overlapping_temps(self) -> None:
		"""Temps that are simultaneously live should interfere."""
		body = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRCopy(dest=_t("b"), source=_c(2)),
			IRBinOp(dest=_t("c"), left=_t("a"), op="+", right=_t("b")),
			IRReturn(value=_t("c")),
		]
		cfg = CFG(body)
		analyzer = LivenessAnalyzer(cfg)
		ig = analyzer.interference_graph()
		assert "b" in ig["a"]
		assert "a" in ig["b"]

	def test_loop_interference(self) -> None:
		"""Temps in a loop with overlapping lifetimes should interfere."""
		body = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("sum"), source=_c(0)),
			IRCopy(dest=_t("i"), source=_c(0)),
			IRJump(target="loop"),
			IRLabelInstr("loop"),
			IRBinOp(dest=_t("sum"), left=_t("sum"), op="+", right=_t("i")),
			IRBinOp(dest=_t("i"), left=_t("i"), op="+", right=_c(1)),
			IRBinOp(dest=_t("cond"), left=_t("i"), op="<", right=_c(10)),
			IRCondJump(condition=_t("cond"), true_label="loop", false_label="exit"),
			IRLabelInstr("exit"),
			IRReturn(value=_t("sum")),
		]
		cfg = CFG(body)
		analyzer = LivenessAnalyzer(cfg)
		ig = analyzer.interference_graph()
		# sum and i are simultaneously live in the loop
		assert "i" in ig["sum"]
		assert "sum" in ig["i"]


# ---------------------------------------------------------------------------
# Tests: graph coloring produces valid colorings
# ---------------------------------------------------------------------------

class TestGraphColoring:
	def test_single_temp_gets_register(self) -> None:
		"""A single temp should be assigned a register."""
		func = _simple_func("f", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(42)),
			IRReturn(value=_t("x")),
		])
		allocator = RegisterAllocator(func)
		result = allocator.allocate()
		assert "x" in result
		assert result["x"] in ALLOCATABLE_REGS

	def test_non_interfering_temps_can_share_register(self) -> None:
		"""Non-interfering temps may receive the same register."""
		func = _simple_func("f", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRReturn(value=_t("a")),
		])
		allocator = RegisterAllocator(func)
		result = allocator.allocate()
		assert "a" in result
		assert result["a"] in ALLOCATABLE_REGS

	def test_interfering_temps_get_different_registers(self) -> None:
		"""Interfering temps must get different registers."""
		func = _simple_func("f", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRCopy(dest=_t("b"), source=_c(2)),
			IRBinOp(dest=_t("c"), left=_t("a"), op="+", right=_t("b")),
			IRReturn(value=_t("c")),
		])
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# a and b interfere, so they must have different registers
		assert "a" in result
		assert "b" in result
		assert result["a"] != result["b"]
		assert result["a"] in ALLOCATABLE_REGS
		assert result["b"] in ALLOCATABLE_REGS

	def test_coloring_valid_on_diamond_cfg(self) -> None:
		"""All simultaneously-live temps must have distinct colors across branches."""
		func = _simple_func("f", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRCopy(dest=_t("y"), source=_c(2)),
			IRCopy(dest=_t("cond"), source=_c(0)),
			IRCondJump(condition=_t("cond"), true_label="then", false_label="else_"),
			IRLabelInstr("then"),
			IRBinOp(dest=_t("r"), left=_t("x"), op="+", right=_t("y")),
			IRJump(target="merge"),
			IRLabelInstr("else_"),
			IRBinOp(dest=_t("r"), left=_t("x"), op="-", right=_t("y")),
			IRJump(target="merge"),
			IRLabelInstr("merge"),
			IRReturn(value=_t("r")),
		])
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# Validate: check the coloring against the interference graph
		cfg = CFG(func.body)
		analyzer = LivenessAnalyzer(cfg)
		ig = analyzer.interference_graph()
		for node, neighbors in ig.items():
			if node in result:
				for neighbor in neighbors:
					if neighbor in result:
						assert result[node] != result[neighbor], (
							f"{node}={result[node]} conflicts with {neighbor}={result[neighbor]}"
						)

	def test_k_temps_all_interfering(self) -> None:
		"""K temps all live simultaneously should all get different registers."""
		# Create K temps that are all live at the same point
		body: list = [IRLabelInstr("entry")]
		for i in range(K):
			body.append(IRCopy(dest=_t(f"t{i}"), source=_c(i)))
		# Use all K temps in a chain to keep them live
		result_temp = _t("t0")
		for i in range(1, K):
			new_dest = _t(f"sum{i}")
			body.append(IRBinOp(dest=new_dest, left=result_temp, op="+", right=_t(f"t{i}")))
			result_temp = new_dest
		body.append(IRReturn(value=result_temp))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# All K temps should get registers, all different
		assigned_regs = set()
		for i in range(K):
			name = f"t{i}"
			assert name in result, f"{name} should be assigned a register"
			assigned_regs.add(result[name])
		assert len(assigned_regs) == K

	def test_call_crossing_temps_get_callee_saved(self) -> None:
		"""Temps live across a function call should only get callee-saved registers."""
		func = _simple_func("f", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(42)),
			IRCall(dest=_t("y"), function_name="bar", args=[]),
			IRBinOp(dest=_t("z"), left=_t("x"), op="+", right=_t("y")),
			IRReturn(value=_t("z")),
		])
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# x is live across the call, so it must be in a callee-saved register
		assert "x" in result
		assert result["x"] in CALLEE_SAVED_REGS, (
			f"x should be callee-saved but got {result['x']}"
		)


# ---------------------------------------------------------------------------
# Tests: spill handling when pressure exceeds available registers
# ---------------------------------------------------------------------------

class TestSpillHandling:
	def test_spill_when_too_many_interfering(self) -> None:
		"""More than K simultaneously-live temps forces spilling."""
		num_temps = K + 2
		body: list = [IRLabelInstr("entry")]
		# Define all temps
		for i in range(num_temps):
			body.append(IRCopy(dest=_t(f"t{i}"), source=_c(i)))
		# Use all temps together to make them all simultaneously live
		result_temp = _t("t0")
		for i in range(1, num_temps):
			new_dest = _t(f"sum{i}")
			body.append(IRBinOp(dest=new_dest, left=result_temp, op="+", right=_t(f"t{i}")))
			result_temp = new_dest
		body.append(IRReturn(value=result_temp))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# Some temps should be spilled (not in the mapping)
		assigned = {name for name in [f"t{i}" for i in range(num_temps)] if name in result}
		spilled = {f"t{i}" for i in range(num_temps)} - assigned
		assert len(spilled) >= 1, "At least one temp should be spilled"

		# All assigned temps should have valid, distinct registers for interfering pairs
		cfg = CFG(func.body)
		analyzer = LivenessAnalyzer(cfg)
		ig = analyzer.interference_graph()
		for node in assigned:
			for neighbor in ig.get(node, set()):
				if neighbor in result and node in result:
					assert result[node] != result[neighbor]

	def test_spilled_temps_work_with_codegen(self) -> None:
		"""Spilled temps should still generate valid assembly via stack slots."""
		num_temps = K + 2
		body: list = [IRLabelInstr("entry")]
		for i in range(num_temps):
			body.append(IRCopy(dest=_t(f"t{i}"), source=_c(i)))
		result_temp = _t("t0")
		for i in range(1, num_temps):
			new_dest = _t(f"sum{i}")
			body.append(IRBinOp(dest=new_dest, left=result_temp, op="+", right=_t(f"t{i}")))
			result_temp = new_dest
		body.append(IRReturn(value=result_temp))

		func = _simple_func("f", body)
		program = IRProgram(functions=[func])

		allocator = RegisterAllocator(func)
		reg_map = allocator.allocate()
		regalloc_maps = {"f": reg_map} if reg_map else {}

		codegen = CodeGenerator(regalloc_maps=regalloc_maps)
		assembly = codegen.generate(program)

		# Should produce valid assembly (no crash)
		assert ".globl f" in assembly
		assert "ret" in assembly

	def test_empty_function_no_crash(self) -> None:
		"""Empty function body should produce empty allocation without error."""
		func = _simple_func("f", [
			IRLabelInstr("entry"),
			IRReturn(),
		])
		allocator = RegisterAllocator(func)
		result = allocator.allocate()
		assert result == {}


# ---------------------------------------------------------------------------
# Tests: end-to-end with codegen
# ---------------------------------------------------------------------------

class TestEndToEnd:
	def test_simple_add_uses_registers(self) -> None:
		"""Simple add function should use registers instead of all stack ops."""
		func = IRFunction(
			name="add",
			params=[_t("a"), _t("b")],
			body=[
				IRLabelInstr("entry"),
				IRBinOp(dest=_t("result"), left=_t("a"), op="+", right=_t("b")),
				IRReturn(value=_t("result")),
			],
			return_type=IRType.INT,
			param_types=[IRType.INT, IRType.INT],
		)
		program = IRProgram(functions=[func])
		reg_maps = allocate_registers(program)
		codegen = CodeGenerator(regalloc_maps=reg_maps)
		assembly = codegen.generate(program)

		# Should contain register names from the allocatable set
		has_alloc_reg = any(reg in assembly for reg in ALLOCATABLE_REGS)
		assert has_alloc_reg, "Assembly should use allocatable registers"
		assert ".globl add" in assembly
		assert "ret" in assembly

	def test_regalloc_disabled_fallback(self) -> None:
		"""Without regalloc maps, codegen uses the existing stack-only path."""
		func = IRFunction(
			name="add",
			params=[_t("a"), _t("b")],
			body=[
				IRLabelInstr("entry"),
				IRBinOp(dest=_t("result"), left=_t("a"), op="+", right=_t("b")),
				IRReturn(value=_t("result")),
			],
			return_type=IRType.INT,
			param_types=[IRType.INT, IRType.INT],
		)
		program = IRProgram(functions=[func])

		# Without regalloc
		codegen_no_regalloc = CodeGenerator()
		asm_no_regalloc = codegen_no_regalloc.generate(program)

		# Should use stack (%rbp-relative) addressing, not allocatable regs
		assert "(%rbp)" in asm_no_regalloc
		assert ".globl add" in asm_no_regalloc

	def test_codegen_with_regalloc_correctness(self) -> None:
		"""Verify that regalloc-generated assembly is structurally valid."""
		func = _simple_func("compute", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(10)),
			IRCopy(dest=_t("b"), source=_c(20)),
			IRBinOp(dest=_t("c"), left=_t("a"), op="+", right=_t("b")),
			IRBinOp(dest=_t("d"), left=_t("c"), op="*", right=_c(2)),
			IRReturn(value=_t("d")),
		])
		program = IRProgram(functions=[func])
		reg_maps = allocate_registers(program)
		codegen = CodeGenerator(regalloc_maps=reg_maps)
		assembly = codegen.generate(program)

		# Verify structural elements
		assert ".globl compute" in assembly
		assert "pushq %rbp" in assembly
		assert "movq %rsp, %rbp" in assembly
		assert "ret" in assembly

		# With regalloc, some operations should reference allocatable regs
		lines = assembly.split("\n")
		uses_allocatable = False
		for line in lines:
			for reg in ALLOCATABLE_REGS:
				if reg in line:
					uses_allocatable = True
					break
		assert uses_allocatable

	def test_callee_saved_regs_are_preserved(self) -> None:
		"""Callee-saved registers used by regalloc should be saved and restored."""
		func = _simple_func("f", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRReturn(value=_t("x")),
		])
		program = IRProgram(functions=[func])
		reg_maps = allocate_registers(program)
		codegen = CodeGenerator(regalloc_maps=reg_maps)
		assembly = codegen.generate(program)

		# Find which callee-saved register was assigned
		func_map = reg_maps.get("f", {})
		callee_used = [r for r in func_map.values() if r in CALLEE_SAVED_REGS]

		# Each callee-saved register should be saved and restored
		for reg in callee_used:
			# Save: movq %reg, offset(%rbp)
			save_pattern = f"movq {reg},"
			restore_pattern = f", {reg}"
			assert save_pattern in assembly, f"{reg} should be saved in prologue"
			assert restore_pattern in assembly, f"{reg} should be restored in epilogue"

	def test_loop_with_regalloc(self) -> None:
		"""Loop code with regalloc should produce valid assembly."""
		func = _simple_func("sum_to_n", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("sum"), source=_c(0)),
			IRCopy(dest=_t("i"), source=_c(1)),
			IRJump(target="loop"),
			IRLabelInstr("loop"),
			IRBinOp(dest=_t("sum"), left=_t("sum"), op="+", right=_t("i")),
			IRBinOp(dest=_t("i"), left=_t("i"), op="+", right=_c(1)),
			IRBinOp(dest=_t("cond"), left=_t("i"), op="<=", right=_c(10)),
			IRCondJump(condition=_t("cond"), true_label="loop", false_label="done"),
			IRLabelInstr("done"),
			IRReturn(value=_t("sum")),
		])
		program = IRProgram(functions=[func])
		reg_maps = allocate_registers(program)
		codegen = CodeGenerator(regalloc_maps=reg_maps)
		assembly = codegen.generate(program)

		assert ".globl sum_to_n" in assembly
		assert "ret" in assembly
		assert "jmp loop" in assembly

	def test_allocate_registers_multi_function(self) -> None:
		"""allocate_registers should handle multiple functions."""
		func1 = _simple_func("foo", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRReturn(value=_t("x")),
		])
		func2 = _simple_func("bar", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("y"), source=_c(2)),
			IRReturn(value=_t("y")),
		])
		program = IRProgram(functions=[func1, func2])
		reg_maps = allocate_registers(program)

		# Both functions should get register allocations
		assert "foo" in reg_maps
		assert "bar" in reg_maps
		assert "x" in reg_maps["foo"]
		assert "y" in reg_maps["bar"]

	def test_full_pipeline_c_source(self) -> None:
		"""End-to-end: compile C source with regalloc, verify assembly uses registers."""
		from compiler.ir_gen import IRGenerator
		from compiler.lexer import Lexer
		from compiler.parser import Parser
		from compiler.semantic import SemanticAnalyzer

		source = """
		int square(int x) {
			int result = x * x;
			return result;
		}
		"""
		tokens = Lexer(source).tokenize()
		ast = Parser(tokens).parse()
		errors = SemanticAnalyzer().analyze(ast)
		assert not errors

		ir_program = IRGenerator().generate(ast)
		reg_maps = allocate_registers(ir_program)
		codegen = CodeGenerator(regalloc_maps=reg_maps)
		assembly = codegen.generate(ir_program)

		assert ".globl square" in assembly
		assert "ret" in assembly
		# Should use at least one allocatable register
		has_alloc_reg = any(reg in assembly for reg in ALLOCATABLE_REGS)
		assert has_alloc_reg, f"Expected allocatable registers in assembly:\n{assembly}"


# ---------------------------------------------------------------------------
# Tests: move coalescing
# ---------------------------------------------------------------------------

class TestMoveCoalescing:
	def test_copy_chain_coalesced_same_register(self) -> None:
		"""A chain of copies (a -> b -> c) with no interference should coalesce."""
		func = _simple_func("f", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(42)),
			IRCopy(dest=_t("b"), source=_t("a")),
			IRCopy(dest=_t("c"), source=_t("b")),
			IRReturn(value=_t("c")),
		])
		allocator = RegisterAllocator(func)
		result = allocator.allocate()
		# All three temps should be coalesced to the same register
		assert "a" in result
		assert "b" in result
		assert "c" in result
		assert result["a"] == result["b"] == result["c"]

	def test_interfering_copies_not_coalesced(self) -> None:
		"""Copies between interfering temps must not be coalesced."""
		func = _simple_func("f", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRCopy(dest=_t("b"), source=_c(2)),
			# a and b are both live here
			IRCopy(dest=_t("c"), source=_t("a")),
			IRBinOp(dest=_t("d"), left=_t("c"), op="+", right=_t("b")),
			IRReturn(value=_t("d")),
		])
		allocator = RegisterAllocator(func)
		result = allocator.allocate()
		# a and b interfere, so they must have different registers
		assert result["a"] != result["b"]

	def test_coalesced_copy_eliminates_movq(self) -> None:
		"""When source and dest are coalesced, codegen should not emit a reg-to-reg movq for the copy."""
		func = _simple_func("f", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(10)),
			IRCopy(dest=_t("b"), source=_t("a")),
			IRReturn(value=_t("b")),
		])
		program = IRProgram(functions=[func])
		reg_maps = allocate_registers(program)

		# Verify coalescing happened: a and b must share the same register
		func_map = reg_maps.get("f", {})
		assert func_map["a"] == func_map["b"], (
			f"a and b should be coalesced to same register: a={func_map.get('a')}, b={func_map.get('b')}"
		)

		codegen = CodeGenerator(regalloc_maps=reg_maps)
		assembly = codegen.generate(program)

		# Count reg-to-reg movq between allocatable registers (not const loads or stack ops)
		alloc_set = set(ALLOCATABLE_REGS)
		reg_to_reg_moves = 0
		for line in assembly.splitlines():
			stripped = line.strip()
			if not stripped.startswith("movq"):
				continue
			parts = stripped.split()
			if len(parts) >= 2:
				src = parts[1].rstrip(",")
				dst = parts[2] if len(parts) > 2 else ""
				if src in alloc_set and dst in alloc_set:
					reg_to_reg_moves += 1

		# With coalescing, the copy b=a should produce no reg-to-reg move
		assert reg_to_reg_moves == 0, (
			f"Expected 0 reg-to-reg moves between allocatable regs, got {reg_to_reg_moves}.\n{assembly}"
		)

	def test_multiple_copies_reduced_moves(self) -> None:
		"""Multiple sequential copies should produce fewer moves with coalescing."""
		func = _simple_func("f", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(5)),
			IRCopy(dest=_t("y"), source=_t("x")),
			IRCopy(dest=_t("z"), source=_t("y")),
			IRCopy(dest=_t("w"), source=_t("z")),
			IRReturn(value=_t("w")),
		])
		allocator = RegisterAllocator(func)
		result = allocator.allocate()
		# All should coalesce to same register
		assert result["x"] == result["y"] == result["z"] == result["w"]

	def test_coalescing_preserves_correctness(self) -> None:
		"""Coalescing must not violate the interference graph constraint."""
		func = _simple_func("f", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRCopy(dest=_t("b"), source=_c(2)),
			IRBinOp(dest=_t("c"), left=_t("a"), op="+", right=_t("b")),
			IRCopy(dest=_t("d"), source=_t("c")),
			IRBinOp(dest=_t("e"), left=_t("d"), op="*", right=_c(3)),
			IRReturn(value=_t("e")),
		])
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# Validate coloring against interference graph
		cfg = CFG(func.body)
		analyzer = LivenessAnalyzer(cfg)
		ig = analyzer.interference_graph()
		for node, neighbors in ig.items():
			if node in result:
				for neighbor in neighbors:
					if neighbor in result:
						assert result[node] != result[neighbor], (
							f"{node}={result[node]} conflicts with {neighbor}={result[neighbor]}"
						)

	def test_spill_heuristic_prefers_low_use(self) -> None:
		"""The improved spill heuristic should prefer spilling less-used temps."""
		# Create K+1 temps all live simultaneously, where one is used much more
		num_temps = K + 1
		body: list = [IRLabelInstr("entry")]
		for i in range(num_temps):
			body.append(IRCopy(dest=_t(f"t{i}"), source=_c(i)))
		# Use t0 multiple times to increase its use count
		body.append(IRBinOp(dest=_t("extra1"), left=_t("t0"), op="+", right=_t("t0")))
		body.append(IRBinOp(dest=_t("extra2"), left=_t("extra1"), op="+", right=_t("t0")))
		# Use all temps to keep them live
		result_temp = _t("extra2")
		for i in range(1, num_temps):
			new_dest = _t(f"sum{i}")
			body.append(IRBinOp(dest=new_dest, left=result_temp, op="+", right=_t(f"t{i}")))
			result_temp = new_dest
		body.append(IRReturn(value=result_temp))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()
		# t0 is heavily used, so it should NOT be spilled
		assert "t0" in result, "Heavily-used temp t0 should not be spilled"
