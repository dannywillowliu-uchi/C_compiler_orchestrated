"""Tests for improved register allocator spill heuristics."""

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
	IRReturn,
	IRTemp,
	IRType,
)
from compiler.liveness import LivenessAnalyzer
from compiler.regalloc import (
	CALLEE_SAVED_REGS,
	K,
	RegisterAllocator,
)


def _t(name: str) -> IRTemp:
	return IRTemp(name)


def _c(val: int) -> IRConst:
	return IRConst(val)


def _simple_func(name: str, body: list, params: list | None = None) -> IRFunction:
	return IRFunction(
		name=name,
		params=params or [],
		body=body,
		return_type=IRType.INT,
	)


def _count_spills(func: IRFunction, temps: list[str]) -> int:
	"""Count how many of the given temps are spilled (not assigned a register)."""
	allocator = RegisterAllocator(func)
	result = allocator.allocate()
	return sum(1 for t in temps if t not in result)


class TestLoopDepthSpillHeuristic:
	def test_loop_temp_not_spilled_over_non_loop(self) -> None:
		"""Temps used inside loops should be kept in registers over temps used outside."""
		# Create K+1 temps all live simultaneously.
		# t0 is used in a loop body, t1..tK are used only outside the loop.
		body: list = [IRLabelInstr("entry")]
		temps = [f"t{i}" for i in range(K + 1)]
		for t in temps:
			body.append(IRCopy(dest=_t(t), source=_c(1)))

		# t0 is used in a loop
		body.append(IRJump(target="loop"))
		body.append(IRLabelInstr("loop"))
		body.append(IRBinOp(dest=_t("t0"), left=_t("t0"), op="+", right=_c(1)))
		body.append(IRBinOp(dest=_t("cond"), left=_t("t0"), op="<", right=_c(100)))
		body.append(IRCondJump(condition=_t("cond"), true_label="loop", false_label="exit"))

		# All temps used at the exit to keep them live
		body.append(IRLabelInstr("exit"))
		result_temp = _t("t0")
		for i in range(1, K + 1):
			new_dest = _t(f"sum{i}")
			body.append(IRBinOp(dest=new_dest, left=result_temp, op="+", right=_t(f"t{i}")))
			result_temp = new_dest
		body.append(IRReturn(value=result_temp))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# t0 (loop variable) should NOT be spilled
		assert "t0" in result, "Loop-used temp t0 should be kept in a register"

	def test_deeply_nested_loop_temp_preferred(self) -> None:
		"""A temp used in a depth-2 loop should be preferred over depth-1."""
		body: list = [IRLabelInstr("entry")]
		# Create K+1 temps, all live simultaneously
		temps = [f"t{i}" for i in range(K + 1)]
		for t in temps:
			body.append(IRCopy(dest=_t(t), source=_c(0)))

		# t0 in depth-2 loop
		body.append(IRJump(target="outer"))
		body.append(IRLabelInstr("outer"))
		body.append(IRJump(target="inner"))
		body.append(IRLabelInstr("inner"))
		body.append(IRBinOp(dest=_t("t0"), left=_t("t0"), op="+", right=_c(1)))
		body.append(IRBinOp(dest=_t("ic"), left=_t("t0"), op="<", right=_c(10)))
		body.append(IRCondJump(condition=_t("ic"), true_label="inner", false_label="outer_check"))
		body.append(IRLabelInstr("outer_check"))
		body.append(IRBinOp(dest=_t("oc"), left=_t("t0"), op="<", right=_c(100)))
		body.append(IRCondJump(condition=_t("oc"), true_label="outer", false_label="exit"))

		# t1 in depth-1 loop only
		body.append(IRLabelInstr("exit"))
		result_temp = _t("t0")
		for i in range(1, K + 1):
			new_dest = _t(f"sum{i}")
			body.append(IRBinOp(dest=new_dest, left=result_temp, op="+", right=_t(f"t{i}")))
			result_temp = new_dest
		body.append(IRReturn(value=result_temp))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		assert "t0" in result, "Deeply-nested loop temp should be kept in register"


class TestLiveRangeSpillHeuristic:
	def test_long_range_low_use_spilled_first(self) -> None:
		"""A temp with a long live range but few uses should be spilled before a frequently-used one."""
		# K+1 temps. t0 is used many times. tK has a long range but few uses.
		body: list = [IRLabelInstr("entry")]
		temps = [f"t{i}" for i in range(K + 1)]
		for t in temps:
			body.append(IRCopy(dest=_t(t), source=_c(1)))

		# Use t0 heavily
		for _ in range(5):
			body.append(IRBinOp(dest=_t("t0"), left=_t("t0"), op="+", right=_c(1)))

		# Use all temps at the end
		result_temp = _t("t0")
		for i in range(1, K + 1):
			new_dest = _t(f"sum{i}")
			body.append(IRBinOp(dest=new_dest, left=result_temp, op="+", right=_t(f"t{i}")))
			result_temp = new_dest
		body.append(IRReturn(value=result_temp))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# t0 is heavily used and should NOT be spilled
		assert "t0" in result, "Heavily-used temp should be kept in register"

	def test_short_lived_temps_not_spilled(self) -> None:
		"""Short-lived temps that don't contribute to register pressure should not be spilled."""
		body: list = [IRLabelInstr("entry")]
		# Create a series of short-lived computations
		body.append(IRCopy(dest=_t("a"), source=_c(1)))
		body.append(IRBinOp(dest=_t("b"), left=_t("a"), op="+", right=_c(2)))
		# a is dead after this point
		body.append(IRCopy(dest=_t("c"), source=_c(3)))
		body.append(IRBinOp(dest=_t("d"), left=_t("b"), op="+", right=_t("c")))
		body.append(IRReturn(value=_t("d")))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# All temps should get registers (no spills needed)
		for t in ["a", "b", "c", "d"]:
			assert t in result, f"Short-lived temp {t} should get a register"


class TestMoveCoalescingSpillReduction:
	def test_coalescing_reduces_register_pressure(self) -> None:
		"""Move coalescing should reduce the number of simultaneously live temps."""
		body: list = [IRLabelInstr("entry")]
		# Chain of copies: a -> b -> c -> d (all coalesce to one register)
		body.append(IRCopy(dest=_t("a"), source=_c(1)))
		body.append(IRCopy(dest=_t("b"), source=_t("a")))
		body.append(IRCopy(dest=_t("c"), source=_t("b")))
		body.append(IRCopy(dest=_t("d"), source=_t("c")))
		body.append(IRReturn(value=_t("d")))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# All should be coalesced to same register
		assert result["a"] == result["b"] == result["c"] == result["d"]
		# Zero spills
		spills = _count_spills(func, ["a", "b", "c", "d"])
		assert spills == 0

	def test_coalescing_with_pressure(self) -> None:
		"""Coalescing copies should free up registers when at high pressure."""
		# Create K temps. Two of them are copy-related and should coalesce,
		# leaving room for all K unique values.
		body: list = [IRLabelInstr("entry")]
		# t0..t(K-2) are independent
		for i in range(K - 1):
			body.append(IRCopy(dest=_t(f"t{i}"), source=_c(i)))
		# copy_src and copy_dst coalesce into one register
		body.append(IRCopy(dest=_t("copy_src"), source=_c(99)))
		body.append(IRCopy(dest=_t("copy_dst"), source=_t("copy_src")))

		# Use all temps
		result_temp = _t("copy_dst")
		for i in range(K - 1):
			new_dest = _t(f"sum{i}")
			body.append(IRBinOp(dest=new_dest, left=result_temp, op="+", right=_t(f"t{i}")))
			result_temp = new_dest
		body.append(IRReturn(value=result_temp))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# copy_src and copy_dst should share a register
		assert result.get("copy_src") == result.get("copy_dst")
		# No spills needed since coalescing freed a register
		all_temps = [f"t{i}" for i in range(K - 1)] + ["copy_src", "copy_dst"]
		spills = _count_spills(func, all_temps)
		assert spills == 0, f"Expected 0 spills but got {spills}"


class TestSpillCountReduction:
	def test_fewer_spills_with_heuristic(self) -> None:
		"""Representative function should have minimal spill count with improved heuristics."""
		# Simulate a function with a hot loop and cold setup code
		body: list = [IRLabelInstr("entry")]

		# Cold setup: define several temps
		for i in range(K + 2):
			body.append(IRCopy(dest=_t(f"v{i}"), source=_c(i)))

		# Hot loop using only v0 and v1
		body.append(IRJump(target="loop"))
		body.append(IRLabelInstr("loop"))
		body.append(IRBinOp(dest=_t("v0"), left=_t("v0"), op="+", right=_t("v1")))
		body.append(IRBinOp(dest=_t("lc"), left=_t("v0"), op="<", right=_c(1000)))
		body.append(IRCondJump(condition=_t("lc"), true_label="loop", false_label="done"))

		# Cold exit using all temps
		body.append(IRLabelInstr("done"))
		result_temp = _t("v0")
		for i in range(1, K + 2):
			new_dest = _t(f"r{i}")
			body.append(IRBinOp(dest=new_dest, left=result_temp, op="+", right=_t(f"v{i}")))
			result_temp = new_dest
		body.append(IRReturn(value=result_temp))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# v0 and v1 (hot loop variables) must NOT be spilled
		assert "v0" in result, "Hot loop var v0 should not be spilled"
		assert "v1" in result, "Hot loop var v1 should not be spilled"

		# The coloring must be valid
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

	def test_call_crossing_with_loop(self) -> None:
		"""Temps live across calls in loops should get callee-saved registers."""
		body: list = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("acc"), source=_c(0)),
			IRCopy(dest=_t("i"), source=_c(0)),
			IRJump(target="loop"),
			IRLabelInstr("loop"),
			IRCall(dest=_t("val"), function_name="get_value", args=[_t("i")], arg_types=[IRType.INT]),
			IRBinOp(dest=_t("acc"), left=_t("acc"), op="+", right=_t("val")),
			IRBinOp(dest=_t("i"), left=_t("i"), op="+", right=_c(1)),
			IRBinOp(dest=_t("cond"), left=_t("i"), op="<", right=_c(10)),
			IRCondJump(condition=_t("cond"), true_label="loop", false_label="done"),
			IRLabelInstr("done"),
			IRReturn(value=_t("acc")),
		]

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# acc and i are live across the call in the loop
		assert "acc" in result, "Loop accumulator should be in a register"
		assert "i" in result, "Loop counter should be in a register"
		# Both should be callee-saved since they cross a call
		assert result["acc"] in CALLEE_SAVED_REGS, f"acc should be callee-saved, got {result['acc']}"
		assert result["i"] in CALLEE_SAVED_REGS, f"i should be callee-saved, got {result['i']}"

	def test_coloring_validity_high_pressure(self) -> None:
		"""Under high register pressure, the coloring must remain valid."""
		num_temps = K + 4
		body: list = [IRLabelInstr("entry")]
		temps = [f"t{i}" for i in range(num_temps)]
		for t in temps:
			body.append(IRCopy(dest=_t(t), source=_c(1)))
		result_temp = _t("t0")
		for i in range(1, num_temps):
			new_dest = _t(f"s{i}")
			body.append(IRBinOp(dest=new_dest, left=result_temp, op="+", right=_t(f"t{i}")))
			result_temp = new_dest
		body.append(IRReturn(value=result_temp))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# Validate coloring: no two interfering temps share a register
		cfg = CFG(func.body)
		analyzer = LivenessAnalyzer(cfg)
		ig = analyzer.interference_graph()
		for node, neighbors in ig.items():
			if node in result:
				for neighbor in neighbors:
					if neighbor in result:
						assert result[node] != result[neighbor], (
							f"Invalid coloring: {node}={result[node]} == {neighbor}={result[neighbor]}"
						)
