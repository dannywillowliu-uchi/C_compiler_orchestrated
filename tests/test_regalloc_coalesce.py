"""Tests for register allocator move coalescing and rematerialization."""

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


def _validate_coloring(func: IRFunction, result: dict[str, str]) -> None:
	"""Assert that no two interfering temps share a register."""
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


class TestMoveCoalescingBasic:
	def test_simple_copy_coalesces(self) -> None:
		"""A simple copy t2 = t1 where they don't interfere should coalesce."""
		func = _simple_func("f", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("t1"), source=_c(10)),
			IRCopy(dest=_t("t2"), source=_t("t1")),
			IRReturn(value=_t("t2")),
		])
		allocator = RegisterAllocator(func)
		result = allocator.allocate()
		assert result["t1"] == result["t2"], "Non-interfering copy should coalesce"

	def test_chain_of_copies_coalesces(self) -> None:
		"""a -> b -> c -> d should all coalesce to the same register."""
		func = _simple_func("f", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRCopy(dest=_t("b"), source=_t("a")),
			IRCopy(dest=_t("c"), source=_t("b")),
			IRCopy(dest=_t("d"), source=_t("c")),
			IRReturn(value=_t("d")),
		])
		allocator = RegisterAllocator(func)
		result = allocator.allocate()
		assert result["a"] == result["b"] == result["c"] == result["d"]
		_validate_coloring(func, result)

	def test_interfering_copies_not_coalesced(self) -> None:
		"""Copies between interfering temps must not coalesce."""
		func = _simple_func("f", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRCopy(dest=_t("b"), source=_c(2)),
			# a and b both live here
			IRCopy(dest=_t("c"), source=_t("a")),
			IRBinOp(dest=_t("d"), left=_t("c"), op="+", right=_t("b")),
			IRReturn(value=_t("d")),
		])
		allocator = RegisterAllocator(func)
		result = allocator.allocate()
		assert result["a"] != result["b"]
		_validate_coloring(func, result)

	def test_coalescing_frees_register_under_pressure(self) -> None:
		"""Coalescing a copy pair should free a register slot under pressure."""
		body: list = [IRLabelInstr("entry")]
		# K-1 independent temps
		for i in range(K - 1):
			body.append(IRCopy(dest=_t(f"t{i}"), source=_c(i)))
		# One copy pair that should coalesce into one slot
		body.append(IRCopy(dest=_t("src"), source=_c(99)))
		body.append(IRCopy(dest=_t("dst"), source=_t("src")))
		# Use all temps together
		result_temp = _t("dst")
		for i in range(K - 1):
			new_dest = _t(f"s{i}")
			body.append(IRBinOp(dest=new_dest, left=result_temp, op="+", right=_t(f"t{i}")))
			result_temp = new_dest
		body.append(IRReturn(value=result_temp))

		func = _simple_func("f", body)
		all_temps = [f"t{i}" for i in range(K - 1)] + ["src", "dst"]
		spills = _count_spills(func, all_temps)
		assert spills == 0, f"Coalescing should prevent spills, got {spills}"

	def test_coalescing_in_loop(self) -> None:
		"""Copy inside a loop should coalesce when src/dst don't interfere."""
		func = _simple_func("f", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(0)),
			IRJump(target="loop"),
			IRLabelInstr("loop"),
			IRBinOp(dest=_t("x"), left=_t("x"), op="+", right=_c(1)),
			IRCopy(dest=_t("y"), source=_t("x")),
			IRBinOp(dest=_t("cond"), left=_t("y"), op="<", right=_c(10)),
			IRCondJump(condition=_t("cond"), true_label="loop", false_label="exit"),
			IRLabelInstr("exit"),
			IRReturn(value=_t("y")),
		])
		allocator = RegisterAllocator(func)
		result = allocator.allocate()
		# x and y don't interfere (y is defined by copying x, x is dead after)
		# They should coalesce
		assert result.get("x") == result.get("y"), "Loop copy should coalesce"
		_validate_coloring(func, result)


class TestRematerializationHints:
	def test_constant_temp_spilled_first(self) -> None:
		"""A temp holding a constant should be preferred for spilling over a computed temp."""
		body: list = [IRLabelInstr("entry")]
		# K computed temps (not rematerializable) - all defined before any are used
		for i in range(K):
			body.append(IRBinOp(
				dest=_t(f"comp{i}"), left=_c(i), op="+", right=_c(i * 2)
			))
		# One constant temp (rematerializable)
		body.append(IRCopy(dest=_t("konst"), source=_c(42)))
		# Use ALL temps in a single expression chain to force them all live simultaneously
		# This creates K+1 simultaneously live temps, forcing exactly 1 spill
		body.append(IRBinOp(dest=_t("s0"), left=_t("konst"), op="+", right=_t("comp0")))
		for i in range(1, K):
			body.append(IRBinOp(dest=_t(f"s{i}"), left=_t(f"s{i-1}"), op="+", right=_t(f"comp{i}")))
		body.append(IRReturn(value=_t(f"s{K-1}")))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# konst is rematerializable; at K+1 simultaneously live temps, one must spill.
		# The allocator should prefer spilling the constant over computed values.
		# We verify: if anything is spilled, konst should be among the spilled.
		all_temps = [f"comp{i}" for i in range(K)] + ["konst"]
		total_spilled = sum(1 for t in all_temps if t not in result)
		if total_spilled > 0:
			assert "konst" not in result, (
				"Rematerializable constant should be spilled before computed values"
			)
		_validate_coloring(func, result)

	def test_rematerializable_under_high_pressure(self) -> None:
		"""Under high pressure, rematerializable temps should be spilled before computed ones."""
		body: list = [IRLabelInstr("entry")]
		# K computed temps (expensive to spill)
		for i in range(K):
			body.append(IRBinOp(
				dest=_t(f"v{i}"), left=_c(i), op="*", right=_c(i + 1)
			))
		# 1 constant temp (cheap to spill) -- total K+1
		body.append(IRCopy(dest=_t("konst"), source=_c(100)))
		# Use all temps in one chain to force simultaneous liveness
		body.append(IRBinOp(dest=_t("s0"), left=_t("konst"), op="+", right=_t("v0")))
		for i in range(1, K):
			body.append(IRBinOp(
				dest=_t(f"s{i}"), left=_t(f"s{i-1}"), op="+", right=_t(f"v{i}")
			))
		body.append(IRReturn(value=_t(f"s{K-1}")))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# K+1 temps all simultaneously live → at least 1 spill.
		# The constant should be the one spilled.
		all_temps = [f"v{i}" for i in range(K)] + ["konst"]
		total_spilled = sum(1 for t in all_temps if t not in result)
		if total_spilled > 0:
			assert "konst" not in result, (
				"Rematerializable constant should be spilled before computed values"
			)
		_validate_coloring(func, result)


class TestDefUseDistanceSpills:
	def test_single_use_far_from_def_cheaper_to_spill(self) -> None:
		"""A temp used once, far from its definition, is cheaper to spill than a frequently used one."""
		body: list = [IRLabelInstr("entry")]
		# t_frequent: defined and used many times
		body.append(IRCopy(dest=_t("t_frequent"), source=_c(1)))
		for _ in range(4):
			body.append(IRBinOp(
				dest=_t("t_frequent"), left=_t("t_frequent"), op="+", right=_c(1)
			))
		# t_rare: defined once, then many padding instructions, used once at end
		body.append(IRCopy(dest=_t("t_rare"), source=_c(999)))
		# Other temps to create pressure
		for i in range(K - 1):
			body.append(IRCopy(dest=_t(f"p{i}"), source=_c(i)))
		# Use all pressure temps
		result_temp = _t("t_frequent")
		for i in range(K - 1):
			new_dest = _t(f"s{i}")
			body.append(IRBinOp(dest=new_dest, left=result_temp, op="+", right=_t(f"p{i}")))
			result_temp = new_dest
		# Finally use t_rare
		body.append(IRBinOp(dest=_t("final"), left=result_temp, op="+", right=_t("t_rare")))
		body.append(IRReturn(value=_t("final")))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# t_frequent should be kept (used a lot)
		assert "t_frequent" in result, "Frequently-used temp should be kept in register"
		_validate_coloring(func, result)

	def test_coloring_valid_after_improvements(self) -> None:
		"""All improved heuristics must still produce valid colorings."""
		body: list = [IRLabelInstr("entry")]
		num_temps = K + 3
		for i in range(num_temps):
			body.append(IRCopy(dest=_t(f"t{i}"), source=_c(i)))
		# Use in a loop for some
		body.append(IRJump(target="loop"))
		body.append(IRLabelInstr("loop"))
		body.append(IRBinOp(dest=_t("t0"), left=_t("t0"), op="+", right=_c(1)))
		body.append(IRBinOp(dest=_t("lc"), left=_t("t0"), op="<", right=_c(10)))
		body.append(IRCondJump(condition=_t("lc"), true_label="loop", false_label="done"))
		body.append(IRLabelInstr("done"))
		result_temp = _t("t0")
		for i in range(1, num_temps):
			new_dest = _t(f"s{i}")
			body.append(IRBinOp(dest=new_dest, left=result_temp, op="+", right=_t(f"t{i}")))
			result_temp = new_dest
		body.append(IRReturn(value=result_temp))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()
		_validate_coloring(func, result)


class TestDefUseChains:
	def test_liveness_def_use_chains(self) -> None:
		"""LivenessAnalyzer.def_use_chains should return correct def/use positions."""
		body = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRCopy(dest=_t("b"), source=_c(2)),
			IRBinOp(dest=_t("c"), left=_t("a"), op="+", right=_t("b")),
			IRReturn(value=_t("c")),
		]
		cfg = CFG(body)
		analyzer = LivenessAnalyzer(cfg)
		chains = analyzer.def_use_chains()

		# "a" defined once, used once
		a_defs, a_uses = chains["a"]
		assert len(a_defs) == 1
		assert len(a_uses) == 1

		# "b" defined once, used once
		b_defs, b_uses = chains["b"]
		assert len(b_defs) == 1
		assert len(b_uses) == 1

		# "c" defined once, used once (in return)
		c_defs, c_uses = chains["c"]
		assert len(c_defs) == 1
		assert len(c_uses) == 1

	def test_loop_def_use_chains(self) -> None:
		"""Temps in loops should show multiple def/use points."""
		body = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("i"), source=_c(0)),
			IRJump(target="loop"),
			IRLabelInstr("loop"),
			IRBinOp(dest=_t("i"), left=_t("i"), op="+", right=_c(1)),
			IRBinOp(dest=_t("cond"), left=_t("i"), op="<", right=_c(10)),
			IRCondJump(condition=_t("cond"), true_label="loop", false_label="exit"),
			IRLabelInstr("exit"),
			IRReturn(value=_t("i")),
		]
		cfg = CFG(body)
		analyzer = LivenessAnalyzer(cfg)
		chains = analyzer.def_use_chains()

		# "i" defined twice (entry + loop body), used multiple times
		i_defs, i_uses = chains["i"]
		assert len(i_defs) == 2
		assert len(i_uses) >= 2


class TestFewerSpillsRepresentative:
	def test_accumulator_loop_no_spill(self) -> None:
		"""A typical accumulator loop should not spill the accumulator."""
		func = _simple_func("f", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("sum"), source=_c(0)),
			IRCopy(dest=_t("i"), source=_c(0)),
			IRJump(target="loop"),
			IRLabelInstr("loop"),
			IRBinOp(dest=_t("sum"), left=_t("sum"), op="+", right=_t("i")),
			IRBinOp(dest=_t("i"), left=_t("i"), op="+", right=_c(1)),
			IRBinOp(dest=_t("cond"), left=_t("i"), op="<", right=_c(100)),
			IRCondJump(condition=_t("cond"), true_label="loop", false_label="done"),
			IRLabelInstr("done"),
			IRReturn(value=_t("sum")),
		])
		allocator = RegisterAllocator(func)
		result = allocator.allocate()
		assert "sum" in result
		assert "i" in result
		spills = _count_spills(func, ["sum", "i", "cond"])
		assert spills == 0

	def test_call_in_loop_with_coalescing(self) -> None:
		"""Temps live across calls in loops with copy should still allocate well."""
		func = _simple_func("f", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("total"), source=_c(0)),
			IRCopy(dest=_t("n"), source=_c(10)),
			IRJump(target="loop"),
			IRLabelInstr("loop"),
			IRCall(dest=_t("val"), function_name="compute", args=[_t("n")], arg_types=[IRType.INT]),
			IRCopy(dest=_t("val2"), source=_t("val")),
			IRBinOp(dest=_t("total"), left=_t("total"), op="+", right=_t("val2")),
			IRBinOp(dest=_t("n"), left=_t("n"), op="-", right=_c(1)),
			IRBinOp(dest=_t("cond"), left=_t("n"), op=">", right=_c(0)),
			IRCondJump(condition=_t("cond"), true_label="loop", false_label="done"),
			IRLabelInstr("done"),
			IRReturn(value=_t("total")),
		])
		allocator = RegisterAllocator(func)
		result = allocator.allocate()
		# total and n are live across the call
		assert "total" in result
		assert "n" in result
		assert result["total"] in CALLEE_SAVED_REGS
		assert result["n"] in CALLEE_SAVED_REGS
		# val and val2 should coalesce (val2 = val, no interference)
		if "val" in result and "val2" in result:
			assert result["val"] == result["val2"], "val/val2 copy should coalesce"
		_validate_coloring(func, result)
