"""Tests for improved register allocator: move coalescing, use-density spill cost, single-use preference."""

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
	_compute_use_density,
	_count_temp_uses,
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


def _allocate(func: IRFunction) -> dict[str, str]:
	allocator = RegisterAllocator(func)
	return allocator.allocate()


def _validate_coloring(func: IRFunction, result: dict[str, str]) -> None:
	"""Assert no two interfering temps share a register."""
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


def _count_spills(func: IRFunction, temps: list[str]) -> int:
	result = _allocate(func)
	return sum(1 for t in temps if t not in result)


# ---------------------------------------------------------------------------
# Move coalescing tests
# ---------------------------------------------------------------------------


class TestMoveCoalescingV2:
	def test_coalesce_eliminates_copy_in_tight_loop(self) -> None:
		"""Copy inside a tight loop body should coalesce, reducing move overhead."""
		func = _simple_func("tight_loop", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("acc"), source=_c(0)),
			IRCopy(dest=_t("i"), source=_c(0)),
			IRJump(target="hdr"),
			IRLabelInstr("hdr"),
			IRBinOp(dest=_t("cond"), left=_t("i"), op="<", right=_c(1000)),
			IRCondJump(condition=_t("cond"), true_label="body", false_label="exit"),
			IRLabelInstr("body"),
			IRBinOp(dest=_t("tmp"), left=_t("acc"), op="+", right=_t("i")),
			IRCopy(dest=_t("acc2"), source=_t("tmp")),  # should coalesce with tmp
			IRCopy(dest=_t("acc"), source=_t("acc2")),
			IRBinOp(dest=_t("i"), left=_t("i"), op="+", right=_c(1)),
			IRJump(target="hdr"),
			IRLabelInstr("exit"),
			IRReturn(value=_t("acc")),
		])
		result = _allocate(func)
		# tmp and acc2 don't interfere, should share a register
		if "tmp" in result and "acc2" in result:
			assert result["tmp"] == result["acc2"], "tmp->acc2 copy should coalesce"
		_validate_coloring(func, result)

	def test_coalesce_multiple_copies_in_sequence(self) -> None:
		"""Chain a -> b -> c -> d should all get the same register."""
		func = _simple_func("chain", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(42)),
			IRCopy(dest=_t("b"), source=_t("a")),
			IRCopy(dest=_t("c"), source=_t("b")),
			IRCopy(dest=_t("d"), source=_t("c")),
			IRReturn(value=_t("d")),
		])
		result = _allocate(func)
		regs = {result.get(t) for t in ["a", "b", "c", "d"]}
		regs.discard(None)
		assert len(regs) == 1, f"Copy chain should coalesce to one register, got {regs}"

	def test_coalesce_does_not_merge_interfering(self) -> None:
		"""Two temps that are simultaneously live must not coalesce even with a copy."""
		func = _simple_func("no_merge", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRCopy(dest=_t("y"), source=_c(2)),
			# x and y both live here
			IRCopy(dest=_t("z"), source=_t("x")),
			IRBinOp(dest=_t("r"), left=_t("z"), op="+", right=_t("y")),
			IRReturn(value=_t("r")),
		])
		result = _allocate(func)
		assert result["x"] != result["y"], "Interfering temps must not share register"
		_validate_coloring(func, result)

	def test_coalesce_across_call_boundary(self) -> None:
		"""Copy of call result should coalesce with the result temp."""
		func = _simple_func("call_copy", [
			IRLabelInstr("entry"),
			IRCall(dest=_t("ret"), function_name="foo", args=[], arg_types=[]),
			IRCopy(dest=_t("v"), source=_t("ret")),
			IRBinOp(dest=_t("r"), left=_t("v"), op="+", right=_c(1)),
			IRReturn(value=_t("r")),
		])
		result = _allocate(func)
		if "ret" in result and "v" in result:
			assert result["ret"] == result["v"], "Copy of call result should coalesce"
		_validate_coloring(func, result)


# ---------------------------------------------------------------------------
# Use-density spill cost tests
# ---------------------------------------------------------------------------


class TestUseDensitySpillCost:
	def test_density_computation_sparse_vs_dense(self) -> None:
		"""A temp used sparsely over a long range has lower density than one used densely."""
		body = [
			IRLabelInstr("entry"),
			# dense: defined and used repeatedly in close proximity
			IRCopy(dest=_t("dense"), source=_c(0)),
			IRBinOp(dest=_t("dense"), left=_t("dense"), op="+", right=_c(1)),
			IRBinOp(dest=_t("dense"), left=_t("dense"), op="+", right=_c(2)),
			IRBinOp(dest=_t("dense"), left=_t("dense"), op="+", right=_c(3)),
			# sparse: defined early, padding, used once at end
			IRCopy(dest=_t("sparse"), source=_c(99)),
			IRCopy(dest=_t("pad1"), source=_c(1)),
			IRCopy(dest=_t("pad2"), source=_c(2)),
			IRCopy(dest=_t("pad3"), source=_c(3)),
			IRCopy(dest=_t("pad4"), source=_c(4)),
			IRBinOp(dest=_t("r"), left=_t("dense"), op="+", right=_t("sparse")),
			IRReturn(value=_t("r")),
		]
		cfg = CFG(body)
		analyzer = LivenessAnalyzer(cfg)
		density = _compute_use_density(cfg, analyzer)
		assert density["dense"] > density["sparse"], (
			f"Dense temp ({density['dense']}) should have higher density than sparse ({density['sparse']})"
		)

	def test_high_density_temp_kept_over_low_density(self) -> None:
		"""Under pressure, a low-density temp should be spilled before a high-density one."""
		body: list = [IRLabelInstr("entry")]
		# K-1 filler temps to create pressure
		for i in range(K - 1):
			body.append(IRCopy(dest=_t(f"f{i}"), source=_c(i)))
		# high-density: used many times in a short span
		body.append(IRCopy(dest=_t("hd"), source=_c(0)))
		body.append(IRBinOp(dest=_t("hd"), left=_t("hd"), op="+", right=_c(1)))
		body.append(IRBinOp(dest=_t("hd"), left=_t("hd"), op="+", right=_c(2)))
		# low-density: defined early, used once much later
		body.append(IRCopy(dest=_t("ld"), source=_c(999)))
		# padding to extend ld's range
		for i in range(4):
			body.append(IRCopy(dest=_t(f"pad{i}"), source=_c(i + 100)))
		# Use everything
		acc = _t("hd")
		for i in range(K - 1):
			nd = _t(f"s{i}")
			body.append(IRBinOp(dest=nd, left=acc, op="+", right=_t(f"f{i}")))
			acc = nd
		body.append(IRBinOp(dest=_t("final"), left=acc, op="+", right=_t("ld")))
		body.append(IRReturn(value=_t("final")))

		func = _simple_func("density_pref", body)
		result = _allocate(func)
		# hd should be kept more aggressively than ld
		assert "hd" in result, "High-density temp should be allocated a register"
		_validate_coloring(func, result)


# ---------------------------------------------------------------------------
# Single-use temp spill preference
# ---------------------------------------------------------------------------


class TestSingleUseSpillPreference:
	def test_single_use_spilled_before_multi_use(self) -> None:
		"""A temp used only once should be spilled before a temp used many times."""
		body: list = [IRLabelInstr("entry")]
		# K temps that are used multiple times (expensive to spill)
		for i in range(K):
			body.append(IRCopy(dest=_t(f"multi{i}"), source=_c(i)))
			body.append(IRBinOp(dest=_t(f"multi{i}"), left=_t(f"multi{i}"), op="+", right=_c(1)))
			body.append(IRBinOp(dest=_t(f"multi{i}"), left=_t(f"multi{i}"), op="+", right=_c(2)))
		# 1 temp used only once (cheap to spill)
		body.append(IRCopy(dest=_t("single"), source=_c(777)))
		# Use all temps together to force simultaneous liveness
		acc = _t("single")
		for i in range(K):
			nd = _t(f"r{i}")
			body.append(IRBinOp(dest=nd, left=acc, op="+", right=_t(f"multi{i}")))
			acc = nd
		body.append(IRReturn(value=acc))

		func = _simple_func("single_use_spill", body)
		result = _allocate(func)

		# K+1 live at the use chain start; at least 1 must spill
		all_temps = [f"multi{i}" for i in range(K)] + ["single"]
		spilled = [t for t in all_temps if t not in result]
		if spilled:
			assert "single" in spilled, (
				f"Single-use temp should be spilled first, but spilled={spilled}"
			)
		_validate_coloring(func, result)

	def test_single_use_count_detected(self) -> None:
		"""_count_temp_uses correctly identifies single-use temps."""
		body = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("once"), source=_c(1)),
			IRCopy(dest=_t("twice"), source=_c(2)),
			IRBinOp(dest=_t("twice"), left=_t("twice"), op="+", right=_c(1)),
			IRBinOp(dest=_t("r"), left=_t("once"), op="+", right=_t("twice")),
			IRReturn(value=_t("r")),
		]
		func = _simple_func("count_test", body)
		counts = _count_temp_uses(func)
		# "once" appears: def(1) + use(1) = 2 refs
		assert counts["once"] == 2
		# "twice" appears: def(1) + use_in_binop(1) + def(1) + use(1) = 4 refs
		assert counts["twice"] >= 4


# ---------------------------------------------------------------------------
# Fewer spills in common patterns
# ---------------------------------------------------------------------------


class TestFewerSpillsTightLoop:
	def test_tight_accumulator_loop_zero_spills(self) -> None:
		"""A simple accumulator loop with counter should produce zero spills."""
		func = _simple_func("accum_loop", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("sum"), source=_c(0)),
			IRCopy(dest=_t("i"), source=_c(0)),
			IRJump(target="hdr"),
			IRLabelInstr("hdr"),
			IRBinOp(dest=_t("cond"), left=_t("i"), op="<", right=_c(100)),
			IRCondJump(condition=_t("cond"), true_label="body", false_label="done"),
			IRLabelInstr("body"),
			IRBinOp(dest=_t("sum"), left=_t("sum"), op="+", right=_t("i")),
			IRBinOp(dest=_t("i"), left=_t("i"), op="+", right=_c(1)),
			IRJump(target="hdr"),
			IRLabelInstr("done"),
			IRReturn(value=_t("sum")),
		])
		spills = _count_spills(func, ["sum", "i", "cond"])
		assert spills == 0, f"Tight loop should have 0 spills, got {spills}"

	def test_nested_loop_inner_temps_not_spilled(self) -> None:
		"""Inner loop temporaries should not be spilled in favor of outer ones."""
		func = _simple_func("nested_loop", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("outer_sum"), source=_c(0)),
			IRCopy(dest=_t("j"), source=_c(0)),
			IRJump(target="outer_hdr"),
			IRLabelInstr("outer_hdr"),
			IRBinOp(dest=_t("oc"), left=_t("j"), op="<", right=_c(10)),
			IRCondJump(condition=_t("oc"), true_label="inner_init", false_label="exit"),
			IRLabelInstr("inner_init"),
			IRCopy(dest=_t("i"), source=_c(0)),
			IRJump(target="inner_hdr"),
			IRLabelInstr("inner_hdr"),
			IRBinOp(dest=_t("ic"), left=_t("i"), op="<", right=_c(10)),
			IRCondJump(condition=_t("ic"), true_label="inner_body", false_label="outer_inc"),
			IRLabelInstr("inner_body"),
			IRBinOp(dest=_t("outer_sum"), left=_t("outer_sum"), op="+", right=_t("i")),
			IRBinOp(dest=_t("i"), left=_t("i"), op="+", right=_c(1)),
			IRJump(target="inner_hdr"),
			IRLabelInstr("outer_inc"),
			IRBinOp(dest=_t("j"), left=_t("j"), op="+", right=_c(1)),
			IRJump(target="outer_hdr"),
			IRLabelInstr("exit"),
			IRReturn(value=_t("outer_sum")),
		])
		result = _allocate(func)
		# Inner loop temps should all get registers
		for t in ["i", "outer_sum", "j"]:
			assert t in result, f"Loop temp {t} should get a register"
		_validate_coloring(func, result)

	def test_loop_with_multiple_accumulators(self) -> None:
		"""Multiple accumulators in a loop should all get registers when K is sufficient."""
		func = _simple_func("multi_accum", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(0)),
			IRCopy(dest=_t("b"), source=_c(0)),
			IRCopy(dest=_t("c"), source=_c(0)),
			IRCopy(dest=_t("i"), source=_c(0)),
			IRJump(target="hdr"),
			IRLabelInstr("hdr"),
			IRBinOp(dest=_t("cond"), left=_t("i"), op="<", right=_c(50)),
			IRCondJump(condition=_t("cond"), true_label="body", false_label="done"),
			IRLabelInstr("body"),
			IRBinOp(dest=_t("a"), left=_t("a"), op="+", right=_t("i")),
			IRBinOp(dest=_t("b"), left=_t("b"), op="+", right=_t("a")),
			IRBinOp(dest=_t("c"), left=_t("c"), op="+", right=_t("b")),
			IRBinOp(dest=_t("i"), left=_t("i"), op="+", right=_c(1)),
			IRJump(target="hdr"),
			IRLabelInstr("done"),
			IRBinOp(dest=_t("r"), left=_t("a"), op="+", right=_t("b")),
			IRBinOp(dest=_t("r2"), left=_t("r"), op="+", right=_t("c")),
			IRReturn(value=_t("r2")),
		])
		spills = _count_spills(func, ["a", "b", "c", "i"])
		assert spills == 0, f"Multiple accumulators should fit in {K} registers, got {spills} spills"


class TestFewerSpillsFunctionCalls:
	def test_call_sequence_with_live_temps(self) -> None:
		"""Temps live across a call sequence should use callee-saved registers."""
		func = _simple_func("call_seq", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(10)),
			IRCopy(dest=_t("y"), source=_c(20)),
			IRCall(dest=_t("r1"), function_name="foo", args=[_t("x")], arg_types=[IRType.INT]),
			# x and y still live after call
			IRBinOp(dest=_t("s1"), left=_t("r1"), op="+", right=_t("y")),
			IRCall(dest=_t("r2"), function_name="bar", args=[_t("s1")], arg_types=[IRType.INT]),
			IRBinOp(dest=_t("s2"), left=_t("r2"), op="+", right=_t("x")),
			IRReturn(value=_t("s2")),
		])
		result = _allocate(func)
		# x and y are live across calls -> should get callee-saved regs
		assert "x" in result, "Temp live across call should get register"
		assert "y" in result, "Temp live across call should get register"
		assert result["x"] in CALLEE_SAVED_REGS, "Call-crossing temp should use callee-saved reg"
		assert result["y"] in CALLEE_SAVED_REGS, "Call-crossing temp should use callee-saved reg"
		_validate_coloring(func, result)

	def test_multiple_calls_in_loop(self) -> None:
		"""Loop with call and copy coalescing should minimize spills."""
		func = _simple_func("calls_in_loop", [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("total"), source=_c(0)),
			IRCopy(dest=_t("n"), source=_c(5)),
			IRJump(target="hdr"),
			IRLabelInstr("hdr"),
			IRBinOp(dest=_t("cond"), left=_t("n"), op=">", right=_c(0)),
			IRCondJump(condition=_t("cond"), true_label="body", false_label="done"),
			IRLabelInstr("body"),
			IRCall(dest=_t("v"), function_name="compute", args=[_t("n")], arg_types=[IRType.INT]),
			IRCopy(dest=_t("v2"), source=_t("v")),  # should coalesce
			IRBinOp(dest=_t("total"), left=_t("total"), op="+", right=_t("v2")),
			IRBinOp(dest=_t("n"), left=_t("n"), op="-", right=_c(1)),
			IRJump(target="hdr"),
			IRLabelInstr("done"),
			IRReturn(value=_t("total")),
		])
		result = _allocate(func)
		assert "total" in result
		assert "n" in result
		# v and v2 should coalesce
		if "v" in result and "v2" in result:
			assert result["v"] == result["v2"], "v/v2 copy should coalesce in loop"
		_validate_coloring(func, result)

	def test_call_sequence_no_unnecessary_spills(self) -> None:
		"""Sequential calls with few live temps should not spill."""
		func = _simple_func("seq_calls", [
			IRLabelInstr("entry"),
			IRCall(dest=_t("a"), function_name="f1", args=[], arg_types=[]),
			IRCall(dest=_t("b"), function_name="f2", args=[], arg_types=[]),
			IRCall(dest=_t("c"), function_name="f3", args=[], arg_types=[]),
			IRBinOp(dest=_t("s1"), left=_t("a"), op="+", right=_t("b")),
			IRBinOp(dest=_t("s2"), left=_t("s1"), op="+", right=_t("c")),
			IRReturn(value=_t("s2")),
		])
		spills = _count_spills(func, ["a", "b", "c"])
		assert spills == 0, f"Sequential calls with 3 temps should not spill, got {spills}"


# ---------------------------------------------------------------------------
# Coloring validity across all patterns
# ---------------------------------------------------------------------------


class TestColoringValidityV2:
	def test_high_pressure_coloring_valid(self) -> None:
		"""Under high register pressure, coloring must remain valid."""
		body: list = [IRLabelInstr("entry")]
		n = K + 4
		for i in range(n):
			body.append(IRCopy(dest=_t(f"t{i}"), source=_c(i)))
		# Force all live simultaneously
		acc = _t("t0")
		for i in range(1, n):
			nd = _t(f"s{i}")
			body.append(IRBinOp(dest=nd, left=acc, op="+", right=_t(f"t{i}")))
			acc = nd
		body.append(IRReturn(value=acc))
		func = _simple_func("high_pressure", body)
		result = _allocate(func)
		_validate_coloring(func, result)

	def test_mixed_coalesce_and_spill_valid(self) -> None:
		"""Mix of coalescing and spilling should produce valid coloring."""
		body: list = [IRLabelInstr("entry")]
		# Create pressure with coalesce candidates -- each pair is independent
		for i in range(K):
			body.append(IRCopy(dest=_t(f"a{i}"), source=_c(i)))
			body.append(IRCopy(dest=_t(f"b{i}"), source=_t(f"a{i}")))
		# Use all b temps
		acc = _t("b0")
		for i in range(1, K):
			nd = _t(f"r{i}")
			body.append(IRBinOp(dest=nd, left=acc, op="+", right=_t(f"b{i}")))
			acc = nd
		body.append(IRReturn(value=acc))
		func = _simple_func("mixed", body)
		result = _allocate(func)
		_validate_coloring(func, result)
		# At least some a/b pairs should coalesce (not all may due to live range overlap)
		coalesced = 0
		for i in range(K):
			a, b = f"a{i}", f"b{i}"
			if a in result and b in result and result[a] == result[b]:
				coalesced += 1
		assert coalesced > 0, "At least some a/b copy pairs should coalesce"
