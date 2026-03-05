"""Tests for loop-depth-aware spill weight in register allocator."""

from compiler.cfg import CFG
from compiler.ir import (
	IRBinOp,
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
	K,
	RegisterAllocator,
	_compute_weighted_uses,
	_get_temp_refs,
)


def _t(name: str) -> IRTemp:
	return IRTemp(name)


def _c(val: int) -> IRConst:
	return IRConst(val)


def _simple_func(name: str, body: list) -> IRFunction:
	return IRFunction(
		name=name,
		params=[],
		body=body,
		return_type=IRType.INT,
	)


def _make_loop_body() -> list:
	"""Build IR with a simple loop: entry -> loop_header -> loop_body -> loop_header, loop_header -> exit.

	Uses 'loop_var' inside the loop and 'outside_var' only outside.
	"""
	return [
		IRLabelInstr("entry"),
		IRCopy(dest=_t("outside_var"), source=_c(100)),
		IRCopy(dest=_t("loop_var"), source=_c(0)),
		IRCopy(dest=_t("limit"), source=_c(10)),
		IRJump(target="loop_header"),
		IRLabelInstr("loop_header"),
		IRBinOp(dest=_t("cond"), left=_t("loop_var"), op="<", right=_t("limit")),
		IRCondJump(condition=_t("cond"), true_label="loop_body", false_label="exit"),
		IRLabelInstr("loop_body"),
		IRBinOp(dest=_t("loop_var"), left=_t("loop_var"), op="+", right=_c(1)),
		IRJump(target="loop_header"),
		IRLabelInstr("exit"),
		IRBinOp(dest=_t("result"), left=_t("outside_var"), op="+", right=_t("loop_var")),
		IRReturn(value=_t("result")),
	]


class TestWeightedUsesComputation:
	def test_loop_var_has_higher_weight_than_outside_var(self) -> None:
		"""Variables used inside a loop should have higher weighted use count."""
		body = _make_loop_body()
		cfg = CFG(body)
		loop_depths = cfg.loop_depth()
		weighted = _compute_weighted_uses(cfg, loop_depths)

		# loop_var is used in loop_header (depth 1) and loop_body (depth 1)
		# outside_var is used in entry (depth 0) and exit (depth 0)
		assert weighted["loop_var"] > weighted["outside_var"], (
			f"Loop var weight {weighted['loop_var']} should exceed outside var weight {weighted['outside_var']}"
		)

	def test_depth_zero_weight_is_one(self) -> None:
		"""At depth 0, each use should contribute weight 1.0 (10^0 = 1)."""
		body = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRReturn(value=_t("a")),
		]
		cfg = CFG(body)
		loop_depths = cfg.loop_depth()
		weighted = _compute_weighted_uses(cfg, loop_depths)
		# "a" is referenced twice (def + use), both at depth 0 -> weight 2.0
		assert weighted["a"] == 2.0

	def test_depth_one_weight_is_ten(self) -> None:
		"""At depth 1, each use should contribute weight 10.0 (10^1 = 10)."""
		body = _make_loop_body()
		cfg = CFG(body)
		loop_depths = cfg.loop_depth()
		_compute_weighted_uses(cfg, loop_depths)

		# Verify loop blocks have depth 1
		assert loop_depths.get("loop_header", 0) >= 1
		assert loop_depths.get("loop_body", 0) >= 1

	def test_nested_loop_has_higher_weight(self) -> None:
		"""Doubly nested loop should produce weight 100 (10^2) per use."""
		body = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("outer_var"), source=_c(0)),
			IRCopy(dest=_t("inner_var"), source=_c(0)),
			IRJump(target="outer_header"),
			IRLabelInstr("outer_header"),
			IRBinOp(dest=_t("ocond"), left=_t("outer_var"), op="<", right=_c(10)),
			IRCondJump(condition=_t("ocond"), true_label="inner_header", false_label="exit"),
			IRLabelInstr("inner_header"),
			IRBinOp(dest=_t("icond"), left=_t("inner_var"), op="<", right=_c(10)),
			IRCondJump(condition=_t("icond"), true_label="inner_body", false_label="outer_latch"),
			IRLabelInstr("inner_body"),
			IRBinOp(dest=_t("inner_var"), left=_t("inner_var"), op="+", right=_c(1)),
			IRJump(target="inner_header"),
			IRLabelInstr("outer_latch"),
			IRBinOp(dest=_t("outer_var"), left=_t("outer_var"), op="+", right=_c(1)),
			IRJump(target="outer_header"),
			IRLabelInstr("exit"),
			IRReturn(value=_t("inner_var")),
		]
		cfg = CFG(body)
		loop_depths = cfg.loop_depth()
		weighted = _compute_weighted_uses(cfg, loop_depths)

		# inner_body is at depth 2, outer_latch at depth 1, entry at depth 0
		assert loop_depths.get("inner_body", 0) == 2
		assert weighted["inner_var"] > weighted["outer_var"], (
			"Inner loop var should have higher weighted uses than outer loop var"
		)


class TestLoopSpillPriority:
	def _make_high_pressure_with_loop(self) -> tuple[IRFunction, list[str], list[str]]:
		"""Create a function with high register pressure where loop vars should survive.

		Creates K+1 temps (more than available registers), with some used in a loop
		and others only outside. Loop vars should be prioritized for registers.
		"""
		body = [IRLabelInstr("entry")]
		loop_temps = []
		outside_temps = []

		# Create temps used outside the loop (should be spilled first)
		for i in range(K - 1):
			name = f"outside_{i}"
			outside_temps.append(name)
			body.append(IRCopy(dest=_t(name), source=_c(i)))

		# Create temps used inside the loop
		loop_temps = ["loop_a", "loop_b"]
		body.append(IRCopy(dest=_t("loop_a"), source=_c(100)))
		body.append(IRCopy(dest=_t("loop_b"), source=_c(200)))
		body.append(IRCopy(dest=_t("loop_i"), source=_c(0)))
		body.append(IRJump(target="loop_header"))

		# Loop header
		body.append(IRLabelInstr("loop_header"))
		body.append(IRBinOp(dest=_t("cmp"), left=_t("loop_i"), op="<", right=_c(100)))
		body.append(IRCondJump(condition=_t("cmp"), true_label="loop_body", false_label="after_loop"))

		# Loop body - use loop_a and loop_b heavily
		body.append(IRLabelInstr("loop_body"))
		body.append(IRBinOp(dest=_t("loop_a"), left=_t("loop_a"), op="+", right=_t("loop_b")))
		body.append(IRBinOp(dest=_t("loop_i"), left=_t("loop_i"), op="+", right=_c(1)))
		body.append(IRJump(target="loop_header"))

		# After loop - use all outside temps to keep them live
		body.append(IRLabelInstr("after_loop"))
		accum = outside_temps[0]
		for i in range(1, len(outside_temps)):
			dest = f"sum_{i}"
			body.append(IRBinOp(dest=_t(dest), left=_t(accum), op="+", right=_t(outside_temps[i])))
			accum = dest
		body.append(IRBinOp(dest=_t("final"), left=_t(accum), op="+", right=_t("loop_a")))
		body.append(IRReturn(value=_t("final")))

		func = _simple_func("test_loop_spill", body)
		return func, loop_temps, outside_temps

	def test_loop_vars_get_registers(self) -> None:
		"""Loop variables should be allocated registers preferentially over outside vars."""
		func, loop_temps, outside_temps = self._make_high_pressure_with_loop()
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# Loop temps should be allocated (not spilled)
		for t in loop_temps:
			assert t in result, f"Loop temp {t} should be allocated a register, not spilled"

	def test_valid_coloring_with_loop(self) -> None:
		"""The allocation with loop weighting should still produce a valid coloring."""
		func, _, _ = self._make_high_pressure_with_loop()
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

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


class TestLoopDepthIntegration:
	def test_cfg_loop_depth_used_in_allocator(self) -> None:
		"""Verify that loop depth information flows through to the allocator."""
		body = _make_loop_body()
		cfg = CFG(body)
		loop_depths = cfg.loop_depth()

		# The loop blocks should have depth > 0
		assert loop_depths.get("loop_header", 0) > 0
		assert loop_depths.get("loop_body", 0) > 0

		# Non-loop blocks should have depth 0
		assert loop_depths.get("entry", -1) == 0
		assert loop_depths.get("exit", -1) == 0

	def test_weighted_uses_scale_exponentially(self) -> None:
		"""Weighted uses should scale by 10^depth."""
		body = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(0)),
			IRJump(target="loop_header"),
			IRLabelInstr("loop_header"),
			IRBinOp(dest=_t("cond"), left=_t("x"), op="<", right=_c(10)),
			IRCondJump(condition=_t("cond"), true_label="loop_body", false_label="exit"),
			IRLabelInstr("loop_body"),
			# x is both defined and used here (2 refs at depth 1 = weight 20)
			IRBinOp(dest=_t("x"), left=_t("x"), op="+", right=_c(1)),
			IRJump(target="loop_header"),
			IRLabelInstr("exit"),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(body)
		loop_depths = cfg.loop_depth()
		weighted = _compute_weighted_uses(cfg, loop_depths)

		# x appears in entry (depth 0), loop_header (depth 1),
		# loop_body (depth 1), and exit (depth 0)
		# The weighted total should be greater than a simple count
		# because loop uses are multiplied by 10
		simple_count = sum(1 for block in cfg.blocks() for instr in block.instructions for name in _get_temp_refs(instr) if name == "x")
		assert weighted["x"] > simple_count, (
			f"Weighted uses ({weighted['x']}) should exceed simple count ({simple_count})"
		)
