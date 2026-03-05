"""Tests for register allocator short-lived temporary spill reduction."""

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
	_compute_use_distances,
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


class TestUseDistanceComputation:
	def test_immediate_use(self) -> None:
		"""A temp defined and used in the very next instruction has distance 1."""
		body = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRReturn(value=_t("a")),
		]
		cfg = CFG(body)
		distances = _compute_use_distances(cfg)
		assert distances["a"] == 1

	def test_distance_two(self) -> None:
		"""A temp defined then used 2 instructions later has distance 2."""
		body = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRCopy(dest=_t("b"), source=_c(2)),
			IRBinOp(dest=_t("c"), left=_t("a"), op="+", right=_t("b")),
			IRReturn(value=_t("c")),
		]
		cfg = CFG(body)
		distances = _compute_use_distances(cfg)
		assert distances["a"] == 2
		assert distances["b"] == 1

	def test_long_lived_temp(self) -> None:
		"""A temp used many instructions after its definition has a large distance."""
		body = [IRLabelInstr("entry")]
		body.append(IRCopy(dest=_t("long"), source=_c(1)))
		for i in range(10):
			body.append(IRCopy(dest=_t(f"filler{i}"), source=_c(i)))
		body.append(IRBinOp(dest=_t("res"), left=_t("long"), op="+", right=_c(0)))
		body.append(IRReturn(value=_t("res")))
		cfg = CFG(body)
		distances = _compute_use_distances(cfg)
		assert distances["long"] >= 10


class TestShortLivedNotSpilled:
	def test_short_lived_temps_kept_in_registers(self) -> None:
		"""Short-lived temps (distance <= 2) should be kept in registers even under pressure."""
		# Create K+1 temps all live at the same time.
		# One is short-lived (defined, used once, dead).
		# The rest are long-lived. The short-lived one should NOT be spilled.
		body: list = [IRLabelInstr("entry")]

		# Define K long-lived temps
		for i in range(K):
			body.append(IRCopy(dest=_t(f"long{i}"), source=_c(i)))

		# Short-lived temp: defined and used within 1 instruction
		body.append(IRCopy(dest=_t("short"), source=_c(99)))
		body.append(IRBinOp(dest=_t("result"), left=_t("short"), op="+", right=_c(1)))

		# Now use all long-lived temps to keep them alive across the short-lived region
		acc = _t("result")
		for i in range(K):
			new_dest = _t(f"sum{i}")
			body.append(IRBinOp(dest=new_dest, left=acc, op="+", right=_t(f"long{i}")))
			acc = new_dest
		body.append(IRReturn(value=acc))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# The short-lived temp should NOT be spilled
		assert "short" in result, "Short-lived temp 'short' should be in a register"

	def test_multiple_short_lived_sequential(self) -> None:
		"""Sequential short-lived temps should all get registers since they don't overlap."""
		body: list = [IRLabelInstr("entry")]
		body.append(IRCopy(dest=_t("a"), source=_c(1)))
		body.append(IRBinOp(dest=_t("b"), left=_t("a"), op="+", right=_c(2)))
		# a is dead; b is used next
		body.append(IRCopy(dest=_t("c"), source=_c(3)))
		body.append(IRBinOp(dest=_t("d"), left=_t("b"), op="+", right=_t("c")))
		# b and c are dead
		body.append(IRCopy(dest=_t("e"), source=_c(4)))
		body.append(IRBinOp(dest=_t("f"), left=_t("d"), op="+", right=_t("e")))
		body.append(IRReturn(value=_t("f")))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		for t in ["a", "b", "c", "d", "e", "f"]:
			assert t in result, f"Short-lived temp '{t}' should get a register"

	def test_short_lived_preferred_over_long_lived(self) -> None:
		"""When choosing what to spill, long-lived temps should be spilled before short-lived ones."""
		body: list = [IRLabelInstr("entry")]

		# Create K long-lived temps
		long_temps = [f"long{i}" for i in range(K)]
		for t in long_temps:
			body.append(IRCopy(dest=_t(t), source=_c(1)))

		# Add filler instructions to extend the long-lived range
		for i in range(5):
			body.append(IRBinOp(dest=_t(f"filler{i}"), left=_t("long0"), op="+", right=_c(i)))

		# Now define a short-lived temp that overlaps with all long temps
		body.append(IRCopy(dest=_t("short_temp"), source=_c(42)))
		body.append(IRBinOp(dest=_t("use_short"), left=_t("short_temp"), op="+", right=_c(1)))

		# Use all long-lived temps
		acc = _t("use_short")
		for t in long_temps:
			new_dest = _t(f"s_{t}")
			body.append(IRBinOp(dest=new_dest, left=acc, op="+", right=_t(t)))
			acc = new_dest
		body.append(IRReturn(value=acc))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# short_temp should be in a register (not spilled)
		assert "short_temp" in result, "Short-lived temp should be kept in register over long-lived ones"


class TestSpillReductionSimpleExpressions:
	def test_simple_arithmetic_no_spills(self) -> None:
		"""Simple expression: (a + b) * (c - d) should produce zero spills."""
		body: list = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRCopy(dest=_t("b"), source=_c(2)),
			IRBinOp(dest=_t("sum"), left=_t("a"), op="+", right=_t("b")),
			IRCopy(dest=_t("c"), source=_c(3)),
			IRCopy(dest=_t("d"), source=_c(4)),
			IRBinOp(dest=_t("diff"), left=_t("c"), op="-", right=_t("d")),
			IRBinOp(dest=_t("result"), left=_t("sum"), op="*", right=_t("diff")),
			IRReturn(value=_t("result")),
		]

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		all_temps = ["a", "b", "sum", "c", "d", "diff", "result"]
		spilled = [t for t in all_temps if t not in result]
		assert len(spilled) == 0, f"Expected 0 spills for simple expression, got {len(spilled)}: {spilled}"

	def test_chained_operations_no_spills(self) -> None:
		"""Chained binary operations with short-lived intermediates produce no spills."""
		body: list = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(10)),
			IRBinOp(dest=_t("t1"), left=_t("x"), op="+", right=_c(1)),
			IRBinOp(dest=_t("t2"), left=_t("t1"), op="*", right=_c(2)),
			IRBinOp(dest=_t("t3"), left=_t("t2"), op="-", right=_c(3)),
			IRBinOp(dest=_t("t4"), left=_t("t3"), op="+", right=_c(4)),
			IRReturn(value=_t("t4")),
		]

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		all_temps = ["x", "t1", "t2", "t3", "t4"]
		spilled = [t for t in all_temps if t not in result]
		assert len(spilled) == 0, f"Expected 0 spills for chained ops, got {len(spilled)}: {spilled}"

	def test_conditional_expression_short_lived(self) -> None:
		"""Short-lived temps in conditional branches should not be spilled."""
		body: list = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(5)),
			IRBinOp(dest=_t("cmp"), left=_t("x"), op=">", right=_c(0)),
			IRCondJump(condition=_t("cmp"), true_label="pos", false_label="neg"),
			IRLabelInstr("pos"),
			IRBinOp(dest=_t("r1"), left=_t("x"), op="+", right=_c(1)),
			IRJump(target="end"),
			IRLabelInstr("neg"),
			IRBinOp(dest=_t("r2"), left=_t("x"), op="-", right=_c(1)),
			IRJump(target="end"),
			IRLabelInstr("end"),
			IRReturn(value=_t("x")),
		]

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# x is used across branches but short-lived temps should all get registers
		assert "x" in result, "Temp 'x' should be in a register"


class TestColoringValidity:
	def test_short_lived_coloring_valid(self) -> None:
		"""Coloring must be valid even with short-lived temp bias."""
		body: list = [IRLabelInstr("entry")]

		# Create pressure with K+2 temps
		for i in range(K + 2):
			body.append(IRCopy(dest=_t(f"v{i}"), source=_c(i)))

		# Short-lived computation
		body.append(IRCopy(dest=_t("short"), source=_c(99)))
		body.append(IRBinOp(dest=_t("sr"), left=_t("short"), op="+", right=_c(1)))

		# Use all long-lived temps
		acc = _t("sr")
		for i in range(K + 2):
			new_dest = _t(f"r{i}")
			body.append(IRBinOp(dest=new_dest, left=acc, op="+", right=_t(f"v{i}")))
			acc = new_dest
		body.append(IRReturn(value=acc))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# Validate: no two interfering temps share a color
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

	def test_mixed_short_long_lived_valid(self) -> None:
		"""Mixed short and long-lived temps produce a valid coloring."""
		body: list = [IRLabelInstr("entry")]

		# Long-lived setup
		for i in range(K):
			body.append(IRCopy(dest=_t(f"L{i}"), source=_c(i)))

		# Interleaved short-lived computations
		for i in range(3):
			body.append(IRCopy(dest=_t(f"s{i}"), source=_c(i * 10)))
			body.append(IRBinOp(dest=_t(f"s{i}r"), left=_t(f"s{i}"), op="+", right=_c(1)))

		# Use all long-lived
		acc = _t("s2r")
		for i in range(K):
			new_dest = _t(f"u{i}")
			body.append(IRBinOp(dest=new_dest, left=acc, op="+", right=_t(f"L{i}")))
			acc = new_dest
		body.append(IRReturn(value=acc))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# Validate coloring
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
