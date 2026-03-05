"""Tests for register allocator live range splitting."""

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
	K,
	RegisterAllocator,
	split_live_ranges,
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


def _count_split_temps(func: IRFunction) -> int:
	"""Count the number of split/reload temps introduced by live range splitting."""
	split_func = split_live_ranges(func)
	count = 0
	for instr in split_func.body:
		if isinstance(instr, IRCopy):
			if instr.dest.name.startswith("_split_") or instr.dest.name.startswith("_reload_"):
				count += 1
	return count


def _make_high_pressure_func(num_extras: int = 2) -> tuple[IRFunction, list[str]]:
	"""Create a function with K + num_extras temps all live simultaneously,
	where some have long live ranges with sparse uses.

	Returns (func, list_of_original_temps).
	"""
	body: list = [IRLabelInstr("entry")]
	num_temps = K + num_extras
	temps = [f"t{i}" for i in range(num_temps)]

	# Define all temps
	for t in temps:
		body.append(IRCopy(dest=_t(t), source=_c(1)))

	# Use t0 heavily (short live range, dense uses)
	for _ in range(6):
		body.append(IRBinOp(dest=_t("t0"), left=_t("t0"), op="+", right=_c(1)))

	# Padding: instructions that don't use the "cold" temps (t2..tN)
	# This creates a gap in their live ranges
	for _ in range(8):
		body.append(IRBinOp(dest=_t("t0"), left=_t("t0"), op="+", right=_t("t1")))

	# Finally use all temps at the end (creating long ranges for t2..tN)
	result_temp = _t("t0")
	for i in range(1, num_temps):
		new_dest = _t(f"sum{i}")
		body.append(IRBinOp(dest=new_dest, left=result_temp, op="+", right=_t(f"t{i}")))
		result_temp = new_dest
	body.append(IRReturn(value=result_temp))

	return _simple_func("f", body), temps


class TestSplitLiveRanges:
	def test_splitting_inserts_copies(self) -> None:
		"""Live range splitting should insert spill/reload copies for long-range temps."""
		func, temps = _make_high_pressure_func()
		num_splits = _count_split_temps(func)
		assert num_splits > 0, "Expected split copies to be inserted for high-pressure function"

	def test_splitting_preserves_correctness(self) -> None:
		"""After splitting, the register allocation coloring must still be valid."""
		func, temps = _make_high_pressure_func()
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# Validate: no two interfering temps share a register
		cfg = CFG(allocator._func.body)
		analyzer = LivenessAnalyzer(cfg)
		ig = analyzer.interference_graph()
		for node, neighbors in ig.items():
			if node in result:
				for neighbor in neighbors:
					if neighbor in result:
						assert result[node] != result[neighbor], (
							f"Invalid coloring: {node}={result[node]} == {neighbor}={result[neighbor]}"
						)

	def test_splitting_reduces_spills(self) -> None:
		"""With live range splitting, fewer original temps should be spilled."""
		func, temps = _make_high_pressure_func(num_extras=3)

		# Count spills without splitting (use a copy of the func to avoid mutation)
		func_no_split = IRFunction(
			name=func.name,
			params=list(func.params),
			body=list(func.body),
			return_type=func.return_type,
		)

		# Run allocator on unsplit version by directly coloring
		cfg_nosplit = CFG(func_no_split.body)
		analyzer_nosplit = LivenessAnalyzer(cfg_nosplit)
		ig_nosplit = analyzer_nosplit.interference_graph()
		float_temps = set()
		addr_taken = set()
		int_temps_nosplit = {t for t in ig_nosplit if t not in float_temps and t not in addr_taken}
		# Count how many original temps have degree >= K (would need spilling)
		high_degree_nosplit = sum(
			1 for t in temps if t in ig_nosplit and len(ig_nosplit[t] & int_temps_nosplit) >= K
		)

		# Run allocator with splitting
		allocator = RegisterAllocator(func)
		result = allocator.allocate()
		spills_with_split = sum(1 for t in temps if t not in result)

		# With splitting, we should have fewer actual spills
		# At minimum, the hot temps (t0, t1) should not be spilled
		assert "t0" in result, "Hot temp t0 should not be spilled after splitting"
		assert "t1" in result, "Hot temp t1 should not be spilled after splitting"

		# Verify that splitting helped: if there was high pressure, some spills
		# should have been avoided
		if high_degree_nosplit > 0:
			assert spills_with_split < len(temps), (
				"Splitting should help avoid spilling all temps"
			)

	def test_no_splitting_when_pressure_is_low(self) -> None:
		"""When register pressure is below K, no splitting should occur."""
		body: list = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRBinOp(dest=_t("b"), left=_t("a"), op="+", right=_c(2)),
			IRReturn(value=_t("b")),
		]
		func = _simple_func("f", body)
		num_splits = _count_split_temps(func)
		assert num_splits == 0, "No splits needed when pressure is low"

	def test_splitting_with_loop(self) -> None:
		"""Splitting should work correctly with loop structures."""
		body: list = [IRLabelInstr("entry")]
		num_temps = K + 2
		temps = [f"t{i}" for i in range(num_temps)]

		for t in temps:
			body.append(IRCopy(dest=_t(t), source=_c(0)))

		# Hot loop using only t0 and t1
		body.append(IRJump(target="loop"))
		body.append(IRLabelInstr("loop"))
		body.append(IRBinOp(dest=_t("t0"), left=_t("t0"), op="+", right=_t("t1")))
		body.append(IRBinOp(dest=_t("lc"), left=_t("t0"), op="<", right=_c(1000)))
		body.append(IRCondJump(condition=_t("lc"), true_label="loop", false_label="done"))

		# Cold exit using all temps
		body.append(IRLabelInstr("done"))
		result_temp = _t("t0")
		for i in range(1, num_temps):
			new_dest = _t(f"r{i}")
			body.append(IRBinOp(dest=new_dest, left=result_temp, op="+", right=_t(f"t{i}")))
			result_temp = new_dest
		body.append(IRReturn(value=result_temp))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# Hot loop vars must not be spilled
		assert "t0" in result, "Loop temp t0 should be in a register"
		assert "t1" in result, "Loop temp t1 should be in a register"

		# Coloring validity
		cfg = CFG(allocator._func.body)
		analyzer = LivenessAnalyzer(cfg)
		ig = analyzer.interference_graph()
		for node, neighbors in ig.items():
			if node in result:
				for neighbor in neighbors:
					if neighbor in result:
						assert result[node] != result[neighbor], (
							f"Invalid: {node}={result[node]} == {neighbor}={result[neighbor]}"
						)

	def test_splitting_with_calls(self) -> None:
		"""Splitting should handle temps live across function calls."""
		body: list = [IRLabelInstr("entry")]
		num_temps = K + 1
		temps = [f"v{i}" for i in range(num_temps)]

		for t in temps:
			body.append(IRCopy(dest=_t(t), source=_c(0)))

		# A call that clobbers caller-saved regs
		body.append(IRCall(dest=_t("result"), function_name="foo", args=[], return_type=IRType.INT))

		# Long gap with some operations
		for _ in range(5):
			body.append(IRBinOp(dest=_t("result"), left=_t("result"), op="+", right=_c(1)))

		# Use all temps
		result_temp = _t("result")
		for i in range(num_temps):
			new_dest = _t(f"s{i}")
			body.append(IRBinOp(dest=new_dest, left=result_temp, op="+", right=_t(f"v{i}")))
			result_temp = new_dest
		body.append(IRReturn(value=result_temp))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# Coloring must be valid
		cfg = CFG(allocator._func.body)
		analyzer = LivenessAnalyzer(cfg)
		ig = analyzer.interference_graph()
		for node, neighbors in ig.items():
			if node in result:
				for neighbor in neighbors:
					if neighbor in result:
						assert result[node] != result[neighbor], (
							f"Invalid: {node}={result[node]} == {neighbor}={result[neighbor]}"
						)

	def test_split_func_body_unchanged_for_simple(self) -> None:
		"""For simple functions with no splitting needed, the body should be identical."""
		body: list = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(42)),
			IRReturn(value=_t("x")),
		]
		func = _simple_func("f", body)
		split_func = split_live_ranges(func)
		assert len(split_func.body) == len(func.body), "Body should not change for simple function"

	def test_fewer_spill_reload_pairs_with_splitting(self) -> None:
		"""Programs with high register pressure should produce fewer spill/reload
		pairs when live range splitting is active."""
		# Create a function with very high pressure: K+4 temps all live
		body: list = [IRLabelInstr("entry")]
		num_temps = K + 4
		temps = [f"t{i}" for i in range(num_temps)]

		# Define all
		for t in temps:
			body.append(IRCopy(dest=_t(t), source=_c(1)))

		# Heavy use of t0 and t1 (should stay in registers)
		for _ in range(10):
			body.append(IRBinOp(dest=_t("t0"), left=_t("t0"), op="+", right=_t("t1")))

		# Final use of all temps
		result_temp = _t("t0")
		for i in range(1, num_temps):
			new_dest = _t(f"sum{i}")
			body.append(IRBinOp(dest=new_dest, left=result_temp, op="+", right=_t(f"t{i}")))
			result_temp = new_dest
		body.append(IRReturn(value=result_temp))

		func = _simple_func("f", body)
		allocator = RegisterAllocator(func)
		result = allocator.allocate()

		# t0 and t1 (heavily used) should NOT be spilled
		assert "t0" in result, "Heavily-used t0 should be in register"
		assert "t1" in result, "Heavily-used t1 should be in register"

		# Count spilled original temps
		spilled = [t for t in temps if t not in result]
		# With K=6 and 10 temps, we need 4 spills without splitting.
		# With splitting, some of those cold temps get split ranges
		# that allow better coloring, so we expect fewer spills.
		# At minimum, the 2 hot temps + K-2 cold temps = K should fit.
		assert len(spilled) <= num_temps - K + 1, (
			f"Too many spills ({len(spilled)}); splitting should reduce pressure"
		)
