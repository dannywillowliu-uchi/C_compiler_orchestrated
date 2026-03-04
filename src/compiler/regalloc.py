"""Graph-coloring register allocator using Chaitin-Briggs algorithm."""

from __future__ import annotations

from compiler.cfg import CFG
from compiler.ir import (
	IRBinOp,
	IRCall,
	IRConvert,
	IRCopy,
	IRFunction,
	IRLoad,
	IRProgram,
	IRReturn,
	IRStore,
	IRTemp,
	IRType,
	IRUnaryOp,
)
from compiler.liveness import LivenessAnalyzer

ALLOCATABLE_REGS: list[str] = ["%rbx", "%r12", "%r13", "%r14", "%r15", "%r10"]
CALLEE_SAVED_REGS: set[str] = {"%rbx", "%r12", "%r13", "%r14", "%r15"}
K: int = len(ALLOCATABLE_REGS)

_FLOAT_TYPES: set[IRType] = {IRType.FLOAT, IRType.DOUBLE}


def _collect_float_temps(func: IRFunction) -> set[str]:
	"""Identify temps that are used in floating-point contexts."""
	floats: set[str] = set()
	for instr in func.body:
		if isinstance(instr, (IRBinOp, IRUnaryOp)) and instr.ir_type in _FLOAT_TYPES:
			floats.add(instr.dest.name)
			if isinstance(instr, IRBinOp):
				if isinstance(instr.left, IRTemp):
					floats.add(instr.left.name)
				if isinstance(instr.right, IRTemp):
					floats.add(instr.right.name)
			elif isinstance(instr, IRUnaryOp):
				if isinstance(instr.operand, IRTemp):
					floats.add(instr.operand.name)
		elif isinstance(instr, IRCopy) and instr.ir_type in _FLOAT_TYPES:
			floats.add(instr.dest.name)
			if isinstance(instr.source, IRTemp):
				floats.add(instr.source.name)
		elif isinstance(instr, IRLoad) and instr.ir_type in _FLOAT_TYPES:
			floats.add(instr.dest.name)
		elif isinstance(instr, IRStore) and instr.ir_type in _FLOAT_TYPES:
			if isinstance(instr.value, IRTemp):
				floats.add(instr.value.name)
		elif isinstance(instr, IRConvert):
			if instr.to_type in _FLOAT_TYPES:
				floats.add(instr.dest.name)
			if instr.from_type in _FLOAT_TYPES and isinstance(instr.source, IRTemp):
				floats.add(instr.source.name)
		elif isinstance(instr, IRCall) and instr.dest is not None and instr.return_type in _FLOAT_TYPES:
			floats.add(instr.dest.name)
		elif isinstance(instr, IRReturn) and instr.ir_type in _FLOAT_TYPES:
			if isinstance(instr.value, IRTemp):
				floats.add(instr.value.name)
	return floats


def _find_call_crossing_temps(cfg: CFG, analyzer: LivenessAnalyzer) -> set[str]:
	"""Find temps that are live across function call instructions."""
	call_crossing: set[str] = set()
	for block in cfg.blocks():
		for i, instr in enumerate(block.instructions):
			if isinstance(instr, IRCall):
				live_after = analyzer.get_live_at_point(block, i + 1)
				if instr.dest is not None:
					live_after = live_after - {instr.dest.name}
				call_crossing |= live_after
	return call_crossing


class RegisterAllocator:
	"""Chaitin-Briggs graph-coloring register allocator for integer temps."""

	def __init__(self, func: IRFunction) -> None:
		self._func = func

	def allocate(self) -> dict[str, str]:
		"""Run register allocation, returning temp_name -> physical register mapping.

		Temps not in the returned mapping should use stack slots (spilled or float).
		"""
		cfg = CFG(self._func.body)
		analyzer = LivenessAnalyzer(cfg)
		interference = analyzer.interference_graph()

		if not interference:
			return {}

		float_temps = _collect_float_temps(self._func)
		call_crossing = _find_call_crossing_temps(cfg, analyzer)

		# Build integer-only interference subgraph
		int_temps = {t for t in interference if t not in float_temps}
		int_graph: dict[str, set[str]] = {}
		for t in int_temps:
			int_graph[t] = interference[t] & int_temps

		return self._color_graph(int_graph, call_crossing)

	def _color_graph(
		self,
		graph: dict[str, set[str]],
		call_crossing: set[str],
	) -> dict[str, str]:
		"""Chaitin-Briggs graph coloring. Returns temp -> register mapping."""
		if not graph:
			return {}

		adj: dict[str, set[str]] = {n: set(neighbors) for n, neighbors in graph.items()}
		remaining: set[str] = set(adj.keys())
		stack: list[tuple[str, bool]] = []  # (node, is_potential_spill)

		# Simplify phase: repeatedly remove low-degree nodes, spill when stuck
		while remaining:
			found = False
			for node in sorted(remaining):
				degree = len(adj[node] & remaining)
				if degree < K:
					stack.append((node, False))
					remaining.remove(node)
					found = True
					break

			if not found:
				# Pick potential spill: highest degree node
				spill_node = max(remaining, key=lambda n: len(adj[n] & remaining))
				stack.append((spill_node, True))
				remaining.remove(spill_node)

		# Select phase: pop from stack and assign colors
		coloring: dict[str, str] = {}
		for node, _is_potential_spill in reversed(stack):
			used_colors: set[str] = set()
			for neighbor in adj[node]:
				if neighbor in coloring:
					used_colors.add(coloring[neighbor])

			if node in call_crossing:
				available = [r for r in ALLOCATABLE_REGS if r in CALLEE_SAVED_REGS and r not in used_colors]
			else:
				available = [r for r in ALLOCATABLE_REGS if r not in used_colors]

			if available:
				coloring[node] = available[0]
			# else: actual spill — temp stays on stack

		return coloring


def allocate_registers(program: IRProgram) -> dict[str, dict[str, str]]:
	"""Run register allocation on all functions in the program.

	Returns a mapping: function_name -> {temp_name -> physical_register}.
	"""
	result: dict[str, dict[str, str]] = {}
	for func in program.functions:
		allocator = RegisterAllocator(func)
		mapping = allocator.allocate()
		if mapping:
			result[func.name] = mapping
	return result
