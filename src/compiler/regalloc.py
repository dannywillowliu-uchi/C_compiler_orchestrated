"""Graph-coloring register allocator using Chaitin-Briggs algorithm with move coalescing."""

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


class _UnionFind:
	"""Union-find (disjoint set) for coalescing temp names."""

	def __init__(self) -> None:
		self._parent: dict[str, str] = {}

	def find(self, x: str) -> str:
		if x not in self._parent:
			self._parent[x] = x
		while self._parent[x] != x:
			self._parent[x] = self._parent[self._parent[x]]
			x = self._parent[x]
		return x

	def union(self, a: str, b: str) -> None:
		ra, rb = self.find(a), self.find(b)
		if ra != rb:
			self._parent[rb] = ra

	def same(self, a: str, b: str) -> bool:
		return self.find(a) == self.find(b)


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


def _collect_move_edges(func: IRFunction) -> list[tuple[str, str]]:
	"""Collect (src, dst) pairs from IRCopy instructions where source is a temp."""
	moves: list[tuple[str, str]] = []
	for instr in func.body:
		if isinstance(instr, IRCopy) and isinstance(instr.source, IRTemp):
			moves.append((instr.source.name, instr.dest.name))
	return moves


def _count_temp_uses(func: IRFunction) -> dict[str, int]:
	"""Count how many times each temp is referenced (defined or used) in the function."""
	counts: dict[str, int] = {}
	for instr in func.body:
		if isinstance(instr, IRBinOp):
			counts[instr.dest.name] = counts.get(instr.dest.name, 0) + 1
			if isinstance(instr.left, IRTemp):
				counts[instr.left.name] = counts.get(instr.left.name, 0) + 1
			if isinstance(instr.right, IRTemp):
				counts[instr.right.name] = counts.get(instr.right.name, 0) + 1
		elif isinstance(instr, IRUnaryOp):
			counts[instr.dest.name] = counts.get(instr.dest.name, 0) + 1
			if isinstance(instr.operand, IRTemp):
				counts[instr.operand.name] = counts.get(instr.operand.name, 0) + 1
		elif isinstance(instr, IRCopy):
			counts[instr.dest.name] = counts.get(instr.dest.name, 0) + 1
			if isinstance(instr.source, IRTemp):
				counts[instr.source.name] = counts.get(instr.source.name, 0) + 1
		elif isinstance(instr, IRLoad):
			counts[instr.dest.name] = counts.get(instr.dest.name, 0) + 1
			if isinstance(instr.address, IRTemp):
				counts[instr.address.name] = counts.get(instr.address.name, 0) + 1
		elif isinstance(instr, IRStore):
			if isinstance(instr.address, IRTemp):
				counts[instr.address.name] = counts.get(instr.address.name, 0) + 1
			if isinstance(instr.value, IRTemp):
				counts[instr.value.name] = counts.get(instr.value.name, 0) + 1
		elif isinstance(instr, IRCall):
			if instr.dest is not None:
				counts[instr.dest.name] = counts.get(instr.dest.name, 0) + 1
			for arg in instr.args:
				if isinstance(arg, IRTemp):
					counts[arg.name] = counts.get(arg.name, 0) + 1
		elif isinstance(instr, IRReturn):
			if instr.value is not None and isinstance(instr.value, IRTemp):
				counts[instr.value.name] = counts.get(instr.value.name, 0) + 1
		elif isinstance(instr, IRConvert):
			counts[instr.dest.name] = counts.get(instr.dest.name, 0) + 1
			if isinstance(instr.source, IRTemp):
				counts[instr.source.name] = counts.get(instr.source.name, 0) + 1
	return counts


class RegisterAllocator:
	"""Chaitin-Briggs graph-coloring register allocator with move coalescing."""

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
		move_edges = _collect_move_edges(self._func)
		use_counts = _count_temp_uses(self._func)

		# Build integer-only interference subgraph
		int_temps = {t for t in interference if t not in float_temps}
		int_graph: dict[str, set[str]] = {}
		for t in int_temps:
			int_graph[t] = interference[t] & int_temps

		# Coalesce moves
		coalesce_map = self._coalesce(int_graph, move_edges, int_temps, call_crossing)

		# Build coalesced graph
		coalesced_graph, rep_map = self._build_coalesced_graph(int_graph, coalesce_map)

		# Map call_crossing and use_counts to representatives
		coalesced_call_crossing: set[str] = set()
		for t in call_crossing:
			rep = coalesce_map.find(t) if t in int_temps else t
			if rep in coalesced_graph:
				coalesced_call_crossing.add(rep)

		coalesced_use_counts: dict[str, int] = {}
		for t in int_temps:
			rep = coalesce_map.find(t)
			coalesced_use_counts[rep] = coalesced_use_counts.get(rep, 0) + use_counts.get(t, 1)

		# Color the coalesced graph
		coloring = self._color_graph(coalesced_graph, coalesced_call_crossing, coalesced_use_counts)

		# Expand coloring back to all original temps
		result: dict[str, str] = {}
		for t in int_temps:
			rep = coalesce_map.find(t)
			if rep in coloring:
				result[t] = coloring[rep]

		return result

	def _coalesce(
		self,
		graph: dict[str, set[str]],
		move_edges: list[tuple[str, str]],
		int_temps: set[str],
		call_crossing: set[str],
	) -> _UnionFind:
		"""Aggressive coalescing: merge non-interfering move-related pairs.

		Uses George's criterion: coalesce (a, b) if every neighbor of b that has
		degree >= K already interferes with a. This is safe because it never
		increases the number of nodes with degree >= K.
		"""
		uf = _UnionFind()
		for t in int_temps:
			uf.find(t)

		# Build a mutable adjacency with live degree tracking
		adj: dict[str, set[str]] = {n: set(neighbors) for n, neighbors in graph.items()}

		for src, dst in move_edges:
			if src not in int_temps or dst not in int_temps:
				continue

			rep_src = uf.find(src)
			rep_dst = uf.find(dst)
			if rep_src == rep_dst:
				continue

			# Don't coalesce if they interfere
			if rep_dst in adj.get(rep_src, set()):
				continue

			# George's criterion: for every neighbor t of rep_dst,
			# either degree(t) < K or t already interferes with rep_src
			neighbors_dst = adj.get(rep_dst, set())
			neighbors_src = adj.get(rep_src, set())

			safe = True
			for neighbor in neighbors_dst:
				if uf.find(neighbor) == rep_src:
					continue
				effective_degree = len(adj.get(neighbor, set()))
				if effective_degree >= K and neighbor not in neighbors_src:
					safe = False
					break

			if not safe:
				continue

			# Don't coalesce if it would mix call-crossing constraint incorrectly:
			# if one is call-crossing, merged result must be too
			# (this is fine, just propagate the constraint)

			# Merge rep_dst into rep_src
			uf.union(rep_src, rep_dst)
			new_rep = uf.find(rep_src)
			other = rep_dst if new_rep == rep_src else rep_src

			# Merge adjacency: new_rep inherits all neighbors of other
			for neighbor in adj.get(other, set()):
				if uf.find(neighbor) == new_rep:
					continue
				adj.setdefault(new_rep, set()).add(neighbor)
				adj.setdefault(neighbor, set()).add(new_rep)
				adj[neighbor].discard(other)

			# Remove old node from adj
			if other in adj:
				del adj[other]
			# Clean up stale references to other in new_rep's neighbors
			adj.get(new_rep, set()).discard(other)

		return uf

	def _build_coalesced_graph(
		self,
		graph: dict[str, set[str]],
		uf: _UnionFind,
	) -> tuple[dict[str, set[str]], dict[str, str]]:
		"""Rebuild interference graph using coalesced representatives."""
		reps: set[str] = set()
		rep_map: dict[str, str] = {}
		for node in graph:
			rep = uf.find(node)
			reps.add(rep)
			rep_map[node] = rep

		coalesced: dict[str, set[str]] = {r: set() for r in reps}
		for node, neighbors in graph.items():
			rn = uf.find(node)
			for neighbor in neighbors:
				rneighbor = uf.find(neighbor)
				if rn != rneighbor:
					coalesced[rn].add(rneighbor)
					coalesced.setdefault(rneighbor, set()).add(rn)

		return coalesced, rep_map

	def _color_graph(
		self,
		graph: dict[str, set[str]],
		call_crossing: set[str],
		use_counts: dict[str, int] | None = None,
	) -> dict[str, str]:
		"""Chaitin-Briggs graph coloring with improved spill heuristic."""
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
				# Spill heuristic: degree / use_count (prefer spilling low-use, high-degree nodes)
				def spill_cost(n: str) -> float:
					degree = len(adj[n] & remaining)
					uses = (use_counts or {}).get(n, 1)
					return degree / max(uses, 1)

				spill_node = max(remaining, key=spill_cost)
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
			# else: actual spill -- temp stays on stack

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
