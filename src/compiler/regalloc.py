"""Graph-coloring register allocator using Chaitin-Briggs algorithm with move coalescing."""

from __future__ import annotations

from compiler.cfg import CFG
from compiler.ir import (
	IRAddrOf,
	IRBinOp,
	IRBulkCopy,
	IRCall,
	IRConst,
	IRConvert,
	IRCopy,
	IRFloatConst,
	IRFunction,
	IRGlobalRef,
	IRInstruction,
	IRLabelInstr,
	IRLoad,
	IRProgram,
	IRReturn,
	IRStore,
	IRTemp,
	IRType,
	IRUnaryOp,
	IRVaArg,
	IRVaCopy,
	IRVaEnd,
	IRVaStart,
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


def _get_temp_refs(instr: IRInstruction) -> list[str]:
	"""Return all temp names referenced (defined or used) by an instruction."""
	refs: list[str] = []
	if isinstance(instr, IRBinOp):
		refs.append(instr.dest.name)
		if isinstance(instr.left, IRTemp):
			refs.append(instr.left.name)
		if isinstance(instr.right, IRTemp):
			refs.append(instr.right.name)
	elif isinstance(instr, IRUnaryOp):
		refs.append(instr.dest.name)
		if isinstance(instr.operand, IRTemp):
			refs.append(instr.operand.name)
	elif isinstance(instr, IRCopy):
		refs.append(instr.dest.name)
		if isinstance(instr.source, IRTemp):
			refs.append(instr.source.name)
	elif isinstance(instr, IRLoad):
		refs.append(instr.dest.name)
		if isinstance(instr.address, IRTemp):
			refs.append(instr.address.name)
	elif isinstance(instr, IRStore):
		if isinstance(instr.address, IRTemp):
			refs.append(instr.address.name)
		if isinstance(instr.value, IRTemp):
			refs.append(instr.value.name)
	elif isinstance(instr, IRCall):
		if instr.dest is not None:
			refs.append(instr.dest.name)
		for arg in instr.args:
			if isinstance(arg, IRTemp):
				refs.append(arg.name)
	elif isinstance(instr, IRReturn):
		if instr.value is not None and isinstance(instr.value, IRTemp):
			refs.append(instr.value.name)
	elif isinstance(instr, IRConvert):
		refs.append(instr.dest.name)
		if isinstance(instr.source, IRTemp):
			refs.append(instr.source.name)
	elif isinstance(instr, IRAddrOf):
		refs.append(instr.dest.name)
		refs.append(instr.source.name)
	elif isinstance(instr, IRVaStart):
		if isinstance(instr.ap_addr, IRTemp):
			refs.append(instr.ap_addr.name)
	elif isinstance(instr, IRVaArg):
		refs.append(instr.dest.name)
		if isinstance(instr.ap_addr, IRTemp):
			refs.append(instr.ap_addr.name)
	elif isinstance(instr, IRVaEnd):
		if isinstance(instr.ap_addr, IRTemp):
			refs.append(instr.ap_addr.name)
	elif isinstance(instr, IRVaCopy):
		if isinstance(instr.dest_addr, IRTemp):
			refs.append(instr.dest_addr.name)
		if isinstance(instr.src_addr, IRTemp):
			refs.append(instr.src_addr.name)
	elif isinstance(instr, IRBulkCopy):
		if isinstance(instr.dest_addr, IRTemp):
			refs.append(instr.dest_addr.name)
		if isinstance(instr.src_addr, IRTemp):
			refs.append(instr.src_addr.name)
	return refs


def _count_temp_uses(func: IRFunction) -> dict[str, int]:
	"""Count how many times each temp is referenced (defined or used) in the function."""
	counts: dict[str, int] = {}
	for instr in func.body:
		for name in _get_temp_refs(instr):
			counts[name] = counts.get(name, 0) + 1
	return counts


def _compute_weighted_uses(cfg: CFG, loop_depths: dict[str, int]) -> dict[str, float]:
	"""Count temp uses weighted by loop depth: uses at depth d count as 10^d."""
	weighted: dict[str, float] = {}
	for block in cfg.blocks():
		depth = loop_depths.get(block.label, 0)
		weight = 10.0 ** depth
		for instr in block.instructions:
			for name in _get_temp_refs(instr):
				weighted[name] = weighted.get(name, 0.0) + weight
	return weighted


def _find_rematerializable_temps(func: IRFunction) -> set[str]:
	"""Identify temps that can be rematerialized instead of spilled.

	A temp is rematerializable if it's defined by loading a constant or a
	global reference -- these can be recomputed cheaply instead of
	spilling to memory.
	"""
	remat: set[str] = set()
	# Track which temps are defined more than once (can't safely remat)
	def_counts: dict[str, int] = {}
	for instr in func.body:
		if isinstance(instr, IRCopy):
			def_counts[instr.dest.name] = def_counts.get(instr.dest.name, 0) + 1
			if isinstance(instr.source, (IRConst, IRFloatConst, IRGlobalRef)):
				remat.add(instr.dest.name)
		elif isinstance(instr, (IRBinOp, IRUnaryOp, IRLoad, IRCall, IRConvert, IRAddrOf)):
			def_counts[instr.dest.name] = def_counts.get(instr.dest.name, 0) + 1
			# Remove from remat if redefined by non-const instruction
			remat.discard(instr.dest.name)

	# Exclude temps defined more than once (e.g. in loops)
	for name, count in def_counts.items():
		if count > 1:
			remat.discard(name)

	return remat


def _compute_use_density(cfg: CFG, analyzer: LivenessAnalyzer) -> dict[str, float]:
	"""Compute use density: number of uses / live range length.

	Temps used infrequently over a long range have low density and are
	cheap to spill. Temps used frequently over a short range are expensive.
	"""
	first_seen: dict[str, int] = {}
	last_seen: dict[str, int] = {}
	use_count: dict[str, int] = {}
	pos = 0
	for block in cfg.blocks():
		for instr in block.instructions:
			refs = _get_temp_refs(instr)
			for name in refs:
				if name not in first_seen:
					first_seen[name] = pos
				last_seen[name] = pos
				use_count[name] = use_count.get(name, 0) + 1
			pos += 1
	density: dict[str, float] = {}
	for name in first_seen:
		span = max(last_seen[name] - first_seen[name], 1)
		density[name] = use_count.get(name, 1) / span
	return density


def _compute_use_distances(cfg: CFG) -> dict[str, int]:
	"""Compute use distance for each temp: gap between its definition and last use.

	Returns the number of instructions between def and last use.  Temps with
	a distance of 0-2 are considered "short-lived" and should be prioritised
	for register retention (i.e. expensive to spill).
	"""
	def_pos: dict[str, int] = {}
	last_use_pos: dict[str, int] = {}
	pos = 0
	for block in cfg.blocks():
		for instr in block.instructions:
			if isinstance(instr, IRLabelInstr):
				continue
			for name in _get_temp_refs(instr):
				if _instr_defs_temp(instr, name) and name not in def_pos:
					def_pos[name] = pos
				if _instr_uses_temp(instr, name):
					last_use_pos[name] = pos
			pos += 1
	distances: dict[str, int] = {}
	for name in def_pos:
		if name in last_use_pos:
			distances[name] = last_use_pos[name] - def_pos[name]
		else:
			distances[name] = 0
	return distances


def _compute_live_range_lengths(cfg: CFG, analyzer: LivenessAnalyzer) -> dict[str, int]:
	"""Compute approximate live range length for each temp (number of instructions it spans)."""
	first_seen: dict[str, int] = {}
	last_seen: dict[str, int] = {}
	pos = 0
	for block in cfg.blocks():
		for i, instr in enumerate(block.instructions):
			live = analyzer.get_live_at_point(block, i)
			for name in live:
				if name not in first_seen:
					first_seen[name] = pos
				last_seen[name] = pos
			pos += 1
	lengths: dict[str, int] = {}
	for name in first_seen:
		lengths[name] = max(last_seen[name] - first_seen[name], 1)
	return lengths


def _replace_temp_in_value(val: object, old_name: str, new_name: str) -> object:
	"""Replace a temp name in an IRValue, returning the (possibly new) value."""
	if isinstance(val, IRTemp) and val.name == old_name:
		return IRTemp(new_name)
	return val


def _replace_temp_uses_in_instr(instr: IRInstruction, old_name: str, new_name: str) -> IRInstruction:
	"""Return a copy of instr with uses (not defs) of old_name replaced by new_name."""
	if isinstance(instr, IRBinOp):
		return IRBinOp(
			dest=instr.dest,
			left=_replace_temp_in_value(instr.left, old_name, new_name),
			op=instr.op,
			right=_replace_temp_in_value(instr.right, old_name, new_name),
			ir_type=instr.ir_type,
			is_unsigned=instr.is_unsigned,
		)
	elif isinstance(instr, IRUnaryOp):
		return IRUnaryOp(
			dest=instr.dest,
			op=instr.op,
			operand=_replace_temp_in_value(instr.operand, old_name, new_name),
			ir_type=instr.ir_type,
		)
	elif isinstance(instr, IRCopy):
		return IRCopy(
			dest=instr.dest,
			source=_replace_temp_in_value(instr.source, old_name, new_name),
			ir_type=instr.ir_type,
		)
	elif isinstance(instr, IRLoad):
		return IRLoad(
			dest=instr.dest,
			address=_replace_temp_in_value(instr.address, old_name, new_name),
			ir_type=instr.ir_type,
		)
	elif isinstance(instr, IRStore):
		return IRStore(
			address=_replace_temp_in_value(instr.address, old_name, new_name),
			value=_replace_temp_in_value(instr.value, old_name, new_name),
			ir_type=instr.ir_type,
		)
	elif isinstance(instr, IRCall):
		new_args = [_replace_temp_in_value(a, old_name, new_name) for a in instr.args]
		new_fv = _replace_temp_in_value(instr.func_value, old_name, new_name) if instr.func_value else instr.func_value
		return IRCall(
			dest=instr.dest,
			function_name=instr.function_name,
			args=new_args,
			arg_types=list(instr.arg_types),
			return_type=instr.return_type,
			indirect=instr.indirect,
			func_value=new_fv,
		)
	elif isinstance(instr, IRReturn):
		return IRReturn(
			value=_replace_temp_in_value(instr.value, old_name, new_name) if instr.value else instr.value,
			ir_type=instr.ir_type,
		)
	elif isinstance(instr, IRConvert):
		return IRConvert(
			dest=instr.dest,
			source=_replace_temp_in_value(instr.source, old_name, new_name),
			from_type=instr.from_type,
			to_type=instr.to_type,
		)
	elif isinstance(instr, IRAddrOf):
		new_src = instr.source
		if instr.source.name == old_name:
			new_src = IRTemp(new_name)
		return IRAddrOf(dest=instr.dest, source=new_src)
	elif isinstance(instr, IRVaStart):
		return IRVaStart(
			ap_addr=_replace_temp_in_value(instr.ap_addr, old_name, new_name),
			num_named_gp=instr.num_named_gp,
		)
	elif isinstance(instr, IRVaArg):
		return IRVaArg(
			dest=instr.dest,
			ap_addr=_replace_temp_in_value(instr.ap_addr, old_name, new_name),
			ir_type=instr.ir_type,
		)
	elif isinstance(instr, IRVaEnd):
		return IRVaEnd(
			ap_addr=_replace_temp_in_value(instr.ap_addr, old_name, new_name),
		)
	elif isinstance(instr, IRVaCopy):
		return IRVaCopy(
			dest_addr=_replace_temp_in_value(instr.dest_addr, old_name, new_name),
			src_addr=_replace_temp_in_value(instr.src_addr, old_name, new_name),
		)
	return instr


def _instr_uses_temp(instr: IRInstruction, name: str) -> bool:
	"""Check if an instruction uses (reads) a temp by name."""
	if isinstance(instr, IRBinOp):
		return (isinstance(instr.left, IRTemp) and instr.left.name == name) or \
			(isinstance(instr.right, IRTemp) and instr.right.name == name)
	elif isinstance(instr, IRUnaryOp):
		return isinstance(instr.operand, IRTemp) and instr.operand.name == name
	elif isinstance(instr, IRCopy):
		return isinstance(instr.source, IRTemp) and instr.source.name == name
	elif isinstance(instr, IRLoad):
		return isinstance(instr.address, IRTemp) and instr.address.name == name
	elif isinstance(instr, IRStore):
		return (isinstance(instr.address, IRTemp) and instr.address.name == name) or \
			(isinstance(instr.value, IRTemp) and instr.value.name == name)
	elif isinstance(instr, IRCall):
		return any(isinstance(a, IRTemp) and a.name == name for a in instr.args)
	elif isinstance(instr, IRReturn):
		return instr.value is not None and isinstance(instr.value, IRTemp) and instr.value.name == name
	elif isinstance(instr, IRConvert):
		return isinstance(instr.source, IRTemp) and instr.source.name == name
	elif isinstance(instr, IRAddrOf):
		return instr.source.name == name
	elif isinstance(instr, IRVaArg):
		return isinstance(instr.ap_addr, IRTemp) and instr.ap_addr.name == name
	elif isinstance(instr, IRVaStart):
		return isinstance(instr.ap_addr, IRTemp) and instr.ap_addr.name == name
	elif isinstance(instr, IRVaEnd):
		return isinstance(instr.ap_addr, IRTemp) and instr.ap_addr.name == name
	elif isinstance(instr, IRVaCopy):
		return (isinstance(instr.dest_addr, IRTemp) and instr.dest_addr.name == name) or \
			(isinstance(instr.src_addr, IRTemp) and instr.src_addr.name == name)
	return False


def _instr_defs_temp(instr: IRInstruction, name: str) -> bool:
	"""Check if an instruction defines (writes) a temp by name."""
	if isinstance(instr, (IRBinOp, IRUnaryOp, IRCopy, IRLoad, IRConvert, IRAddrOf, IRVaArg)):
		return instr.dest.name == name
	if isinstance(instr, IRCall) and instr.dest is not None:
		return instr.dest.name == name
	return False


def split_live_ranges(func: IRFunction) -> IRFunction:
	"""Split long live ranges to reduce register pressure.

	For temps with long live ranges that span many instructions but are only
	used in a few spots, insert spill/reload copies to break the range into
	shorter segments. This makes it easier for the graph colorer to find a
	valid K-coloring without actual stack spills.
	"""
	body = list(func.body)
	cfg = CFG(body)
	analyzer = LivenessAnalyzer(cfg)

	# Compute per-instruction pressure on the flat body
	# We work on the flat instruction list for simplicity
	pressure: list[int] = []
	block_map: list[str] = []
	flat_idx = 0
	block_instr_map: dict[int, tuple[str, int]] = {}
	for block in cfg.blocks():
		for i in range(len(block.instructions)):
			live = analyzer.get_live_at_point(block, i)
			pressure.append(len(live))
			block_map.append(block.label)
			block_instr_map[flat_idx] = (block.label, i)
			flat_idx += 1

	if not pressure:
		return func

	max_pressure = max(pressure)
	if max_pressure <= K:
		return func

	# Identify address-taken temps (never split these)
	addr_taken: set[str] = set()
	for instr in body:
		if isinstance(instr, IRAddrOf):
			addr_taken.add(instr.source.name)

	float_temps = _collect_float_temps(func)

	# Build flat instruction index for the body (including labels)
	# Map each instruction in body to its position
	def_positions: dict[str, list[int]] = {}
	use_positions: dict[str, list[int]] = {}
	for idx, instr in enumerate(body):
		if isinstance(instr, IRLabelInstr):
			continue
		for name in _get_temp_refs(instr):
			if _instr_defs_temp(instr, name):
				def_positions.setdefault(name, []).append(idx)
			if _instr_uses_temp(instr, name):
				use_positions.setdefault(name, []).append(idx)

	# Find candidates: temps with long ranges and sparse uses, not float/addr-taken
	all_temps = set(def_positions.keys()) | set(use_positions.keys())
	candidates: list[tuple[str, int, int]] = []  # (name, range_len, num_uses)
	for name in all_temps:
		if name in addr_taken or name in float_temps:
			continue
		defs = def_positions.get(name, [])
		uses = use_positions.get(name, [])
		if not defs or not uses:
			continue
		# Only split temps with a single definition (SSA-like)
		if len(defs) != 1:
			continue
		first = defs[0]
		last = max(uses)
		range_len = last - first
		if range_len < 4:
			continue
		num_uses = len(uses)
		# Only split if the density is low (few uses over long range)
		if num_uses > range_len // 2:
			continue
		candidates.append((name, range_len, num_uses))

	# Sort by range length descending (split longest ranges first)
	candidates.sort(key=lambda x: -x[1])

	# Limit splits to avoid excessive IR growth
	split_counter = 0
	max_splits = len(body) // 2

	# Track insertions: list of (position, instruction) to insert
	insertions: list[tuple[int, IRInstruction]] = []
	# Track use replacements: (position, old_name, new_name)
	replacements: list[tuple[int, str, str]] = []

	for name, range_len, num_uses in candidates:
		if split_counter >= max_splits:
			break

		defs = def_positions[name]
		uses = sorted(use_positions.get(name, []))
		def_pos = defs[0]

		if len(uses) < 2:
			continue

		# Find gaps between consecutive use points where we could split
		# A "gap" is a span between two consecutive uses that's long enough
		# and where pressure is high
		all_points = sorted([def_pos] + uses)

		for gap_idx in range(len(all_points) - 1):
			if split_counter >= max_splits:
				break

			start = all_points[gap_idx]
			end = all_points[gap_idx + 1]
			gap_len = end - start

			if gap_len < 3:
				continue

			# Check if pressure is high in this gap
			# Use the flat pressure array - but we need to map body indices
			# to pressure indices. Since labels can shift things, use a
			# simpler heuristic: check if any instruction in the gap has
			# high pressure by scanning the CFG-based pressure.
			high_pressure = False
			for block in cfg.blocks():
				for bi in range(len(block.instructions)):
					if bi < len(pressure) and pressure[bi] >= K:
						high_pressure = True
						break
				if high_pressure:
					break

			if not high_pressure:
				continue

			# Insert spill after `start` and reload before `end`
			spill_name = f"_split_{name}_{split_counter}"
			split_counter += 1

			# Spill: _split_X = name (copy right after start instruction)
			spill_instr = IRCopy(dest=IRTemp(spill_name), source=IRTemp(name))
			insertions.append((start + 1, spill_instr))

			# Reload: name_reload = _split_X (copy right before end instruction)
			reload_name = f"_reload_{name}_{split_counter}"
			reload_instr = IRCopy(dest=IRTemp(reload_name), source=IRTemp(spill_name))
			insertions.append((end, reload_instr))

			# Replace uses of `name` at position `end` with `reload_name`
			replacements.append((end, name, reload_name))

			# Only split the first qualifying gap per temp to keep things simple
			break

	if not insertions and not replacements:
		return func

	# Apply replacements first (before insertions shift indices)
	new_body = list(body)
	for pos, old_name, new_name in replacements:
		if 0 <= pos < len(new_body):
			new_body[pos] = _replace_temp_uses_in_instr(new_body[pos], old_name, new_name)

	# Sort insertions by position (descending) to maintain correct indices
	insertions.sort(key=lambda x: -x[0])
	for pos, instr in insertions:
		new_body.insert(pos, instr)

	return IRFunction(
		name=func.name,
		params=list(func.params),
		body=new_body,
		return_type=func.return_type,
		param_types=list(func.param_types),
		storage_class=func.storage_class,
		is_prototype=func.is_prototype,
		is_variadic=func.is_variadic,
	)


class RegisterAllocator:
	"""Chaitin-Briggs graph-coloring register allocator with move coalescing."""

	def __init__(self, func: IRFunction) -> None:
		self._func = func

	def allocate(self) -> dict[str, str]:
		"""Run register allocation, returning temp_name -> physical register mapping.

		Temps not in the returned mapping should use stack slots (spilled or float).
		"""
		# Apply live range splitting before allocation
		split_func = split_live_ranges(self._func)
		self._func = split_func

		cfg = CFG(self._func.body)
		analyzer = LivenessAnalyzer(cfg)
		interference = analyzer.interference_graph()

		if not interference:
			return {}

		# Collect temps whose address is taken -- they must stay on the stack
		addr_taken: set[str] = set()
		for instr in self._func.body:
			if isinstance(instr, IRAddrOf):
				addr_taken.add(instr.source.name)

		float_temps = _collect_float_temps(self._func)
		call_crossing = _find_call_crossing_temps(cfg, analyzer)
		move_edges = _collect_move_edges(self._func)
		use_counts = _count_temp_uses(self._func)

		# Compute loop-aware spill heuristic data
		loop_depths = cfg.loop_depth()
		weighted_uses = _compute_weighted_uses(cfg, loop_depths)
		live_range_lengths = _compute_live_range_lengths(cfg, analyzer)
		remat_temps = _find_rematerializable_temps(self._func)
		use_density = _compute_use_density(cfg, analyzer)
		use_distances = _compute_use_distances(cfg)

		# Build integer-only interference subgraph (exclude address-taken temps)
		int_temps = {t for t in interference if t not in float_temps and t not in addr_taken}
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
		coalesced_weighted_uses: dict[str, float] = {}
		coalesced_range_lengths: dict[str, int] = {}
		coalesced_remat: set[str] = set()
		coalesced_density: dict[str, float] = {}
		coalesced_use_distances: dict[str, int] = {}
		for t in int_temps:
			rep = coalesce_map.find(t)
			coalesced_use_counts[rep] = coalesced_use_counts.get(rep, 0) + use_counts.get(t, 1)
			coalesced_weighted_uses[rep] = coalesced_weighted_uses.get(rep, 0.0) + weighted_uses.get(t, 1.0)
			coalesced_range_lengths[rep] = max(
				coalesced_range_lengths.get(rep, 1), live_range_lengths.get(t, 1)
			)
			if t in remat_temps:
				coalesced_remat.add(rep)
			else:
				# If any member is not rematerializable, the group isn't
				coalesced_remat.discard(rep)
			coalesced_density[rep] = max(
				coalesced_density.get(rep, 0.0), use_density.get(t, 1.0)
			)
			# For use distances, take the minimum (shortest-lived member dominates)
			coalesced_use_distances[rep] = min(
				coalesced_use_distances.get(rep, 999999), use_distances.get(t, 999999)
			)

		# Color the coalesced graph
		coloring = self._color_graph(
			coalesced_graph, coalesced_call_crossing, coalesced_use_counts,
			coalesced_weighted_uses, coalesced_range_lengths, coalesced_remat,
			coalesced_density, coalesced_use_distances,
		)

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
		weighted_uses: dict[str, float] | None = None,
		range_lengths: dict[str, int] | None = None,
		remat_temps: set[str] | None = None,
		use_density: dict[str, float] | None = None,
		use_distances: dict[str, int] | None = None,
	) -> dict[str, str]:
		"""Chaitin-Briggs graph coloring with improved spill heuristic.

		Spill cost considers:
		- Loop-depth-weighted usage frequency (hot temps are expensive to spill)
		- Live range length (long ranges contribute more pressure)
		- Rematerialization (constants can be recomputed, nearly free to spill)
		- Use density (infrequent uses over long ranges are cheap to spill)
		- Use distance (short-lived temps are very expensive to spill)
		"""
		if not graph:
			return {}

		adj: dict[str, set[str]] = {n: set(neighbors) for n, neighbors in graph.items()}
		remaining: set[str] = set(adj.keys())
		stack: list[tuple[str, bool]] = []  # (node, is_potential_spill)

		_remat = remat_temps or set()
		_density = use_density or {}
		_use_dist = use_distances or {}

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
				# Spill heuristic: pick the node with lowest spill cost.
				# Rematerializable temps are preferred for spilling (cost ~0).
				# Short-lived temps (use distance <= 2) get a large multiplier
				# to avoid spilling them -- they don't benefit from spilling
				# since they'd need immediate reload.
				def spill_cost(n: str) -> float:
					degree = len(adj[n] & remaining)
					wu = (weighted_uses or {}).get(n, 0.0)
					if wu == 0.0:
						wu = float((use_counts or {}).get(n, 1))
					rl = (range_lengths or {}).get(n, 1)

					if n in _remat:
						return wu * 0.1 / max(degree, 1)

					# Factor in use density: low density = cheaper to spill
					density = _density.get(n, 1.0)
					base_cost = (wu * (1.0 + density)) / max(degree * rl, 1)

					# Single-use temporaries: temps referenced only once or
					# twice (def + one use) are cheap to spill because
					# only one reload is needed.  Discount their cost.
					uc = (use_counts or {}).get(n, 1)
					if uc <= 2:
						base_cost *= 0.5

					# Short-lived temporaries: defined and used within 1-2
					# instructions. Spilling these is wasteful since the
					# spill store + reload would cost more than the value's
					# entire lifetime. Multiply cost to strongly prefer
					# evicting longer-lived values instead.
					dist = _use_dist.get(n, 999999)
					if dist <= 2:
						base_cost *= 100.0
					elif dist <= 4:
						base_cost *= 10.0

					return base_cost

				spill_node = min(remaining, key=spill_cost)
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
