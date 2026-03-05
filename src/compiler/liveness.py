"""Dataflow-based liveness analysis on the CFG."""

from __future__ import annotations

from compiler.cfg import BasicBlock, CFG
from compiler.ir import (
	IRAddrOf,
	IRAlloc,
	IRBinOp,
	IRCall,
	IRCondJump,
	IRConvert,
	IRCopy,
	IRInstruction,
	IRLoad,
	IRParam,
	IRReturn,
	IRStore,
	IRTemp,
	IRUnaryOp,
)


def _used_temps(instr: IRInstruction) -> set[str]:
	"""Return the set of temporary variable names used (read) by an instruction."""
	used: set[str] = set()

	if isinstance(instr, IRBinOp):
		if isinstance(instr.left, IRTemp):
			used.add(instr.left.name)
		if isinstance(instr.right, IRTemp):
			used.add(instr.right.name)
	elif isinstance(instr, IRUnaryOp):
		if isinstance(instr.operand, IRTemp):
			used.add(instr.operand.name)
	elif isinstance(instr, IRCopy):
		if isinstance(instr.source, IRTemp):
			used.add(instr.source.name)
	elif isinstance(instr, IRLoad):
		if isinstance(instr.address, IRTemp):
			used.add(instr.address.name)
	elif isinstance(instr, IRStore):
		if isinstance(instr.address, IRTemp):
			used.add(instr.address.name)
		if isinstance(instr.value, IRTemp):
			used.add(instr.value.name)
	elif isinstance(instr, IRCondJump):
		if isinstance(instr.condition, IRTemp):
			used.add(instr.condition.name)
	elif isinstance(instr, IRReturn):
		if instr.value is not None and isinstance(instr.value, IRTemp):
			used.add(instr.value.name)
	elif isinstance(instr, IRCall):
		for arg in instr.args:
			if isinstance(arg, IRTemp):
				used.add(arg.name)
	elif isinstance(instr, IRParam):
		if isinstance(instr.value, IRTemp):
			used.add(instr.value.name)
	elif isinstance(instr, IRConvert):
		if isinstance(instr.source, IRTemp):
			used.add(instr.source.name)
	elif isinstance(instr, IRAddrOf):
		used.add(instr.source.name)

	return used


def _defined_temp(instr: IRInstruction) -> str | None:
	"""Return the temporary variable name defined (written) by an instruction, or None."""
	if isinstance(instr, (IRAddrOf, IRBinOp, IRUnaryOp, IRCopy, IRLoad, IRConvert, IRAlloc)):
		return instr.dest.name
	if isinstance(instr, IRCall) and instr.dest is not None:
		return instr.dest.name
	return None


def _compute_gen_kill(block: BasicBlock) -> tuple[set[str], set[str]]:
	"""Compute gen and kill sets for a basic block.

	gen  = variables used before being defined in the block
	kill = variables defined anywhere in the block
	"""
	gen: set[str] = set()
	kill: set[str] = set()

	for instr in block.instructions:
		used = _used_temps(instr)
		# A use counts as gen only if the variable wasn't already killed in this block.
		gen |= used - kill
		defined = _defined_temp(instr)
		if defined is not None:
			kill.add(defined)

	return gen, kill


class LivenessAnalyzer:
	"""Computes liveness information on a CFG using iterative dataflow analysis."""

	def __init__(self, cfg: CFG) -> None:
		self._cfg = cfg
		self._live_in: dict[str, set[str]] = {}
		self._live_out: dict[str, set[str]] = {}
		self._gen: dict[str, set[str]] = {}
		self._kill: dict[str, set[str]] = {}
		self._computed = False

	def _ensure_computed(self) -> None:
		if not self._computed:
			self._run()
			self._computed = True

	def _run(self) -> None:
		"""Run iterative fixed-point liveness analysis."""
		blocks = self._cfg.blocks()
		if not blocks:
			return

		# Initialize gen/kill for every block.
		for block in blocks:
			gen, kill = _compute_gen_kill(block)
			self._gen[block.label] = gen
			self._kill[block.label] = kill
			self._live_in[block.label] = set()
			self._live_out[block.label] = set()

		# Iterate until fixed point. Process in reverse order for faster convergence.
		changed = True
		while changed:
			changed = False
			for block in reversed(blocks):
				lbl = block.label

				# live_out[B] = union of live_in[S] for all successors S
				new_out: set[str] = set()
				for succ in block.successors:
					new_out |= self._live_in[succ.label]

				# live_in[B] = gen[B] | (live_out[B] - kill[B])
				new_in = self._gen[lbl] | (new_out - self._kill[lbl])

				if new_in != self._live_in[lbl] or new_out != self._live_out[lbl]:
					changed = True
					self._live_in[lbl] = new_in
					self._live_out[lbl] = new_out

	def compute_liveness(self) -> dict[str, tuple[set[str], set[str]]]:
		"""Compute liveness and return a dict mapping block label to (live_in, live_out)."""
		self._ensure_computed()
		result: dict[str, tuple[set[str], set[str]]] = {}
		for block in self._cfg.blocks():
			lbl = block.label
			result[lbl] = (set(self._live_in[lbl]), set(self._live_out[lbl]))
		return result

	def get_live_at_point(self, block: BasicBlock, instruction_index: int) -> set[str]:
		"""Return the set of live variables just before the given instruction index.

		Index 0 means live-in of the block. An index equal to len(instructions)
		means the live-out of the block.
		"""
		self._ensure_computed()
		instrs = block.instructions

		if instruction_index < 0 or instruction_index > len(instrs):
			raise IndexError(f"instruction_index {instruction_index} out of range for block '{block.label}'")

		# Start from live_out and walk backwards to the requested point.
		live = set(self._live_out[block.label])

		for i in range(len(instrs) - 1, instruction_index - 1, -1):
			if i < instruction_index:
				break
			instr = instrs[i]
			# Remove defined variable (it's not live before this point unless also used).
			defined = _defined_temp(instr)
			if defined is not None:
				live.discard(defined)
			# Add used variables (they must be live before this instruction).
			live |= _used_temps(instr)

		return live

	def find_critical_edges(self) -> list[tuple[str, str]]:
		"""Find critical edges in the CFG.

		A critical edge goes from a block with multiple successors to a
		block with multiple predecessors. These edges complicate liveness
		because inserting code (e.g. for phi resolution) requires splitting.
		"""
		edges: list[tuple[str, str]] = []
		for block in self._cfg.blocks():
			if len(block.successors) > 1:
				for succ in block.successors:
					if len(succ.predecessors) > 1:
						edges.append((block.label, succ.label))
		return edges

	def variables_live_across_calls(self) -> dict[str, set[str]]:
		"""Find variables that are live across function call instructions.

		Returns a dict mapping block labels to the set of variable names
		that are live across at least one IRCall in that block. These
		variables would need caller-saved registers or spilling.
		"""
		self._ensure_computed()
		result: dict[str, set[str]] = {}

		for block in self._cfg.blocks():
			across_calls: set[str] = set()
			instrs = block.instructions
			live = set(self._live_out[block.label])

			for instr in reversed(instrs):
				defined = _defined_temp(instr)
				if defined is not None:
					live.discard(defined)
				live |= _used_temps(instr)

				if isinstance(instr, IRCall):
					# Variables live *after* the call (minus the call's dest)
					# are the ones that must survive the call.
					post_call_live = set(live)
					# The call args are consumed by the call, not live across it.
					for arg in instr.args:
						if isinstance(arg, IRTemp):
							post_call_live.discard(arg.name)
					# The dest is defined by the call, not live across it.
					if instr.dest is not None:
						post_call_live.discard(instr.dest.name)
					across_calls |= post_call_live

			if across_calls:
				result[block.label] = across_calls

		return result

	def merge_point_live_ins(self) -> dict[str, dict[str, list[str]]]:
		"""Analyze variables live-in at merge points (blocks with >1 predecessor).

		Returns a dict mapping merge block labels to a dict of variable
		name -> list of predecessor labels that contribute that variable
		as live-out. This identifies phi-like merge requirements.
		"""
		self._ensure_computed()
		result: dict[str, dict[str, list[str]]] = {}

		for block in self._cfg.blocks():
			if len(block.predecessors) <= 1:
				continue
			live_in = self._live_in[block.label]
			if not live_in:
				continue
			var_sources: dict[str, list[str]] = {}
			for var in live_in:
				sources: list[str] = []
				for pred in block.predecessors:
					if var in self._live_out[pred.label]:
						sources.append(pred.label)
				var_sources[var] = sources
			result[block.label] = var_sources

		return result

	def def_use_chains(self) -> dict[str, tuple[list[tuple[str, int]], list[tuple[str, int]]]]:
		"""Compute def-use chains for each temporary variable.

		Returns a dict mapping temp name to (defs, uses) where each is a list
		of (block_label, instruction_index) pairs indicating where the temp
		is defined and used respectively.
		"""
		defs: dict[str, list[tuple[str, int]]] = {}
		uses: dict[str, list[tuple[str, int]]] = {}
		for block in self._cfg.blocks():
			for i, instr in enumerate(block.instructions):
				defined = _defined_temp(instr)
				if defined is not None:
					defs.setdefault(defined, []).append((block.label, i))
				for name in _used_temps(instr):
					uses.setdefault(name, []).append((block.label, i))
		result: dict[str, tuple[list[tuple[str, int]], list[tuple[str, int]]]] = {}
		all_temps = set(defs.keys()) | set(uses.keys())
		for t in all_temps:
			result[t] = (defs.get(t, []), uses.get(t, []))
		return result

	def interference_graph(self) -> dict[str, set[str]]:
		"""Build an interference graph: maps each variable to its set of interfering variables.

		Two variables interfere if they are simultaneously live at some program point.
		"""
		self._ensure_computed()
		graph: dict[str, set[str]] = {}

		def _add_edge(a: str, b: str) -> None:
			if a == b:
				return
			graph.setdefault(a, set()).add(b)
			graph.setdefault(b, set()).add(a)

		for block in self._cfg.blocks():
			instrs = block.instructions
			live = set(self._live_out[block.label])

			# Ensure all variables seen are in the graph.
			for var in live:
				graph.setdefault(var, set())

			# Walk instructions backwards, updating live set and adding edges.
			for instr in reversed(instrs):
				defined = _defined_temp(instr)
				if defined is not None:
					graph.setdefault(defined, set())
					# The defined variable interferes with everything currently live
					# (except itself). For copy instructions, the source and dest
					# don't interfere (standard move coalescing optimization).
					for var in live:
						if isinstance(instr, IRCopy) and isinstance(instr.source, IRTemp) and var == instr.source.name:
							continue
						_add_edge(defined, var)
					# Remove defined from live set.
					live.discard(defined)

				# Add used variables to live set.
				used = _used_temps(instr)
				live |= used
				for var in used:
					graph.setdefault(var, set())

		return graph
