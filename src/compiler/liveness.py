"""Dataflow-based liveness analysis on the CFG."""

from __future__ import annotations

from compiler.cfg import BasicBlock, CFG
from compiler.ir import (
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

	return used


def _defined_temp(instr: IRInstruction) -> str | None:
	"""Return the temporary variable name defined (written) by an instruction, or None."""
	if isinstance(instr, (IRBinOp, IRUnaryOp, IRCopy, IRLoad, IRConvert, IRAlloc)):
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
