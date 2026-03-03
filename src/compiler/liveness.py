"""Liveness analysis pass on a Control Flow Graph."""

from __future__ import annotations

from src.compiler.cfg import BasicBlock, CFG
from src.compiler.ir import (
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
	IRValue,
)


def _temps_in(value: IRValue) -> set[str]:
	"""Extract temporary variable names from an IR value."""
	if isinstance(value, IRTemp):
		return {value.name}
	return set()


def instruction_def(instr: IRInstruction) -> set[str]:
	"""Return the set of variable names defined (written) by an instruction."""
	if isinstance(instr, (IRBinOp, IRUnaryOp, IRCopy, IRLoad, IRConvert, IRAlloc)):
		return {instr.dest.name}
	if isinstance(instr, IRCall) and instr.dest is not None:
		return {instr.dest.name}
	return set()


def instruction_use(instr: IRInstruction) -> set[str]:
	"""Return the set of variable names used (read) by an instruction."""
	if isinstance(instr, IRBinOp):
		return _temps_in(instr.left) | _temps_in(instr.right)
	if isinstance(instr, IRUnaryOp):
		return _temps_in(instr.operand)
	if isinstance(instr, IRCopy):
		return _temps_in(instr.source)
	if isinstance(instr, IRLoad):
		return _temps_in(instr.address)
	if isinstance(instr, IRStore):
		return _temps_in(instr.address) | _temps_in(instr.value)
	if isinstance(instr, IRCondJump):
		return _temps_in(instr.condition)
	if isinstance(instr, IRCall):
		result: set[str] = set()
		for arg in instr.args:
			result |= _temps_in(arg)
		return result
	if isinstance(instr, IRReturn) and instr.value is not None:
		return _temps_in(instr.value)
	if isinstance(instr, IRParam):
		return _temps_in(instr.value)
	if isinstance(instr, IRConvert):
		return _temps_in(instr.source)
	return set()


def _block_def_use(block: BasicBlock) -> tuple[set[str], set[str]]:
	"""Compute the def and use sets for a basic block.

	use(B) = variables read before being defined within B.
	def(B) = variables defined within B.
	"""
	def_set: set[str] = set()
	use_set: set[str] = set()
	for instr in block.instructions:
		# Variables read by this instruction that haven't been defined yet in this block
		# are "upward exposed" uses.
		use_set |= instruction_use(instr) - def_set
		def_set |= instruction_def(instr)
	return def_set, use_set


class LivenessAnalysis:
	"""Computes live variable information for a CFG using iterative backward dataflow."""

	def __init__(self, cfg: CFG) -> None:
		self._cfg = cfg
		self._def: dict[str, set[str]] = {}
		self._use: dict[str, set[str]] = {}
		self._live_in: dict[str, set[str]] = {}
		self._live_out: dict[str, set[str]] = {}

		self._compute()

	def _compute(self) -> None:
		blocks = self._cfg.blocks()
		if not blocks:
			return

		# Compute per-block def/use sets.
		for block in blocks:
			d, u = _block_def_use(block)
			self._def[block.label] = d
			self._use[block.label] = u
			self._live_in[block.label] = set()
			self._live_out[block.label] = set()

		# Iterative backward dataflow until fixed point.
		changed = True
		while changed:
			changed = False
			for block in reversed(blocks):
				lbl = block.label
				# live_out(B) = union of live_in(successors)
				new_out: set[str] = set()
				for succ in block.successors:
					new_out |= self._live_in[succ.label]

				# live_in(B) = use(B) | (live_out(B) - def(B))
				new_in = self._use[lbl] | (new_out - self._def[lbl])

				if new_in != self._live_in[lbl] or new_out != self._live_out[lbl]:
					changed = True
					self._live_in[lbl] = new_in
					self._live_out[lbl] = new_out

	def def_set(self, block: BasicBlock) -> set[str]:
		"""Variables defined (written) in the block."""
		return self._def.get(block.label, set())

	def use_set(self, block: BasicBlock) -> set[str]:
		"""Variables used (read before defined) in the block."""
		return self._use.get(block.label, set())

	def live_in(self, block: BasicBlock) -> set[str]:
		"""Variables live at block entry."""
		return self._live_in.get(block.label, set())

	def live_out(self, block: BasicBlock) -> set[str]:
		"""Variables live at block exit."""
		return self._live_out.get(block.label, set())

	def live_at_point(self, block: BasicBlock, instruction_index: int) -> set[str]:
		"""Return the set of variables live just *before* the given instruction.

		Computes backward from live_out of the block to the requested point.
		"""
		instrs = block.instructions
		if not instrs:
			return self.live_in(block).copy()
		if instruction_index < 0 or instruction_index >= len(instrs):
			raise IndexError(
				f"instruction_index {instruction_index} out of range for block "
				f"'{block.label}' with {len(instrs)} instructions"
			)

		# Start from live_out and walk backward to the instruction.
		live = self.live_out(block).copy()
		for i in range(len(instrs) - 1, instruction_index - 1, -1):
			instr = instrs[i]
			# Remove definitions, then add uses.
			live -= instruction_def(instr)
			live |= instruction_use(instr)
		return live
