"""Control Flow Graph module for IR analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from compiler.ir import (
	IRCondJump,
	IRInstruction,
	IRJump,
	IRLabelInstr,
	IRReturn,
)


@dataclass
class NaturalLoop:
	"""A natural loop identified by back edges to a common header."""

	header: str
	body: set[str]
	back_edges: list[tuple[str, str]]  # list of (tail_label, header_label)


@dataclass
class BasicBlock:
	"""A basic block: a sequence of IR instructions with no internal branches.

	Execution enters at the top and exits at the bottom.
	"""

	label: str
	instructions: list[IRInstruction] = field(default_factory=list)
	_successors: list[BasicBlock] = field(default_factory=list, repr=False)
	_predecessors: list[BasicBlock] = field(default_factory=list, repr=False)

	@property
	def successors(self) -> list[BasicBlock]:
		return self._successors

	@property
	def predecessors(self) -> list[BasicBlock]:
		return self._predecessors

	def add_successor(self, block: BasicBlock) -> None:
		if block not in self._successors:
			self._successors.append(block)
		if self not in block._predecessors:
			block._predecessors.append(self)

	def is_empty(self) -> bool:
		return len(self.instructions) == 0

	def terminator(self) -> IRInstruction | None:
		"""Return the last instruction if it is a terminator (jump/condjump/return)."""
		if not self.instructions:
			return None
		last = self.instructions[-1]
		if isinstance(last, (IRJump, IRCondJump, IRReturn)):
			return last
		return None

	def __hash__(self) -> int:
		return hash(self.label)

	def __eq__(self, other: object) -> bool:
		if not isinstance(other, BasicBlock):
			return NotImplemented
		return self.label == other.label


class CFG:
	"""Control Flow Graph built from a list of IR instructions."""

	def __init__(self, instructions: list[IRInstruction]) -> None:
		self._blocks: dict[str, BasicBlock] = {}
		self._block_order: list[str] = []
		self._entry_label: str | None = None
		self._exit_labels: list[str] = []
		self._build(instructions)

	def _build(self, instructions: list[IRInstruction]) -> None:
		if not instructions:
			return

		# Phase 1: Split instructions into basic blocks.
		# A new block starts at: the very first instruction, after a jump/condjump/return,
		# or at a label instruction.
		raw_blocks: list[tuple[str, list[IRInstruction]]] = []
		current_label: str | None = None
		current_instrs: list[IRInstruction] = []
		label_counter = 0

		def _gen_label() -> str:
			nonlocal label_counter
			label_counter += 1
			return f"__bb{label_counter}"

		def _flush() -> None:
			nonlocal current_label, current_instrs
			if current_label is not None or current_instrs:
				lbl = current_label if current_label is not None else _gen_label()
				raw_blocks.append((lbl, current_instrs))
				current_label = None
				current_instrs = []

		for instr in instructions:
			if isinstance(instr, IRLabelInstr):
				# A label starts a new block.  Flush previous block first.
				if current_instrs or current_label is not None:
					_flush()
				current_label = instr.name
			else:
				if current_label is None and not current_instrs and not raw_blocks:
					# Very first instruction without a leading label.
					current_label = _gen_label()
				elif current_label is None and not current_instrs and raw_blocks:
					# Previous block ended with terminator; this is dead code or
					# an implicit new block.
					current_label = _gen_label()
				current_instrs.append(instr)
				if isinstance(instr, (IRJump, IRCondJump, IRReturn)):
					_flush()

		_flush()

		# Phase 2: Create BasicBlock objects.
		for lbl, instrs in raw_blocks:
			block = BasicBlock(label=lbl, instructions=instrs)
			self._blocks[lbl] = block
			self._block_order.append(lbl)

		if not self._block_order:
			return

		self._entry_label = self._block_order[0]

		# Phase 3: Connect edges.
		for i, lbl in enumerate(self._block_order):
			block = self._blocks[lbl]
			term = block.terminator()

			if isinstance(term, IRJump):
				target = self._blocks.get(term.target)
				if target is not None:
					block.add_successor(target)
			elif isinstance(term, IRCondJump):
				true_target = self._blocks.get(term.true_label)
				false_target = self._blocks.get(term.false_label)
				if true_target is not None:
					block.add_successor(true_target)
				if false_target is not None:
					block.add_successor(false_target)
			elif isinstance(term, IRReturn):
				self._exit_labels.append(lbl)
			else:
				# Fall-through to the next block.
				if i + 1 < len(self._block_order):
					next_block = self._blocks[self._block_order[i + 1]]
					block.add_successor(next_block)
				else:
					# Last block with no terminator is an implicit exit.
					self._exit_labels.append(lbl)

	# -----------------------------------------------------------------------
	# Public API
	# -----------------------------------------------------------------------

	def get_block(self, label: str) -> BasicBlock | None:
		"""Return the basic block with the given label, or None."""
		return self._blocks.get(label)

	def predecessors(self, block: BasicBlock) -> list[BasicBlock]:
		"""Return the predecessors of a block."""
		return block.predecessors

	def successors(self, block: BasicBlock) -> list[BasicBlock]:
		"""Return the successors of a block."""
		return block.successors

	def blocks(self) -> list[BasicBlock]:
		"""Return all basic blocks in order."""
		return [self._blocks[lbl] for lbl in self._block_order]

	@property
	def entry_block(self) -> BasicBlock | None:
		"""Return the entry block (first block)."""
		if self._entry_label is None:
			return None
		return self._blocks[self._entry_label]

	def exit_blocks(self) -> list[BasicBlock]:
		"""Return blocks that end with a return or have no successors."""
		return [self._blocks[lbl] for lbl in self._exit_labels]

	def all_labels(self) -> list[str]:
		"""Return all block labels in order."""
		return list(self._block_order)

	def reachable_blocks(self) -> set[BasicBlock]:
		"""Return the set of blocks reachable from the entry block."""
		if self.entry_block is None:
			return set()
		visited: set[BasicBlock] = set()
		worklist = [self.entry_block]
		while worklist:
			block = worklist.pop()
			if block in visited:
				continue
			visited.add(block)
			for succ in block.successors:
				if succ not in visited:
					worklist.append(succ)
		return visited

	def unreachable_blocks(self) -> list[BasicBlock]:
		"""Return blocks that are not reachable from the entry block."""
		reachable = self.reachable_blocks()
		return [b for b in self.blocks() if b not in reachable]

	# -----------------------------------------------------------------------
	# Dominator and loop analysis
	# -----------------------------------------------------------------------

	def _compute_postorder(self) -> list[BasicBlock]:
		"""Compute postorder traversal of reachable blocks."""
		if self.entry_block is None:
			return []
		visited: set[BasicBlock] = set()
		postorder: list[BasicBlock] = []
		stack: list[tuple[BasicBlock, bool]] = [(self.entry_block, False)]
		while stack:
			block, processed = stack.pop()
			if processed:
				postorder.append(block)
				continue
			if block in visited:
				continue
			visited.add(block)
			stack.append((block, True))
			for succ in reversed(block.successors):
				if succ not in visited:
					stack.append((succ, False))
		return postorder

	def _compute_idom(self) -> dict[BasicBlock, BasicBlock]:
		"""Compute immediate dominators using Cooper-Harvey-Kennedy algorithm.

		Returns dict where idom[entry] = entry (self-loop convention).
		Only includes reachable blocks.
		"""
		if self.entry_block is None:
			return {}

		postorder = self._compute_postorder()
		rpo = list(reversed(postorder))
		rpo_number: dict[BasicBlock, int] = {block: i for i, block in enumerate(rpo)}

		entry = self.entry_block
		idom: dict[BasicBlock, BasicBlock] = {entry: entry}

		def intersect(b1: BasicBlock, b2: BasicBlock) -> BasicBlock:
			finger1, finger2 = b1, b2
			while finger1 is not finger2:
				while rpo_number[finger1] > rpo_number[finger2]:
					finger1 = idom[finger1]
				while rpo_number[finger2] > rpo_number[finger1]:
					finger2 = idom[finger2]
			return finger1

		changed = True
		while changed:
			changed = False
			for block in rpo:
				if block is entry:
					continue
				processed_preds = [p for p in block.predecessors if p in idom]
				if not processed_preds:
					continue
				new_idom = processed_preds[0]
				for p in processed_preds[1:]:
					new_idom = intersect(new_idom, p)
				if idom.get(block) is not new_idom:
					idom[block] = new_idom
					changed = True

		return idom

	def compute_dominators(self) -> dict[str, str | None]:
		"""Compute immediate dominators using the iterative dataflow algorithm.

		Returns a dict mapping each reachable block label to its immediate
		dominator's label. The entry block maps to None.
		"""
		idom = self._compute_idom()
		result: dict[str, str | None] = {}
		entry = self.entry_block
		for block, dom in idom.items():
			if block is entry:
				result[block.label] = None
			else:
				result[block.label] = dom.label
		return result

	def compute_dominance_frontiers(self) -> dict[str, set[str]]:
		"""Compute dominance frontiers for all reachable blocks.

		Returns a dict mapping each block label to its dominance frontier
		(set of labels).
		"""
		idom = self._compute_idom()
		df: dict[str, set[str]] = {block.label: set() for block in idom}

		for block in idom:
			preds = [p for p in block.predecessors if p in idom]
			if len(preds) < 2:
				continue
			for p in preds:
				runner = p
				while runner is not idom[block]:
					df[runner.label].add(block.label)
					runner = idom[runner]

		return df

	def find_natural_loops(self) -> list[NaturalLoop]:
		"""Find natural loops by identifying back edges and computing loop bodies.

		A back edge is an edge n -> h where h dominates n. Loops with the
		same header are merged into a single NaturalLoop.
		"""
		idom = self._compute_idom()
		entry = self.entry_block
		if entry is None:
			return []

		def dominates(a: BasicBlock, b: BasicBlock) -> bool:
			"""Check if a dominates b by walking idom tree from b to root."""
			current = b
			while True:
				if current is a:
					return True
				parent = idom.get(current)
				if parent is None or parent is current:
					return False
				current = parent

		merged: dict[str, tuple[set[str], list[tuple[str, str]]]] = {}

		for block in idom:
			for succ in block.successors:
				if succ not in idom:
					continue
				if not dominates(succ, block):
					continue
				header = succ.label
				tail = block.label
				body: set[str] = {header}
				if tail != header:
					body.add(tail)
					worklist = [tail]
					while worklist:
						node_label = worklist.pop()
						node = self.get_block(node_label)
						if node is None:
							continue
						for pred in node.predecessors:
							if pred.label not in body and pred in idom:
								body.add(pred.label)
								worklist.append(pred.label)

				if header in merged:
					merged[header][0].update(body)
					merged[header][1].append((tail, header))
				else:
					merged[header] = (body, [(tail, header)])

		return [
			NaturalLoop(header=h, body=b, back_edges=edges)
			for h, (b, edges) in merged.items()
		]

	def loop_depth(self) -> dict[str, int]:
		"""Return nesting depth per block. 0 means not in any loop."""
		loops = self.find_natural_loops()
		depth: dict[str, int] = {label: 0 for label in self.all_labels()}
		for loop in loops:
			for label in loop.body:
				if label in depth:
					depth[label] += 1
		return depth
