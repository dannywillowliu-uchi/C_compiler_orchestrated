"""Edge-case tests for CFG construction and basic block splitting."""

from compiler.cfg import CFG, BasicBlock
from compiler.ir import (
	IRCondJump,
	IRConst,
	IRCopy,
	IRInstruction,
	IRJump,
	IRLabelInstr,
	IRReturn,
	IRTemp,
)


class TestEmptyCFG:
	"""CFG built from no instructions."""

	def test_empty_instructions(self):
		cfg = CFG([])
		assert cfg.entry_block is None
		assert cfg.blocks() == []
		assert cfg.all_labels() == []
		assert cfg.exit_blocks() == []

	def test_empty_reachable(self):
		cfg = CFG([])
		assert cfg.reachable_blocks() == set()
		assert cfg.unreachable_blocks() == []

	def test_empty_dominators(self):
		cfg = CFG([])
		assert cfg.compute_dominators() == {}

	def test_empty_dominance_frontiers(self):
		cfg = CFG([])
		assert cfg.compute_dominance_frontiers() == {}

	def test_empty_natural_loops(self):
		cfg = CFG([])
		assert cfg.find_natural_loops() == []

	def test_empty_loop_depth(self):
		cfg = CFG([])
		assert cfg.loop_depth() == {}


class TestSingleBlockFunctions:
	"""Functions with a single basic block."""

	def test_only_return(self):
		cfg = CFG([IRReturn()])
		assert len(cfg.blocks()) == 1
		assert len(cfg.exit_blocks()) == 1
		assert cfg.entry_block is cfg.exit_blocks()[0]

	def test_return_with_value(self):
		cfg = CFG([IRReturn(value=IRConst(42))])
		block = cfg.entry_block
		assert block is not None
		assert len(block.instructions) == 1
		assert block.successors == []
		assert block.predecessors == []

	def test_single_instruction_no_terminator(self):
		"""A single non-terminator instruction: implicit exit."""
		cfg = CFG([IRCopy(dest=IRTemp("t0"), source=IRConst(1))])
		assert len(cfg.blocks()) == 1
		assert len(cfg.exit_blocks()) == 1
		assert cfg.entry_block.terminator() is None

	def test_label_only_no_instructions(self):
		"""A label with no following instructions creates an empty block."""
		cfg = CFG([IRLabelInstr("L0")])
		assert len(cfg.blocks()) == 1
		block = cfg.get_block("L0")
		assert block is not None
		assert block.is_empty()
		assert len(cfg.exit_blocks()) == 1

	def test_single_block_dominators(self):
		cfg = CFG([IRReturn()])
		doms = cfg.compute_dominators()
		entry = cfg.entry_block
		assert doms[entry.label] is None

	def test_single_block_loop_depth_zero(self):
		cfg = CFG([IRReturn()])
		depths = cfg.loop_depth()
		assert all(d == 0 for d in depths.values())


class TestBlockSplitting:
	"""Test that blocks are correctly split at labels and terminators."""

	def test_label_starts_new_block(self):
		instrs = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(1)),
			IRLabelInstr("L1"),
			IRCopy(dest=IRTemp("t1"), source=IRConst(2)),
		]
		cfg = CFG(instrs)
		assert len(cfg.blocks()) == 2
		# First block falls through to L1
		b0 = cfg.blocks()[0]
		b1 = cfg.get_block("L1")
		assert b1 in b0.successors
		assert b0 in b1.predecessors

	def test_jump_ends_block(self):
		instrs = [
			IRLabelInstr("start"),
			IRJump(target="end"),
			IRLabelInstr("end"),
			IRReturn(),
		]
		cfg = CFG(instrs)
		start = cfg.get_block("start")
		end = cfg.get_block("end")
		assert end in start.successors
		assert start not in end.successors

	def test_condjump_creates_two_successors(self):
		instrs = [
			IRLabelInstr("entry"),
			IRCondJump(condition=IRTemp("c"), true_label="T", false_label="F"),
			IRLabelInstr("T"),
			IRReturn(value=IRConst(1)),
			IRLabelInstr("F"),
			IRReturn(value=IRConst(0)),
		]
		cfg = CFG(instrs)
		entry = cfg.get_block("entry")
		assert len(entry.successors) == 2
		succ_labels = {s.label for s in entry.successors}
		assert succ_labels == {"T", "F"}

	def test_return_ends_block_no_fallthrough(self):
		instrs = [
			IRLabelInstr("A"),
			IRReturn(),
			IRLabelInstr("B"),
			IRReturn(),
		]
		cfg = CFG(instrs)
		a = cfg.get_block("A")
		b = cfg.get_block("B")
		assert b not in a.successors
		assert a not in b.predecessors

	def test_consecutive_labels(self):
		"""Two labels in a row: each gets its own block."""
		instrs = [
			IRLabelInstr("L1"),
			IRLabelInstr("L2"),
			IRReturn(),
		]
		cfg = CFG(instrs)
		l1 = cfg.get_block("L1")
		l2 = cfg.get_block("L2")
		assert l1 is not None
		assert l2 is not None
		# L1 is empty and falls through to L2
		assert l1.is_empty()
		assert l2 in l1.successors

	def test_multiple_instructions_in_block(self):
		instrs = [
			IRLabelInstr("blk"),
			IRCopy(dest=IRTemp("t0"), source=IRConst(1)),
			IRCopy(dest=IRTemp("t1"), source=IRConst(2)),
			IRCopy(dest=IRTemp("t2"), source=IRConst(3)),
			IRReturn(),
		]
		cfg = CFG(instrs)
		blk = cfg.get_block("blk")
		assert len(blk.instructions) == 4  # 3 copies + return


class TestUnreachableBlocks:
	"""Blocks that cannot be reached from entry."""

	def test_unreachable_after_return(self):
		instrs = [
			IRLabelInstr("entry"),
			IRReturn(),
			IRLabelInstr("dead"),
			IRCopy(dest=IRTemp("t0"), source=IRConst(99)),
			IRReturn(),
		]
		cfg = CFG(instrs)
		unreachable = cfg.unreachable_blocks()
		labels = {b.label for b in unreachable}
		assert "dead" in labels

	def test_unreachable_after_unconditional_jump(self):
		instrs = [
			IRLabelInstr("entry"),
			IRJump(target="target"),
			IRLabelInstr("skipped"),
			IRReturn(),
			IRLabelInstr("target"),
			IRReturn(),
		]
		cfg = CFG(instrs)
		unreachable = cfg.unreachable_blocks()
		labels = {b.label for b in unreachable}
		assert "skipped" in labels
		assert "target" not in labels

	def test_reachable_excludes_unreachable(self):
		instrs = [
			IRLabelInstr("entry"),
			IRReturn(),
			IRLabelInstr("orphan"),
			IRReturn(),
		]
		cfg = CFG(instrs)
		reachable = cfg.reachable_blocks()
		reachable_labels = {b.label for b in reachable}
		assert "entry" in reachable_labels
		assert "orphan" not in reachable_labels


class TestGotoCrossingLoopBoundaries:
	"""Goto (IRJump) that jumps into or out of a loop body."""

	def test_goto_into_loop_body(self):
		"""Jump from outside into the middle of a loop."""
		instrs = [
			IRLabelInstr("entry"),
			IRJump(target="loop_body"),
			IRLabelInstr("loop_header"),
			IRCondJump(condition=IRTemp("c"), true_label="loop_body", false_label="exit"),
			IRLabelInstr("loop_body"),
			IRCopy(dest=IRTemp("t0"), source=IRConst(1)),
			IRJump(target="loop_header"),
			IRLabelInstr("exit"),
			IRReturn(),
		]
		cfg = CFG(instrs)
		loop_body = cfg.get_block("loop_body")
		# loop_body should have predecessors from both entry and loop_header
		pred_labels = {p.label for p in loop_body.predecessors}
		assert "entry" in pred_labels
		assert "loop_header" in pred_labels

	def test_goto_out_of_loop(self):
		"""Jump from inside loop to outside (like a break via goto)."""
		instrs = [
			IRLabelInstr("loop_top"),
			IRCondJump(condition=IRTemp("c"), true_label="body", false_label="done"),
			IRLabelInstr("body"),
			IRCondJump(condition=IRTemp("escape"), true_label="outside", false_label="continue"),
			IRLabelInstr("continue"),
			IRJump(target="loop_top"),
			IRLabelInstr("outside"),
			IRReturn(value=IRConst(99)),
			IRLabelInstr("done"),
			IRReturn(),
		]
		cfg = CFG(instrs)
		body = cfg.get_block("body")
		succ_labels = {s.label for s in body.successors}
		assert "outside" in succ_labels
		assert "continue" in succ_labels


class TestDeeplyNestedLoops:
	"""Multiple levels of loop nesting."""

	def _make_nested_loop_instrs(self, depth: int) -> list[IRInstruction]:
		"""Generate IR for `depth` nested while-like loops."""
		instrs: list[IRInstruction] = []
		for i in range(depth):
			instrs.append(IRLabelInstr(f"loop{i}_header"))
			instrs.append(
				IRCondJump(
					condition=IRTemp(f"c{i}"),
					true_label=f"loop{i}_body",
					false_label=f"loop{i}_exit",
				)
			)
			instrs.append(IRLabelInstr(f"loop{i}_body"))
		# Innermost body
		instrs.append(IRCopy(dest=IRTemp("inner"), source=IRConst(0)))
		# Close loops in reverse
		for i in range(depth - 1, -1, -1):
			instrs.append(IRJump(target=f"loop{i}_header"))
			instrs.append(IRLabelInstr(f"loop{i}_exit"))
		instrs.append(IRReturn())
		return instrs

	def test_two_nested_loops(self):
		cfg = CFG(self._make_nested_loop_instrs(2))
		loops = cfg.find_natural_loops()
		assert len(loops) >= 2

	def test_three_nested_loop_depths(self):
		cfg = CFG(self._make_nested_loop_instrs(3))
		depths = cfg.loop_depth()
		# Innermost body block should have depth 3
		# loop2_body contains the innermost copy + jump to loop2_header
		# Actually the copy is in loop2_body, so it's inside all 3 loops
		inner_depth = max(depths.values())
		assert inner_depth == 3

	def test_five_nested_loops(self):
		cfg = CFG(self._make_nested_loop_instrs(5))
		loops = cfg.find_natural_loops()
		assert len(loops) >= 5
		depths = cfg.loop_depth()
		assert max(depths.values()) == 5

	def test_nested_loop_dominators(self):
		cfg = CFG(self._make_nested_loop_instrs(2))
		doms = cfg.compute_dominators()
		# Inner loop header should be dominated by outer loop body (or header)
		assert doms.get("loop1_header") is not None


class TestSwitchInsideLoop:
	"""Switch-like IR pattern (multiple condjumps) inside a loop."""

	def test_switch_in_loop(self):
		instrs = [
			IRLabelInstr("loop_top"),
			IRCondJump(condition=IRTemp("done"), true_label="exit", false_label="switch_start"),
			IRLabelInstr("switch_start"),
			IRCondJump(condition=IRTemp("case1"), true_label="case1_body", false_label="check2"),
			IRLabelInstr("check2"),
			IRCondJump(condition=IRTemp("case2"), true_label="case2_body", false_label="default_body"),
			IRLabelInstr("case1_body"),
			IRCopy(dest=IRTemp("r"), source=IRConst(1)),
			IRJump(target="loop_top"),
			IRLabelInstr("case2_body"),
			IRCopy(dest=IRTemp("r"), source=IRConst(2)),
			IRJump(target="loop_top"),
			IRLabelInstr("default_body"),
			IRCopy(dest=IRTemp("r"), source=IRConst(0)),
			IRJump(target="loop_top"),
			IRLabelInstr("exit"),
			IRReturn(),
		]
		cfg = CFG(instrs)
		# All case bodies should jump back to loop_top
		for case_label in ["case1_body", "case2_body", "default_body"]:
			block = cfg.get_block(case_label)
			succ_labels = {s.label for s in block.successors}
			assert "loop_top" in succ_labels

		# Should detect a natural loop
		loops = cfg.find_natural_loops()
		assert len(loops) >= 1
		loop_headers = {lp.header for lp in loops}
		assert "loop_top" in loop_headers

	def test_switch_in_loop_multiple_back_edges(self):
		"""Loop with 3 back edges from different case arms."""
		instrs = [
			IRLabelInstr("header"),
			IRCondJump(condition=IRTemp("x"), true_label="c1", false_label="c2"),
			IRLabelInstr("c1"),
			IRJump(target="header"),
			IRLabelInstr("c2"),
			IRCondJump(condition=IRTemp("y"), true_label="c3", false_label="c4"),
			IRLabelInstr("c3"),
			IRJump(target="header"),
			IRLabelInstr("c4"),
			IRJump(target="header"),
		]
		cfg = CFG(instrs)
		loops = cfg.find_natural_loops()
		assert len(loops) == 1
		loop = loops[0]
		assert loop.header == "header"
		assert len(loop.back_edges) == 3


class TestManyLabels:
	"""Functions with many labels to stress block splitting."""

	def test_ten_sequential_labels(self):
		instrs = []
		for i in range(10):
			instrs.append(IRLabelInstr(f"L{i}"))
			instrs.append(IRCopy(dest=IRTemp(f"t{i}"), source=IRConst(i)))
		instrs.append(IRReturn())
		cfg = CFG(instrs)
		assert len(cfg.blocks()) == 10
		# Each block falls through to the next (except last which returns)
		for i in range(9):
			block = cfg.get_block(f"L{i}")
			succ_labels = {s.label for s in block.successors}
			assert f"L{i+1}" in succ_labels

	def test_fifty_labels_all_jumping_to_exit(self):
		instrs = []
		for i in range(50):
			instrs.append(IRLabelInstr(f"B{i}"))
			instrs.append(IRJump(target="exit"))
		instrs.append(IRLabelInstr("exit"))
		instrs.append(IRReturn())
		cfg = CFG(instrs)
		assert len(cfg.blocks()) == 51
		exit_block = cfg.get_block("exit")
		assert len(exit_block.predecessors) == 50

	def test_many_labels_reachability(self):
		"""Chain of 20 blocks: only entry is directly reachable via jumps."""
		instrs = [
			IRLabelInstr("entry"),
			IRJump(target="exit"),
		]
		for i in range(20):
			instrs.append(IRLabelInstr(f"dead{i}"))
			instrs.append(IRCopy(dest=IRTemp(f"t{i}"), source=IRConst(i)))
		instrs.append(IRLabelInstr("exit"))
		instrs.append(IRReturn())
		cfg = CFG(instrs)
		unreachable = cfg.unreachable_blocks()
		unreachable_labels = {b.label for b in unreachable}
		for i in range(20):
			assert f"dead{i}" in unreachable_labels


class TestEdgeConstruction:
	"""Verify edges are correctly constructed for various patterns."""

	def test_diamond_cfg(self):
		"""Classic if-then-else diamond shape."""
		instrs = [
			IRLabelInstr("entry"),
			IRCondJump(condition=IRTemp("c"), true_label="then", false_label="else"),
			IRLabelInstr("then"),
			IRCopy(dest=IRTemp("r"), source=IRConst(1)),
			IRJump(target="merge"),
			IRLabelInstr("else"),
			IRCopy(dest=IRTemp("r"), source=IRConst(0)),
			IRJump(target="merge"),
			IRLabelInstr("merge"),
			IRReturn(),
		]
		cfg = CFG(instrs)
		merge = cfg.get_block("merge")
		pred_labels = {p.label for p in merge.predecessors}
		assert pred_labels == {"then", "else"}
		assert len(cfg.exit_blocks()) == 1

	def test_self_loop(self):
		"""A block that jumps to itself (infinite loop)."""
		instrs = [
			IRLabelInstr("loop"),
			IRJump(target="loop"),
		]
		cfg = CFG(instrs)
		loop_block = cfg.get_block("loop")
		assert loop_block in loop_block.successors
		assert loop_block in loop_block.predecessors
		loops = cfg.find_natural_loops()
		assert len(loops) == 1
		assert loops[0].header == "loop"

	def test_no_duplicate_edges(self):
		"""Adding same successor twice should not create duplicates."""
		instrs = [
			IRLabelInstr("entry"),
			IRCondJump(condition=IRTemp("c"), true_label="target", false_label="target"),
			IRLabelInstr("target"),
			IRReturn(),
		]
		cfg = CFG(instrs)
		entry = cfg.get_block("entry")
		# Both branches go to same target, but should appear once in successors
		assert entry.successors.count(cfg.get_block("target")) == 1

	def test_jump_to_nonexistent_label(self):
		"""Jump to a label not in the instruction list: no edge created."""
		instrs = [
			IRLabelInstr("entry"),
			IRJump(target="nowhere"),
		]
		cfg = CFG(instrs)
		entry = cfg.get_block("entry")
		assert entry.successors == []

	def test_condjump_one_branch_missing(self):
		"""CondJump where one target doesn't exist."""
		instrs = [
			IRLabelInstr("entry"),
			IRCondJump(condition=IRTemp("c"), true_label="exists", false_label="missing"),
			IRLabelInstr("exists"),
			IRReturn(),
		]
		cfg = CFG(instrs)
		entry = cfg.get_block("entry")
		assert len(entry.successors) == 1
		assert entry.successors[0].label == "exists"

	def test_fallthrough_without_label(self):
		"""Instructions without an explicit label get auto-generated labels."""
		instrs = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(1)),
			IRCopy(dest=IRTemp("t1"), source=IRConst(2)),
			IRReturn(),
		]
		cfg = CFG(instrs)
		assert len(cfg.blocks()) == 1
		assert cfg.entry_block is not None


class TestDominanceAndFrontiers:
	"""Dominance and dominance frontier edge cases."""

	def test_linear_chain_dominators(self):
		"""A -> B -> C: A dominates B, B dominates C."""
		instrs = [
			IRLabelInstr("A"),
			IRJump(target="B"),
			IRLabelInstr("B"),
			IRJump(target="C"),
			IRLabelInstr("C"),
			IRReturn(),
		]
		cfg = CFG(instrs)
		doms = cfg.compute_dominators()
		assert doms["A"] is None
		assert doms["B"] == "A"
		assert doms["C"] == "B"

	def test_diamond_dominance_frontier(self):
		instrs = [
			IRLabelInstr("entry"),
			IRCondJump(condition=IRTemp("c"), true_label="L", false_label="R"),
			IRLabelInstr("L"),
			IRJump(target="merge"),
			IRLabelInstr("R"),
			IRJump(target="merge"),
			IRLabelInstr("merge"),
			IRReturn(),
		]
		cfg = CFG(instrs)
		df = cfg.compute_dominance_frontiers()
		# L and R have merge in their dominance frontier
		assert "merge" in df["L"]
		assert "merge" in df["R"]
		# entry and merge have empty frontiers
		assert df["entry"] == set()
		assert df["merge"] == set()

	def test_unreachable_not_in_dominators(self):
		instrs = [
			IRLabelInstr("entry"),
			IRReturn(),
			IRLabelInstr("dead"),
			IRReturn(),
		]
		cfg = CFG(instrs)
		doms = cfg.compute_dominators()
		assert "dead" not in doms


class TestBasicBlockProperties:
	"""Test BasicBlock dataclass methods directly."""

	def test_is_empty(self):
		block = BasicBlock(label="empty")
		assert block.is_empty()

	def test_not_empty(self):
		block = BasicBlock(label="full", instructions=[IRReturn()])
		assert not block.is_empty()

	def test_terminator_jump(self):
		block = BasicBlock(label="b", instructions=[
			IRCopy(dest=IRTemp("t"), source=IRConst(1)),
			IRJump(target="next"),
		])
		assert isinstance(block.terminator(), IRJump)

	def test_terminator_none_for_non_terminator(self):
		block = BasicBlock(label="b", instructions=[
			IRCopy(dest=IRTemp("t"), source=IRConst(1)),
		])
		assert block.terminator() is None

	def test_terminator_empty_block(self):
		block = BasicBlock(label="b")
		assert block.terminator() is None

	def test_block_equality(self):
		b1 = BasicBlock(label="same")
		b2 = BasicBlock(label="same")
		assert b1 == b2

	def test_block_inequality(self):
		b1 = BasicBlock(label="a")
		b2 = BasicBlock(label="b")
		assert b1 != b2

	def test_block_hash(self):
		b1 = BasicBlock(label="x")
		b2 = BasicBlock(label="x")
		assert hash(b1) == hash(b2)
		s = {b1, b2}
		assert len(s) == 1

	def test_add_successor_bidirectional(self):
		a = BasicBlock(label="a")
		b = BasicBlock(label="b")
		a.add_successor(b)
		assert b in a.successors
		assert a in b.predecessors

	def test_add_successor_no_duplicates(self):
		a = BasicBlock(label="a")
		b = BasicBlock(label="b")
		a.add_successor(b)
		a.add_successor(b)
		assert a.successors.count(b) == 1
		assert b.predecessors.count(a) == 1
