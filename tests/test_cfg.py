"""Tests for the Control Flow Graph module."""

from compiler.cfg import BasicBlock, CFG
from compiler.ir import (
	IRBinOp,
	IRCondJump,
	IRConst,
	IRCopy,
	IRJump,
	IRLabelInstr,
	IRReturn,
	IRTemp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t(name: str) -> IRTemp:
	return IRTemp(name)


def _c(val: int) -> IRConst:
	return IRConst(val)


# ---------------------------------------------------------------------------
# Tests: linear code
# ---------------------------------------------------------------------------

class TestLinearCode:
	def test_single_block_from_linear_code(self) -> None:
		"""Linear code with no branches produces a single basic block."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRBinOp(dest=_t("y"), left=_t("x"), op="+", right=_c(2)),
			IRReturn(value=_t("y")),
		]
		cfg = CFG(instrs)
		assert len(cfg.blocks()) == 1
		assert cfg.entry_block is not None
		assert cfg.entry_block.label == "entry"
		assert len(cfg.entry_block.instructions) == 3  # excludes the label instr
		assert cfg.entry_block.successors == []
		assert cfg.entry_block.predecessors == []

	def test_linear_code_without_label(self) -> None:
		"""Linear code without a leading label still forms a single block."""
		instrs = [
			IRCopy(dest=_t("a"), source=_c(42)),
			IRReturn(value=_t("a")),
		]
		cfg = CFG(instrs)
		assert len(cfg.blocks()) == 1
		assert cfg.entry_block is not None

	def test_all_labels(self) -> None:
		"""all_labels() returns labels in order."""
		instrs = [
			IRLabelInstr("start"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		assert cfg.all_labels() == ["start"]


# ---------------------------------------------------------------------------
# Tests: conditional branches
# ---------------------------------------------------------------------------

class TestConditionalBranch:
	def test_if_else_creates_diamond(self) -> None:
		"""if-else pattern creates a diamond CFG: entry -> then/else -> merge."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("cond"), source=_c(1)),
			IRCondJump(condition=_t("cond"), true_label="then", false_label="else"),
			IRLabelInstr("then"),
			IRCopy(dest=_t("x"), source=_c(10)),
			IRJump(target="merge"),
			IRLabelInstr("else"),
			IRCopy(dest=_t("x"), source=_c(20)),
			IRJump(target="merge"),
			IRLabelInstr("merge"),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(instrs)

		assert len(cfg.blocks()) == 4
		assert cfg.all_labels() == ["entry", "then", "else", "merge"]

		entry = cfg.get_block("entry")
		then_b = cfg.get_block("then")
		else_b = cfg.get_block("else")
		merge = cfg.get_block("merge")

		assert entry is not None
		assert then_b is not None
		assert else_b is not None
		assert merge is not None

		# entry -> then, else
		assert then_b in cfg.successors(entry)
		assert else_b in cfg.successors(entry)

		# then -> merge, else -> merge
		assert merge in cfg.successors(then_b)
		assert merge in cfg.successors(else_b)

		# merge predecessors
		assert then_b in cfg.predecessors(merge)
		assert else_b in cfg.predecessors(merge)

		# entry is the entry block
		assert cfg.entry_block is entry

	def test_condjump_successors_count(self) -> None:
		"""A conditional jump block should have exactly 2 successors."""
		instrs = [
			IRLabelInstr("entry"),
			IRCondJump(condition=_t("x"), true_label="a", false_label="b"),
			IRLabelInstr("a"),
			IRReturn(value=_c(1)),
			IRLabelInstr("b"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		entry = cfg.get_block("entry")
		assert entry is not None
		assert len(cfg.successors(entry)) == 2


# ---------------------------------------------------------------------------
# Tests: loops and back-edges
# ---------------------------------------------------------------------------

class TestLoops:
	def test_while_loop_back_edge(self) -> None:
		"""A while loop creates a back-edge from the loop body to the header."""
		# while (x < 10) { x = x + 1; }
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(0)),
			IRJump(target="loop_header"),
			IRLabelInstr("loop_header"),
			IRBinOp(dest=_t("cmp"), left=_t("x"), op="<", right=_c(10)),
			IRCondJump(condition=_t("cmp"), true_label="loop_body", false_label="loop_exit"),
			IRLabelInstr("loop_body"),
			IRBinOp(dest=_t("x"), left=_t("x"), op="+", right=_c(1)),
			IRJump(target="loop_header"),
			IRLabelInstr("loop_exit"),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(instrs)

		assert len(cfg.blocks()) == 4
		header = cfg.get_block("loop_header")
		body = cfg.get_block("loop_body")
		exit_b = cfg.get_block("loop_exit")

		assert header is not None
		assert body is not None
		assert exit_b is not None

		# back-edge: body -> header
		assert header in cfg.successors(body)
		# header -> body (true), header -> exit (false)
		assert body in cfg.successors(header)
		assert exit_b in cfg.successors(header)
		# header has predecessors: entry and body
		assert len(cfg.predecessors(header)) == 2

	def test_nested_loop(self) -> None:
		"""Nested loops create multiple back-edges."""
		instrs = [
			IRLabelInstr("entry"),
			IRJump(target="outer_header"),
			IRLabelInstr("outer_header"),
			IRCondJump(condition=_t("i"), true_label="inner_header", false_label="exit"),
			IRLabelInstr("inner_header"),
			IRCondJump(condition=_t("j"), true_label="inner_body", false_label="outer_latch"),
			IRLabelInstr("inner_body"),
			IRBinOp(dest=_t("j"), left=_t("j"), op="+", right=_c(1)),
			IRJump(target="inner_header"),
			IRLabelInstr("outer_latch"),
			IRBinOp(dest=_t("i"), left=_t("i"), op="+", right=_c(1)),
			IRJump(target="outer_header"),
			IRLabelInstr("exit"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)

		assert len(cfg.blocks()) == 6
		# inner back-edge
		inner_body = cfg.get_block("inner_body")
		inner_header = cfg.get_block("inner_header")
		assert inner_body is not None
		assert inner_header is not None
		assert inner_header in cfg.successors(inner_body)

		# outer back-edge
		outer_latch = cfg.get_block("outer_latch")
		outer_header = cfg.get_block("outer_header")
		assert outer_latch is not None
		assert outer_header is not None
		assert outer_header in cfg.successors(outer_latch)


# ---------------------------------------------------------------------------
# Tests: unreachable blocks
# ---------------------------------------------------------------------------

class TestUnreachable:
	def test_unreachable_block_detected(self) -> None:
		"""A block that no jump targets should be detected as unreachable."""
		instrs = [
			IRLabelInstr("entry"),
			IRReturn(value=_c(0)),
			IRLabelInstr("dead"),
			IRCopy(dest=_t("x"), source=_c(99)),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(instrs)

		assert len(cfg.blocks()) == 2
		unreachable = cfg.unreachable_blocks()
		assert len(unreachable) == 1
		assert unreachable[0].label == "dead"

	def test_all_reachable(self) -> None:
		"""When all blocks are connected, unreachable_blocks returns empty."""
		instrs = [
			IRLabelInstr("entry"),
			IRJump(target="next"),
			IRLabelInstr("next"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		assert cfg.unreachable_blocks() == []

	def test_reachable_blocks_subset(self) -> None:
		"""reachable_blocks returns only blocks reachable from entry."""
		instrs = [
			IRLabelInstr("a"),
			IRJump(target="b"),
			IRLabelInstr("b"),
			IRReturn(value=_c(0)),
			IRLabelInstr("c"),
			IRReturn(value=_c(1)),
		]
		cfg = CFG(instrs)
		reachable = cfg.reachable_blocks()
		labels = {b.label for b in reachable}
		assert labels == {"a", "b"}


# ---------------------------------------------------------------------------
# Tests: empty blocks and edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
	def test_empty_instructions(self) -> None:
		"""Empty instruction list produces an empty CFG."""
		cfg = CFG([])
		assert cfg.blocks() == []
		assert cfg.entry_block is None
		assert cfg.all_labels() == []

	def test_empty_block_with_jump(self) -> None:
		"""A label followed immediately by a jump creates a block with one instruction."""
		instrs = [
			IRLabelInstr("entry"),
			IRJump(target="target"),
			IRLabelInstr("target"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		entry = cfg.get_block("entry")
		assert entry is not None
		assert len(entry.instructions) == 1  # just the jump

	def test_consecutive_labels(self) -> None:
		"""Consecutive labels each create their own block."""
		instrs = [
			IRLabelInstr("a"),
			IRLabelInstr("b"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		# "a" is empty (no instructions before "b" label) with fall-through to "b"
		a = cfg.get_block("a")
		b = cfg.get_block("b")
		assert a is not None
		assert b is not None
		assert b in cfg.successors(a)

	def test_get_block_nonexistent(self) -> None:
		"""get_block returns None for a label that doesn't exist."""
		cfg = CFG([IRLabelInstr("x"), IRReturn(value=_c(0))])
		assert cfg.get_block("nonexistent") is None

	def test_exit_blocks(self) -> None:
		"""exit_blocks returns blocks ending with return."""
		instrs = [
			IRLabelInstr("entry"),
			IRCondJump(condition=_t("c"), true_label="a", false_label="b"),
			IRLabelInstr("a"),
			IRReturn(value=_c(1)),
			IRLabelInstr("b"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		exits = cfg.exit_blocks()
		labels = {b.label for b in exits}
		assert labels == {"a", "b"}

	def test_fall_through_without_terminator(self) -> None:
		"""A block without a terminator falls through to the next block."""
		instrs = [
			IRLabelInstr("first"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRLabelInstr("second"),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(instrs)
		first = cfg.get_block("first")
		second = cfg.get_block("second")
		assert first is not None
		assert second is not None
		assert second in cfg.successors(first)
		assert first in cfg.predecessors(second)


# ---------------------------------------------------------------------------
# Tests: BasicBlock methods
# ---------------------------------------------------------------------------

class TestBasicBlock:
	def test_is_empty(self) -> None:
		"""is_empty returns True for blocks with no instructions."""
		b = BasicBlock(label="empty")
		assert b.is_empty()
		b.instructions.append(IRReturn())
		assert not b.is_empty()

	def test_terminator_returns_jump(self) -> None:
		"""terminator() returns the last instruction if it is a jump."""
		b = BasicBlock(label="test", instructions=[
			IRCopy(dest=_t("x"), source=_c(1)),
			IRJump(target="next"),
		])
		term = b.terminator()
		assert isinstance(term, IRJump)
		assert term.target == "next"

	def test_terminator_returns_none_for_non_terminator(self) -> None:
		"""terminator() returns None if the last instruction is not a terminator."""
		b = BasicBlock(label="test", instructions=[
			IRCopy(dest=_t("x"), source=_c(1)),
		])
		assert b.terminator() is None

	def test_terminator_empty_block(self) -> None:
		"""terminator() returns None for empty blocks."""
		b = BasicBlock(label="empty")
		assert b.terminator() is None

	def test_block_equality(self) -> None:
		"""Two blocks with the same label are equal."""
		a = BasicBlock(label="same")
		b = BasicBlock(label="same")
		assert a == b
		assert hash(a) == hash(b)

	def test_block_inequality(self) -> None:
		"""Two blocks with different labels are not equal."""
		a = BasicBlock(label="one")
		b = BasicBlock(label="two")
		assert a != b
