"""Tests for CFG dominator tree computation and natural loop detection."""

from compiler.cfg import CFG
from compiler.ir import (
	IRBinOp,
	IRCondJump,
	IRConst,
	IRJump,
	IRLabelInstr,
	IRReturn,
	IRTemp,
)


def _t(name: str) -> IRTemp:
	return IRTemp(name)


def _c(val: int) -> IRConst:
	return IRConst(val)


# ---------------------------------------------------------------------------
# Tests: compute_dominators
# ---------------------------------------------------------------------------


class TestComputeDominators:
	def test_single_block(self) -> None:
		instrs = [
			IRLabelInstr("entry"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		idom = cfg.compute_dominators()
		assert idom == {"entry": None}

	def test_linear_cfg(self) -> None:
		"""In a linear CFG, each block is dominated by the previous one."""
		instrs = [
			IRLabelInstr("entry"),
			IRJump(target="middle"),
			IRLabelInstr("middle"),
			IRJump(target="exit"),
			IRLabelInstr("exit"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		idom = cfg.compute_dominators()
		assert idom["entry"] is None
		assert idom["middle"] == "entry"
		assert idom["exit"] == "middle"

	def test_diamond_cfg(self) -> None:
		"""Diamond CFG: entry -> then/else -> merge. All idoms point to entry."""
		instrs = [
			IRLabelInstr("entry"),
			IRCondJump(condition=_t("c"), true_label="then", false_label="else"),
			IRLabelInstr("then"),
			IRJump(target="merge"),
			IRLabelInstr("else"),
			IRJump(target="merge"),
			IRLabelInstr("merge"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		idom = cfg.compute_dominators()
		assert idom["entry"] is None
		assert idom["then"] == "entry"
		assert idom["else"] == "entry"
		assert idom["merge"] == "entry"

	def test_while_loop(self) -> None:
		"""While loop: header dominated by entry, body by header, exit by header."""
		instrs = [
			IRLabelInstr("entry"),
			IRJump(target="header"),
			IRLabelInstr("header"),
			IRCondJump(condition=_t("c"), true_label="body", false_label="exit"),
			IRLabelInstr("body"),
			IRJump(target="header"),
			IRLabelInstr("exit"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		idom = cfg.compute_dominators()
		assert idom["entry"] is None
		assert idom["header"] == "entry"
		assert idom["body"] == "header"
		assert idom["exit"] == "header"

	def test_nested_if(self) -> None:
		"""Nested if produces nested dominator relationships."""
		instrs = [
			IRLabelInstr("entry"),
			IRCondJump(condition=_t("c1"), true_label="a", false_label="b"),
			IRLabelInstr("a"),
			IRCondJump(condition=_t("c2"), true_label="c", false_label="d"),
			IRLabelInstr("c"),
			IRJump(target="e"),
			IRLabelInstr("d"),
			IRJump(target="e"),
			IRLabelInstr("e"),
			IRJump(target="f"),
			IRLabelInstr("b"),
			IRJump(target="f"),
			IRLabelInstr("f"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		idom = cfg.compute_dominators()
		assert idom["entry"] is None
		assert idom["a"] == "entry"
		assert idom["b"] == "entry"
		assert idom["c"] == "a"
		assert idom["d"] == "a"
		assert idom["e"] == "a"
		assert idom["f"] == "entry"

	def test_empty_cfg(self) -> None:
		cfg = CFG([])
		assert cfg.compute_dominators() == {}

	def test_unreachable_blocks_excluded(self) -> None:
		"""Unreachable blocks are not included in the dominator map."""
		instrs = [
			IRLabelInstr("entry"),
			IRReturn(value=_c(0)),
			IRLabelInstr("dead"),
			IRReturn(value=_c(1)),
		]
		cfg = CFG(instrs)
		idom = cfg.compute_dominators()
		assert "entry" in idom
		assert "dead" not in idom

	def test_nested_loops(self) -> None:
		"""Nested loops: outer header dominates inner header."""
		instrs = [
			IRLabelInstr("entry"),
			IRJump(target="outer"),
			IRLabelInstr("outer"),
			IRCondJump(condition=_t("i"), true_label="inner", false_label="exit"),
			IRLabelInstr("inner"),
			IRCondJump(condition=_t("j"), true_label="inner_body", false_label="outer_latch"),
			IRLabelInstr("inner_body"),
			IRJump(target="inner"),
			IRLabelInstr("outer_latch"),
			IRJump(target="outer"),
			IRLabelInstr("exit"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		idom = cfg.compute_dominators()
		assert idom["entry"] is None
		assert idom["outer"] == "entry"
		assert idom["inner"] == "outer"
		assert idom["inner_body"] == "inner"
		assert idom["outer_latch"] == "inner"
		assert idom["exit"] == "outer"


# ---------------------------------------------------------------------------
# Tests: compute_dominance_frontiers
# ---------------------------------------------------------------------------


class TestComputeDominanceFrontiers:
	def test_linear_cfg(self) -> None:
		"""Linear CFG has empty dominance frontiers."""
		instrs = [
			IRLabelInstr("a"),
			IRJump(target="b"),
			IRLabelInstr("b"),
			IRJump(target="c"),
			IRLabelInstr("c"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		df = cfg.compute_dominance_frontiers()
		for label in ["a", "b", "c"]:
			assert df[label] == set()

	def test_diamond_cfg(self) -> None:
		"""In a diamond, merge is in DF of both then and else."""
		instrs = [
			IRLabelInstr("entry"),
			IRCondJump(condition=_t("c"), true_label="then", false_label="else"),
			IRLabelInstr("then"),
			IRJump(target="merge"),
			IRLabelInstr("else"),
			IRJump(target="merge"),
			IRLabelInstr("merge"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		df = cfg.compute_dominance_frontiers()
		assert df["entry"] == set()
		assert df["then"] == {"merge"}
		assert df["else"] == {"merge"}
		assert df["merge"] == set()

	def test_while_loop(self) -> None:
		"""In a while loop, header is in DF of body and itself."""
		instrs = [
			IRLabelInstr("entry"),
			IRJump(target="header"),
			IRLabelInstr("header"),
			IRCondJump(condition=_t("c"), true_label="body", false_label="exit"),
			IRLabelInstr("body"),
			IRJump(target="header"),
			IRLabelInstr("exit"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		df = cfg.compute_dominance_frontiers()
		assert "header" in df["body"]
		assert "header" in df["header"]
		assert "header" not in df["entry"]
		assert df["exit"] == set()

	def test_nested_if_frontiers(self) -> None:
		"""Nested if: inner merge has DF at outer merge."""
		instrs = [
			IRLabelInstr("entry"),
			IRCondJump(condition=_t("c1"), true_label="a", false_label="b"),
			IRLabelInstr("a"),
			IRCondJump(condition=_t("c2"), true_label="c", false_label="d"),
			IRLabelInstr("c"),
			IRJump(target="e"),
			IRLabelInstr("d"),
			IRJump(target="e"),
			IRLabelInstr("e"),
			IRJump(target="f"),
			IRLabelInstr("b"),
			IRJump(target="f"),
			IRLabelInstr("f"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		df = cfg.compute_dominance_frontiers()
		assert df["c"] == {"e"}
		assert df["d"] == {"e"}
		assert df["a"] == {"f"}
		assert df["b"] == {"f"}
		assert df["e"] == {"f"}


# ---------------------------------------------------------------------------
# Tests: find_natural_loops
# ---------------------------------------------------------------------------


class TestFindNaturalLoops:
	def test_no_loops(self) -> None:
		"""Diamond CFG has no natural loops."""
		instrs = [
			IRLabelInstr("entry"),
			IRCondJump(condition=_t("c"), true_label="a", false_label="b"),
			IRLabelInstr("a"),
			IRJump(target="exit"),
			IRLabelInstr("b"),
			IRJump(target="exit"),
			IRLabelInstr("exit"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		assert cfg.find_natural_loops() == []

	def test_simple_while_loop(self) -> None:
		instrs = [
			IRLabelInstr("entry"),
			IRJump(target="header"),
			IRLabelInstr("header"),
			IRCondJump(condition=_t("c"), true_label="body", false_label="exit"),
			IRLabelInstr("body"),
			IRJump(target="header"),
			IRLabelInstr("exit"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		loops = cfg.find_natural_loops()
		assert len(loops) == 1
		assert loops[0].header == "header"
		assert loops[0].body == {"header", "body"}
		assert ("body", "header") in loops[0].back_edges

	def test_self_loop(self) -> None:
		"""A block that jumps to itself forms a self-loop."""
		instrs = [
			IRLabelInstr("entry"),
			IRJump(target="loop"),
			IRLabelInstr("loop"),
			IRJump(target="loop"),
		]
		cfg = CFG(instrs)
		loops = cfg.find_natural_loops()
		assert len(loops) == 1
		assert loops[0].header == "loop"
		assert loops[0].body == {"loop"}

	def test_nested_loops(self) -> None:
		"""Nested loops produce two separate NaturalLoop entries."""
		instrs = [
			IRLabelInstr("entry"),
			IRJump(target="outer"),
			IRLabelInstr("outer"),
			IRCondJump(condition=_t("i"), true_label="inner", false_label="exit"),
			IRLabelInstr("inner"),
			IRCondJump(condition=_t("j"), true_label="inner_body", false_label="outer_latch"),
			IRLabelInstr("inner_body"),
			IRJump(target="inner"),
			IRLabelInstr("outer_latch"),
			IRJump(target="outer"),
			IRLabelInstr("exit"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		loops = cfg.find_natural_loops()
		assert len(loops) == 2
		headers = {loop.header for loop in loops}
		assert headers == {"outer", "inner"}

		inner_loop = next(lp for lp in loops if lp.header == "inner")
		assert inner_loop.body == {"inner", "inner_body"}

		outer_loop = next(lp for lp in loops if lp.header == "outer")
		assert "outer" in outer_loop.body
		assert "inner" in outer_loop.body
		assert "inner_body" in outer_loop.body
		assert "outer_latch" in outer_loop.body

	def test_irreducible_cfg_no_natural_loops(self) -> None:
		"""Irreducible control flow: A <-> B with neither dominating the other."""
		instrs = [
			IRLabelInstr("entry"),
			IRCondJump(condition=_t("c"), true_label="A", false_label="B"),
			IRLabelInstr("A"),
			IRJump(target="B"),
			IRLabelInstr("B"),
			IRCondJump(condition=_t("d"), true_label="A", false_label="exit"),
			IRLabelInstr("exit"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		loops = cfg.find_natural_loops()
		# Neither A nor B dominates the other, so no back edges exist
		assert len(loops) == 0

	def test_loop_with_multiple_latches(self) -> None:
		"""Multiple back edges to same header are merged into one loop."""
		instrs = [
			IRLabelInstr("entry"),
			IRJump(target="header"),
			IRLabelInstr("header"),
			IRCondJump(condition=_t("c"), true_label="body", false_label="exit"),
			IRLabelInstr("body"),
			IRCondJump(condition=_t("d"), true_label="latch1", false_label="latch2"),
			IRLabelInstr("latch1"),
			IRJump(target="header"),
			IRLabelInstr("latch2"),
			IRJump(target="header"),
			IRLabelInstr("exit"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		loops = cfg.find_natural_loops()
		assert len(loops) == 1
		assert loops[0].header == "header"
		assert loops[0].body == {"header", "body", "latch1", "latch2"}
		assert len(loops[0].back_edges) == 2

	def test_do_while_loop(self) -> None:
		"""Do-while: body executes before the condition check."""
		instrs = [
			IRLabelInstr("entry"),
			IRJump(target="body"),
			IRLabelInstr("body"),
			IRBinOp(dest=_t("x"), left=_t("x"), op="+", right=_c(1)),
			IRBinOp(dest=_t("cmp"), left=_t("x"), op="<", right=_c(10)),
			IRCondJump(condition=_t("cmp"), true_label="body", false_label="exit"),
			IRLabelInstr("exit"),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(instrs)
		loops = cfg.find_natural_loops()
		assert len(loops) == 1
		assert loops[0].header == "body"
		assert loops[0].body == {"body"}

	def test_empty_cfg(self) -> None:
		cfg = CFG([])
		assert cfg.find_natural_loops() == []


# ---------------------------------------------------------------------------
# Tests: loop_depth
# ---------------------------------------------------------------------------


class TestLoopDepth:
	def test_no_loops(self) -> None:
		instrs = [
			IRLabelInstr("entry"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		depth = cfg.loop_depth()
		assert depth["entry"] == 0

	def test_single_loop(self) -> None:
		instrs = [
			IRLabelInstr("entry"),
			IRJump(target="header"),
			IRLabelInstr("header"),
			IRCondJump(condition=_t("c"), true_label="body", false_label="exit"),
			IRLabelInstr("body"),
			IRJump(target="header"),
			IRLabelInstr("exit"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		depth = cfg.loop_depth()
		assert depth["entry"] == 0
		assert depth["header"] == 1
		assert depth["body"] == 1
		assert depth["exit"] == 0

	def test_nested_loops(self) -> None:
		"""Inner loop blocks have depth 2, outer-only blocks have depth 1."""
		instrs = [
			IRLabelInstr("entry"),
			IRJump(target="outer"),
			IRLabelInstr("outer"),
			IRCondJump(condition=_t("i"), true_label="inner", false_label="exit"),
			IRLabelInstr("inner"),
			IRCondJump(condition=_t("j"), true_label="inner_body", false_label="outer_latch"),
			IRLabelInstr("inner_body"),
			IRJump(target="inner"),
			IRLabelInstr("outer_latch"),
			IRJump(target="outer"),
			IRLabelInstr("exit"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		depth = cfg.loop_depth()
		assert depth["entry"] == 0
		assert depth["outer"] == 1
		assert depth["inner"] == 2
		assert depth["inner_body"] == 2
		assert depth["outer_latch"] == 1
		assert depth["exit"] == 0

	def test_empty_cfg(self) -> None:
		cfg = CFG([])
		assert cfg.loop_depth() == {}

	def test_irreducible_has_no_depth(self) -> None:
		"""Irreducible flow produces depth 0 for cycle members."""
		instrs = [
			IRLabelInstr("entry"),
			IRCondJump(condition=_t("c"), true_label="A", false_label="B"),
			IRLabelInstr("A"),
			IRJump(target="B"),
			IRLabelInstr("B"),
			IRCondJump(condition=_t("d"), true_label="A", false_label="exit"),
			IRLabelInstr("exit"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		depth = cfg.loop_depth()
		assert depth["A"] == 0
		assert depth["B"] == 0
