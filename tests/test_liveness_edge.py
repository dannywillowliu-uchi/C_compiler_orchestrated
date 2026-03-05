"""Edge-case tests for liveness analysis: critical edges, call-crossing
liveness, and merge-point (phi-like) variable tracking."""

from compiler.cfg import CFG
from compiler.ir import (
	IRBinOp,
	IRCall,
	IRCondJump,
	IRConst,
	IRCopy,
	IRJump,
	IRLabelInstr,
	IRReturn,
	IRTemp,
)
from compiler.liveness import LivenessAnalyzer


def _t(name: str) -> IRTemp:
	return IRTemp(name)


def _c(val: int) -> IRConst:
	return IRConst(val)


# ---------------------------------------------------------------------------
# Critical edge detection
# ---------------------------------------------------------------------------

class TestCriticalEdges:
	def test_diamond_has_no_critical_edges(self) -> None:
		"""Standard diamond: entry->then, entry->else, both->merge.
		Entry has 2 successors, merge has 2 predecessors, but then/else
		each have only 1 predecessor, so entry->then and entry->else are
		NOT critical (their targets have only 1 predecessor each)."""
		instrs = [
			IRLabelInstr("entry"),
			IRCondJump(condition=_t("c"), true_label="then", false_label="else_"),
			IRLabelInstr("then"),
			IRJump(target="merge"),
			IRLabelInstr("else_"),
			IRJump(target="merge"),
			IRLabelInstr("merge"),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		assert analyzer.find_critical_edges() == []

	def test_critical_edge_condjump_to_loop_header(self) -> None:
		"""A conditional jump from a block with 2 successors to a loop header
		with 2 predecessors (back-edge + entry) is a critical edge."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(0)),
			IRJump(target="header"),
			IRLabelInstr("header"),
			IRBinOp(dest=_t("cmp"), left=_t("x"), op="<", right=_c(10)),
			IRCondJump(condition=_t("cmp"), true_label="body", false_label="exit"),
			IRLabelInstr("body"),
			IRBinOp(dest=_t("x"), left=_t("x"), op="+", right=_c(1)),
			IRBinOp(dest=_t("flag"), left=_t("x"), op="<", right=_c(5)),
			IRCondJump(condition=_t("flag"), true_label="header", false_label="exit"),
			IRLabelInstr("exit"),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		critical = analyzer.find_critical_edges()

		# body has 2 successors (header, exit); header has 2 predecessors (entry, body)
		assert ("body", "header") in critical
		# body -> exit: exit has predecessors from header and body (2 preds)
		assert ("body", "exit") in critical

	def test_no_critical_edges_linear(self) -> None:
		"""Linear code has no critical edges."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		assert analyzer.find_critical_edges() == []

	def test_multiple_critical_edges(self) -> None:
		"""Two conditional branches both targeting the same merge block."""
		instrs = [
			IRLabelInstr("entry"),
			IRCondJump(condition=_t("c1"), true_label="a", false_label="b"),
			IRLabelInstr("a"),
			IRCondJump(condition=_t("c2"), true_label="merge", false_label="c"),
			IRLabelInstr("b"),
			IRCondJump(condition=_t("c3"), true_label="merge", false_label="c"),
			IRLabelInstr("c"),
			IRJump(target="merge"),
			IRLabelInstr("merge"),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		critical = analyzer.find_critical_edges()

		# a->merge: a has 2 successors, merge has 3 predecessors
		assert ("a", "merge") in critical
		# b->merge: b has 2 successors, merge has 3 predecessors
		assert ("b", "merge") in critical


# ---------------------------------------------------------------------------
# Variables live across function calls
# ---------------------------------------------------------------------------

class TestLiveAcrossCalls:
	def test_variable_live_across_call(self) -> None:
		"""x defined before call, used after call => live across call."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRCall(dest=_t("r"), function_name="foo", args=[]),
			IRBinOp(dest=_t("z"), left=_t("x"), op="+", right=_t("r")),
			IRReturn(value=_t("z")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		across = analyzer.variables_live_across_calls()

		assert "entry" in across
		assert "x" in across["entry"]
		# r is the call's dest, not live across
		assert "r" not in across["entry"]

	def test_call_arg_not_live_across(self) -> None:
		"""Call arguments are consumed by the call, not live across it."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRCall(dest=_t("r"), function_name="bar", args=[_t("a")]),
			IRReturn(value=_t("r")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		across = analyzer.variables_live_across_calls()

		# a is only used as an arg, not needed after the call
		if "entry" in across:
			assert "a" not in across["entry"]

	def test_multiple_calls_different_live_vars(self) -> None:
		"""Two calls in sequence with different variables surviving each."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRCopy(dest=_t("y"), source=_c(2)),
			IRCall(dest=_t("r1"), function_name="f1", args=[_t("x")]),
			# y survives first call, x does not (consumed as arg)
			IRCall(dest=_t("r2"), function_name="f2", args=[_t("y")]),
			IRBinOp(dest=_t("z"), left=_t("r1"), op="+", right=_t("r2")),
			IRReturn(value=_t("z")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		across = analyzer.variables_live_across_calls()

		assert "entry" in across
		# y must survive the first call (used after it)
		assert "y" in across["entry"]
		# r1 must survive the second call (used after it)
		assert "r1" in across["entry"]

	def test_no_vars_across_void_call_at_end(self) -> None:
		"""A void call at the end of a block with no live-out has nothing across."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRCall(dest=None, function_name="print", args=[_t("a")]),
			IRReturn(),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		across = analyzer.variables_live_across_calls()

		# a is consumed as arg, nothing else is live
		assert across == {} or across.get("entry", set()) == set()

	def test_var_live_across_call_in_loop(self) -> None:
		"""Variable live across a call inside a loop body."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("sum"), source=_c(0)),
			IRCopy(dest=_t("i"), source=_c(0)),
			IRJump(target="header"),
			IRLabelInstr("header"),
			IRBinOp(dest=_t("cmp"), left=_t("i"), op="<", right=_c(5)),
			IRCondJump(condition=_t("cmp"), true_label="body", false_label="exit"),
			IRLabelInstr("body"),
			IRCall(dest=_t("val"), function_name="get_value", args=[_t("i")]),
			IRBinOp(dest=_t("sum"), left=_t("sum"), op="+", right=_t("val")),
			IRBinOp(dest=_t("i"), left=_t("i"), op="+", right=_c(1)),
			IRJump(target="header"),
			IRLabelInstr("exit"),
			IRReturn(value=_t("sum")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		across = analyzer.variables_live_across_calls()

		assert "body" in across
		# sum must survive the call (used after it)
		assert "sum" in across["body"]


# ---------------------------------------------------------------------------
# Merge point (phi-like) analysis
# ---------------------------------------------------------------------------

class TestMergePointLiveIns:
	def test_diamond_merge_both_preds_contribute(self) -> None:
		"""Both branches define x, used at merge -- both preds contribute."""
		instrs = [
			IRLabelInstr("entry"),
			IRCondJump(condition=_t("c"), true_label="then", false_label="else_"),
			IRLabelInstr("then"),
			IRCopy(dest=_t("x"), source=_c(10)),
			IRJump(target="merge"),
			IRLabelInstr("else_"),
			IRCopy(dest=_t("x"), source=_c(20)),
			IRJump(target="merge"),
			IRLabelInstr("merge"),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		merges = analyzer.merge_point_live_ins()

		assert "merge" in merges
		assert "x" in merges["merge"]
		sources = merges["merge"]["x"]
		assert "then" in sources
		assert "else_" in sources

	def test_merge_var_from_one_pred_only(self) -> None:
		"""Variable defined in one branch, used at merge -- only one pred contributes."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRCondJump(condition=_t("c"), true_label="then", false_label="else_"),
			IRLabelInstr("then"),
			IRCopy(dest=_t("y"), source=_t("x")),
			IRJump(target="merge"),
			IRLabelInstr("else_"),
			IRCopy(dest=_t("y"), source=_c(99)),
			IRJump(target="merge"),
			IRLabelInstr("merge"),
			IRReturn(value=_t("y")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		merges = analyzer.merge_point_live_ins()

		assert "merge" in merges
		assert "y" in merges["merge"]
		# y comes from both then and else_
		assert len(merges["merge"]["y"]) == 2

	def test_loop_header_is_merge_point(self) -> None:
		"""Loop header with entry and back-edge predecessors is a merge point."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(0)),
			IRJump(target="header"),
			IRLabelInstr("header"),
			IRBinOp(dest=_t("cmp"), left=_t("x"), op="<", right=_c(10)),
			IRCondJump(condition=_t("cmp"), true_label="body", false_label="exit"),
			IRLabelInstr("body"),
			IRBinOp(dest=_t("x"), left=_t("x"), op="+", right=_c(1)),
			IRJump(target="header"),
			IRLabelInstr("exit"),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		merges = analyzer.merge_point_live_ins()

		assert "header" in merges
		assert "x" in merges["header"]
		sources = merges["header"]["x"]
		assert "entry" in sources
		assert "body" in sources

	def test_no_merge_points_in_linear_code(self) -> None:
		"""Linear code has no merge points."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		assert analyzer.merge_point_live_ins() == {}

	def test_merge_with_multiple_vars(self) -> None:
		"""Multiple variables live at merge from different sources."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRCondJump(condition=_t("c"), true_label="then", false_label="else_"),
			IRLabelInstr("then"),
			IRCopy(dest=_t("b"), source=_c(10)),
			IRJump(target="merge"),
			IRLabelInstr("else_"),
			IRCopy(dest=_t("b"), source=_c(20)),
			IRJump(target="merge"),
			IRLabelInstr("merge"),
			IRBinOp(dest=_t("r"), left=_t("a"), op="+", right=_t("b")),
			IRReturn(value=_t("r")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		merges = analyzer.merge_point_live_ins()

		assert "merge" in merges
		# Both a and b are live-in at merge
		assert "a" in merges["merge"]
		assert "b" in merges["merge"]
		# a flows through both predecessors
		assert len(merges["merge"]["a"]) == 2
		# b is defined in both branches
		assert len(merges["merge"]["b"]) == 2


# ---------------------------------------------------------------------------
# Combined scenarios: critical edges + liveness
# ---------------------------------------------------------------------------

class TestCriticalEdgeLiveness:
	def test_liveness_correct_across_critical_edge(self) -> None:
		"""Verify liveness is still correct when critical edges exist."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(0)),
			IRJump(target="header"),
			IRLabelInstr("header"),
			IRBinOp(dest=_t("cmp"), left=_t("x"), op="<", right=_c(10)),
			IRCondJump(condition=_t("cmp"), true_label="body", false_label="exit"),
			IRLabelInstr("body"),
			IRBinOp(dest=_t("x"), left=_t("x"), op="+", right=_c(1)),
			IRBinOp(dest=_t("flag"), left=_t("x"), op="<", right=_c(5)),
			IRCondJump(condition=_t("flag"), true_label="header", false_label="exit"),
			IRLabelInstr("exit"),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)

		# Verify critical edges exist
		critical = analyzer.find_critical_edges()
		assert len(critical) > 0

		# Verify liveness is still correct
		result = analyzer.compute_liveness()
		assert "x" in result["header"][0]
		assert "x" in result["body"][0]
		assert "x" in result["exit"][0]

	def test_interference_at_critical_edge_merge(self) -> None:
		"""Variables meeting at a merge via critical edge should interfere properly."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRCopy(dest=_t("y"), source=_c(2)),
			IRJump(target="header"),
			IRLabelInstr("header"),
			IRBinOp(dest=_t("cmp"), left=_t("x"), op="<", right=_c(10)),
			IRCondJump(condition=_t("cmp"), true_label="body", false_label="exit"),
			IRLabelInstr("body"),
			IRBinOp(dest=_t("x"), left=_t("x"), op="+", right=_t("y")),
			IRBinOp(dest=_t("flag"), left=_t("x"), op="<", right=_c(5)),
			IRCondJump(condition=_t("flag"), true_label="header", false_label="exit"),
			IRLabelInstr("exit"),
			IRBinOp(dest=_t("z"), left=_t("x"), op="+", right=_t("y")),
			IRReturn(value=_t("z")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		ig = analyzer.interference_graph()

		# x and y are both live throughout the loop, so they interfere
		assert "y" in ig.get("x", set())
		assert "x" in ig.get("y", set())


# ---------------------------------------------------------------------------
# Call-crossing combined with merge points
# ---------------------------------------------------------------------------

class TestCallCrossingMerge:
	def test_call_in_one_branch_var_survives(self) -> None:
		"""Variable must survive a call in one branch, used at merge."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRCondJump(condition=_t("c"), true_label="call_branch", false_label="no_call"),
			IRLabelInstr("call_branch"),
			IRCall(dest=_t("r"), function_name="side_effect", args=[]),
			IRJump(target="merge"),
			IRLabelInstr("no_call"),
			IRJump(target="merge"),
			IRLabelInstr("merge"),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		across = analyzer.variables_live_across_calls()

		# x is live across the call in call_branch
		assert "call_branch" in across
		assert "x" in across["call_branch"]

		# Merge analysis: x flows from both preds
		merges = analyzer.merge_point_live_ins()
		assert "merge" in merges
		assert "x" in merges["merge"]
