"""Comprehensive tests for liveness analysis on CFG."""

from src.compiler.cfg import CFG
from src.compiler.ir import (
	IRAlloc,
	IRBinOp,
	IRCall,
	IRCondJump,
	IRConst,
	IRConvert,
	IRCopy,
	IRJump,
	IRLabelInstr,
	IRLoad,
	IRParam,
	IRReturn,
	IRStore,
	IRTemp,
	IRType,
	IRUnaryOp,
)
from src.compiler.liveness import LivenessAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t(name: str) -> IRTemp:
	return IRTemp(name)


def _c(val: int) -> IRConst:
	return IRConst(val)


# ---------------------------------------------------------------------------
# Tests: simple linear code
# ---------------------------------------------------------------------------

class TestSimpleLinearCode:
	def test_single_assignment_and_return(self) -> None:
		"""x = 1; return x; -- x is live between the two instructions."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		result = analyzer.compute_liveness()

		live_in, live_out = result["entry"]
		assert "x" not in live_in
		assert live_out == set()

	def test_chain_of_assignments(self) -> None:
		"""a = 1; b = a + 2; c = b + 3; return c;"""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRBinOp(dest=_t("b"), left=_t("a"), op="+", right=_c(2)),
			IRBinOp(dest=_t("c"), left=_t("b"), op="+", right=_c(3)),
			IRReturn(value=_t("c")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		result = analyzer.compute_liveness()

		live_in, live_out = result["entry"]
		assert live_in == set()
		assert live_out == set()

	def test_multiple_uses_of_same_variable(self) -> None:
		"""x = 1; y = x + x; return y;"""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRBinOp(dest=_t("y"), left=_t("x"), op="+", right=_t("x")),
			IRReturn(value=_t("y")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)

		block = cfg.get_block("entry")
		assert block is not None
		live = analyzer.get_live_at_point(block, 1)
		assert "x" in live

	def test_live_at_each_instruction(self) -> None:
		"""Verify live sets at every program point in a linear block."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),       # idx 0
			IRCopy(dest=_t("b"), source=_c(2)),       # idx 1
			IRBinOp(dest=_t("c"), left=_t("a"), op="+", right=_t("b")),  # idx 2
			IRReturn(value=_t("c")),                   # idx 3
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)

		block = cfg.get_block("entry")
		assert block is not None

		assert analyzer.get_live_at_point(block, 0) == set()
		assert analyzer.get_live_at_point(block, 1) == {"a"}
		assert analyzer.get_live_at_point(block, 2) == {"a", "b"}
		assert analyzer.get_live_at_point(block, 3) == {"c"}
		# After all instructions (live-out)
		assert analyzer.get_live_at_point(block, 4) == set()


# ---------------------------------------------------------------------------
# Tests: branches and joins (diamond CFG)
# ---------------------------------------------------------------------------

class TestBranchesJoins:
	def test_if_else_both_define(self) -> None:
		"""Diamond CFG where both branches define x, used after merge."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("cond"), source=_c(1)),
			IRCondJump(condition=_t("cond"), true_label="then", false_label="else_"),
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
		result = analyzer.compute_liveness()

		assert "x" in result["merge"][0]
		assert "x" in result["then"][1]
		assert "x" in result["else_"][1]
		assert "x" not in result["entry"][0]

	def test_variable_used_in_one_branch(self) -> None:
		"""'a' used in then-branch but not else-branch."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(5)),
			IRCopy(dest=_t("cond"), source=_c(1)),
			IRCondJump(condition=_t("cond"), true_label="then", false_label="else_"),
			IRLabelInstr("then"),
			IRCopy(dest=_t("b"), source=_t("a")),
			IRJump(target="merge"),
			IRLabelInstr("else_"),
			IRCopy(dest=_t("b"), source=_c(99)),
			IRJump(target="merge"),
			IRLabelInstr("merge"),
			IRReturn(value=_t("b")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		result = analyzer.compute_liveness()

		assert "a" in result["entry"][1]
		assert "a" in result["then"][0]
		assert "a" not in result["else_"][0]

	def test_variable_live_across_both_branches(self) -> None:
		"""Variable defined before branch, used after merge."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRCopy(dest=_t("cond"), source=_c(0)),
			IRCondJump(condition=_t("cond"), true_label="then", false_label="else_"),
			IRLabelInstr("then"),
			IRCopy(dest=_t("y"), source=_c(10)),
			IRJump(target="merge"),
			IRLabelInstr("else_"),
			IRCopy(dest=_t("y"), source=_c(20)),
			IRJump(target="merge"),
			IRLabelInstr("merge"),
			IRBinOp(dest=_t("z"), left=_t("x"), op="+", right=_t("y")),
			IRReturn(value=_t("z")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		result = analyzer.compute_liveness()

		# x must be live through both branches
		assert "x" in result["then"][0]
		assert "x" in result["then"][1]
		assert "x" in result["else_"][0]
		assert "x" in result["else_"][1]
		assert "x" in result["merge"][0]


# ---------------------------------------------------------------------------
# Tests: loops requiring iteration to converge
# ---------------------------------------------------------------------------

class TestLoops:
	def test_simple_while_loop(self) -> None:
		"""while (x < 10) { x = x + 1; } return x;"""
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
		result = analyzer.compute_liveness()

		assert "x" in result["header"][0]
		assert "x" in result["header"][1]
		assert "x" in result["body"][0]
		assert "x" in result["body"][1]
		assert "x" in result["exit"][0]

	def test_loop_with_accumulator(self) -> None:
		"""sum = 0; i = 0; while (i < 5) { sum += i; i++; } return sum;"""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("sum"), source=_c(0)),
			IRCopy(dest=_t("i"), source=_c(0)),
			IRJump(target="header"),
			IRLabelInstr("header"),
			IRBinOp(dest=_t("cmp"), left=_t("i"), op="<", right=_c(5)),
			IRCondJump(condition=_t("cmp"), true_label="body", false_label="exit"),
			IRLabelInstr("body"),
			IRBinOp(dest=_t("sum"), left=_t("sum"), op="+", right=_t("i")),
			IRBinOp(dest=_t("i"), left=_t("i"), op="+", right=_c(1)),
			IRJump(target="header"),
			IRLabelInstr("exit"),
			IRReturn(value=_t("sum")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		result = analyzer.compute_liveness()

		assert "sum" in result["header"][0]
		assert "i" in result["header"][0]
		assert "sum" in result["body"][0]
		assert "i" in result["body"][0]

	def test_convergence_with_back_edge(self) -> None:
		"""Verify analysis converges when back-edge introduces new live variables."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(0)),
			IRCopy(dest=_t("b"), source=_c(0)),
			IRJump(target="loop"),
			IRLabelInstr("loop"),
			IRCopy(dest=_t("a"), source=_t("b")),
			IRBinOp(dest=_t("b"), left=_t("a"), op="+", right=_c(1)),
			IRBinOp(dest=_t("cond"), left=_t("b"), op="<", right=_c(10)),
			IRCondJump(condition=_t("cond"), true_label="loop", false_label="done"),
			IRLabelInstr("done"),
			IRReturn(value=_t("a")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		result = analyzer.compute_liveness()

		# b must be live-in at loop (used by a = b before being redefined)
		assert "b" in result["loop"][0]
		# a must be live-out of loop (used in done block via back-edge or exit)
		assert "a" in result["loop"][1]

	def test_loop_live_at_point(self) -> None:
		"""Detailed live-at-point within a loop body."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("i"), source=_c(0)),
			IRCopy(dest=_t("sum"), source=_c(0)),
			IRJump(target="loop"),
			IRLabelInstr("loop"),
			IRBinOp(dest=_t("sum"), left=_t("sum"), op="+", right=_t("i")),
			IRBinOp(dest=_t("i"), left=_t("i"), op="+", right=_c(1)),
			IRCondJump(condition=_t("i"), true_label="loop", false_label="exit"),
			IRLabelInstr("exit"),
			IRReturn(value=_t("sum")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		loop_block = cfg.get_block("loop")
		assert loop_block is not None

		# Before sum = sum + i: both live
		assert analyzer.get_live_at_point(loop_block, 0) == {"i", "sum"}
		# Before i = i + 1: i live (used), sum live (flows out)
		live1 = analyzer.get_live_at_point(loop_block, 1)
		assert "i" in live1
		assert "sum" in live1
		# Before condjump: i live (condition), sum live (flows out)
		live2 = analyzer.get_live_at_point(loop_block, 2)
		assert "i" in live2
		assert "sum" in live2


# ---------------------------------------------------------------------------
# Tests: nested loops
# ---------------------------------------------------------------------------

class TestNestedLoops:
	def test_nested_loop_liveness(self) -> None:
		"""Outer loop variable must remain live through inner loop."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("i"), source=_c(0)),
			IRJump(target="outer_hdr"),
			IRLabelInstr("outer_hdr"),
			IRBinOp(dest=_t("ocmp"), left=_t("i"), op="<", right=_c(3)),
			IRCondJump(condition=_t("ocmp"), true_label="inner_init", false_label="exit"),
			IRLabelInstr("inner_init"),
			IRCopy(dest=_t("j"), source=_c(0)),
			IRJump(target="inner_hdr"),
			IRLabelInstr("inner_hdr"),
			IRBinOp(dest=_t("icmp"), left=_t("j"), op="<", right=_c(5)),
			IRCondJump(condition=_t("icmp"), true_label="inner_body", false_label="outer_latch"),
			IRLabelInstr("inner_body"),
			IRBinOp(dest=_t("j"), left=_t("j"), op="+", right=_c(1)),
			IRJump(target="inner_hdr"),
			IRLabelInstr("outer_latch"),
			IRBinOp(dest=_t("i"), left=_t("i"), op="+", right=_c(1)),
			IRJump(target="outer_hdr"),
			IRLabelInstr("exit"),
			IRReturn(value=_t("i")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		result = analyzer.compute_liveness()

		# i must be live throughout the inner loop
		assert "i" in result["inner_hdr"][0]
		assert "i" in result["inner_body"][0]
		assert "i" in result["inner_body"][1]
		# j lives in the inner loop
		assert "j" in result["inner_hdr"][0]
		assert "j" in result["inner_body"][0]

	def test_nested_loop_both_vars_live_at_outer(self) -> None:
		"""Both i and j should NOT be live at outer header (j is initialized inside)."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("i"), source=_c(0)),
			IRJump(target="outer_hdr"),
			IRLabelInstr("outer_hdr"),
			IRBinOp(dest=_t("ocmp"), left=_t("i"), op="<", right=_c(3)),
			IRCondJump(condition=_t("ocmp"), true_label="inner_init", false_label="exit"),
			IRLabelInstr("inner_init"),
			IRCopy(dest=_t("j"), source=_c(0)),
			IRJump(target="inner_hdr"),
			IRLabelInstr("inner_hdr"),
			IRBinOp(dest=_t("icmp"), left=_t("j"), op="<", right=_c(5)),
			IRCondJump(condition=_t("icmp"), true_label="inner_body", false_label="outer_latch"),
			IRLabelInstr("inner_body"),
			IRBinOp(dest=_t("j"), left=_t("j"), op="+", right=_c(1)),
			IRJump(target="inner_hdr"),
			IRLabelInstr("outer_latch"),
			IRBinOp(dest=_t("i"), left=_t("i"), op="+", right=_c(1)),
			IRJump(target="outer_hdr"),
			IRLabelInstr("exit"),
			IRReturn(value=_t("i")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		result = analyzer.compute_liveness()

		# i is live at outer_hdr, j is NOT (defined fresh in inner_init)
		assert "i" in result["outer_hdr"][0]
		assert "j" not in result["outer_hdr"][0]


# ---------------------------------------------------------------------------
# Tests: dead variables
# ---------------------------------------------------------------------------

class TestDeadVariables:
	def test_unused_variable_not_live(self) -> None:
		"""A variable defined but never used should not be live anywhere."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("dead"), source=_c(42)),
			IRCopy(dest=_t("alive"), source=_c(1)),
			IRReturn(value=_t("alive")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		result = analyzer.compute_liveness()

		live_in, live_out = result["entry"]
		assert "dead" not in live_in
		assert "dead" not in live_out

	def test_overwritten_before_use(self) -> None:
		"""x defined, then overwritten before use -- first def is dead."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),   # idx 0
			IRCopy(dest=_t("x"), source=_c(2)),   # idx 1
			IRReturn(value=_t("x")),               # idx 2
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)

		block = cfg.get_block("entry")
		assert block is not None

		# Before first copy: x not live
		assert "x" not in analyzer.get_live_at_point(block, 0)
		# Before second copy: x not live (about to be overwritten, no use at idx 1)
		assert "x" not in analyzer.get_live_at_point(block, 1)
		# Before return: x IS live
		assert "x" in analyzer.get_live_at_point(block, 2)

	def test_multiple_dead_variables(self) -> None:
		"""a = 1; b = 2; c = 3; return a; -- b and c are dead."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRCopy(dest=_t("b"), source=_c(2)),
			IRCopy(dest=_t("c"), source=_c(3)),
			IRReturn(value=_t("a")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		block = cfg.blocks()[0]

		assert analyzer.get_live_at_point(block, 0) == set()
		# After a is defined, before b: a is live
		assert "a" in analyzer.get_live_at_point(block, 1)
		assert "b" not in analyzer.get_live_at_point(block, 2)
		assert "c" not in analyzer.get_live_at_point(block, 3)
		assert "a" in analyzer.get_live_at_point(block, 3)


# ---------------------------------------------------------------------------
# Tests: function parameters (variables live-in at entry)
# ---------------------------------------------------------------------------

class TestFunctionParameters:
	def test_parameter_live_at_entry(self) -> None:
		"""Parameters used but not defined in the function are live-in at entry."""
		instrs = [
			IRLabelInstr("entry"),
			IRBinOp(dest=_t("result"), left=_t("param_a"), op="+", right=_t("param_b")),
			IRReturn(value=_t("result")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		result = analyzer.compute_liveness()

		live_in, _ = result["entry"]
		assert "param_a" in live_in
		assert "param_b" in live_in

	def test_parameter_used_in_loop(self) -> None:
		"""Parameter used inside loop body stays live through loop."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("i"), source=_c(0)),
			IRJump(target="loop"),
			IRLabelInstr("loop"),
			IRBinOp(dest=_t("i"), left=_t("i"), op="+", right=_t("step")),
			IRBinOp(dest=_t("cmp"), left=_t("i"), op="<", right=_t("limit")),
			IRCondJump(condition=_t("cmp"), true_label="loop", false_label="done"),
			IRLabelInstr("done"),
			IRReturn(value=_t("i")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		result = analyzer.compute_liveness()

		assert "step" in result["entry"][0]
		assert "limit" in result["entry"][0]
		assert "step" in result["loop"][0]
		assert "limit" in result["loop"][0]


# ---------------------------------------------------------------------------
# Tests: empty blocks
# ---------------------------------------------------------------------------

class TestEmptyBlocks:
	def test_empty_block_passes_liveness_through(self) -> None:
		"""An empty block should pass liveness from successors to predecessors."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRJump(target="empty"),
			IRLabelInstr("empty"),
			IRJump(target="use"),
			IRLabelInstr("use"),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		result = analyzer.compute_liveness()

		assert "x" in result["empty"][0]
		assert "x" in result["empty"][1]


# ---------------------------------------------------------------------------
# Tests: interference graph
# ---------------------------------------------------------------------------

class TestInterferenceGraph:
	def test_non_overlapping_lifetimes_no_interference(self) -> None:
		"""Sequential non-overlapping variables don't interfere."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRReturn(value=_t("a")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		ig = analyzer.interference_graph()

		assert ig.get("a", set()) == set()

	def test_overlapping_lifetimes_interfere(self) -> None:
		"""a = 1; b = 2; c = a + b -- a and b overlap."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRCopy(dest=_t("b"), source=_c(2)),
			IRBinOp(dest=_t("c"), left=_t("a"), op="+", right=_t("b")),
			IRReturn(value=_t("c")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		ig = analyzer.interference_graph()

		assert "b" in ig.get("a", set())
		assert "a" in ig.get("b", set())
		# c doesn't interfere with a or b (they die when c is born)
		assert "a" not in ig.get("c", set())
		assert "b" not in ig.get("c", set())

	def test_copy_coalescing_no_interference(self) -> None:
		"""For copy x = y, x and y should NOT interfere (move coalescing)."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("y"), source=_c(5)),
			IRCopy(dest=_t("x"), source=_t("y")),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		ig = analyzer.interference_graph()

		assert "y" not in ig.get("x", set())
		assert "x" not in ig.get("y", set())

	def test_interference_across_branches(self) -> None:
		"""Variables live across branches should interfere."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRCopy(dest=_t("y"), source=_c(2)),
			IRCopy(dest=_t("cond"), source=_c(0)),
			IRCondJump(condition=_t("cond"), true_label="then", false_label="else_"),
			IRLabelInstr("then"),
			IRBinOp(dest=_t("r"), left=_t("x"), op="+", right=_t("y")),
			IRJump(target="merge"),
			IRLabelInstr("else_"),
			IRBinOp(dest=_t("r"), left=_t("x"), op="-", right=_t("y")),
			IRJump(target="merge"),
			IRLabelInstr("merge"),
			IRReturn(value=_t("r")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)
		ig = analyzer.interference_graph()

		assert "y" in ig.get("x", set())
		assert "x" in ig.get("y", set())


# ---------------------------------------------------------------------------
# Tests: various IR instruction types
# ---------------------------------------------------------------------------

class TestInstructionTypes:
	def test_store_instruction_uses(self) -> None:
		"""IRStore uses both address and value."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("addr"), source=_c(100)),
			IRCopy(dest=_t("val"), source=_c(42)),
			IRStore(address=_t("addr"), value=_t("val")),
			IRReturn(),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)

		block = cfg.get_block("entry")
		assert block is not None
		live = analyzer.get_live_at_point(block, 2)
		assert "addr" in live
		assert "val" in live

	def test_load_instruction(self) -> None:
		"""IRLoad defines dest and uses address."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("ptr"), source=_c(100)),
			IRLoad(dest=_t("val"), address=_t("ptr")),
			IRReturn(value=_t("val")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)

		block = cfg.get_block("entry")
		assert block is not None
		assert "ptr" in analyzer.get_live_at_point(block, 1)
		live_after_load = analyzer.get_live_at_point(block, 2)
		assert "val" in live_after_load
		assert "ptr" not in live_after_load

	def test_call_instruction(self) -> None:
		"""IRCall defines dest and uses args."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRCopy(dest=_t("b"), source=_c(2)),
			IRCall(dest=_t("result"), function_name="foo", args=[_t("a"), _t("b")]),
			IRReturn(value=_t("result")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)

		block = cfg.get_block("entry")
		assert block is not None

		live_before_call = analyzer.get_live_at_point(block, 2)
		assert "a" in live_before_call
		assert "b" in live_before_call

		live_after_call = analyzer.get_live_at_point(block, 3)
		assert "result" in live_after_call
		assert "a" not in live_after_call
		assert "b" not in live_after_call

	def test_void_call(self) -> None:
		"""A call with no dest (void) only uses args."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRCall(dest=None, function_name="print_int", args=[_t("a")]),
			IRReturn(),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)

		block = cfg.get_block("entry")
		assert block is not None
		assert "a" in analyzer.get_live_at_point(block, 1)
		assert "a" not in analyzer.get_live_at_point(block, 2)

	def test_param_instruction(self) -> None:
		"""IRParam uses value."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(5)),
			IRParam(value=_t("x")),
			IRCall(dest=_t("r"), function_name="bar", args=[]),
			IRReturn(value=_t("r")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)

		block = cfg.get_block("entry")
		assert block is not None
		assert "x" in analyzer.get_live_at_point(block, 1)

	def test_convert_instruction(self) -> None:
		"""IRConvert defines dest and uses source."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(5)),
			IRConvert(dest=_t("fx"), source=_t("x"), from_type=IRType.INT, to_type=IRType.FLOAT),
			IRReturn(value=_t("fx")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)

		block = cfg.get_block("entry")
		assert block is not None
		assert "x" in analyzer.get_live_at_point(block, 1)
		live_after = analyzer.get_live_at_point(block, 2)
		assert "fx" in live_after
		assert "x" not in live_after

	def test_alloc_instruction(self) -> None:
		"""IRAlloc defines dest but uses nothing."""
		instrs = [
			IRLabelInstr("entry"),
			IRAlloc(dest=_t("buf"), size=16),
			IRReturn(value=_t("buf")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)

		block = cfg.get_block("entry")
		assert block is not None
		assert analyzer.get_live_at_point(block, 0) == set()
		assert "buf" in analyzer.get_live_at_point(block, 1)

	def test_unary_op_instruction(self) -> None:
		"""IRUnaryOp defines dest and uses operand."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(5)),
			IRUnaryOp(dest=_t("neg"), op="-", operand=_t("x")),
			IRReturn(value=_t("neg")),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)

		block = cfg.get_block("entry")
		assert block is not None
		assert "x" in analyzer.get_live_at_point(block, 1)
		live_after = analyzer.get_live_at_point(block, 2)
		assert "neg" in live_after
		assert "x" not in live_after


# ---------------------------------------------------------------------------
# Tests: empty CFG
# ---------------------------------------------------------------------------

class TestEmptyCFG:
	def test_empty_cfg_returns_empty(self) -> None:
		cfg = CFG([])
		analyzer = LivenessAnalyzer(cfg)
		assert analyzer.compute_liveness() == {}
		assert analyzer.interference_graph() == {}


# ---------------------------------------------------------------------------
# Tests: get_live_at_point edge cases
# ---------------------------------------------------------------------------

class TestGetLiveAtPointEdgeCases:
	def test_out_of_range_raises(self) -> None:
		instrs = [
			IRLabelInstr("entry"),
			IRReturn(value=_c(0)),
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)

		block = cfg.get_block("entry")
		assert block is not None

		try:
			analyzer.get_live_at_point(block, -1)
			assert False, "Should have raised IndexError"
		except IndexError:
			pass

		try:
			analyzer.get_live_at_point(block, 5)
			assert False, "Should have raised IndexError"
		except IndexError:
			pass

	def test_live_at_point_with_redefinition(self) -> None:
		"""x = 1; y = x + 1; x = 2; return x; -- first x is dead after y uses it."""
		instrs = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),       # idx 0
			IRBinOp(dest=_t("y"), left=_t("x"), op="+", right=_c(1)),  # idx 1
			IRCopy(dest=_t("x"), source=_c(2)),        # idx 2
			IRReturn(value=_t("x")),                   # idx 3
		]
		cfg = CFG(instrs)
		analyzer = LivenessAnalyzer(cfg)

		block = cfg.get_block("entry")
		assert block is not None

		# Before idx 1 (y = x+1): x is live
		assert "x" in analyzer.get_live_at_point(block, 1)
		# Before idx 2 (x = 2): x is NOT live (about to be overwritten)
		assert "x" not in analyzer.get_live_at_point(block, 2)
		# Before idx 3 (return x): x is live
		assert "x" in analyzer.get_live_at_point(block, 3)
