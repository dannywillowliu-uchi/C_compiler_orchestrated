"""Tests for liveness analysis on hand-crafted IR."""

from src.compiler.cfg import CFG
from src.compiler.ir import (
	IRBinOp,
	IRCondJump,
	IRConst,
	IRCopy,
	IRJump,
	IRLabelInstr,
	IRReturn,
	IRTemp,
)
from src.compiler.liveness import LivenessAnalysis


def _t(name: str) -> IRTemp:
	return IRTemp(name)


def _c(val: int) -> IRConst:
	return IRConst(val)


class TestSimpleLinearCode:
	"""Linear code with no branches."""

	def test_sequential_defs_and_uses(self) -> None:
		# x = 1
		# y = x + 2
		# return y
		ir = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRBinOp(dest=_t("y"), left=_t("x"), op="+", right=_c(2)),
			IRReturn(value=_t("y")),
		]
		cfg = CFG(ir)
		la = LivenessAnalysis(cfg)
		block = cfg.blocks()[0]

		assert la.def_set(block) == {"x", "y"}
		assert la.use_set(block) == set()
		assert la.live_in(block) == set()
		assert la.live_out(block) == set()

	def test_live_at_point_linear(self) -> None:
		# x = 1          (idx 0)
		# y = x + 2      (idx 1)
		# return y        (idx 2)
		ir = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRBinOp(dest=_t("y"), left=_t("x"), op="+", right=_c(2)),
			IRReturn(value=_t("y")),
		]
		cfg = CFG(ir)
		la = LivenessAnalysis(cfg)
		block = cfg.blocks()[0]

		# Before 'x = 1': nothing is live (all defined locally before use)
		live0 = la.live_at_point(block, 0)
		assert live0 == set()

		# Before 'y = x + 2': x is live
		live1 = la.live_at_point(block, 1)
		assert live1 == {"x"}

		# Before 'return y': y is live
		live2 = la.live_at_point(block, 2)
		assert live2 == {"y"}


class TestBranchingCode:
	"""Code with if-else branching."""

	def test_branch_liveness(self) -> None:
		# entry:
		#   x = 1
		#   if cond goto then else goto else_
		# then:
		#   y = x + 1
		#   jump done
		# else_:
		#   y = x + 2
		#   jump done
		# done:
		#   return y
		ir = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRCopy(dest=_t("cond"), source=_c(1)),
			IRCondJump(condition=_t("cond"), true_label="then", false_label="else_"),
			IRLabelInstr("then"),
			IRBinOp(dest=_t("y"), left=_t("x"), op="+", right=_c(1)),
			IRJump(target="done"),
			IRLabelInstr("else_"),
			IRBinOp(dest=_t("y"), left=_t("x"), op="+", right=_c(2)),
			IRJump(target="done"),
			IRLabelInstr("done"),
			IRReturn(value=_t("y")),
		]
		cfg = CFG(ir)
		la = LivenessAnalysis(cfg)

		entry = cfg.get_block("entry")
		then_block = cfg.get_block("then")
		else_block = cfg.get_block("else_")
		done_block = cfg.get_block("done")
		assert entry is not None
		assert then_block is not None
		assert else_block is not None
		assert done_block is not None

		# x must be live out of entry (used in then and else)
		assert "x" in la.live_out(entry)

		# y must be live into done
		assert "y" in la.live_in(done_block)

		# x must be live into both then and else blocks
		assert "x" in la.live_in(then_block)
		assert "x" in la.live_in(else_block)

		# After done, nothing is live (return consumes y)
		assert la.live_out(done_block) == set()


class TestLoopCode:
	"""Loop code requiring fixed-point iteration."""

	def test_loop_liveness(self) -> None:
		# entry:
		#   i = 0
		#   sum = 0
		#   jump loop
		# loop:
		#   sum = sum + i
		#   i = i + 1
		#   if i goto loop else goto exit
		# exit:
		#   return sum
		ir = [
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
		cfg = CFG(ir)
		la = LivenessAnalysis(cfg)

		loop_block = cfg.get_block("loop")
		entry = cfg.get_block("entry")
		exit_block = cfg.get_block("exit")
		assert loop_block is not None
		assert entry is not None
		assert exit_block is not None

		# i and sum are live at the top of the loop (back-edge carries them)
		assert la.live_in(loop_block) == {"i", "sum"}

		# sum is live into exit (used by return)
		assert "sum" in la.live_in(exit_block)

		# i and sum are live out of entry (they flow into the loop)
		assert la.live_out(entry) == {"i", "sum"}

	def test_loop_live_at_point(self) -> None:
		# Same loop IR as above.
		ir = [
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
		cfg = CFG(ir)
		la = LivenessAnalysis(cfg)
		loop_block = cfg.get_block("loop")
		assert loop_block is not None

		# Before 'sum = sum + i' (idx 0): both i and sum are live
		assert la.live_at_point(loop_block, 0) == {"i", "sum"}

		# Before 'i = i + 1' (idx 1): i is live (used), sum is live (flows out)
		live1 = la.live_at_point(loop_block, 1)
		assert "i" in live1
		assert "sum" in live1

		# Before 'if i goto ...' (idx 2): i is live (condition), sum is live (flows out)
		live2 = la.live_at_point(loop_block, 2)
		assert "i" in live2
		assert "sum" in live2


class TestDeadVariables:
	"""Variables defined but never used should not be live."""

	def test_dead_variable_not_live(self) -> None:
		# x = 1
		# dead = 42   (never used)
		# return x
		ir = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRCopy(dest=_t("dead"), source=_c(42)),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(ir)
		la = LivenessAnalysis(cfg)
		block = cfg.blocks()[0]

		assert "dead" not in la.live_in(block)
		assert "dead" not in la.live_out(block)

	def test_dead_variable_live_at_point(self) -> None:
		# x = 1          (idx 0)
		# dead = 42      (idx 1)
		# return x       (idx 2)
		ir = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("x"), source=_c(1)),
			IRCopy(dest=_t("dead"), source=_c(42)),
			IRReturn(value=_t("x")),
		]
		cfg = CFG(ir)
		la = LivenessAnalysis(cfg)
		block = cfg.blocks()[0]

		# Before 'dead = 42': x is live (used by return), dead is not
		live1 = la.live_at_point(block, 1)
		assert "x" in live1
		assert "dead" not in live1

		# Before 'return x': x is live, dead is not
		live2 = la.live_at_point(block, 2)
		assert "x" in live2
		assert "dead" not in live2

	def test_multiple_dead_variables(self) -> None:
		# a = 1
		# b = 2
		# c = 3
		# return a
		ir = [
			IRLabelInstr("entry"),
			IRCopy(dest=_t("a"), source=_c(1)),
			IRCopy(dest=_t("b"), source=_c(2)),
			IRCopy(dest=_t("c"), source=_c(3)),
			IRReturn(value=_t("a")),
		]
		cfg = CFG(ir)
		la = LivenessAnalysis(cfg)
		block = cfg.blocks()[0]

		# Only a is ever used; b and c are dead
		assert la.live_in(block) == set()
		assert la.live_out(block) == set()
		assert "b" not in la.live_at_point(block, 2)
		assert "c" not in la.live_at_point(block, 3)
		assert "a" in la.live_at_point(block, 2)
		assert "a" in la.live_at_point(block, 3)


class TestEmptyCFG:
	"""Edge case: empty instruction list."""

	def test_empty_cfg(self) -> None:
		cfg = CFG([])
		la = LivenessAnalysis(cfg)
		assert la._live_in == {}
		assert la._live_out == {}
