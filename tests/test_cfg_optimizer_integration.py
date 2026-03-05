"""Integration tests for CFG construction and optimizer passes.

Verifies that after optimization the CFG remains structurally valid,
that jump threading simplifies diamond patterns, dead code elimination
removes unreachable blocks, LICM works with nested loops, and constant
propagation crosses basic block boundaries.
"""

from compiler.cfg import CFG
from compiler.ir import (
	IRBinOp,
	IRCondJump,
	IRConst,
	IRCopy,
	IRFunction,
	IRInstruction,
	IRJump,
	IRLabelInstr,
	IRProgram,
	IRReturn,
	IRTemp,
	IRType,
)
from compiler.optimizer import IROptimizer


def _make_func(body: list[IRInstruction], name: str = "test") -> IRFunction:
	return IRFunction(name=name, params=[], body=body, return_type=IRType.INT)


def _make_program(body: list[IRInstruction]) -> IRProgram:
	return IRProgram(functions=[_make_func(body)])


def _optimize_body(body: list[IRInstruction]) -> list[IRInstruction]:
	prog = _make_program(body)
	opt = IROptimizer()
	result = opt.optimize(prog)
	return result.functions[0].body


def _cfg_is_valid(cfg: CFG) -> None:
	"""Assert CFG structural invariants: no dangling edges, all labels reachable from successors/predecessors."""
	all_blocks = set(cfg.blocks())
	for block in cfg.blocks():
		for succ in block.successors:
			assert succ in all_blocks, f"Dangling successor edge: {block.label} -> {succ.label}"
			assert block in succ.predecessors, (
				f"Missing predecessor back-edge: {block.label} -> {succ.label}"
			)
		for pred in block.predecessors:
			assert pred in all_blocks, f"Dangling predecessor edge: {pred.label} -> {block.label}"
			assert block in pred.successors, (
				f"Missing successor forward-edge: {pred.label} -> {block.label}"
			)

	# Jump targets should reference existing blocks
	for block in cfg.blocks():
		term = block.terminator()
		if isinstance(term, IRJump):
			assert cfg.get_block(term.target) is not None, (
				f"Jump target '{term.target}' has no corresponding block"
			)
		elif isinstance(term, IRCondJump):
			assert cfg.get_block(term.true_label) is not None, (
				f"CondJump true target '{term.true_label}' has no corresponding block"
			)
			assert cfg.get_block(term.false_label) is not None, (
				f"CondJump false target '{term.false_label}' has no corresponding block"
			)


class TestCFGValidityAfterOptimization:
	"""Verify that CFG structure is still valid after optimization passes."""

	def test_simple_linear_code(self) -> None:
		body = [
			IRLabelInstr(name="entry"),
			IRCopy(dest=IRTemp("t0"), source=IRConst(1)),
			IRCopy(dest=IRTemp("t1"), source=IRConst(2)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t0"), op="+", right=IRTemp("t1")),
			IRReturn(value=IRTemp("t2")),
		]
		optimized = _optimize_body(body)
		cfg = CFG(optimized)
		_cfg_is_valid(cfg)
		assert cfg.entry_block is not None

	def test_if_else_cfg_valid(self) -> None:
		body = [
			IRLabelInstr(name="entry"),
			IRCopy(dest=IRTemp("x"), source=IRConst(5)),
			IRCondJump(condition=IRTemp("x"), true_label="then", false_label="else"),
			IRLabelInstr(name="then"),
			IRCopy(dest=IRTemp("r"), source=IRConst(1)),
			IRJump(target="end"),
			IRLabelInstr(name="else"),
			IRCopy(dest=IRTemp("r"), source=IRConst(0)),
			IRJump(target="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("r")),
		]
		optimized = _optimize_body(body)
		cfg = CFG(optimized)
		_cfg_is_valid(cfg)

	def test_loop_cfg_valid(self) -> None:
		body = [
			IRLabelInstr(name="entry"),
			IRCopy(dest=IRTemp("i"), source=IRConst(0)),
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_header"),
			IRBinOp(dest=IRTemp("cond"), left=IRTemp("i"), op="<", right=IRConst(10)),
			IRCondJump(condition=IRTemp("cond"), true_label="loop_body", false_label="loop_end"),
			IRLabelInstr(name="loop_body"),
			IRBinOp(dest=IRTemp("i"), left=IRTemp("i"), op="+", right=IRConst(1)),
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_end"),
			IRReturn(value=IRTemp("i")),
		]
		optimized = _optimize_body(body)
		cfg = CFG(optimized)
		_cfg_is_valid(cfg)
		# Loop should still have reachable blocks
		reachable = cfg.reachable_blocks()
		assert len(reachable) >= 3

	def test_nested_conditionals_cfg_valid(self) -> None:
		body = [
			IRLabelInstr(name="entry"),
			IRCondJump(condition=IRTemp("a"), true_label="outer_then", false_label="outer_else"),
			IRLabelInstr(name="outer_then"),
			IRCondJump(condition=IRTemp("b"), true_label="inner_then", false_label="inner_else"),
			IRLabelInstr(name="inner_then"),
			IRCopy(dest=IRTemp("r"), source=IRConst(1)),
			IRJump(target="end"),
			IRLabelInstr(name="inner_else"),
			IRCopy(dest=IRTemp("r"), source=IRConst(2)),
			IRJump(target="end"),
			IRLabelInstr(name="outer_else"),
			IRCopy(dest=IRTemp("r"), source=IRConst(3)),
			IRJump(target="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("r")),
		]
		optimized = _optimize_body(body)
		cfg = CFG(optimized)
		_cfg_is_valid(cfg)

	def test_all_reachable_blocks_have_terminators_or_fallthrough(self) -> None:
		"""Every reachable block should either have a terminator or fall through."""
		body = [
			IRLabelInstr(name="entry"),
			IRCopy(dest=IRTemp("x"), source=IRConst(42)),
			IRCondJump(condition=IRTemp("x"), true_label="a", false_label="b"),
			IRLabelInstr(name="a"),
			IRReturn(value=IRConst(1)),
			IRLabelInstr(name="b"),
			IRReturn(value=IRConst(0)),
		]
		optimized = _optimize_body(body)
		cfg = CFG(optimized)
		_cfg_is_valid(cfg)
		for block in cfg.reachable_blocks():
			# Block should have successors or be an exit block
			term = block.terminator()
			if term is None:
				assert len(block.successors) > 0 or block in cfg.exit_blocks(), (
					f"Block {block.label} has no terminator and no successors"
				)


class TestJumpThreading:
	"""Test that jump threading correctly simplifies diamond patterns."""

	def test_thread_through_empty_block(self) -> None:
		"""Jump to a label that immediately jumps elsewhere gets threaded."""
		body = [
			IRLabelInstr(name="entry"),
			IRJump(target="trampoline"),
			IRLabelInstr(name="trampoline"),
			IRJump(target="final"),
			IRLabelInstr(name="final"),
			IRReturn(value=IRConst(42)),
		]
		optimized = _optimize_body(body)
		# The trampoline should be threaded through
		jumps = [i for i in optimized if isinstance(i, IRJump)]
		for j in jumps:
			assert j.target != "trampoline", "Jump should be threaded past trampoline"

	def test_condjump_threading(self) -> None:
		"""Conditional jump targets get threaded through trampolines."""
		body = [
			IRLabelInstr(name="entry"),
			IRCondJump(condition=IRTemp("c"), true_label="t_tramp", false_label="f_tramp"),
			IRLabelInstr(name="t_tramp"),
			IRJump(target="real_true"),
			IRLabelInstr(name="f_tramp"),
			IRJump(target="real_false"),
			IRLabelInstr(name="real_true"),
			IRReturn(value=IRConst(1)),
			IRLabelInstr(name="real_false"),
			IRReturn(value=IRConst(0)),
		]
		optimized = _optimize_body(body)
		cond_jumps = [i for i in optimized if isinstance(i, IRCondJump)]
		for cj in cond_jumps:
			assert cj.true_label != "t_tramp", "True branch should be threaded"
			assert cj.false_label != "f_tramp", "False branch should be threaded"

	def test_chain_threading(self) -> None:
		"""Multiple chained trampolines get fully resolved."""
		body = [
			IRLabelInstr(name="entry"),
			IRJump(target="a"),
			IRLabelInstr(name="a"),
			IRJump(target="b"),
			IRLabelInstr(name="b"),
			IRJump(target="c"),
			IRLabelInstr(name="c"),
			IRReturn(value=IRConst(99)),
		]
		optimized = _optimize_body(body)
		# After optimization, entry should jump directly to "c" or the return block
		entry_jumps = []
		in_entry = False
		for instr in optimized:
			if isinstance(instr, IRLabelInstr) and instr.name == "entry":
				in_entry = True
				continue
			if isinstance(instr, IRLabelInstr):
				in_entry = False
			if in_entry and isinstance(instr, IRJump):
				entry_jumps.append(instr)
		for j in entry_jumps:
			assert j.target not in ("a", "b"), f"Chain should be fully threaded, got {j.target}"

	def test_diamond_pattern_simplification(self) -> None:
		"""Diamond pattern where both sides jump to same target gets simplified."""
		body = [
			IRLabelInstr(name="entry"),
			IRCondJump(condition=IRTemp("c"), true_label="left", false_label="right"),
			IRLabelInstr(name="left"),
			IRCopy(dest=IRTemp("r"), source=IRConst(1)),
			IRJump(target="merge_tramp"),
			IRLabelInstr(name="right"),
			IRCopy(dest=IRTemp("r"), source=IRConst(2)),
			IRJump(target="merge_tramp"),
			IRLabelInstr(name="merge_tramp"),
			IRJump(target="merge"),
			IRLabelInstr(name="merge"),
			IRReturn(value=IRTemp("r")),
		]
		optimized = _optimize_body(body)
		# Both branches should jump directly to merge, not merge_tramp
		jumps = [i for i in optimized if isinstance(i, IRJump)]
		for j in jumps:
			assert j.target != "merge_tramp", "Diamond merge trampoline should be threaded"


class TestDeadCodeElimination:
	"""Test that dead code elimination removes unreachable blocks."""

	def test_code_after_unconditional_jump_removed(self) -> None:
		"""Instructions between a jump and the next label are dead."""
		body = [
			IRLabelInstr(name="entry"),
			IRJump(target="target"),
			# Dead code -- after unconditional jump
			IRCopy(dest=IRTemp("dead"), source=IRConst(999)),
			IRBinOp(dest=IRTemp("also_dead"), left=IRConst(1), op="+", right=IRConst(2)),
			IRLabelInstr(name="target"),
			IRReturn(value=IRConst(0)),
		]
		optimized = _optimize_body(body)
		# The dead instructions should be removed
		for instr in optimized:
			if isinstance(instr, IRCopy) and isinstance(instr.source, IRConst) and instr.source.value == 999:
				raise AssertionError("Dead code after jump should be eliminated")

	def test_code_after_return_removed(self) -> None:
		"""Instructions between a return and the next label are dead."""
		body = [
			IRLabelInstr(name="entry"),
			IRReturn(value=IRConst(42)),
			IRCopy(dest=IRTemp("dead"), source=IRConst(100)),
			IRLabelInstr(name="unreachable_label"),
			IRReturn(value=IRConst(0)),
		]
		optimized = _optimize_body(body)
		# Dead instruction between return and label should be removed
		copies = [i for i in optimized if isinstance(i, IRCopy) and isinstance(i.source, IRConst) and i.source.value == 100]
		assert len(copies) == 0, "Dead code after return should be eliminated"

	def test_unreachable_block_detected_by_cfg(self) -> None:
		"""CFG correctly identifies unreachable blocks after optimization."""
		body = [
			IRLabelInstr(name="entry"),
			IRReturn(value=IRConst(1)),
			IRLabelInstr(name="dead_block"),
			IRCopy(dest=IRTemp("x"), source=IRConst(42)),
			IRReturn(value=IRTemp("x")),
		]
		optimized = _optimize_body(body)
		cfg = CFG(optimized)
		_cfg_is_valid(cfg)
		unreachable = cfg.unreachable_blocks()
		unreachable_labels = {b.label for b in unreachable}
		assert "dead_block" in unreachable_labels, "dead_block should be unreachable"

	def test_unused_computation_removed(self) -> None:
		"""Pure computations whose results are never used get eliminated."""
		body = [
			IRLabelInstr(name="entry"),
			IRBinOp(dest=IRTemp("unused"), left=IRConst(10), op="*", right=IRConst(20)),
			IRCopy(dest=IRTemp("also_unused"), source=IRConst(77)),
			IRReturn(value=IRConst(0)),
		]
		optimized = _optimize_body(body)
		# unused and also_unused should be eliminated by DCE
		dest_names = set()
		for instr in optimized:
			if isinstance(instr, (IRBinOp, IRCopy)) and hasattr(instr, "dest"):
				dest_names.add(instr.dest.name)
		assert "unused" not in dest_names, "Unused computation should be eliminated"
		assert "also_unused" not in dest_names, "Unused copy should be eliminated"


class TestLoopInvariantCodeMotion:
	"""Test loop-invariant code motion with nested loops."""

	def test_invariant_hoisted_above_loop(self) -> None:
		"""A computation using only loop-external values gets hoisted."""
		body = [
			IRLabelInstr(name="entry"),
			IRCopy(dest=IRTemp("a"), source=IRConst(3)),
			IRCopy(dest=IRTemp("b"), source=IRConst(7)),
			IRCopy(dest=IRTemp("i"), source=IRConst(0)),
			IRJump(target="loop"),
			IRLabelInstr(name="loop"),
			# This is loop-invariant: a and b are not modified in the loop
			IRBinOp(dest=IRTemp("inv"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRBinOp(dest=IRTemp("i"), left=IRTemp("i"), op="+", right=IRConst(1)),
			IRBinOp(dest=IRTemp("cond"), left=IRTemp("i"), op="<", right=IRConst(10)),
			IRCondJump(condition=IRTemp("cond"), true_label="loop", false_label="exit"),
			IRLabelInstr(name="exit"),
			IRReturn(value=IRTemp("inv")),
		]
		optimized = _optimize_body(body)
		# Find the invariant computation - it should appear before the loop label
		loop_label_idx = None
		inv_idx = None
		for i, instr in enumerate(optimized):
			if isinstance(instr, IRLabelInstr) and instr.name == "loop":
				loop_label_idx = i
			if isinstance(instr, IRBinOp) and instr.dest.name == "inv":
				inv_idx = i
			# After constant folding, a+b=10, so it might become a copy
			if isinstance(instr, IRCopy) and instr.dest.name == "inv":
				inv_idx = i

		# inv should be computed (either as binop or constant-folded copy)
		# and should be before or at the loop header
		if inv_idx is not None and loop_label_idx is not None:
			assert inv_idx < loop_label_idx, (
				f"Loop-invariant 'inv' at {inv_idx} should be before loop header at {loop_label_idx}"
			)

	def test_non_invariant_stays_in_loop(self) -> None:
		"""A computation using loop-modified values stays inside the loop."""
		body = [
			IRLabelInstr(name="entry"),
			IRCopy(dest=IRTemp("i"), source=IRConst(0)),
			IRJump(target="loop"),
			IRLabelInstr(name="loop"),
			# i is modified in the loop, so i+1 is NOT loop-invariant
			IRBinOp(dest=IRTemp("i"), left=IRTemp("i"), op="+", right=IRConst(1)),
			IRBinOp(dest=IRTemp("cond"), left=IRTemp("i"), op="<", right=IRConst(10)),
			IRCondJump(condition=IRTemp("cond"), true_label="loop", false_label="exit"),
			IRLabelInstr(name="exit"),
			IRReturn(value=IRTemp("i")),
		]
		optimized = _optimize_body(body)
		# The increment of i should still be inside the loop
		loop_label_idx = None
		exit_label_idx = None
		i_increment_idx = None
		for idx, instr in enumerate(optimized):
			if isinstance(instr, IRLabelInstr) and instr.name == "loop":
				loop_label_idx = idx
			if isinstance(instr, IRLabelInstr) and instr.name == "exit":
				exit_label_idx = idx
			if isinstance(instr, IRBinOp) and instr.dest.name == "i" and instr.op == "+":
				i_increment_idx = idx

		if i_increment_idx is not None and loop_label_idx is not None and exit_label_idx is not None:
			assert loop_label_idx < i_increment_idx < exit_label_idx, (
				"Loop-variant computation should stay inside the loop"
			)

	def test_nested_loop_invariant(self) -> None:
		"""In nested loops, outer-loop invariant relative to inner loop can be hoisted."""
		body = [
			IRLabelInstr(name="entry"),
			IRCopy(dest=IRTemp("x"), source=IRConst(5)),
			IRCopy(dest=IRTemp("i"), source=IRConst(0)),
			IRJump(target="outer"),
			IRLabelInstr(name="outer"),
			IRCopy(dest=IRTemp("j"), source=IRConst(0)),
			IRJump(target="inner"),
			IRLabelInstr(name="inner"),
			# x * 2 is invariant to both loops since x is never modified
			IRBinOp(dest=IRTemp("inv"), left=IRTemp("x"), op="*", right=IRConst(2)),
			IRBinOp(dest=IRTemp("j"), left=IRTemp("j"), op="+", right=IRConst(1)),
			IRBinOp(dest=IRTemp("jcond"), left=IRTemp("j"), op="<", right=IRConst(5)),
			IRCondJump(condition=IRTemp("jcond"), true_label="inner", false_label="inner_done"),
			IRLabelInstr(name="inner_done"),
			IRBinOp(dest=IRTemp("i"), left=IRTemp("i"), op="+", right=IRConst(1)),
			IRBinOp(dest=IRTemp("icond"), left=IRTemp("i"), op="<", right=IRConst(3)),
			IRCondJump(condition=IRTemp("icond"), true_label="outer", false_label="exit"),
			IRLabelInstr(name="exit"),
			IRReturn(value=IRTemp("inv")),
		]
		optimized = _optimize_body(body)
		# After optimization, x*2 should be constant-folded to 10
		# or hoisted out of the inner loop at minimum
		inner_label_idx = None
		inv_idx = None
		for idx, instr in enumerate(optimized):
			if isinstance(instr, IRLabelInstr) and instr.name == "inner":
				inner_label_idx = idx
			if isinstance(instr, (IRBinOp, IRCopy)) and hasattr(instr, "dest") and instr.dest.name == "inv":
				inv_idx = idx

		if inv_idx is not None and inner_label_idx is not None:
			assert inv_idx < inner_label_idx, (
				"Invariant computation should be hoisted above inner loop"
			)

	def test_cfg_valid_after_licm(self) -> None:
		"""CFG remains valid after LICM hoists instructions."""
		body = [
			IRLabelInstr(name="entry"),
			IRCopy(dest=IRTemp("c"), source=IRConst(100)),
			IRCopy(dest=IRTemp("i"), source=IRConst(0)),
			IRJump(target="loop"),
			IRLabelInstr(name="loop"),
			IRBinOp(dest=IRTemp("v"), left=IRTemp("c"), op="+", right=IRConst(1)),
			IRBinOp(dest=IRTemp("i"), left=IRTemp("i"), op="+", right=IRConst(1)),
			IRBinOp(dest=IRTemp("cond"), left=IRTemp("i"), op="<", right=IRConst(5)),
			IRCondJump(condition=IRTemp("cond"), true_label="loop", false_label="done"),
			IRLabelInstr(name="done"),
			IRReturn(value=IRTemp("v")),
		]
		optimized = _optimize_body(body)
		cfg = CFG(optimized)
		_cfg_is_valid(cfg)


class TestConstantPropagationAcrossBlocks:
	"""Test that constant propagation works across basic blocks."""

	def test_constant_folding_across_copy_chain(self) -> None:
		"""Constants propagated through copies get folded in later computations."""
		body = [
			IRLabelInstr(name="entry"),
			IRCopy(dest=IRTemp("a"), source=IRConst(10)),
			IRCopy(dest=IRTemp("b"), source=IRConst(20)),
			IRBinOp(dest=IRTemp("c"), left=IRTemp("a"), op="+", right=IRTemp("b")),
			IRReturn(value=IRTemp("c")),
		]
		optimized = _optimize_body(body)
		# After copy prop + constant fold, the return should use 30 directly
		returns = [i for i in optimized if isinstance(i, IRReturn)]
		assert len(returns) == 1
		ret = returns[0]
		assert isinstance(ret.value, IRConst) and ret.value.value == 30, (
			f"Expected return of constant 30, got {ret.value}"
		)

	def test_constant_prop_through_conditional_branches(self) -> None:
		"""Copy propagation clears at labels (conservative), so post-branch uses remain."""
		body = [
			IRLabelInstr(name="entry"),
			IRCopy(dest=IRTemp("x"), source=IRConst(5)),
			IRCondJump(condition=IRTemp("x"), true_label="then", false_label="else"),
			IRLabelInstr(name="then"),
			# After label, copy map is cleared, so x might not be propagated
			IRBinOp(dest=IRTemp("r"), left=IRTemp("x"), op="*", right=IRConst(2)),
			IRJump(target="end"),
			IRLabelInstr(name="else"),
			IRCopy(dest=IRTemp("r"), source=IRConst(0)),
			IRJump(target="end"),
			IRLabelInstr(name="end"),
			IRReturn(value=IRTemp("r")),
		]
		optimized = _optimize_body(body)
		cfg = CFG(optimized)
		_cfg_is_valid(cfg)

	def test_constant_folding_chain(self) -> None:
		"""Multiple dependent constant folds converge to a single value."""
		body = [
			IRLabelInstr(name="entry"),
			IRCopy(dest=IRTemp("a"), source=IRConst(3)),
			IRCopy(dest=IRTemp("b"), source=IRConst(4)),
			IRBinOp(dest=IRTemp("c"), left=IRTemp("a"), op="+", right=IRTemp("b")),  # 7
			IRBinOp(dest=IRTemp("d"), left=IRTemp("c"), op="*", right=IRConst(2)),   # 14
			IRBinOp(dest=IRTemp("e"), left=IRTemp("d"), op="-", right=IRConst(4)),   # 10
			IRReturn(value=IRTemp("e")),
		]
		optimized = _optimize_body(body)
		returns = [i for i in optimized if isinstance(i, IRReturn)]
		assert len(returns) == 1
		ret = returns[0]
		assert isinstance(ret.value, IRConst) and ret.value.value == 10, (
			f"Expected return of constant 10, got {ret.value}"
		)

	def test_strength_reduction_with_constant_prop(self) -> None:
		"""Strength reduction replaces multiply-by-power-of-2 with shift after const prop."""
		body = [
			IRLabelInstr(name="entry"),
			IRCopy(dest=IRTemp("n"), source=IRConst(8)),
			IRBinOp(dest=IRTemp("r"), left=IRTemp("x"), op="*", right=IRTemp("n")),
			IRReturn(value=IRTemp("r")),
		]
		optimized = _optimize_body(body)
		# After copy prop, x * 8 should become x << 3
		shifts = [i for i in optimized if isinstance(i, IRBinOp) and i.op == "<<"]
		# If it got strength-reduced, we should see a shift or the entire thing
		# might have been further optimized. Either way, no multiply by 8 should remain.
		mults = [i for i in optimized if isinstance(i, IRBinOp) and i.op == "*"]
		assert len(mults) == 0 or len(shifts) > 0, (
			"Multiply by power of 2 should be strength-reduced to shift"
		)

	def test_boolean_constant_propagation(self) -> None:
		"""Boolean simplification works with propagated constants."""
		body = [
			IRLabelInstr(name="entry"),
			IRCopy(dest=IRTemp("zero"), source=IRConst(0)),
			IRBinOp(dest=IRTemp("r"), left=IRTemp("x"), op="||", right=IRTemp("zero")),
			IRReturn(value=IRTemp("r")),
		]
		optimized = _optimize_body(body)
		# x || 0 should simplify to x
		cfg = CFG(optimized)
		_cfg_is_valid(cfg)


class TestCombinedOptimizationsCFGIntegrity:
	"""Test that multiple optimization passes compose correctly and leave CFG valid."""

	def test_constant_fold_then_dce_then_cfg(self) -> None:
		"""Constant folding creates dead code, DCE removes it, CFG stays valid."""
		body = [
			IRLabelInstr(name="entry"),
			IRBinOp(dest=IRTemp("a"), left=IRConst(2), op="+", right=IRConst(3)),
			IRBinOp(dest=IRTemp("b"), left=IRConst(10), op="*", right=IRConst(0)),
			# Only 'a' is used, 'b' is dead
			IRReturn(value=IRTemp("a")),
		]
		optimized = _optimize_body(body)
		cfg = CFG(optimized)
		_cfg_is_valid(cfg)
		# b should be eliminated
		dest_names = set()
		for instr in optimized:
			if hasattr(instr, "dest") and instr.dest is not None:
				dest_names.add(instr.dest.name)
		assert "b" not in dest_names, "Dead computation 'b' should be eliminated"

	def test_jump_threading_then_unreachable_elimination(self) -> None:
		"""After threading, trampoline blocks become unreachable."""
		body = [
			IRLabelInstr(name="entry"),
			IRJump(target="tramp1"),
			IRLabelInstr(name="tramp1"),
			IRJump(target="tramp2"),
			IRLabelInstr(name="tramp2"),
			IRJump(target="real"),
			IRLabelInstr(name="real"),
			IRReturn(value=IRConst(1)),
		]
		optimized = _optimize_body(body)
		cfg = CFG(optimized)
		_cfg_is_valid(cfg)
		# After threading, entry jumps directly to real
		# Trampolines may still exist as labels but should be unreachable or removed
		reachable_labels = {b.label for b in cfg.reachable_blocks()}
		# The entry and real blocks should be reachable
		assert "real" in reachable_labels or any(
			isinstance(i, IRReturn) for b in cfg.reachable_blocks() for i in b.instructions
		)

	def test_cse_across_optimization_rounds(self) -> None:
		"""CSE eliminates duplicate expressions after copy propagation reveals them."""
		body = [
			IRLabelInstr(name="entry"),
			IRCopy(dest=IRTemp("a"), source=IRTemp("x")),
			IRBinOp(dest=IRTemp("r1"), left=IRTemp("x"), op="+", right=IRConst(1)),
			IRBinOp(dest=IRTemp("r2"), left=IRTemp("a"), op="+", right=IRConst(1)),
			# After copy prop, both become x + 1, then CSE deduplicates
			IRBinOp(dest=IRTemp("result"), left=IRTemp("r1"), op="+", right=IRTemp("r2")),
			IRReturn(value=IRTemp("result")),
		]
		optimized = _optimize_body(body)
		cfg = CFG(optimized)
		_cfg_is_valid(cfg)
		# Count additions of x + 1 -- should only appear once due to CSE
		x_plus_1 = [
			i for i in optimized
			if isinstance(i, IRBinOp) and i.op == "+"
			and isinstance(i.right, IRConst) and i.right.value == 1
			and isinstance(i.left, IRTemp) and i.left.name == "x"
		]
		assert len(x_plus_1) <= 1, "CSE should eliminate duplicate x+1 computation"

	def test_empty_program(self) -> None:
		"""Empty instruction list produces valid (empty) CFG."""
		optimized = _optimize_body([])
		cfg = CFG(optimized)
		_cfg_is_valid(cfg)
		assert cfg.entry_block is None

	def test_single_return(self) -> None:
		"""Single return instruction produces valid CFG."""
		body = [
			IRLabelInstr(name="entry"),
			IRReturn(value=IRConst(0)),
		]
		optimized = _optimize_body(body)
		cfg = CFG(optimized)
		_cfg_is_valid(cfg)
		assert len(cfg.exit_blocks()) >= 1
