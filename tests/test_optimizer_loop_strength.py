"""Tests for loop strength reduction optimization pass."""

from compiler.ir import (
	IRBinOp,
	IRCondJump,
	IRConst,
	IRCopy,
	IRFunction,
	IRJump,
	IRLabelInstr,
	IRProgram,
	IRReturn,
	IRTemp,
	IRType,
)
from compiler.optimizer import IROptimizer


def _make_func(body, name="test", params=None):
	"""Helper to wrap instructions in a single-function program."""
	return IRProgram(functions=[
		IRFunction(name=name, params=params or [], body=body, return_type=IRType.INT)
	])


def _opt(body):
	"""Optimize a function body and return the resulting instructions."""
	prog = _make_func(body)
	result = IROptimizer().optimize(prog)
	return result.functions[0].body


def _lsr_only(body):
	"""Run only the loop strength reduction pass."""
	return IROptimizer()._loop_strength_reduction(body)


def _make_simple_loop(mul_left, mul_right, iv_step=1, iv_name="i"):
	"""Build a simple loop: i=0; while(i<n) { t = mul_left * mul_right; i = i + step; }

	Returns the instruction body for the loop.
	"""
	return [
		# i = 0
		IRCopy(dest=IRTemp(iv_name), source=IRConst(0)),
		IRLabelInstr(name="loop_header"),
		# if i < n goto loop_body else loop_exit
		IRCondJump(condition=IRTemp(iv_name), true_label="loop_body", false_label="loop_exit"),
		IRLabelInstr(name="loop_body"),
		# t = mul_left * mul_right
		IRBinOp(dest=IRTemp("t"), left=mul_left, op="*", right=mul_right),
		# i = i + step
		IRBinOp(dest=IRTemp(iv_name), left=IRTemp(iv_name), op="+", right=IRConst(iv_step)),
		IRJump(target="loop_header"),
		IRLabelInstr(name="loop_exit"),
		IRReturn(value=IRTemp("t")),
	]


class TestBasicLoopStrengthReduction:
	"""Test basic i*const -> accumulator pattern."""

	def test_mul_iv_by_constant(self):
		"""t = i * 4 in a loop should become an accumulator."""
		body = _make_simple_loop(IRTemp("i"), IRConst(4))
		result = _lsr_only(body)
		# The original in-loop multiply is gone (replaced by copy), but
		# there should be an initialization multiply before the loop
		in_loop = False
		loop_muls = []
		pre_loop_muls = []
		for instr in result:
			if isinstance(instr, IRLabelInstr) and instr.name == "loop_header":
				in_loop = True
			if isinstance(instr, IRLabelInstr) and instr.name == "loop_exit":
				in_loop = False
			if isinstance(instr, IRBinOp) and instr.op == "*":
				if in_loop:
					loop_muls.append(instr)
				else:
					pre_loop_muls.append(instr)
		# No multiply inside the loop
		assert len(loop_muls) == 0
		# There is an initializing multiply before the loop
		assert len(pre_loop_muls) == 1
		assert isinstance(pre_loop_muls[0].right, IRConst) and pre_loop_muls[0].right.value == 4

	def test_mul_constant_by_iv(self):
		"""t = 4 * i in a loop should also be reduced."""
		body = _make_simple_loop(IRConst(4), IRTemp("i"))
		result = _lsr_only(body)
		loop_muls = []
		in_loop = False
		for instr in result:
			if isinstance(instr, IRLabelInstr) and instr.name == "loop_header":
				in_loop = True
			if isinstance(instr, IRLabelInstr) and instr.name == "loop_exit":
				in_loop = False
			if in_loop and isinstance(instr, IRBinOp) and instr.op == "*":
				loop_muls.append(instr)
		assert len(loop_muls) == 0

	def test_accumulator_increment_inserted(self):
		"""After the IV update, an accumulator increment should be inserted."""
		body = _make_simple_loop(IRTemp("i"), IRConst(4))
		result = _lsr_only(body)
		# Find the IV update (i = i + 1) and check the next instruction
		for idx, instr in enumerate(result):
			if (
				isinstance(instr, IRBinOp)
				and isinstance(instr.left, IRTemp)
				and instr.left.name == "i"
				and instr.op == "+"
				and isinstance(instr.right, IRConst)
				and instr.right.value == 1
				and instr.dest.name == "i"
			):
				# Next instruction should be accum = accum + 4 (stride * step = 4 * 1)
				nxt = result[idx + 1]
				assert isinstance(nxt, IRBinOp)
				assert nxt.op == "+"
				assert isinstance(nxt.right, IRConst) and nxt.right.value == 4
				break
		else:
			raise AssertionError("IV update not found")

	def test_copy_replaces_multiply(self):
		"""The multiply inside the loop should be replaced with an IRCopy."""
		body = _make_simple_loop(IRTemp("i"), IRConst(4))
		result = _lsr_only(body)
		# Inside the loop body, find the instruction that defines 't'
		in_loop = False
		for instr in result:
			if isinstance(instr, IRLabelInstr) and instr.name == "loop_body":
				in_loop = True
			if isinstance(instr, IRLabelInstr) and instr.name == "loop_exit":
				in_loop = False
			if in_loop and isinstance(instr, IRCopy) and instr.dest.name == "t":
				# t is now assigned from the accumulator
				assert instr.source.name.startswith("__lsr_")
				break
		else:
			raise AssertionError("Copy from accumulator not found in loop body")


class TestStepSizes:
	"""Test different IV step sizes."""

	def test_step_2_stride_3(self):
		"""i += 2, t = i * 3 -> accumulator increments by 6."""
		body = _make_simple_loop(IRTemp("i"), IRConst(3), iv_step=2)
		result = _lsr_only(body)
		# Find the accumulator increment
		accum_increments = [
			instr for instr in result
			if isinstance(instr, IRBinOp)
			and instr.op == "+"
			and isinstance(instr.right, IRConst)
			and instr.dest.name.startswith("__lsr_")
		]
		assert len(accum_increments) == 1
		assert accum_increments[0].right.value == 6  # stride(3) * step(2)

	def test_step_negative(self):
		"""i -= 1 (i = i + (-1)), t = i * 5 -> accumulator increments by -5."""
		body = [
			IRCopy(dest=IRTemp("i"), source=IRConst(100)),
			IRLabelInstr(name="loop_header"),
			IRCondJump(condition=IRTemp("i"), true_label="loop_body", false_label="loop_exit"),
			IRLabelInstr(name="loop_body"),
			IRBinOp(dest=IRTemp("t"), left=IRTemp("i"), op="*", right=IRConst(5)),
			# i = i - 1 (using subtraction)
			IRBinOp(dest=IRTemp("i"), left=IRTemp("i"), op="-", right=IRConst(1)),
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_exit"),
			IRReturn(value=IRTemp("t")),
		]
		result = _lsr_only(body)
		accum_increments = [
			instr for instr in result
			if isinstance(instr, IRBinOp)
			and instr.op == "+"
			and instr.dest.name.startswith("__lsr_")
		]
		assert len(accum_increments) == 1
		# stride(5) * step(-1) = -5
		assert accum_increments[0].right.value == -5


class TestNoReduction:
	"""Cases where loop strength reduction should NOT apply."""

	def test_no_loop(self):
		"""No loop means no reduction."""
		body = [
			IRBinOp(dest=IRTemp("t"), left=IRTemp("x"), op="*", right=IRConst(4)),
			IRReturn(value=IRTemp("t")),
		]
		result = _lsr_only(body)
		assert result == body

	def test_non_iv_multiply(self):
		"""Multiply of two non-induction variables should not be reduced."""
		body = [
			IRCopy(dest=IRTemp("i"), source=IRConst(0)),
			IRLabelInstr(name="loop_header"),
			IRCondJump(condition=IRTemp("i"), true_label="loop_body", false_label="loop_exit"),
			IRLabelInstr(name="loop_body"),
			# Multiply of two non-IV temps
			IRBinOp(dest=IRTemp("t"), left=IRTemp("x"), op="*", right=IRConst(4)),
			IRBinOp(dest=IRTemp("i"), left=IRTemp("i"), op="+", right=IRConst(1)),
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_exit"),
			IRReturn(value=IRTemp("t")),
		]
		result = _lsr_only(body)
		# The multiply should remain since x is not an induction variable
		in_loop_muls = []
		in_loop = False
		for instr in result:
			if isinstance(instr, IRLabelInstr) and instr.name == "loop_header":
				in_loop = True
			if isinstance(instr, IRLabelInstr) and instr.name == "loop_exit":
				in_loop = False
			if in_loop and isinstance(instr, IRBinOp) and instr.op == "*":
				in_loop_muls.append(instr)
		assert len(in_loop_muls) == 1

	def test_iv_multiply_by_temp_not_const(self):
		"""t = i * x where x is a temp (not const) should not be reduced."""
		body = [
			IRCopy(dest=IRTemp("i"), source=IRConst(0)),
			IRLabelInstr(name="loop_header"),
			IRCondJump(condition=IRTemp("i"), true_label="loop_body", false_label="loop_exit"),
			IRLabelInstr(name="loop_body"),
			IRBinOp(dest=IRTemp("t"), left=IRTemp("i"), op="*", right=IRTemp("x")),
			IRBinOp(dest=IRTemp("i"), left=IRTemp("i"), op="+", right=IRConst(1)),
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_exit"),
			IRReturn(value=IRTemp("t")),
		]
		result = _lsr_only(body)
		in_loop_muls = []
		in_loop = False
		for instr in result:
			if isinstance(instr, IRLabelInstr) and instr.name == "loop_header":
				in_loop = True
			if isinstance(instr, IRLabelInstr) and instr.name == "loop_exit":
				in_loop = False
			if in_loop and isinstance(instr, IRBinOp) and instr.op == "*":
				in_loop_muls.append(instr)
		assert len(in_loop_muls) == 1

	def test_multi_def_iv_not_reduced(self):
		"""An IV defined more than once in the loop should not be reduced."""
		body = [
			IRCopy(dest=IRTemp("i"), source=IRConst(0)),
			IRLabelInstr(name="loop_header"),
			IRCondJump(condition=IRTemp("i"), true_label="loop_body", false_label="loop_exit"),
			IRLabelInstr(name="loop_body"),
			IRBinOp(dest=IRTemp("t"), left=IRTemp("i"), op="*", right=IRConst(4)),
			IRBinOp(dest=IRTemp("i"), left=IRTemp("i"), op="+", right=IRConst(1)),
			# Second definition of i -- breaks IV detection
			IRBinOp(dest=IRTemp("i"), left=IRTemp("i"), op="+", right=IRConst(1)),
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_exit"),
			IRReturn(value=IRTemp("t")),
		]
		result = _lsr_only(body)
		in_loop_muls = []
		in_loop = False
		for instr in result:
			if isinstance(instr, IRLabelInstr) and instr.name == "loop_header":
				in_loop = True
			if isinstance(instr, IRLabelInstr) and instr.name == "loop_exit":
				in_loop = False
			if in_loop and isinstance(instr, IRBinOp) and instr.op == "*":
				in_loop_muls.append(instr)
		assert len(in_loop_muls) == 1

	def test_empty_body(self):
		"""Empty body should be returned unchanged."""
		assert _lsr_only([]) == []


class TestIntegration:
	"""Test that loop strength reduction integrates with the full optimizer."""

	def test_full_optimizer_applies_lsr(self):
		"""The full optimizer should apply loop strength reduction."""
		body = _make_simple_loop(IRTemp("i"), IRConst(8))
		result = _opt(body)
		# After full optimization, there should be no multiply inside the loop
		in_loop = False
		loop_muls = []
		for instr in result:
			if isinstance(instr, IRLabelInstr) and instr.name == "loop_header":
				in_loop = True
			if isinstance(instr, IRLabelInstr) and instr.name == "loop_exit":
				in_loop = False
			if in_loop and isinstance(instr, IRBinOp) and instr.op == "*":
				loop_muls.append(instr)
		assert len(loop_muls) == 0

	def test_preserves_semantics_simple(self):
		"""The initialization multiply should use the IV's initial value."""
		body = _make_simple_loop(IRTemp("i"), IRConst(4))
		result = _lsr_only(body)
		# Find the initialization multiply (before loop header)
		pre_loop = []
		for instr in result:
			if isinstance(instr, IRLabelInstr) and instr.name == "loop_header":
				break
			pre_loop.append(instr)
		init_muls = [i for i in pre_loop if isinstance(i, IRBinOp) and i.op == "*"]
		assert len(init_muls) == 1
		# The init multiply should use the IV name to capture its pre-loop value
		assert isinstance(init_muls[0].left, IRTemp) and init_muls[0].left.name == "i"
		assert isinstance(init_muls[0].right, IRConst) and init_muls[0].right.value == 4

	def test_type_preserved(self):
		"""The accumulator should preserve the IR type of the original multiply."""
		body = [
			IRCopy(dest=IRTemp("i"), source=IRConst(0)),
			IRLabelInstr(name="loop_header"),
			IRCondJump(condition=IRTemp("i"), true_label="loop_body", false_label="loop_exit"),
			IRLabelInstr(name="loop_body"),
			IRBinOp(
				dest=IRTemp("t"), left=IRTemp("i"), op="*",
				right=IRConst(4), ir_type=IRType.LONG,
			),
			IRBinOp(
				dest=IRTemp("i"), left=IRTemp("i"), op="+",
				right=IRConst(1), ir_type=IRType.LONG,
			),
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_exit"),
			IRReturn(value=IRTemp("t")),
		]
		result = _lsr_only(body)
		# Find the copy that replaces the multiply
		for instr in result:
			if isinstance(instr, IRCopy) and instr.dest.name == "t":
				assert instr.ir_type == IRType.LONG
				break
		else:
			raise AssertionError("Copy from accumulator not found")


class TestMultipleReductions:
	"""Test multiple multiplies in the same loop."""

	def test_two_derived_from_same_iv(self):
		"""Two different multiplies of the same IV should both be reduced."""
		body = [
			IRCopy(dest=IRTemp("i"), source=IRConst(0)),
			IRLabelInstr(name="loop_header"),
			IRCondJump(condition=IRTemp("i"), true_label="loop_body", false_label="loop_exit"),
			IRLabelInstr(name="loop_body"),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("i"), op="*", right=IRConst(4)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("i"), op="*", right=IRConst(8)),
			IRBinOp(dest=IRTemp("i"), left=IRTemp("i"), op="+", right=IRConst(1)),
			IRJump(target="loop_header"),
			IRLabelInstr(name="loop_exit"),
			IRReturn(value=IRTemp("t1")),
		]
		result = _lsr_only(body)
		in_loop = False
		loop_muls = []
		for instr in result:
			if isinstance(instr, IRLabelInstr) and instr.name == "loop_header":
				in_loop = True
			if isinstance(instr, IRLabelInstr) and instr.name == "loop_exit":
				in_loop = False
			if in_loop and isinstance(instr, IRBinOp) and instr.op == "*":
				loop_muls.append(instr)
		assert len(loop_muls) == 0
		# Both t1 and t2 should now be copies from accumulators
		loop_copies = []
		in_loop = False
		for instr in result:
			if isinstance(instr, IRLabelInstr) and instr.name == "loop_body":
				in_loop = True
			if isinstance(instr, IRJump):
				in_loop = False
			if in_loop and isinstance(instr, IRCopy) and instr.dest.name in ("t1", "t2"):
				loop_copies.append(instr)
		assert len(loop_copies) == 2
