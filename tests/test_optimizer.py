"""Tests for IR optimizer: constant folding, dead code elimination, copy propagation,
strength reduction, jump threading, and unreachable code elimination."""

from compiler.ir import (
	IRBinOp,
	IRCall,
	IRCondJump,
	IRConst,
	IRCopy,
	IRFunction,
	IRJump,
	IRLabelInstr,
	IRLoad,
	IRParam,
	IRProgram,
	IRReturn,
	IRStore,
	IRTemp,
	IRType,
	IRUnaryOp,
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


def _fold_only(body):
	"""Run only constant folding (single pass, no DCE/propagation)."""
	return IROptimizer()._constant_fold(body)


def _dce_only(body):
	"""Run only dead code elimination (single pass)."""
	return IROptimizer()._dead_code_elimination(body)


def _propagate_only(body):
	"""Run only copy propagation (single pass)."""
	return IROptimizer()._copy_propagation(body)


def _strength_only(body):
	"""Run only strength reduction (single pass)."""
	return IROptimizer()._strength_reduction(body)


def _thread_only(body):
	"""Run only jump threading."""
	return IROptimizer()._jump_threading(body)


def _unreachable_only(body):
	"""Run only unreachable code elimination."""
	return IROptimizer()._unreachable_elimination(body)


# ── Constant Folding (isolated) ──


class TestConstantFolding:
	def test_add(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(3), op="+", right=IRConst(4))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(7)

	def test_subtract(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(10), op="-", right=IRConst(3))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(7)

	def test_multiply(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(6), op="*", right=IRConst(7))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(42)

	def test_divide(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(20), op="/", right=IRConst(4))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(5)

	def test_divide_by_zero_preserved(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(1), op="/", right=IRConst(0))]
		result = _fold_only(body)
		assert isinstance(result[0], IRBinOp)

	def test_modulo(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(17), op="%", right=IRConst(5))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(2)

	def test_modulo_by_zero_preserved(self):
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(1), op="%", right=IRConst(0))]
		result = _fold_only(body)
		assert isinstance(result[0], IRBinOp)

	def test_comparison_ops(self):
		for op, a, b, expected in [
			("<", 1, 2, 1), ("<", 2, 1, 0),
			(">", 3, 1, 1), (">", 1, 3, 0),
			("<=", 2, 2, 1), (">=", 2, 2, 1),
			("==", 5, 5, 1), ("==", 5, 6, 0),
			("!=", 5, 6, 1), ("!=", 5, 5, 0),
		]:
			body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(a), op=op, right=IRConst(b))]
			result = _fold_only(body)
			assert isinstance(result[0], IRCopy), f"Failed for {a} {op} {b}"
			assert result[0].source == IRConst(expected), f"Expected {expected} for {a} {op} {b}"

	def test_bitwise_ops(self):
		for op, a, b, expected in [
			("&", 0b1100, 0b1010, 0b1000),
			("|", 0b1100, 0b1010, 0b1110),
			("^", 0b1100, 0b1010, 0b0110),
			("<<", 1, 3, 8),
			(">>", 16, 2, 4),
		]:
			body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(a), op=op, right=IRConst(b))]
			result = _fold_only(body)
			assert result[0].source == IRConst(expected)

	def test_logical_ops(self):
		for op, a, b, expected in [
			("&&", 1, 1, 1), ("&&", 1, 0, 0), ("&&", 0, 1, 0),
			("||", 0, 0, 0), ("||", 1, 0, 1), ("||", 0, 1, 1),
		]:
			body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(a), op=op, right=IRConst(b))]
			result = _fold_only(body)
			assert result[0].source == IRConst(expected)

	def test_unary_negate(self):
		body = [IRUnaryOp(dest=IRTemp("t0"), op="-", operand=IRConst(42))]
		result = _fold_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(-42)

	def test_unary_bitwise_not(self):
		body = [IRUnaryOp(dest=IRTemp("t0"), op="~", operand=IRConst(0))]
		result = _fold_only(body)
		assert result[0].source == IRConst(-1)

	def test_unary_logical_not(self):
		body = [IRUnaryOp(dest=IRTemp("t0"), op="!", operand=IRConst(0))]
		result = _fold_only(body)
		assert result[0].source == IRConst(1)

	def test_no_fold_when_non_const(self):
		body = [IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="+", right=IRConst(1))]
		result = _fold_only(body)
		assert isinstance(result[0], IRBinOp)

	def test_chain_fold_with_full_optimizer(self):
		"""Constant folding + copy propagation enables chained folding."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(2), op="+", right=IRConst(3)),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="*", right=IRConst(4)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _opt(body)
		# After all passes: t0=5, propagated into t1=5*4=20, DCE removes dead copies
		assert len(result) == 1
		assert isinstance(result[0], IRReturn)
		assert result[0].value == IRConst(20)


# ── Dead Code Elimination (isolated) ──


class TestDeadCodeElimination:
	def test_remove_unused_copy(self):
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(42)),
			IRReturn(value=None),
		]
		result = _dce_only(body)
		assert len(result) == 1
		assert isinstance(result[0], IRReturn)

	def test_remove_unused_binop(self):
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(1), op="+", right=IRConst(2)),
			IRReturn(value=None),
		]
		result = _dce_only(body)
		assert len(result) == 1
		assert isinstance(result[0], IRReturn)

	def test_remove_unused_unaryop(self):
		body = [
			IRUnaryOp(dest=IRTemp("t0"), op="-", operand=IRConst(5)),
			IRReturn(value=None),
		]
		result = _dce_only(body)
		assert len(result) == 1

	def test_keep_used_instruction(self):
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(42)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _dce_only(body)
		assert len(result) == 2

	def test_keep_call_with_unused_dest(self):
		"""Calls have side effects and must not be removed."""
		body = [
			IRCall(dest=IRTemp("t0"), function_name="printf", args=[IRConst(0)]),
			IRReturn(value=None),
		]
		result = _dce_only(body)
		assert len(result) == 2
		assert isinstance(result[0], IRCall)

	def test_keep_store(self):
		body = [
			IRStore(address=IRTemp("t0"), value=IRConst(1)),
			IRReturn(value=None),
		]
		result = _dce_only(body)
		assert len(result) == 2

	def test_cascading_dce(self):
		"""Removing t1 makes t0 dead too (via iterative optimization)."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(1)),
			IRCopy(dest=IRTemp("t1"), source=IRTemp("t0")),
			IRReturn(value=None),
		]
		result = _opt(body)
		assert len(result) == 1
		assert isinstance(result[0], IRReturn)


# ── Copy Propagation (isolated) ──


class TestCopyPropagation:
	def test_basic_propagation(self):
		body = [
			IRCopy(dest=IRTemp("t1"), source=IRTemp("t0")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _propagate_only(body)
		ret = [i for i in result if isinstance(i, IRReturn)][0]
		assert ret.value == IRTemp("t0")

	def test_propagation_into_binop(self):
		body = [
			IRCopy(dest=IRTemp("t1"), source=IRTemp("t0")),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t1"), op="+", right=IRConst(1)),
		]
		result = _propagate_only(body)
		binops = [i for i in result if isinstance(i, IRBinOp)]
		assert binops[0].left == IRTemp("t0")

	def test_propagation_chain(self):
		"""t2 = copy t1; t3 = copy t2 => t3 should resolve to t1."""
		body = [
			IRCopy(dest=IRTemp("t2"), source=IRTemp("t1")),
			IRCopy(dest=IRTemp("t3"), source=IRTemp("t2")),
			IRReturn(value=IRTemp("t3")),
		]
		result = _propagate_only(body)
		ret = [i for i in result if isinstance(i, IRReturn)][0]
		assert ret.value == IRTemp("t1")

	def test_propagation_const(self):
		"""Copy propagation with constants."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(42)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _propagate_only(body)
		ret = [i for i in result if isinstance(i, IRReturn)][0]
		assert ret.value == IRConst(42)

	def test_propagation_killed_by_redef(self):
		"""Redefinition of source invalidates the copy mapping."""
		body = [
			IRCopy(dest=IRTemp("t1"), source=IRTemp("t0")),
			IRCopy(dest=IRTemp("t0"), source=IRConst(99)),
			IRReturn(value=IRTemp("t1")),
		]
		result = _propagate_only(body)
		# t1 -> t0 mapping is invalidated when t0 is redefined
		# so t1 stays as t1 in the return
		ret = [i for i in result if isinstance(i, IRReturn)][0]
		assert ret.value == IRTemp("t1")

	def test_propagation_cleared_at_label(self):
		"""Copy mappings are cleared at labels (conservative at join points)."""
		body = [
			IRCopy(dest=IRTemp("t1"), source=IRTemp("t0")),
			IRLabelInstr(name="L1"),
			IRReturn(value=IRTemp("t1")),
		]
		result = _propagate_only(body)
		ret = [i for i in result if isinstance(i, IRReturn)][0]
		assert ret.value == IRTemp("t1")

	def test_propagation_in_store(self):
		body = [
			IRCopy(dest=IRTemp("t1"), source=IRTemp("t0")),
			IRStore(address=IRTemp("t1"), value=IRConst(5)),
		]
		result = _propagate_only(body)
		stores = [i for i in result if isinstance(i, IRStore)]
		assert stores[0].address == IRTemp("t0")

	def test_propagation_in_condjump(self):
		body = [
			IRCopy(dest=IRTemp("t1"), source=IRTemp("t0")),
			IRCondJump(condition=IRTemp("t1"), true_label="L1", false_label="L2"),
		]
		result = _propagate_only(body)
		cjs = [i for i in result if isinstance(i, IRCondJump)]
		assert cjs[0].condition == IRTemp("t0")

	def test_propagation_in_call_args(self):
		body = [
			IRCopy(dest=IRTemp("t1"), source=IRTemp("t0")),
			IRCall(dest=IRTemp("t2"), function_name="foo", args=[IRTemp("t1")]),
		]
		result = _propagate_only(body)
		calls = [i for i in result if isinstance(i, IRCall)]
		assert calls[0].args[0] == IRTemp("t0")

	def test_propagation_in_param(self):
		body = [
			IRCopy(dest=IRTemp("t1"), source=IRTemp("t0")),
			IRParam(value=IRTemp("t1")),
		]
		result = _propagate_only(body)
		params = [i for i in result if isinstance(i, IRParam)]
		assert params[0].value == IRTemp("t0")

	def test_propagation_in_load(self):
		body = [
			IRCopy(dest=IRTemp("t1"), source=IRTemp("t0")),
			IRLoad(dest=IRTemp("t2"), address=IRTemp("t1")),
		]
		result = _propagate_only(body)
		loads = [i for i in result if isinstance(i, IRLoad)]
		assert loads[0].address == IRTemp("t0")


# ── Combined Passes ──


class TestCombinedOptimizations:
	def test_fold_then_propagate_then_dce(self):
		"""Full pipeline: fold constants, propagate, eliminate dead code."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(10), op="+", right=IRConst(20)),
			IRCopy(dest=IRTemp("t1"), source=IRTemp("t0")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _opt(body)
		# t0 = 30 (folded), t1 = copy t0 propagated to 30, DCE removes dead copies
		assert len(result) == 1
		assert isinstance(result[0], IRReturn)
		assert result[0].value == IRConst(30)

	def test_idempotent(self):
		"""Running the optimizer twice yields the same result."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(1), op="+", right=IRConst(2)),
			IRCopy(dest=IRTemp("t1"), source=IRTemp("t0")),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t1"), op="*", right=IRConst(3)),
			IRReturn(value=IRTemp("t2")),
		]
		prog = _make_func(body)
		opt = IROptimizer()
		first = opt.optimize(prog)
		second = opt.optimize(first)
		assert first.functions[0].body == second.functions[0].body

	def test_preserves_side_effects(self):
		"""Calls and stores must survive optimization."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(1)),
			IRCall(dest=None, function_name="print", args=[IRTemp("t0")]),
			IRStore(address=IRTemp("ptr"), value=IRConst(42)),
			IRReturn(value=None),
		]
		result = _opt(body)
		assert any(isinstance(i, IRCall) for i in result)
		assert any(isinstance(i, IRStore) for i in result)

	def test_multiple_functions(self):
		"""Optimizer handles programs with multiple functions."""
		f1 = IRFunction(
			name="foo", params=[], return_type=IRType.INT,
			body=[
				IRBinOp(dest=IRTemp("t0"), left=IRConst(1), op="+", right=IRConst(2)),
				IRReturn(value=IRTemp("t0")),
			],
		)
		f2 = IRFunction(
			name="bar", params=[], return_type=IRType.INT,
			body=[
				IRCopy(dest=IRTemp("t0"), source=IRConst(99)),
				IRReturn(value=IRTemp("t0")),
			],
		)
		prog = IRProgram(functions=[f1, f2])
		result = IROptimizer().optimize(prog)
		assert len(result.functions) == 2
		# foo: 1+2=3 folded and propagated
		assert result.functions[0].body[0].value == IRConst(3)
		# bar: constant propagated into return
		assert result.functions[1].body[0].value == IRConst(99)

	def test_complex_cfg_preserved(self):
		"""Constant condition is folded: condjump becomes unconditional jump to true_label."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(1)),
			IRCondJump(condition=IRTemp("t0"), true_label="L1", false_label="L2"),
			IRLabelInstr(name="L1"),
			IRReturn(value=IRConst(1)),
			IRLabelInstr(name="L2"),
			IRReturn(value=IRConst(0)),
		]
		result = _opt(body)
		# Const propagated then condjump folded to unconditional jump
		assert not any(isinstance(i, IRCondJump) for i in result)
		jumps = [i for i in result if isinstance(i, IRJump)]
		assert any(j.target == "L1" for j in jumps)

	def test_empty_function(self):
		result = _opt([])
		assert result == []

	def test_load_not_eliminated(self):
		"""IRLoad may have side effects (memory read) and should not be DCE'd."""
		body = [
			IRLoad(dest=IRTemp("t0"), address=IRTemp("ptr")),
			IRReturn(value=None),
		]
		result = _opt(body)
		assert any(isinstance(i, IRLoad) for i in result)

	def test_nested_expressions(self):
		"""(2 + 3) * (4 + 5) should fold completely."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(2), op="+", right=IRConst(3)),
			IRBinOp(dest=IRTemp("t1"), left=IRConst(4), op="+", right=IRConst(5)),
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t0"), op="*", right=IRTemp("t1")),
			IRReturn(value=IRTemp("t2")),
		]
		result = _opt(body)
		assert len(result) == 1
		assert result[0].value == IRConst(45)

	def test_partial_fold(self):
		"""Only foldable parts are folded; non-const operands remain."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(3), op="+", right=IRConst(4)),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="*", right=IRTemp("x")),
			IRReturn(value=IRTemp("t1")),
		]
		result = _opt(body)
		# t0 folds to 7, propagated into t1 = 7 * x
		binops = [i for i in result if isinstance(i, IRBinOp)]
		assert len(binops) == 1
		assert binops[0].left == IRConst(7)
		assert binops[0].right == IRTemp("x")


# ── Strength Reduction (isolated) ──


class TestStrengthReduction:
	def test_multiply_by_power_of_2(self):
		"""a * 4 -> a << 2"""
		body = [IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="*", right=IRConst(4))]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "<<"
		assert result[0].left == IRTemp("a")
		assert result[0].right == IRConst(2)

	def test_multiply_by_power_of_2_left(self):
		"""4 * a -> a << 2 (commutative)"""
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(4), op="*", right=IRTemp("a"))]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "<<"
		assert result[0].left == IRTemp("a")
		assert result[0].right == IRConst(2)

	def test_multiply_by_8(self):
		"""a * 8 -> a << 3"""
		body = [IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="*", right=IRConst(8))]
		result = _strength_only(body)
		assert result[0].op == "<<"
		assert result[0].right == IRConst(3)

	def test_multiply_by_2(self):
		"""a * 2 -> a << 1"""
		body = [IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="*", right=IRConst(2))]
		result = _strength_only(body)
		assert result[0].op == "<<"
		assert result[0].right == IRConst(1)

	def test_multiply_by_0(self):
		"""a * 0 -> 0"""
		body = [IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="*", right=IRConst(0))]
		result = _strength_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(0)

	def test_multiply_by_0_left(self):
		"""0 * a -> 0"""
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(0), op="*", right=IRTemp("a"))]
		result = _strength_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRConst(0)

	def test_multiply_by_1(self):
		"""a * 1 -> copy a"""
		body = [IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="*", right=IRConst(1))]
		result = _strength_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRTemp("a")

	def test_multiply_by_1_left(self):
		"""1 * a -> copy a"""
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(1), op="*", right=IRTemp("a"))]
		result = _strength_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRTemp("a")

	def test_add_0_right(self):
		"""a + 0 -> copy a"""
		body = [IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="+", right=IRConst(0))]
		result = _strength_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRTemp("a")

	def test_add_0_left(self):
		"""0 + a -> copy a"""
		body = [IRBinOp(dest=IRTemp("t0"), left=IRConst(0), op="+", right=IRTemp("a"))]
		result = _strength_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRTemp("a")

	def test_subtract_0(self):
		"""a - 0 -> copy a"""
		body = [IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="-", right=IRConst(0))]
		result = _strength_only(body)
		assert isinstance(result[0], IRCopy)
		assert result[0].source == IRTemp("a")

	def test_non_power_of_2_preserved(self):
		"""a * 3 is not a power of 2 and should stay as multiply."""
		body = [IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="*", right=IRConst(3))]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "*"

	def test_non_matching_op_preserved(self):
		"""Division is not reduced."""
		body = [IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="/", right=IRConst(2))]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "/"

	def test_subtract_nonzero_preserved(self):
		"""a - 5 stays as subtract."""
		body = [IRBinOp(dest=IRTemp("t0"), left=IRTemp("a"), op="-", right=IRConst(5))]
		result = _strength_only(body)
		assert isinstance(result[0], IRBinOp)
		assert result[0].op == "-"


# ── Jump Threading (isolated) ──


class TestJumpThreading:
	def test_simple_thread(self):
		"""jump L1; L1: jump L2 -> jump L2"""
		body = [
			IRJump(target="L1"),
			IRLabelInstr(name="L1"),
			IRJump(target="L2"),
			IRLabelInstr(name="L2"),
			IRReturn(value=None),
		]
		result = _thread_only(body)
		assert result[0] == IRJump(target="L2")

	def test_transitive_thread(self):
		"""jump L1; L1: jump L2; L2: jump L3 -> jump L3"""
		body = [
			IRJump(target="L1"),
			IRLabelInstr(name="L1"),
			IRJump(target="L2"),
			IRLabelInstr(name="L2"),
			IRJump(target="L3"),
			IRLabelInstr(name="L3"),
			IRReturn(value=None),
		]
		result = _thread_only(body)
		assert result[0] == IRJump(target="L3")

	def test_condjump_true_threaded(self):
		"""if cond goto L1 else L2; L1: jump L3 -> if cond goto L3 else L2"""
		body = [
			IRCondJump(condition=IRTemp("c"), true_label="L1", false_label="L2"),
			IRLabelInstr(name="L1"),
			IRJump(target="L3"),
			IRLabelInstr(name="L2"),
			IRReturn(value=IRConst(0)),
			IRLabelInstr(name="L3"),
			IRReturn(value=IRConst(1)),
		]
		result = _thread_only(body)
		cj = result[0]
		assert isinstance(cj, IRCondJump)
		assert cj.true_label == "L3"
		assert cj.false_label == "L2"

	def test_condjump_false_threaded(self):
		"""Thread the false label of a conditional jump."""
		body = [
			IRCondJump(condition=IRTemp("c"), true_label="L1", false_label="L2"),
			IRLabelInstr(name="L1"),
			IRReturn(value=IRConst(1)),
			IRLabelInstr(name="L2"),
			IRJump(target="L3"),
			IRLabelInstr(name="L3"),
			IRReturn(value=IRConst(0)),
		]
		result = _thread_only(body)
		cj = result[0]
		assert isinstance(cj, IRCondJump)
		assert cj.true_label == "L1"
		assert cj.false_label == "L3"

	def test_condjump_both_threaded(self):
		"""Thread both labels of a conditional jump."""
		body = [
			IRCondJump(condition=IRTemp("c"), true_label="L1", false_label="L2"),
			IRLabelInstr(name="L1"),
			IRJump(target="L3"),
			IRLabelInstr(name="L2"),
			IRJump(target="L4"),
			IRLabelInstr(name="L3"),
			IRReturn(value=IRConst(1)),
			IRLabelInstr(name="L4"),
			IRReturn(value=IRConst(0)),
		]
		result = _thread_only(body)
		cj = result[0]
		assert isinstance(cj, IRCondJump)
		assert cj.true_label == "L3"
		assert cj.false_label == "L4"

	def test_no_thread_when_label_has_code(self):
		"""No threading when label is followed by a non-jump instruction."""
		body = [
			IRJump(target="L1"),
			IRLabelInstr(name="L1"),
			IRCopy(dest=IRTemp("t0"), source=IRConst(1)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _thread_only(body)
		assert result[0] == IRJump(target="L1")

	def test_self_loop_not_infinite(self):
		"""L1: jump L1 should not cause infinite loop in threading."""
		body = [
			IRJump(target="L1"),
			IRLabelInstr(name="L1"),
			IRJump(target="L1"),
		]
		result = _thread_only(body)
		assert result[0] == IRJump(target="L1")


# ── Unreachable Code Elimination (isolated) ──


class TestUnreachableElimination:
	def test_remove_after_jump(self):
		"""Code between jump and next label is removed."""
		body = [
			IRJump(target="L1"),
			IRCopy(dest=IRTemp("t0"), source=IRConst(42)),
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="+", right=IRConst(1)),
			IRLabelInstr(name="L1"),
			IRReturn(value=None),
		]
		result = _unreachable_only(body)
		assert len(result) == 3
		assert isinstance(result[0], IRJump)
		assert isinstance(result[1], IRLabelInstr)
		assert isinstance(result[2], IRReturn)

	def test_remove_after_return(self):
		"""Code between return and next label is removed."""
		body = [
			IRReturn(value=IRConst(0)),
			IRCopy(dest=IRTemp("t0"), source=IRConst(1)),
			IRLabelInstr(name="L1"),
			IRReturn(value=IRConst(1)),
		]
		result = _unreachable_only(body)
		assert len(result) == 3
		assert isinstance(result[0], IRReturn)
		assert isinstance(result[1], IRLabelInstr)
		assert isinstance(result[2], IRReturn)

	def test_no_removal_when_reachable(self):
		"""Code that is reachable is preserved."""
		body = [
			IRCopy(dest=IRTemp("t0"), source=IRConst(1)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _unreachable_only(body)
		assert len(result) == 2

	def test_multiple_unreachable_regions(self):
		"""Multiple separate unreachable regions are all removed."""
		body = [
			IRJump(target="L1"),
			IRCopy(dest=IRTemp("dead1"), source=IRConst(1)),
			IRLabelInstr(name="L1"),
			IRJump(target="L2"),
			IRCopy(dest=IRTemp("dead2"), source=IRConst(2)),
			IRLabelInstr(name="L2"),
			IRReturn(value=None),
		]
		result = _unreachable_only(body)
		assert len(result) == 5
		copies = [i for i in result if isinstance(i, IRCopy)]
		assert len(copies) == 0

	def test_label_restores_reachability(self):
		"""A label after unreachable code restores reachability."""
		body = [
			IRJump(target="L1"),
			IRCopy(dest=IRTemp("dead"), source=IRConst(1)),
			IRLabelInstr(name="L1"),
			IRCopy(dest=IRTemp("live"), source=IRConst(2)),
			IRReturn(value=IRTemp("live")),
		]
		result = _unreachable_only(body)
		copies = [i for i in result if isinstance(i, IRCopy)]
		assert len(copies) == 1
		assert copies[0].dest == IRTemp("live")


# ── New Combined Pass Tests ──


class TestCombinedNewPasses:
	def test_strength_reduction_with_constant_fold(self):
		"""Strength reduction + folding: const * power_of_2 folds directly."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRConst(5), op="*", right=IRConst(4)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _opt(body)
		assert len(result) == 1
		assert result[0].value == IRConst(20)

	def test_jump_threading_with_unreachable(self):
		"""Jump threading + unreachable elimination working together."""
		body = [
			IRCondJump(condition=IRTemp("c"), true_label="L1", false_label="L2"),
			IRLabelInstr(name="L1"),
			IRJump(target="L3"),
			IRLabelInstr(name="L2"),
			IRReturn(value=IRConst(0)),
			IRLabelInstr(name="L3"),
			IRReturn(value=IRConst(1)),
		]
		result = _opt(body)
		# L1 jumps to L3, so condjump should be threaded to L3
		cjs = [i for i in result if isinstance(i, IRCondJump)]
		assert cjs[0].true_label == "L3"

	def test_strength_plus_propagation(self):
		"""Strength reduction creates copies that propagation can resolve."""
		body = [
			IRBinOp(dest=IRTemp("t0"), left=IRTemp("x"), op="*", right=IRConst(1)),
			IRReturn(value=IRTemp("t0")),
		]
		result = _opt(body)
		# * 1 -> copy x, propagated into return, DCE removes copy
		assert len(result) == 1
		assert isinstance(result[0], IRReturn)
		assert result[0].value == IRTemp("x")

	def test_all_passes_combined(self):
		"""All optimization passes work together on a complex function."""
		body = [
			# Constant fold: 2 + 3 = 5
			IRBinOp(dest=IRTemp("t0"), left=IRConst(2), op="+", right=IRConst(3)),
			# Strength reduction: t0 * 4 -> t0 << 2, then folded to 5 << 2 = 20
			IRBinOp(dest=IRTemp("t1"), left=IRTemp("t0"), op="*", right=IRConst(4)),
			# Add 0 -> copy
			IRBinOp(dest=IRTemp("t2"), left=IRTemp("t1"), op="+", right=IRConst(0)),
			IRReturn(value=IRTemp("t2")),
		]
		result = _opt(body)
		assert len(result) == 1
		assert result[0].value == IRConst(20)

	def test_unreachable_after_early_return(self):
		"""Unreachable elimination removes dead code after early return."""
		body = [
			IRReturn(value=IRConst(42)),
			IRBinOp(dest=IRTemp("t0"), left=IRConst(1), op="+", right=IRConst(2)),
			IRCopy(dest=IRTemp("t1"), source=IRTemp("t0")),
		]
		result = _opt(body)
		assert len(result) == 1
		assert isinstance(result[0], IRReturn)
		assert result[0].value == IRConst(42)
