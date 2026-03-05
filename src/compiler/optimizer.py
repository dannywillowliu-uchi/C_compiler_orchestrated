"""IR optimization passes: constant folding, dead code elimination, copy propagation,
strength reduction, jump threading, unreachable code elimination, common subexpression
elimination, loop-invariant code motion, and dead store elimination."""

from __future__ import annotations

from compiler.cfg import CFG
from compiler.ir import (
	IRAddrOf,
	IRAlloc,
	IRBinOp,
	IRCall,
	IRCondJump,
	IRConst,
	IRConvert,
	IRCopy,
	IRFloatConst,
	IRFunction,
	IRInstruction,
	IRJump,
	IRLoad,
	IRParam,
	IRProgram,
	IRReturn,
	IRStore,
	IRTemp,
	IRUnaryOp,
	IRVaArg,
	IRVaCopy,
	IRVaEnd,
	IRVaStart,
	IRValue,
	IRLabelInstr,
	IRType,
	ir_type_byte_width,
	ir_type_is_integer,
)
from compiler.liveness import LivenessAnalyzer, _defined_temp, _used_temps


class IROptimizer:
	"""Optimizes an IRProgram through constant folding, copy propagation, and DCE."""

	def optimize(self, program: IRProgram) -> IRProgram:
		"""Apply optimization passes to all functions until convergence."""
		return IRProgram(functions=[self._optimize_function(f) for f in program.functions])

	def _optimize_function(self, func: IRFunction) -> IRFunction:
		"""Apply all passes in a loop until no changes occur."""
		body = list(func.body)
		changed = True
		while changed:
			changed = False
			for pass_fn in (
				self._constant_fold,
				self._algebraic_simplification,
				self._convert_const_propagation,
				self._boolean_simplification,
				self._strength_reduction,
				self._copy_propagation,
				self._convert_elimination,
				self._cse,
				self._constant_condjump_folding,
				self._dead_store_elimination,
				self._dead_code_elimination,
				self._jump_threading,
				self._unreachable_elimination,
				self._licm,
			):
				new_body = pass_fn(body)
				if new_body != body:
					changed = True
					body = new_body
		return IRFunction(
			name=func.name, params=func.params, body=body, return_type=func.return_type,
			param_types=func.param_types, storage_class=func.storage_class,
		)

	# -- Constant Folding --

	def _constant_fold(self, body: list[IRInstruction]) -> list[IRInstruction]:
		"""Replace operations on two constants with the computed result."""
		result: list[IRInstruction] = []
		for instr in body:
			if isinstance(instr, IRBinOp):
				left_const = isinstance(instr.left, (IRConst, IRFloatConst))
				right_const = isinstance(instr.right, (IRConst, IRFloatConst))
				if left_const and right_const:
					left_is_float = isinstance(instr.left, IRFloatConst)
					right_is_float = isinstance(instr.right, IRFloatConst)
					if left_is_float or right_is_float:
						lv = float(instr.left.value)
						rv = float(instr.right.value)
						ir_type = (
							instr.left.ir_type if left_is_float
							else instr.right.ir_type
						)
						folded = self._eval_float_binop(lv, instr.op, rv)
						if folded is not None:
							if isinstance(folded, int):
								result.append(IRCopy(dest=instr.dest, source=IRConst(folded)))
							else:
								result.append(IRCopy(dest=instr.dest, source=IRFloatConst(folded, ir_type=ir_type)))
							continue
					else:
						folded = self._eval_binop(instr.left.value, instr.op, instr.right.value)
						if folded is not None:
							result.append(IRCopy(dest=instr.dest, source=IRConst(folded)))
							continue
			elif isinstance(instr, IRUnaryOp):
				if isinstance(instr.operand, IRFloatConst):
					folded = self._eval_float_unaryop(instr.op, instr.operand.value)
					if folded is not None:
						result.append(IRCopy(dest=instr.dest, source=IRFloatConst(folded, ir_type=instr.operand.ir_type)))
						continue
				elif isinstance(instr.operand, IRConst):
					folded = self._eval_unaryop(instr.op, instr.operand.value)
					if folded is not None:
						result.append(IRCopy(dest=instr.dest, source=IRConst(folded)))
						continue
			result.append(instr)
		return result

	def _eval_binop(self, left: int, op: str, right: int) -> int | None:
		"""Evaluate a binary operation on two integer constants."""
		if op == "/" and right != 0:
			return int(left / right) if left / right == left // right else left // right
		if op == "%" and right != 0:
			return left % right
		ops: dict[str, object] = {
			"+": lambda a, b: a + b,
			"-": lambda a, b: a - b,
			"*": lambda a, b: a * b,
			"&": lambda a, b: a & b,
			"|": lambda a, b: a | b,
			"^": lambda a, b: a ^ b,
			"<<": lambda a, b: a << b,
			">>": lambda a, b: a >> b,
			"<": lambda a, b: int(a < b),
			">": lambda a, b: int(a > b),
			"<=": lambda a, b: int(a <= b),
			">=": lambda a, b: int(a >= b),
			"==": lambda a, b: int(a == b),
			"!=": lambda a, b: int(a != b),
			"&&": lambda a, b: int(bool(a) and bool(b)),
			"||": lambda a, b: int(bool(a) or bool(b)),
		}
		fn = ops.get(op)
		if fn is not None:
			return fn(left, right)  # type: ignore[operator]
		return None

	def _eval_unaryop(self, op: str, operand: int) -> int | None:
		"""Evaluate a unary operation on an integer constant."""
		if op == "-":
			return -operand
		if op == "~":
			return ~operand
		if op == "!":
			return int(not operand)
		return None

	def _eval_float_binop(self, left: float, op: str, right: float) -> float | int | None:
		"""Evaluate a binary operation on float constants. Comparisons return int."""
		if op == "/" and right == 0.0:
			return None
		arithmetic: dict[str, object] = {
			"+": lambda a, b: a + b,
			"-": lambda a, b: a - b,
			"*": lambda a, b: a * b,
			"/": lambda a, b: a / b,
		}
		fn = arithmetic.get(op)
		if fn is not None:
			return fn(left, right)  # type: ignore[operator]
		comparisons: dict[str, object] = {
			"<": lambda a, b: int(a < b),
			">": lambda a, b: int(a > b),
			"<=": lambda a, b: int(a <= b),
			">=": lambda a, b: int(a >= b),
			"==": lambda a, b: int(a == b),
			"!=": lambda a, b: int(a != b),
		}
		fn = comparisons.get(op)
		if fn is not None:
			return fn(left, right)  # type: ignore[operator]
		return None

	def _eval_float_unaryop(self, op: str, operand: float) -> float | None:
		"""Evaluate a unary operation on a float constant."""
		if op == "-":
			return -operand
		return None

	# -- Algebraic Simplification --

	def _algebraic_simplification(self, body: list[IRInstruction]) -> list[IRInstruction]:
		"""Simplify binary operations using algebraic identities.

		Handles identity elements, absorption, and idempotent patterns where
		one operand is a constant or both operands are the same temp.
		"""
		result: list[IRInstruction] = []
		for instr in body:
			if isinstance(instr, IRBinOp):
				simplified = self._simplify_algebraic(instr)
				if simplified is not None:
					result.append(simplified)
					continue
			result.append(instr)
		return result

	def _simplify_algebraic(self, instr: IRBinOp) -> IRInstruction | None:
		"""Try to simplify a binary op using algebraic identities."""
		left, op, right = instr.left, instr.op, instr.right

		# --- Identity with constant on right: x OP const ---
		if isinstance(right, IRConst):
			rv = right.value
			# x + 0 -> x
			if op == "+" and rv == 0:
				return IRCopy(dest=instr.dest, source=left)
			# x - 0 -> x
			if op == "-" and rv == 0:
				return IRCopy(dest=instr.dest, source=left)
			# x * 1 -> x
			if op == "*" and rv == 1:
				return IRCopy(dest=instr.dest, source=left)
			# x * 0 -> 0
			if op == "*" and rv == 0:
				return IRCopy(dest=instr.dest, source=IRConst(0))
			# x & 0 -> 0
			if op == "&" and rv == 0:
				return IRCopy(dest=instr.dest, source=IRConst(0))
			# x | 0 -> x
			if op == "|" and rv == 0:
				return IRCopy(dest=instr.dest, source=left)
			# x ^ 0 -> x
			if op == "^" and rv == 0:
				return IRCopy(dest=instr.dest, source=left)
			# x << 0 -> x
			if op == "<<" and rv == 0:
				return IRCopy(dest=instr.dest, source=left)
			# x >> 0 -> x
			if op == ">>" and rv == 0:
				return IRCopy(dest=instr.dest, source=left)

		# --- Identity with constant on left: const OP x ---
		if isinstance(left, IRConst):
			lv = left.value
			# 0 + x -> x
			if op == "+" and lv == 0:
				return IRCopy(dest=instr.dest, source=right)
			# 1 * x -> x
			if op == "*" and lv == 1:
				return IRCopy(dest=instr.dest, source=right)
			# 0 * x -> 0
			if op == "*" and lv == 0:
				return IRCopy(dest=instr.dest, source=IRConst(0))
			# 0 & x -> 0
			if op == "&" and lv == 0:
				return IRCopy(dest=instr.dest, source=IRConst(0))
			# 0 | x -> x
			if op == "|" and lv == 0:
				return IRCopy(dest=instr.dest, source=right)
			# 0 ^ x -> x
			if op == "^" and lv == 0:
				return IRCopy(dest=instr.dest, source=right)

		# --- Same-operand patterns: x OP x ---
		if isinstance(left, IRTemp) and isinstance(right, IRTemp) and left.name == right.name:
			# x - x -> 0
			if op == "-":
				return IRCopy(dest=instr.dest, source=IRConst(0))
			# x & x -> x
			if op == "&":
				return IRCopy(dest=instr.dest, source=left)
			# x | x -> x
			if op == "|":
				return IRCopy(dest=instr.dest, source=left)
			# x ^ x -> 0
			if op == "^":
				return IRCopy(dest=instr.dest, source=IRConst(0))

		return None

	# -- Redundant Convert Elimination --

	def _convert_elimination(self, body: list[IRInstruction]) -> list[IRInstruction]:
		"""Eliminate redundant IRConvert instructions."""
		result: list[IRInstruction] = []
		# Track converts: dest_name -> (original_source, from_type, to_type)
		convert_map: dict[str, tuple[IRValue, IRType, IRType]] = {}

		for instr in body:
			if isinstance(instr, IRLabelInstr):
				convert_map.clear()
				result.append(instr)
				continue

			if isinstance(instr, IRConvert):
				# Pattern 1: No-op conversion (same type)
				if instr.from_type == instr.to_type:
					result.append(IRCopy(dest=instr.dest, source=instr.source))
					continue

				# Check for chained converts
				if isinstance(instr.source, IRTemp) and instr.source.name in convert_map:
					orig_source, orig_from, intermediate_to = convert_map[instr.source.name]
					# Pattern 2: Round-trip with wider intermediate -> use original value
					if (
						instr.to_type == orig_from
						and ir_type_byte_width(intermediate_to) >= ir_type_byte_width(orig_from)
					):
						result.append(IRCopy(dest=instr.dest, source=orig_source))
						convert_map[instr.dest.name] = (orig_source, orig_from, instr.to_type)
						continue
					# Pattern 3: Collapse chained converts into a single convert
					result.append(IRConvert(
						dest=instr.dest, source=orig_source,
						from_type=orig_from, to_type=instr.to_type,
					))
					convert_map[instr.dest.name] = (orig_source, orig_from, instr.to_type)
					continue

				convert_map[instr.dest.name] = (instr.source, instr.from_type, instr.to_type)
				result.append(instr)
				continue

			# Kill mappings for any redefined temp
			dest = self._get_dest(instr)
			if dest is not None:
				convert_map.pop(dest.name, None)

			result.append(instr)

		return result

	# -- Copy Propagation --

	def _copy_propagation(self, body: list[IRInstruction]) -> list[IRInstruction]:
		"""When t2 = copy t1, replace subsequent uses of t2 with t1."""
		copies: dict[str, IRValue] = {}
		result: list[IRInstruction] = []
		for instr in body:
			# Conservative: clear at labels (join points from jumps)
			if isinstance(instr, IRLabelInstr):
				copies.clear()
				result.append(instr)
				continue

			# Substitute uses first
			new_instr = self._substitute(instr, copies)

			# Kill mappings for any redefined temp
			dest = self._get_dest(new_instr)
			if dest is not None:
				copies.pop(dest.name, None)
				to_remove = [
					k for k, v in copies.items()
					if isinstance(v, IRTemp) and v.name == dest.name
				]
				for k in to_remove:
					del copies[k]

			# Track new copy relationships
			if isinstance(new_instr, IRCopy) and isinstance(new_instr.source, (IRTemp, IRConst, IRFloatConst)):
				source = new_instr.source
				if isinstance(source, IRTemp) and source.name in copies:
					source = copies[source.name]
				copies[new_instr.dest.name] = source

			result.append(new_instr)
		return result

	def _substitute(self, instr: IRInstruction, copies: dict[str, IRValue]) -> IRInstruction:
		"""Replace IRTemp operands according to the copy map."""
		def resolve(val: IRValue) -> IRValue:
			if isinstance(val, IRTemp) and val.name in copies:
				return copies[val.name]
			return val

		if isinstance(instr, IRBinOp):
			nl, nr = resolve(instr.left), resolve(instr.right)
			if nl is not instr.left or nr is not instr.right:
				return IRBinOp(dest=instr.dest, left=nl, op=instr.op, right=nr)
		elif isinstance(instr, IRUnaryOp):
			no = resolve(instr.operand)
			if no is not instr.operand:
				return IRUnaryOp(dest=instr.dest, op=instr.op, operand=no)
		elif isinstance(instr, IRCopy):
			ns = resolve(instr.source)
			if ns is not instr.source:
				return IRCopy(dest=instr.dest, source=ns)
		elif isinstance(instr, IRLoad):
			na = resolve(instr.address)
			if na is not instr.address:
				return IRLoad(dest=instr.dest, address=na)
		elif isinstance(instr, IRStore):
			na, nv = resolve(instr.address), resolve(instr.value)
			if na is not instr.address or nv is not instr.value:
				return IRStore(address=na, value=nv)
		elif isinstance(instr, IRCondJump):
			nc = resolve(instr.condition)
			if nc is not instr.condition:
				return IRCondJump(condition=nc, true_label=instr.true_label, false_label=instr.false_label)
		elif isinstance(instr, IRCall):
			new_args = [resolve(a) for a in instr.args]
			if any(new_args[i] is not instr.args[i] for i in range(len(new_args))):
				return IRCall(dest=instr.dest, function_name=instr.function_name, args=new_args)
		elif isinstance(instr, IRReturn):
			if instr.value is not None:
				nv = resolve(instr.value)
				if nv is not instr.value:
					return IRReturn(value=nv)
		elif isinstance(instr, IRParam):
			nv = resolve(instr.value)
			if nv is not instr.value:
				return IRParam(value=nv)
		elif isinstance(instr, IRConvert):
			ns = resolve(instr.source)
			if ns is not instr.source:
				return IRConvert(dest=instr.dest, source=ns, from_type=instr.from_type, to_type=instr.to_type)
		return instr

	# -- Dead Code Elimination --

	def _dead_code_elimination(self, body: list[IRInstruction]) -> list[IRInstruction]:
		"""Remove pure instructions whose dest is never read."""
		used: set[str] = set()
		for instr in body:
			for val in self._get_uses(instr):
				if isinstance(val, IRTemp):
					used.add(val.name)
		return [
			instr for instr in body
			if not (isinstance(instr, (IRBinOp, IRUnaryOp, IRCopy, IRConvert)) and instr.dest.name not in used)
		]

	# -- Strength Reduction --

	def _strength_reduction(self, body: list[IRInstruction]) -> list[IRInstruction]:
		"""Replace expensive operations with cheaper equivalents."""
		result: list[IRInstruction] = []
		for instr in body:
			if isinstance(instr, IRBinOp):
				reduced = self._reduce_binop(instr)
				if reduced is not None:
					result.append(reduced)
					continue
			result.append(instr)
		return result

	def _reduce_binop(self, instr: IRBinOp) -> IRInstruction | None:
		"""Try to reduce a binary operation to a cheaper form."""
		left, op, right = instr.left, instr.op, instr.right
		if op == "*":
			# * 0 -> 0
			if isinstance(right, IRConst) and right.value == 0:
				return IRCopy(dest=instr.dest, source=IRConst(0))
			if isinstance(left, IRConst) and left.value == 0:
				return IRCopy(dest=instr.dest, source=IRConst(0))
			# * 1 -> copy
			if isinstance(right, IRConst) and right.value == 1:
				return IRCopy(dest=instr.dest, source=left)
			if isinstance(left, IRConst) and left.value == 1:
				return IRCopy(dest=instr.dest, source=right)
			# * power_of_2 -> left shift
			if isinstance(right, IRConst) and right.value > 1 and (right.value & (right.value - 1)) == 0:
				return IRBinOp(dest=instr.dest, left=left, op="<<", right=IRConst(right.value.bit_length() - 1))
			if isinstance(left, IRConst) and left.value > 1 and (left.value & (left.value - 1)) == 0:
				return IRBinOp(dest=instr.dest, left=right, op="<<", right=IRConst(left.value.bit_length() - 1))
		elif op == "/":
			# x / 1 -> x
			if isinstance(right, IRConst) and right.value == 1:
				return IRCopy(dest=instr.dest, source=left)
			# unsigned x / power_of_2 -> right shift
			if isinstance(right, IRConst) and right.value > 1 and (right.value & (right.value - 1)) == 0:
				shift = right.value.bit_length() - 1
				return IRBinOp(dest=instr.dest, left=left, op=">>", right=IRConst(shift))
		elif op == "%":
			# x % 1 -> 0
			if isinstance(right, IRConst) and right.value == 1:
				return IRCopy(dest=instr.dest, source=IRConst(0))
			# unsigned x % power_of_2 -> bitwise AND with (power_of_2 - 1)
			if isinstance(right, IRConst) and right.value > 1 and (right.value & (right.value - 1)) == 0:
				mask = right.value - 1
				return IRBinOp(dest=instr.dest, left=left, op="&", right=IRConst(mask))
		elif op == "+":
			if isinstance(right, IRConst) and right.value == 0:
				return IRCopy(dest=instr.dest, source=left)
			if isinstance(left, IRConst) and left.value == 0:
				return IRCopy(dest=instr.dest, source=right)
		elif op == "-":
			if isinstance(right, IRConst) and right.value == 0:
				return IRCopy(dest=instr.dest, source=left)
		return None

	# -- Jump Threading --

	def _jump_threading(self, body: list[IRInstruction]) -> list[IRInstruction]:
		"""Redirect jumps through labels that immediately jump elsewhere."""
		changed = True
		while changed:
			changed = False
			# Build map: label -> final target (for labels followed by unconditional jump)
			label_target: dict[str, str] = {}
			for i, instr in enumerate(body):
				if isinstance(instr, IRLabelInstr) and i + 1 < len(body) and isinstance(body[i + 1], IRJump):
					label_target[instr.name] = body[i + 1].target
			if not label_target:
				break

			def resolve(label: str) -> str:
				visited: set[str] = set()
				while label in label_target and label not in visited:
					visited.add(label)
					label = label_target[label]
				return label

			new_body: list[IRInstruction] = []
			for instr in body:
				if isinstance(instr, IRJump):
					new_target = resolve(instr.target)
					if new_target != instr.target:
						new_body.append(IRJump(target=new_target))
						changed = True
						continue
				elif isinstance(instr, IRCondJump):
					new_true = resolve(instr.true_label)
					new_false = resolve(instr.false_label)
					if new_true != instr.true_label or new_false != instr.false_label:
						new_body.append(IRCondJump(
							condition=instr.condition, true_label=new_true, false_label=new_false,
						))
						changed = True
						continue
				new_body.append(instr)
			body = new_body
		return body

	# -- Unreachable Code Elimination --

	def _unreachable_elimination(self, body: list[IRInstruction]) -> list[IRInstruction]:
		"""Remove instructions between an unconditional jump/return and the next label."""
		result: list[IRInstruction] = []
		unreachable = False
		for instr in body:
			if isinstance(instr, IRLabelInstr):
				unreachable = False
				result.append(instr)
			elif unreachable:
				continue
			else:
				result.append(instr)
				if isinstance(instr, (IRJump, IRReturn)):
					unreachable = True
		return result

	# -- Constant Conditional Jump Folding --

	def _constant_condjump_folding(self, body: list[IRInstruction]) -> list[IRInstruction]:
		"""Replace conditional jumps on constant conditions with unconditional jumps."""
		result: list[IRInstruction] = []
		for instr in body:
			if isinstance(instr, IRCondJump) and isinstance(instr.condition, (IRConst, IRFloatConst)):
				if isinstance(instr.condition, IRFloatConst):
					is_true = instr.condition.value != 0.0
				else:
					is_true = instr.condition.value != 0
				if is_true:
					result.append(IRJump(target=instr.true_label))
				else:
					result.append(IRJump(target=instr.false_label))
				continue
			result.append(instr)
		return result

	# -- Common Subexpression Elimination --

	def _cse(self, body: list[IRInstruction]) -> list[IRInstruction]:
		"""Replace duplicate computations with a copy of the first result."""
		available: dict[tuple, IRTemp] = {}
		result: list[IRInstruction] = []

		for instr in body:
			if isinstance(instr, IRLabelInstr):
				available.clear()
				result.append(instr)
				continue

			if isinstance(instr, IRStore):
				available.clear()
				result.append(instr)
				continue

			expr_key = self._expr_key(instr)
			if expr_key is not None and expr_key in available:
				result.append(IRCopy(dest=instr.dest, source=available[expr_key]))
				continue

			dest = self._get_dest(instr)
			if dest is not None:
				self._invalidate_cse(available, dest.name)

			if expr_key is not None:
				uses = self._get_uses(instr)
				if not any(isinstance(u, IRTemp) and u.name == dest.name for u in uses):
					available[expr_key] = instr.dest

			result.append(instr)

		return result

	@staticmethod
	def _expr_key(instr: IRInstruction) -> tuple | None:
		"""Return a hashable key for the expression, or None if not CSE-able."""
		if isinstance(instr, IRBinOp):
			return ("binop", instr.op, instr.left, instr.right)
		if isinstance(instr, IRUnaryOp):
			return ("unaryop", instr.op, instr.operand)
		return None

	@staticmethod
	def _invalidate_cse(available: dict[tuple, IRTemp], name: str) -> None:
		"""Remove entries whose dest or operands reference the given temp name."""
		def _refs_name(key: tuple) -> bool:
			for part in key[2:]:
				if isinstance(part, IRTemp) and part.name == name:
					return True
			return False

		to_remove = [
			key for key, dest in available.items()
			if dest.name == name or _refs_name(key)
		]
		for key in to_remove:
			del available[key]

	# -- Loop-Invariant Code Motion --

	def _licm(self, body: list[IRInstruction]) -> list[IRInstruction]:
		"""Hoist loop-invariant computations above loop headers using CFG natural loop detection.

		A computation is loop-invariant if all its operands are either constants,
		defined outside the loop, or themselves loop-invariant (transitive).
		Only pure instructions (IRBinOp, IRUnaryOp, IRCopy, IRConvert) are candidates.
		"""
		if not body:
			return body

		cfg = CFG(body)
		loops = cfg.find_natural_loops()
		if not loops:
			return body

		# Collect all instructions per block label
		block_instrs: dict[str, list[IRInstruction]] = {}
		for block in cfg.blocks():
			block_instrs[block.label] = block.instructions

		# Process one loop at a time; return early on first change to re-enter
		for loop in loops:
			# Gather all definitions inside the loop
			loop_defs: dict[str, list[IRInstruction]] = {}
			for label in loop.body:
				for instr in block_instrs.get(label, []):
					d = self._get_dest(instr)
					if d is not None:
						loop_defs.setdefault(d.name, []).append(instr)

			# An operand is invariant if it is a constant, defined outside the loop,
			# or defined exactly once inside the loop by a loop-invariant instruction.
			invariant_set: set[int] = set()  # set of id(instr) for invariant instrs

			def _is_operand_invariant(val: IRValue) -> bool:
				if isinstance(val, (IRConst, IRFloatConst)):
					return True
				if isinstance(val, IRTemp):
					if val.name not in loop_defs:
						return True
					defs = loop_defs[val.name]
					if len(defs) == 1 and id(defs[0]) in invariant_set:
						return True
				return False

			# Fixed-point iteration to find all loop-invariant instructions
			changed = True
			while changed:
				changed = False
				for label in loop.body:
					for instr in block_instrs.get(label, []):
						if id(instr) in invariant_set:
							continue
						# Only hoist pure computations
						if not isinstance(instr, (IRBinOp, IRUnaryOp, IRCopy, IRConvert)):
							continue
						# Destination must be defined exactly once in the loop
						d = self._get_dest(instr)
						if d is not None and len(loop_defs.get(d.name, [])) != 1:
							continue
						operands = self._get_uses(instr)
						if all(_is_operand_invariant(op) for op in operands):
							invariant_set.add(id(instr))
							changed = True

			if not invariant_set:
				continue

			# Hoist: place invariant instructions before the loop header label
			header_label = loop.header
			hoisted = [instr for instr in body if id(instr) in invariant_set]
			hoist_ids = invariant_set

			new_body: list[IRInstruction] = []
			for instr in body:
				if isinstance(instr, IRLabelInstr) and instr.name == header_label:
					new_body.extend(hoisted)
				if id(instr) not in hoist_ids:
					new_body.append(instr)

			return new_body

		return body

	# -- Constant Propagation through IRConvert --

	def _convert_const_propagation(self, body: list[IRInstruction]) -> list[IRInstruction]:
		"""Fold IRConvert of constant values into the resulting constant."""
		result: list[IRInstruction] = []
		for instr in body:
			if isinstance(instr, IRConvert):
				folded = self._fold_convert(instr)
				if folded is not None:
					result.append(folded)
					continue
			result.append(instr)
		return result

	def _fold_convert(self, instr: IRConvert) -> IRCopy | None:
		"""Try to fold a convert of a constant into a copy of the result constant."""
		source = instr.source
		to_type = instr.to_type

		if isinstance(source, IRConst):
			val = source.value
			if ir_type_is_integer(to_type):
				truncated = self._truncate_int(val, to_type)
				return IRCopy(dest=instr.dest, source=IRConst(truncated, ir_type=to_type))
			if to_type in (IRType.FLOAT, IRType.DOUBLE):
				return IRCopy(dest=instr.dest, source=IRFloatConst(float(val), ir_type=to_type))
		elif isinstance(source, IRFloatConst):
			val = source.value
			if ir_type_is_integer(to_type):
				truncated = self._truncate_int(int(val), to_type)
				return IRCopy(dest=instr.dest, source=IRConst(truncated, ir_type=to_type))
			if to_type in (IRType.FLOAT, IRType.DOUBLE):
				return IRCopy(dest=instr.dest, source=IRFloatConst(val, ir_type=to_type))
		return None

	@staticmethod
	def _truncate_int(value: int, ir_type: IRType) -> int:
		"""Truncate an integer value to fit the given IR integer type."""
		widths = {
			IRType.BOOL: 1,
			IRType.CHAR: 8,
			IRType.SHORT: 16,
			IRType.INT: 32,
			IRType.LONG: 64,
		}
		bits = widths.get(ir_type)
		if bits is None:
			return value
		if ir_type == IRType.BOOL:
			return 1 if value else 0
		mask = (1 << bits) - 1
		truncated = value & mask
		# Sign-extend for signed types
		if truncated >= (1 << (bits - 1)):
			truncated -= (1 << bits)
		return truncated

	# -- Boolean Simplification --

	def _boolean_simplification(self, body: list[IRInstruction]) -> list[IRInstruction]:
		"""Simplify boolean operations: comparisons with 0/1, double negation, logical ops with constants."""
		# Track which temps are negations: temp_name -> negated_operand
		negation_map: dict[str, IRValue] = {}
		result: list[IRInstruction] = []

		for instr in body:
			if isinstance(instr, IRLabelInstr):
				negation_map.clear()
				result.append(instr)
				continue

			# Double negation elimination: !!x -> x (when x is boolean-valued)
			if isinstance(instr, IRUnaryOp) and instr.op == "!":
				if isinstance(instr.operand, IRTemp) and instr.operand.name in negation_map:
					original = negation_map[instr.operand.name]
					result.append(IRCopy(dest=instr.dest, source=original))
					continue

			# Simplify comparisons with 0
			if isinstance(instr, IRBinOp) and isinstance(instr.right, IRConst):
				simplified = self._simplify_bool_binop(instr)
				if simplified is not None:
					# Track if the simplified result is a negation
					if isinstance(simplified, IRUnaryOp) and simplified.op == "!":
						negation_map[simplified.dest.name] = simplified.operand
					else:
						dest = self._get_dest(simplified)
						if dest is not None:
							negation_map.pop(dest.name, None)
					result.append(simplified)
					continue

			# Simplify comparisons with 0 on left side
			if isinstance(instr, IRBinOp) and isinstance(instr.left, IRConst):
				simplified = self._simplify_bool_binop_left(instr)
				if simplified is not None:
					if isinstance(simplified, IRUnaryOp) and simplified.op == "!":
						negation_map[simplified.dest.name] = simplified.operand
					else:
						dest = self._get_dest(simplified)
						if dest is not None:
							negation_map.pop(dest.name, None)
					result.append(simplified)
					continue

			# Kill mappings for redefined temps, then track negations
			dest = self._get_dest(instr)
			if dest is not None:
				negation_map.pop(dest.name, None)

			# Track negations after killing old mappings
			if isinstance(instr, IRUnaryOp) and instr.op == "!":
				negation_map[instr.dest.name] = instr.operand

			result.append(instr)
		return result

	def _simplify_bool_binop(self, instr: IRBinOp) -> IRInstruction | None:
		"""Simplify binary ops where the right operand is a constant 0 or 1."""
		right_val = instr.right.value if isinstance(instr.right, IRConst) else None
		if right_val is None:
			return None

		op = instr.op
		# x == 0 -> !x
		if op == "==" and right_val == 0:
			return IRUnaryOp(dest=instr.dest, op="!", operand=instr.left)
		# x != 0 -> !!x (will be folded by double negation if already boolean)
		# Actually just leave x != 0 as-is since it's not simpler
		# x && 0 -> 0
		if op == "&&" and right_val == 0:
			return IRCopy(dest=instr.dest, source=IRConst(0))
		# x && 1 -> !!x (convert to boolean) - but simpler: !(!x)
		# Actually x && 1 -> x != 0, leave as unary !(!x) via later passes
		# x || 0 -> x (as boolean: x != 0, but keep as copy if already boolean)
		if op == "||" and right_val == 0:
			return IRCopy(dest=instr.dest, source=instr.left)
		# x || 1 -> 1
		if op == "||" and right_val == 1:
			return IRCopy(dest=instr.dest, source=IRConst(1))
		# x && 1 -> x (preserves boolean value since && already produces 0/1)
		if op == "&&" and right_val == 1:
			return IRCopy(dest=instr.dest, source=instr.left)
		return None

	def _simplify_bool_binop_left(self, instr: IRBinOp) -> IRInstruction | None:
		"""Simplify binary ops where the left operand is a constant 0 or 1."""
		left_val = instr.left.value if isinstance(instr.left, IRConst) else None
		if left_val is None:
			return None

		op = instr.op
		# 0 == x -> !x
		if op == "==" and left_val == 0:
			return IRUnaryOp(dest=instr.dest, op="!", operand=instr.right)
		# 0 && x -> 0
		if op == "&&" and left_val == 0:
			return IRCopy(dest=instr.dest, source=IRConst(0))
		# 1 && x -> x
		if op == "&&" and left_val == 1:
			return IRCopy(dest=instr.dest, source=instr.right)
		# 0 || x -> x
		if op == "||" and left_val == 0:
			return IRCopy(dest=instr.dest, source=instr.right)
		# 1 || x -> 1
		if op == "||" and left_val == 1:
			return IRCopy(dest=instr.dest, source=IRConst(1))
		return None

	# -- Dead Store Elimination --

	def _dead_store_elimination(self, body: list[IRInstruction]) -> list[IRInstruction]:
		"""Eliminate stores/copies whose destination is dead (not live after the instruction).

		Uses CFG-based liveness analysis for precise per-instruction liveness, and
		within-block forward analysis to find IRStore instructions that are overwritten
		before being read.
		"""
		if not body:
			return body

		cfg = CFG(body)
		analyzer = LivenessAnalyzer(cfg)
		liveness = analyzer.compute_liveness()

		dead_ids: set[int] = set()

		# Phase 1: Liveness-based elimination of pure definitions.
		# A definition is dead if the dest temp is not live after the instruction.
		for block in cfg.blocks():
			instrs = block.instructions
			live = set(liveness[block.label][1])  # live_out

			for i in range(len(instrs) - 1, -1, -1):
				instr = instrs[i]
				defined = _defined_temp(instr)
				used = _used_temps(instr)

				if defined is not None and defined not in live:
					if isinstance(instr, (IRCopy, IRBinOp, IRUnaryOp, IRConvert)):
						dead_ids.add(id(instr))
						continue

				if defined is not None:
					live.discard(defined)
				live |= used

		# Phase 2: Within-block IRStore elimination.
		# If the same address is stored to again without an intervening load, call,
		# or address-taking operation, the earlier store is dead.

		# Find temps whose address is taken (aliased) anywhere in the function.
		aliased: set[str] = set()
		for instr in body:
			if isinstance(instr, IRAddrOf) and isinstance(instr.source, IRTemp):
				aliased.add(instr.source.name)

		for block in cfg.blocks():
			instrs = block.instructions
			pending_stores: dict[str, int] = {}

			for i, instr in enumerate(instrs):
				if isinstance(instr, IRStore) and isinstance(instr.address, IRTemp):
					addr = instr.address.name
					if addr not in aliased and addr in pending_stores:
						dead_ids.add(id(instrs[pending_stores[addr]]))
					if addr not in aliased:
						pending_stores[addr] = i
				elif isinstance(instr, IRLoad):
					if isinstance(instr.address, IRTemp):
						pending_stores.pop(instr.address.name, None)
					else:
						pending_stores.clear()
				elif isinstance(instr, (IRCall, IRVaStart, IRVaArg, IRVaEnd, IRVaCopy, IRAddrOf)):
					pending_stores.clear()

		if not dead_ids:
			return body

		return [instr for instr in body if id(instr) not in dead_ids]

	# -- Helpers --

	def _get_dest(self, instr: IRInstruction) -> IRTemp | None:
		"""Return the destination temp written by an instruction, if any."""
		if isinstance(instr, (IRAddrOf, IRBinOp, IRUnaryOp, IRCopy, IRLoad, IRAlloc, IRConvert, IRVaArg)):
			return instr.dest
		if isinstance(instr, IRCall):
			return instr.dest
		return None

	def _get_uses(self, instr: IRInstruction) -> list[IRValue]:
		"""Return all values read by an instruction."""
		if isinstance(instr, IRAddrOf):
			return [instr.source]
		if isinstance(instr, IRBinOp):
			return [instr.left, instr.right]
		if isinstance(instr, IRUnaryOp):
			return [instr.operand]
		if isinstance(instr, IRCopy):
			return [instr.source]
		if isinstance(instr, IRLoad):
			return [instr.address]
		if isinstance(instr, IRStore):
			return [instr.address, instr.value]
		if isinstance(instr, IRCondJump):
			return [instr.condition]
		if isinstance(instr, IRCall):
			return list(instr.args)
		if isinstance(instr, IRReturn):
			return [instr.value] if instr.value is not None else []
		if isinstance(instr, IRParam):
			return [instr.value]
		if isinstance(instr, IRConvert):
			return [instr.source]
		if isinstance(instr, IRVaStart):
			return [instr.ap_addr]
		if isinstance(instr, IRVaArg):
			return [instr.ap_addr]
		if isinstance(instr, IRVaEnd):
			return [instr.ap_addr]
		if isinstance(instr, IRVaCopy):
			return [instr.dest_addr, instr.src_addr]
		return []
