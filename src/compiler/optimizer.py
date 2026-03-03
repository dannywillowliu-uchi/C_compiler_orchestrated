"""IR optimization passes: constant folding, dead code elimination, copy propagation,
strength reduction, jump threading, unreachable code elimination, common subexpression
elimination, and loop-invariant code motion."""

from __future__ import annotations

from compiler.ir import (
	IRAlloc,
	IRBinOp,
	IRCall,
	IRCondJump,
	IRConst,
	IRCopy,
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
	IRValue,
	IRLabelInstr,
)


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
				self._strength_reduction,
				self._copy_propagation,
				self._cse,
				self._dead_code_elimination,
				self._jump_threading,
				self._unreachable_elimination,
				self._licm,
			):
				new_body = pass_fn(body)
				if new_body != body:
					changed = True
					body = new_body
		return IRFunction(name=func.name, params=func.params, body=body, return_type=func.return_type)

	# -- Constant Folding --

	def _constant_fold(self, body: list[IRInstruction]) -> list[IRInstruction]:
		"""Replace operations on two constants with the computed result."""
		result: list[IRInstruction] = []
		for instr in body:
			if isinstance(instr, IRBinOp) and isinstance(instr.left, IRConst) and isinstance(instr.right, IRConst):
				folded = self._eval_binop(instr.left.value, instr.op, instr.right.value)
				if folded is not None:
					result.append(IRCopy(dest=instr.dest, source=IRConst(folded)))
					continue
			elif isinstance(instr, IRUnaryOp) and isinstance(instr.operand, IRConst):
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
			if isinstance(new_instr, IRCopy) and isinstance(new_instr.source, (IRTemp, IRConst)):
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
			if not (isinstance(instr, (IRBinOp, IRUnaryOp, IRCopy)) and instr.dest.name not in used)
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
		"""Hoist loop-invariant computations above loop headers."""
		label_pos: dict[str, int] = {}
		for i, instr in enumerate(body):
			if isinstance(instr, IRLabelInstr):
				label_pos[instr.name] = i

		for i, instr in enumerate(body):
			targets: list[str] = []
			if isinstance(instr, IRJump):
				targets.append(instr.target)
			elif isinstance(instr, IRCondJump):
				targets.extend([instr.true_label, instr.false_label])

			for target in targets:
				if target not in label_pos or label_pos[target] > i:
					continue
				header = label_pos[target]
				back_edge = i

				loop_defs: set[str] = set()
				for j in range(header, back_edge + 1):
					d = self._get_dest(body[j])
					if d is not None:
						loop_defs.add(d.name)

				to_hoist: list[int] = []
				for j in range(header + 1, back_edge):
					jnstr = body[j]
					if not isinstance(jnstr, (IRBinOp, IRUnaryOp)):
						continue
					d = self._get_dest(jnstr)
					if d is not None:
						other_defs = sum(
							1 for k in range(header, back_edge + 1)
							if k != j
							and self._get_dest(body[k]) is not None
							and self._get_dest(body[k]).name == d.name
						)
						if other_defs > 0:
							continue
					operands = self._get_uses(jnstr)
					if all(
						isinstance(op, IRConst) or (isinstance(op, IRTemp) and op.name not in loop_defs)
						for op in operands
					):
						to_hoist.append(j)

				if to_hoist:
					hoisted = [body[j] for j in to_hoist]
					hoist_set = set(to_hoist)
					new_body: list[IRInstruction] = []
					for j, b_instr in enumerate(body):
						if j == header:
							new_body.extend(hoisted)
						if j not in hoist_set:
							new_body.append(b_instr)
					return new_body

		return body

	# -- Helpers --

	def _get_dest(self, instr: IRInstruction) -> IRTemp | None:
		"""Return the destination temp written by an instruction, if any."""
		if isinstance(instr, (IRBinOp, IRUnaryOp, IRCopy, IRLoad, IRAlloc)):
			return instr.dest
		if isinstance(instr, IRCall):
			return instr.dest
		return None

	def _get_uses(self, instr: IRInstruction) -> list[IRValue]:
		"""Return all values read by an instruction."""
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
		return []
