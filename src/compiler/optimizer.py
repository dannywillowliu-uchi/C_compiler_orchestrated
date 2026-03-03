"""IR optimization passes: constant folding, dead code elimination, copy propagation."""

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
			new_body = self._constant_fold(body)
			if new_body != body:
				changed = True
				body = new_body
			new_body = self._copy_propagation(body)
			if new_body != body:
				changed = True
				body = new_body
			new_body = self._dead_code_elimination(body)
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
