"""AST-to-IR lowering pass: walks AST nodes and produces three-address code."""

from __future__ import annotations

from compiler.ast_nodes import (
	ArraySubscript,
	ASTVisitor,
	Assignment,
	BinaryOp,
	BreakStmt,
	CharLiteral,
	CompoundAssignment,
	CompoundStmt,
	ContinueStmt,
	DoWhileStmt,
	ExprStmt,
	ForStmt,
	FunctionCall,
	FunctionDecl,
	Identifier,
	IfStmt,
	IntLiteral,
	ParamDecl,
	Program,
	ReturnStmt,
	StringLiteral,
	TypeSpec,
	UnaryOp,
	VarDecl,
	WhileStmt,
)
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

_TYPE_MAP: dict[str, IRType] = {
	"int": IRType.INT,
	"char": IRType.CHAR,
	"void": IRType.VOID,
}

_SIZE_MAP: dict[str, int] = {
	"int": 4,
	"char": 1,
	"void": 0,
}


def _resolve_ir_type(ts: TypeSpec) -> IRType:
	if ts.pointer_count > 0:
		return IRType.POINTER
	return _TYPE_MAP.get(ts.base_type, IRType.INT)


def _resolve_size(ts: TypeSpec) -> int:
	if ts.pointer_count > 0:
		return 8
	return _SIZE_MAP.get(ts.base_type, 4)


class IRGenerator(ASTVisitor):
	"""Translates an AST into a three-address code IRProgram."""

	def __init__(self) -> None:
		self._temp_counter: int = 0
		self._label_counter: int = 0
		self._instructions: list[IRInstruction] = []
		self._locals: dict[str, IRTemp] = {}
		self._local_types: dict[str, TypeSpec] = {}
		self._local_array: dict[str, list[int]] = {}
		self._functions: list[IRFunction] = []
		self._loop_stack: list[tuple[str, str]] = []  # (continue_label, break_label)

	# ------------------------------------------------------------------
	# Helpers
	# ------------------------------------------------------------------

	def _new_temp(self) -> IRTemp:
		name = f"t{self._temp_counter}"
		self._temp_counter += 1
		return IRTemp(name)

	def _new_label(self, prefix: str = "L") -> str:
		name = f"{prefix}{self._label_counter}"
		self._label_counter += 1
		return name

	def _emit(self, instr: IRInstruction) -> None:
		self._instructions.append(instr)

	# ------------------------------------------------------------------
	# Public entry point
	# ------------------------------------------------------------------

	def generate(self, program: Program) -> IRProgram:
		"""Lower an entire AST Program to an IRProgram."""
		self.visit(program)
		return IRProgram(functions=list(self._functions))

	# ------------------------------------------------------------------
	# Top-level
	# ------------------------------------------------------------------

	def visit_program(self, node: Program) -> None:
		for decl in node.declarations:
			self.visit(decl)

	# ------------------------------------------------------------------
	# Declarations
	# ------------------------------------------------------------------

	def visit_function_decl(self, node: FunctionDecl) -> None:
		old_instructions = self._instructions
		old_locals = self._locals
		old_types = self._local_types
		old_arrays = self._local_array
		self._instructions = []
		self._locals = {}
		self._local_types = {}
		self._local_array = {}

		params: list[IRTemp] = []
		for param in node.params:
			p_temp = self._new_temp()
			params.append(p_temp)
			self._locals[param.name] = p_temp
			self._local_types[param.name] = param.type_spec

		self.visit(node.body)

		self._functions.append(
			IRFunction(
				name=node.name,
				params=params,
				body=self._instructions,
				return_type=_resolve_ir_type(node.return_type),
			)
		)

		self._instructions = old_instructions
		self._locals = old_locals
		self._local_types = old_types
		self._local_array = old_arrays

	def visit_var_decl(self, node: VarDecl) -> None:
		dest = self._new_temp()
		self._locals[node.name] = dest
		self._local_types[node.name] = node.type_spec
		if node.array_sizes is not None and len(node.array_sizes) > 0:
			# Compute total array size: product of all dimensions * element_size
			element_size = _resolve_size(node.type_spec)
			total_elements = 1
			size_vals: list[int] = []
			for size_expr in node.array_sizes:
				if isinstance(size_expr, IntLiteral):
					total_elements *= size_expr.value
					size_vals.append(size_expr.value)
			self._local_array[node.name] = size_vals
			self._emit(IRAlloc(dest=dest, size=element_size * total_elements))
		else:
			self._emit(IRAlloc(dest=dest, size=_resolve_size(node.type_spec)))
		if node.initializer is not None:
			val = self.visit(node.initializer)
			self._emit(IRCopy(dest=dest, source=val))

	def visit_param_decl(self, node: ParamDecl) -> IRTemp:
		temp = self._new_temp()
		self._locals[node.name] = temp
		return temp

	# ------------------------------------------------------------------
	# Expressions  (each returns an IRValue representing the result)
	# ------------------------------------------------------------------

	def visit_int_literal(self, node: IntLiteral) -> IRConst:
		return IRConst(node.value)

	def visit_char_literal(self, node: CharLiteral) -> IRConst:
		return IRConst(ord(node.value))

	def visit_string_literal(self, node: StringLiteral) -> IRConst:
		return IRConst(0)

	def visit_identifier(self, node: Identifier) -> IRTemp:
		src = self._locals.get(node.name)
		if src is not None:
			dest = self._new_temp()
			self._emit(IRCopy(dest=dest, source=src))
			return dest
		return IRTemp(node.name)

	def visit_binary_op(self, node: BinaryOp) -> IRTemp:
		if node.op == "&&":
			return self._emit_short_circuit_and(node)
		if node.op == "||":
			return self._emit_short_circuit_or(node)
		left = self.visit(node.left)
		right = self.visit(node.right)
		dest = self._new_temp()
		self._emit(IRBinOp(dest=dest, left=left, op=node.op, right=right))
		return dest

	def _emit_short_circuit_and(self, node: BinaryOp) -> IRTemp:
		"""a && b: if a is falsy, result is 0; otherwise result is !!b."""
		result = self._new_temp()
		eval_right = self._new_label("and_right")
		false_label = self._new_label("and_false")
		end_label = self._new_label("and_end")

		left = self.visit(node.left)
		self._emit(IRCondJump(condition=left, true_label=eval_right, false_label=false_label))

		# Left was truthy -> evaluate right
		self._emit(IRLabelInstr(name=eval_right))
		right = self.visit(node.right)
		norm = self._new_temp()
		self._emit(IRBinOp(dest=norm, left=right, op="!=", right=IRConst(0)))
		self._emit(IRCopy(dest=result, source=norm))
		self._emit(IRJump(target=end_label))

		# Left was falsy -> result = 0
		self._emit(IRLabelInstr(name=false_label))
		self._emit(IRCopy(dest=result, source=IRConst(0)))
		self._emit(IRJump(target=end_label))

		self._emit(IRLabelInstr(name=end_label))
		return result

	def _emit_short_circuit_or(self, node: BinaryOp) -> IRTemp:
		"""a || b: if a is truthy, result is 1; otherwise result is !!b."""
		result = self._new_temp()
		eval_right = self._new_label("or_right")
		true_label = self._new_label("or_true")
		end_label = self._new_label("or_end")

		left = self.visit(node.left)
		self._emit(IRCondJump(condition=left, true_label=true_label, false_label=eval_right))

		# Left was falsy -> evaluate right
		self._emit(IRLabelInstr(name=eval_right))
		right = self.visit(node.right)
		norm = self._new_temp()
		self._emit(IRBinOp(dest=norm, left=right, op="!=", right=IRConst(0)))
		self._emit(IRCopy(dest=result, source=norm))
		self._emit(IRJump(target=end_label))

		# Left was truthy -> result = 1
		self._emit(IRLabelInstr(name=true_label))
		self._emit(IRCopy(dest=result, source=IRConst(1)))
		self._emit(IRJump(target=end_label))

		self._emit(IRLabelInstr(name=end_label))
		return result

	def visit_unary_op(self, node: UnaryOp) -> IRTemp:
		if node.op == "*":
			# Pointer dereference: load from the pointer value
			ptr = self.visit(node.operand)
			dest = self._new_temp()
			self._emit(IRLoad(dest=dest, address=ptr))
			return dest
		if node.op == "&":
			# Address-of: return the stack address of the variable
			if isinstance(node.operand, Identifier):
				src = self._locals.get(node.operand.name)
				if src is not None:
					return src
			operand = self.visit(node.operand)
			return operand
		if node.op in ("++", "--"):
			# Prefix ++/--: load, add/sub 1, store back
			if isinstance(node.operand, Identifier):
				target = self._locals.get(node.operand.name)
				if target is not None:
					current = self._new_temp()
					self._emit(IRCopy(dest=current, source=target))
					result = self._new_temp()
					delta_op = "+" if node.op == "++" else "-"
					self._emit(IRBinOp(dest=result, left=current, op=delta_op, right=IRConst(1)))
					self._emit(IRCopy(dest=target, source=result))
					return result
			operand = self.visit(node.operand)
			result = self._new_temp()
			delta_op = "+" if node.op == "++" else "-"
			self._emit(IRBinOp(dest=result, left=operand, op=delta_op, right=IRConst(1)))
			return result
		operand = self.visit(node.operand)
		dest = self._new_temp()
		self._emit(IRUnaryOp(dest=dest, op=node.op, operand=operand))
		return dest

	def visit_assignment(self, node: Assignment) -> IRTemp:
		val = self.visit(node.value)
		if isinstance(node.target, ArraySubscript):
			addr = self._compute_array_addr(node.target)
			self._emit(IRStore(address=addr, value=val))
			return val if isinstance(val, IRTemp) else self._new_temp()
		if isinstance(node.target, UnaryOp) and node.target.op == "*":
			# *p = v -> store through pointer
			addr = self.visit(node.target.operand)
			self._emit(IRStore(address=addr, value=val))
			return val if isinstance(val, IRTemp) else self._new_temp()
		if isinstance(node.target, Identifier):
			target_temp = self._locals.get(node.target.name)
			if target_temp is not None:
				self._emit(IRCopy(dest=target_temp, source=val))
				return target_temp
		target = self.visit(node.target)
		if isinstance(target, IRTemp):
			self._emit(IRCopy(dest=target, source=val))
			return target
		dest = self._new_temp()
		self._emit(IRCopy(dest=dest, source=val))
		return dest

	def visit_function_call(self, node: FunctionCall) -> IRTemp:
		arg_vals = [self.visit(arg) for arg in node.arguments]
		for av in arg_vals:
			self._emit(IRParam(value=av))
		dest = self._new_temp()
		self._emit(IRCall(dest=dest, function_name=node.name, args=arg_vals))
		return dest

	def _compute_array_addr(self, node: ArraySubscript) -> IRTemp:
		"""Compute the address of an array element: base + index * element_size."""
		base = self.visit(node.array)
		index = self.visit(node.index)
		# Determine element size from base type
		element_size = 4  # default int
		if isinstance(node.array, Identifier):
			ts = self._local_types.get(node.array.name)
			if ts is not None:
				element_size = _resolve_size(ts)
		size_val = IRConst(element_size)
		offset = self._new_temp()
		self._emit(IRBinOp(dest=offset, left=index, op="*", right=size_val))
		addr = self._new_temp()
		self._emit(IRBinOp(dest=addr, left=base, op="+", right=offset))
		return addr

	def visit_array_subscript(self, node: ArraySubscript) -> IRTemp:
		addr = self._compute_array_addr(node)
		dest = self._new_temp()
		self._emit(IRLoad(dest=dest, address=addr))
		return dest

	# ------------------------------------------------------------------
	# Statements
	# ------------------------------------------------------------------

	def visit_expr_stmt(self, node: ExprStmt) -> None:
		self.visit(node.expression)

	def visit_return_stmt(self, node: ReturnStmt) -> None:
		val = self.visit(node.expression)
		self._emit(IRReturn(value=val))

	def visit_compound_stmt(self, node: CompoundStmt) -> None:
		for stmt in node.statements:
			self.visit(stmt)

	def visit_if_stmt(self, node: IfStmt) -> None:
		cond = self.visit(node.condition)
		then_label = self._new_label("if_then")
		else_label = self._new_label("if_else")
		end_label = self._new_label("if_end")

		if node.else_branch is not None:
			self._emit(IRCondJump(condition=cond, true_label=then_label, false_label=else_label))
			self._emit(IRLabelInstr(name=then_label))
			self.visit(node.then_branch)
			self._emit(IRJump(target=end_label))
			self._emit(IRLabelInstr(name=else_label))
			self.visit(node.else_branch)
			self._emit(IRLabelInstr(name=end_label))
		else:
			self._emit(IRCondJump(condition=cond, true_label=then_label, false_label=end_label))
			self._emit(IRLabelInstr(name=then_label))
			self.visit(node.then_branch)
			self._emit(IRLabelInstr(name=end_label))

	def visit_while_stmt(self, node: WhileStmt) -> None:
		loop_start = self._new_label("while_start")
		loop_body = self._new_label("while_body")
		loop_end = self._new_label("while_end")

		self._loop_stack.append((loop_start, loop_end))
		self._emit(IRLabelInstr(name=loop_start))
		cond = self.visit(node.condition)
		self._emit(IRCondJump(condition=cond, true_label=loop_body, false_label=loop_end))
		self._emit(IRLabelInstr(name=loop_body))
		self.visit(node.body)
		self._emit(IRJump(target=loop_start))
		self._emit(IRLabelInstr(name=loop_end))
		self._loop_stack.pop()

	def visit_for_stmt(self, node: ForStmt) -> None:
		if node.init is not None:
			self.visit(node.init)

		loop_start = self._new_label("for_start")
		loop_body = self._new_label("for_body")
		loop_update = self._new_label("for_update")
		loop_end = self._new_label("for_end")

		self._loop_stack.append((loop_update, loop_end))
		self._emit(IRLabelInstr(name=loop_start))
		if node.condition is not None:
			cond = self.visit(node.condition)
			self._emit(IRCondJump(condition=cond, true_label=loop_body, false_label=loop_end))
		self._emit(IRLabelInstr(name=loop_body))
		self.visit(node.body)
		self._emit(IRLabelInstr(name=loop_update))
		if node.update is not None:
			self.visit(node.update)
		self._emit(IRJump(target=loop_start))
		self._emit(IRLabelInstr(name=loop_end))
		self._loop_stack.pop()

	def visit_do_while_stmt(self, node: DoWhileStmt) -> None:
		loop_body = self._new_label("do_body")
		loop_cond = self._new_label("do_cond")
		loop_end = self._new_label("do_end")

		self._loop_stack.append((loop_cond, loop_end))
		self._emit(IRLabelInstr(name=loop_body))
		self.visit(node.body)
		self._emit(IRLabelInstr(name=loop_cond))
		cond = self.visit(node.condition)
		self._emit(IRCondJump(condition=cond, true_label=loop_body, false_label=loop_end))
		self._emit(IRLabelInstr(name=loop_end))
		self._loop_stack.pop()

	def visit_break_stmt(self, node: BreakStmt) -> None:
		_, break_label = self._loop_stack[-1]
		self._emit(IRJump(target=break_label))

	def visit_continue_stmt(self, node: ContinueStmt) -> None:
		continue_label, _ = self._loop_stack[-1]
		self._emit(IRJump(target=continue_label))

	def visit_compound_assignment(self, node: CompoundAssignment) -> None:
		if isinstance(node.target, ArraySubscript):
			addr = self._compute_array_addr(node.target)
			# Load current value
			current = self._new_temp()
			self._emit(IRLoad(dest=current, address=addr))
			# Compute new value
			rhs = self.visit(node.value)
			result = self._new_temp()
			self._emit(IRBinOp(dest=result, left=current, op=node.op, right=rhs))
			# Store back (recompute address since addr temp may have been clobbered)
			addr2 = self._compute_array_addr(node.target)
			self._emit(IRStore(address=addr2, value=result))
		elif isinstance(node.target, Identifier):
			target_temp = self._locals.get(node.target.name)
			if target_temp is None:
				target_temp = IRTemp(node.target.name)
			# Read current value
			current = self._new_temp()
			self._emit(IRCopy(dest=current, source=target_temp))
			# Compute new value
			rhs = self.visit(node.value)
			result = self._new_temp()
			self._emit(IRBinOp(dest=result, left=current, op=node.op, right=rhs))
			# Write back
			self._emit(IRCopy(dest=target_temp, source=result))

	def visit_type_spec(self, node: TypeSpec) -> None:
		pass
