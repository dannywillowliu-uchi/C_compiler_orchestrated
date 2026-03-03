"""AST-to-IR lowering pass: walks AST nodes and produces three-address code."""

from __future__ import annotations

from compiler.ast_nodes import (
	ASTVisitor,
	Assignment,
	BinaryOp,
	CharLiteral,
	CompoundStmt,
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
	IRParam,
	IRProgram,
	IRReturn,
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
		self._functions: list[IRFunction] = []

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
		self._instructions = []
		self._locals = {}

		params: list[IRTemp] = []
		for param in node.params:
			p_temp = self._new_temp()
			params.append(p_temp)
			self._locals[param.name] = p_temp

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

	def visit_var_decl(self, node: VarDecl) -> None:
		dest = self._new_temp()
		self._locals[node.name] = dest
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
		left = self.visit(node.left)
		right = self.visit(node.right)
		dest = self._new_temp()
		self._emit(IRBinOp(dest=dest, left=left, op=node.op, right=right))
		return dest

	def visit_unary_op(self, node: UnaryOp) -> IRTemp:
		operand = self.visit(node.operand)
		dest = self._new_temp()
		self._emit(IRUnaryOp(dest=dest, op=node.op, operand=operand))
		return dest

	def visit_assignment(self, node: Assignment) -> IRTemp:
		val = self.visit(node.value)
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

		self._emit(IRLabelInstr(name=loop_start))
		cond = self.visit(node.condition)
		self._emit(IRCondJump(condition=cond, true_label=loop_body, false_label=loop_end))
		self._emit(IRLabelInstr(name=loop_body))
		self.visit(node.body)
		self._emit(IRJump(target=loop_start))
		self._emit(IRLabelInstr(name=loop_end))

	def visit_for_stmt(self, node: ForStmt) -> None:
		if node.init is not None:
			self.visit(node.init)

		loop_start = self._new_label("for_start")
		loop_body = self._new_label("for_body")
		loop_end = self._new_label("for_end")

		self._emit(IRLabelInstr(name=loop_start))
		if node.condition is not None:
			cond = self.visit(node.condition)
			self._emit(IRCondJump(condition=cond, true_label=loop_body, false_label=loop_end))
		self._emit(IRLabelInstr(name=loop_body))
		self.visit(node.body)
		if node.update is not None:
			self.visit(node.update)
		self._emit(IRJump(target=loop_start))
		self._emit(IRLabelInstr(name=loop_end))

	def visit_type_spec(self, node: TypeSpec) -> None:
		pass
