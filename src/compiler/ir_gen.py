"""AST-to-IR lowering pass: walks AST nodes and produces three-address code."""

from __future__ import annotations

from compiler.ast_nodes import (
	ArraySubscript,
	ASTVisitor,
	Assignment,
	BinaryOp,
	BreakStmt,
	CaseClause,
	CastExpr,
	CharLiteral,
	CompoundAssignment,
	CompoundStmt,
	ContinueStmt,
	DoWhileStmt,
	EnumDecl,
	ExprStmt,
	FloatLiteral,
	ForStmt,
	FunctionCall,
	FunctionDecl,
	InitializerList,
	TypedefDecl,
	Identifier,
	IfStmt,
	IntLiteral,
	MemberAccess,
	ParamDecl,
	PostfixExpr,
	Program,
	ReturnStmt,
	SizeofExpr,
	StringLiteral,
	StructDecl,
	StructMember,
	SwitchStmt,
	TernaryExpr,
	TypeSpec,
	UnaryOp,
	UnionDecl,
	VarDecl,
	WhileStmt,
)
from compiler.ir import (
	IRAlloc,
	IRBinOp,
	IRCall,
	IRCondJump,
	IRConst,
	IRConvert,
	IRCopy,
	IRFloatConst,
	IRFunction,
	IRGlobalRef,
	IRGlobalVar,
	IRInstruction,
	IRJump,
	IRLabelInstr,
	IRLoad,
	IRParam,
	IRProgram,
	IRReturn,
	IRStore,
	IRStringData,
	IRTemp,
	IRType,
	IRUnaryOp,
	IRValue,
)

_TYPE_MAP: dict[str, IRType] = {
	"int": IRType.INT,
	"char": IRType.CHAR,
	"void": IRType.VOID,
	"float": IRType.FLOAT,
	"double": IRType.DOUBLE,
}

_SIZE_MAP: dict[str, int] = {
	"int": 4,
	"char": 1,
	"void": 0,
	"float": 4,
	"double": 8,
}

_ALIGN_MAP: dict[str, int] = {
	"int": 4,
	"char": 1,
	"void": 1,
	"float": 4,
	"double": 8,
}

_FLOAT_TYPES = {IRType.FLOAT, IRType.DOUBLE}


def _resolve_ir_type(ts: TypeSpec) -> IRType:
	if ts.pointer_count > 0:
		return IRType.POINTER
	# Width modifiers: long/long long -> POINTER-sized int (INT for now, no INT64)
	if ts.width_modifier in ("long", "long long"):
		return IRType.INT
	if ts.width_modifier == "short":
		return IRType.INT
	return _TYPE_MAP.get(ts.base_type, IRType.INT)


def _resolve_size(ts: TypeSpec) -> int:
	if ts.pointer_count > 0:
		return 8
	if ts.width_modifier == "short":
		return 2
	if ts.width_modifier in ("long", "long long"):
		return 8
	return _SIZE_MAP.get(ts.base_type, 4)


def _resolve_alignment(ts: TypeSpec) -> int:
	"""Return the natural alignment of a primitive type."""
	if ts.pointer_count > 0:
		return 8
	if ts.width_modifier == "short":
		return 2
	if ts.width_modifier in ("long", "long long"):
		return 8
	return _ALIGN_MAP.get(ts.base_type, 4)


def _align_to(offset: int, alignment: int) -> int:
	"""Round *offset* up to the next multiple of *alignment*."""
	return (offset + alignment - 1) & ~(alignment - 1)


class IRGenerator(ASTVisitor):
	"""Translates an AST into a three-address code IRProgram."""

	def __init__(self) -> None:
		self._temp_counter: int = 0
		self._label_counter: int = 0
		self._instructions: list[IRInstruction] = []
		self._locals: dict[str, IRTemp] = {}
		self._local_types: dict[str, TypeSpec] = {}
		self._local_array: dict[str, list[int]] = {}
		self._temp_types: dict[str, IRType] = {}
		self._functions: list[IRFunction] = []
		self._globals: list[IRGlobalVar] = []
		self._global_names: set[str] = set()
		self._in_function: bool = False
		self._loop_stack: list[tuple[str, str]] = []  # (continue_label, break_label)
		self._structs: dict[str, list[StructMember]] = {}  # struct name -> members
		self._unions: dict[str, list[StructMember]] = {}  # union name -> members
		self._string_data: list[IRStringData] = []
		self._string_counter: int = 0
		self._enum_constants: dict[str, int] = {}
		self._func_ptr_locals: set[str] = set()
		self._known_functions: set[str] = set()

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

	def _value_ir_type(self, val: IRValue) -> IRType:
		"""Determine the IR type of a value."""
		if isinstance(val, IRFloatConst):
			return val.ir_type
		if isinstance(val, IRTemp):
			return self._temp_types.get(val.name, IRType.INT)
		return IRType.INT

	def _set_temp_type(self, temp: IRTemp, ir_type: IRType) -> None:
		self._temp_types[temp.name] = ir_type

	def _is_float_type(self, ir_type: IRType) -> bool:
		return ir_type in _FLOAT_TYPES

	def _resolve_local_ir_type(self, name: str) -> IRType:
		"""Get the IRType for a local variable by name."""
		ts = self._local_types.get(name)
		if ts is not None:
			return _resolve_ir_type(ts)
		return IRType.INT

	# ------------------------------------------------------------------
	# Public entry point
	# ------------------------------------------------------------------

	def generate(self, program: Program) -> IRProgram:
		"""Lower an entire AST Program to an IRProgram."""
		self.visit(program)
		return IRProgram(
			functions=list(self._functions),
			globals=list(self._globals),
			string_data=list(self._string_data),
		)

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
		self._known_functions.add(node.name)
		if node.body is None:
			return
		old_instructions = self._instructions
		old_locals = self._locals
		old_types = self._local_types
		old_arrays = self._local_array
		old_temp_types = self._temp_types
		old_in_function = self._in_function
		old_func_ptr_locals = self._func_ptr_locals
		self._instructions = []
		self._locals = {}
		self._local_types = {}
		self._local_array = {}
		self._temp_types = {}
		self._in_function = True
		self._func_ptr_locals = set()

		params: list[IRTemp] = []
		param_types: list[IRType] = []
		for param in node.params:
			p_temp = self._new_temp()
			params.append(p_temp)
			self._locals[param.name] = p_temp
			self._local_types[param.name] = param.type_spec
			if param.type_spec.is_function_pointer:
				self._func_ptr_locals.add(param.name)
				p_ir_type = IRType.POINTER
			else:
				p_ir_type = _resolve_ir_type(param.type_spec)
			param_types.append(p_ir_type)
			self._set_temp_type(p_temp, p_ir_type)

		self.visit(node.body)

		self._functions.append(
			IRFunction(
				name=node.name,
				params=params,
				body=self._instructions,
				return_type=_resolve_ir_type(node.return_type),
				param_types=param_types,
			)
		)

		self._instructions = old_instructions
		self._locals = old_locals
		self._local_types = old_types
		self._local_array = old_arrays
		self._temp_types = old_temp_types
		self._in_function = old_in_function
		self._func_ptr_locals = old_func_ptr_locals

	def visit_var_decl(self, node: VarDecl) -> None:
		if not self._in_function:
			# Global variable declaration
			ir_type = _resolve_ir_type(node.type_spec) if not node.type_spec.is_function_pointer else IRType.POINTER
			if isinstance(node.initializer, InitializerList):
				init_values = self._collect_init_values(node.initializer)
				total_size = 0
				if node.array_sizes:
					element_size = _resolve_size(node.type_spec)
					for se in node.array_sizes:
						if isinstance(se, IntLiteral):
							total_size = se.value * element_size
				elif node.type_spec.base_type.startswith("struct "):
					struct_name = node.type_spec.base_type[len("struct "):]
					total_size = self._compute_struct_size(struct_name)
				# Pad with zeros if needed
				element_size = _resolve_size(node.type_spec) if node.array_sizes else 4
				total_slots = total_size // element_size if element_size > 0 else len(init_values)
				while len(init_values) < total_slots:
					init_values.append(0)
				self._globals.append(IRGlobalVar(
					name=node.name, ir_type=ir_type,
					initializer_values=init_values, total_size=total_size,
				))
			else:
				init_val: int | None = None
				if node.initializer is not None and isinstance(node.initializer, IntLiteral):
					init_val = node.initializer.value
				self._globals.append(IRGlobalVar(name=node.name, ir_type=ir_type, initializer=init_val))
			self._global_names.add(node.name)
			if node.type_spec.is_function_pointer:
				self._func_ptr_locals.add(node.name)
			return

		is_fp = node.type_spec.is_function_pointer
		dest = self._new_temp()
		self._locals[node.name] = dest
		self._local_types[node.name] = node.type_spec
		if is_fp:
			self._func_ptr_locals.add(node.name)
			var_ir_type = IRType.POINTER
		else:
			var_ir_type = _resolve_ir_type(node.type_spec)
		self._set_temp_type(dest, var_ir_type)
		if node.array_sizes is not None and len(node.array_sizes) > 0:
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
			alloc_size = 8 if is_fp else self._resolve_type_size(node.type_spec)
			self._emit(IRAlloc(dest=dest, size=alloc_size))
		if node.initializer is not None:
			if isinstance(node.initializer, InitializerList):
				self._emit_initializer_list(dest, node)
			elif is_fp:
				val = self._emit_func_ptr_value(node.initializer)
				self._emit(IRCopy(dest=dest, source=val, ir_type=IRType.POINTER))
			else:
				val = self.visit(node.initializer)
				val_type = self._value_ir_type(val)
				if self._is_float_type(var_ir_type) and not self._is_float_type(val_type):
					converted = self._new_temp()
					self._set_temp_type(converted, var_ir_type)
					self._emit(IRConvert(dest=converted, source=val, from_type=val_type, to_type=var_ir_type))
					val = converted
				elif not self._is_float_type(var_ir_type) and self._is_float_type(val_type):
					converted = self._new_temp()
					self._set_temp_type(converted, var_ir_type)
					self._emit(IRConvert(dest=converted, source=val, from_type=val_type, to_type=var_ir_type))
					val = converted
				self._emit(IRCopy(dest=dest, source=val, ir_type=var_ir_type))

	def visit_param_decl(self, node: ParamDecl) -> IRTemp:
		temp = self._new_temp()
		self._locals[node.name] = temp
		return temp

	# ------------------------------------------------------------------
	# Expressions  (each returns an IRValue representing the result)
	# ------------------------------------------------------------------

	def visit_int_literal(self, node: IntLiteral) -> IRConst:
		return IRConst(node.value)

	def visit_float_literal(self, node: FloatLiteral) -> IRFloatConst:
		ir_type = IRType.FLOAT if node.suffix == "f" else IRType.DOUBLE
		return IRFloatConst(value=node.value, ir_type=ir_type)

	def visit_char_literal(self, node: CharLiteral) -> IRConst:
		return IRConst(ord(node.value))

	def visit_string_literal(self, node: StringLiteral) -> IRGlobalRef:
		label = f".str{self._string_counter}"
		self._string_counter += 1
		self._string_data.append(IRStringData(label=label, value=node.value))
		return IRGlobalRef(label)

	def visit_identifier(self, node: Identifier) -> IRTemp | IRConst:
		if node.name in self._enum_constants:
			return IRConst(self._enum_constants[node.name])
		src = self._locals.get(node.name)
		if src is not None:
			dest = self._new_temp()
			src_type = self._resolve_local_ir_type(node.name)
			self._set_temp_type(dest, src_type)
			self._emit(IRCopy(dest=dest, source=src, ir_type=src_type))
			return dest
		if node.name in self._global_names:
			dest = self._new_temp()
			self._emit(IRLoad(dest=dest, address=IRGlobalRef(node.name)))
			return dest
		return IRTemp(node.name)

	def visit_binary_op(self, node: BinaryOp) -> IRTemp:
		if node.op == "&&":
			return self._emit_short_circuit_and(node)
		if node.op == "||":
			return self._emit_short_circuit_or(node)
		left = self.visit(node.left)
		right = self.visit(node.right)
		left_type = self._value_ir_type(left)
		right_type = self._value_ir_type(right)
		# Determine result type: promote to float/double if either operand is float
		if self._is_float_type(left_type) or self._is_float_type(right_type):
			# Promote to the wider float type (double > float)
			if left_type == IRType.DOUBLE or right_type == IRType.DOUBLE:
				result_type = IRType.DOUBLE
			else:
				result_type = IRType.FLOAT
			# Convert int operand to float if needed
			if not self._is_float_type(left_type):
				conv = self._new_temp()
				self._set_temp_type(conv, result_type)
				self._emit(IRConvert(dest=conv, source=left, from_type=left_type, to_type=result_type))
				left = conv
			elif left_type != result_type and self._is_float_type(left_type):
				conv = self._new_temp()
				self._set_temp_type(conv, result_type)
				self._emit(IRConvert(dest=conv, source=left, from_type=left_type, to_type=result_type))
				left = conv
			if not self._is_float_type(right_type):
				conv = self._new_temp()
				self._set_temp_type(conv, result_type)
				self._emit(IRConvert(dest=conv, source=right, from_type=right_type, to_type=result_type))
				right = conv
			elif right_type != result_type and self._is_float_type(right_type):
				conv = self._new_temp()
				self._set_temp_type(conv, result_type)
				self._emit(IRConvert(dest=conv, source=right, from_type=right_type, to_type=result_type))
				right = conv
			dest = self._new_temp()
			# Comparison ops return int even for float operands
			if node.op in ("<", ">", "<=", ">=", "==", "!="):
				self._set_temp_type(dest, IRType.INT)
			else:
				self._set_temp_type(dest, result_type)
			self._emit(IRBinOp(dest=dest, left=left, op=node.op, right=right, ir_type=result_type))
			return dest
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
			# Address-of a function -> get function address
			if isinstance(node.operand, Identifier) and node.operand.name in self._known_functions:
				dest = self._new_temp()
				self._set_temp_type(dest, IRType.POINTER)
				self._emit(IRCopy(dest=dest, source=IRGlobalRef(node.operand.name), ir_type=IRType.POINTER))
				return dest
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
		op_type = self._value_ir_type(operand)
		dest = self._new_temp()
		if self._is_float_type(op_type):
			self._set_temp_type(dest, op_type)
			self._emit(IRUnaryOp(dest=dest, op=node.op, operand=operand, ir_type=op_type))
		else:
			self._emit(IRUnaryOp(dest=dest, op=node.op, operand=operand))
		return dest

	def visit_assignment(self, node: Assignment) -> IRTemp:
		# Check if target is a function pointer variable
		if isinstance(node.target, Identifier) and node.target.name in self._func_ptr_locals:
			val = self._emit_func_ptr_value(node.value)
			target_temp = self._locals.get(node.target.name)
			if target_temp is not None:
				self._emit(IRCopy(dest=target_temp, source=val, ir_type=IRType.POINTER))
				return target_temp
		val = self.visit(node.value)
		if isinstance(node.target, ArraySubscript):
			addr = self._compute_array_addr(node.target)
			val_type = self._value_ir_type(val)
			self._emit(IRStore(address=addr, value=val, ir_type=val_type))
			return val if isinstance(val, IRTemp) else self._new_temp()
		if isinstance(node.target, UnaryOp) and node.target.op == "*":
			addr = self.visit(node.target.operand)
			val_type = self._value_ir_type(val)
			self._emit(IRStore(address=addr, value=val, ir_type=val_type))
			return val if isinstance(val, IRTemp) else self._new_temp()
		if isinstance(node.target, MemberAccess):
			addr = self._compute_member_addr(node.target)
			val_type = self._value_ir_type(val)
			self._emit(IRStore(address=addr, value=val, ir_type=val_type))
			return val if isinstance(val, IRTemp) else self._new_temp()
		if isinstance(node.target, Identifier):
			target_temp = self._locals.get(node.target.name)
			if target_temp is not None:
				target_type = self._resolve_local_ir_type(node.target.name)
				val_type = self._value_ir_type(val)
				if self._is_float_type(target_type) and not self._is_float_type(val_type):
					conv = self._new_temp()
					self._set_temp_type(conv, target_type)
					self._emit(IRConvert(dest=conv, source=val, from_type=val_type, to_type=target_type))
					val = conv
				elif not self._is_float_type(target_type) and self._is_float_type(val_type):
					conv = self._new_temp()
					self._set_temp_type(conv, target_type)
					self._emit(IRConvert(dest=conv, source=val, from_type=val_type, to_type=target_type))
					val = conv
				self._emit(IRCopy(dest=target_temp, source=val, ir_type=target_type))
				return target_temp
			if node.target.name in self._global_names:
				self._emit(IRStore(address=IRGlobalRef(node.target.name), value=val))
				return val if isinstance(val, IRTemp) else self._new_temp()
		target = self.visit(node.target)
		if isinstance(target, IRTemp):
			self._emit(IRCopy(dest=target, source=val))
			return target
		dest = self._new_temp()
		self._emit(IRCopy(dest=dest, source=val))
		return dest

	def _emit_func_ptr_value(self, node: object) -> IRValue:
		"""Emit IR to get a function address from an expression (func name, &func, or another fp)."""
		if isinstance(node, Identifier):
			# Bare function name -> address of function
			if node.name in self._known_functions:
				dest = self._new_temp()
				self._set_temp_type(dest, IRType.POINTER)
				self._emit(IRCopy(dest=dest, source=IRGlobalRef(node.name), ir_type=IRType.POINTER))
				return dest
			# Another function pointer variable
			if node.name in self._func_ptr_locals:
				src = self._locals.get(node.name)
				if src is not None:
					dest = self._new_temp()
					self._set_temp_type(dest, IRType.POINTER)
					self._emit(IRCopy(dest=dest, source=src, ir_type=IRType.POINTER))
					return dest
		if isinstance(node, UnaryOp) and node.op == "&":
			if isinstance(node.operand, Identifier) and node.operand.name in self._known_functions:
				dest = self._new_temp()
				self._set_temp_type(dest, IRType.POINTER)
				self._emit(IRCopy(dest=dest, source=IRGlobalRef(node.operand.name), ir_type=IRType.POINTER))
				return dest
		# Fallback: evaluate expression
		return self.visit(node)

	def visit_function_call(self, node: FunctionCall) -> IRTemp:
		arg_vals = [self.visit(arg) for arg in node.arguments]
		arg_types = [self._value_ir_type(av) for av in arg_vals]
		for av in arg_vals:
			self._emit(IRParam(value=av))
		dest = self._new_temp()
		# Check if this is an indirect call through a function pointer
		if node.name in self._func_ptr_locals:
			fp_temp = self._locals.get(node.name)
			if fp_temp is not None:
				func_val = self._new_temp()
				self._set_temp_type(func_val, IRType.POINTER)
				self._emit(IRCopy(dest=func_val, source=fp_temp, ir_type=IRType.POINTER))
				self._emit(IRCall(
					dest=dest, function_name=node.name, args=arg_vals,
					arg_types=arg_types, indirect=True, func_value=func_val,
				))
				return dest
		self._emit(IRCall(
			dest=dest, function_name=node.name, args=arg_vals,
			arg_types=arg_types,
		))
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
		val_type = self._value_ir_type(val)
		self._emit(IRReturn(value=val, ir_type=val_type))

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
			if isinstance(node.init, list):
				for decl in node.init:
					self.visit(decl)
			else:
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
		arith_op = node.op[:-1] if node.op.endswith("=") else node.op
		if isinstance(node.target, ArraySubscript):
			addr = self._compute_array_addr(node.target)
			# Load current value
			current = self._new_temp()
			self._emit(IRLoad(dest=current, address=addr))
			# Compute new value
			rhs = self.visit(node.value)
			result = self._new_temp()
			self._emit(IRBinOp(dest=result, left=current, op=arith_op, right=rhs))
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
			self._emit(IRBinOp(dest=result, left=current, op=arith_op, right=rhs))
			# Write back
			self._emit(IRCopy(dest=target_temp, source=result))

	def visit_type_spec(self, node: TypeSpec) -> None:
		pass

	# ------------------------------------------------------------------
	# Switch / Case
	# ------------------------------------------------------------------

	def visit_switch_stmt(self, node: SwitchStmt) -> None:
		expr_val = self.visit(node.expression)
		end_label = self._new_label("switch_end")

		# Push switch onto loop stack so break jumps to end_label
		self._loop_stack.append((end_label, end_label))

		# Build case labels and default label
		case_labels: list[tuple[CaseClause, str]] = []
		default_label: str | None = None
		for clause in node.cases:
			lbl = self._new_label("case")
			if clause.value is None:
				default_label = lbl
			case_labels.append((clause, lbl))

		# Emit jump table: compare expr to each case value
		for clause, lbl in case_labels:
			if clause.value is not None:
				case_val = self.visit(clause.value)
				cmp = self._new_temp()
				self._emit(IRBinOp(dest=cmp, left=expr_val, op="==", right=case_val))
				next_check = self._new_label("case_check")
				self._emit(IRCondJump(condition=cmp, true_label=lbl, false_label=next_check))
				self._emit(IRLabelInstr(name=next_check))

		# After all checks, jump to default or end
		if default_label is not None:
			self._emit(IRJump(target=default_label))
		else:
			self._emit(IRJump(target=end_label))

		# Emit case bodies (fallthrough by default, break exits)
		for clause, lbl in case_labels:
			self._emit(IRLabelInstr(name=lbl))
			for stmt in clause.statements:
				self.visit(stmt)

		self._emit(IRLabelInstr(name=end_label))
		self._loop_stack.pop()

	def visit_case_clause(self, node: CaseClause) -> None:
		# Case clauses are handled inline by visit_switch_stmt
		pass

	# ------------------------------------------------------------------
	# Ternary expression
	# ------------------------------------------------------------------

	def visit_ternary_expr(self, node: TernaryExpr) -> IRTemp:
		cond = self.visit(node.condition)
		result = self._new_temp()
		true_label = self._new_label("tern_true")
		false_label = self._new_label("tern_false")
		end_label = self._new_label("tern_end")

		self._emit(IRCondJump(condition=cond, true_label=true_label, false_label=false_label))

		self._emit(IRLabelInstr(name=true_label))
		true_val = self.visit(node.true_expr)
		self._emit(IRCopy(dest=result, source=true_val))
		self._emit(IRJump(target=end_label))

		self._emit(IRLabelInstr(name=false_label))
		false_val = self.visit(node.false_expr)
		self._emit(IRCopy(dest=result, source=false_val))
		self._emit(IRJump(target=end_label))

		self._emit(IRLabelInstr(name=end_label))
		return result

	# ------------------------------------------------------------------
	# Sizeof expression
	# ------------------------------------------------------------------

	def visit_sizeof_expr(self, node: SizeofExpr) -> IRConst:
		if node.type_operand is not None:
			ts = node.type_operand
			key = ts.base_type
			if key.startswith("struct "):
				key = key[len("struct "):]
			elif key.startswith("union "):
				key = key[len("union "):]
			if ts.pointer_count == 0:
				if key in self._structs:
					return IRConst(self._compute_struct_size(key))
				if key in self._unions:
					return IRConst(self._compute_union_size(key))
			return IRConst(_resolve_size(ts))
		# sizeof(expr) -- compute based on expression type heuristic
		# For simplicity, default to 4 (int-sized)
		return IRConst(4)

	def _resolve_type_size(self, ts: TypeSpec) -> int:
		"""Resolve the size of a type, handling struct/union types."""
		return self._resolve_member_size(ts)

	def _compute_struct_size(self, name: str) -> int:
		"""Compute the total size of a struct including alignment padding."""
		members = self._structs.get(name, [])
		offset = 0
		max_align = 1
		for m in members:
			align = self._resolve_type_alignment(m.type_spec)
			max_align = max(max_align, align)
			offset = _align_to(offset, align)
			offset += self._resolve_member_size(m.type_spec)
		# Pad total size to the struct's overall alignment
		return _align_to(offset, max_align)

	def _compute_union_size(self, name: str) -> int:
		"""Compute the size of a union (largest member, padded to max alignment)."""
		members = self._unions.get(name, [])
		if not members:
			return 0
		max_size = 0
		max_align = 1
		for m in members:
			max_size = max(max_size, self._resolve_member_size(m.type_spec))
			max_align = max(max_align, self._resolve_type_alignment(m.type_spec))
		return _align_to(max_size, max_align)

	def _resolve_member_size(self, ts: TypeSpec) -> int:
		"""Resolve the size of a member type, handling nested structs/unions."""
		if ts.pointer_count > 0:
			return 8
		key = ts.base_type
		if key.startswith("struct "):
			sname = key[len("struct "):]
			if sname in self._structs:
				return self._compute_struct_size(sname)
		elif key.startswith("union "):
			uname = key[len("union "):]
			if uname in self._unions:
				return self._compute_union_size(uname)
		return _resolve_size(ts)

	def _resolve_type_alignment(self, ts: TypeSpec) -> int:
		"""Resolve the alignment of a type, handling nested structs/unions."""
		if ts.pointer_count > 0:
			return 8
		key = ts.base_type
		if key.startswith("struct "):
			sname = key[len("struct "):]
			members = self._structs.get(sname, [])
			if members:
				return max(self._resolve_type_alignment(m.type_spec) for m in members)
			return 4
		if key.startswith("union "):
			uname = key[len("union "):]
			members = self._unions.get(uname, [])
			if members:
				return max(self._resolve_type_alignment(m.type_spec) for m in members)
			return 4
		return _resolve_alignment(ts)

	# ------------------------------------------------------------------
	# Postfix increment/decrement
	# ------------------------------------------------------------------

	def visit_postfix_expr(self, node: PostfixExpr) -> IRTemp:
		# Load current value (this is the result -- old value)
		if isinstance(node.operand, Identifier):
			target = self._locals.get(node.operand.name)
			if target is not None:
				old_val = self._new_temp()
				self._emit(IRCopy(dest=old_val, source=target))
				new_val = self._new_temp()
				delta_op = "+" if node.op == "++" else "-"
				self._emit(IRBinOp(dest=new_val, left=old_val, op=delta_op, right=IRConst(1)))
				self._emit(IRCopy(dest=target, source=new_val))
				return old_val
		if isinstance(node.operand, ArraySubscript):
			addr = self._compute_array_addr(node.operand)
			old_val = self._new_temp()
			self._emit(IRLoad(dest=old_val, address=addr))
			new_val = self._new_temp()
			delta_op = "+" if node.op == "++" else "-"
			self._emit(IRBinOp(dest=new_val, left=old_val, op=delta_op, right=IRConst(1)))
			addr2 = self._compute_array_addr(node.operand)
			self._emit(IRStore(address=addr2, value=new_val))
			return old_val
		# Fallback: evaluate operand, compute new val (side effect may be lost)
		operand = self.visit(node.operand)
		old_val = self._new_temp()
		self._emit(IRCopy(dest=old_val, source=operand))
		new_val = self._new_temp()
		delta_op = "+" if node.op == "++" else "-"
		self._emit(IRBinOp(dest=new_val, left=old_val, op=delta_op, right=IRConst(1)))
		return old_val

	# ------------------------------------------------------------------
	# Struct declarations and member access
	# ------------------------------------------------------------------

	def visit_enum_decl(self, node: EnumDecl) -> None:
		next_value = 0
		for const in node.constants:
			if const.value is not None and isinstance(const.value, IntLiteral):
				next_value = const.value.value
			self._enum_constants[const.name] = next_value
			next_value += 1

	def visit_typedef_decl(self, node: TypedefDecl) -> None:
		if node.struct_decl is not None:
			self.visit_struct_decl(node.struct_decl)
		if node.enum_decl is not None:
			self.visit_enum_decl(node.enum_decl)
		if node.union_decl is not None:
			self.visit_union_decl(node.union_decl)

	def visit_struct_decl(self, node: StructDecl) -> None:
		self._structs[node.name] = list(node.members)

	def visit_union_decl(self, node: UnionDecl) -> None:
		self._unions[node.name] = list(node.members)

	def visit_initializer_list(self, node: InitializerList) -> IRConst:
		# Should not be visited directly; handled by _emit_initializer_list
		return IRConst(0)

	def _emit_initializer_list(self, dest: IRTemp, node: VarDecl) -> None:
		"""Emit IR stores for each element in an initializer list."""
		init_list = node.initializer
		assert isinstance(init_list, InitializerList)
		is_array = node.array_sizes is not None and len(node.array_sizes) > 0
		base_type = node.type_spec.base_type

		if is_array:
			element_size = _resolve_size(node.type_spec)
			# Determine total array size
			total_elements = 0
			if node.array_sizes:
				for se in node.array_sizes:
					if isinstance(se, IntLiteral):
						total_elements = se.value
			total_elements = max(total_elements, len(init_list.elements))
			for i in range(total_elements):
				if i < len(init_list.elements):
					val = self.visit(init_list.elements[i])
				else:
					val = IRConst(0)
				offset = self._new_temp()
				self._emit(IRBinOp(dest=offset, left=IRConst(i), op="*", right=IRConst(element_size)))
				addr = self._new_temp()
				self._emit(IRBinOp(dest=addr, left=dest, op="+", right=offset))
				self._emit(IRStore(address=addr, value=val))
		elif base_type.startswith("struct "):
			struct_name = base_type[len("struct "):]
			members = self._structs.get(struct_name, [])
			field_offset = 0
			for i, elem in enumerate(init_list.elements):
				if i >= len(members):
					break
				align = self._resolve_type_alignment(members[i].type_spec)
				field_offset = _align_to(field_offset, align)
				if isinstance(elem, InitializerList):
					# Nested initializer for array/struct member
					member_addr = self._new_temp()
					self._emit(IRBinOp(dest=member_addr, left=dest, op="+", right=IRConst(field_offset)))
					member_type = members[i].type_spec
					member_size = self._resolve_member_size(member_type)
					for j, sub_elem in enumerate(elem.elements):
						sub_val = self.visit(sub_elem)
						sub_offset = self._new_temp()
						self._emit(IRBinOp(dest=sub_offset, left=IRConst(j), op="*", right=IRConst(member_size)))
						sub_addr = self._new_temp()
						self._emit(IRBinOp(dest=sub_addr, left=member_addr, op="+", right=sub_offset))
						self._emit(IRStore(address=sub_addr, value=sub_val))
				else:
					val = self.visit(elem)
					addr = self._new_temp()
					self._emit(IRBinOp(dest=addr, left=dest, op="+", right=IRConst(field_offset)))
					self._emit(IRStore(address=addr, value=val))
				field_offset += self._resolve_member_size(members[i].type_spec)
			# Zero-fill remaining members
			for i in range(len(init_list.elements), len(members)):
				align = self._resolve_type_alignment(members[i].type_spec)
				field_offset = _align_to(field_offset, align)
				addr = self._new_temp()
				self._emit(IRBinOp(dest=addr, left=dest, op="+", right=IRConst(field_offset)))
				self._emit(IRStore(address=addr, value=IRConst(0)))
				field_offset += self._resolve_member_size(members[i].type_spec)

	def _collect_init_values(self, init_list: InitializerList) -> list[int]:
		"""Collect constant integer values from an initializer list (for globals)."""
		values: list[int] = []
		for elem in init_list.elements:
			if isinstance(elem, IntLiteral):
				values.append(elem.value)
			elif isinstance(elem, InitializerList):
				values.extend(self._collect_init_values(elem))
			else:
				values.append(0)
		return values

	def _compute_member_addr(self, node: MemberAccess) -> IRTemp:
		"""Compute the memory address of a struct/union member."""
		base = self.visit(node.object)
		type_name = self._resolve_aggregate_name(node.object)
		is_union = type_name in self._unions

		if is_union:
			# Union: all members at offset 0
			return base

		# Struct: compute field offset
		offset = self._compute_field_offset(type_name, node.member)
		addr = self._new_temp()
		self._emit(IRBinOp(dest=addr, left=base, op="+", right=IRConst(offset)))
		return addr

	def visit_member_access(self, node: MemberAccess) -> IRTemp:
		addr = self._compute_member_addr(node)
		dest = self._new_temp()
		self._emit(IRLoad(dest=dest, address=addr))
		return dest

	def visit_cast_expr(self, node: CastExpr) -> IRTemp:
		"""Handle casts, including int<->float conversions."""
		val = self.visit(node.operand)
		val_type = self._value_ir_type(val)
		target_ir_type = _resolve_ir_type(node.target_type)
		dest = self._new_temp()
		self._set_temp_type(dest, target_ir_type)
		if self._is_float_type(target_ir_type) != self._is_float_type(val_type):
			self._emit(IRConvert(dest=dest, source=val, from_type=val_type, to_type=target_ir_type))
		elif self._is_float_type(target_ir_type) and self._is_float_type(val_type) and target_ir_type != val_type:
			self._emit(IRConvert(dest=dest, source=val, from_type=val_type, to_type=target_ir_type))
		else:
			self._emit(IRCopy(dest=dest, source=val, ir_type=target_ir_type))
		return dest

	def _resolve_struct_name(self, node: object) -> str:
		"""Try to determine the struct type name from an AST node."""
		return self._resolve_aggregate_name(node)

	def _resolve_aggregate_name(self, node: object) -> str:
		"""Try to determine the struct/union type name from an AST node."""
		if isinstance(node, Identifier):
			ts = self._local_types.get(node.name)
			if ts is not None:
				name = ts.base_type
				if name.startswith("struct "):
					name = name[len("struct "):]
				elif name.startswith("union "):
					name = name[len("union "):]
				return name
		return ""

	def _compute_field_offset(self, struct_name: str, field_name: str) -> int:
		members = self._structs.get(struct_name, [])
		offset = 0
		for m in members:
			align = self._resolve_type_alignment(m.type_spec)
			offset = _align_to(offset, align)
			if m.name == field_name:
				return offset
			offset += self._resolve_member_size(m.type_spec)
		return offset
