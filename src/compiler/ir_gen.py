"""AST-to-IR lowering pass: walks AST nodes and produces three-address code."""

from __future__ import annotations

from compiler.ast_nodes import (
	ASTNode,
	ArraySubscript,
	ASTVisitor,
	Assignment,
	BinaryOp,
	BreakStmt,
	CaseClause,
	CastExpr,
	CharLiteral,
	CommaExpr,
	CompoundAssignment,
	CompoundLiteral,
	CompoundStmt,
	ContinueStmt,
	DesignatedInit,
	DoWhileStmt,
	EnumDecl,
	ExprStmt,
	FloatLiteral,
	ForStmt,
	FunctionCall,
	FunctionDecl,
	GotoStmt,
	InitializerList,
	LabelStmt,
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
	StaticAssertDecl,
	StringLiteral,
	StructDecl,
	StructMember,
	SwitchStmt,
	TernaryExpr,
	TypeSpec,
	UnaryOp,
	UnionDecl,
	VaCopyExpr,
	VaArgExpr,
	VaEndExpr,
	VaStartExpr,
	VarDecl,
	WhileStmt,
)
from compiler.ir import (
	IRAddrOf,
	IRAlloc,
	IRBinOp,
	IRBulkCopy,
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
	IRVaArg,
	IRVaCopy,
	IRVaEnd,
	IRVaStart,
	IRValue,
)

_TYPE_MAP: dict[str, IRType] = {
	"int": IRType.INT,
	"char": IRType.CHAR,
	"short": IRType.SHORT,
	"long": IRType.LONG,
	"void": IRType.VOID,
	"float": IRType.FLOAT,
	"double": IRType.DOUBLE,
	"_Bool": IRType.BOOL,
}

_SIZE_MAP: dict[str, int] = {
	"int": 4,
	"char": 1,
	"short": 2,
	"long": 8,
	"void": 0,
	"float": 4,
	"double": 8,
	"_Bool": 1,
}

_ALIGN_MAP: dict[str, int] = {
	"int": 4,
	"char": 1,
	"short": 2,
	"long": 8,
	"void": 1,
	"float": 4,
	"double": 8,
	"_Bool": 1,
}

_FLOAT_TYPES = {IRType.FLOAT, IRType.DOUBLE}


def _resolve_ir_type(ts: TypeSpec) -> IRType:
	if ts.pointer_count > 0:
		return IRType.POINTER
	if ts.width_modifier in ("long", "long long"):
		return IRType.LONG
	if ts.width_modifier == "short":
		return IRType.SHORT
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


_UNSIGNED_TYPEDEFS = frozenset({
	"uint8_t", "uint16_t", "uint32_t", "uint64_t",
	"size_t", "uintptr_t",
})


def _is_type_unsigned(ts: TypeSpec) -> bool:
	"""Check if a TypeSpec represents an unsigned type (including typedefs)."""
	if ts.signedness == "unsigned":
		return True
	if ts.base_type in _UNSIGNED_TYPEDEFS:
		return True
	if ts.base_type.startswith("unsigned "):
		return True
	return False


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
		self._local_alloc: set[str] = set()
		self._temp_types: dict[str, IRType] = {}
		self._temp_unsigned: dict[str, bool] = {}
		self._functions: list[IRFunction] = []
		self._globals: list[IRGlobalVar] = []
		self._global_names: set[str] = set()
		self._global_types: dict[str, TypeSpec] = {}
		self._global_array: dict[str, list[int]] = {}
		self._in_function: bool = False
		self._current_function_name: str = ""
		self._loop_stack: list[tuple[str, str]] = []  # (continue_label, break_label)
		self._structs: dict[str, list[StructMember]] = {}  # struct name -> members
		self._unions: dict[str, list[StructMember]] = {}  # union name -> members
		self._string_data: list[IRStringData] = []
		self._string_counter: int = 0
		self._enum_constants: dict[str, int] = {}
		self._func_ptr_locals: set[str] = set()
		self._known_functions: set[str] = set()
		self._user_labels: dict[str, str] = {}  # C label name -> IR label name
		self._static_local_map: dict[str, str] = {}  # local name -> mangled global name
		self._temp_pointee_size: dict[str, int] = {}  # pointer temp -> element size
		self._current_function_is_variadic: bool = False
		self._current_function_params: list[str] = []
		# Bitfield layout: struct_name -> {member_name: (byte_offset, bit_offset, bit_width, storage_size)}
		self._bitfield_layouts: dict[str, dict[str, tuple[int, int, int, int]]] = {}

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
		if isinstance(val, IRConst):
			return val.ir_type
		if isinstance(val, IRTemp):
			return self._temp_types.get(val.name, IRType.INT)
		return IRType.INT

	def _set_temp_type(self, temp: IRTemp, ir_type: IRType) -> None:
		self._temp_types[temp.name] = ir_type

	def _is_float_type(self, ir_type: IRType) -> bool:
		return ir_type in _FLOAT_TYPES

	def _emit_bool_normalize(self, val: IRValue) -> IRValue:
		"""Normalize a value for _Bool storage: any non-zero becomes 1 (C99 6.3.1.2)."""
		result = self._new_temp()
		self._set_temp_type(result, IRType.BOOL)
		self._emit(IRBinOp(dest=result, left=val, op="!=", right=IRConst(0), ir_type=IRType.INT))
		return result

	def _is_unsigned_value(self, val: IRValue) -> bool:
		"""Check if a value originates from an unsigned type."""
		if isinstance(val, IRTemp):
			return self._temp_unsigned.get(val.name, False)
		return False

	def _resolve_local_ir_type(self, name: str) -> IRType:
		"""Get the IRType for a local variable by name."""
		ts = self._local_types.get(name)
		if ts is not None:
			return _resolve_ir_type(ts)
		return IRType.INT

	def _pointee_size_from_type(self, ts: TypeSpec) -> int:
		"""Get the element size that a pointer type points to."""
		if ts.pointer_count > 1:
			return 8  # pointer to pointer -> pointer size
		deref = TypeSpec(base_type=ts.base_type, width_modifier=ts.width_modifier, signedness=ts.signedness)
		return self._resolve_member_size(deref)

	def _get_pointee_size(self, val: IRValue) -> int:
		"""Get the pointed-to element size for a pointer value."""
		if isinstance(val, IRTemp):
			return self._temp_pointee_size.get(val.name, 1)
		return 1

	def _local_pointee_size(self, name: str) -> int:
		"""Get the pointed-to element size for a local pointer variable."""
		ts = self._local_types.get(name)
		if ts is not None and ts.pointer_count > 0:
			return self._pointee_size_from_type(ts)
		return 1

	def _read_simple_rvalue(self, node: ASTNode) -> IRValue | None:
		"""Return a direct IRValue for simple expressions without emitting instructions.

		Returns the value directly for literals, enum constants, and local variable
		temporaries. Returns None for complex expressions that require codegen.
		"""
		if isinstance(node, IntLiteral):
			return self.visit_int_literal(node)
		if isinstance(node, CharLiteral):
			return self.visit_char_literal(node)
		if isinstance(node, FloatLiteral):
			return self.visit_float_literal(node)
		if isinstance(node, Identifier):
			if node.name in self._enum_constants:
				return IRConst(self._enum_constants[node.name])
			# Static locals and globals need IRLoad instructions; skip them.
			if node.name in self._static_local_map or node.name in self._global_names:
				return None
			src = self._locals.get(node.name)
			if src is not None:
				return src
		return None

	def _visit_rvalue(self, node: ASTNode) -> IRValue:
		"""Visit an expression, avoiding unnecessary copies for simple values."""
		simple = self._read_simple_rvalue(node)
		if simple is not None:
			return simple
		return self.visit(node)

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
			# Prototype-only or extern declaration: emit a stub IRFunction
			self._functions.append(
				IRFunction(
					name=node.name,
					params=[],
					body=[],
					return_type=_resolve_ir_type(node.return_type),
					param_types=[],
					storage_class=node.storage_class,
					is_prototype=True,
				)
			)
			return
		old_instructions = self._instructions
		old_locals = self._locals
		old_types = self._local_types
		old_arrays = self._local_array
		old_allocs = self._local_alloc
		old_temp_types = self._temp_types
		old_temp_unsigned = self._temp_unsigned
		old_in_function = self._in_function
		old_func_ptr_locals = self._func_ptr_locals
		old_user_labels = self._user_labels
		old_function_name = self._current_function_name
		old_static_local_map = self._static_local_map
		old_pointee_size = self._temp_pointee_size
		old_is_variadic = self._current_function_is_variadic
		old_func_params = self._current_function_params
		self._instructions = []
		self._locals = {}
		self._local_types = {}
		self._local_array = {}
		self._local_alloc = set()
		self._temp_types = {}
		self._temp_unsigned = {}
		self._temp_pointee_size = {}
		self._in_function = True
		self._current_function_name = node.name
		self._current_function_is_variadic = node.is_variadic
		self._current_function_params = [p.name for p in node.params]
		self._func_ptr_locals = set()
		self._user_labels = {}
		self._static_local_map = {}

		params: list[IRTemp] = []
		param_types: list[IRType] = []
		struct_param_fixups: list[tuple[str, IRTemp, str]] = []  # (param_name, param_temp, agg_name)
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
			if param.type_spec.signedness == "unsigned":
				self._temp_unsigned[p_temp.name] = True
			if p_ir_type == IRType.POINTER and param.type_spec.pointer_count > 0:
				self._temp_pointee_size[p_temp.name] = self._pointee_size_from_type(param.type_spec)
			# Check if param is a by-value struct/union (needs stack copy for member access)
			if param.type_spec.pointer_count == 0:
				base = param.type_spec.base_type
				agg_name = None
				if base.startswith("struct ") and base[len("struct "):] in self._structs:
					agg_name = base[len("struct "):]
				elif base.startswith("union ") and base[len("union "):] in self._unions:
					agg_name = base[len("union "):]
				if agg_name is not None:
					struct_param_fixups.append((param.name, p_temp, agg_name))

		# For struct/union params passed by value: allocate stack space and bulk-copy from caller's address
		for param_name, p_temp, agg_name in struct_param_fixups:
			if agg_name in self._unions:
				size = self._compute_union_size(agg_name)
			else:
				size = self._compute_struct_size(agg_name)
			stack_addr = self._new_temp()
			self._set_temp_type(stack_addr, IRType.POINTER)
			self._emit(IRAlloc(dest=stack_addr, size=size))
			self._emit(IRBulkCopy(dest_addr=stack_addr, src_addr=p_temp, size=size))
			self._locals[param_name] = stack_addr
			self._local_alloc.add(param_name)

		self.visit(node.body)

		self._functions.append(
			IRFunction(
				name=node.name,
				params=params,
				body=self._instructions,
				return_type=_resolve_ir_type(node.return_type),
				param_types=param_types,
				storage_class=node.storage_class,
				is_variadic=node.is_variadic,
			)
		)

		self._instructions = old_instructions
		self._locals = old_locals
		self._local_types = old_types
		self._local_array = old_arrays
		self._local_alloc = old_allocs
		self._temp_types = old_temp_types
		self._temp_unsigned = old_temp_unsigned
		self._in_function = old_in_function
		self._current_function_name = old_function_name
		self._func_ptr_locals = old_func_ptr_locals
		self._current_function_is_variadic = old_is_variadic
		self._current_function_params = old_func_params
		self._user_labels = old_user_labels
		self._static_local_map = old_static_local_map
		self._temp_pointee_size = old_pointee_size

	def _emit_global_var(self, name: str, node: VarDecl, storage_class: str | None = None) -> None:
		"""Emit a global variable declaration with proper initializer handling."""
		ir_type = _resolve_ir_type(node.type_spec) if not node.type_spec.is_function_pointer else IRType.POINTER
		sc = storage_class if storage_class is not None else node.storage_class
		# Global char array initialized from string literal
		if (
			node.array_sizes is not None
			and len(node.array_sizes) > 0
			and node.type_spec.base_type == "char"
			and node.type_spec.pointer_count == 0
			and isinstance(node.initializer, StringLiteral)
		):
			string_val = node.initializer.value
			total_size = 0
			for se in node.array_sizes:
				val = self._eval_const_expr(se)
				if val is not None:
					total_size = val
			string_bytes = [ord(ch) for ch in string_val] + [0]
			init_values = list(string_bytes[:total_size])
			while len(init_values) < total_size:
				init_values.append(0)
			self._globals.append(IRGlobalVar(
				name=name, ir_type=IRType.CHAR,
				initializer_values=init_values, total_size=total_size,
				storage_class=sc,
			))
		elif isinstance(node.initializer, InitializerList):
			init_values = self._collect_init_values(node.initializer)
			total_size = 0
			if node.array_sizes:
				element_size = _resolve_size(node.type_spec)
				for se in node.array_sizes:
					val = self._eval_const_expr(se)
					if val is not None:
						total_size = val * element_size
			elif node.type_spec.base_type.startswith("struct "):
				struct_name = node.type_spec.base_type[len("struct "):]
				total_size = self._compute_struct_size(struct_name)
			element_size = _resolve_size(node.type_spec) if node.array_sizes else 4
			total_slots = total_size // element_size if element_size > 0 else len(init_values)
			while len(init_values) < total_slots:
				init_values.append(0)
			self._globals.append(IRGlobalVar(
				name=name, ir_type=ir_type,
				initializer_values=init_values, total_size=total_size,
				storage_class=sc,
			))
		else:
			init_val: int | None = None
			float_init: float | None = None
			string_label: str | None = None
			symbol_init: str | None = None
			symbol_offset: int = 0
			if node.initializer is not None:
				if isinstance(node.initializer, FloatLiteral):
					float_init = node.initializer.value
				elif isinstance(node.initializer, StringLiteral):
					label = f".str{self._string_counter}"
					self._string_counter += 1
					self._string_data.append(IRStringData(label=label, value=node.initializer.value))
					string_label = label
				elif isinstance(node.initializer, UnaryOp) and node.initializer.op == "&" and isinstance(node.initializer.operand, Identifier):
					symbol_init = node.initializer.operand.name
				elif isinstance(node.initializer, UnaryOp) and node.initializer.op == "&" and isinstance(node.initializer.operand, MemberAccess):
					ma = node.initializer.operand
					if isinstance(ma.object, Identifier):
						symbol_init = ma.object.name
						type_name = self._resolve_aggregate_name(ma.object)
						if type_name and type_name not in self._unions:
							symbol_offset = self._compute_field_offset(type_name, ma.member)
					elif isinstance(ma.object, ArraySubscript) and isinstance(ma.object.array, Identifier):
						# &array[idx].member
						symbol_init = ma.object.array.name
						idx = self._eval_const_expr(ma.object.index)
						arr_ts = self._global_types.get(ma.object.array.name)
						if idx is not None and arr_ts is not None:
							elem_size = _resolve_size(arr_ts)
							symbol_offset = idx * elem_size
							type_name = self._resolve_aggregate_name(ma.object.array)
							if type_name and type_name not in self._unions:
								symbol_offset += self._compute_field_offset(type_name, ma.member)
				elif isinstance(node.initializer, UnaryOp) and node.initializer.op == "&" and isinstance(node.initializer.operand, ArraySubscript):
					subscript = node.initializer.operand
					if isinstance(subscript.array, Identifier):
						symbol_init = subscript.array.name
						idx = self._eval_const_expr(subscript.index)
						if idx is not None:
							# Element size from the array's base type, not the pointer type
							arr_ts = self._global_types.get(subscript.array.name)
							if arr_ts is not None:
								elem_size = _resolve_size(arr_ts)
							else:
								elem_size = 4
							symbol_offset = idx * elem_size
				elif isinstance(node.initializer, UnaryOp) and node.initializer.op == "-" and isinstance(node.initializer.operand, FloatLiteral):
					float_init = -node.initializer.operand.value
				elif isinstance(node.initializer, Identifier) and node.initializer.name in self._global_names:
					# Bare global name as initializer (array decay to pointer, or address of global)
					init_name = node.initializer.name
					if init_name in self._global_array or init_name in self._known_functions:
						symbol_init = init_name
					else:
						# Could be another global variable - try const eval first
						const_val = self._eval_const_expr(node.initializer)
						if const_val is not None:
							init_val = const_val
				else:
					const_val = self._eval_const_expr(node.initializer)
					if const_val is not None:
						init_val = const_val
			total_size = 0
			if node.array_sizes:
				element_size = _resolve_size(node.type_spec)
				total_elements = 1
				for se in node.array_sizes:
					val = self._eval_const_expr(se)
					if val is not None:
						total_elements *= val
				total_size = total_elements * element_size
			elif node.type_spec.pointer_count == 0:
				base = node.type_spec.base_type
				if base.startswith("struct "):
					sn = base[len("struct "):]
					if sn in self._structs:
						total_size = self._compute_struct_size(sn)
				elif base.startswith("union "):
					un = base[len("union "):]
					if un in self._unions:
						total_size = self._compute_union_size(un)
			self._globals.append(IRGlobalVar(
				name=name, ir_type=ir_type, initializer=init_val,
				total_size=total_size,
				storage_class=sc, float_initializer=float_init,
				string_label=string_label,
				symbol_initializer=symbol_init,
				symbol_initializer_offset=symbol_offset,
			))
		self._global_names.add(name)

	def visit_var_decl(self, node: VarDecl) -> None:
		if not self._in_function:
			self._emit_global_var(node.name, node)
			self._global_types[node.name] = node.type_spec
			if node.array_sizes:
				size_vals = []
				for se in node.array_sizes:
					val = self._eval_const_expr(se)
					if val is not None:
						size_vals.append(val)
				if size_vals:
					self._global_array[node.name] = size_vals
			if node.type_spec.is_function_pointer:
				self._func_ptr_locals.add(node.name)
			return

		# Static local: emit as global with mangled name, reference via IRGlobalRef
		if node.storage_class == "static":
			mangled = f"{self._current_function_name}.{node.name}"
			self._emit_global_var(mangled, node, storage_class="static")
			self._static_local_map[node.name] = mangled
			self._local_types[node.name] = node.type_spec
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
		if node.type_spec.signedness == "unsigned":
			self._temp_unsigned[dest.name] = True
		if var_ir_type == IRType.POINTER and node.type_spec.pointer_count > 0:
			self._temp_pointee_size[dest.name] = self._pointee_size_from_type(node.type_spec)
		if node.array_sizes is not None and len(node.array_sizes) > 0:
			element_size = _resolve_size(node.type_spec)
			total_elements = 1
			size_vals: list[int] = []
			for size_expr in node.array_sizes:
				val = self._eval_const_expr(size_expr)
				if val is not None:
					total_elements *= val
					size_vals.append(val)
			self._local_array[node.name] = size_vals
			self._emit(IRAlloc(dest=dest, size=element_size * total_elements))
			self._local_alloc.add(node.name)
		else:
			alloc_size = 8 if is_fp else self._resolve_type_size(node.type_spec)
			self._emit(IRAlloc(dest=dest, size=alloc_size))
			# Only structs, unions and arrays use the alloc pointer; scalars store values directly
			base = node.type_spec.base_type
			if node.type_spec.pointer_count == 0 and (
				(base.startswith("struct ") and base[len("struct "):] in self._structs) or
				(base.startswith("union ") and base[len("union "):] in self._unions)
			):
				self._local_alloc.add(node.name)
		if node.initializer is not None:
			if isinstance(node.initializer, InitializerList):
				self._emit_initializer_list(dest, node)
			elif (
				node.array_sizes is not None
				and len(node.array_sizes) > 0
				and node.type_spec.base_type == "char"
				and node.type_spec.pointer_count == 0
				and isinstance(node.initializer, StringLiteral)
			):
				self._emit_char_array_from_string(dest, node)
			elif is_fp:
				val = self._emit_func_ptr_value(node.initializer)
				self._emit(IRCopy(dest=dest, source=val, ir_type=IRType.POINTER))
			else:
				val = self._visit_rvalue(node.initializer)
				# Check for struct/union initialization from another variable
				agg_name = self._is_local_aggregate(node.name)
				if agg_name is not None:
					self._emit_aggregate_copy(dest, val, agg_name)
				else:
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
					if var_ir_type == IRType.BOOL:
						val = self._emit_bool_normalize(val)
					self._emit(IRCopy(dest=dest, source=val, ir_type=var_ir_type))

	def visit_param_decl(self, node: ParamDecl) -> IRTemp:
		temp = self._new_temp()
		self._locals[node.name] = temp
		return temp

	# ------------------------------------------------------------------
	# Expressions  (each returns an IRValue representing the result)
	# ------------------------------------------------------------------

	def visit_int_literal(self, node: IntLiteral) -> IRConst:
		suffix = node.suffix.lower() if node.suffix else ""
		if suffix in ("l", "ll"):
			return IRConst(node.value, ir_type=IRType.LONG)
		if suffix == "u":
			return IRConst(node.value, ir_type=IRType.INT, is_unsigned=True)
		if suffix in ("ul", "ull"):
			return IRConst(node.value, ir_type=IRType.LONG, is_unsigned=True)
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
		# Static local: load from mangled global
		if node.name in self._static_local_map:
			mangled = self._static_local_map[node.name]
			dest = self._new_temp()
			ir_type = self._resolve_local_ir_type(node.name)
			self._set_temp_type(dest, ir_type)
			ts = self._local_types.get(node.name)
			load_unsigned = ts is not None and ts.signedness == "unsigned"
			self._emit(IRLoad(dest=dest, address=IRGlobalRef(mangled), ir_type=ir_type, is_unsigned=load_unsigned))
			return dest
		src = self._locals.get(node.name)
		if src is not None:
			dest = self._new_temp()
			src_type = self._resolve_local_ir_type(node.name)
			self._set_temp_type(dest, src_type)
			ts = self._local_types.get(node.name)
			if ts is not None and ts.signedness == "unsigned":
				self._temp_unsigned[dest.name] = True
			if src_type == IRType.POINTER and ts is not None and ts.pointer_count > 0:
				self._temp_pointee_size[dest.name] = self._pointee_size_from_type(ts)
			self._emit(IRCopy(dest=dest, source=src, ir_type=src_type))
			return dest
		if node.name in self._global_names:
			dest = self._new_temp()
			global_ts = self._global_types.get(node.name)
			# For global arrays, return address (array decays to pointer)
			if node.name in self._global_array:
				self._set_temp_type(dest, IRType.POINTER)
				self._emit(IRCopy(dest=dest, source=IRGlobalRef(node.name), ir_type=IRType.POINTER))
				return dest
			# For global aggregates (struct/union), return address, not loaded value
			if global_ts is not None and global_ts.pointer_count == 0:
				base = global_ts.base_type
				if (base.startswith("struct ") and base[len("struct "):] in self._structs) or \
				   (base.startswith("union ") and base[len("union "):] in self._unions):
					self._set_temp_type(dest, IRType.POINTER)
					self._emit(IRCopy(dest=dest, source=IRGlobalRef(node.name), ir_type=IRType.POINTER))
					return dest
			if global_ts is not None:
				ir_type = _resolve_ir_type(global_ts)
				self._set_temp_type(dest, ir_type)
				load_unsigned = global_ts.signedness == "unsigned"
				if load_unsigned:
					self._temp_unsigned[dest.name] = True
				self._emit(IRLoad(dest=dest, address=IRGlobalRef(node.name), ir_type=ir_type, is_unsigned=load_unsigned))
			else:
				self._emit(IRLoad(dest=dest, address=IRGlobalRef(node.name)))
			return dest
		return IRTemp(node.name)

	def visit_binary_op(self, node: BinaryOp) -> IRTemp:
		if node.op == "&&":
			return self._emit_short_circuit_and(node)
		if node.op == "||":
			return self._emit_short_circuit_or(node)
		left = self._visit_rvalue(node.left)
		right = self._visit_rvalue(node.right)
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
		# Pointer arithmetic scaling
		left_is_ptr = left_type == IRType.POINTER
		right_is_ptr = right_type == IRType.POINTER
		if (left_is_ptr or right_is_ptr) and node.op in ("+", "-"):
			if left_is_ptr and right_is_ptr and node.op == "-":
				# pointer - pointer -> element count
				raw_diff = self._new_temp()
				self._emit(IRBinOp(dest=raw_diff, left=left, op="-", right=right, ir_type=IRType.POINTER))
				elem_size = self._get_pointee_size(left)
				if elem_size > 1:
					dest = self._new_temp()
					self._emit(IRBinOp(dest=dest, left=raw_diff, op="/", right=IRConst(elem_size)))
					return dest
				return raw_diff
			elif left_is_ptr:
				# pointer +/- integer: scale the integer operand
				elem_size = self._get_pointee_size(left)
				if elem_size > 1:
					scaled = self._new_temp()
					self._emit(IRBinOp(dest=scaled, left=right, op="*", right=IRConst(elem_size)))
					right = scaled
				dest = self._new_temp()
				self._set_temp_type(dest, IRType.POINTER)
				self._temp_pointee_size[dest.name] = self._get_pointee_size(left)
				self._emit(IRBinOp(dest=dest, left=left, op=node.op, right=right, ir_type=IRType.POINTER))
				return dest
			else:
				# integer + pointer
				elem_size = self._get_pointee_size(right)
				if elem_size > 1:
					scaled = self._new_temp()
					self._emit(IRBinOp(dest=scaled, left=left, op="*", right=IRConst(elem_size)))
					left = scaled
				dest = self._new_temp()
				self._set_temp_type(dest, IRType.POINTER)
				self._temp_pointee_size[dest.name] = self._get_pointee_size(right)
				self._emit(IRBinOp(dest=dest, left=left, op="+", right=right, ir_type=IRType.POINTER))
				return dest

		is_unsigned = self._is_unsigned_value(left) or self._is_unsigned_value(right)
		dest = self._new_temp()
		if is_unsigned:
			self._temp_unsigned[dest.name] = True
		self._emit(IRBinOp(dest=dest, left=left, op=node.op, right=right, is_unsigned=is_unsigned))
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
			# Infer load type from the pointer's pointee type
			load_type = IRType.INT
			operand_ts = self._infer_expr_type(node.operand)
			# If pointee is a struct/union, return pointer as address (no scalar load)
			if operand_ts is not None and operand_ts.pointer_count == 1:
				base = operand_ts.base_type
				agg_name = None
				if base.startswith("struct ") and base[len("struct "):] in self._structs:
					agg_name = base[len("struct "):]
				elif base.startswith("union ") and base[len("union "):] in self._unions:
					agg_name = base[len("union "):]
				if agg_name is not None:
					self._set_temp_type(ptr, IRType.POINTER)
					return ptr
			dest = self._new_temp()
			if operand_ts is not None and operand_ts.pointer_count > 1:
				load_type = IRType.POINTER
				self._set_temp_type(dest, IRType.POINTER)
			elif operand_ts is not None and operand_ts.pointer_count == 1:
				load_type = _resolve_ir_type(TypeSpec(
					base_type=operand_ts.base_type,
					pointer_count=0,
					width_modifier=operand_ts.width_modifier,
					signedness=operand_ts.signedness,
				))
				self._set_temp_type(dest, load_type)
			load_unsigned = operand_ts is not None and operand_ts.signedness == "unsigned"
			self._emit(IRLoad(dest=dest, address=ptr, ir_type=load_type, is_unsigned=load_unsigned))
			return dest
		if node.op == "&":
			# Address-of a function -> get function address
			if isinstance(node.operand, Identifier) and node.operand.name in self._known_functions:
				dest = self._new_temp()
				self._set_temp_type(dest, IRType.POINTER)
				self._emit(IRCopy(dest=dest, source=IRGlobalRef(node.operand.name), ir_type=IRType.POINTER))
				return dest
			# Address-of static local
			if isinstance(node.operand, Identifier) and node.operand.name in self._static_local_map:
				mangled = self._static_local_map[node.operand.name]
				dest = self._new_temp()
				self._set_temp_type(dest, IRType.POINTER)
				self._emit(IRCopy(dest=dest, source=IRGlobalRef(mangled), ir_type=IRType.POINTER))
				return dest
			# Address-of: emit IRAddrOf to get the stack address of the variable
			if isinstance(node.operand, Identifier):
				# Address-of global variable -> return IRGlobalRef (the address itself)
				if node.operand.name in self._global_names:
					dest = self._new_temp()
					self._set_temp_type(dest, IRType.POINTER)
					global_ts = self._global_types.get(node.operand.name)
					if global_ts is not None:
						self._temp_pointee_size[dest.name] = self._resolve_member_size(global_ts)
					self._emit(IRCopy(dest=dest, source=IRGlobalRef(node.operand.name), ir_type=IRType.POINTER))
					return dest
				src = self._locals.get(node.operand.name)
				if src is not None:
					dest = self._new_temp()
					self._set_temp_type(dest, IRType.POINTER)
					ts = self._local_types.get(node.operand.name)
					if ts is not None:
						self._temp_pointee_size[dest.name] = self._resolve_member_size(ts)
					if node.operand.name in self._local_alloc:
						# Alloc'd local: temp already holds pointer to data
						self._emit(IRCopy(dest=dest, source=src, ir_type=IRType.POINTER))
					else:
						# Parameter: need actual address of stack slot
						self._emit(IRAddrOf(dest=dest, source=src))
					return dest
			# Address-of array subscript: return the computed address without loading
			if isinstance(node.operand, ArraySubscript):
				addr = self._compute_array_addr(node.operand)
				dest = self._new_temp()
				self._set_temp_type(dest, IRType.POINTER)
				self._emit(IRCopy(dest=dest, source=addr, ir_type=IRType.POINTER))
				# Determine pointee size from array base
				base = node.operand
				while isinstance(base, ArraySubscript):
					base = base.array
				if isinstance(base, Identifier):
					ts = self._local_types.get(base.name)
					if ts is not None:
						self._temp_pointee_size[dest.name] = self._resolve_member_size(ts)
				return dest
			# Address-of member access: return computed member address without loading
			if isinstance(node.operand, MemberAccess):
				addr = self._compute_member_addr(node.operand)
				dest = self._new_temp()
				self._set_temp_type(dest, IRType.POINTER)
				self._emit(IRCopy(dest=dest, source=addr, ir_type=IRType.POINTER))
				return dest
			operand = self.visit(node.operand)
			return operand
		if node.op in ("++", "--"):
			# Prefix ++/-- on static locals
			if isinstance(node.operand, Identifier) and node.operand.name in self._static_local_map:
				mangled = self._static_local_map[node.operand.name]
				ir_type = self._resolve_local_ir_type(node.operand.name)
				current = self._new_temp()
				self._set_temp_type(current, ir_type)
				self._emit(IRLoad(dest=current, address=IRGlobalRef(mangled), ir_type=ir_type))
				result = self._new_temp()
				self._set_temp_type(result, ir_type)
				delta_op = "+" if node.op == "++" else "-"
				delta_val = IRConst(self._local_pointee_size(node.operand.name)) if ir_type == IRType.POINTER else IRConst(1)
				self._emit(IRBinOp(dest=result, left=current, op=delta_op, right=delta_val, ir_type=ir_type))
				self._emit(IRStore(address=IRGlobalRef(mangled), value=result, ir_type=ir_type))
				return result
			# Prefix ++/--: load, add/sub 1, store back
			if isinstance(node.operand, Identifier):
				target = self._locals.get(node.operand.name)
				if target is not None:
					ir_type = self._resolve_local_ir_type(node.operand.name)
					current = self._new_temp()
					self._set_temp_type(current, ir_type)
					self._emit(IRCopy(dest=current, source=target, ir_type=ir_type))
					result = self._new_temp()
					self._set_temp_type(result, ir_type)
					delta_op = "+" if node.op == "++" else "-"
					if ir_type == IRType.POINTER:
						delta_val: IRValue = IRConst(self._local_pointee_size(node.operand.name))
					elif self._is_float_type(ir_type):
						delta_val = IRFloatConst(1.0, ir_type=ir_type)
					else:
						delta_val = IRConst(1)
					self._emit(IRBinOp(dest=result, left=current, op=delta_op, right=delta_val, ir_type=ir_type))
					self._emit(IRCopy(dest=target, source=result, ir_type=ir_type))
					return result
				# Global variable prefix ++/--
				if node.operand.name in self._global_names:
					ir_type = _resolve_ir_type(self._global_types.get(node.operand.name, TypeSpec(base_type="int")))
					current = self._new_temp()
					self._set_temp_type(current, ir_type)
					self._emit(IRLoad(dest=current, address=IRGlobalRef(node.operand.name), ir_type=ir_type))
					result = self._new_temp()
					self._set_temp_type(result, ir_type)
					delta_op = "+" if node.op == "++" else "-"
					if self._is_float_type(ir_type):
						delta_val2: IRValue = IRFloatConst(1.0, ir_type=ir_type)
					else:
						delta_val2 = IRConst(1)
					self._emit(IRBinOp(dest=result, left=current, op=delta_op, right=delta_val2, ir_type=ir_type))
					self._emit(IRStore(address=IRGlobalRef(node.operand.name), value=result, ir_type=ir_type))
					return result
			if isinstance(node.operand, MemberAccess):
				bf_info = self._get_bitfield_info(node.operand)
				if bf_info is not None:
					current = self._bitfield_read(node.operand, bf_info)
					result = self._new_temp()
					delta_op = "+" if node.op == "++" else "-"
					self._emit(IRBinOp(dest=result, left=current, op=delta_op, right=IRConst(1)))
					self._bitfield_write(node.operand, bf_info, result)
					return result
				addr = self._compute_member_addr(node.operand)
				member_type = self._member_ir_type(node.operand)
				current = self._new_temp()
				self._emit(IRLoad(dest=current, address=addr, ir_type=member_type))
				result = self._new_temp()
				delta_op = "+" if node.op == "++" else "-"
				self._emit(IRBinOp(dest=result, left=current, op=delta_op, right=IRConst(1)))
				addr2 = self._compute_member_addr(node.operand)
				self._emit(IRStore(address=addr2, value=result, ir_type=member_type))
				return result
			operand = self.visit(node.operand)
			result = self._new_temp()
			delta_op = "+" if node.op == "++" else "-"
			self._emit(IRBinOp(dest=result, left=operand, op=delta_op, right=IRConst(1)))
			return result
		operand = self._visit_rvalue(node.operand)
		# Constant-fold unary minus on literals
		if node.op == "-" and isinstance(operand, IRConst):
			return IRConst(-operand.value, ir_type=operand.ir_type, is_unsigned=operand.is_unsigned)
		if node.op == "-" and isinstance(operand, IRFloatConst):
			return IRFloatConst(-operand.value, ir_type=operand.ir_type)
		if node.op == "~" and isinstance(operand, IRConst):
			return IRConst(~operand.value, ir_type=operand.ir_type, is_unsigned=operand.is_unsigned)
		if node.op == "!" and isinstance(operand, IRConst):
			return IRConst(int(operand.value == 0), ir_type=IRType.INT)
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
		val = self._visit_rvalue(node.value)
		# Static local assignment: store to mangled global
		if isinstance(node.target, Identifier) and node.target.name in self._static_local_map:
			mangled = self._static_local_map[node.target.name]
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
			if target_type == IRType.BOOL:
				val = self._emit_bool_normalize(val)
			self._emit(IRStore(address=IRGlobalRef(mangled), value=val, ir_type=target_type))
			return val if isinstance(val, IRTemp) else self._new_temp()
		if isinstance(node.target, ArraySubscript):
			addr = self._compute_array_addr(node.target)
			elem_type = self._array_element_ir_type(node.target)
			self._emit(IRStore(address=addr, value=val, ir_type=elem_type))
			return val if isinstance(val, IRTemp) else self._new_temp()
		if isinstance(node.target, UnaryOp) and node.target.op == "*":
			addr = self.visit(node.target.operand)
			# Check if we're assigning to a dereferenced pointer-to-struct/union
			agg_name = self._deref_aggregate_name(node.target.operand)
			if agg_name is not None:
				self._emit_aggregate_copy(addr, val, agg_name)
				return addr
			val_type = self._value_ir_type(val)
			self._emit(IRStore(address=addr, value=val, ir_type=val_type))
			return val if isinstance(val, IRTemp) else self._new_temp()
		if isinstance(node.target, MemberAccess):
			bf_info = self._get_bitfield_info(node.target)
			if bf_info is not None:
				self._bitfield_write(node.target, bf_info, val)
				return val if isinstance(val, IRTemp) else self._new_temp()
			addr = self._compute_member_addr(node.target)
			member_ts = self._resolve_member_type_spec(node.target)
			if self._member_is_aggregate(member_ts):
				nested_name = member_ts.base_type
				if nested_name.startswith("struct "):
					nested_name = nested_name[len("struct "):]
				elif nested_name.startswith("union "):
					nested_name = nested_name[len("union "):]
				self._emit_aggregate_copy(addr, val, nested_name)
				return addr
			member_type = self._member_ir_type(node.target)
			self._emit(IRStore(address=addr, value=val, ir_type=member_type))
			return val if isinstance(val, IRTemp) else self._new_temp()
		if isinstance(node.target, Identifier):
			target_temp = self._locals.get(node.target.name)
			if target_temp is not None:
				agg_name = self._is_local_aggregate(node.target.name)
				if agg_name is not None:
					self._emit_aggregate_copy(target_temp, val, agg_name)
					return target_temp
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
				if target_type == IRType.BOOL:
					val = self._emit_bool_normalize(val)
				self._emit(IRCopy(dest=target_temp, source=val, ir_type=target_type))
				return target_temp
			if node.target.name in self._global_names:
				agg_name = self._is_global_aggregate(node.target.name)
				if agg_name is not None:
					dest_addr = self._new_temp()
					self._set_temp_type(dest_addr, IRType.POINTER)
					self._emit(IRCopy(dest=dest_addr, source=IRGlobalRef(node.target.name), ir_type=IRType.POINTER))
					self._emit_aggregate_copy(dest_addr, val, agg_name)
					return dest_addr
				global_ts = self._global_types.get(node.target.name)
				if global_ts is not None and global_ts.base_type == "_Bool" and global_ts.pointer_count == 0:
					val = self._emit_bool_normalize(val)
				store_type = _resolve_ir_type(global_ts) if global_ts is not None else IRType.INT
				self._emit(IRStore(address=IRGlobalRef(node.target.name), value=val, ir_type=store_type))
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
		# Indirect call via expression callee (e.g. (*fp)(args), arr[i](args))
		if node.callee is not None:
			func_val = self.visit(node.callee)
			if not isinstance(func_val, IRTemp):
				tmp = self._new_temp()
				self._set_temp_type(tmp, IRType.POINTER)
				self._emit(IRCopy(dest=tmp, source=func_val, ir_type=IRType.POINTER))
				func_val = tmp
			self._emit(IRCall(
				dest=dest, function_name="<indirect>", args=arg_vals,
				arg_types=arg_types, indirect=True, func_value=func_val,
			))
			return dest
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

	def _array_element_ir_type(self, node: ArraySubscript) -> IRType:
		"""Determine the IR type of an array element for correct load/store sizing."""
		current: ASTNode = node
		while isinstance(current, ArraySubscript):
			current = current.array
		if isinstance(current, CastExpr):
			ts = current.target_type
			if ts.pointer_count > 0:
				deref = TypeSpec(base_type=ts.base_type, width_modifier=ts.width_modifier, signedness=ts.signedness, pointer_count=ts.pointer_count - 1)
				return _resolve_ir_type(deref)
		if isinstance(current, Identifier):
			ts = self._local_types.get(current.name) or self._global_types.get(current.name)
			if ts is not None:
				dims = self._local_array.get(current.name) or self._global_array.get(current.name, [])
				if ts.pointer_count > 0 and dims:
					# Array of pointers: element type is the pointer type itself
					return IRType.POINTER
				if ts.pointer_count > 0:
					# Pointer subscript: element type is the pointee type
					deref = TypeSpec(base_type=ts.base_type, width_modifier=ts.width_modifier, signedness=ts.signedness, pointer_count=ts.pointer_count - 1)
					return _resolve_ir_type(deref)
				return _resolve_ir_type(ts)
		return IRType.INT

	def _compute_array_addr(self, node: ArraySubscript) -> IRTemp:
		"""Compute the address of an array element, handling multi-dimensional arrays.

		For int a[3][4], a[i][j] computes: base + i*16 + j*4
		(stride for dimension d = element_size * product(dims[d+1:]))
		"""
		# Unwrap nested subscripts: a[i][j] -> indices=[i, j], base_node=Identifier("a")
		index_nodes: list[ASTNode] = []
		current: ASTNode = node
		while isinstance(current, ArraySubscript):
			index_nodes.append(current.index)
			current = current.array
		index_nodes.reverse()

		# For global arrays, get the address instead of loading the value
		is_global_array = isinstance(current, Identifier) and current.name in self._global_array
		if is_global_array:
			base = self._new_temp()
			self._set_temp_type(base, IRType.POINTER)
			self._emit(IRCopy(dest=base, source=IRGlobalRef(current.name), ir_type=IRType.POINTER))
		else:
			base = self.visit(current)

		# Determine element size and array dimensions from the base identifier
		element_size = 4  # default int
		dims: list[int] = []
		if isinstance(current, CastExpr):
			# Cast expression as array base: use cast target type for element size
			cast_ts = current.target_type
			if cast_ts.pointer_count > 0:
				element_size = self._pointee_size_from_type(cast_ts)
		elif isinstance(current, Identifier):
			ts = self._local_types.get(current.name) or self._global_types.get(current.name)
			dims = self._local_array.get(current.name) or self._global_array.get(current.name, [])
			if ts is not None:
				if ts.pointer_count > 0 and dims:
					# Array of pointers: element size is pointer size (8)
					element_size = 8
				elif ts.pointer_count > 0:
					# Pointer subscript: element size is pointee size
					element_size = self._pointee_size_from_type(ts)
				else:
					element_size = self._resolve_member_size(ts)

		addr = base
		for d, idx_node in enumerate(index_nodes):
			idx = self.visit(idx_node)
			# stride = element_size * product(dims[d+1:])
			stride = element_size
			for remaining_dim in dims[d + 1:]:
				stride *= remaining_dim
			offset = self._new_temp()
			self._emit(IRBinOp(dest=offset, left=idx, op="*", right=IRConst(stride)))
			new_addr = self._new_temp()
			self._emit(IRBinOp(dest=new_addr, left=addr, op="+", right=offset))
			addr = new_addr
		return addr

	def visit_array_subscript(self, node: ArraySubscript) -> IRTemp:
		addr = self._compute_array_addr(node)
		elem_type = self._array_element_ir_type(node)
		dest = self._new_temp()
		self._set_temp_type(dest, elem_type)
		self._emit(IRLoad(dest=dest, address=addr, ir_type=elem_type))
		return dest

	# ------------------------------------------------------------------
	# Statements
	# ------------------------------------------------------------------

	def visit_expr_stmt(self, node: ExprStmt) -> None:
		self.visit(node.expression)

	def visit_return_stmt(self, node: ReturnStmt) -> None:
		val = self._visit_rvalue(node.expression)
		val_type = self._value_ir_type(val)
		self._emit(IRReturn(value=val, ir_type=val_type))

	def visit_compound_stmt(self, node: CompoundStmt) -> None:
		for stmt in node.statements:
			self.visit(stmt)

	def visit_if_stmt(self, node: IfStmt) -> None:
		cond = self._visit_rvalue(node.condition)
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
		cond = self._visit_rvalue(node.condition)
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
			cond = self._visit_rvalue(node.condition)
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
		cond = self._visit_rvalue(node.condition)
		self._emit(IRCondJump(condition=cond, true_label=loop_body, false_label=loop_end))
		self._emit(IRLabelInstr(name=loop_end))
		self._loop_stack.pop()

	def visit_break_stmt(self, node: BreakStmt) -> None:
		_, break_label = self._loop_stack[-1]
		self._emit(IRJump(target=break_label))

	def visit_continue_stmt(self, node: ContinueStmt) -> None:
		continue_label, _ = self._loop_stack[-1]
		self._emit(IRJump(target=continue_label))

	def _get_user_label(self, name: str) -> str:
		if name not in self._user_labels:
			self._user_labels[name] = self._new_label("usr_")
		return self._user_labels[name]

	def visit_goto_stmt(self, node: GotoStmt) -> None:
		ir_label = self._get_user_label(node.label)
		self._emit(IRJump(target=ir_label))

	def visit_label_stmt(self, node: LabelStmt) -> None:
		ir_label = self._get_user_label(node.label)
		self._emit(IRLabelInstr(name=ir_label))
		self.visit(node.statement)

	def visit_compound_assignment(self, node: CompoundAssignment) -> None:
		arith_op = node.op[:-1] if node.op.endswith("=") else node.op
		if isinstance(node.target, ArraySubscript):
			addr = self._compute_array_addr(node.target)
			# Load current value
			current = self._new_temp()
			self._emit(IRLoad(dest=current, address=addr))
			# Compute new value
			rhs = self._visit_rvalue(node.value)
			result = self._new_temp()
			self._emit(IRBinOp(dest=result, left=current, op=arith_op, right=rhs))
			# Store back (recompute address since addr temp may have been clobbered)
			addr2 = self._compute_array_addr(node.target)
			self._emit(IRStore(address=addr2, value=result))
		elif isinstance(node.target, MemberAccess):
			bf_info = self._get_bitfield_info(node.target)
			if bf_info is not None:
				current = self._bitfield_read(node.target, bf_info)
				rhs = self._visit_rvalue(node.value)
				result = self._new_temp()
				self._emit(IRBinOp(dest=result, left=current, op=arith_op, right=rhs))
				self._bitfield_write(node.target, bf_info, result)
			else:
				addr = self._compute_member_addr(node.target)
				member_type = self._member_ir_type(node.target)
				current = self._new_temp()
				self._emit(IRLoad(dest=current, address=addr, ir_type=member_type))
				rhs = self._visit_rvalue(node.value)
				result = self._new_temp()
				self._emit(IRBinOp(dest=result, left=current, op=arith_op, right=rhs))
				addr2 = self._compute_member_addr(node.target)
				self._emit(IRStore(address=addr2, value=result, ir_type=member_type))
		elif isinstance(node.target, UnaryOp) and node.target.op == "*":
			addr = self.visit(node.target.operand)
			current = self._new_temp()
			self._emit(IRLoad(dest=current, address=addr))
			rhs = self._visit_rvalue(node.value)
			result = self._new_temp()
			self._emit(IRBinOp(dest=result, left=current, op=arith_op, right=rhs))
			addr2 = self.visit(node.target.operand)
			self._emit(IRStore(address=addr2, value=result))
		elif isinstance(node.target, Identifier) and node.target.name in self._static_local_map:
			mangled = self._static_local_map[node.target.name]
			ir_type = self._resolve_local_ir_type(node.target.name)
			current = self._new_temp()
			self._set_temp_type(current, ir_type)
			self._emit(IRLoad(dest=current, address=IRGlobalRef(mangled), ir_type=ir_type))
			rhs = self._visit_rvalue(node.value)
			result = self._new_temp()
			self._set_temp_type(result, ir_type)
			self._emit(IRBinOp(dest=result, left=current, op=arith_op, right=rhs, ir_type=ir_type))
			self._emit(IRStore(address=IRGlobalRef(mangled), value=result, ir_type=ir_type))
		elif isinstance(node.target, Identifier) and node.target.name in self._global_names:
			ir_type = IRType.INT
			global_ts = self._global_types.get(node.target.name)
			if global_ts is not None:
				ir_type = _resolve_ir_type(global_ts)
			current = self._new_temp()
			self._set_temp_type(current, ir_type)
			self._emit(IRLoad(dest=current, address=IRGlobalRef(node.target.name), ir_type=ir_type))
			rhs = self._visit_rvalue(node.value)
			result = self._new_temp()
			self._set_temp_type(result, ir_type)
			self._emit(IRBinOp(dest=result, left=current, op=arith_op, right=rhs, ir_type=ir_type))
			self._emit(IRStore(address=IRGlobalRef(node.target.name), value=result, ir_type=ir_type))
		elif isinstance(node.target, Identifier):
			target_temp = self._locals.get(node.target.name)
			if target_temp is None:
				target_temp = IRTemp(node.target.name)
			# Read current value
			current = self._new_temp()
			self._emit(IRCopy(dest=current, source=target_temp))
			# Compute new value
			rhs = self._visit_rvalue(node.value)
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

		# Separate pre-switch statements from case/default clauses
		pre_switch_clauses: list[CaseClause] = []
		regular_clauses: list[CaseClause] = []
		for clause in node.cases:
			if clause.is_pre_switch:
				pre_switch_clauses.append(clause)
			else:
				regular_clauses.append(clause)

		# Build case labels and default label
		case_labels: list[tuple[CaseClause, str]] = []
		default_label: str | None = None
		for clause in regular_clauses:
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

		# Emit pre-switch statements (reachable only via goto)
		for clause in pre_switch_clauses:
			for stmt in clause.statements:
				self.visit(stmt)

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

	def _infer_expr_type(self, node: ASTNode) -> TypeSpec | None:
		"""Infer the C type of an expression AST node without emitting IR."""
		if isinstance(node, Identifier):
			return self._local_types.get(node.name) or self._global_types.get(node.name)
		if isinstance(node, CharLiteral):
			return TypeSpec(base_type="char")
		if isinstance(node, IntLiteral):
			return TypeSpec(base_type="int")
		if isinstance(node, FloatLiteral):
			if node.suffix == "f":
				return TypeSpec(base_type="float")
			return TypeSpec(base_type="double")
		if isinstance(node, StringLiteral):
			return TypeSpec(base_type="char", pointer_count=1)
		if isinstance(node, CastExpr):
			return node.target_type
		if isinstance(node, UnaryOp):
			if node.op == "*":
				inner = self._infer_expr_type(node.operand)
				if inner and inner.pointer_count > 0:
					return TypeSpec(
						base_type=inner.base_type,
						pointer_count=inner.pointer_count - 1,
						width_modifier=inner.width_modifier,
						signedness=inner.signedness,
					)
			elif node.op == "&":
				inner = self._infer_expr_type(node.operand)
				if inner:
					return TypeSpec(
						base_type=inner.base_type,
						pointer_count=inner.pointer_count + 1,
						width_modifier=inner.width_modifier,
						signedness=inner.signedness,
					)
			return self._infer_expr_type(node.operand)
		if isinstance(node, ArraySubscript):
			arr_type = self._infer_expr_type(node.array)
			if arr_type and arr_type.pointer_count > 0:
				return TypeSpec(
					base_type=arr_type.base_type,
					pointer_count=arr_type.pointer_count - 1,
					width_modifier=arr_type.width_modifier,
					signedness=arr_type.signedness,
				)
			return arr_type
		if isinstance(node, MemberAccess):
			obj_type = self._infer_expr_type(node.object)
			if obj_type:
				key = obj_type.base_type
				if key.startswith("struct "):
					key = key[len("struct "):]
				members = self._structs.get(key)
				if members:
					for m in members:
						if m.name == node.member:
							return m.type_spec
		if isinstance(node, SizeofExpr):
			return TypeSpec(base_type="int")
		if isinstance(node, PostfixExpr):
			return self._infer_expr_type(node.operand)
		return None

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
		# sizeof(expr) -- infer the expression's type
		if node.operand is not None:
			# sizeof(array_variable) returns total array size
			if isinstance(node.operand, Identifier):
				name = node.operand.name
				ts = self._local_types.get(name) or self._global_types.get(name)
				if ts is not None:
					dims = self._local_array.get(name) or self._global_array.get(name)
					if dims:
						elem_size = self._resolve_member_size(ts)
						total = elem_size
						for d in dims:
							total *= d
						return IRConst(total)
					return IRConst(self._resolve_member_size(ts))
			ts = self._infer_expr_type(node.operand)
			if ts is not None:
				return IRConst(self._resolve_member_size(ts))
		return IRConst(4)

	def _resolve_type_size(self, ts: TypeSpec) -> int:
		"""Resolve the size of a type, handling struct/union types."""
		return self._resolve_member_size(ts)

	def _compute_bitfield_layout(self, name: str) -> None:
		"""Pre-compute bitfield layout for a struct, storing byte_offset, bit_offset, bit_width, storage_size."""
		members = self._structs.get(name, [])
		has_bitfields = any(m.bit_width is not None for m in members)
		if not has_bitfields:
			return
		layout: dict[str, tuple[int, int, int, int]] = {}
		byte_offset = 0
		bit_offset = 0  # bits used within current storage unit
		storage_size = 0  # current storage unit size in bytes
		max_align = 1
		for m in members:
			if m.bit_width is not None:
				type_size = _resolve_size(m.type_spec)
				if m.bit_width == 0:
					# Zero-width bitfield: force alignment to next storage unit boundary
					if bit_offset > 0:
						byte_offset += storage_size
						bit_offset = 0
						storage_size = 0
					continue
				if storage_size == 0:
					# Start new storage unit
					align = _resolve_alignment(m.type_spec)
					max_align = max(max_align, align)
					byte_offset = _align_to(byte_offset, align)
					storage_size = type_size
					bit_offset = 0
				elif bit_offset + m.bit_width > storage_size * 8:
					# Doesn't fit in current storage unit, start new one
					byte_offset += storage_size
					align = _resolve_alignment(m.type_spec)
					max_align = max(max_align, align)
					byte_offset = _align_to(byte_offset, align)
					storage_size = type_size
					bit_offset = 0
				if m.name:
					layout[m.name] = (byte_offset, bit_offset, m.bit_width, storage_size)
				bit_offset += m.bit_width
			else:
				# Non-bitfield member: flush any pending bitfield storage
				if bit_offset > 0:
					byte_offset += storage_size
					bit_offset = 0
					storage_size = 0
				align = self._resolve_type_alignment(m.type_spec)
				max_align = max(max_align, align)
				byte_offset = _align_to(byte_offset, align)
				byte_offset += self._resolve_full_member_size(m)
		self._bitfield_layouts[name] = layout

	def _compute_struct_size(self, name: str) -> int:
		"""Compute the total size of a struct including alignment padding."""
		members = self._structs.get(name, [])
		has_bitfields = any(m.bit_width is not None for m in members)
		if not has_bitfields:
			offset = 0
			max_align = 1
			for m in members:
				align = self._resolve_type_alignment(m.type_spec)
				max_align = max(max_align, align)
				offset = _align_to(offset, align)
				offset += self._resolve_full_member_size(m)
			return _align_to(offset, max_align)
		# Bitfield-aware size computation
		byte_offset = 0
		bit_offset = 0
		storage_size = 0
		max_align = 1
		for m in members:
			if m.bit_width is not None:
				type_size = _resolve_size(m.type_spec)
				if m.bit_width == 0:
					if bit_offset > 0:
						byte_offset += storage_size
						bit_offset = 0
						storage_size = 0
					continue
				if storage_size == 0:
					align = _resolve_alignment(m.type_spec)
					max_align = max(max_align, align)
					byte_offset = _align_to(byte_offset, align)
					storage_size = type_size
					bit_offset = 0
				elif bit_offset + m.bit_width > storage_size * 8:
					byte_offset += storage_size
					align = _resolve_alignment(m.type_spec)
					max_align = max(max_align, align)
					byte_offset = _align_to(byte_offset, align)
					storage_size = type_size
					bit_offset = 0
				bit_offset += m.bit_width
			else:
				if bit_offset > 0:
					byte_offset += storage_size
					bit_offset = 0
					storage_size = 0
				align = self._resolve_type_alignment(m.type_spec)
				max_align = max(max_align, align)
				byte_offset = _align_to(byte_offset, align)
				byte_offset += self._resolve_full_member_size(m)
		if bit_offset > 0:
			byte_offset += storage_size
		return _align_to(byte_offset, max_align)

	def _compute_union_size(self, name: str) -> int:
		"""Compute the size of a union (largest member, padded to max alignment)."""
		members = self._unions.get(name, [])
		if not members:
			return 0
		max_size = 0
		max_align = 1
		for m in members:
			max_size = max(max_size, self._resolve_full_member_size(m))
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

	def _resolve_full_member_size(self, m) -> int:
		"""Resolve the total size of a struct member including array dimensions."""
		base_size = self._resolve_member_size(m.type_spec)
		if m.array_dims:
			for dim in m.array_dims:
				dim_val = self._eval_const_expr(dim)
				if dim_val is not None and dim_val > 0:
					base_size *= dim_val
		return base_size

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
		if isinstance(node.operand, Identifier) and node.operand.name in self._static_local_map:
			mangled = self._static_local_map[node.operand.name]
			ir_type = self._resolve_local_ir_type(node.operand.name)
			old_val = self._new_temp()
			self._set_temp_type(old_val, ir_type)
			self._emit(IRLoad(dest=old_val, address=IRGlobalRef(mangled), ir_type=ir_type))
			new_val = self._new_temp()
			self._set_temp_type(new_val, ir_type)
			delta_op = "+" if node.op == "++" else "-"
			delta_val = IRConst(self._local_pointee_size(node.operand.name)) if ir_type == IRType.POINTER else IRConst(1)
			self._emit(IRBinOp(dest=new_val, left=old_val, op=delta_op, right=delta_val, ir_type=ir_type))
			self._emit(IRStore(address=IRGlobalRef(mangled), value=new_val, ir_type=ir_type))
			return old_val
		if isinstance(node.operand, Identifier):
			target = self._locals.get(node.operand.name)
			if target is not None:
				ir_type = self._resolve_local_ir_type(node.operand.name)
				old_val = self._new_temp()
				self._set_temp_type(old_val, ir_type)
				self._emit(IRCopy(dest=old_val, source=target, ir_type=ir_type))
				new_val = self._new_temp()
				self._set_temp_type(new_val, ir_type)
				delta_op = "+" if node.op == "++" else "-"
				if ir_type == IRType.POINTER:
					delta_val: IRValue = IRConst(self._local_pointee_size(node.operand.name))
				elif self._is_float_type(ir_type):
					delta_val = IRFloatConst(1.0, ir_type=ir_type)
				else:
					delta_val = IRConst(1)
				self._emit(IRBinOp(dest=new_val, left=old_val, op=delta_op, right=delta_val, ir_type=ir_type))
				self._emit(IRCopy(dest=target, source=new_val, ir_type=ir_type))
				return old_val
			# Global variable postfix ++/--
			if node.operand.name in self._global_names:
				ir_type = _resolve_ir_type(self._global_types.get(node.operand.name, TypeSpec(base_type="int")))
				old_val = self._new_temp()
				self._set_temp_type(old_val, ir_type)
				self._emit(IRLoad(dest=old_val, address=IRGlobalRef(node.operand.name), ir_type=ir_type))
				new_val = self._new_temp()
				self._set_temp_type(new_val, ir_type)
				delta_op = "+" if node.op == "++" else "-"
				if self._is_float_type(ir_type):
					delta_val2: IRValue = IRFloatConst(1.0, ir_type=ir_type)
				else:
					delta_val2 = IRConst(1)
				self._emit(IRBinOp(dest=new_val, left=old_val, op=delta_op, right=delta_val2, ir_type=ir_type))
				self._emit(IRStore(address=IRGlobalRef(node.operand.name), value=new_val, ir_type=ir_type))
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
		if isinstance(node.operand, MemberAccess):
			bf_info = self._get_bitfield_info(node.operand)
			if bf_info is not None:
				old_val = self._bitfield_read(node.operand, bf_info)
				new_val = self._new_temp()
				delta_op = "+" if node.op == "++" else "-"
				self._emit(IRBinOp(dest=new_val, left=old_val, op=delta_op, right=IRConst(1)))
				self._bitfield_write(node.operand, bf_info, new_val)
				return old_val
			addr = self._compute_member_addr(node.operand)
			member_type = self._member_ir_type(node.operand)
			old_val = self._new_temp()
			self._emit(IRLoad(dest=old_val, address=addr, ir_type=member_type))
			new_val = self._new_temp()
			delta_op = "+" if node.op == "++" else "-"
			self._emit(IRBinOp(dest=new_val, left=old_val, op=delta_op, right=IRConst(1)))
			addr2 = self._compute_member_addr(node.operand)
			self._emit(IRStore(address=addr2, value=new_val, ir_type=member_type))
			return old_val
		if isinstance(node.operand, UnaryOp) and node.operand.op == "*":
			addr = self.visit(node.operand.operand)
			old_val = self._new_temp()
			self._emit(IRLoad(dest=old_val, address=addr))
			new_val = self._new_temp()
			delta_op = "+" if node.op == "++" else "-"
			self._emit(IRBinOp(dest=new_val, left=old_val, op=delta_op, right=IRConst(1)))
			addr2 = self.visit(node.operand.operand)
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
			if const.value is not None:
				evaluated = self._eval_const_expr(const.value)
				if evaluated is not None:
					next_value = evaluated
			self._enum_constants[const.name] = next_value
			next_value += 1

	def visit_static_assert_decl(self, node: StaticAssertDecl) -> None:
		pass  # compile-time only; already checked by semantic analyzer

	def visit_typedef_decl(self, node: TypedefDecl) -> None:
		if node.struct_decl is not None:
			self.visit_struct_decl(node.struct_decl)
		if node.enum_decl is not None:
			self.visit_enum_decl(node.enum_decl)
		if node.union_decl is not None:
			self.visit_union_decl(node.union_decl)

	def visit_struct_decl(self, node: StructDecl) -> None:
		self._structs[node.name] = list(node.members)
		self._compute_bitfield_layout(node.name)

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
			element_ir_type = _resolve_ir_type(node.type_spec)
			# Determine total array size
			total_elements = 0
			if node.array_sizes:
				for se in node.array_sizes:
					val = self._eval_const_expr(se)
					if val is not None:
						total_elements = val
			total_elements = max(total_elements, len(init_list.elements))

			has_designated = any(isinstance(e, DesignatedInit) for e in init_list.elements)
			if has_designated:
				# Zero-fill the whole array first, then apply designated values
				for i in range(total_elements):
					off = self._new_temp()
					self._emit(IRBinOp(dest=off, left=IRConst(i), op="*", right=IRConst(element_size)))
					a = self._new_temp()
					self._emit(IRBinOp(dest=a, left=dest, op="+", right=off))
					self._emit(IRStore(address=a, value=IRConst(0), ir_type=element_ir_type))
				positional_idx = 0
				for elem in init_list.elements:
					if isinstance(elem, DesignatedInit):
						assert elem.index is not None
						if isinstance(elem.index, IntLiteral):
							idx = elem.index.value
						else:
							idx = positional_idx
						val = self.visit(elem.value)
						off = self._new_temp()
						self._emit(IRBinOp(dest=off, left=IRConst(idx), op="*", right=IRConst(element_size)))
						a = self._new_temp()
						self._emit(IRBinOp(dest=a, left=dest, op="+", right=off))
						self._emit(IRStore(address=a, value=val, ir_type=element_ir_type))
						positional_idx = idx + 1
					else:
						val = self.visit(elem)
						off = self._new_temp()
						self._emit(IRBinOp(dest=off, left=IRConst(positional_idx), op="*", right=IRConst(element_size)))
						a = self._new_temp()
						self._emit(IRBinOp(dest=a, left=dest, op="+", right=off))
						self._emit(IRStore(address=a, value=val, ir_type=element_ir_type))
						positional_idx += 1
			else:
				for i in range(total_elements):
					if i < len(init_list.elements):
						val = self.visit(init_list.elements[i])
					else:
						val = IRConst(0)
					offset = self._new_temp()
					self._emit(IRBinOp(dest=offset, left=IRConst(i), op="*", right=IRConst(element_size)))
					addr = self._new_temp()
					self._emit(IRBinOp(dest=addr, left=dest, op="+", right=offset))
					self._emit(IRStore(address=addr, value=val, ir_type=element_ir_type))
		elif base_type.startswith("struct "):
			struct_name = base_type[len("struct "):]
			members = self._structs.get(struct_name, [])
			has_designated = any(isinstance(e, DesignatedInit) for e in init_list.elements)

			if has_designated:
				# Zero-fill the whole struct first
				self._zero_fill_struct(dest, members)
				positional_idx = 0
				for elem in init_list.elements:
					if isinstance(elem, DesignatedInit):
						assert elem.field_name is not None
						field_offset = self._compute_field_offset(struct_name, elem.field_name)
						val = self.visit(elem.value)
						a = self._new_temp()
						self._emit(IRBinOp(dest=a, left=dest, op="+", right=IRConst(field_offset)))
						self._emit(IRStore(address=a, value=val))
						# Update positional index to field after designated one
						for mi, m in enumerate(members):
							if m.name == elem.field_name:
								positional_idx = mi + 1
								break
					else:
						if positional_idx >= len(members):
							break
						m = members[positional_idx]
						align = self._resolve_type_alignment(m.type_spec)
						field_offset = self._compute_field_offset_by_index(struct_name, positional_idx)
						if isinstance(elem, InitializerList):
							member_addr = self._new_temp()
							self._emit(IRBinOp(dest=member_addr, left=dest, op="+", right=IRConst(field_offset)))
							member_size = self._resolve_member_size(m.type_spec)
							for j, sub_elem in enumerate(elem.elements):
								sub_val = self.visit(sub_elem)
								sub_off = self._new_temp()
								self._emit(IRBinOp(dest=sub_off, left=IRConst(j), op="*", right=IRConst(member_size)))
								sub_addr = self._new_temp()
								self._emit(IRBinOp(dest=sub_addr, left=member_addr, op="+", right=sub_off))
								self._emit(IRStore(address=sub_addr, value=sub_val))
						else:
							val = self.visit(elem)
							a = self._new_temp()
							self._emit(IRBinOp(dest=a, left=dest, op="+", right=IRConst(field_offset)))
							self._emit(IRStore(address=a, value=val))
						positional_idx += 1
			else:
				bf_layout = self._bitfield_layouts.get(struct_name, {})
				has_bitfields = any(m.bit_width is not None for m in members)
				if has_bitfields and bf_layout:
					# Pack bitfield values into storage units
					# First, zero-fill entire struct
					struct_size = self._compute_struct_size(struct_name)
					for off in range(0, struct_size, 4):
						a = self._new_temp()
						self._emit(IRBinOp(dest=a, left=dest, op="+", right=IRConst(off)))
						sz = min(4, struct_size - off)
						self._emit(IRStore(address=a, value=IRConst(0), ir_type=IRType.INT if sz >= 4 else IRType.CHAR))
					# Group bitfield init values by storage unit byte_offset
					storage_units: dict[int, int] = {}  # byte_offset -> packed value
					for i, elem in enumerate(init_list.elements):
						if i >= len(members):
							break
						m = members[i]
						if m.bit_width is not None and m.name and m.name in bf_layout:
							byte_off, bit_off, bit_width, storage_sz = bf_layout[m.name]
							const_val = self._eval_const_expr(elem) if hasattr(elem, 'accept') else None
							if const_val is None and isinstance(elem, IntLiteral):
								const_val = elem.value
							if const_val is not None:
								mask = (1 << bit_width) - 1
								packed = (const_val & mask) << bit_off
								storage_units[byte_off] = storage_units.get(byte_off, 0) | packed
						else:
							# Non-bitfield member
							field_offset = self._compute_field_offset_by_index(struct_name, i)
							if isinstance(elem, InitializerList):
								member_addr = self._new_temp()
								self._emit(IRBinOp(dest=member_addr, left=dest, op="+", right=IRConst(field_offset)))
								member_size = self._resolve_member_size(m.type_spec)
								for j, sub_elem in enumerate(elem.elements):
									sub_val = self.visit(sub_elem)
									sub_off = self._new_temp()
									self._emit(IRBinOp(dest=sub_off, left=IRConst(j), op="*", right=IRConst(member_size)))
									sub_addr = self._new_temp()
									self._emit(IRBinOp(dest=sub_addr, left=member_addr, op="+", right=sub_off))
									self._emit(IRStore(address=sub_addr, value=sub_val))
							else:
								val = self.visit(elem)
								a = self._new_temp()
								self._emit(IRBinOp(dest=a, left=dest, op="+", right=IRConst(field_offset)))
								self._emit(IRStore(address=a, value=val))
					# Write packed bitfield storage units
					for byte_off, packed_val in storage_units.items():
						a = self._new_temp()
						self._emit(IRBinOp(dest=a, left=dest, op="+", right=IRConst(byte_off)))
						self._emit(IRStore(address=a, value=IRConst(packed_val), ir_type=IRType.INT))
				else:
					field_offset = 0
					for i, elem in enumerate(init_list.elements):
						if i >= len(members):
							break
						align = self._resolve_type_alignment(members[i].type_spec)
						field_offset = _align_to(field_offset, align)
						if isinstance(elem, InitializerList):
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
						field_offset += self._resolve_full_member_size(members[i])
					# Zero-fill remaining members
					for i in range(len(init_list.elements), len(members)):
						align = self._resolve_type_alignment(members[i].type_spec)
						field_offset = _align_to(field_offset, align)
						addr = self._new_temp()
						self._emit(IRBinOp(dest=addr, left=dest, op="+", right=IRConst(field_offset)))
						self._emit(IRStore(address=addr, value=IRConst(0)))
						field_offset += self._resolve_full_member_size(members[i])

	def _is_local_aggregate(self, name: str) -> str | None:
		"""Return the struct/union name if a local variable is an aggregate type, else None."""
		ts = self._local_types.get(name)
		if ts is None or ts.pointer_count > 0:
			return None
		base = ts.base_type
		if base.startswith("struct "):
			sname = base[len("struct "):]
			if sname in self._structs:
				return sname
		if base.startswith("union "):
			uname = base[len("union "):]
			if uname in self._unions:
				return uname
		return None

	def _is_global_aggregate(self, name: str) -> str | None:
		"""Return the struct/union name if a global variable is an aggregate type, else None."""
		ts = self._global_types.get(name)
		if ts is None or ts.pointer_count > 0:
			return None
		base = ts.base_type
		if base.startswith("struct "):
			sname = base[len("struct "):]
			if sname in self._structs:
				return sname
		if base.startswith("union "):
			uname = base[len("union "):]
			if uname in self._unions:
				return uname
		return None

	def _deref_aggregate_name(self, expr: ASTNode) -> str | None:
		"""If expr is a pointer-to-struct/union, return the aggregate name, else None."""
		ts = None
		if isinstance(expr, Identifier):
			ts = self._local_types.get(expr.name) or self._global_types.get(expr.name)
		elif isinstance(expr, MemberAccess):
			ts = self._resolve_member_type_spec(expr)
		elif isinstance(expr, CastExpr):
			ts = expr.target_type
		if ts is not None and ts.pointer_count >= 1:
			base = ts.base_type
			if base.startswith("struct "):
				sname = base[len("struct "):]
				if sname in self._structs:
					return sname
			if base.startswith("union "):
				uname = base[len("union "):]
				if uname in self._unions:
					return uname
		return None

	def _emit_aggregate_copy(self, dest_addr: IRTemp, src_addr: IRTemp, type_name: str) -> None:
		"""Emit a bulk copy IR instruction for copying a struct or union."""
		is_union = type_name in self._unions
		if is_union:
			size = self._compute_union_size(type_name)
		else:
			size = self._compute_struct_size(type_name)
		self._emit(IRBulkCopy(dest_addr=dest_addr, src_addr=src_addr, size=size))

	def _emit_memcopy_by_words(self, dest_addr: IRTemp, src_addr: IRTemp, size: int) -> None:
		"""Copy 'size' bytes from src to dest using a bulk copy instruction."""
		self._emit(IRBulkCopy(dest_addr=dest_addr, src_addr=src_addr, size=size))

	def _zero_fill_struct(self, dest: IRTemp, members: list[StructMember]) -> None:
		"""Zero-fill all members of a struct."""
		field_offset = 0
		for m in members:
			align = self._resolve_type_alignment(m.type_spec)
			field_offset = _align_to(field_offset, align)
			addr = self._new_temp()
			self._emit(IRBinOp(dest=addr, left=dest, op="+", right=IRConst(field_offset)))
			self._emit(IRStore(address=addr, value=IRConst(0)))
			field_offset += self._resolve_full_member_size(m)

	def _compute_field_offset_by_index(self, struct_name: str, field_index: int) -> int:
		"""Compute the byte offset of a struct field by its index."""
		members = self._structs.get(struct_name, [])
		# Check bitfield layout
		bf_layout = self._bitfield_layouts.get(struct_name)
		if bf_layout and field_index < len(members):
			m = members[field_index]
			if m.name and m.name in bf_layout:
				return bf_layout[m.name][0]
		has_bitfields = any(m.bit_width is not None for m in members)
		if not has_bitfields:
			offset = 0
			for i, m in enumerate(members):
				align = self._resolve_type_alignment(m.type_spec)
				offset = _align_to(offset, align)
				if i == field_index:
					return offset
				offset += self._resolve_full_member_size(m)
			return offset
		# Bitfield-aware computation
		byte_offset = 0
		bit_offset = 0
		storage_size = 0
		for i, m in enumerate(members):
			if m.bit_width is not None:
				type_size = _resolve_size(m.type_spec)
				if m.bit_width == 0:
					if bit_offset > 0:
						byte_offset += storage_size
						bit_offset = 0
						storage_size = 0
					continue
				if storage_size == 0:
					align = _resolve_alignment(m.type_spec)
					byte_offset = _align_to(byte_offset, align)
					storage_size = type_size
					bit_offset = 0
				elif bit_offset + m.bit_width > storage_size * 8:
					byte_offset += storage_size
					align = _resolve_alignment(m.type_spec)
					byte_offset = _align_to(byte_offset, align)
					storage_size = type_size
					bit_offset = 0
				if i == field_index:
					return byte_offset
				bit_offset += m.bit_width
			else:
				if bit_offset > 0:
					byte_offset += storage_size
					bit_offset = 0
					storage_size = 0
				align = self._resolve_type_alignment(m.type_spec)
				byte_offset = _align_to(byte_offset, align)
				if i == field_index:
					return byte_offset
				byte_offset += self._resolve_full_member_size(m)
		return byte_offset

	def visit_designated_init(self, node: DesignatedInit) -> IRValue:
		return self.visit(node.value)

	def _emit_char_array_from_string(self, dest: IRTemp, node: VarDecl) -> None:
		"""Emit byte-wise stores for a char array initialized from a string literal."""
		assert isinstance(node.initializer, StringLiteral)
		string_val = node.initializer.value
		# Determine the total array size from array_sizes
		total_size = 0
		if node.array_sizes:
			for se in node.array_sizes:
				val = self._eval_const_expr(se)
				if val is not None:
					total_size = val
		# String bytes + null terminator
		string_bytes = [ord(ch) for ch in string_val] + [0]
		for i in range(total_size):
			val = IRConst(string_bytes[i] if i < len(string_bytes) else 0)
			addr = self._new_temp()
			self._emit(IRBinOp(dest=addr, left=dest, op="+", right=IRConst(i)))
			self._emit(IRStore(address=addr, value=val, ir_type=IRType.CHAR))

	def _eval_const_expr(self, node: ASTNode) -> int | None:
		"""Evaluate a compile-time constant expression, returning an int or None."""
		if isinstance(node, IntLiteral):
			return node.value
		if isinstance(node, CharLiteral):
			return ord(node.value)
		if isinstance(node, Identifier):
			if node.name in self._enum_constants:
				return self._enum_constants[node.name]
			return None
		if isinstance(node, UnaryOp):
			operand = self._eval_const_expr(node.operand)
			if operand is None:
				return None
			if node.op == "-":
				return -operand
			if node.op == "+":
				return operand
			if node.op == "~":
				return ~operand
			if node.op == "!":
				return 0 if operand else 1
			return None
		if isinstance(node, BinaryOp):
			left = self._eval_const_expr(node.left)
			right = self._eval_const_expr(node.right)
			if left is None or right is None:
				return None
			if node.op == "+":
				return left + right
			if node.op == "-":
				return left - right
			if node.op == "*":
				return left * right
			if node.op == "/" and right != 0:
				return int(left / right)
			if node.op == "%" and right != 0:
				return left % right
			if node.op == "<<":
				return left << right
			if node.op == ">>":
				return left >> right
			if node.op == "&":
				return left & right
			if node.op == "|":
				return left | right
			if node.op == "^":
				return left ^ right
			if node.op == "&&":
				return 1 if (left and right) else 0
			if node.op == "||":
				return 1 if (left or right) else 0
			if node.op == "==":
				return 1 if left == right else 0
			if node.op == "!=":
				return 1 if left != right else 0
			if node.op == "<":
				return 1 if left < right else 0
			if node.op == ">":
				return 1 if left > right else 0
			if node.op == "<=":
				return 1 if left <= right else 0
			if node.op == ">=":
				return 1 if left >= right else 0
			return None
		if isinstance(node, SizeofExpr):
			if node.type_operand is not None:
				ts = node.type_operand
				key = ts.base_type
				if key.startswith("struct "):
					key = key[len("struct "):]
				elif key.startswith("union "):
					key = key[len("union "):]
				if ts.pointer_count == 0:
					if key in self._structs:
						return self._compute_struct_size(key)
					if key in self._unions:
						return self._compute_union_size(key)
				return _resolve_size(ts)
			if node.operand is not None:
				if isinstance(node.operand, Identifier):
					name = node.operand.name
					ts = self._local_types.get(name) or self._global_types.get(name)
					if ts is not None:
						dims = self._local_array.get(name) or self._global_array.get(name)
						if dims:
							total = self._resolve_member_size(ts)
							for d in dims:
								total *= d
							return total
						return self._resolve_member_size(ts)
				ts = self._infer_expr_type(node.operand)
				if ts is not None:
					return self._resolve_member_size(ts)
			return 4
		if isinstance(node, CastExpr):
			return self._eval_const_expr(node.operand)
		if isinstance(node, TernaryExpr):
			cond = self._eval_const_expr(node.condition)
			if cond is None:
				return None
			return self._eval_const_expr(node.true_expr if cond else node.false_expr)
		return None

	def _collect_init_values(self, init_list: InitializerList) -> list[int]:
		"""Collect constant integer values from an initializer list (for globals)."""
		values: list[int] = []
		for elem in init_list.elements:
			if isinstance(elem, InitializerList):
				values.extend(self._collect_init_values(elem))
			else:
				result = self._eval_const_expr(elem)
				values.append(result if result is not None else 0)
		return values

	def _resolve_member_type_spec(self, node: MemberAccess) -> TypeSpec | None:
		"""Resolve the TypeSpec of the accessed member from struct/union definitions."""
		type_name = self._resolve_aggregate_name(node.object)
		if not type_name:
			return None
		members = self._structs.get(type_name) or self._unions.get(type_name, [])
		for m in members:
			if m.name == node.member:
				return m.type_spec
		return None

	def _member_is_aggregate(self, ts: TypeSpec | None) -> bool:
		"""Check if a TypeSpec refers to a struct or union (not a pointer to one)."""
		if ts is None or ts.pointer_count > 0:
			return False
		base = ts.base_type
		if base.startswith("struct "):
			return base[len("struct "):] in self._structs
		if base.startswith("union "):
			return base[len("union "):] in self._unions
		return False

	def _member_is_array(self, node: MemberAccess) -> bool:
		"""Check if a member access refers to an array member (has array_dims)."""
		type_name = self._resolve_aggregate_name(node.object)
		if not type_name:
			return False
		members = self._structs.get(type_name) or self._unions.get(type_name, [])
		for m in members:
			if m.name == node.member and m.array_dims:
				return True
		return False

	def _member_ir_type(self, node: MemberAccess) -> IRType:
		"""Resolve the IRType for a struct/union member access."""
		ts = self._resolve_member_type_spec(node)
		if ts is not None:
			return _resolve_ir_type(ts)
		return IRType.INT

	def _compute_member_addr(self, node: MemberAccess) -> IRTemp:
		"""Compute the memory address of a struct/union member."""
		if node.is_arrow:
			# Arrow access: object is a pointer, visit to get pointer value (= struct address)
			base = self.visit(node.object)
		elif isinstance(node.object, UnaryOp) and node.object.op == "*":
			# (*ptr).member is equivalent to ptr->member: use pointer value as base
			base = self.visit(node.object.operand)
		elif isinstance(node.object, ArraySubscript):
			# Dot access on array element: need address, not loaded value
			base = self._compute_array_addr(node.object)
		elif isinstance(node.object, MemberAccess):
			# Nested dot access: get address of parent member
			base = self._compute_member_addr(node.object)
		else:
			# Identifier (local/global struct) - visit returns address for aggregates
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

	def _get_bitfield_info(self, node: MemberAccess) -> tuple[int, int, int, int] | None:
		"""Return (byte_offset, bit_offset, bit_width, storage_size) if member is a bitfield."""
		type_name = self._resolve_aggregate_name(node.object)
		if not type_name:
			return None
		bf_layout = self._bitfield_layouts.get(type_name)
		if bf_layout and node.member in bf_layout:
			return bf_layout[node.member]
		return None

	def _bitfield_read(self, node: MemberAccess, bf_info: tuple[int, int, int, int]) -> IRTemp:
		"""Generate IR for reading a bitfield: load storage unit, shift right, mask, sign-extend."""
		byte_offset, bit_offset, bit_width, storage_size = bf_info
		storage_ir_type = {1: IRType.CHAR, 2: IRType.SHORT, 4: IRType.INT, 8: IRType.LONG}.get(
			storage_size, IRType.INT
		)
		addr = self._compute_member_addr(node)
		raw = self._new_temp()
		self._emit(IRLoad(dest=raw, address=addr, ir_type=storage_ir_type))
		if bit_offset > 0:
			shifted = self._new_temp()
			self._emit(IRBinOp(dest=shifted, left=raw, op=">>", right=IRConst(bit_offset), is_unsigned=True))
		else:
			shifted = raw
		mask_val = (1 << bit_width) - 1
		result = self._new_temp()
		self._emit(IRBinOp(dest=result, left=shifted, op="&", right=IRConst(mask_val)))
		# Sign-extend for signed bitfields
		is_unsigned = False
		member_ts = self._resolve_member_type_spec(node)
		if member_ts is not None and member_ts.signedness == "unsigned":
			is_unsigned = True
		if not is_unsigned and bit_width < 32:
			# Sign-extend: shift left then arithmetic shift right
			shift_amt = 64 - bit_width
			shl = self._new_temp()
			self._emit(IRBinOp(dest=shl, left=result, op="<<", right=IRConst(shift_amt)))
			sar = self._new_temp()
			self._emit(IRBinOp(dest=sar, left=shl, op=">>", right=IRConst(shift_amt)))
			return sar
		return result

	def _bitfield_write(self, node: MemberAccess, bf_info: tuple[int, int, int, int], value: IRValue) -> None:
		"""Generate IR for writing a bitfield: load, clear bits, shift value, OR, store."""
		byte_offset, bit_offset, bit_width, storage_size = bf_info
		storage_ir_type = {1: IRType.CHAR, 2: IRType.SHORT, 4: IRType.INT, 8: IRType.LONG}.get(
			storage_size, IRType.INT
		)
		addr = self._compute_member_addr(node)
		# Load current storage unit
		old = self._new_temp()
		self._emit(IRLoad(dest=old, address=addr, ir_type=storage_ir_type))
		# Clear the bitfield bits: old & ~(mask << bit_offset)
		mask_val = (1 << bit_width) - 1
		clear_mask = ~(mask_val << bit_offset)
		cleared = self._new_temp()
		self._emit(IRBinOp(dest=cleared, left=old, op="&", right=IRConst(clear_mask)))
		# Mask and shift the new value: (value & mask) << bit_offset
		masked_val = self._new_temp()
		self._emit(IRBinOp(dest=masked_val, left=value, op="&", right=IRConst(mask_val)))
		if bit_offset > 0:
			shifted_val = self._new_temp()
			self._emit(IRBinOp(dest=shifted_val, left=masked_val, op="<<", right=IRConst(bit_offset)))
		else:
			shifted_val = masked_val
		# OR them together and store
		combined = self._new_temp()
		self._emit(IRBinOp(dest=combined, left=cleared, op="|", right=shifted_val))
		addr2 = self._compute_member_addr(node)
		self._emit(IRStore(address=addr2, value=combined, ir_type=storage_ir_type))

	def visit_member_access(self, node: MemberAccess) -> IRTemp:
		# Check for bitfield access
		bf_info = self._get_bitfield_info(node)
		if bf_info is not None:
			return self._bitfield_read(node, bf_info)
		addr = self._compute_member_addr(node)
		member_ts = self._resolve_member_type_spec(node)
		# For struct/union-typed members, return the address directly (no scalar load)
		if self._member_is_aggregate(member_ts):
			return addr
		# For array members, return address (array decays to pointer)
		if self._member_is_array(node):
			self._set_temp_type(addr, IRType.POINTER)
			return addr
		member_type = _resolve_ir_type(member_ts) if member_ts is not None else IRType.INT
		dest = self._new_temp()
		self._set_temp_type(dest, member_type)
		self._emit(IRLoad(dest=dest, address=addr, ir_type=member_type))
		return dest

	def visit_cast_expr(self, node: CastExpr) -> IRTemp:
		"""Handle casts, including int<->float conversions."""
		val = self.visit(node.operand)
		val_type = self._value_ir_type(val)
		target_ir_type = _resolve_ir_type(node.target_type)
		dest = self._new_temp()
		self._set_temp_type(dest, target_ir_type)
		# Track signedness change even for same-width casts
		cast_unsigned = _is_type_unsigned(node.target_type)
		if cast_unsigned:
			self._temp_unsigned[dest.name] = True
		if self._is_float_type(target_ir_type) != self._is_float_type(val_type):
			self._emit(IRConvert(dest=dest, source=val, from_type=val_type, to_type=target_ir_type, is_unsigned=cast_unsigned))
		elif self._is_float_type(target_ir_type) and self._is_float_type(val_type) and target_ir_type != val_type:
			self._emit(IRConvert(dest=dest, source=val, from_type=val_type, to_type=target_ir_type))
		elif val_type != target_ir_type:
			self._emit(IRConvert(dest=dest, source=val, from_type=val_type, to_type=target_ir_type, is_unsigned=cast_unsigned))
		elif target_ir_type in (IRType.CHAR, IRType.SHORT, IRType.INT):
			# Same type but may need sign/zero extension or truncation
			self._emit(IRConvert(dest=dest, source=val, from_type=val_type, to_type=target_ir_type, is_unsigned=cast_unsigned))
		else:
			self._emit(IRCopy(dest=dest, source=val, ir_type=target_ir_type))
		return dest

	def visit_compound_literal(self, node: CompoundLiteral) -> IRTemp:
		"""Handle compound literals: allocate stack space, initialize, return address."""
		ts = node.type_spec
		base_type = ts.base_type
		dest = self._new_temp()
		self._set_temp_type(dest, IRType.POINTER)

		if base_type.startswith("struct "):
			struct_name = base_type[len("struct "):]
			members = self._structs.get(struct_name, [])
			total_size = self._compute_struct_size(struct_name)
			self._emit(IRAlloc(dest=dest, size=total_size))
			# Emit stores for each element
			has_designated = any(isinstance(e, DesignatedInit) for e in node.init_list.elements)
			if has_designated:
				self._zero_fill_struct(dest, members)
				positional_idx = 0
				for elem in node.init_list.elements:
					if isinstance(elem, DesignatedInit) and elem.field_name is not None:
						field_offset = self._compute_field_offset(struct_name, elem.field_name)
						val = self.visit(elem.value)
						a = self._new_temp()
						self._emit(IRBinOp(dest=a, left=dest, op="+", right=IRConst(field_offset)))
						self._emit(IRStore(address=a, value=val))
						for mi, m in enumerate(members):
							if m.name == elem.field_name:
								positional_idx = mi + 1
								break
					else:
						if positional_idx >= len(members):
							break
						field_offset = self._compute_field_offset_by_index(struct_name, positional_idx)
						val = self.visit(elem)
						a = self._new_temp()
						self._emit(IRBinOp(dest=a, left=dest, op="+", right=IRConst(field_offset)))
						self._emit(IRStore(address=a, value=val))
						positional_idx += 1
			else:
				for i, elem in enumerate(node.init_list.elements):
					if i >= len(members):
						break
					field_offset = self._compute_field_offset_by_index(struct_name, i)
					val = self.visit(elem)
					a = self._new_temp()
					self._emit(IRBinOp(dest=a, left=dest, op="+", right=IRConst(field_offset)))
					self._emit(IRStore(address=a, value=val))
		elif base_type.startswith("union "):
			union_name = base_type[len("union "):]
			total_size = self._compute_union_size(union_name)
			self._emit(IRAlloc(dest=dest, size=total_size))
			if node.init_list.elements:
				elem = node.init_list.elements[0]
				if isinstance(elem, DesignatedInit) and elem.field_name is not None:
					val = self.visit(elem.value)
				else:
					val = self.visit(elem)
				self._emit(IRStore(address=dest, value=val))
		else:
			# Scalar or array compound literal
			element_size = _resolve_size(ts)
			num_elements = len(node.init_list.elements)
			if num_elements <= 1:
				total_size = element_size
			else:
				total_size = element_size * num_elements
			self._emit(IRAlloc(dest=dest, size=total_size))
			for i, elem in enumerate(node.init_list.elements):
				val = self.visit(elem)
				if num_elements > 1:
					offset = self._new_temp()
					self._emit(IRBinOp(dest=offset, left=IRConst(i), op="*", right=IRConst(element_size)))
					addr = self._new_temp()
					self._emit(IRBinOp(dest=addr, left=dest, op="+", right=offset))
					self._emit(IRStore(address=addr, value=val))
				else:
					self._emit(IRStore(address=dest, value=val))
		return dest

	def visit_comma_expr(self, node: CommaExpr) -> IRValue:
		"""Evaluate left for side effects, return right's value."""
		self.visit(node.left)
		return self.visit(node.right)

	def visit_va_start_expr(self, node: VaStartExpr) -> IRValue:
		ap_val = self.visit(node.ap)
		# Count named GP params to compute initial gp_offset
		num_named_gp = len(self._current_function_params)
		# Allocate 24-byte va_list struct and store its address in ap
		va_struct = self._new_temp()
		self._set_temp_type(va_struct, IRType.POINTER)
		self._emit(IRAlloc(dest=va_struct, size=24))
		# Store struct address into the ap variable
		self._emit(IRStore(address=ap_val, value=va_struct, ir_type=IRType.POINTER))
		# Emit IRVaStart to initialize the struct
		self._emit(IRVaStart(ap_addr=va_struct, num_named_gp=num_named_gp))
		return IRConst(0, ir_type=IRType.VOID)

	def visit_va_arg_expr(self, node: VaArgExpr) -> IRValue:
		ap_val = self.visit(node.ap)
		# Load the va_list struct address from ap
		struct_addr = self._new_temp()
		self._set_temp_type(struct_addr, IRType.POINTER)
		self._emit(IRLoad(dest=struct_addr, address=ap_val, ir_type=IRType.POINTER))
		dest = self._new_temp()
		ir_type = _resolve_ir_type(node.arg_type)
		self._set_temp_type(dest, ir_type)
		self._emit(IRVaArg(dest=dest, ap_addr=struct_addr, ir_type=ir_type))
		return dest

	def visit_va_end_expr(self, node: VaEndExpr) -> IRValue:
		ap_val = self.visit(node.ap)
		struct_addr = self._new_temp()
		self._set_temp_type(struct_addr, IRType.POINTER)
		self._emit(IRLoad(dest=struct_addr, address=ap_val, ir_type=IRType.POINTER))
		self._emit(IRVaEnd(ap_addr=struct_addr))
		return IRConst(0, ir_type=IRType.VOID)

	def visit_va_copy_expr(self, node: VaCopyExpr) -> IRValue:
		dest_val = self.visit(node.dest)
		src_val = self.visit(node.src)
		# Load src struct address
		src_struct = self._new_temp()
		self._set_temp_type(src_struct, IRType.POINTER)
		self._emit(IRLoad(dest=src_struct, address=src_val, ir_type=IRType.POINTER))
		# Allocate new struct for dest, copy src into it
		dest_struct = self._new_temp()
		self._set_temp_type(dest_struct, IRType.POINTER)
		self._emit(IRAlloc(dest=dest_struct, size=24))
		self._emit(IRStore(address=dest_val, value=dest_struct, ir_type=IRType.POINTER))
		self._emit(IRVaCopy(dest_addr=dest_struct, src_addr=src_struct))
		return IRConst(0, ir_type=IRType.VOID)

	def _resolve_struct_name(self, node: object) -> str:
		"""Try to determine the struct type name from an AST node."""
		return self._resolve_aggregate_name(node)

	def _resolve_aggregate_name(self, node: object) -> str:
		"""Try to determine the struct/union type name from an AST node."""
		if isinstance(node, Identifier):
			ts = self._local_types.get(node.name) or self._global_types.get(node.name)
			if ts is not None:
				name = ts.base_type
				if name.startswith("struct "):
					name = name[len("struct "):]
				elif name.startswith("union "):
					name = name[len("union "):]
				return name
		if isinstance(node, UnaryOp) and node.op == "*":
			# Dereference of pointer-to-struct: resolve pointee type
			inner_ts = self._infer_expr_type(node.operand)
			if inner_ts is not None and inner_ts.pointer_count >= 1:
				name = inner_ts.base_type
				if name.startswith("struct "):
					name = name[len("struct "):]
				elif name.startswith("union "):
					name = name[len("union "):]
				return name
		if isinstance(node, CastExpr):
			# Cast to struct pointer: resolve from target type
			ts = node.target_type
			if ts.pointer_count >= 1:
				name = ts.base_type
				if name.startswith("struct "):
					name = name[len("struct "):]
				elif name.startswith("union "):
					name = name[len("union "):]
				return name
		if isinstance(node, MemberAccess):
			parent_name = self._resolve_aggregate_name(node.object)
			if not parent_name:
				return ""
			members = self._structs.get(parent_name) or self._unions.get(parent_name, [])
			for m in members:
				if m.name == node.member:
					base = m.type_spec.base_type
					if base.startswith("struct "):
						base = base[len("struct "):]
					elif base.startswith("union "):
						base = base[len("union "):]
					return base
		return ""

	def _compute_field_offset(self, struct_name: str, field_name: str) -> int:
		# Check bitfield layout first
		bf_layout = self._bitfield_layouts.get(struct_name)
		if bf_layout and field_name in bf_layout:
			return bf_layout[field_name][0]  # byte_offset
		members = self._structs.get(struct_name, [])
		has_bitfields = any(m.bit_width is not None for m in members)
		if not has_bitfields:
			offset = 0
			for m in members:
				align = self._resolve_type_alignment(m.type_spec)
				offset = _align_to(offset, align)
				if m.name == field_name:
					return offset
				offset += self._resolve_full_member_size(m)
			return offset
		# Walk through with bitfield-aware offsets for non-bitfield members
		byte_offset = 0
		bit_offset = 0
		storage_size = 0
		for m in members:
			if m.bit_width is not None:
				type_size = _resolve_size(m.type_spec)
				if m.bit_width == 0:
					if bit_offset > 0:
						byte_offset += storage_size
						bit_offset = 0
						storage_size = 0
					continue
				if storage_size == 0:
					align = _resolve_alignment(m.type_spec)
					byte_offset = _align_to(byte_offset, align)
					storage_size = type_size
					bit_offset = 0
				elif bit_offset + m.bit_width > storage_size * 8:
					byte_offset += storage_size
					align = _resolve_alignment(m.type_spec)
					byte_offset = _align_to(byte_offset, align)
					storage_size = type_size
					bit_offset = 0
				if m.name == field_name:
					return byte_offset
				bit_offset += m.bit_width
			else:
				if bit_offset > 0:
					byte_offset += storage_size
					bit_offset = 0
					storage_size = 0
				align = self._resolve_type_alignment(m.type_spec)
				byte_offset = _align_to(byte_offset, align)
				if m.name == field_name:
					return byte_offset
				byte_offset += self._resolve_full_member_size(m)
		return byte_offset
