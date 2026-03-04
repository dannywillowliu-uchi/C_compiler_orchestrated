"""Semantic analysis: symbol table management and type checking for the C compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
	Identifier,
	IfStmt,
	InitializerList,
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
	TypedefDecl,
	TypeSpec,
	UnaryOp,
	UnionDecl,
	VarDecl,
	WhileStmt,
)

if TYPE_CHECKING:
	from compiler.ast_nodes import ASTNode


class SemanticError(Exception):
	"""Raised when semantic analysis detects a violation."""

	def __init__(self, message: str, line: int = 0, col: int = 0) -> None:
		self.line = line
		self.col = col
		super().__init__(f"{line}:{col}: {message}")


@dataclass
class Symbol:
	"""A symbol table entry."""

	name: str
	type_spec: TypeSpec
	scope_depth: int
	is_function: bool = False
	param_types: list[TypeSpec] = field(default_factory=list)
	is_array: bool = False
	array_sizes: list[int] = field(default_factory=list)
	is_prototype: bool = False
	storage_class: str | None = None
	is_function_pointer: bool = False


class SymbolTable:
	"""Nested lexical scope symbol table."""

	def __init__(self) -> None:
		self._scopes: list[dict[str, Symbol]] = [{}]
		self._depth: int = 0

	@property
	def depth(self) -> int:
		return self._depth

	def push_scope(self) -> None:
		self._depth += 1
		self._scopes.append({})

	def pop_scope(self) -> None:
		if self._depth == 0:
			raise RuntimeError("Cannot pop global scope")
		self._scopes.pop()
		self._depth -= 1

	def define(self, symbol: Symbol) -> None:
		current = self._scopes[-1]
		if symbol.name in current:
			raise SemanticError(
				f"redefinition of '{symbol.name}' in the same scope",
				line=0,
				col=0,
			)
		current[symbol.name] = symbol

	def lookup(self, name: str) -> Symbol | None:
		for scope in reversed(self._scopes):
			if name in scope:
				return scope[name]
		return None

	def lookup_current_scope(self, name: str) -> Symbol | None:
		return self._scopes[-1].get(name)


# Type compatibility helpers

_NUMERIC_TYPES = {"int", "char", "short", "long", "float", "double"}
_ARITHMETIC_OPS = {"+", "-", "*", "/", "%"}
_COMPARISON_OPS = {"<", ">", "<=", ">=", "==", "!="}
_LOGICAL_OPS = {"&&", "||"}
_BITWISE_OPS = {"&", "|", "^", "<<", ">>"}


def _type_size(ts: TypeSpec) -> int:
	"""Compute size in bytes for a type considering width modifiers."""
	if ts.pointer_count > 0:
		return 8
	wm = ts.width_modifier
	if wm == "short":
		return 2
	if wm == "long":
		return 8
	if wm == "long long":
		return 8
	base_sizes = {"int": 4, "char": 1, "float": 4, "double": 8, "short": 2, "long": 8, "void": 0}
	return base_sizes.get(ts.base_type, 4)


def _is_numeric(ts: TypeSpec) -> bool:
	return ts.base_type in _NUMERIC_TYPES and ts.pointer_count == 0


def _is_pointer(ts: TypeSpec) -> bool:
	return ts.pointer_count > 0


def _is_array_or_pointer(ts: TypeSpec) -> bool:
	return ts.pointer_count > 0


def _types_compatible(left: TypeSpec, right: TypeSpec) -> bool:
	"""Check whether two types are assignment-compatible."""
	if left == right:
		return True
	# Numeric types are mutually compatible (implicit conversion in C)
	if _is_numeric(left) and _is_numeric(right):
		return True
	# Pointer to pointer of same base (ignoring void*)
	if _is_pointer(left) and _is_pointer(right):
		if left.base_type == "void" or right.base_type == "void":
			return True
		return left.base_type == right.base_type and left.pointer_count == right.pointer_count
	return False


def _result_type(left: TypeSpec, op: str, right: TypeSpec) -> TypeSpec:
	"""Determine the result type of a binary operation."""
	if op in _COMPARISON_OPS or op in _LOGICAL_OPS:
		return TypeSpec(base_type="int")
	# Pointer arithmetic: ptr + int or int + ptr
	if op in ("+", "-"):
		if _is_pointer(left) and _is_numeric(right):
			return left
		if _is_numeric(left) and _is_pointer(right) and op == "+":
			return right
	return left


class SemanticAnalyzer(ASTVisitor):
	"""Walks the AST performing symbol resolution and basic type checking."""

	def __init__(self) -> None:
		self.symbols = SymbolTable()
		self.errors: list[SemanticError] = []
		self._current_function: FunctionDecl | None = None
		self._loop_depth: int = 0
		self._switch_depth: int = 0
		self._struct_types: dict[str, StructDecl] = {}
		self._union_types: dict[str, UnionDecl] = {}
		self._typedef_types: dict[str, TypeSpec] = {}

	def analyze(self, node: ASTNode) -> list[SemanticError]:
		self.visit(node)
		if self.errors:
			raise self.errors[0]
		return self.errors

	def _error(self, msg: str, node: ASTNode) -> None:
		err = SemanticError(msg, line=node.loc.line, col=node.loc.col)
		self.errors.append(err)

	def _define_symbol(
		self,
		name: str,
		type_spec: TypeSpec,
		node: ASTNode,
		is_function: bool = False,
		param_types: list[TypeSpec] | None = None,
	) -> None:
		sym = Symbol(
			name=name,
			type_spec=type_spec,
			scope_depth=self.symbols.depth,
			is_function=is_function,
			param_types=param_types or [],
		)
		existing = self.symbols.lookup_current_scope(name)
		if existing is not None:
			self._error(f"redefinition of '{name}' in the same scope", node)
			return
		self.symbols.define(sym)

	def _resolve_type(self, ts: TypeSpec) -> TypeSpec:
		"""Resolve a type spec, following typedef chains to the underlying type."""
		if ts.is_function_pointer:
			return ts
		if ts.base_type in self._typedef_types:
			resolved = self._typedef_types[ts.base_type]
			if resolved.is_function_pointer:
				return resolved
			return TypeSpec(
				base_type=resolved.base_type,
				pointer_count=resolved.pointer_count + ts.pointer_count,
			)
		return ts

	def _check_duplicate_qualifiers(self, ts: TypeSpec, node: ASTNode) -> None:
		"""Warn on duplicate qualifiers like 'const const'."""
		seen: set[str] = set()
		for q in ts.qualifiers:
			if q in seen:
				self._error(f"duplicate '{q}' qualifier", node)
			seen.add(q)

	# --- Visitor methods ---

	def visit_program(self, node: Program) -> None:
		for decl in node.declarations:
			self.visit(decl)

	def visit_function_decl(self, node: FunctionDecl) -> TypeSpec:
		node.return_type = self._resolve_type(node.return_type)
		self._check_duplicate_qualifiers(node.return_type, node)
		for p in node.params:
			p.type_spec = self._resolve_type(p.type_spec)
		param_types = [p.type_spec for p in node.params]
		existing = self.symbols.lookup_current_scope(node.name)
		if existing is not None and existing.is_function and existing.is_prototype and node.body is not None:
			existing.is_prototype = False
		elif node.body is None:
			sym = Symbol(
				name=node.name,
				type_spec=node.return_type,
				scope_depth=self.symbols.depth,
				is_function=True,
				param_types=param_types,
				is_prototype=True,
			)
			if existing is not None:
				self._error(f"redefinition of '{node.name}' in the same scope", node)
				return node.return_type
			self.symbols.define(sym)
			return node.return_type
		else:
			self._define_symbol(
				node.name, node.return_type, node, is_function=True, param_types=param_types
			)
		if node.body is None:
			return node.return_type
		prev_func = self._current_function
		self._current_function = node
		self.symbols.push_scope()
		for param in node.params:
			self.visit(param)
		for stmt in node.body.statements:
			self.visit(stmt)
		self.symbols.pop_scope()
		self._current_function = prev_func
		return node.return_type

	def visit_param_decl(self, node: ParamDecl) -> TypeSpec:
		node.type_spec = self._resolve_type(node.type_spec)
		if node.type_spec.is_function_pointer:
			sym = Symbol(
				name=node.name,
				type_spec=node.type_spec,
				scope_depth=self.symbols.depth,
				is_function_pointer=True,
			)
			existing = self.symbols.lookup_current_scope(node.name)
			if existing is not None:
				self._error(f"redefinition of '{node.name}' in the same scope", node)
			else:
				self.symbols.define(sym)
		else:
			self._define_symbol(node.name, node.type_spec, node)
		return node.type_spec

	def visit_var_decl(self, node: VarDecl) -> TypeSpec:
		node.type_spec = self._resolve_type(node.type_spec)
		self._check_duplicate_qualifiers(node.type_spec, node)

		is_fp = node.type_spec.is_function_pointer

		if node.initializer is not None:
			if isinstance(node.initializer, InitializerList):
				self._check_initializer_list(node)
			else:
				if is_fp:
					self._check_func_ptr_initializer(node)
				else:
					init_type = self.visit(node.initializer)
					if init_type is not None and not _types_compatible(node.type_spec, init_type):
						self._error(
							f"incompatible types in initialization of '{node.name}': "
							f"'{node.type_spec.base_type}' and '{init_type.base_type}'",
							node,
						)
		is_array = node.array_sizes is not None and len(node.array_sizes) > 0
		array_size_vals: list[int] = []
		if is_array:
			for size_expr in node.array_sizes:  # type: ignore[union-attr]
				if isinstance(size_expr, IntLiteral) and size_expr.value == 0:
					# Infer size from initializer list
					if isinstance(node.initializer, InitializerList):
						inferred = len(node.initializer.elements)
						size_expr.value = inferred
						array_size_vals.append(inferred)
					else:
						self._error("array size must be specified or inferred from initializer", node)
				else:
					size_type = self.visit(size_expr)
					if size_type is not None and not _is_numeric(size_type):
						self._error("array size must be an integer expression", node)
					if isinstance(size_expr, IntLiteral):
						array_size_vals.append(size_expr.value)
		sym = Symbol(
			name=node.name,
			type_spec=node.type_spec,
			scope_depth=self.symbols.depth,
			is_array=is_array,
			array_sizes=array_size_vals,
			storage_class=node.storage_class,
			is_function_pointer=is_fp,
		)
		existing = self.symbols.lookup_current_scope(node.name)
		if existing is not None:
			self._error(f"redefinition of '{node.name}' in the same scope", node)
			return node.type_spec
		self.symbols.define(sym)
		return node.type_spec

	def _check_func_ptr_initializer(self, node: VarDecl) -> None:
		"""Validate function pointer initializer (fp = func_name or fp = &func_name)."""
		init = node.initializer
		assert init is not None
		# Allow bare function name: fp = add
		if isinstance(init, Identifier):
			sym = self.symbols.lookup(init.name)
			if sym is None:
				self._error(f"use of undeclared identifier '{init.name}'", init)
			elif not sym.is_function and not sym.is_function_pointer:
				self._error(f"'{init.name}' is not a function or function pointer", init)
			return
		# Allow address-of function: fp = &add
		if isinstance(init, UnaryOp) and init.op == "&":
			if isinstance(init.operand, Identifier):
				sym = self.symbols.lookup(init.operand.name)
				if sym is None:
					self._error(f"use of undeclared identifier '{init.operand.name}'", init)
				elif not sym.is_function:
					self._error(f"'{init.operand.name}' is not a function", init)
				return
		# General expression -- just visit it
		self.visit(init)

	def _check_initializer_list(self, node: VarDecl) -> None:
		"""Type-check an initializer list against the variable's type."""
		init_list = node.initializer
		assert isinstance(init_list, InitializerList)
		is_array = node.array_sizes is not None and len(node.array_sizes) > 0
		base_type = node.type_spec.base_type

		if is_array:
			# Check element count vs declared array size
			for size_expr in node.array_sizes:  # type: ignore[union-attr]
				if isinstance(size_expr, IntLiteral) and size_expr.value > 0:
					if len(init_list.elements) > size_expr.value:
						self._error("excess elements in array initializer", node)
			# Type-check each element
			for elem in init_list.elements:
				if isinstance(elem, InitializerList):
					self.visit_initializer_list(elem)
				else:
					elem_t = self.visit(elem)
					if elem_t is not None and not _types_compatible(node.type_spec, elem_t):
						self._error("incompatible type in array initializer element", elem)
		elif base_type.startswith("struct "):
			struct_name = base_type[len("struct "):]
			if struct_name in self._struct_types:
				struct_decl = self._struct_types[struct_name]
				if len(init_list.elements) > len(struct_decl.members):
					self._error("excess elements in struct initializer", node)
				for i, elem in enumerate(init_list.elements):
					if isinstance(elem, InitializerList):
						self.visit_initializer_list(elem)
					else:
						elem_t = self.visit(elem)
						if i < len(struct_decl.members) and elem_t is not None:
							member_type = struct_decl.members[i].type_spec
							if not _types_compatible(member_type, elem_t):
								self._error(
									f"incompatible type for member '{struct_decl.members[i].name}'",
									elem,
								)
			else:
				for elem in init_list.elements:
					self.visit(elem)
		else:
			for elem in init_list.elements:
				self.visit(elem)

	def visit_initializer_list(self, node: InitializerList) -> TypeSpec | None:
		for elem in node.elements:
			self.visit(elem)
		return None

	def visit_compound_stmt(self, node: CompoundStmt) -> None:
		self.symbols.push_scope()
		for stmt in node.statements:
			self.visit(stmt)
		self.symbols.pop_scope()

	def visit_return_stmt(self, node: ReturnStmt) -> None:
		ret_type = self.visit(node.expression)
		if self._current_function is not None and ret_type is not None:
			expected = self._current_function.return_type
			if not _types_compatible(expected, ret_type):
				self._error(
					f"incompatible return type: expected '{expected.base_type}', "
					f"got '{ret_type.base_type}'",
					node,
				)

	def visit_if_stmt(self, node: IfStmt) -> None:
		self.visit(node.condition)
		self.visit(node.then_branch)
		if node.else_branch is not None:
			self.visit(node.else_branch)

	def visit_while_stmt(self, node: WhileStmt) -> None:
		self.visit(node.condition)
		self._loop_depth += 1
		self.visit(node.body)
		self._loop_depth -= 1

	def visit_for_stmt(self, node: ForStmt) -> None:
		self.symbols.push_scope()
		if node.init is not None:
			self.visit(node.init)
		if node.condition is not None:
			self.visit(node.condition)
		if node.update is not None:
			self.visit(node.update)
		self._loop_depth += 1
		self.visit(node.body)
		self._loop_depth -= 1
		self.symbols.pop_scope()

	def visit_do_while_stmt(self, node: DoWhileStmt) -> None:
		self._loop_depth += 1
		self.visit(node.body)
		self._loop_depth -= 1
		self.visit(node.condition)

	def visit_break_stmt(self, node: BreakStmt) -> None:
		if self._loop_depth == 0 and self._switch_depth == 0:
			self._error("break statement not within a loop or switch", node)

	def visit_continue_stmt(self, node: ContinueStmt) -> None:
		if self._loop_depth == 0:
			self._error("continue statement not within a loop", node)

	def visit_compound_assignment(self, node: CompoundAssignment) -> TypeSpec | None:
		target_type = self.visit(node.target)
		value_type = self.visit(node.value)
		if target_type is None or value_type is None:
			return None
		# Arithmetic compound ops require numeric types
		if node.op in _ARITHMETIC_OPS:
			if not _is_numeric(target_type) and not _is_pointer(target_type):
				self._error(
					f"incompatible type for compound assignment operator '{node.op}='",
					node,
				)
				return None
			if not _is_numeric(value_type):
				self._error(
					f"incompatible type for compound assignment operator '{node.op}='",
					node,
				)
				return None
			# Modulo requires both numeric (no pointers)
			if node.op == "%" and (not _is_numeric(target_type) or not _is_numeric(value_type)):
				self._error(
					f"incompatible types for operator '{node.op}=': "
					f"'{target_type.base_type}' and '{value_type.base_type}'",
					node,
				)
				return None
		elif not _types_compatible(target_type, value_type):
			self._error(
				f"incompatible types in compound assignment: "
				f"'{target_type.base_type}' and '{value_type.base_type}'",
				node,
			)
		return target_type

	def visit_expr_stmt(self, node: ExprStmt) -> None:
		self.visit(node.expression)

	def visit_binary_op(self, node: BinaryOp) -> TypeSpec | None:
		left_type = self.visit(node.left)
		right_type = self.visit(node.right)
		if left_type is None or right_type is None:
			return None
		if node.op in _ARITHMETIC_OPS or node.op in _BITWISE_OPS:
			# Allow pointer arithmetic
			if not (_is_numeric(left_type) or _is_pointer(left_type)) or not (
				_is_numeric(right_type) or _is_pointer(right_type)
			):
				self._error(
					f"incompatible types for operator '{node.op}': "
					f"'{left_type.base_type}' and '{right_type.base_type}'",
					node,
				)
				return None
			if node.op in _BITWISE_OPS or node.op == "%":
				if not _is_numeric(left_type) or not _is_numeric(right_type):
					self._error(
						f"incompatible types for operator '{node.op}': "
						f"'{left_type.base_type}' and '{right_type.base_type}'",
						node,
					)
					return None
		elif node.op in _COMPARISON_OPS or node.op in _LOGICAL_OPS:
			pass  # comparisons/logical ops work on any scalar
		return _result_type(left_type, node.op, right_type)

	def visit_unary_op(self, node: UnaryOp) -> TypeSpec | None:
		# Special case: &function_name -> function pointer (address of function)
		if node.op == "&" and isinstance(node.operand, Identifier):
			sym = self.symbols.lookup(node.operand.name)
			if sym is not None and sym.is_function:
				return TypeSpec(base_type="void", pointer_count=1)
		operand_type = self.visit(node.operand)
		if operand_type is None:
			return None
		if node.op == "&":
			return TypeSpec(
				base_type=operand_type.base_type,
				pointer_count=operand_type.pointer_count + 1,
			)
		if node.op == "*":
			if operand_type.pointer_count < 1:
				self._error("dereference of non-pointer type", node)
				return None
			return TypeSpec(
				base_type=operand_type.base_type,
				pointer_count=operand_type.pointer_count - 1,
			)
		return operand_type

	def visit_int_literal(self, node: IntLiteral) -> TypeSpec:
		return TypeSpec(base_type="int")

	def visit_float_literal(self, node: FloatLiteral) -> TypeSpec:
		if node.suffix == "f":
			return TypeSpec(base_type="float")
		return TypeSpec(base_type="double")

	def visit_string_literal(self, node: StringLiteral) -> TypeSpec:
		return TypeSpec(base_type="char", pointer_count=1)

	def visit_char_literal(self, node: CharLiteral) -> TypeSpec:
		return TypeSpec(base_type="char")

	def visit_identifier(self, node: Identifier) -> TypeSpec | None:
		sym = self.symbols.lookup(node.name)
		if sym is None:
			self._error(f"use of undeclared identifier '{node.name}'", node)
			return None
		# Function pointer variables return a pointer type for type-compat purposes
		if sym.is_function_pointer:
			return TypeSpec(base_type="void", pointer_count=1)
		return sym.type_spec

	def visit_assignment(self, node: Assignment) -> TypeSpec | None:
		target_type = self.visit(node.target)
		# Check if target is a function pointer
		if isinstance(node.target, Identifier):
			sym = self.symbols.lookup(node.target.name)
			if sym is not None and sym.is_function_pointer:
				# Validate the RHS is a function or function pointer
				if isinstance(node.value, Identifier):
					rhs_sym = self.symbols.lookup(node.value.name)
					if rhs_sym is None:
						self._error(f"use of undeclared identifier '{node.value.name}'", node.value)
					elif not rhs_sym.is_function and not rhs_sym.is_function_pointer:
						self._error(f"'{node.value.name}' is not a function or function pointer", node.value)
					return target_type
				if isinstance(node.value, UnaryOp) and node.value.op == "&":
					if isinstance(node.value.operand, Identifier):
						rhs_sym = self.symbols.lookup(node.value.operand.name)
						if rhs_sym is None:
							self._error(f"use of undeclared identifier '{node.value.operand.name}'", node.value)
						elif not rhs_sym.is_function:
							self._error(f"'{node.value.operand.name}' is not a function", node.value)
						return target_type
				# General expression
				self.visit(node.value)
				return target_type
		value_type = self.visit(node.value)
		if target_type is None or value_type is None:
			return None
		if not _types_compatible(target_type, value_type):
			self._error(
				f"incompatible types in assignment: "
				f"'{target_type.base_type}' and '{value_type.base_type}'",
				node,
			)
		return target_type

	def visit_function_call(self, node: FunctionCall) -> TypeSpec | None:
		sym = self.symbols.lookup(node.name)
		if sym is None:
			self._error(f"call to undeclared function '{node.name}'", node)
			return None
		# Handle indirect calls through function pointers
		if sym.is_function_pointer or (sym.type_spec is not None and sym.type_spec.is_function_pointer):
			fp_type = sym.type_spec
			expected_params = fp_type.func_ptr_params
			if len(node.arguments) != len(expected_params):
				self._error(
					f"function pointer '{node.name}' expects {len(expected_params)} arguments, "
					f"got {len(node.arguments)}",
					node,
				)
			for arg in node.arguments:
				self.visit(arg)
			return fp_type.func_ptr_return_type
		if not sym.is_function:
			self._error(f"'{node.name}' is not a function", node)
			return None
		if len(node.arguments) != len(sym.param_types):
			self._error(
				f"function '{node.name}' expects {len(sym.param_types)} arguments, "
				f"got {len(node.arguments)}",
				node,
			)
		for arg in node.arguments:
			self.visit(arg)
		return sym.type_spec

	def visit_array_subscript(self, node: ArraySubscript) -> TypeSpec | None:
		base_type = self.visit(node.array)
		index_type = self.visit(node.index)
		if base_type is None:
			return None
		# Check base is array or pointer
		is_base_array = False
		if isinstance(node.array, Identifier):
			sym = self.symbols.lookup(node.array.name)
			if sym is not None:
				is_base_array = sym.is_array
		if not is_base_array and not _is_pointer(base_type):
			self._error("subscript requires array or pointer type", node)
			return None
		# Check index is integer
		if index_type is not None and not _is_numeric(index_type):
			self._error("array index must be an integer", node)
			return None
		# Result type: dereference one pointer level, or base element type for arrays
		if _is_pointer(base_type):
			return TypeSpec(
				base_type=base_type.base_type,
				pointer_count=base_type.pointer_count - 1,
			)
		return TypeSpec(base_type=base_type.base_type)

	def visit_switch_stmt(self, node: SwitchStmt) -> None:
		self.visit(node.expression)
		self._switch_depth += 1
		seen_values: set[int | str] = set()
		has_default = False
		for case in node.cases:
			if case.value is None:
				if has_default:
					self._error("duplicate default label in switch", case)
				has_default = True
			else:
				if not isinstance(case.value, (IntLiteral, CharLiteral)):
					self._error("case expression must be a constant integer", case)
				else:
					val: int | str = case.value.value
					if val in seen_values:
						self._error(f"duplicate case value {val!r}", case)
					seen_values.add(val)
			for stmt in case.statements:
				self.visit(stmt)
		self._switch_depth -= 1

	def visit_case_clause(self, node: CaseClause) -> None:
		if node.value is not None:
			self.visit(node.value)
		for stmt in node.statements:
			self.visit(stmt)

	def visit_ternary_expr(self, node: TernaryExpr) -> TypeSpec | None:
		cond_type = self.visit(node.condition)
		true_type = self.visit(node.true_expr)
		false_type = self.visit(node.false_expr)
		if cond_type is not None and not (_is_numeric(cond_type) or _is_pointer(cond_type)):
			self._error("ternary condition must be a scalar type", node)
		if true_type is not None and false_type is not None:
			if not _types_compatible(true_type, false_type):
				self._error(
					f"incompatible types in ternary branches: "
					f"'{true_type.base_type}' and '{false_type.base_type}'",
					node,
				)
		return true_type

	def visit_sizeof_expr(self, node: SizeofExpr) -> TypeSpec:
		if node.operand is not None:
			self.visit(node.operand)
		return TypeSpec(base_type="int")

	def visit_postfix_expr(self, node: PostfixExpr) -> TypeSpec | None:
		operand_type = self.visit(node.operand)
		if not self._is_lvalue(node.operand):
			self._error("operand of postfix operator must be an lvalue", node)
		return operand_type

	def visit_cast_expr(self, node: CastExpr) -> TypeSpec:
		operand_type = self.visit(node.operand)
		node.target_type = self._resolve_type(node.target_type)
		target = node.target_type
		if operand_type is not None:
			# Allow: numeric-to-numeric, pointer-to-pointer, numeric-to-pointer, pointer-to-numeric
			src_numeric = _is_numeric(operand_type)
			src_pointer = _is_pointer(operand_type)
			tgt_numeric = _is_numeric(target)
			tgt_pointer = _is_pointer(target)
			if not ((src_numeric and tgt_numeric)
				or (src_pointer and tgt_pointer)
				or (src_numeric and tgt_pointer)
				or (src_pointer and tgt_numeric)):
				self._error(
					f"invalid cast from '{operand_type.base_type}' to '{target.base_type}'",
					node,
				)
		return target

	def _is_lvalue(self, node: ASTNode) -> bool:
		if isinstance(node, Identifier):
			return True
		if isinstance(node, ArraySubscript):
			return True
		if isinstance(node, MemberAccess):
			return True
		if isinstance(node, UnaryOp) and node.op == "*":
			return True
		return False

	def visit_enum_decl(self, node: EnumDecl) -> None:
		next_value = 0
		for const in node.constants:
			if const.value is not None:
				if isinstance(const.value, IntLiteral):
					next_value = const.value.value
				else:
					self.visit(const.value)
			self._define_symbol(const.name, TypeSpec(base_type="int"), const)
			next_value += 1

	def visit_struct_decl(self, node: StructDecl) -> None:
		if node.name in self._struct_types:
			self._error(f"redefinition of struct '{node.name}'", node)
			return
		seen: set[str] = set()
		for member in node.members:
			if member.name in seen:
				self._error(f"duplicate member '{member.name}' in struct '{node.name}'", member)
			else:
				seen.add(member.name)
		self._struct_types[node.name] = node

	def visit_union_decl(self, node: UnionDecl) -> None:
		if node.name in self._union_types:
			self._error(f"redefinition of union '{node.name}'", node)
			return
		seen: set[str] = set()
		for member in node.members:
			if member.name in seen:
				self._error(f"duplicate member '{member.name}' in union '{node.name}'", member)
			else:
				seen.add(member.name)
		self._union_types[node.name] = node

	def visit_typedef_decl(self, node: TypedefDecl) -> None:
		if node.struct_decl is not None:
			self.visit_struct_decl(node.struct_decl)
		if node.enum_decl is not None:
			self.visit_enum_decl(node.enum_decl)
		if node.union_decl is not None:
			self.visit_union_decl(node.union_decl)
		resolved = self._resolve_type(node.type_spec)
		if node.name in self._typedef_types:
			self._error(f"redefinition of typedef '{node.name}'", node)
			return
		self._typedef_types[node.name] = resolved

	def visit_struct_member(self, node: StructMember) -> TypeSpec:
		return node.type_spec

	def visit_member_access(self, node: MemberAccess) -> TypeSpec | None:
		obj_type = self.visit(node.object)
		if obj_type is None:
			return None
		if node.is_arrow:
			if obj_type.pointer_count < 1:
				self._error("member access with '->' requires pointer to struct/union", node)
				return None
			base_name = obj_type.base_type
		else:
			if obj_type.pointer_count > 0:
				self._error("member access with '.' requires non-pointer struct/union (use '->' instead)", node)
				return None
			base_name = obj_type.base_type
		# Strip "struct " or "union " prefix if present
		is_union = False
		if base_name.startswith("struct "):
			base_name = base_name[len("struct "):]
		elif base_name.startswith("union "):
			base_name = base_name[len("union "):]
			is_union = True
		# Look up in union types first, then struct types
		if is_union or base_name in self._union_types:
			if base_name not in self._union_types:
				self._error(f"member access on non-union type '{obj_type.base_type}'", node)
				return None
			union_decl = self._union_types[base_name]
			for member in union_decl.members:
				if member.name == node.member:
					return member.type_spec
			self._error(
				f"no member named '{node.member}' in union '{base_name}'",
				node,
			)
			return None
		if base_name not in self._struct_types:
			self._error(f"member access on non-struct type '{obj_type.base_type}'", node)
			return None
		struct_decl = self._struct_types[base_name]
		for member in struct_decl.members:
			if member.name == node.member:
				return member.type_spec
		self._error(
			f"no member named '{node.member}' in struct '{base_name}'",
			node,
		)
		return None

	def visit_type_spec(self, node: TypeSpec) -> TypeSpec:
		return self._resolve_type(node)
