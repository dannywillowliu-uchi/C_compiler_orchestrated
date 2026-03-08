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
	Identifier,
	IfStmt,
	InitializerList,
	IntLiteral,
	LabelStmt,
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
	TypedefDecl,
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
from compiler.const_eval import ConstExprEvaluator

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
	is_variadic: bool = False


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

_NUMERIC_TYPES = {
	"int", "char", "short", "long", "float", "double", "_Bool", "bool",
	"signed char", "unsigned char",
	"unsigned short", "signed short",
	"unsigned int", "signed int",
	"unsigned long", "signed long",
	"long long", "unsigned long long", "signed long long",
	"long double",
	# Standard typedef'd numeric types
	"size_t", "ptrdiff_t", "wchar_t",
	"int8_t", "int16_t", "int32_t", "int64_t",
	"uint8_t", "uint16_t", "uint32_t", "uint64_t",
	"intptr_t", "uintptr_t", "intmax_t", "uintmax_t",
	"__float80",
}
_ARITHMETIC_OPS = {"+", "-", "*", "/", "%"}
_COMPARISON_OPS = {"<", ">", "<=", ">=", "==", "!="}
_LOGICAL_OPS = {"&&", "||"}
_BITWISE_OPS = {"&", "|", "^", "<<", ">>"}


_TYPEDEF_RANK: dict[str, tuple[int, bool]] = {
	"int8_t": (1, False), "uint8_t": (1, True),
	"int16_t": (2, False), "uint16_t": (2, True),
	"int32_t": (3, False), "uint32_t": (3, True),
	"int64_t": (5, False), "uint64_t": (5, True),
	"intptr_t": (4, False), "uintptr_t": (4, True),
	"intmax_t": (5, False), "uintmax_t": (5, True),
	"size_t": (4, True), "ptrdiff_t": (4, False),
	"wchar_t": (3, False),
}

def _type_rank(ts: TypeSpec) -> int:
	"""Return the conversion rank of a type (higher = wider).

	Rank values: char=1, short=2, int=3, long=4, long_long=5, float=6, double=7.
	"""
	if ts.base_type == "double":
		return 7
	if ts.base_type == "float":
		return 6
	if ts.base_type in _TYPEDEF_RANK:
		return _TYPEDEF_RANK[ts.base_type][0]
	if ts.width_modifier == "long long":
		return 5
	if ts.base_type == "long" or ts.width_modifier == "long":
		return 4
	if ts.base_type == "short" or ts.width_modifier == "short":
		return 2
	if ts.base_type == "char":
		return 1
	if ts.base_type in ("_Bool", "bool"):
		return 0
	# int or anything else
	return 3


def _is_unsigned(ts: TypeSpec) -> bool:
	if ts.base_type in _TYPEDEF_RANK:
		return _TYPEDEF_RANK[ts.base_type][1]
	return ts.signedness == "unsigned"


def _typespec_from_rank(rank: int, unsigned: bool) -> TypeSpec:
	"""Build a TypeSpec for a given integer/float rank."""
	if rank == 7:
		return TypeSpec(base_type="double")
	if rank == 6:
		return TypeSpec(base_type="float")
	signedness = "unsigned" if unsigned else None
	if rank == 5:
		return TypeSpec(base_type="int", width_modifier="long long", signedness=signedness)
	if rank == 4:
		return TypeSpec(base_type="int", width_modifier="long", signedness=signedness)
	# rank <= 3 all promote to int
	return TypeSpec(base_type="int", signedness=signedness)


def _integer_promote(ts: TypeSpec) -> TypeSpec:
	"""Apply C integer promotion rules: char and short promote to int."""
	if ts.pointer_count > 0:
		return ts
	rank = _type_rank(ts)
	if rank < 3:  # char or short -> int
		return TypeSpec(base_type="int")
	return ts


def _usual_arithmetic_conversions(left: TypeSpec, right: TypeSpec) -> TypeSpec:
	"""Apply C usual arithmetic conversions to find the common type."""
	l_rank = _type_rank(left)
	r_rank = _type_rank(right)

	# If either is double, result is double
	if l_rank == 7 or r_rank == 7:
		return TypeSpec(base_type="double")
	# If either is float, result is float
	if l_rank == 6 or r_rank == 6:
		return TypeSpec(base_type="float")

	# Apply integer promotions first
	left = _integer_promote(left)
	right = _integer_promote(right)
	l_rank = _type_rank(left)
	r_rank = _type_rank(right)
	l_unsigned = _is_unsigned(left)
	r_unsigned = _is_unsigned(right)

	# Same rank and signedness
	if l_rank == r_rank and l_unsigned == r_unsigned:
		return left

	# Both same signedness: convert to higher rank
	if l_unsigned == r_unsigned:
		return _typespec_from_rank(max(l_rank, r_rank), l_unsigned)

	# One is unsigned, one is signed
	if l_unsigned:
		u_rank, s_rank = l_rank, r_rank
	else:
		u_rank, s_rank = r_rank, l_rank

	# Unsigned rank >= signed rank -> unsigned type wins
	if u_rank >= s_rank:
		return _typespec_from_rank(u_rank, True)
	# Signed rank can represent all unsigned values -> signed wins
	return _typespec_from_rank(s_rank, False)


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
	base_sizes = {"int": 4, "char": 1, "float": 4, "double": 8, "short": 2, "long": 8, "void": 0, "_Bool": 1}
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
	# Same struct/union types are compatible (compare by name, not object identity)
	if (left.base_type.startswith("struct ") or left.base_type.startswith("union ")):
		if left.base_type == right.base_type and left.pointer_count == right.pointer_count:
			return True
	# Numeric types are mutually compatible (implicit conversion in C)
	if _is_numeric(left) and _is_numeric(right):
		return True
	# Pointer to pointer of same base (ignoring void*)
	if _is_pointer(left) and _is_pointer(right):
		if left.base_type == "void" or right.base_type == "void":
			return True
		return left.base_type == right.base_type and left.pointer_count == right.pointer_count
	# Allow integer 0 (NULL) to be assigned to pointer
	if _is_pointer(left) and _is_numeric(right):
		return True
	return False


def _result_type(left: TypeSpec, op: str, right: TypeSpec) -> TypeSpec:
	"""Determine the result type of a binary operation."""
	if op in _COMPARISON_OPS or op in _LOGICAL_OPS:
		return TypeSpec(base_type="int")
	# Pointer arithmetic: ptr + int, int + ptr, ptr - int, ptr - ptr
	if op in ("+", "-"):
		if _is_pointer(left) and _is_numeric(right):
			return left
		if _is_numeric(left) and _is_pointer(right) and op == "+":
			return right
		if _is_pointer(left) and _is_pointer(right) and op == "-":
			return TypeSpec(base_type="int")
	# Shift operators: result type is the promoted left operand
	if op in ("<<", ">>"):
		return _integer_promote(left)
	# Arithmetic and bitwise: apply usual arithmetic conversions
	if op in _ARITHMETIC_OPS or op in _BITWISE_OPS:
		return _usual_arithmetic_conversions(left, right)
	return left


class SemanticAnalyzer(ASTVisitor):
	"""Walks the AST performing symbol resolution and basic type checking."""

	def __init__(self) -> None:
		self.symbols = SymbolTable()
		self.errors: list[SemanticError] = []
		self.warnings: list[str] = []
		self._current_function: FunctionDecl | None = None
		self._loop_depth: int = 0
		self._switch_depth: int = 0
		self._struct_types: dict[str, StructDecl] = {}
		self._union_types: dict[str, UnionDecl] = {}
		self._predefined_typedef_names: set[str] = {
			"int8_t", "uint8_t", "int16_t", "uint16_t",
			"int32_t", "uint32_t", "int64_t", "uint64_t",
			"intptr_t", "uintptr_t",
			"size_t", "ptrdiff_t", "wchar_t",
			"intmax_t", "uintmax_t",
			"nullptr_t",
		}
		self._typedef_types: dict[str, TypeSpec] = {
			"int8_t": TypeSpec(base_type="signed char"),
			"uint8_t": TypeSpec(base_type="unsigned char"),
			"int16_t": TypeSpec(base_type="short"),
			"uint16_t": TypeSpec(base_type="unsigned short"),
			"int32_t": TypeSpec(base_type="int"),
			"uint32_t": TypeSpec(base_type="unsigned int"),
			"int64_t": TypeSpec(base_type="long long"),
			"uint64_t": TypeSpec(base_type="unsigned long long"),
			"intptr_t": TypeSpec(base_type="long"),
			"uintptr_t": TypeSpec(base_type="unsigned long"),
			"size_t": TypeSpec(base_type="unsigned long"),
			"ptrdiff_t": TypeSpec(base_type="long"),
			"wchar_t": TypeSpec(base_type="int"),
			"intmax_t": TypeSpec(base_type="long long"),
			"uintmax_t": TypeSpec(base_type="unsigned long long"),
			"nullptr_t": TypeSpec(base_type="void", pointer_count=1),
		}
		self._in_sizeof_or_addressof: bool = False
		self._label_defs: dict[str, ASTNode] = {}
		self._goto_refs: list[tuple[str, ASTNode]] = []
		self._enum_constants: dict[str, int] = {}

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
		is_variadic: bool = False,
	) -> None:
		sym = Symbol(
			name=name,
			type_spec=type_spec,
			scope_depth=self.symbols.depth,
			is_function=is_function,
			param_types=param_types or [],
			is_variadic=is_variadic,
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
				signedness=resolved.signedness,
				width_modifier=resolved.width_modifier,
				qualifiers=list(set(resolved.qualifiers + ts.qualifiers)),
			)
		return ts

	def _check_duplicate_qualifiers(self, ts: TypeSpec, node: ASTNode) -> None:
		"""Warn on duplicate qualifiers like 'const const'."""
		seen: set[str] = set()
		for q in ts.qualifiers:
			if q in seen:
				self._error(f"duplicate '{q}' qualifier", node)
			seen.add(q)

	def _check_type_modifiers(self, ts: TypeSpec, node: ASTNode) -> None:
		"""Reject illegal type modifier combinations."""
		base = ts.base_type
		sign = ts.signedness
		width = ts.width_modifier

		# signed/unsigned cannot apply to float or double
		if sign in ("signed", "unsigned") and base in ("float", "double"):
			self._error(f"'{sign}' cannot be used with '{base}'", node)

		# short cannot combine with long
		if width == "short":
			if base in ("float", "double", "long"):
				self._error(f"'short' cannot be used with '{base}'", node)
		elif width in ("long", "long long"):
			if base in ("short", "char"):
				self._error(f"'{width}' cannot be used with '{base}'", node)

	# --- Visitor methods ---

	def visit_program(self, node: Program) -> None:
		for decl in node.declarations:
			self.visit(decl)

	def visit_function_decl(self, node: FunctionDecl) -> TypeSpec:
		node.return_type = self._resolve_type(node.return_type)
		self._check_duplicate_qualifiers(node.return_type, node)
		self._check_type_modifiers(node.return_type, node)
		for p in node.params:
			p.type_spec = self._resolve_type(p.type_spec)
		param_types = [p.type_spec for p in node.params]
		existing = self.symbols.lookup_current_scope(node.name)
		if existing is not None and existing.is_function and existing.is_prototype and node.body is not None:
			existing.is_prototype = False
			existing.param_types = param_types
			existing.is_variadic = node.is_variadic
		elif node.body is None:
			sym = Symbol(
				name=node.name,
				type_spec=node.return_type,
				scope_depth=self.symbols.depth,
				is_function=True,
				param_types=param_types,
				is_prototype=True,
				is_variadic=node.is_variadic,
			)
			if existing is not None:
				if existing.is_function and existing.is_prototype:
					return node.return_type
				self._error(f"redefinition of '{node.name}' in the same scope", node)
				return node.return_type
			self.symbols.define(sym)
			return node.return_type
		else:
			self._define_symbol(
				node.name, node.return_type, node, is_function=True,
				param_types=param_types, is_variadic=node.is_variadic,
			)
		if node.body is None:
			return node.return_type
		prev_func = self._current_function
		prev_label_defs = self._label_defs
		prev_goto_refs = self._goto_refs
		self._current_function = node
		self._label_defs = {}
		self._goto_refs = []
		self.symbols.push_scope()
		for param in node.params:
			self.visit(param)
		for stmt in node.body.statements:
			self.visit(stmt)
		for goto_label, goto_node in self._goto_refs:
			if goto_label not in self._label_defs:
				self._error(f"use of undeclared label '{goto_label}'", goto_node)
		# Warn if non-void function may fall off without returning
		if (
			node.return_type.base_type != "void" or node.return_type.pointer_count > 0
		) and node.body is not None:
			stmts = node.body.statements
			if not stmts or not isinstance(stmts[-1], ReturnStmt):
				self.warnings.append(
					f"control reaches end of non-void function '{node.name}'"
				)
		self.symbols.pop_scope()
		self._current_function = prev_func
		self._label_defs = prev_label_defs
		self._goto_refs = prev_goto_refs
		return node.return_type

	def visit_param_decl(self, node: ParamDecl) -> TypeSpec:
		node.type_spec = self._resolve_type(node.type_spec)
		self._check_type_modifiers(node.type_spec, node)
		if not node.name:
			return node.type_spec
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
		self._check_type_modifiers(node.type_spec, node)

		is_fp = node.type_spec.is_function_pointer

		# Detect char array initialized from string literal
		is_array = node.array_sizes is not None and len(node.array_sizes) > 0
		_is_char_array_str_init = (
			is_array
			and node.type_spec.base_type == "char"
			and node.type_spec.pointer_count == 0
			and isinstance(node.initializer, StringLiteral)
		)

		if node.initializer is not None:
			if isinstance(node.initializer, InitializerList):
				self._check_initializer_list(node)
			elif _is_char_array_str_init:
				pass  # char array from string literal is always valid; size checked below
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
		array_size_vals: list[int] = []
		if is_array:
			for size_expr in node.array_sizes:  # type: ignore[union-attr]
				if isinstance(size_expr, IntLiteral) and size_expr.value == 0:
					# Infer size from initializer
					if isinstance(node.initializer, InitializerList):
						inferred = len(node.initializer.elements)
						size_expr.value = inferred
						array_size_vals.append(inferred)
					elif _is_char_array_str_init:
						inferred = len(node.initializer.value) + 1  # +1 for null terminator
						size_expr.value = inferred
						array_size_vals.append(inferred)
					elif node.storage_class == "extern" or self.symbols.depth() == 0:
						# Extern/file-scope incomplete arrays are allowed
						array_size_vals.append(0)
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
			# C allows compatible redeclarations at file scope:
			# - duplicate extern declarations
			# - extern decl followed by definition (or vice versa)
			# - multiple tentative definitions (no initializer)
			# Only reject true redefinitions (both have initializers, neither extern).
			both_have_init = (
				existing.storage_class != "extern"
				and node.storage_class != "extern"
				and getattr(existing, '_has_initializer', False)
				and node.initializer is not None
			)
			if both_have_init:
				self._error(f"redefinition of '{node.name}' in the same scope", node)
				return node.type_spec
			# Update symbol table entry if this is a real definition replacing a declaration
			if existing.storage_class == "extern" and node.storage_class != "extern":
				existing.storage_class = node.storage_class
				existing.type_spec = node.type_spec
				existing.is_array = is_array
				existing.array_sizes = array_size_vals
				existing._has_initializer = node.initializer is not None
			elif node.initializer is not None:
				existing._has_initializer = True
			return node.type_spec
		if node.initializer is not None:
			sym._has_initializer = True
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
			# Check element count vs declared array size (only non-designated positional count)
			declared_size = 0
			for size_expr in node.array_sizes:  # type: ignore[union-attr]
				if isinstance(size_expr, IntLiteral) and size_expr.value > 0:
					declared_size = size_expr.value
			# Validate designated initializers
			for elem in init_list.elements:
				if isinstance(elem, DesignatedInit):
					if elem.field_name is not None:
						self._error("field designator in array initializer", elem)
					elif elem.index is not None:
						idx_type = self.visit(elem.index)
						if idx_type is not None and not _is_numeric(idx_type):
							self._error("array index designator must be an integer", elem)
						if isinstance(elem.index, IntLiteral) and declared_size > 0:
							if elem.index.value < 0 or elem.index.value >= declared_size:
								self._error(
									f"array index {elem.index.value} out of range for array of size {declared_size}",
									elem,
								)
					self._visit_init_value(elem.value, node.type_spec)
				elif isinstance(elem, InitializerList):
					self.visit_initializer_list(elem)
				else:
					self.visit(elem)
			# Check positional count doesn't exceed total array capacity
			if declared_size > 0 and not any(isinstance(e, DesignatedInit) for e in init_list.elements):
				# For multi-dimensional arrays, compute total capacity
				total_capacity = 1
				for size_expr in node.array_sizes:  # type: ignore[union-attr]
					if isinstance(size_expr, IntLiteral) and size_expr.value > 0:
						total_capacity *= size_expr.value
				if len(init_list.elements) > total_capacity:
					self._error("excess elements in array initializer", node)
		elif base_type.startswith("struct "):
			struct_name = base_type[len("struct "):]
			if struct_name in self._struct_types:
				struct_decl = self._struct_types[struct_name]
				member_names = {m.name for m in struct_decl.members}
				positional_idx = 0
				for elem in init_list.elements:
					if isinstance(elem, DesignatedInit):
						if elem.index is not None:
							self._error("array index designator in struct initializer", elem)
						elif elem.field_name is not None:
							if elem.field_name not in member_names:
								self._error(
									f"struct '{struct_name}' has no member '{elem.field_name}'",
									elem,
								)
							else:
								# Find the member for type checking
								for m in struct_decl.members:
									if m.name == elem.field_name:
										self._visit_init_value(elem.value, m.type_spec)
										break
						continue
					if isinstance(elem, InitializerList):
						self.visit_initializer_list(elem)
					else:
						elem_t = self.visit(elem)
						if positional_idx < len(struct_decl.members) and elem_t is not None:
							member_type = struct_decl.members[positional_idx].type_spec
							if not _types_compatible(member_type, elem_t):
								self._error(
									f"incompatible type for member '{struct_decl.members[positional_idx].name}'",
									elem,
								)
					positional_idx += 1
				# Check excess positional elements
				if not any(isinstance(e, DesignatedInit) for e in init_list.elements):
					if len(init_list.elements) > len(struct_decl.members):
						self._error("excess elements in struct initializer", node)
			else:
				for elem in init_list.elements:
					self.visit(elem)
		else:
			for elem in init_list.elements:
				self.visit(elem)

	def _visit_init_value(self, value: ASTNode, expected_type: TypeSpec) -> None:
		"""Visit an initializer value and check type compatibility."""
		if isinstance(value, InitializerList):
			self.visit_initializer_list(value)
		else:
			val_type = self.visit(value)
			resolved_expected = self._resolve_type(expected_type)
			if val_type is not None and not _types_compatible(resolved_expected, val_type):
				self._error(
					"incompatible type in designated initializer",
					value,
				)

	def visit_designated_init(self, node: DesignatedInit) -> TypeSpec | None:
		return self.visit(node.value)

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
		is_void_func = (
			self._current_function is not None
			and self._current_function.return_type.base_type == "void"
			and self._current_function.return_type.pointer_count == 0
		)
		# In C, returning a value from a void function is allowed (just a warning)
		if is_void_func and node.has_expression:
			self.warnings.append("void function should not return a value")
			self.visit(node.expression)
			return
		# Bare return in void function is fine; skip type check
		if is_void_func and not node.has_expression:
			return
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
			if isinstance(node.init, list):
				for decl in node.init:
					self.visit(decl)
			else:
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

	def visit_goto_stmt(self, node: GotoStmt) -> None:
		self._goto_refs.append((node.label, node))

	def visit_label_stmt(self, node: LabelStmt) -> None:
		if node.label in self._label_defs:
			self._error(f"redefinition of label '{node.label}'", node)
		else:
			self._label_defs[node.label] = node
		self.visit(node.statement)

	def visit_compound_assignment(self, node: CompoundAssignment) -> TypeSpec | None:
		target_type = self.visit(node.target)
		value_type = self.visit(node.value)
		# Const check
		if self._is_const_target(node.target):
			self._error("assignment to const variable", node)
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
		elif node.op in _BITWISE_OPS:
			if not _is_numeric(target_type) or not _is_numeric(value_type):
				self._error(
					f"incompatible types for bitwise compound assignment '{node.op}='",
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
			# Pointer arithmetic validation
			if _is_pointer(left_type) or _is_pointer(right_type):
				if node.op == "+":
					if _is_pointer(left_type) and _is_pointer(right_type):
						self._error("addition of two pointers is not allowed", node)
						return None
					if not (_is_numeric(left_type) or _is_numeric(right_type)):
						self._error(
							f"incompatible types for operator '+': "
							f"'{left_type.base_type}' and '{right_type.base_type}'",
							node,
						)
						return None
				elif node.op == "-":
					# ptr - ptr OK, ptr - int OK, int - ptr error
					if _is_pointer(right_type) and not _is_pointer(left_type):
						self._error(
							"subtraction of pointer from integer is not allowed",
							node,
						)
						return None
				else:
					# *, /, %, bitwise ops not allowed with pointers
					self._error(
						f"invalid operands to '{node.op}': pointer type not allowed",
						node,
					)
					return None
			elif not _is_numeric(left_type) or not _is_numeric(right_type):
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
			# Check: cannot take address of register variable (C89 6.3.3.2)
			if sym is not None and sym.storage_class == "register":
				self._error(
					f"address of register variable '{node.operand.name}' requested",
					node,
				)
		# Address-of: inhibit array decay for the operand
		if node.op == "&":
			old_flag = self._in_sizeof_or_addressof
			self._in_sizeof_or_addressof = True
			operand_type = self.visit(node.operand)
			self._in_sizeof_or_addressof = old_flag
			if operand_type is None:
				return None
			return TypeSpec(
				base_type=operand_type.base_type,
				pointer_count=operand_type.pointer_count + 1,
			)
		operand_type = self.visit(node.operand)
		if operand_type is None:
			return None
		if node.op in ("++", "--"):
			if self._is_const_target(node.operand):
				self._error("increment/decrement of const variable", node)
		if node.op == "*":
			if operand_type.pointer_count < 1:
				# Arrays decay to pointers and can be dereferenced
				if isinstance(node.operand, Identifier):
					sym = self.symbols.lookup(node.operand.name)
					if sym is not None and sym.is_array:
						return TypeSpec(base_type=operand_type.base_type)
				self._error("dereference of non-pointer type", node)
				return None
			return TypeSpec(
				base_type=operand_type.base_type,
				pointer_count=operand_type.pointer_count - 1,
			)
		# Unary minus and bitwise not: integer promotion applies
		if node.op in ("-", "~"):
			return _integer_promote(operand_type)
		# Logical not always yields int
		if node.op == "!":
			return TypeSpec(base_type="int")
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
		# Array-to-pointer decay: arrays decay to pointers in expression contexts
		# except inside sizeof or address-of (&)
		if sym.is_array and not self._in_sizeof_or_addressof:
			return TypeSpec(
				base_type=sym.type_spec.base_type,
				pointer_count=sym.type_spec.pointer_count + 1,
			)
		return sym.type_spec

	def visit_assignment(self, node: Assignment) -> TypeSpec | None:
		target_type = self.visit(node.target)
		# Const check
		if self._is_const_target(node.target):
			self._error("assignment to const variable", node)
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
		# Indirect call via expression callee (e.g. (*fp)(args), arr[i](args))
		if node.callee is not None:
			self.visit(node.callee)
			for arg in node.arguments:
				self.visit(arg)
			return TypeSpec(base_type="int")
		sym = self.symbols.lookup(node.name)
		if sym is None:
			# C89 implicit function declaration: assume int return, unspecified params
			self.warnings.append(
				f"implicit declaration of function '{node.name}'"
			)
			implicit_type = TypeSpec(base_type="int")
			implicit_sym = Symbol(
				name=node.name,
				type_spec=implicit_type,
				scope_depth=0,
				is_function=True,
				param_types=[],
				is_variadic=True,
			)
			self.symbols.define(implicit_sym)
			for arg in node.arguments:
				self.visit(arg)
			return implicit_type
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
			if sym.storage_class == "extern":
				# Local extern function declarations are parsed as VarDecl;
				# allow calls through them
				for arg in node.arguments:
					self.visit(arg)
				return sym.type_spec
			self._error(f"'{node.name}' is not a function", node)
			return None
		# In C, a prototype `int f()` means unspecified params (accept any args)
		# A definition `int f() { ... }` with zero params means zero args
		has_unspecified_params = sym.is_prototype and len(sym.param_types) == 0 and not sym.is_variadic
		if sym.is_variadic:
			if len(node.arguments) < len(sym.param_types):
				self._error(
					f"variadic function '{node.name}' requires at least "
					f"{len(sym.param_types)} arguments, got {len(node.arguments)}",
					node,
				)
		elif has_unspecified_params:
			pass  # Unprototyped function: accept any number of arguments
		elif len(node.arguments) != len(sym.param_types):
			self._error(
				f"function '{node.name}' expects {len(sym.param_types)} arguments, "
				f"got {len(node.arguments)}",
				node,
			)
		for arg in node.arguments:
			self.visit(arg)
		return sym.type_spec

	def visit_array_subscript(self, node: ArraySubscript) -> TypeSpec | None:
		# Array subscript always needs array decay on the base expression
		old_flag = self._in_sizeof_or_addressof
		self._in_sizeof_or_addressof = False
		base_type = self.visit(node.array)
		self._in_sizeof_or_addressof = old_flag
		index_type = self.visit(node.index)
		if base_type is None:
			return None
		# Check base is array or pointer
		is_base_array = False
		if isinstance(node.array, Identifier):
			sym = self.symbols.lookup(node.array.name)
			if sym is not None:
				is_base_array = sym.is_array
		elif isinstance(node.array, ArraySubscript):
			# Nested subscript (multi-dim array): result of inner subscript is still subscriptable
			is_base_array = True
		elif isinstance(node.array, MemberAccess):
			# Member access may yield an array type; trust the type system
			is_base_array = True
		elif isinstance(node.array, (UnaryOp, CastExpr, FunctionCall, CompoundLiteral)):
			# Complex expressions (dereference, cast, function call) may produce arrays/pointers
			is_base_array = True
		if not is_base_array and not _is_pointer(base_type):
			self._error("subscript requires array or pointer type", node)
			return None
		# Check index is integer
		if index_type is not None and not _is_numeric(index_type):
			self._error("array index must be an integer", node)
			return None
		# Result type: dereference one pointer level (arrays decay to pointers, so subscript undoes one level)
		if _is_pointer(base_type):
			return TypeSpec(
				base_type=base_type.base_type,
				pointer_count=base_type.pointer_count - 1,
			)
		return TypeSpec(base_type=base_type.base_type)

	def visit_switch_stmt(self, node: SwitchStmt) -> None:
		self.visit(node.expression)
		self._switch_depth += 1
		seen_values: set[int] = set()
		has_default = False
		evaluator = self._get_const_evaluator()
		for case in node.cases:
			if case.is_pre_switch:
				pass  # Pre-switch statements (before first case/default)
			elif case.value is None:
				if has_default:
					self._error("duplicate default label in switch", case)
				has_default = True
			else:
				val = evaluator.evaluate(case.value)
				if val is not None:
					if val in seen_values:
						self._error(f"duplicate case value {val!r}", case)
					seen_values.add(val)
				else:
					self._error("case expression must be a constant integer", case)
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
			# Apply usual arithmetic conversions to determine result type
			if _is_numeric(true_type) and _is_numeric(false_type):
				return _usual_arithmetic_conversions(true_type, false_type)
			# Pointer types: result is the pointer type
			if _is_pointer(true_type):
				return true_type
			if _is_pointer(false_type):
				return false_type
		return true_type

	def visit_sizeof_expr(self, node: SizeofExpr) -> TypeSpec:
		if node.operand is not None:
			old_flag = self._in_sizeof_or_addressof
			self._in_sizeof_or_addressof = True
			self.visit(node.operand)
			self._in_sizeof_or_addressof = old_flag
		return TypeSpec(base_type="int")

	def visit_postfix_expr(self, node: PostfixExpr) -> TypeSpec | None:
		operand_type = self.visit(node.operand)
		if not self._is_lvalue(node.operand):
			self._error("operand of postfix operator must be an lvalue", node)
		if self._is_const_target(node.operand):
			self._error("increment/decrement of const variable", node)
		return operand_type

	def visit_cast_expr(self, node: CastExpr) -> TypeSpec:
		operand_type = self.visit(node.operand)
		node.target_type = self._resolve_type(node.target_type)
		target = node.target_type
		if operand_type is not None:
			# Only reject casts involving struct/union types (non-scalar)
			src_struct = operand_type.base_type.startswith(("struct ", "union ")) and operand_type.pointer_count == 0
			tgt_struct = target.base_type.startswith(("struct ", "union ")) and target.pointer_count == 0
			if src_struct or tgt_struct:
				self._error(
					f"invalid cast from '{operand_type.base_type}' to '{target.base_type}'",
					node,
				)
		return target

	def visit_compound_literal(self, node: CompoundLiteral) -> TypeSpec:
		node.type_spec = self._resolve_type(node.type_spec)
		self.visit(node.init_list)
		return node.type_spec

	def visit_comma_expr(self, node: CommaExpr) -> TypeSpec | None:
		self.visit(node.left)
		return self.visit(node.right)

	def visit_va_start_expr(self, node: VaStartExpr) -> TypeSpec | None:
		self.visit(node.ap)
		return TypeSpec(base_type="void")

	def visit_va_arg_expr(self, node: VaArgExpr) -> TypeSpec | None:
		self.visit(node.ap)
		return node.arg_type

	def visit_va_end_expr(self, node: VaEndExpr) -> TypeSpec | None:
		self.visit(node.ap)
		return TypeSpec(base_type="void")

	def visit_va_copy_expr(self, node: VaCopyExpr) -> TypeSpec | None:
		self.visit(node.dest)
		self.visit(node.src)
		return TypeSpec(base_type="void")

	def _is_const_target(self, node: ASTNode) -> bool:
		"""Return True if *node* refers to a const-qualified variable.

		For pointer types like ``const int *p``, the const applies to the
		pointee, not the pointer itself, so ``p`` is NOT const.
		"""
		if isinstance(node, Identifier):
			sym = self.symbols.lookup(node.name)
			if sym is not None and "const" in sym.type_spec.qualifiers:
				# const int *p  ->  p is mutable (pointee is const)
				if sym.type_spec.pointer_count > 0:
					return False
				return True
		return False

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
				evaluator = self._get_const_evaluator()
				val = evaluator.evaluate(const.value)
				if val is not None:
					next_value = val
				else:
					self.visit(const.value)
			self._enum_constants[const.name] = next_value
			self._define_symbol(const.name, TypeSpec(base_type="int"), const)
			next_value += 1

	def visit_struct_decl(self, node: StructDecl) -> None:
		existing = self._struct_types.get(node.name)
		if existing is not None:
			if len(node.members) == 0:
				# Repeated forward declaration is fine
				return
			if len(existing.members) > 0:
				# Full definition already exists
				self._error(f"redefinition of struct '{node.name}'", node)
				return
		seen: set[str] = set()
		for member in node.members:
			if member.name and member.name in seen:
				self._error(f"duplicate member '{member.name}' in struct '{node.name}'", member)
			elif member.name:
				seen.add(member.name)
			self._validate_bitfield(member, node.name)
		self._struct_types[node.name] = node
		for sa in node.static_asserts:
			self.visit_static_assert_decl(sa)

	def _validate_bitfield(self, member: StructMember, container_name: str) -> None:
		if member.bit_width is None:
			return
		if member.type_spec.pointer_count > 0:
			self._error(f"bitfield '{member.name}' cannot be a pointer type", member)
			return
		if member.array_dims:
			self._error(f"bitfield '{member.name}' cannot be an array", member)
			return
		base = member.type_spec.base_type
		type_widths = {
			"int": 32, "unsigned int": 32, "signed int": 32,
			"char": 8, "unsigned char": 8, "signed char": 8,
			"short": 16, "unsigned short": 16, "signed short": 16,
			"long": 64, "unsigned long": 64, "signed long": 64,
			"long long": 64, "unsigned long long": 64, "signed long long": 64,
			"_Bool": 1, "bool": 1,
			"int8_t": 8, "uint8_t": 8,
			"int16_t": 16, "uint16_t": 16,
			"int32_t": 32, "uint32_t": 32,
			"int64_t": 64, "uint64_t": 64,
			"intptr_t": 64, "uintptr_t": 64,
			"size_t": 64, "ptrdiff_t": 64,
		}
		max_width = type_widths.get(base, 32)
		if member.type_spec.width_modifier == "short":
			max_width = 16
		elif member.type_spec.width_modifier in ("long", "long long"):
			max_width = 64
		if member.bit_width < 0:
			self._error(f"bitfield width cannot be negative in '{container_name}'", member)
		elif member.bit_width == 0 and member.name:
			self._error(f"named bitfield '{member.name}' cannot have zero width", member)
		elif member.bit_width > max_width:
			self._error(
				f"bitfield '{member.name or '(unnamed)'}' width {member.bit_width} exceeds type width {max_width}",
				member,
			)

	def visit_union_decl(self, node: UnionDecl) -> None:
		existing = self._union_types.get(node.name)
		if existing is not None:
			if len(node.members) == 0:
				return
			if len(existing.members) > 0:
				self._error(f"redefinition of union '{node.name}'", node)
				return
		seen: set[str] = set()
		for member in node.members:
			if member.name and member.name in seen:
				self._error(f"duplicate member '{member.name}' in union '{node.name}'", member)
			elif member.name:
				seen.add(member.name)
			self._validate_bitfield(member, node.name)
		self._union_types[node.name] = node
		for sa in node.static_asserts:
			self.visit_static_assert_decl(sa)

	def visit_typedef_decl(self, node: TypedefDecl) -> None:
		if node.struct_decl is not None:
			self.visit_struct_decl(node.struct_decl)
		if node.enum_decl is not None:
			self.visit_enum_decl(node.enum_decl)
		if node.union_decl is not None:
			self.visit_union_decl(node.union_decl)
		resolved = self._resolve_type(node.type_spec)
		if node.name in self._typedef_types and node.name not in self._predefined_typedef_names:
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
				f"union {base_name} has no member {node.member}",
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
			f"struct {base_name} has no member {node.member}",
			node,
		)
		return None

	def visit_static_assert_decl(self, node: StaticAssertDecl) -> None:
		val = self._try_eval_constant(node.expression)
		if val is None:
			self._error("_Static_assert expression is not a constant expression", node)
			return
		if val == 0:
			msg = node.message
			from compiler.lexer import interpret_c_escapes
			msg = interpret_c_escapes(msg)
			self._error(f"static assertion failed: \"{msg}\"", node)

	def _get_const_evaluator(self) -> ConstExprEvaluator:
		"""Create a ConstExprEvaluator with current semantic state."""
		return ConstExprEvaluator(
			enum_constants=self._enum_constants,
			struct_types=self._struct_types,
			union_types=self._union_types,
			typedef_map=self._typedef_types,
		)

	def _try_eval_constant(self, node: ASTNode) -> int | None:
		"""Try to evaluate an AST expression as a compile-time integer constant."""
		return self._get_const_evaluator().evaluate(node)

	def visit_type_spec(self, node: TypeSpec) -> TypeSpec:
		return self._resolve_type(node)
