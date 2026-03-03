"""Semantic analysis: symbol table management and type checking for the C compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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


def _is_numeric(ts: TypeSpec) -> bool:
	return ts.base_type in _NUMERIC_TYPES and ts.pointer_count == 0


def _is_pointer(ts: TypeSpec) -> bool:
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

	# --- Visitor methods ---

	def visit_program(self, node: Program) -> None:
		for decl in node.declarations:
			self.visit(decl)

	def visit_function_decl(self, node: FunctionDecl) -> TypeSpec:
		param_types = [p.type_spec for p in node.params]
		self._define_symbol(
			node.name, node.return_type, node, is_function=True, param_types=param_types
		)
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
		self._define_symbol(node.name, node.type_spec, node)
		return node.type_spec

	def visit_var_decl(self, node: VarDecl) -> TypeSpec:
		if node.initializer is not None:
			init_type = self.visit(node.initializer)
			if init_type is not None and not _types_compatible(node.type_spec, init_type):
				self._error(
					f"incompatible types in initialization of '{node.name}': "
					f"'{node.type_spec.base_type}' and '{init_type.base_type}'",
					node,
				)
		self._define_symbol(node.name, node.type_spec, node)
		return node.type_spec

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
		self.visit(node.body)

	def visit_for_stmt(self, node: ForStmt) -> None:
		self.symbols.push_scope()
		if node.init is not None:
			self.visit(node.init)
		if node.condition is not None:
			self.visit(node.condition)
		if node.update is not None:
			self.visit(node.update)
		self.visit(node.body)
		self.symbols.pop_scope()

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

	def visit_string_literal(self, node: StringLiteral) -> TypeSpec:
		return TypeSpec(base_type="char", pointer_count=1)

	def visit_char_literal(self, node: CharLiteral) -> TypeSpec:
		return TypeSpec(base_type="char")

	def visit_identifier(self, node: Identifier) -> TypeSpec | None:
		sym = self.symbols.lookup(node.name)
		if sym is None:
			self._error(f"use of undeclared identifier '{node.name}'", node)
			return None
		return sym.type_spec

	def visit_assignment(self, node: Assignment) -> TypeSpec | None:
		target_type = self.visit(node.target)
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

	def visit_type_spec(self, node: TypeSpec) -> TypeSpec:
		return node
