"""AST node definitions for the C compiler subset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from typing import Any


@dataclass
class SourceLocation:
	"""Source position in the original C file."""

	line: int
	col: int


@dataclass
class ASTNode:
	"""Base class for all AST nodes."""

	loc: SourceLocation = field(default_factory=lambda: SourceLocation(0, 0))

	def accept(self, visitor: ASTVisitor) -> Any:
		raise NotImplementedError


class ASTVisitor:
	"""Base visitor for traversing AST nodes. Override visit_* methods as needed."""

	def visit(self, node: ASTNode) -> Any:
		return node.accept(self)

	def visit_program(self, node: Program) -> Any:
		return None

	def visit_function_decl(self, node: FunctionDecl) -> Any:
		return None

	def visit_param_decl(self, node: ParamDecl) -> Any:
		return None

	def visit_var_decl(self, node: VarDecl) -> Any:
		return None

	def visit_compound_stmt(self, node: CompoundStmt) -> Any:
		return None

	def visit_return_stmt(self, node: ReturnStmt) -> Any:
		return None

	def visit_if_stmt(self, node: IfStmt) -> Any:
		return None

	def visit_while_stmt(self, node: WhileStmt) -> Any:
		return None

	def visit_for_stmt(self, node: ForStmt) -> Any:
		return None

	def visit_expr_stmt(self, node: ExprStmt) -> Any:
		return None

	def visit_binary_op(self, node: BinaryOp) -> Any:
		return None

	def visit_unary_op(self, node: UnaryOp) -> Any:
		return None

	def visit_int_literal(self, node: IntLiteral) -> Any:
		return None

	def visit_string_literal(self, node: StringLiteral) -> Any:
		return None

	def visit_char_literal(self, node: CharLiteral) -> Any:
		return None

	def visit_identifier(self, node: Identifier) -> Any:
		return None

	def visit_assignment(self, node: Assignment) -> Any:
		return None

	def visit_function_call(self, node: FunctionCall) -> Any:
		return None

	def visit_type_spec(self, node: TypeSpec) -> Any:
		return None


# --- Type node ---


@dataclass
class TypeSpec(ASTNode):
	"""Type specifier: base type name with optional pointer indirection."""

	base_type: str = ""
	pointer_count: int = 0

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_type_spec(self)


# --- Expression nodes ---


@dataclass
class IntLiteral(ASTNode):
	"""Integer literal, e.g. 42."""

	value: int = 0

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_int_literal(self)


@dataclass
class StringLiteral(ASTNode):
	"""String literal, e.g. "hello"."""

	value: str = ""

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_string_literal(self)


@dataclass
class CharLiteral(ASTNode):
	"""Character literal, e.g. 'a'."""

	value: str = ""

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_char_literal(self)


@dataclass
class Identifier(ASTNode):
	"""Variable or function name reference."""

	name: str = ""

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_identifier(self)


@dataclass
class BinaryOp(ASTNode):
	"""Binary operation: left op right."""

	left: ASTNode = field(default_factory=ASTNode)
	op: str = ""
	right: ASTNode = field(default_factory=ASTNode)

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_binary_op(self)


@dataclass
class UnaryOp(ASTNode):
	"""Unary operation: op operand (prefix or postfix)."""

	op: str = ""
	operand: ASTNode = field(default_factory=ASTNode)
	prefix: bool = True

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_unary_op(self)


@dataclass
class Assignment(ASTNode):
	"""Assignment: target = value."""

	target: ASTNode = field(default_factory=ASTNode)
	value: ASTNode = field(default_factory=ASTNode)

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_assignment(self)


@dataclass
class FunctionCall(ASTNode):
	"""Function call: name(arguments)."""

	name: str = ""
	arguments: list[ASTNode] = field(default_factory=list)

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_function_call(self)


# --- Statement nodes ---


@dataclass
class ExprStmt(ASTNode):
	"""Expression statement wrapper."""

	expression: ASTNode = field(default_factory=ASTNode)

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_expr_stmt(self)


@dataclass
class ReturnStmt(ASTNode):
	"""Return statement with expression."""

	expression: ASTNode = field(default_factory=ASTNode)

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_return_stmt(self)


@dataclass
class CompoundStmt(ASTNode):
	"""Block of statements enclosed in braces."""

	statements: list[ASTNode] = field(default_factory=list)

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_compound_stmt(self)


@dataclass
class IfStmt(ASTNode):
	"""If statement with optional else branch."""

	condition: ASTNode = field(default_factory=ASTNode)
	then_branch: ASTNode = field(default_factory=ASTNode)
	else_branch: ASTNode | None = None

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_if_stmt(self)


@dataclass
class WhileStmt(ASTNode):
	"""While loop."""

	condition: ASTNode = field(default_factory=ASTNode)
	body: ASTNode = field(default_factory=ASTNode)

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_while_stmt(self)


@dataclass
class ForStmt(ASTNode):
	"""For loop: for(init; condition; update) body."""

	init: ASTNode | None = None
	condition: ASTNode | None = None
	update: ASTNode | None = None
	body: ASTNode = field(default_factory=ASTNode)

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_for_stmt(self)


# --- Declaration nodes ---


@dataclass
class ParamDecl(ASTNode):
	"""Function parameter declaration."""

	type_spec: TypeSpec = field(default_factory=TypeSpec)
	name: str = ""

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_param_decl(self)


@dataclass
class VarDecl(ASTNode):
	"""Variable declaration with optional initializer."""

	type_spec: TypeSpec = field(default_factory=TypeSpec)
	name: str = ""
	initializer: ASTNode | None = None

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_var_decl(self)


@dataclass
class FunctionDecl(ASTNode):
	"""Function declaration: return_type name(params) body."""

	return_type: TypeSpec = field(default_factory=TypeSpec)
	name: str = ""
	params: list[ParamDecl] = field(default_factory=list)
	body: CompoundStmt = field(default_factory=CompoundStmt)

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_function_decl(self)


# --- Top-level node ---


@dataclass
class Program(ASTNode):
	"""Top-level AST node holding all declarations."""

	declarations: list[ASTNode] = field(default_factory=list)

	def accept(self, visitor: ASTVisitor) -> Any:
		return visitor.visit_program(self)
