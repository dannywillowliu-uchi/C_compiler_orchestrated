"""Compile-time constant expression evaluator for C.

Evaluates constant expressions at compile time for:
- _Static_assert condition evaluation
- Switch/case label deduplication
- Enum member values
- Array dimension validation
- Bitfield width validation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from compiler.ast_nodes import (
	BinaryOp,
	CastExpr,
	CharLiteral,
	CommaExpr,
	Identifier,
	IntLiteral,
	SizeofExpr,
	TernaryExpr,
	TypeSpec,
	UnaryOp,
)

if TYPE_CHECKING:
	from compiler.ast_nodes import ASTNode


class ConstEvalError(Exception):
	"""Raised when a constant expression cannot be evaluated."""

	def __init__(self, message: str, line: int = 0, col: int = 0) -> None:
		self.line = line
		self.col = col
		super().__init__(message)


# Standard sizes for sizeof evaluation (x86-64 LP64)
_SIZE_MAP: dict[str, int] = {
	"char": 1,
	"unsigned char": 1,
	"signed char": 1,
	"short": 2,
	"unsigned short": 2,
	"int": 4,
	"unsigned int": 4,
	"long": 8,
	"unsigned long": 8,
	"long long": 8,
	"unsigned long long": 8,
	"float": 4,
	"double": 8,
	"long double": 16,
	"void": 1,
	"_Bool": 1,
}


class ConstExprEvaluator:
	"""Evaluates AST expressions as compile-time integer constants.

	Handles: integer literals, char literals, enum constants, sizeof,
	arithmetic/bitwise/logical/comparison operators, casts, and ternary expressions.
	"""

	def __init__(
		self,
		enum_constants: dict[str, int] | None = None,
		struct_types: dict | None = None,
		union_types: dict | None = None,
		typedef_map: dict[str, TypeSpec] | None = None,
	) -> None:
		self._enum_constants: dict[str, int] = enum_constants or {}
		self._struct_types = struct_types or {}
		self._union_types = union_types or {}
		self._typedef_map: dict[str, TypeSpec] = typedef_map or {}

	def evaluate(self, node: ASTNode) -> int | None:
		"""Evaluate an AST node as a compile-time integer constant.

		Returns the integer value if the expression is a valid constant expression,
		or None if it cannot be evaluated at compile time.
		"""
		if isinstance(node, IntLiteral):
			return node.value

		if isinstance(node, CharLiteral):
			return self._eval_char(node)

		if isinstance(node, Identifier):
			return self._enum_constants.get(node.name)

		if isinstance(node, UnaryOp):
			return self._eval_unary(node)

		if isinstance(node, BinaryOp):
			return self._eval_binary(node)

		if isinstance(node, TernaryExpr):
			return self._eval_ternary(node)

		if isinstance(node, SizeofExpr):
			return self._eval_sizeof(node)

		if isinstance(node, CastExpr):
			return self._eval_cast(node)

		if isinstance(node, CommaExpr):
			return self._eval_comma(node)

		return None

	def _eval_char(self, node: CharLiteral) -> int:
		"""Evaluate a character literal to its integer value."""
		from compiler.lexer import interpret_c_escapes

		ch = interpret_c_escapes(node.value)
		return ord(ch[0]) if ch else 0

	def _eval_unary(self, node: UnaryOp) -> int | None:
		if not node.prefix:
			return None
		operand = self.evaluate(node.operand)
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

	def _eval_binary(self, node: BinaryOp) -> int | None:
		# Short-circuit for logical operators
		if node.op == "&&":
			left = self.evaluate(node.left)
			if left is None:
				return None
			if not left:
				return 0
			right = self.evaluate(node.right)
			if right is None:
				return None
			return 1 if right else 0

		if node.op == "||":
			left = self.evaluate(node.left)
			if left is None:
				return None
			if left:
				return 1
			right = self.evaluate(node.right)
			if right is None:
				return None
			return 1 if right else 0

		left = self.evaluate(node.left)
		right = self.evaluate(node.right)
		if left is None or right is None:
			return None

		op = node.op
		if op == "+":
			return left + right
		if op == "-":
			return left - right
		if op == "*":
			return left * right
		if op == "/":
			if right == 0:
				return None
			# C integer division truncates toward zero
			if (left < 0) != (right < 0) and left % right != 0:
				return -(abs(left) // abs(right))
			return left // right
		if op == "%":
			if right == 0:
				return None
			# C modulo: result has same sign as dividend
			result = abs(left) % abs(right)
			return -result if left < 0 else result
		if op == "<<":
			if right < 0:
				return None
			return left << right
		if op == ">>":
			if right < 0:
				return None
			return left >> right
		if op == "&":
			return left & right
		if op == "|":
			return left | right
		if op == "^":
			return left ^ right
		if op == "==":
			return 1 if left == right else 0
		if op == "!=":
			return 1 if left != right else 0
		if op == "<":
			return 1 if left < right else 0
		if op == ">":
			return 1 if left > right else 0
		if op == "<=":
			return 1 if left <= right else 0
		if op == ">=":
			return 1 if left >= right else 0
		return None

	def _eval_ternary(self, node: TernaryExpr) -> int | None:
		cond = self.evaluate(node.condition)
		if cond is None:
			return None
		return self.evaluate(node.true_expr if cond else node.false_expr)

	def _eval_sizeof(self, node: SizeofExpr) -> int | None:
		"""Evaluate sizeof for constant expression contexts."""
		if node.type_operand is not None:
			return self._sizeof_type(node.type_operand)
		if node.operand is not None:
			return self._sizeof_expr(node.operand)
		return None

	def _sizeof_type(self, ts: TypeSpec) -> int | None:
		"""Compute sizeof for a type specifier."""
		if ts.pointer_count > 0:
			return 8

		base = ts.base_type

		# Resolve typedefs
		if base in self._typedef_map:
			resolved = self._typedef_map[base]
			return self._sizeof_type(resolved)

		# Handle width modifiers
		if ts.width_modifier == "short":
			return 2
		if ts.width_modifier in ("long", "long long"):
			if base in ("int", ""):
				return 8 if ts.width_modifier == "long long" else 8
			if base == "double":
				return 16

		# Handle signedness with no explicit base (e.g., "unsigned" alone means "unsigned int")
		if ts.signedness and not base:
			return 4

		# Build full type name for lookup
		full_name = base
		if ts.signedness and base in ("char", ""):
			full_name = f"{ts.signedness} {base}".strip()

		if full_name in _SIZE_MAP:
			return _SIZE_MAP[full_name]

		# Struct/union sizeof
		if base.startswith("struct ") or base in self._struct_types:
			name = base.replace("struct ", "") if base.startswith("struct ") else base
			return self._sizeof_struct(name)
		if base.startswith("union ") or base in self._union_types:
			name = base.replace("union ", "") if base.startswith("union ") else base
			return self._sizeof_union(name)

		# Enum types are int-sized
		if base.startswith("enum "):
			return 4

		return _SIZE_MAP.get(base)

	def _sizeof_struct(self, name: str) -> int | None:
		"""Compute sizeof for a struct (simplified: sum of member sizes, no padding)."""
		decl = self._struct_types.get(name)
		if decl is None:
			return None
		total = 0
		for member in decl.members:
			size = self._sizeof_type(member.type_spec)
			if size is None:
				return None
			# Handle arrays
			for dim in member.array_dims:
				dim_val = self.evaluate(dim)
				if dim_val is None:
					return None
				size *= dim_val
			total += size
		return total

	def _sizeof_union(self, name: str) -> int | None:
		"""Compute sizeof for a union (max of member sizes)."""
		decl = self._union_types.get(name)
		if decl is None:
			return None
		max_size = 0
		for member in decl.members:
			size = self._sizeof_type(member.type_spec)
			if size is None:
				return None
			if size > max_size:
				max_size = size
		return max_size

	def _sizeof_expr(self, expr: ASTNode) -> int | None:
		"""Compute sizeof for an expression (by type inference)."""
		if isinstance(expr, IntLiteral):
			return 4
		if isinstance(expr, CharLiteral):
			return 1
		if isinstance(expr, Identifier):
			if expr.name in self._enum_constants:
				return 4
			return None
		return None

	def _eval_cast(self, node: CastExpr) -> int | None:
		"""Evaluate a cast expression on a constant."""
		val = self.evaluate(node.operand)
		if val is None:
			return None
		target = node.target_type
		if target.pointer_count > 0:
			return val
		base = target.base_type
		width = target.width_modifier

		# Truncate/sign-extend based on target type
		if base == "char" or (base == "" and width is None and target.signedness == "signed"):
			if target.signedness == "unsigned":
				return val & 0xFF
			return _sign_extend(val & 0xFF, 8)
		if base == "_Bool":
			return 0 if val == 0 else 1
		if width == "short" or base == "short":
			if target.signedness == "unsigned":
				return val & 0xFFFF
			return _sign_extend(val & 0xFFFF, 16)
		if width in ("long", "long long") or base in ("long",):
			if target.signedness == "unsigned":
				return val & 0xFFFFFFFFFFFFFFFF
			return _sign_extend(val & 0xFFFFFFFFFFFFFFFF, 64)
		# Default: int
		if target.signedness == "unsigned":
			return val & 0xFFFFFFFF
		return _sign_extend(val & 0xFFFFFFFF, 32)

	def _eval_comma(self, node: CommaExpr) -> int | None:
		"""Comma expression: evaluate both, return right."""
		self.evaluate(node.left)
		return self.evaluate(node.right)


def _sign_extend(val: int, bits: int) -> int:
	"""Sign-extend a value from the given bit width to Python int."""
	sign_bit = 1 << (bits - 1)
	if val & sign_bit:
		return val - (1 << bits)
	return val
