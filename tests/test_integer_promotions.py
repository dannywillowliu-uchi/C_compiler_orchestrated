"""Tests for C integer promotion and usual arithmetic conversion rules."""

from compiler.ast_nodes import (
	BinaryOp,
	CharLiteral,
	CompoundStmt,
	FloatLiteral,
	FunctionDecl,
	Identifier,
	IntLiteral,
	Program,
	ReturnStmt,
	SourceLocation,
	TernaryExpr,
	TypeSpec,
	UnaryOp,
	VarDecl,
)
from compiler.semantic import (
	SemanticAnalyzer,
	_integer_promote,
	_type_rank,
	_typespec_from_rank,
	_usual_arithmetic_conversions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def loc() -> SourceLocation:
	return SourceLocation(line=1, col=1)


def make_type(base: str, **kw) -> TypeSpec:
	return TypeSpec(base_type=base, loc=loc(), **kw)


class _TypeCapture(SemanticAnalyzer):
	"""Subclass that captures the type of the return expression."""

	def __init__(self) -> None:
		super().__init__()
		self.captured_type: TypeSpec | None = None

	def visit_return_stmt(self, node: ReturnStmt) -> None:
		if node.has_expression:
			self.captured_type = self.visit(node.expression)


def _analyze_expr_type(expr_node, declarations=None):
	"""Build a minimal program that declares variables and evaluates expr_node,
	returning the type inferred by the semantic analyzer."""
	if declarations is None:
		declarations = []

	# Use a wide return type so we don't get type errors
	stmts = declarations + [ReturnStmt(expression=expr_node, has_expression=True, loc=loc())]
	func = FunctionDecl(
		name="test_fn",
		return_type=make_type("double"),
		params=[],
		body=CompoundStmt(statements=stmts, loc=loc()),
		loc=loc(),
	)
	prog = Program(declarations=[func], loc=loc())
	analyzer = _TypeCapture()
	analyzer.analyze(prog)
	return analyzer.captured_type


def _var_decl(name: str, ts: TypeSpec) -> VarDecl:
	return VarDecl(name=name, type_spec=ts, loc=loc())


def _id(name: str) -> Identifier:
	return Identifier(name=name, loc=loc())


def _binop(left, op, right) -> BinaryOp:
	return BinaryOp(left=left, op=op, right=right, loc=loc())


def _unary(op, operand) -> UnaryOp:
	return UnaryOp(op=op, operand=operand, loc=loc())


def _ternary(cond, true_expr, false_expr) -> TernaryExpr:
	return TernaryExpr(condition=cond, true_expr=true_expr, false_expr=false_expr, loc=loc())


# ---------------------------------------------------------------------------
# Unit tests for _type_rank
# ---------------------------------------------------------------------------

class TestTypeRank:
	def test_char_rank(self) -> None:
		assert _type_rank(make_type("char")) == 1

	def test_short_rank_via_width_modifier(self) -> None:
		assert _type_rank(make_type("int", width_modifier="short")) == 2

	def test_short_rank_via_base_type(self) -> None:
		assert _type_rank(make_type("short")) == 2

	def test_int_rank(self) -> None:
		assert _type_rank(make_type("int")) == 3

	def test_long_rank_via_width_modifier(self) -> None:
		assert _type_rank(make_type("int", width_modifier="long")) == 4

	def test_long_rank_via_base_type(self) -> None:
		assert _type_rank(make_type("long")) == 4

	def test_long_long_rank(self) -> None:
		assert _type_rank(make_type("int", width_modifier="long long")) == 5

	def test_float_rank(self) -> None:
		assert _type_rank(make_type("float")) == 6

	def test_double_rank(self) -> None:
		assert _type_rank(make_type("double")) == 7

	def test_unsigned_does_not_affect_rank(self) -> None:
		assert _type_rank(make_type("int", signedness="unsigned")) == 3
		assert _type_rank(make_type("char", signedness="unsigned")) == 1


# ---------------------------------------------------------------------------
# Unit tests for _integer_promote
# ---------------------------------------------------------------------------

class TestIntegerPromote:
	def test_char_promotes_to_int(self) -> None:
		result = _integer_promote(make_type("char"))
		assert result.base_type == "int"
		assert result.pointer_count == 0

	def test_short_promotes_to_int(self) -> None:
		result = _integer_promote(make_type("int", width_modifier="short"))
		assert result.base_type == "int"
		assert result.width_modifier is None

	def test_short_base_type_promotes_to_int(self) -> None:
		result = _integer_promote(make_type("short"))
		assert result.base_type == "int"

	def test_int_stays_int(self) -> None:
		ts = make_type("int")
		result = _integer_promote(ts)
		assert result.base_type == "int"

	def test_long_stays_long(self) -> None:
		ts = make_type("int", width_modifier="long")
		result = _integer_promote(ts)
		assert result.base_type == "int"
		assert result.width_modifier == "long"

	def test_float_stays_float(self) -> None:
		ts = make_type("float")
		result = _integer_promote(ts)
		assert result.base_type == "float"

	def test_double_stays_double(self) -> None:
		ts = make_type("double")
		result = _integer_promote(ts)
		assert result.base_type == "double"

	def test_pointer_not_promoted(self) -> None:
		ts = make_type("char", pointer_count=1)
		result = _integer_promote(ts)
		assert result.base_type == "char"
		assert result.pointer_count == 1


# ---------------------------------------------------------------------------
# Unit tests for _typespec_from_rank
# ---------------------------------------------------------------------------

class TestTypespecFromRank:
	def test_rank_1_gives_int(self) -> None:
		result = _typespec_from_rank(1, False)
		assert result.base_type == "int"

	def test_rank_3_gives_int(self) -> None:
		result = _typespec_from_rank(3, False)
		assert result.base_type == "int"

	def test_rank_4_gives_long(self) -> None:
		result = _typespec_from_rank(4, False)
		assert result.width_modifier == "long"

	def test_rank_5_gives_long_long(self) -> None:
		result = _typespec_from_rank(5, False)
		assert result.width_modifier == "long long"

	def test_rank_6_gives_float(self) -> None:
		result = _typespec_from_rank(6, False)
		assert result.base_type == "float"

	def test_rank_7_gives_double(self) -> None:
		result = _typespec_from_rank(7, False)
		assert result.base_type == "double"

	def test_unsigned_int(self) -> None:
		result = _typespec_from_rank(3, True)
		assert result.base_type == "int"
		assert result.signedness == "unsigned"

	def test_unsigned_long(self) -> None:
		result = _typespec_from_rank(4, True)
		assert result.signedness == "unsigned"
		assert result.width_modifier == "long"


# ---------------------------------------------------------------------------
# Unit tests for _usual_arithmetic_conversions
# ---------------------------------------------------------------------------

class TestUsualArithmeticConversions:
	# Float / double dominance
	def test_int_plus_double_gives_double(self) -> None:
		result = _usual_arithmetic_conversions(make_type("int"), make_type("double"))
		assert result.base_type == "double"

	def test_double_plus_int_gives_double(self) -> None:
		result = _usual_arithmetic_conversions(make_type("double"), make_type("int"))
		assert result.base_type == "double"

	def test_float_plus_double_gives_double(self) -> None:
		result = _usual_arithmetic_conversions(make_type("float"), make_type("double"))
		assert result.base_type == "double"

	def test_int_plus_float_gives_float(self) -> None:
		result = _usual_arithmetic_conversions(make_type("int"), make_type("float"))
		assert result.base_type == "float"

	def test_float_plus_int_gives_float(self) -> None:
		result = _usual_arithmetic_conversions(make_type("float"), make_type("int"))
		assert result.base_type == "float"

	def test_char_plus_float_gives_float(self) -> None:
		result = _usual_arithmetic_conversions(make_type("char"), make_type("float"))
		assert result.base_type == "float"

	# Integer promotion in conversions
	def test_char_plus_char_gives_int(self) -> None:
		result = _usual_arithmetic_conversions(make_type("char"), make_type("char"))
		assert result.base_type == "int"

	def test_short_plus_short_gives_int(self) -> None:
		result = _usual_arithmetic_conversions(
			make_type("int", width_modifier="short"),
			make_type("int", width_modifier="short"),
		)
		assert result.base_type == "int"
		assert result.width_modifier is None

	def test_char_plus_short_gives_int(self) -> None:
		result = _usual_arithmetic_conversions(
			make_type("char"), make_type("int", width_modifier="short")
		)
		assert result.base_type == "int"

	def test_char_plus_int_gives_int(self) -> None:
		result = _usual_arithmetic_conversions(make_type("char"), make_type("int"))
		assert result.base_type == "int"

	def test_int_plus_int_gives_int(self) -> None:
		result = _usual_arithmetic_conversions(make_type("int"), make_type("int"))
		assert result.base_type == "int"

	# Long promotion
	def test_int_plus_long_gives_long(self) -> None:
		result = _usual_arithmetic_conversions(
			make_type("int"), make_type("int", width_modifier="long")
		)
		assert result.base_type == "int"
		assert result.width_modifier == "long"

	def test_long_plus_int_gives_long(self) -> None:
		result = _usual_arithmetic_conversions(
			make_type("int", width_modifier="long"), make_type("int")
		)
		assert result.width_modifier == "long"

	def test_char_plus_long_gives_long(self) -> None:
		result = _usual_arithmetic_conversions(
			make_type("char"), make_type("int", width_modifier="long")
		)
		assert result.width_modifier == "long"

	def test_long_plus_long_long_gives_long_long(self) -> None:
		result = _usual_arithmetic_conversions(
			make_type("int", width_modifier="long"),
			make_type("int", width_modifier="long long"),
		)
		assert result.width_modifier == "long long"

	# Signed / unsigned interactions
	def test_signed_int_plus_unsigned_int_gives_unsigned_int(self) -> None:
		result = _usual_arithmetic_conversions(
			make_type("int"), make_type("int", signedness="unsigned")
		)
		assert result.signedness == "unsigned"
		assert result.base_type == "int"

	def test_unsigned_int_plus_signed_int_gives_unsigned_int(self) -> None:
		result = _usual_arithmetic_conversions(
			make_type("int", signedness="unsigned"), make_type("int")
		)
		assert result.signedness == "unsigned"

	def test_unsigned_int_plus_signed_long_gives_signed_long(self) -> None:
		# Signed long can represent all unsigned int values, so signed long wins
		result = _usual_arithmetic_conversions(
			make_type("int", signedness="unsigned"),
			make_type("int", width_modifier="long"),
		)
		assert result.width_modifier == "long"
		assert result.signedness is None

	def test_unsigned_long_plus_signed_int_gives_unsigned_long(self) -> None:
		result = _usual_arithmetic_conversions(
			make_type("int", width_modifier="long", signedness="unsigned"),
			make_type("int"),
		)
		assert result.signedness == "unsigned"
		assert result.width_modifier == "long"


# ---------------------------------------------------------------------------
# Integration: binary op type inference via semantic analyzer
# ---------------------------------------------------------------------------

class TestBinaryOpPromotion:
	def test_char_plus_char_returns_int(self) -> None:
		decls = [
			_var_decl("a", make_type("char")),
			_var_decl("b", make_type("char")),
		]
		expr = _binop(_id("a"), "+", _id("b"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "int"

	def test_short_times_short_returns_int(self) -> None:
		decls = [
			_var_decl("a", make_type("int", width_modifier="short")),
			_var_decl("b", make_type("int", width_modifier="short")),
		]
		expr = _binop(_id("a"), "*", _id("b"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "int"
		assert result.width_modifier is None

	def test_int_plus_long_returns_long(self) -> None:
		decls = [
			_var_decl("a", make_type("int")),
			_var_decl("b", make_type("int", width_modifier="long")),
		]
		expr = _binop(_id("a"), "+", _id("b"))
		result = _analyze_expr_type(expr, decls)
		assert result.width_modifier == "long"

	def test_int_plus_float_returns_float(self) -> None:
		decls = [
			_var_decl("a", make_type("int")),
			_var_decl("b", make_type("float")),
		]
		expr = _binop(_id("a"), "+", _id("b"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "float"

	def test_float_plus_double_returns_double(self) -> None:
		decls = [
			_var_decl("a", make_type("float")),
			_var_decl("b", make_type("double")),
		]
		expr = _binop(_id("a"), "+", _id("b"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "double"

	def test_char_minus_int_returns_int(self) -> None:
		decls = [
			_var_decl("a", make_type("char")),
			_var_decl("b", make_type("int")),
		]
		expr = _binop(_id("a"), "-", _id("b"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "int"

	def test_int_div_double_returns_double(self) -> None:
		decls = [
			_var_decl("a", make_type("int")),
			_var_decl("b", make_type("double")),
		]
		expr = _binop(_id("a"), "/", _id("b"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "double"

	def test_char_mod_char_returns_int(self) -> None:
		decls = [
			_var_decl("a", make_type("char")),
			_var_decl("b", make_type("char")),
		]
		expr = _binop(_id("a"), "%", _id("b"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "int"

	def test_bitwise_and_char_int_returns_int(self) -> None:
		decls = [
			_var_decl("a", make_type("char")),
			_var_decl("b", make_type("int")),
		]
		expr = _binop(_id("a"), "&", _id("b"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "int"

	def test_bitwise_or_short_long_returns_long(self) -> None:
		decls = [
			_var_decl("a", make_type("int", width_modifier="short")),
			_var_decl("b", make_type("int", width_modifier="long")),
		]
		expr = _binop(_id("a"), "|", _id("b"))
		result = _analyze_expr_type(expr, decls)
		assert result.width_modifier == "long"

	def test_shift_left_char_returns_int(self) -> None:
		decls = [
			_var_decl("a", make_type("char")),
			_var_decl("b", make_type("int")),
		]
		expr = _binop(_id("a"), "<<", _id("b"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "int"

	def test_shift_right_long_returns_long(self) -> None:
		decls = [
			_var_decl("a", make_type("int", width_modifier="long")),
			_var_decl("b", make_type("int")),
		]
		expr = _binop(_id("a"), ">>", _id("b"))
		result = _analyze_expr_type(expr, decls)
		assert result.width_modifier == "long"

	def test_comparison_returns_int(self) -> None:
		decls = [
			_var_decl("a", make_type("double")),
			_var_decl("b", make_type("float")),
		]
		expr = _binop(_id("a"), "<", _id("b"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "int"

	def test_logical_and_returns_int(self) -> None:
		decls = [
			_var_decl("a", make_type("long")),
			_var_decl("b", make_type("double")),
		]
		expr = _binop(_id("a"), "&&", _id("b"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "int"


# ---------------------------------------------------------------------------
# Integration: unary op type inference
# ---------------------------------------------------------------------------

class TestUnaryOpPromotion:
	def test_negate_char_returns_int(self) -> None:
		decls = [_var_decl("c", make_type("char"))]
		expr = _unary("-", _id("c"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "int"

	def test_negate_short_returns_int(self) -> None:
		decls = [_var_decl("s", make_type("int", width_modifier="short"))]
		expr = _unary("-", _id("s"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "int"
		assert result.width_modifier is None

	def test_negate_int_returns_int(self) -> None:
		decls = [_var_decl("i", make_type("int"))]
		expr = _unary("-", _id("i"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "int"

	def test_negate_long_returns_long(self) -> None:
		decls = [_var_decl("l", make_type("int", width_modifier="long"))]
		expr = _unary("-", _id("l"))
		result = _analyze_expr_type(expr, decls)
		assert result.width_modifier == "long"

	def test_bitwise_not_char_returns_int(self) -> None:
		decls = [_var_decl("c", make_type("char"))]
		expr = _unary("~", _id("c"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "int"

	def test_logical_not_returns_int(self) -> None:
		decls = [_var_decl("d", make_type("double"))]
		expr = _unary("!", _id("d"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "int"

	def test_negate_float_stays_float(self) -> None:
		decls = [_var_decl("f", make_type("float"))]
		expr = _unary("-", _id("f"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "float"


# ---------------------------------------------------------------------------
# Integration: ternary expression type inference
# ---------------------------------------------------------------------------

class TestTernaryPromotion:
	def test_char_int_ternary_returns_int(self) -> None:
		decls = [
			_var_decl("c", make_type("char")),
			_var_decl("i", make_type("int")),
			_var_decl("cond", make_type("int")),
		]
		expr = _ternary(_id("cond"), _id("c"), _id("i"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "int"

	def test_int_double_ternary_returns_double(self) -> None:
		decls = [
			_var_decl("i", make_type("int")),
			_var_decl("d", make_type("double")),
			_var_decl("cond", make_type("int")),
		]
		expr = _ternary(_id("cond"), _id("i"), _id("d"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "double"

	def test_float_double_ternary_returns_double(self) -> None:
		decls = [
			_var_decl("f", make_type("float")),
			_var_decl("d", make_type("double")),
			_var_decl("cond", make_type("int")),
		]
		expr = _ternary(_id("cond"), _id("f"), _id("d"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "double"

	def test_short_long_ternary_returns_long(self) -> None:
		decls = [
			_var_decl("s", make_type("int", width_modifier="short")),
			_var_decl("l", make_type("int", width_modifier="long")),
			_var_decl("cond", make_type("int")),
		]
		expr = _ternary(_id("cond"), _id("s"), _id("l"))
		result = _analyze_expr_type(expr, decls)
		assert result.width_modifier == "long"

	def test_char_char_ternary_returns_int(self) -> None:
		decls = [
			_var_decl("a", make_type("char")),
			_var_decl("b", make_type("char")),
			_var_decl("cond", make_type("int")),
		]
		expr = _ternary(_id("cond"), _id("a"), _id("b"))
		result = _analyze_expr_type(expr, decls)
		assert result.base_type == "int"


# ---------------------------------------------------------------------------
# Literal type inference in expressions
# ---------------------------------------------------------------------------

class TestLiteralPromotionInExpr:
	def test_char_literal_plus_int_literal(self) -> None:
		expr = _binop(CharLiteral(value=ord("A"), loc=loc()), "+", IntLiteral(value=1, loc=loc()))
		result = _analyze_expr_type(expr)
		assert result.base_type == "int"

	def test_int_literal_times_float_literal(self) -> None:
		expr = _binop(IntLiteral(value=2, loc=loc()), "*", FloatLiteral(value=3.14, loc=loc()))
		result = _analyze_expr_type(expr)
		assert result.base_type == "double"  # unadorned float literal is double

	def test_negate_char_literal(self) -> None:
		expr = _unary("-", CharLiteral(value=ord("x"), loc=loc()))
		result = _analyze_expr_type(expr)
		assert result.base_type == "int"
