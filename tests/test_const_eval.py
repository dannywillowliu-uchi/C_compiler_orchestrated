"""Tests for compile-time constant expression evaluator."""

from compiler.ast_nodes import (
	BinaryOp,
	CastExpr,
	CharLiteral,
	CommaExpr,
	Identifier,
	IntLiteral,
	SizeofExpr,
	SourceLocation,
	TernaryExpr,
	TypeSpec,
	UnaryOp,
)
from compiler.const_eval import ConstExprEvaluator, _sign_extend
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer, SemanticError


def loc() -> SourceLocation:
	return SourceLocation(0, 0)


def make_int(val: int) -> IntLiteral:
	return IntLiteral(loc=loc(), value=val)


def make_char(val: str) -> CharLiteral:
	return CharLiteral(loc=loc(), value=val)


def make_id(name: str) -> Identifier:
	return Identifier(loc=loc(), name=name)


def make_binop(left, op: str, right) -> BinaryOp:
	return BinaryOp(loc=loc(), left=left, op=op, right=right)


def make_unary(op: str, operand, prefix: bool = True) -> UnaryOp:
	return UnaryOp(loc=loc(), op=op, operand=operand, prefix=prefix)


def make_ternary(cond, true_expr, false_expr) -> TernaryExpr:
	return TernaryExpr(loc=loc(), condition=cond, true_expr=true_expr, false_expr=false_expr)


def make_sizeof_type(base: str, pointer_count: int = 0, **kwargs) -> SizeofExpr:
	ts = TypeSpec(loc=loc(), base_type=base, pointer_count=pointer_count, **kwargs)
	return SizeofExpr(loc=loc(), type_operand=ts)


def make_cast(base: str, operand, pointer_count: int = 0, **kwargs) -> CastExpr:
	ts = TypeSpec(loc=loc(), base_type=base, pointer_count=pointer_count, **kwargs)
	return CastExpr(loc=loc(), target_type=ts, operand=operand)


def parse(source: str):
	return Parser.from_source(source).parse()


def analyze(source: str) -> list[SemanticError]:
	program = parse(source)
	analyzer = SemanticAnalyzer()
	try:
		analyzer.analyze(program)
	except SemanticError:
		pass
	return analyzer.errors


# --- Direct evaluator tests ---


class TestIntLiterals:
	def test_zero(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_int(0)) == 0

	def test_positive(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_int(42)) == 42

	def test_negative_via_unary(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_unary("-", make_int(5))) == -5

	def test_large_value(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_int(0xFFFFFFFF)) == 0xFFFFFFFF


class TestCharLiterals:
	def test_simple_char(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_char("A")) == 65

	def test_newline_escape(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_char("\\n")) == 10

	def test_null_char(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_char("\\0")) == 0

	def test_tab_escape(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_char("\\t")) == 9


class TestEnumConstants:
	def test_known_enum(self):
		ev = ConstExprEvaluator(enum_constants={"FOO": 10, "BAR": 20})
		assert ev.evaluate(make_id("FOO")) == 10
		assert ev.evaluate(make_id("BAR")) == 20

	def test_unknown_identifier(self):
		ev = ConstExprEvaluator(enum_constants={"FOO": 10})
		assert ev.evaluate(make_id("UNKNOWN")) is None

	def test_enum_in_expression(self):
		ev = ConstExprEvaluator(enum_constants={"X": 3})
		expr = make_binop(make_id("X"), "+", make_int(7))
		assert ev.evaluate(expr) == 10


class TestUnaryOps:
	def test_negate(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_unary("-", make_int(42))) == -42

	def test_positive(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_unary("+", make_int(42))) == 42

	def test_bitwise_not(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_unary("~", make_int(0))) == -1

	def test_logical_not_true(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_unary("!", make_int(5))) == 0

	def test_logical_not_false(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_unary("!", make_int(0))) == 1

	def test_postfix_returns_none(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_unary("++", make_int(1), prefix=False)) is None

	def test_double_negation(self):
		ev = ConstExprEvaluator()
		expr = make_unary("-", make_unary("-", make_int(7)))
		assert ev.evaluate(expr) == 7


class TestArithmeticOps:
	def test_add(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(3), "+", make_int(4))) == 7

	def test_subtract(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(10), "-", make_int(3))) == 7

	def test_multiply(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(6), "*", make_int(7))) == 42

	def test_divide(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(10), "/", make_int(3))) == 3

	def test_divide_negative_truncates_toward_zero(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_unary("-", make_int(7)), "/", make_int(2))) == -3

	def test_divide_by_zero(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(5), "/", make_int(0))) is None

	def test_modulo(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(10), "%", make_int(3))) == 1

	def test_modulo_negative(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_unary("-", make_int(7)), "%", make_int(3))) == -1

	def test_modulo_by_zero(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(5), "%", make_int(0))) is None


class TestBitwiseOps:
	def test_and(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(0xFF), "&", make_int(0x0F))) == 0x0F

	def test_or(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(0xF0), "|", make_int(0x0F))) == 0xFF

	def test_xor(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(0xFF), "^", make_int(0x0F))) == 0xF0

	def test_left_shift(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(1), "<<", make_int(8))) == 256

	def test_right_shift(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(256), ">>", make_int(4))) == 16

	def test_negative_shift_returns_none(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(1), "<<", make_unary("-", make_int(1)))) is None


class TestComparisonOps:
	def test_equal_true(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(5), "==", make_int(5))) == 1

	def test_equal_false(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(5), "==", make_int(3))) == 0

	def test_not_equal(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(5), "!=", make_int(3))) == 1

	def test_less_than(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(3), "<", make_int(5))) == 1

	def test_greater_than(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(5), ">", make_int(3))) == 1

	def test_less_equal(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(5), "<=", make_int(5))) == 1

	def test_greater_equal(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(5), ">=", make_int(6))) == 0


class TestLogicalOps:
	def test_and_true(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(1), "&&", make_int(2))) == 1

	def test_and_false(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(0), "&&", make_int(2))) == 0

	def test_or_true(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(0), "||", make_int(1))) == 1

	def test_or_false(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_binop(make_int(0), "||", make_int(0))) == 0

	def test_and_short_circuit(self):
		"""&& with false left should return 0 even if right is non-evaluable."""
		ev = ConstExprEvaluator()
		expr = make_binop(make_int(0), "&&", make_id("unknown_var"))
		assert ev.evaluate(expr) == 0

	def test_or_short_circuit(self):
		"""|| with true left should return 1 even if right is non-evaluable."""
		ev = ConstExprEvaluator()
		expr = make_binop(make_int(1), "||", make_id("unknown_var"))
		assert ev.evaluate(expr) == 1


class TestTernary:
	def test_true_branch(self):
		ev = ConstExprEvaluator()
		expr = make_ternary(make_int(1), make_int(10), make_int(20))
		assert ev.evaluate(expr) == 10

	def test_false_branch(self):
		ev = ConstExprEvaluator()
		expr = make_ternary(make_int(0), make_int(10), make_int(20))
		assert ev.evaluate(expr) == 20

	def test_nested_ternary(self):
		ev = ConstExprEvaluator()
		inner = make_ternary(make_int(1), make_int(5), make_int(6))
		outer = make_ternary(make_int(0), make_int(99), inner)
		assert ev.evaluate(outer) == 5

	def test_non_constant_condition(self):
		ev = ConstExprEvaluator()
		expr = make_ternary(make_id("x"), make_int(1), make_int(2))
		assert ev.evaluate(expr) is None


class TestSizeof:
	def test_sizeof_int(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_sizeof_type("int")) == 4

	def test_sizeof_char(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_sizeof_type("char")) == 1

	def test_sizeof_long(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_sizeof_type("long")) == 8

	def test_sizeof_double(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_sizeof_type("double")) == 8

	def test_sizeof_float(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_sizeof_type("float")) == 4

	def test_sizeof_pointer(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_sizeof_type("int", pointer_count=1)) == 8

	def test_sizeof_void_pointer(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_sizeof_type("void", pointer_count=1)) == 8

	def test_sizeof_short_modifier(self):
		ev = ConstExprEvaluator()
		expr = SizeofExpr(
			loc=loc(),
			type_operand=TypeSpec(loc=loc(), base_type="int", width_modifier="short"),
		)
		assert ev.evaluate(expr) == 2

	def test_sizeof_bool(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_sizeof_type("_Bool")) == 1

	def test_sizeof_in_expression(self):
		ev = ConstExprEvaluator()
		expr = make_binop(make_sizeof_type("int"), "*", make_int(10))
		assert ev.evaluate(expr) == 40

	def test_sizeof_expr_int_literal(self):
		ev = ConstExprEvaluator()
		expr = SizeofExpr(loc=loc(), operand=make_int(42))
		assert ev.evaluate(expr) == 4

	def test_sizeof_expr_char_literal(self):
		ev = ConstExprEvaluator()
		expr = SizeofExpr(loc=loc(), operand=make_char("x"))
		assert ev.evaluate(expr) == 1


class TestCast:
	def test_cast_to_char(self):
		ev = ConstExprEvaluator()
		expr = make_cast("char", make_int(300))
		# 300 & 0xFF = 44, sign-extended: 44 (positive, no sign extension)
		assert ev.evaluate(expr) == 44

	def test_cast_to_unsigned_char(self):
		ev = ConstExprEvaluator()
		expr = make_cast("char", make_int(300), signedness="unsigned")
		assert ev.evaluate(expr) == 44

	def test_cast_to_char_negative(self):
		ev = ConstExprEvaluator()
		expr = make_cast("char", make_int(200))
		# 200 & 0xFF = 200, sign-extended from 8 bits: -56
		assert ev.evaluate(expr) == -56

	def test_cast_to_int(self):
		ev = ConstExprEvaluator()
		expr = make_cast("int", make_int(42))
		assert ev.evaluate(expr) == 42

	def test_cast_to_unsigned_int(self):
		ev = ConstExprEvaluator()
		expr = make_cast("int", make_unary("-", make_int(1)), signedness="unsigned")
		assert ev.evaluate(expr) == 0xFFFFFFFF

	def test_cast_to_bool(self):
		ev = ConstExprEvaluator()
		assert ev.evaluate(make_cast("_Bool", make_int(42))) == 1
		assert ev.evaluate(make_cast("_Bool", make_int(0))) == 0

	def test_cast_to_pointer(self):
		ev = ConstExprEvaluator()
		expr = make_cast("void", make_int(0), pointer_count=1)
		assert ev.evaluate(expr) == 0

	def test_cast_non_constant(self):
		ev = ConstExprEvaluator()
		expr = make_cast("int", make_id("x"))
		assert ev.evaluate(expr) is None


class TestCommaExpr:
	def test_comma_returns_right(self):
		ev = ConstExprEvaluator()
		expr = CommaExpr(loc=loc(), left=make_int(1), right=make_int(2))
		assert ev.evaluate(expr) == 2


class TestComplexExpressions:
	def test_nested_arithmetic(self):
		ev = ConstExprEvaluator()
		# (3 + 4) * 2 - 1 = 13
		expr = make_binop(
			make_binop(
				make_binop(make_int(3), "+", make_int(4)),
				"*",
				make_int(2),
			),
			"-",
			make_int(1),
		)
		assert ev.evaluate(expr) == 13

	def test_sizeof_plus_enum(self):
		ev = ConstExprEvaluator(enum_constants={"OFFSET": 16})
		expr = make_binop(make_sizeof_type("int"), "+", make_id("OFFSET"))
		assert ev.evaluate(expr) == 20

	def test_bitwise_flags(self):
		ev = ConstExprEvaluator()
		# (1 << 0) | (1 << 2) | (1 << 4) = 1 | 4 | 16 = 21
		flag0 = make_binop(make_int(1), "<<", make_int(0))
		flag2 = make_binop(make_int(1), "<<", make_int(2))
		flag4 = make_binop(make_int(1), "<<", make_int(4))
		expr = make_binop(make_binop(flag0, "|", flag2), "|", flag4)
		assert ev.evaluate(expr) == 21

	def test_conditional_with_comparison(self):
		ev = ConstExprEvaluator()
		# sizeof(int) == 4 ? 32 : 64
		cond = make_binop(make_sizeof_type("int"), "==", make_int(4))
		expr = make_ternary(cond, make_int(32), make_int(64))
		assert ev.evaluate(expr) == 32

	def test_propagates_none(self):
		ev = ConstExprEvaluator()
		expr = make_binop(make_id("unknown"), "+", make_int(1))
		assert ev.evaluate(expr) is None


class TestSignExtend:
	def test_no_extension_needed(self):
		assert _sign_extend(42, 8) == 42

	def test_sign_extend_8bit(self):
		assert _sign_extend(0xFF, 8) == -1

	def test_sign_extend_16bit(self):
		assert _sign_extend(0xFFFF, 16) == -1

	def test_sign_extend_32bit(self):
		assert _sign_extend(0x80000000, 32) == -2147483648


# --- Integration tests via semantic analysis ---


class TestStaticAssertIntegration:
	def test_simple_pass(self):
		errors = analyze("_Static_assert(1, \"ok\");")
		assert len(errors) == 0

	def test_simple_fail(self):
		errors = analyze("_Static_assert(0, \"fail\");")
		assert len(errors) == 1
		assert "static assertion failed" in str(errors[0])

	def test_sizeof_condition(self):
		errors = analyze("_Static_assert(sizeof(int) == 4, \"int must be 4 bytes\");")
		assert len(errors) == 0

	def test_sizeof_condition_fail(self):
		errors = analyze("_Static_assert(sizeof(int) == 8, \"int must be 8 bytes\");")
		assert len(errors) == 1

	def test_arithmetic_expression(self):
		errors = analyze("_Static_assert(2 + 3 == 5, \"math works\");")
		assert len(errors) == 0

	def test_bitwise_expression(self):
		errors = analyze("_Static_assert((0xFF & 0x0F) == 0x0F, \"bitwise and\");")
		assert len(errors) == 0

	def test_ternary_in_static_assert(self):
		errors = analyze("_Static_assert(1 ? 1 : 0, \"ternary true\");")
		assert len(errors) == 0

	def test_logical_operators(self):
		errors = analyze("_Static_assert(1 && 1, \"logical and\");")
		assert len(errors) == 0

	def test_negation(self):
		errors = analyze("_Static_assert(!0, \"not zero\");")
		assert len(errors) == 0

	def test_enum_in_static_assert(self):
		src = """
		enum vals { A = 5, B = 10 };
		_Static_assert(A + B == 15, "enum sum");
		"""
		errors = analyze(src)
		assert len(errors) == 0

	def test_cast_in_static_assert(self):
		errors = analyze("_Static_assert((int)1, \"cast\");")
		assert len(errors) == 0

	def test_char_literal_in_static_assert(self):
		errors = analyze("_Static_assert('A' == 65, \"char value\");")
		assert len(errors) == 0

	def test_complex_expression(self):
		errors = analyze("_Static_assert(sizeof(int) * 2 + 1 > sizeof(char), \"complex\");")
		assert len(errors) == 0


class TestSwitchCaseIntegration:
	def test_constant_case_values(self):
		src = """
		int main() {
			int x = 1;
			switch (x) {
				case 1: return 1;
				case 2: return 2;
				default: return 0;
			}
		}
		"""
		errors = analyze(src)
		assert len(errors) == 0

	def test_duplicate_case_detected(self):
		src = """
		int main() {
			int x = 1;
			switch (x) {
				case 1: return 1;
				case 1: return 2;
			}
		}
		"""
		errors = analyze(src)
		assert len(errors) == 1
		assert "duplicate case" in str(errors[0])

	def test_enum_case_values(self):
		src = """
		enum Color { RED, GREEN, BLUE };
		int main() {
			int c = 0;
			switch (c) {
				case RED: return 0;
				case GREEN: return 1;
				case BLUE: return 2;
			}
		}
		"""
		errors = analyze(src)
		assert len(errors) == 0

	def test_enum_duplicate_case(self):
		src = """
		enum dup { A = 1, B = 1 };
		int main() {
			int x = 0;
			switch (x) {
				case A: return 1;
				case B: return 2;
			}
		}
		"""
		errors = analyze(src)
		assert len(errors) == 1
		assert "duplicate case" in str(errors[0])

	def test_expression_case_values(self):
		src = """
		int main() {
			int x = 0;
			switch (x) {
				case 1 + 1: return 2;
				case 3 * 2: return 6;
				default: return 0;
			}
		}
		"""
		errors = analyze(src)
		assert len(errors) == 0

	def test_char_case_value(self):
		src = """
		int main() {
			int x = 65;
			switch (x) {
				case 'A': return 1;
				case 'B': return 2;
				default: return 0;
			}
		}
		"""
		errors = analyze(src)
		assert len(errors) == 0


class TestEnumConstEval:
	def test_enum_with_expressions(self):
		src = """
		enum flags { A = 1 << 0, B = 1 << 1, C = 1 << 2 };
		_Static_assert(A == 1, "A");
		_Static_assert(B == 2, "B");
		_Static_assert(C == 4, "C");
		"""
		errors = analyze(src)
		assert len(errors) == 0

	def test_enum_with_arithmetic(self):
		src = """
		enum seq { BASE = 100, NEXT = BASE + 1, LAST = NEXT + 1 };
		_Static_assert(BASE == 100, "base");
		_Static_assert(NEXT == 101, "next");
		_Static_assert(LAST == 102, "last");
		"""
		errors = analyze(src)
		assert len(errors) == 0

	def test_enum_auto_increment(self):
		src = """
		enum inc { X = 5, Y, Z };
		_Static_assert(Y == 6, "Y");
		_Static_assert(Z == 7, "Z");
		"""
		errors = analyze(src)
		assert len(errors) == 0

	def test_enum_negative_value(self):
		src = """
		enum neg { NEG = -1, ZERO, ONE };
		_Static_assert(NEG == -1, "neg");
		_Static_assert(ZERO == 0, "zero");
		_Static_assert(ONE == 1, "one");
		"""
		errors = analyze(src)
		assert len(errors) == 0
