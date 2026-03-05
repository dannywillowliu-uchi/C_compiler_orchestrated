"""Edge-case tests for enum, sizeof, and cast corner cases."""

from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer
from compiler.ir_gen import IRGenerator
from compiler.ir import IRConst


def _parse(source: str):
	return Parser.from_source(source).parse()


def _analyze(source: str):
	prog = _parse(source)
	analyzer = SemanticAnalyzer()
	errors = analyzer.analyze(prog)
	return prog, analyzer, errors


def _analyze_ok(source: str):
	prog, analyzer, errors = _analyze(source)
	assert not errors, f"Unexpected semantic errors: {errors}"
	return prog, analyzer


def _ir_gen(source: str):
	prog, _ = _analyze_ok(source)
	gen = IRGenerator()
	return gen.generate(prog)


def _ir_return_value(source: str):
	"""Get the return value from the last instruction of first function."""
	ir = _ir_gen(source)
	ret = ir.functions[0].body[-1]
	assert isinstance(ret.value, IRConst), f"Expected IRConst, got {type(ret.value)}"
	return ret.value.value


# ---------------------------------------------------------------------------
# Enum with positive values and auto-increment
# ---------------------------------------------------------------------------


class TestEnumValues:
	def test_enum_default_values(self):
		src = """
		enum Color { RED, GREEN, BLUE };
		int foo() { return RED; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert analyzer._enum_constants["RED"] == 0
		assert analyzer._enum_constants["GREEN"] == 1
		assert analyzer._enum_constants["BLUE"] == 2

	def test_enum_explicit_start(self):
		src = """
		enum Vals { A = 5, B, C };
		int foo() { return C; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert analyzer._enum_constants["A"] == 5
		assert analyzer._enum_constants["B"] == 6
		assert analyzer._enum_constants["C"] == 7

	def test_enum_large_positive(self):
		src = """
		enum Big { MILLION = 1000000, NEXT };
		int foo() { return NEXT; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert analyzer._enum_constants["MILLION"] == 1000000
		assert analyzer._enum_constants["NEXT"] == 1000001

	def test_enum_max_int(self):
		src = """
		enum Max { MAXVAL = 2147483647 };
		int foo() { return MAXVAL; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert analyzer._enum_constants["MAXVAL"] == 2147483647

	def test_enum_value_gap(self):
		src = """
		enum Sparse { A = 0, B = 100, C = 1000 };
		int foo() { return C; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert analyzer._enum_constants["A"] == 0
		assert analyzer._enum_constants["B"] == 100
		assert analyzer._enum_constants["C"] == 1000

	def test_enum_decreasing_values(self):
		src = """
		enum Dec { X = 10, Y = 5, Z = 1 };
		int foo() { return Z; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert analyzer._enum_constants["X"] == 10
		assert analyzer._enum_constants["Y"] == 5
		assert analyzer._enum_constants["Z"] == 1

	def test_enum_mixed_explicit_auto(self):
		src = """
		enum Mix { A, B, C = 10, D, E = 0, F };
		int foo() { return F; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert analyzer._enum_constants["A"] == 0
		assert analyzer._enum_constants["B"] == 1
		assert analyzer._enum_constants["C"] == 10
		assert analyzer._enum_constants["D"] == 11
		assert analyzer._enum_constants["E"] == 0
		assert analyzer._enum_constants["F"] == 1


# ---------------------------------------------------------------------------
# Enum duplicate values and edge cases
# ---------------------------------------------------------------------------


class TestEnumDuplicatesAndEdge:
	def test_enum_same_explicit_value(self):
		"""Multiple constants with the same explicit value (valid in C)."""
		src = """
		enum Dup { A = 5, B = 5, C = 5 };
		int foo() { return A + B + C; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert analyzer._enum_constants["A"] == 5
		assert analyzer._enum_constants["B"] == 5
		assert analyzer._enum_constants["C"] == 5

	def test_enum_single_constant(self):
		src = """
		enum Single { ONLY };
		int foo() { return ONLY; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert analyzer._enum_constants["ONLY"] == 0

	def test_enum_used_as_array_index(self):
		src = """
		enum Idx { FIRST = 0, SECOND = 1, THIRD = 2 };
		int foo() {
			int arr[3];
			arr[FIRST] = 10;
			arr[SECOND] = 20;
			arr[THIRD] = 30;
			return arr[SECOND];
		}
		"""
		_analyze_ok(src)

	def test_enum_in_ternary(self):
		src = """
		enum Bool { FALSE = 0, TRUE = 1 };
		int foo(int x) { return x ? TRUE : FALSE; }
		"""
		_analyze_ok(src)

	def test_multiple_enums(self):
		src = """
		enum Color { RED, GREEN, BLUE };
		enum Size { SMALL, MEDIUM, LARGE };
		int foo() { return RED + SMALL; }
		"""
		prog, analyzer = _analyze_ok(src)
		assert analyzer._enum_constants["RED"] == 0
		assert analyzer._enum_constants["SMALL"] == 0

	def test_enum_constant_in_var_init(self):
		src = """
		enum Vals { X = 42 };
		int foo() { int v = X; return v; }
		"""
		_analyze_ok(src)

	def test_enum_in_expression(self):
		src = """
		enum Vals { A = 2, B = 3 };
		int foo() { int x = A + B; return x; }
		"""
		_analyze_ok(src)

	def test_enum_in_comparison(self):
		src = """
		enum Level { LOW = 1, HIGH = 10 };
		int foo() { int x = LOW < HIGH; return x; }
		"""
		_analyze_ok(src)

	def test_enum_in_while_condition(self):
		src = """
		enum Limits { LIMIT = 5 };
		int foo() {
			int i = 0;
			while (i < LIMIT) { i++; }
			return i;
		}
		"""
		_analyze_ok(src)

	def test_enum_in_for_loop(self):
		src = """
		enum Range { START = 0, END = 10 };
		int foo() {
			int sum = 0;
			int i;
			for (i = START; i < END; i++) { sum = sum + i; }
			return sum;
		}
		"""
		_analyze_ok(src)


# ---------------------------------------------------------------------------
# Enum negative values via IR (IR generator handles UnaryOp correctly)
# ---------------------------------------------------------------------------


class TestEnumNegativeIR:
	def test_enum_negative_value_ir(self):
		src = """
		enum Sign { NEG = -1, ZERO = 0, POS = 1 };
		int foo() { return NEG; }
		"""
		gen = IRGenerator()
		prog, _ = _analyze_ok(src)
		gen.generate(prog)
		assert gen._enum_constants["NEG"] == -1

	def test_enum_negative_auto_increment_ir(self):
		"""IR generator should handle negative with auto-increment: -5, -4, -3."""
		src = """
		enum Neg { A = -5, B, C };
		int foo() { return C; }
		"""
		gen = IRGenerator()
		prog, _ = _analyze_ok(src)
		gen.generate(prog)
		assert gen._enum_constants["A"] == -5
		assert gen._enum_constants["B"] == -4
		assert gen._enum_constants["C"] == -3

	def test_enum_large_negative_ir(self):
		src = """
		enum Big { X = -1000000 };
		int foo() { return X; }
		"""
		gen = IRGenerator()
		prog, _ = _analyze_ok(src)
		gen.generate(prog)
		assert gen._enum_constants["X"] == -1000000

	def test_enum_mixed_negative_positive_ir(self):
		src = """
		enum Mixed { A = -2, B, C, D = 5, E };
		int foo() { return E; }
		"""
		gen = IRGenerator()
		prog, _ = _analyze_ok(src)
		gen.generate(prog)
		assert gen._enum_constants["A"] == -2
		assert gen._enum_constants["B"] == -1
		assert gen._enum_constants["C"] == 0
		assert gen._enum_constants["D"] == 5
		assert gen._enum_constants["E"] == 6


# ---------------------------------------------------------------------------
# sizeof on primitive types (via IR)
# ---------------------------------------------------------------------------


class TestSizeofPrimitives:
	def test_sizeof_int(self):
		assert _ir_return_value("int foo() { return sizeof(int); }") == 4

	def test_sizeof_char(self):
		assert _ir_return_value("int foo() { return sizeof(char); }") == 1

	def test_sizeof_long(self):
		assert _ir_return_value("int foo() { return sizeof(long); }") == 8

	def test_sizeof_short(self):
		assert _ir_return_value("int foo() { return sizeof(short); }") == 2

	def test_sizeof_pointer(self):
		assert _ir_return_value("int foo() { return sizeof(int *); }") == 8

	def test_sizeof_double(self):
		assert _ir_return_value("int foo() { return sizeof(double); }") == 8

	def test_sizeof_float(self):
		assert _ir_return_value("int foo() { return sizeof(float); }") == 4

	def test_sizeof_long_long(self):
		assert _ir_return_value("int foo() { return sizeof(long long); }") == 8

	def test_sizeof_char_pointer(self):
		assert _ir_return_value("int foo() { return sizeof(char *); }") == 8


# ---------------------------------------------------------------------------
# sizeof on variables and arrays
# ---------------------------------------------------------------------------


class TestSizeofVariablesArrays:
	def test_sizeof_int_variable(self):
		assert _ir_return_value("int foo() { int x = 0; return sizeof(x); }") == 4

	def test_sizeof_char_variable(self):
		assert _ir_return_value("int foo() { char c = 'a'; return sizeof(c); }") == 1

	def test_sizeof_int_array(self):
		assert _ir_return_value("int foo() { int arr[10]; return sizeof(arr); }") == 40

	def test_sizeof_char_array(self):
		assert _ir_return_value("int foo() { char buf[64]; return sizeof(buf); }") == 64

	def test_sizeof_short_array(self):
		assert _ir_return_value("int foo() { short s[5]; return sizeof(s); }") == 10

	def test_sizeof_in_arithmetic(self):
		src = "int foo() { return sizeof(int) + sizeof(char); }"
		_analyze_ok(src)

	def test_sizeof_in_comparison(self):
		src = "int foo() { return sizeof(long) > sizeof(int); }"
		_analyze_ok(src)


# ---------------------------------------------------------------------------
# sizeof on bitfield structs
# ---------------------------------------------------------------------------


class TestSizeofBitfieldStruct:
	def test_sizeof_struct_with_bitfields(self):
		src = """
		struct Flags { int a : 1; int b : 1; int c : 1; };
		int foo() { return sizeof(struct Flags); }
		"""
		val = _ir_return_value(src)
		assert val > 0  # At least 1 byte for the storage unit

	def test_sizeof_struct_mixed_bitfield_normal(self):
		src = """
		struct Mixed { int x; int flags : 4; int y; };
		int foo() { return sizeof(struct Mixed); }
		"""
		val = _ir_return_value(src)
		assert val >= 12  # At least 3 ints worth


# ---------------------------------------------------------------------------
# sizeof on array-of-structs
# ---------------------------------------------------------------------------


class TestSizeofArrayOfStructs:
	def test_sizeof_struct_array(self):
		src = """
		struct Point { int x; int y; };
		int foo() {
			struct Point pts[5];
			return sizeof(pts);
		}
		"""
		val = _ir_return_value(src)
		# struct Point is 8 bytes (2 * 4), array of 5 = 40
		assert val == 40

	def test_sizeof_struct_array_three_members(self):
		src = """
		struct Vec3 { int x; int y; int z; };
		int foo() {
			struct Vec3 vecs[3];
			return sizeof(vecs);
		}
		"""
		val = _ir_return_value(src)
		# struct Vec3 is 12 bytes (3 * 4), array of 3 = 36
		assert val == 36

	def test_sizeof_struct_type_vs_array(self):
		"""sizeof(struct) vs sizeof(array of struct) should differ."""
		src = """
		struct Pair { int a; int b; };
		int foo() {
			struct Pair arr[4];
			int struct_size = sizeof(struct Pair);
			int array_size = sizeof(arr);
			return array_size - struct_size;
		}
		"""
		_analyze_ok(src)

	def test_sizeof_single_element_struct_array(self):
		src = """
		struct S { int val; };
		int foo() {
			struct S arr[1];
			return sizeof(arr);
		}
		"""
		val = _ir_return_value(src)
		assert val == 4


# ---------------------------------------------------------------------------
# Cast chains (int -> char -> int)
# ---------------------------------------------------------------------------


class TestCastChains:
	def test_int_to_char_to_int(self):
		src = """
		int foo() {
			int x = 65;
			char c = (char)x;
			int y = (int)c;
			return y;
		}
		"""
		_analyze_ok(src)

	def test_long_to_short_to_char(self):
		src = """
		int foo() {
			long x = 1000;
			short s = (short)x;
			char c = (char)s;
			return (int)c;
		}
		"""
		_analyze_ok(src)

	def test_char_to_long_to_short(self):
		src = """
		int foo() {
			char c = 'A';
			long l = (long)c;
			short s = (short)l;
			return (int)s;
		}
		"""
		_analyze_ok(src)

	def test_nested_cast_expression(self):
		"""Cast of a cast expression: (int)(char)(long)x."""
		src = """
		int foo() {
			long x = 42;
			int y = (int)(char)(long)x;
			return y;
		}
		"""
		_analyze_ok(src)

	def test_cast_chain_in_return(self):
		src = """
		int foo() {
			long x = 100;
			return (int)(short)(char)x;
		}
		"""
		_analyze_ok(src)

	def test_double_cast_same_type(self):
		src = """
		int foo() {
			int x = 5;
			int y = (int)(int)x;
			return y;
		}
		"""
		_analyze_ok(src)

	def test_cast_chain_with_arithmetic(self):
		src = """
		int foo() {
			int x = 300;
			char c = (char)x;
			int y = (int)c + 1;
			return y;
		}
		"""
		_analyze_ok(src)

	def test_triple_widening_cast(self):
		src = """
		int foo() {
			char c = 'Z';
			short s = (short)c;
			int i = (int)s;
			long l = (long)i;
			return (int)l;
		}
		"""
		_analyze_ok(src)


# ---------------------------------------------------------------------------
# Cast in ternary branches
# ---------------------------------------------------------------------------


class TestCastInTernary:
	def test_cast_in_true_branch(self):
		src = """
		int foo(int x) {
			return x ? (int)'A' : 0;
		}
		"""
		_analyze_ok(src)

	def test_cast_in_both_branches(self):
		src = """
		int foo(int x) {
			char a = 'X';
			char b = 'Y';
			return x ? (int)a : (int)b;
		}
		"""
		_analyze_ok(src)

	def test_cast_in_ternary_condition(self):
		src = """
		int foo() {
			char c = 1;
			return (int)c ? 100 : 200;
		}
		"""
		_analyze_ok(src)

	def test_cast_ternary_different_widths(self):
		src = """
		int foo(int x) {
			short s = 10;
			long l = 20;
			return x ? (int)s : (int)l;
		}
		"""
		_analyze_ok(src)

	def test_cast_ternary_nested(self):
		src = """
		int foo(int x, int y) {
			return x ? (int)(char)y : (int)(short)y;
		}
		"""
		_analyze_ok(src)

	def test_ternary_with_enum_cast(self):
		src = """
		enum Mode { OFF = 0, ON = 1 };
		int foo(int x) {
			return x ? (int)ON : (int)OFF;
		}
		"""
		_analyze_ok(src)


# ---------------------------------------------------------------------------
# Cast with type modifiers (unsigned short -> long, etc.)
# ---------------------------------------------------------------------------


class TestCastTypeModifiers:
	def test_cast_unsigned_short_to_long(self):
		src = """
		int foo() {
			unsigned short us = 50000;
			long l = (long)us;
			return (int)l;
		}
		"""
		_analyze_ok(src)

	def test_cast_long_to_unsigned_short(self):
		src = """
		int foo() {
			long l = 100000;
			unsigned short us = (unsigned short)l;
			return (int)us;
		}
		"""
		_analyze_ok(src)

	def test_cast_signed_to_unsigned(self):
		src = """
		int foo() {
			int x = -1;
			unsigned int u = (unsigned int)x;
			return (int)u;
		}
		"""
		_analyze_ok(src)

	def test_cast_unsigned_to_signed(self):
		src = """
		int foo() {
			unsigned int u = 42;
			int x = (int)u;
			return x;
		}
		"""
		_analyze_ok(src)

	def test_cast_short_to_long_long(self):
		src = """
		int foo() {
			short s = 100;
			long long ll = (long long)s;
			return (int)ll;
		}
		"""
		_analyze_ok(src)

	def test_cast_long_long_to_char(self):
		src = """
		int foo() {
			long long ll = 65;
			char c = (char)ll;
			return (int)c;
		}
		"""
		_analyze_ok(src)

	def test_cast_float_to_long(self):
		src = """
		int foo() {
			float f = 3.14;
			long l = (long)f;
			return (int)l;
		}
		"""
		_analyze_ok(src)

	def test_cast_double_to_short(self):
		src = """
		int foo() {
			double d = 99.9;
			short s = (short)d;
			return (int)s;
		}
		"""
		_analyze_ok(src)

	def test_cast_unsigned_char_to_int(self):
		src = """
		int foo() {
			unsigned char uc = 255;
			int x = (int)uc;
			return x;
		}
		"""
		_analyze_ok(src)

	def test_cast_int_to_unsigned_long(self):
		src = """
		int foo() {
			int x = 42;
			unsigned long ul = (unsigned long)x;
			return (int)ul;
		}
		"""
		_analyze_ok(src)


# ---------------------------------------------------------------------------
# Enum values in IR generation (positive cases)
# ---------------------------------------------------------------------------


class TestEnumIR:
	def test_enum_zero_ir(self):
		assert _ir_return_value("""
		enum Vals { A };
		int foo() { return A; }
		""") == 0

	def test_enum_auto_increment_ir(self):
		assert _ir_return_value("""
		enum Vals { A, B, C };
		int foo() { return C; }
		""") == 2

	def test_enum_explicit_large_ir(self):
		assert _ir_return_value("""
		enum Big { X = 999999 };
		int foo() { return X; }
		""") == 999999

	def test_enum_explicit_then_auto_ir(self):
		assert _ir_return_value("""
		enum Vals { A = 10, B };
		int foo() { return B; }
		""") == 11
