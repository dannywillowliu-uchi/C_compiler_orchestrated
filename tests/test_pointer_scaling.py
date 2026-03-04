"""Tests for pointer arithmetic scaling and long/short IR type mapping."""

from compiler.ir import IRBinOp, IRConst, IRCopy, IRType, ir_type_byte_width
from compiler.ir_gen import IRGenerator
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer


def _generate_ir(source: str):
	prog = Parser.from_source(source).parse()
	SemanticAnalyzer().analyze(prog)
	return IRGenerator().generate(prog)


def _get_function_body(ir_prog, func_name: str = "main"):
	for f in ir_prog.functions:
		if f.name == func_name:
			return f.body
	raise ValueError(f"Function {func_name} not found")


# ---------------------------------------------------------------------------
# Pointer + integer scaling
# ---------------------------------------------------------------------------


class TestPointerPlusInteger:
	def test_int_ptr_plus_int_scales_by_4(self) -> None:
		"""int *p; p + 3 should emit multiply by 4 (sizeof(int))."""
		src = """
		int main() {
			int *p;
			int *q = p + 3;
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		body = _get_function_body(ir_prog)
		# Find the multiply-by-4 instruction for scaling
		mul_instrs = [i for i in body if isinstance(i, IRBinOp) and i.op == "*"]
		assert len(mul_instrs) >= 1, "Expected a multiply for pointer scaling"
		scale_instr = mul_instrs[0]
		assert isinstance(scale_instr.right, IRConst)
		assert scale_instr.right.value == 4

	def test_char_ptr_plus_int_no_scaling(self) -> None:
		"""char *p; p + 3 should NOT emit a multiply (sizeof(char) == 1)."""
		src = """
		int main() {
			char *p;
			char *q = p + 3;
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		body = _get_function_body(ir_prog)
		# Should not have a multiply for scaling since char is 1 byte
		mul_instrs = [i for i in body if isinstance(i, IRBinOp) and i.op == "*"]
		assert len(mul_instrs) == 0, "char pointer add should not scale"

	def test_double_ptr_plus_int_scales_by_8(self) -> None:
		"""double *p; p + 2 should emit multiply by 8 (sizeof(double))."""
		src = """
		int main() {
			double *p;
			double *q = p + 2;
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		body = _get_function_body(ir_prog)
		mul_instrs = [i for i in body if isinstance(i, IRBinOp) and i.op == "*"]
		assert len(mul_instrs) >= 1
		scale_instr = mul_instrs[0]
		assert isinstance(scale_instr.right, IRConst)
		assert scale_instr.right.value == 8

	def test_int_plus_ptr_scales(self) -> None:
		"""3 + p should also scale the integer by sizeof(int)."""
		src = """
		int main() {
			int *p;
			int *q = 3 + p;
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		body = _get_function_body(ir_prog)
		mul_instrs = [i for i in body if isinstance(i, IRBinOp) and i.op == "*"]
		assert len(mul_instrs) >= 1
		scale_instr = mul_instrs[0]
		assert isinstance(scale_instr.right, IRConst)
		assert scale_instr.right.value == 4


# ---------------------------------------------------------------------------
# Pointer - integer scaling
# ---------------------------------------------------------------------------


class TestPointerMinusInteger:
	def test_int_ptr_minus_int_scales_by_4(self) -> None:
		"""int *p; p - 2 should emit multiply by 4 for the subtracted offset."""
		src = """
		int main() {
			int *p;
			int *q = p - 2;
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		body = _get_function_body(ir_prog)
		mul_instrs = [i for i in body if isinstance(i, IRBinOp) and i.op == "*"]
		assert len(mul_instrs) >= 1
		scale_instr = mul_instrs[0]
		assert isinstance(scale_instr.right, IRConst)
		assert scale_instr.right.value == 4


# ---------------------------------------------------------------------------
# Pointer - pointer (element count)
# ---------------------------------------------------------------------------


class TestPointerMinusPointer:
	def test_ptr_diff_divides_by_element_size(self) -> None:
		"""int *p, *q; p - q should produce a division by 4."""
		src = """
		int main() {
			int *p;
			int *q;
			int diff = p - q;
			return diff;
		}
		"""
		ir_prog = _generate_ir(src)
		body = _get_function_body(ir_prog)
		div_instrs = [i for i in body if isinstance(i, IRBinOp) and i.op == "/"]
		assert len(div_instrs) >= 1, "pointer diff should emit division by element size"
		assert isinstance(div_instrs[0].right, IRConst)
		assert div_instrs[0].right.value == 4

	def test_char_ptr_diff_no_division(self) -> None:
		"""char *p, *q; p - q should NOT divide (sizeof(char) == 1)."""
		src = """
		int main() {
			char *p;
			char *q;
			int diff = p - q;
			return diff;
		}
		"""
		ir_prog = _generate_ir(src)
		body = _get_function_body(ir_prog)
		div_instrs = [i for i in body if isinstance(i, IRBinOp) and i.op == "/"]
		assert len(div_instrs) == 0, "char pointer diff should not divide"


# ---------------------------------------------------------------------------
# Pointer postfix ++/--
# ---------------------------------------------------------------------------


class TestPointerPostfix:
	def test_int_ptr_postfix_increment(self) -> None:
		"""int *p; p++ should increment by 4 (sizeof(int))."""
		src = """
		int main() {
			int *p;
			p++;
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		body = _get_function_body(ir_prog)
		add_instrs = [i for i in body if isinstance(i, IRBinOp) and i.op == "+"]
		assert len(add_instrs) >= 1
		# The delta should be 4, not 1
		ptr_add = add_instrs[0]
		assert isinstance(ptr_add.right, IRConst)
		assert ptr_add.right.value == 4

	def test_int_ptr_postfix_decrement(self) -> None:
		"""int *p; p-- should decrement by 4."""
		src = """
		int main() {
			int *p;
			p--;
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		body = _get_function_body(ir_prog)
		sub_instrs = [i for i in body if isinstance(i, IRBinOp) and i.op == "-"]
		assert len(sub_instrs) >= 1
		ptr_sub = sub_instrs[0]
		assert isinstance(ptr_sub.right, IRConst)
		assert ptr_sub.right.value == 4

	def test_char_ptr_postfix_increment_by_1(self) -> None:
		"""char *p; p++ should increment by 1."""
		src = """
		int main() {
			char *p;
			p++;
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		body = _get_function_body(ir_prog)
		add_instrs = [i for i in body if isinstance(i, IRBinOp) and i.op == "+"]
		assert len(add_instrs) >= 1
		assert isinstance(add_instrs[0].right, IRConst)
		assert add_instrs[0].right.value == 1


# ---------------------------------------------------------------------------
# Pointer prefix ++/--
# ---------------------------------------------------------------------------


class TestPointerPrefix:
	def test_int_ptr_prefix_increment(self) -> None:
		"""++p where p is int* should increment by 4."""
		src = """
		int main() {
			int *p;
			++p;
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		body = _get_function_body(ir_prog)
		add_instrs = [i for i in body if isinstance(i, IRBinOp) and i.op == "+"]
		assert len(add_instrs) >= 1
		assert isinstance(add_instrs[0].right, IRConst)
		assert add_instrs[0].right.value == 4

	def test_int_ptr_prefix_decrement(self) -> None:
		"""--p where p is int* should decrement by 4."""
		src = """
		int main() {
			int *p;
			--p;
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		body = _get_function_body(ir_prog)
		sub_instrs = [i for i in body if isinstance(i, IRBinOp) and i.op == "-"]
		assert len(sub_instrs) >= 1
		assert isinstance(sub_instrs[0].right, IRConst)
		assert sub_instrs[0].right.value == 4


# ---------------------------------------------------------------------------
# IR type mapping: long and short
# ---------------------------------------------------------------------------


class TestIRTypeMapping:
	def test_long_maps_to_ir_long(self) -> None:
		"""long x should use IRType.LONG."""
		src = """
		int main() {
			long x = 100;
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		body = _get_function_body(ir_prog)
		# Find the copy instruction that initializes x
		copy_instrs = [i for i in body if isinstance(i, IRCopy) and isinstance(i.source, IRConst) and i.source.value == 100]
		assert len(copy_instrs) >= 1
		assert copy_instrs[0].ir_type == IRType.LONG

	def test_long_long_maps_to_ir_long(self) -> None:
		"""long long x should also use IRType.LONG."""
		src = """
		int main() {
			long long x = 200;
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		body = _get_function_body(ir_prog)
		copy_instrs = [i for i in body if isinstance(i, IRCopy) and isinstance(i.source, IRConst) and i.source.value == 200]
		assert len(copy_instrs) >= 1
		assert copy_instrs[0].ir_type == IRType.LONG

	def test_short_maps_to_ir_short(self) -> None:
		"""short x should use IRType.SHORT."""
		src = """
		int main() {
			short x = 42;
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		body = _get_function_body(ir_prog)
		copy_instrs = [i for i in body if isinstance(i, IRCopy) and isinstance(i.source, IRConst) and i.source.value == 42]
		assert len(copy_instrs) >= 1
		assert copy_instrs[0].ir_type == IRType.SHORT

	def test_long_byte_width_is_8(self) -> None:
		"""IRType.LONG should have 8-byte width."""
		assert ir_type_byte_width(IRType.LONG) == 8

	def test_short_byte_width_is_2(self) -> None:
		"""IRType.SHORT should have 2-byte width."""
		assert ir_type_byte_width(IRType.SHORT) == 2

	def test_long_function_return_type(self) -> None:
		"""A function returning long should have IRType.LONG return type."""
		src = """
		long get_val() {
			return 5000000000;
		}
		int main() {
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		for f in ir_prog.functions:
			if f.name == "get_val":
				assert f.return_type == IRType.LONG
				break
		else:
			raise AssertionError("get_val function not found")

	def test_short_function_return_type(self) -> None:
		"""A function returning short should have IRType.SHORT return type."""
		src = """
		short get_val() {
			return 10;
		}
		int main() {
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		for f in ir_prog.functions:
			if f.name == "get_val":
				assert f.return_type == IRType.SHORT
				break
		else:
			raise AssertionError("get_val function not found")


# ---------------------------------------------------------------------------
# Pointer to pointer arithmetic
# ---------------------------------------------------------------------------


class TestPointerToPointerArithmetic:
	def test_ptr_to_ptr_plus_int_scales_by_8(self) -> None:
		"""int **pp; pp + 1 should scale by 8 (sizeof(int*))."""
		src = """
		int main() {
			int **pp;
			int **qq = pp + 1;
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		body = _get_function_body(ir_prog)
		mul_instrs = [i for i in body if isinstance(i, IRBinOp) and i.op == "*"]
		assert len(mul_instrs) >= 1
		assert isinstance(mul_instrs[0].right, IRConst)
		assert mul_instrs[0].right.value == 8


# ---------------------------------------------------------------------------
# Pointer arithmetic via function parameter
# ---------------------------------------------------------------------------


class TestPointerParamArithmetic:
	def test_param_ptr_plus_int(self) -> None:
		"""Function with int* param: param + n should scale by 4."""
		src = """
		int deref_offset(int *arr, int n) {
			int *p = arr + n;
			return *p;
		}
		int main() {
			return 0;
		}
		"""
		ir_prog = _generate_ir(src)
		body = _get_function_body(ir_prog, "deref_offset")
		mul_instrs = [i for i in body if isinstance(i, IRBinOp) and i.op == "*"]
		assert len(mul_instrs) >= 1
		assert isinstance(mul_instrs[0].right, IRConst)
		assert mul_instrs[0].right.value == 4
