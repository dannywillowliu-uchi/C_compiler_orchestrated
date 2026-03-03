"""Tests for the assembly peephole optimizer."""

from compiler.peephole import PeepholeOptimizer


def _opt(asm: str) -> str:
	"""Helper: optimize assembly text."""
	return PeepholeOptimizer().optimize(asm)


# ---------------------------------------------------------------------------
# Import / smoke tests
# ---------------------------------------------------------------------------


class TestImport:
	def test_import(self) -> None:
		from compiler.peephole import PeepholeOptimizer as PO  # noqa: F811

		assert PO is not None

	def test_empty_input(self) -> None:
		assert _opt("") == ""

	def test_no_change(self) -> None:
		asm = "\tmovq $42, %rax\n\tret"
		assert _opt(asm) == asm


# ---------------------------------------------------------------------------
# Pattern 1: Store-then-reload elimination
# ---------------------------------------------------------------------------


class TestStoreReload:
	def test_basic_store_reload(self) -> None:
		asm = "\tmovq %rax, -8(%rbp)\n\tmovq -8(%rbp), %rax"
		result = _opt(asm)
		assert result == "\tmovq %rax, -8(%rbp)"

	def test_different_offsets_not_eliminated(self) -> None:
		asm = "\tmovq %rax, -8(%rbp)\n\tmovq -16(%rbp), %rax"
		assert _opt(asm) == asm

	def test_different_registers_not_eliminated(self) -> None:
		asm = "\tmovq %rax, -8(%rbp)\n\tmovq -8(%rbp), %rcx"
		assert _opt(asm) == asm

	def test_store_reload_with_surrounding_code(self) -> None:
		lines = [
			"\tpushq %rbp",
			"\tmovq %rax, -8(%rbp)",
			"\tmovq -8(%rbp), %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tpushq %rbp",
			"\tmovq %rax, -8(%rbp)",
			"\tret",
		])
		assert result == expected

	def test_multiple_store_reloads(self) -> None:
		lines = [
			"\tmovq %rax, -8(%rbp)",
			"\tmovq -8(%rbp), %rax",
			"\tmovq %rcx, -16(%rbp)",
			"\tmovq -16(%rbp), %rcx",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tmovq %rax, -8(%rbp)",
			"\tmovq %rcx, -16(%rbp)",
		])
		assert result == expected

	def test_store_reload_with_rcx(self) -> None:
		asm = "\tmovq %rcx, -24(%rbp)\n\tmovq -24(%rbp), %rcx"
		result = _opt(asm)
		assert result == "\tmovq %rcx, -24(%rbp)"


# ---------------------------------------------------------------------------
# Pattern 2: Self-move elimination
# ---------------------------------------------------------------------------


class TestSelfMove:
	def test_self_move_rax(self) -> None:
		asm = "\tmovq %rax, %rax"
		assert _opt(asm) == ""

	def test_self_move_rcx(self) -> None:
		asm = "\tmovq %rcx, %rcx"
		assert _opt(asm) == ""

	def test_self_move_r8(self) -> None:
		asm = "\tmovq %r8, %r8"
		assert _opt(asm) == ""

	def test_non_self_move_preserved(self) -> None:
		asm = "\tmovq %rax, %rcx"
		assert _opt(asm) == asm

	def test_self_move_among_other_code(self) -> None:
		lines = [
			"\tpushq %rbp",
			"\tmovq %rax, %rax",
			"\taddq %rcx, %rax",
			"\tmovq %rbx, %rbx",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tpushq %rbp",
			"\taddq %rcx, %rax",
			"\tret",
		])
		assert result == expected


# ---------------------------------------------------------------------------
# Pattern 3: Zero-cmp collapse
# ---------------------------------------------------------------------------


class TestZeroCmpCollapse:
	def test_basic_zero_cmp(self) -> None:
		asm = "\tmovq $0, %rax\n\tcmpq $0, %rax"
		result = _opt(asm)
		assert result == "\txorq %rax, %rax\n\ttestq %rax, %rax"

	def test_zero_cmp_rcx(self) -> None:
		asm = "\tmovq $0, %rcx\n\tcmpq $0, %rcx"
		result = _opt(asm)
		assert result == "\txorq %rcx, %rcx\n\ttestq %rcx, %rcx"

	def test_different_registers_not_collapsed(self) -> None:
		asm = "\tmovq $0, %rax\n\tcmpq $0, %rcx"
		assert _opt(asm) == asm

	def test_non_zero_constant_not_collapsed(self) -> None:
		asm = "\tmovq $1, %rax\n\tcmpq $0, %rax"
		assert _opt(asm) == asm

	def test_zero_cmp_with_context(self) -> None:
		lines = [
			"\tmovq $0, %rax",
			"\tcmpq $0, %rax",
			"\tjne .L1",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\txorq %rax, %rax",
			"\ttestq %rax, %rax",
			"\tjne .L1",
		])
		assert result == expected


# ---------------------------------------------------------------------------
# Pattern 4: No-op arithmetic elimination
# ---------------------------------------------------------------------------


class TestNoopArith:
	def test_addq_zero(self) -> None:
		asm = "\taddq $0, %rax"
		assert _opt(asm) == ""

	def test_subq_zero(self) -> None:
		asm = "\tsubq $0, %rsp"
		assert _opt(asm) == ""

	def test_non_zero_add_preserved(self) -> None:
		asm = "\taddq $8, %rsp"
		assert _opt(asm) == asm

	def test_non_zero_sub_preserved(self) -> None:
		asm = "\tsubq $16, %rsp"
		assert _opt(asm) == asm

	def test_noop_arith_among_other_code(self) -> None:
		lines = [
			"\tpushq %rbp",
			"\tsubq $0, %rsp",
			"\tmovq $42, %rax",
			"\taddq $0, %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tpushq %rbp",
			"\tmovq $42, %rax",
			"\tret",
		])
		assert result == expected


# ---------------------------------------------------------------------------
# Combined patterns
# ---------------------------------------------------------------------------


class TestCombined:
	def test_multiple_patterns_in_sequence(self) -> None:
		lines = [
			"\tmovq %rax, -8(%rbp)",
			"\tmovq -8(%rbp), %rax",
			"\tmovq %rcx, %rcx",
			"\taddq $0, %rax",
			"\tmovq $0, %rax",
			"\tcmpq $0, %rax",
			"\tsubq $0, %rsp",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			"\tmovq %rax, -8(%rbp)",
			"\txorq %rax, %rax",
			"\ttestq %rax, %rax",
		])
		assert result == expected

	def test_labels_and_directives_preserved(self) -> None:
		lines = [
			".section .text",
			".globl main",
			"main:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tmovq %rax, %rax",
			"\tret",
		]
		result = _opt("\n".join(lines))
		expected = "\n".join([
			".section .text",
			".globl main",
			"main:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tret",
		])
		assert result == expected

	def test_cascading_optimization(self) -> None:
		"""After removing a self-move, a store-reload pair may become adjacent."""
		lines = [
			"\tmovq %rax, -8(%rbp)",
			"\tmovq %rcx, %rcx",
			"\tmovq -8(%rbp), %rax",
		]
		result = _opt("\n".join(lines))
		assert result == "\tmovq %rax, -8(%rbp)"

	def test_realistic_function(self) -> None:
		lines = [
			".section .text",
			".globl add",
			"add:",
			"\tpushq %rbp",
			"\tmovq %rsp, %rbp",
			"\tsubq $16, %rsp",
			"\tmovq %rdi, -8(%rbp)",
			"\tmovq %rsi, -16(%rbp)",
			"\tmovq -8(%rbp), %rax",
			"\tmovq %rax, -8(%rbp)",
			"\tmovq -8(%rbp), %rax",
			"\tmovq -16(%rbp), %rcx",
			"\taddq %rcx, %rax",
			"\tmovq %rax, -8(%rbp)",
			"\tmovq -8(%rbp), %rax",
			"\tmovq %rbp, %rsp",
			"\tpopq %rbp",
			"\tret",
		]
		result = _opt("\n".join(lines))
		# Store-reload pairs should be collapsed
		result_lines = result.split("\n")
		# No line should be a reload of a just-stored value
		for j in range(len(result_lines) - 1):
			if "movq %rax, -8(%rbp)" in result_lines[j]:
				assert result_lines[j + 1] != "\tmovq -8(%rbp), %rax"
