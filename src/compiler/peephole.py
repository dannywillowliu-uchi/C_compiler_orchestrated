"""Assembly peephole optimizer for redundant instruction elimination."""

from __future__ import annotations

import re


class PeepholeOptimizer:
	"""Pattern-based peephole optimizer for x86-64 AT&T syntax assembly."""

	def __init__(self) -> None:
		# Regex patterns compiled once for reuse
		self._movq_store_re = re.compile(
			r"^\tmovq\s+(%\w+),\s+(-?\d+\(%rbp\))$"
		)
		self._movq_load_re = re.compile(
			r"^\tmovq\s+(-?\d+\(%rbp\)),\s+(%\w+)$"
		)
		self._movq_reg_re = re.compile(
			r"^\tmovq\s+(%\w+),\s+(%\w+)$"
		)
		self._movq_zero_re = re.compile(
			r"^\tmovq\s+\$0,\s+(%\w+)$"
		)
		self._cmpq_zero_re = re.compile(
			r"^\tcmpq\s+\$0,\s+(%\w+)$"
		)
		self._addq_zero_re = re.compile(
			r"^\taddq\s+\$0,\s+%\w+$"
		)
		self._subq_zero_re = re.compile(
			r"^\tsubq\s+\$0,\s+%\w+$"
		)
		self._movq_imm_reg_re = re.compile(
			r"^\tmovq\s+\$(-?\d+),\s+(%\w+)$"
		)
		self._addq_reg_reg_re = re.compile(
			r"^\taddq\s+(%\w+),\s+(%\w+)$"
		)
		self._pushq_re = re.compile(r"^\tpushq\s+(%\w+)$")
		self._popq_re = re.compile(r"^\tpopq\s+(%\w+)$")
		self._jmp_re = re.compile(r"^\tjmp\s+(\.\w+)$")
		self._label_re = re.compile(r"^(\.\w+):$")
		# New patterns for address-mode folding and redundant comparison elimination
		self._arith_flag_setting_re = re.compile(
			r"^\t(addq|subq|andq|orq|xorq)\s+[^,]+,\s+(%\w+)$"
		)
		# Extended flag-setting instructions (single-operand or shift forms)
		self._negq_re = re.compile(r"^\tnegq\s+(%\w+)$")
		self._incq_re = re.compile(r"^\tincq\s+(%\w+)$")
		self._decq_re = re.compile(r"^\tdecq\s+(%\w+)$")
		self._shift_re = re.compile(
			r"^\t(shlq|shrq|sarq)\s+\$\d+,\s+(%\w+)$"
		)
		self._testq_self_re = re.compile(
			r"^\ttestq\s+(%\w+),\s+\1$"
		)
		# Conditional jump patterns for cmov optimization
		self._jcc_re = re.compile(r"^\t(je|jne|jg|jge|jl|jle|ja|jae|jb|jbe|js|jns)\s+(\.\w+)$")
		# cmov condition inversion map
		self._jcc_to_cmov = {
			"je": "cmovne", "jne": "cmove",
			"jg": "cmovle", "jge": "cmovl",
			"jl": "cmovge", "jle": "cmovg",
			"ja": "cmovbe", "jae": "cmovb",
			"jb": "cmovae", "jbe": "cmova",
			"js": "cmovns", "jns": "cmovs",
		}
		self._leaq_rip_re = re.compile(
			r"^\tleaq\s+(\w+)\(%rip\),\s+(%\w+)$"
		)
		self._movq_indirect_re = re.compile(
			r"^\tmovq\s+\((%\w+)\),\s+(%\w+)$"
		)
		self._addq_imm_reg_re = re.compile(
			r"^\taddq\s+(%\w+),\s+(%\w+)$"
		)
		# General mov self-move pattern (handles movl, movw, movb in addition to movq)
		self._mov_any_reg_re = re.compile(
			r"^\tmov[lwb]\s+(%\w+),\s+(%\w+)$"
		)
		# movl $0, %reg -> xorl %reg, %reg
		self._movl_zero_re = re.compile(
			r"^\tmovl\s+\$0,\s+(%\w+)$"
		)
		# Strength reduction patterns
		self._imulq_imm_re = re.compile(
			r"^\timulq\s+\$(\d+),\s+(%\w+)$"
		)
		# testq %reg, %reg pattern for redundant cmpq elimination
		self._testq_reg_re = re.compile(
			r"^\ttestq\s+(%\w+),\s+(%\w+)$"
		)
		# Unconditional control flow (jmp or ret)
		self._unconditional_jmp_re = re.compile(r"^\t(jmp|ret)\b")
		# Immediate add/sub for combining adjacent operations
		self._addq_imm_val_re = re.compile(
			r"^\taddq\s+\$(-?\d+),\s+(%\w+)$"
		)
		self._subq_imm_val_re = re.compile(
			r"^\tsubq\s+\$(-?\d+),\s+(%\w+)$"
		)
		# leaq 0(%rA), %rB -> movq %rA, %rB
		self._leaq_zero_offset_re = re.compile(
			r"^\tleaq\s+0\((%\w+)\),\s+(%\w+)$"
		)
		# General instruction destination register extraction (for dead move detection)
		self._instr_dst_re = re.compile(
			r"^\t(?:movq|movl|movw|movb|leaq|xorq|xorl)\s+([^,]+),\s+(%\w+)$"
		)
		# Division strength reduction patterns
		self._cqto_re = re.compile(r"^\tcqto$")
		self._idivq_re = re.compile(r"^\tidivq\s+(%\w+)$")
		self._xorq_rdx_re = re.compile(r"^\txorq\s+%rdx,\s+%rdx$")
		self._divq_re = re.compile(r"^\tdivq\s+(%\w+)$")
		# Push different reg pop: pushq %rA; popq %rB -> movq %rA, %rB
		self._pushq_any_re = self._pushq_re
		self._popq_any_re = self._popq_re
		# addq $imm, %reg pattern for lea folding
		self._addq_imm_to_reg_re = re.compile(
			r"^\taddq\s+\$(-?\d+),\s+(%\w+)$"
		)
		# Extension instruction patterns (movzbl, movsbl, movzwl, movswl, movzbw, movsbw)
		self._ext_re = re.compile(
			r"^\t(movzbl|movsbl|movzwl|movswl|movzbw|movsbw)\s+(%\w+),\s+(%\w+)$"
		)
		# Truncation patterns: movb/movw from register to register
		self._trunc_re = re.compile(
			r"^\tmov([bw])\s+(%\w+),\s+(%\w+)$"
		)
		# testl %reg, %reg
		self._testl_self_re = re.compile(
			r"^\ttestl\s+(%\w+),\s+\1$"
		)
		# Operations that implicitly zero-extend their 32-bit result into 64-bit register:
		# movl, addl, subl, andl, orl, xorl, imull, shll, shrl, sarl
		self._zero_ext_op_re = re.compile(
			r"^\t(movl|addl|subl|andl|orl|xorl|imull|shll|shrl|sarl)\s+[^,]+,\s+(%\w+)$"
		)
		# andl with 8-bit or 16-bit mask (value fits in byte/word range)
		self._andl_imm_re = re.compile(
			r"^\tandl\s+\$(\d+),\s+(%\w+)$"
		)

	def optimize(self, assembly: str) -> str:
		"""Apply peephole optimizations to assembly text and return optimized version."""
		lines = assembly.split("\n")
		changed = True
		while changed:
			lines, changed = self._apply_pass(lines)
		return "\n".join(lines)

	def _apply_pass(self, lines: list[str]) -> tuple[list[str], bool]:
		"""Run one pass of all peephole patterns. Returns (new_lines, changed)."""
		result: list[str] = []
		changed = False
		i = 0
		while i < len(lines):
			# Try 2-instruction window patterns
			if i + 1 < len(lines):
				# Store-then-reload elimination
				opt = self._try_store_reload(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# movq $0 + cmpq $0 -> xorq + testq
				opt = self._try_zero_cmp(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Reverse move elimination: movq %rA, %rB + movq %rB, %rA
				opt = self._try_reverse_move(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Mov chain elimination
				opt = self._try_mov_chain(lines, i)
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# LEA strength reduction
				opt = self._try_lea_reduction(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# LEA folding: movq %rA, %rB + addq $imm, %rB -> leaq imm(%rA), %rB
				opt = self._try_mov_addimm_to_lea(lines, i)
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# LEA folding: movq %rA, %rB + addq %rC, %rB -> leaq (%rA,%rC), %rB
				opt = self._try_mov_addreg_to_lea(lines, i)
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Push/pop elimination
				opt = self._try_push_pop(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Dead store elimination
				opt = self._try_dead_store(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Jump-to-next elimination
				opt = self._try_jump_to_next(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Redundant comparison after flag-setting arithmetic
				opt = self._try_redundant_cmp(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# LEA+load RIP-relative folding
				opt = self._try_lea_load_fold(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Immediate add folding: movq $imm, %rA + addq %rA, %rB -> addq $imm, %rB
				opt = self._try_imm_add_fold(lines, i)
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Combine adjacent addq/subq $imm to same register
				opt = self._try_combine_add_sub(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Redundant consecutive load elimination
				opt = self._try_redundant_load(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Dead load elimination: load to register immediately overwritten
				opt = self._try_dead_load(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Redundant cmpq $0 after testq on same register
				opt = self._try_redundant_cmp_after_test(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Redundant testq after flag-setting arithmetic
				opt = self._try_redundant_test_after_arith(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Redundant cmp/test after negq/incq/decq
				opt = self._try_redundant_cmp_after_unary(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Redundant cmp/test after shift
				opt = self._try_redundant_cmp_after_shift(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Duplicate extension elimination
				opt = self._try_duplicate_extension(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Extension after truncation elimination
				opt = self._try_ext_after_trunc(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# movzbl + testl -> testb folding
				opt = self._try_movzbl_testl_fold(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Redundant movzbl after zero-extending operation
				opt = self._try_redundant_ext_after_zero_ext_op(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Consecutive identical register moves
				opt = self._try_duplicate_move(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Dead register move: movq %rA, %rB where %rB is immediately overwritten
				opt = self._try_dead_reg_move(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

			# 3-instruction window patterns
			if i + 2 < len(lines):
				# Signed division strength reduction:
				# movq $N, %reg + cqto + idivq %reg -> sarq $log2(N), %rax
				opt = self._try_signed_div_strength(lines[i], lines[i + 1], lines[i + 2])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 3
					continue

				# Unsigned division strength reduction:
				# movq $N, %reg + xorq %rdx, %rdx + divq %reg -> shrq $log2(N), %rax
				opt = self._try_unsigned_div_strength(lines[i], lines[i + 1], lines[i + 2])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 3
					continue

			# Conditional branch to cmov optimization (5-line window)
			if i + 4 < len(lines):
				opt = self._try_branch_to_cmov(lines, i)
				if opt is not None:
					result.extend(opt[0])
					changed = True
					i += opt[1]
					continue

			# Push/pop to different register: pushq %rA; popq %rB -> movq %rA, %rB
			if i + 1 < len(lines):
				opt = self._try_push_pop_move(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

			# Dead code elimination after unconditional jmp/ret
			consumed = self._try_dead_code_after_jump(lines, i)
			if consumed is not None:
				result.append(lines[i])
				changed = True
				i = consumed
				continue

			# cmpq $0, %reg -> testq %reg, %reg
			opt = self._try_cmp_zero_to_test(lines[i])
			if opt is not None:
				result.append(opt)
				changed = True
				i += 1
				continue

			# movq $0, %reg -> xorq %reg, %reg
			opt = self._try_mov_zero_to_xor(lines[i])
			if opt is not None:
				result.append(opt)
				changed = True
				i += 1
				continue

			# movl $0, %reg -> xorl %reg, %reg
			opt = self._try_movl_zero_to_xorl(lines[i])
			if opt is not None:
				result.append(opt)
				changed = True
				i += 1
				continue

			# Strength reduction: imulq by power of 2 -> shlq
			opt = self._try_strength_reduction_mul(lines[i])
			if opt is not None:
				if opt:
					result.append(opt)
				changed = True
				i += 1
				continue

			# leaq 0(%rA), %rB -> movq %rA, %rB
			opt = self._try_lea_zero_fold(lines[i])
			if opt is not None:
				result.append(opt)
				changed = True
				i += 1
				continue

			# Self-move elimination (movq and movl/movw/movb)
			if self._is_self_move(lines[i]):
				changed = True
				i += 1
				continue

			# addq $0 / subq $0 elimination
			if self._is_noop_arith(lines[i]):
				changed = True
				i += 1
				continue

			result.append(lines[i])
			i += 1

		return result, changed

	def _try_store_reload(self, line1: str, line2: str) -> list[str] | None:
		"""Remove redundant store followed by immediate reload of same location."""
		store_m = self._movq_store_re.match(line1)
		if store_m is None:
			return None
		load_m = self._movq_load_re.match(line2)
		if load_m is None:
			return None
		if store_m.group(2) == load_m.group(1) and store_m.group(1) == load_m.group(2):
			return [line1]
		return None

	def _try_zero_cmp(self, line1: str, line2: str) -> list[str] | None:
		"""Collapse movq $0, %reg + cmpq $0, %reg -> xorq + testq."""
		mov_m = self._movq_zero_re.match(line1)
		if mov_m is None:
			return None
		cmp_m = self._cmpq_zero_re.match(line2)
		if cmp_m is None:
			return None
		if mov_m.group(1) == cmp_m.group(1):
			reg = mov_m.group(1)
			return [f"\txorq {reg}, {reg}", f"\ttestq {reg}, {reg}"]
		return None

	def _try_mov_chain(self, lines: list[str], idx: int) -> list[str] | None:
		"""Eliminate mov chain: movq %rA, %rB + movq %rB, %rC -> movq %rA, %rC.

		Only fires when %rB is dead after the chain.
		"""
		m1 = self._movq_reg_re.match(lines[idx])
		if m1 is None:
			return None
		m2 = self._movq_reg_re.match(lines[idx + 1])
		if m2 is None:
			return None
		ra, rb1 = m1.group(1), m1.group(2)
		rb2, rc = m2.group(1), m2.group(2)
		if rb1 != rb2:
			return None
		# Skip self-moves (handled by another pattern)
		if ra == rb1 or rb2 == rc:
			return None
		if not self._is_reg_dead_in_window(lines, idx + 2, rb1):
			return None
		return [f"\tmovq {ra}, {rc}"]

	def _try_lea_reduction(self, line1: str, line2: str) -> list[str] | None:
		"""Replace movq $imm, %rA + addq %rB, %rA -> leaq imm(%rB), %rA."""
		m_mov = self._movq_imm_reg_re.match(line1)
		if m_mov is None:
			return None
		m_add = self._addq_reg_reg_re.match(line2)
		if m_add is None:
			return None
		imm = int(m_mov.group(1))
		mov_dst = m_mov.group(2)
		add_src = m_add.group(1)
		add_dst = m_add.group(2)
		if mov_dst != add_dst:
			return None
		if add_src == add_dst:
			return None
		if not (-2147483648 <= imm <= 2147483647):
			return None
		return [f"\tleaq {imm}({add_src}), {add_dst}"]

	def _try_push_pop(self, line1: str, line2: str) -> list[str] | None:
		"""Remove redundant pushq %rA + popq %rA pair."""
		m_push = self._pushq_re.match(line1)
		if m_push is None:
			return None
		m_pop = self._popq_re.match(line2)
		if m_pop is None:
			return None
		if m_push.group(1) == m_pop.group(1):
			return []
		return None

	def _try_dead_store(self, line1: str, line2: str) -> list[str] | None:
		"""Remove first store when two consecutive stores target the same stack location."""
		m1 = self._movq_store_re.match(line1)
		if m1 is None:
			return None
		m2 = self._movq_store_re.match(line2)
		if m2 is None:
			return None
		if m1.group(2) == m2.group(2):
			return [line2]
		return None

	def _try_jump_to_next(self, line1: str, line2: str) -> list[str] | None:
		"""Remove jmp to the immediately following label."""
		m_jmp = self._jmp_re.match(line1)
		if m_jmp is None:
			return None
		m_label = self._label_re.match(line2)
		if m_label is None:
			return None
		if m_jmp.group(1) == m_label.group(1):
			return [line2]
		return None

	def _is_self_move(self, line: str) -> bool:
		"""Check for mov %reg, %reg self-moves (movq, movl, movw, movb)."""
		m = self._movq_reg_re.match(line)
		if m is not None:
			return m.group(1) == m.group(2)
		m = self._mov_any_reg_re.match(line)
		if m is not None:
			return m.group(1) == m.group(2)
		return False

	def _is_noop_arith(self, line: str) -> bool:
		"""Check for addq $0, %reg or subq $0, %reg."""
		return self._addq_zero_re.match(line) is not None or self._subq_zero_re.match(line) is not None

	def _try_cmp_zero_to_test(self, line: str) -> str | None:
		"""Replace 'cmpq $0, %reg' with 'testq %reg, %reg' (shorter encoding)."""
		m = self._cmpq_zero_re.match(line)
		if m is None:
			return None
		reg = m.group(1)
		return f"\ttestq {reg}, {reg}"

	def _try_mov_zero_to_xor(self, line: str) -> str | None:
		"""Replace 'movq $0, %reg' with 'xorq %reg, %reg' (shorter encoding)."""
		m = self._movq_zero_re.match(line)
		if m is None:
			return None
		reg = m.group(1)
		return f"\txorq {reg}, {reg}"

	def _try_movl_zero_to_xorl(self, line: str) -> str | None:
		"""Replace 'movl $0, %reg' with 'xorl %reg, %reg' (shorter, breaks dep chains)."""
		m = self._movl_zero_re.match(line)
		if m is None:
			return None
		reg = m.group(1)
		return f"\txorl {reg}, {reg}"

	def _try_reverse_move(self, line1: str, line2: str) -> list[str] | None:
		"""Eliminate movq %rA, %rB followed by movq %rB, %rA (reverse copy is redundant)."""
		m1 = self._movq_reg_re.match(line1)
		if m1 is None:
			return None
		m2 = self._movq_reg_re.match(line2)
		if m2 is None:
			return None
		ra, rb = m1.group(1), m1.group(2)
		if ra == rb:
			return None
		if m2.group(1) == rb and m2.group(2) == ra:
			return [line1]
		return None

	def _try_strength_reduction_mul(self, line: str) -> str | None:
		"""Replace imulq by power of 2 with shlq. Returns '' to eliminate imulq $1."""
		m = self._imulq_imm_re.match(line)
		if m is None:
			return None
		imm = int(m.group(1))
		reg = m.group(2)
		if imm == 0:
			return f"\txorq {reg}, {reg}"
		if imm == 1:
			return ""
		if imm & (imm - 1) != 0:
			return None
		shift = imm.bit_length() - 1
		return f"\tshlq ${shift}, {reg}"

	def _try_combine_add_sub(self, line1: str, line2: str) -> list[str] | None:
		"""Combine adjacent addq/subq $imm to same register into single instruction."""
		val1, reg1 = self._parse_add_sub_imm(line1)
		if val1 is None:
			return None
		val2, reg2 = self._parse_add_sub_imm(line2)
		if val2 is None:
			return None
		if reg1 != reg2:
			return None
		combined = val1 + val2
		if combined == 0:
			return []
		if combined > 0:
			return [f"\taddq ${combined}, {reg1}"]
		return [f"\tsubq ${-combined}, {reg1}"]

	def _parse_add_sub_imm(self, line: str) -> tuple[int | None, str | None]:
		"""Parse addq/subq $imm, %reg returning (signed_value, reg)."""
		m = self._addq_imm_val_re.match(line)
		if m:
			return int(m.group(1)), m.group(2)
		m = self._subq_imm_val_re.match(line)
		if m:
			return -int(m.group(1)), m.group(2)
		return None, None

	def _try_redundant_cmp(self, line1: str, line2: str) -> list[str] | None:
		"""Eliminate cmp after flag-setting arithmetic on the same register.

		add/sub/and/or/xor already set flags, so a following cmpq $0 on the
		same destination register is redundant.
		"""
		m_arith = self._arith_flag_setting_re.match(line1)
		if m_arith is None:
			return None
		m_cmp = self._cmpq_zero_re.match(line2)
		if m_cmp is None:
			return None
		if m_arith.group(2) == m_cmp.group(1):
			return [line1]
		return None

	def _try_lea_load_fold(self, line1: str, line2: str) -> list[str] | None:
		"""Fold 'leaq sym(%rip), %rA; movq (%rA), %rA' into 'movq sym(%rip), %rA'."""
		m_lea = self._leaq_rip_re.match(line1)
		if m_lea is None:
			return None
		m_load = self._movq_indirect_re.match(line2)
		if m_load is None:
			return None
		sym = m_lea.group(1)
		lea_dst = m_lea.group(2)
		load_base = m_load.group(1)
		load_dst = m_load.group(2)
		if lea_dst == load_base and lea_dst == load_dst:
			return [f"\tmovq {sym}(%rip), {load_dst}"]
		return None

	def _try_imm_add_fold(self, lines: list[str], idx: int) -> list[str] | None:
		"""Fold 'movq $imm, %rA; addq %rA, %rB' into 'addq $imm, %rB'.

		Only fires when the immediate fits in 32 bits and %rA is not used after.
		"""
		m_mov = self._movq_imm_reg_re.match(lines[idx])
		if m_mov is None:
			return None
		m_add = self._addq_imm_reg_re.match(lines[idx + 1])
		if m_add is None:
			return None
		imm = int(m_mov.group(1))
		mov_dst = m_mov.group(2)
		add_src = m_add.group(1)
		add_dst = m_add.group(2)
		if mov_dst != add_src:
			return None
		if mov_dst == add_dst:
			return None
		if not (-2147483648 <= imm <= 2147483647):
			return None
		if not self._is_reg_dead_in_window(lines, idx + 2, mov_dst):
			return None
		return [f"\taddq ${imm}, {add_dst}"]

	def _try_redundant_load(self, line1: str, line2: str) -> list[str] | None:
		"""Eliminate redundant consecutive load of the same memory to the same register."""
		if line1 != line2:
			return None
		m = re.match(r"^\tmov[lqwb]\s+-?\d+\(%\w+\),\s+%\w+$", line1)
		if m is not None:
			return [line1]
		return None

	def _try_dead_load(self, line1: str, line2: str) -> list[str] | None:
		"""Remove a load whose destination register is immediately overwritten."""
		# Check if line1 is a load (mov from memory to register)
		m_load = self._movq_load_re.match(line1)
		if m_load is None:
			return None
		loaded_reg = m_load.group(2)
		# Check if line2 overwrites the same register without reading it
		# movq (from immediate or other non-reading source), leaq (never reads dest)
		m_overwrite = re.match(r"^\t(movq|leaq)\s+([^,]+),\s+(%\w+)$", line2)
		if m_overwrite is not None:
			if m_overwrite.group(3) == loaded_reg and loaded_reg not in m_overwrite.group(2):
				return [line2]
		# xorq %reg, %reg is a zeroing idiom (pure overwrite)
		m_xor = re.match(r"^\txorq\s+(%\w+),\s+(%\w+)$", line2)
		if m_xor is not None and m_xor.group(1) == m_xor.group(2) == loaded_reg:
			return [line2]
		return None

	def _try_redundant_cmp_after_test(self, line1: str, line2: str) -> list[str] | None:
		"""Eliminate cmpq $0 after testq on the same register.

		testq %reg, %reg already sets ZF/SF based on the register value,
		so a following cmpq $0, %reg is redundant.
		"""
		m_test = self._testq_reg_re.match(line1)
		if m_test is None:
			return None
		# testq %reg, %reg (same register)
		if m_test.group(1) != m_test.group(2):
			return None
		m_cmp = self._cmpq_zero_re.match(line2)
		if m_cmp is None:
			return None
		if m_test.group(1) == m_cmp.group(1):
			return [line1]
		return None

	def _try_dead_code_after_jump(self, lines: list[str], idx: int) -> int | None:
		"""Remove unreachable instructions after unconditional jmp/ret until next label.

		Returns the new index to continue from (pointing at the next label),
		or None if this pattern doesn't apply.
		"""
		if not self._unconditional_jmp_re.match(lines[idx]):
			return None
		# Scan forward: skip instructions until we hit a label, directive, or end
		j = idx + 1
		found_dead = False
		while j < len(lines):
			line = lines[j]
			stripped = line.strip()
			# Empty line: stop scanning
			if not stripped:
				break
			# Any label (both .Lxxx: and named labels like while_body1:)
			if stripped.endswith(":"):
				break
			# Directive lines (e.g. .globl, .section, .size)
			if stripped.startswith("."):
				break
			# Only remove tab-indented instructions
			if not line.startswith("\t") or line.startswith("\t."):
				break
			# This is an unreachable instruction - skip it
			found_dead = True
			j += 1
		if not found_dead:
			return None
		return j

	def _try_duplicate_move(self, line1: str, line2: str) -> list[str] | None:
		"""Eliminate consecutive identical register-to-register moves."""
		if line1 != line2:
			return None
		m = self._movq_reg_re.match(line1)
		if m is not None and m.group(1) != m.group(2):
			return [line1]
		return None

	def _try_dead_reg_move(self, line1: str, line2: str) -> list[str] | None:
		"""Eliminate movq %rA, %rB where %rB is immediately overwritten without being read."""
		m_mov = self._movq_reg_re.match(line1)
		if m_mov is None:
			return None
		src, dst = m_mov.group(1), m_mov.group(2)
		if src == dst:
			return None
		# Check if next instruction overwrites dst without reading it
		m_next = self._instr_dst_re.match(line2)
		if m_next is None:
			return None
		next_dst = m_next.group(2)
		next_src = m_next.group(1)
		if next_dst != dst:
			return None
		# Ensure dst is not read in the source operand of the overwriting instruction
		if dst in next_src:
			return None
		return [line2]

	def _try_lea_zero_fold(self, line: str) -> str | None:
		"""Replace 'leaq 0(%rA), %rB' with 'movq %rA, %rB'."""
		m = self._leaq_zero_offset_re.match(line)
		if m is None:
			return None
		src, dst = m.group(1), m.group(2)
		return f"\tmovq {src}, {dst}"

	def _try_signed_div_strength(self, line1: str, line2: str, line3: str) -> list[str] | None:
		"""Replace movq $N, %reg + cqto + idivq %reg -> sarq $log2(N), %rax when N is power of 2."""
		m_mov = self._movq_imm_reg_re.match(line1)
		if m_mov is None:
			return None
		imm = int(m_mov.group(1))
		reg = m_mov.group(2)
		if imm <= 0 or imm & (imm - 1) != 0:
			return None
		if not self._cqto_re.match(line2):
			return None
		m_div = self._idivq_re.match(line3)
		if m_div is None or m_div.group(1) != reg:
			return None
		if imm == 1:
			return []
		shift = imm.bit_length() - 1
		return [f"\tsarq ${shift}, %rax"]

	def _try_unsigned_div_strength(self, line1: str, line2: str, line3: str) -> list[str] | None:
		"""Replace movq $N, %reg + xorq %rdx,%rdx + divq %reg -> shrq $log2(N), %rax."""
		m_mov = self._movq_imm_reg_re.match(line1)
		if m_mov is None:
			return None
		imm = int(m_mov.group(1))
		reg = m_mov.group(2)
		if imm <= 0 or imm & (imm - 1) != 0:
			return None
		if not self._xorq_rdx_re.match(line2):
			return None
		m_div = self._divq_re.match(line3)
		if m_div is None or m_div.group(1) != reg:
			return None
		if imm == 1:
			return []
		shift = imm.bit_length() - 1
		return [f"\tshrq ${shift}, %rax"]

	def _try_push_pop_move(self, line1: str, line2: str) -> list[str] | None:
		"""Replace pushq %rA + popq %rB (different registers) with movq %rA, %rB."""
		m_push = self._pushq_re.match(line1)
		if m_push is None:
			return None
		m_pop = self._popq_re.match(line2)
		if m_pop is None:
			return None
		src, dst = m_push.group(1), m_pop.group(1)
		if src == dst:
			return None  # Same-register case handled by _try_push_pop
		return [f"\tmovq {src}, {dst}"]

	def _try_mov_addimm_to_lea(self, lines: list[str], idx: int) -> list[str] | None:
		"""Fold 'movq %rA, %rB; addq $imm, %rB' into 'leaq imm(%rA), %rB'.

		Only fires when %rA != %rB and %rA is not used after (dead in window).
		"""
		m_mov = self._movq_reg_re.match(lines[idx])
		if m_mov is None:
			return None
		ra, rb = m_mov.group(1), m_mov.group(2)
		if ra == rb:
			return None
		m_add = self._addq_imm_to_reg_re.match(lines[idx + 1])
		if m_add is None:
			return None
		imm = int(m_add.group(1))
		add_dst = m_add.group(2)
		if add_dst != rb:
			return None
		if not (-2147483648 <= imm <= 2147483647):
			return None
		if not self._is_reg_dead_in_window(lines, idx + 2, ra):
			return None
		return [f"\tleaq {imm}({ra}), {rb}"]

	def _try_mov_addreg_to_lea(self, lines: list[str], idx: int) -> list[str] | None:
		"""Fold 'movq %rA, %rB; addq %rC, %rB' into 'leaq (%rA,%rC), %rB'.

		Only fires when all three registers are distinct and %rA is dead after.
		"""
		m_mov = self._movq_reg_re.match(lines[idx])
		if m_mov is None:
			return None
		ra, rb = m_mov.group(1), m_mov.group(2)
		if ra == rb:
			return None
		m_add = self._addq_reg_reg_re.match(lines[idx + 1])
		if m_add is None:
			return None
		rc, add_dst = m_add.group(1), m_add.group(2)
		if add_dst != rb:
			return None
		if rc == rb:
			return None
		if not self._is_reg_dead_in_window(lines, idx + 2, ra):
			return None
		return [f"\tleaq ({ra},{rc}), {rb}"]

	def _try_redundant_test_after_arith(self, line1: str, line2: str) -> list[str] | None:
		"""Eliminate testq %reg, %reg after flag-setting arithmetic on same register.

		add/sub/and/or/xor already set ZF/SF, so a following testq is redundant.
		"""
		m_arith = self._arith_flag_setting_re.match(line1)
		if m_arith is None:
			return None
		m_test = self._testq_self_re.match(line2)
		if m_test is None:
			return None
		if m_arith.group(2) == m_test.group(1):
			return [line1]
		return None

	def _try_redundant_cmp_after_unary(self, line1: str, line2: str) -> list[str] | None:
		"""Eliminate cmpq $0 or testq after negq/incq/decq on same register.

		These single-operand instructions set ZF/SF based on the result.
		"""
		reg = None
		for pat in (self._negq_re, self._incq_re, self._decq_re):
			m = pat.match(line1)
			if m is not None:
				reg = m.group(1)
				break
		if reg is None:
			return None
		# Check for cmpq $0, %reg
		m_cmp = self._cmpq_zero_re.match(line2)
		if m_cmp is not None and m_cmp.group(1) == reg:
			return [line1]
		# Check for testq %reg, %reg
		m_test = self._testq_self_re.match(line2)
		if m_test is not None and m_test.group(1) == reg:
			return [line1]
		return None

	def _try_redundant_cmp_after_shift(self, line1: str, line2: str) -> list[str] | None:
		"""Eliminate cmpq $0 or testq after shift on same register.

		shlq/shrq/sarq set ZF/SF based on the result.
		"""
		m_shift = self._shift_re.match(line1)
		if m_shift is None:
			return None
		reg = m_shift.group(2)
		m_cmp = self._cmpq_zero_re.match(line2)
		if m_cmp is not None and m_cmp.group(1) == reg:
			return [line1]
		m_test = self._testq_self_re.match(line2)
		if m_test is not None and m_test.group(1) == reg:
			return [line1]
		return None

	def _try_branch_to_cmov(self, lines: list[str], idx: int) -> tuple[list[str], int] | None:
		"""Convert simple conditional branch pattern to cmov.

		Pattern (5 lines):
		  jCC .Lfalse
		  movq %rA, %rDst       (or movq mem, %rDst)
		  jmp .Lend
		  .Lfalse:
		  movq %rB, %rDst       (or movq mem, %rDst)

		Becomes:
		  movq %rB, %rDst
		  cmovINV_CC %rA, %rDst

		Only applies when the true-branch source is a register (cmov requires reg/mem src).
		The false-branch can be reg or memory.
		"""
		# Line 0: jCC .Lfalse
		m_jcc = self._jcc_re.match(lines[idx])
		if m_jcc is None:
			return None
		cc = m_jcc.group(1)
		false_label = m_jcc.group(2)

		# Line 1: movq src_true, %rDst
		m_true_mov = re.match(r"^\tmovq\s+([^,]+),\s+(%\w+)$", lines[idx + 1])
		if m_true_mov is None:
			return None
		src_true = m_true_mov.group(1).strip()
		dst = m_true_mov.group(2)

		# Line 2: jmp .Lend
		m_jmp = self._jmp_re.match(lines[idx + 2])
		if m_jmp is None:
			return None
		end_label = m_jmp.group(1)

		# Line 3: .Lfalse:
		m_flabel = self._label_re.match(lines[idx + 3])
		if m_flabel is None or m_flabel.group(1) != false_label:
			return None

		# Line 4: movq src_false, %rDst
		m_false_mov = re.match(r"^\tmovq\s+([^,]+),\s+(%\w+)$", lines[idx + 4])
		if m_false_mov is None:
			return None
		src_false = m_false_mov.group(1).strip()
		false_dst = m_false_mov.group(2)
		if false_dst != dst:
			return None

		# cmov requires register or memory source, not immediate
		if src_true.startswith("$"):
			return None

		# Check if end_label follows immediately after (line idx+5)
		if idx + 5 < len(lines):
			m_elabel = self._label_re.match(lines[idx + 5])
			if m_elabel is not None and m_elabel.group(1) == end_label:
				# Full 6-line pattern: consume all 6 lines
				# The condition was: if CC, jump to false_label (skip true branch)
				# So the true branch executes when CC is NOT met
				# cmov should apply the true value when CC is NOT met
				# i.e., cmovINV_CC src_true, %rDst (inverted condition)
				cmov_cc = self._jcc_to_cmov.get(cc)
				if cmov_cc is None:
					return None
				result = []
				# First load the false value (default)
				if src_false.startswith("$"):
					# Can't use cmov with immediate for false path either,
					# but we can load it with movq
					result.append(f"\tmovq {src_false}, {dst}")
				else:
					result.append(f"\tmovq {src_false}, {dst}")
				# Then conditionally overwrite with true value
				result.append(f"\t{cmov_cc}q {src_true}, {dst}")
				# Keep the end label (other code may jump to it)
				result.append(lines[idx + 5])
				return (result, 6)

		return None

	# --- Register name mapping helpers for extension patterns ---

	# Map from 32-bit register to its 8-bit low counterpart
	_reg32_to_reg8 = {
		"%eax": "%al", "%ebx": "%bl", "%ecx": "%cl", "%edx": "%dl",
		"%esi": "%sil", "%edi": "%dil", "%r8d": "%r8b", "%r9d": "%r9b",
		"%r10d": "%r10b", "%r11d": "%r11b", "%r12d": "%r12b",
		"%r13d": "%r13b", "%r14d": "%r14b", "%r15d": "%r15b",
	}

	# Map from 32-bit register to its 16-bit counterpart
	_reg32_to_reg16 = {
		"%eax": "%ax", "%ebx": "%bx", "%ecx": "%cx", "%edx": "%dx",
		"%esi": "%si", "%edi": "%di", "%r8d": "%r8w", "%r9d": "%r9w",
		"%r10d": "%r10w", "%r11d": "%r11w", "%r12d": "%r12w",
		"%r13d": "%r13w", "%r14d": "%r14w", "%r15d": "%r15w",
	}

	# Map from 8-bit register to its 32-bit counterpart
	_reg8_to_reg32 = {v: k for k, v in _reg32_to_reg8.items()}

	# Map from 16-bit register to its 32-bit counterpart
	_reg16_to_reg32 = {v: k for k, v in _reg32_to_reg16.items()}

	def _regs_same_physical(self, r1: str, r2: str) -> bool:
		"""Check if two register names refer to the same physical register."""
		if r1 == r2:
			return True
		# Normalize both to 32-bit form for comparison
		n1 = self._reg8_to_reg32.get(r1, self._reg16_to_reg32.get(r1, r1))
		n2 = self._reg8_to_reg32.get(r2, self._reg16_to_reg32.get(r2, r2))
		return n1 == n2

	def _try_duplicate_extension(self, line1: str, line2: str) -> list[str] | None:
		"""Remove duplicate extension where the same register is extended twice.

		e.g. movzbl %al, %eax; movzbl %al, %eax -> movzbl %al, %eax
		Also handles case where first extension writes to dest, second re-extends same.
		"""
		m1 = self._ext_re.match(line1)
		if m1 is None:
			return None
		m2 = self._ext_re.match(line2)
		if m2 is None:
			return None
		# Exact duplicate
		if line1 == line2:
			return [line1]
		# Same opcode, same dest, second src is sub-register of first dest
		op1, _, dst1 = m1.group(1), m1.group(2), m1.group(3)
		op2, src2, dst2 = m2.group(1), m2.group(2), m2.group(3)
		if op1 == op2 and dst1 == dst2 and self._regs_same_physical(src2, dst1):
			return [line1]
		return None

	def _try_ext_after_trunc(self, line1: str, line2: str) -> list[str] | None:
		"""Remove extension immediately after truncation to the same width.

		e.g. movb %al, %cl; movzbl %cl, %ecx -> movzbl %al, %ecx
		e.g. movw %ax, %cx; movzwl %cx, %ecx -> movzwl %ax, %ecx
		"""
		m_trunc = self._trunc_re.match(line1)
		if m_trunc is None:
			return None
		trunc_size = m_trunc.group(1)  # 'b' or 'w'
		trunc_src = m_trunc.group(2)
		trunc_dst = m_trunc.group(3)

		m_ext = self._ext_re.match(line2)
		if m_ext is None:
			return None
		ext_op = m_ext.group(1)
		ext_src = m_ext.group(2)
		ext_dst = m_ext.group(3)

		# The extension source must be the truncation destination
		if ext_src != trunc_dst:
			return None

		# Match truncation width to extension type
		if trunc_size == "b" and ext_op in ("movzbl", "movsbl"):
			# Replace with single extension from original source
			return [f"\t{ext_op} {trunc_src}, {ext_dst}"]
		if trunc_size == "w" and ext_op in ("movzwl", "movswl"):
			return [f"\t{ext_op} {trunc_src}, {ext_dst}"]
		return None

	def _try_movzbl_testl_fold(self, line1: str, line2: str) -> list[str] | None:
		"""Fold movzbl %al, %eax + testl %eax, %eax -> testb %al, %al.

		Zero-extending and testing the 32-bit result is equivalent to
		testing the original byte register directly.
		"""
		m_ext = self._ext_re.match(line1)
		if m_ext is None:
			return None
		if m_ext.group(1) != "movzbl":
			return None
		ext_src = m_ext.group(2)
		ext_dst = m_ext.group(3)

		m_test = self._testl_self_re.match(line2)
		if m_test is None:
			return None
		test_reg = m_test.group(1)
		if test_reg != ext_dst:
			return None
		return [f"\ttestb {ext_src}, {ext_src}"]

	def _try_redundant_ext_after_zero_ext_op(self, line1: str, line2: str) -> list[str] | None:
		"""Remove movzbl when source was already zero-extended by a prior 32-bit operation.

		Any 32-bit operation (movl, addl, xorl, etc.) implicitly zero-extends its
		result to 64 bits. A subsequent movzbl on the low byte of that result is
		redundant if the destination is the same register.

		Also handles andl with a byte mask followed by movzbl.
		"""
		m_ext = self._ext_re.match(line2)
		if m_ext is None:
			return None
		ext_op = m_ext.group(1)
		ext_src = m_ext.group(2)
		ext_dst = m_ext.group(3)

		# Check for 32-bit operation on the same register
		m_op = self._zero_ext_op_re.match(line1)
		if m_op is not None:
			op_dst = m_op.group(2)
			# The extension source must be the low sub-register of the 32-bit op dest
			if ext_op == "movzbl" and self._regs_same_physical(ext_src, op_dst) and op_dst == ext_dst:
				return [line1]
			# movzwl is also redundant after a 32-bit op
			if ext_op == "movzwl" and self._regs_same_physical(ext_src, op_dst) and op_dst == ext_dst:
				return [line1]

		# andl with byte-range mask already zeros upper bits
		m_and = self._andl_imm_re.match(line1)
		if m_and is not None:
			mask = int(m_and.group(1))
			and_dst = m_and.group(2)
			if ext_op == "movzbl" and mask <= 0xFF and self._regs_same_physical(ext_src, and_dst) and and_dst == ext_dst:
				return [line1]

		return None

	def _is_reg_dead_in_window(self, lines: list[str], start_idx: int, reg: str) -> bool:
		"""Check if reg is dead (overwritten before read) in a forward window."""
		for i in range(start_idx, len(lines)):
			line = lines[i]
			stripped = line.strip()
			if not stripped or stripped.endswith(":"):
				return False
			# At ret, only %rax is live (return value register)
			if stripped.startswith("ret"):
				return reg != "%rax"
			if stripped.startswith((
				"jmp", "je", "jne", "jg", "jge", "jl", "jle",
				"ja", "jae", "jb", "jbe", "js", "jns", "jo", "jno",
				"call", "syscall",
			)):
				return False
			if reg not in line:
				continue
			# Check if reg is overwritten without being read first
			m = re.match(r"^\t(?:movq|leaq)\s+([^,]+),\s+(.+)$", line)
			if m and m.group(2).strip() == reg and reg not in m.group(1):
				return True
			m = re.match(r"^\tpopq\s+(.+)$", line)
			if m and m.group(1).strip() == reg:
				return True
			# reg is read (or both read and written)
			return False
		return False
