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
		self._leaq_rip_re = re.compile(
			r"^\tleaq\s+(\w+)\(%rip\),\s+(%\w+)$"
		)
		self._movq_indirect_re = re.compile(
			r"^\tmovq\s+\((%\w+)\),\s+(%\w+)$"
		)
		self._addq_imm_reg_re = re.compile(
			r"^\taddq\s+(%\w+),\s+(%\w+)$"
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

			# Self-move elimination
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
		"""Check for movq %reg, %reg (self-move)."""
		m = self._movq_reg_re.match(line)
		if m is None:
			return False
		return m.group(1) == m.group(2)

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
