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
		self._self_move_re = re.compile(
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
				# Pattern 1: store-then-reload elimination
				opt = self._try_store_reload(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

				# Pattern 3: movq $0 + cmpq $0 -> xorq + testq
				opt = self._try_zero_cmp(lines[i], lines[i + 1])
				if opt is not None:
					result.extend(opt)
					changed = True
					i += 2
					continue

			# Pattern 2: self-move elimination
			if self._is_self_move(lines[i]):
				changed = True
				i += 1
				continue

			# Pattern 4: addq $0 / subq $0 elimination
			if self._is_noop_arith(lines[i]):
				changed = True
				i += 1
				continue

			result.append(lines[i])
			i += 1

		return result, changed

	def _try_store_reload(self, line1: str, line2: str) -> list[str] | None:
		"""Remove redundant store followed by immediate reload of same location.

		movq %rax, -8(%rbp)   ->  movq %rax, -8(%rbp)
		movq -8(%rbp), %rax   ->  (removed)
		"""
		store_m = self._movq_store_re.match(line1)
		if store_m is None:
			return None
		load_m = self._movq_load_re.match(line2)
		if load_m is None:
			return None

		store_reg = store_m.group(1)
		store_loc = store_m.group(2)
		load_loc = load_m.group(1)
		load_reg = load_m.group(2)

		if store_loc == load_loc and store_reg == load_reg:
			return [line1]
		return None

	def _try_zero_cmp(self, line1: str, line2: str) -> list[str] | None:
		"""Collapse movq $0, %reg + cmpq $0, %reg -> xorq %reg, %reg + testq %reg, %reg."""
		mov_m = self._movq_zero_re.match(line1)
		if mov_m is None:
			return None
		cmp_m = self._cmpq_zero_re.match(line2)
		if cmp_m is None:
			return None

		mov_reg = mov_m.group(1)
		cmp_reg = cmp_m.group(1)

		if mov_reg == cmp_reg:
			return [f"\txorq {mov_reg}, {mov_reg}", f"\ttestq {mov_reg}, {mov_reg}"]
		return None

	def _is_self_move(self, line: str) -> bool:
		"""Check for movq %reg, %reg (self-move)."""
		m = self._self_move_re.match(line)
		if m is None:
			return False
		return m.group(1) == m.group(2)

	def _is_noop_arith(self, line: str) -> bool:
		"""Check for addq $0, %reg or subq $0, %reg."""
		return self._addq_zero_re.match(line) is not None or self._subq_zero_re.match(line) is not None
