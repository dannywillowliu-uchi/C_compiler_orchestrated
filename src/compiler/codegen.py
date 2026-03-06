"""x86-64 code generator from three-address IR (AT&T syntax, System V AMD64 ABI)."""

from __future__ import annotations

import re
import struct

from compiler.ir import (
	IRAddrOf,
	IRAlloc,
	IRBinOp,
	IRBulkCopy,
	IRCall,
	IRCondJump,
	IRConst,
	IRConvert,
	IRCopy,
	IRFloatConst,
	IRFunction,
	IRGlobalRef,
	IRGlobalVar,
	IRJump,
	IRLabelInstr,
	IRLoad,
	IRParam,
	IRProgram,
	IRReturn,
	IRStore,
	IRTemp,
	IRType,
	IRUnaryOp,
	IRVaArg,
	IRVaCopy,
	IRVaEnd,
	IRVaStart,
	IRValue,
)

# System V AMD64 ABI: first 6 integer/pointer argument registers
_ARG_REGS = ["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"]
# System V AMD64 ABI: first 8 float/double argument registers
_FLOAT_ARG_REGS = ["%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"]

_FLOAT_TYPES = {IRType.FLOAT, IRType.DOUBLE}


def _is_float(ir_type: IRType) -> bool:
	return ir_type in _FLOAT_TYPES


def _ss_sd(ir_type: IRType) -> str:
	"""Return 'ss' for float, 'sd' for double."""
	return "ss" if ir_type == IRType.FLOAT else "sd"


def _s_d(ir_type: IRType) -> str:
	"""Return 's' for float, 'd' for double (for ucomis/movaps etc.)."""
	return "s" if ir_type == IRType.FLOAT else "d"


_CALLEE_SAVED_ALLOC_REGS = ["%rbx", "%r12", "%r13", "%r14", "%r15"]

_GAS_ESCAPE_MAP = {
	"\n": "\\n",
	"\t": "\\t",
	"\0": "\\0",
	"\\": "\\\\",
	'"': '\\"',
	"\a": "\\a",
	"\b": "\\b",
	"\f": "\\f",
	"\r": "\\r",
	"\v": "\\v",
}


def _global_alignment(g: IRGlobalVar) -> int:
	"""Return the required alignment in bytes for a global variable."""
	if g.symbol_initializer is not None or g.string_label is not None:
		return 8
	if g.ir_type in (IRType.POINTER, IRType.LONG, IRType.DOUBLE):
		return 8
	if g.ir_type in (IRType.INT, IRType.FLOAT):
		return 4
	if g.ir_type == IRType.SHORT:
		return 2
	if g.total_size >= 8:
		return 8
	if g.total_size >= 4:
		return 4
	return 1


def _escape_for_gas(s: str) -> str:
	"""Escape a string for use in a GAS .asciz directive."""
	result: list[str] = []
	for ch in s:
		if ch in _GAS_ESCAPE_MAP:
			result.append(_GAS_ESCAPE_MAP[ch])
		elif 0x20 <= ord(ch) <= 0x7E:
			result.append(ch)
		else:
			result.append(f"\\{ord(ch):03o}")
	return "".join(result)


# ---------------------------------------------------------------------------
# Post-regalloc copy propagation on emitted assembly
# ---------------------------------------------------------------------------

# Map sub-register names to their 64-bit parent
_SUB_TO_64: dict[str, str] = {}
for _base, _subs in [
	("%rax", ["%eax", "%ax", "%al", "%ah"]),
	("%rbx", ["%ebx", "%bx", "%bl", "%bh"]),
	("%rcx", ["%ecx", "%cx", "%cl", "%ch"]),
	("%rdx", ["%edx", "%dx", "%dl", "%dh"]),
	("%rsi", ["%esi", "%si", "%sil"]),
	("%rdi", ["%edi", "%di", "%dil"]),
	("%rbp", ["%ebp", "%bp", "%bpl"]),
	("%rsp", ["%esp", "%sp", "%spl"]),
	("%r8", ["%r8d", "%r8w", "%r8b"]),
	("%r9", ["%r9d", "%r9w", "%r9b"]),
	("%r10", ["%r10d", "%r10w", "%r10b"]),
	("%r11", ["%r11d", "%r11w", "%r11b"]),
	("%r12", ["%r12d", "%r12w", "%r12b"]),
	("%r13", ["%r13d", "%r13w", "%r13b"]),
	("%r14", ["%r14d", "%r14w", "%r14b"]),
	("%r15", ["%r15d", "%r15w", "%r15b"]),
]:
	_SUB_TO_64[_base] = _base
	for _s in _subs:
		_SUB_TO_64[_s] = _base

_MOVQ_REG_REG_RE = re.compile(r"^\tmovq\s+(%\w+),\s*(%\w+)$")
_GP_REG_64 = frozenset(_SUB_TO_64[k] for k in _SUB_TO_64)

# Caller-saved registers clobbered by a call instruction
_CALLER_SAVED = frozenset({"%rax", "%rcx", "%rdx", "%rsi", "%rdi", "%r8", "%r9", "%r10", "%r11"})

# Read-only instructions that do not write to their explicit register operands
# (pushq implicitly writes to %rsp, handled separately)
_READ_ONLY_MNEMONICS = frozenset({"cmpq", "cmpl", "cmpw", "cmpb", "testq", "testl", "testw", "testb",
	"pushq", "pushl", "pushw", "ucomiss", "ucomisd", "ucomis"})

# Registers excluded from copy propagation (frame/stack pointers)
_EXCLUDED_REGS = frozenset({"%rbp", "%rsp"})

# Regex to find any GP register reference in an operand
_ANY_GP_REG_RE = re.compile(r"%(?:r(?:ax|bx|cx|dx|si|di|bp|sp|8|9|1[0-5])|e(?:ax|bx|cx|dx|si|di|bp|sp)|"
	r"(?:ax|bx|cx|dx|si|di|bp|sp)|[abcd][lh]|sil|dil|bpl|spl|"
	r"r(?:8|9|1[0-5])[dwb])")


def _to_64(reg: str) -> str | None:
	"""Normalize a register name to its 64-bit parent, or None if not a GP reg."""
	return _SUB_TO_64.get(reg)


def _invalidate(copies: dict[str, str], reg64: str) -> None:
	"""Remove all copy entries involving reg64 as source or destination."""
	copies.pop(reg64, None)
	to_del = [k for k, v in copies.items() if v == reg64]
	for k in to_del:
		del copies[k]


def _resolve(copies: dict[str, str], reg: str) -> str:
	"""Follow the copy chain to find the original source register."""
	seen: set[str] = set()
	while reg in copies and reg not in seen:
		seen.add(reg)
		reg = copies[reg]
	return reg


def _written_reg64(line: str) -> str | None:
	"""Heuristically determine which 64-bit GP register is written by an instruction.

	Returns None if no GP register is written or if it cannot be determined.
	In AT&T syntax, the destination is the last operand for most instructions.
	"""
	stripped = line.lstrip("\t ")
	if not stripped or stripped.startswith(".") or stripped.startswith("#") or stripped.endswith(":"):
		return None

	parts = stripped.split(None, 1)
	if len(parts) < 2:
		# Single-operand instructions like negq %rax, popq %rax, incq %rax
		if len(parts) == 1:
			return None
		return None

	mnemonic = parts[0]

	# Read-only instructions
	if mnemonic in _READ_ONLY_MNEMONICS:
		return None

	operands_str = parts[1]
	operands = operands_str.split(",")
	last_op = operands[-1].strip()

	# Check for GP register in the last operand (destination in AT&T syntax)
	m = _ANY_GP_REG_RE.search(last_op)
	if m:
		return _to_64(m.group())
	return None


def copy_propagate_asm(lines: list[str]) -> list[str]:
	"""Perform copy propagation on assembly lines to eliminate redundant movq reg,reg."""
	copies: dict[str, str] = {}  # dst_reg64 -> src_reg64
	result: list[str] = []

	for line in lines:
		# Labels: reset all copy tracking (potential branch target)
		if line and not line[0].isspace() and line.rstrip().endswith(":"):
			copies.clear()
			result.append(line)
			continue

		stripped = line.lstrip("\t ")

		# Control flow: handle jumps and ret
		if stripped.startswith(("jmp ", "je ", "jne ", "jl ", "jg ", "jle ", "jge ",
			"ja ", "jb ", "jae ", "jbe ", "js ", "jns ", "ret")):
			copies.clear()
			result.append(line)
			continue

		# Call: clobber caller-saved registers
		if stripped.startswith("call ") or stripped.startswith("call\t"):
			for reg in _CALLER_SAVED:
				_invalidate(copies, reg)
			result.append(line)
			continue

		# Try to match movq %reg, %reg
		m = _MOVQ_REG_REG_RE.match(line)
		if m:
			src_str, dst_str = m.group(1), m.group(2)
			src64 = _to_64(src_str)
			dst64 = _to_64(dst_str)

			if src64 and dst64 and src64 != dst64:
				# Skip frame/stack pointer registers
				if src64 in _EXCLUDED_REGS or dst64 in _EXCLUDED_REGS:
					result.append(line)
					continue

				# Both are GP registers
				resolved = _resolve(copies, src64)

				if resolved == dst64:
					# After resolution, this is a self-move -- eliminate
					continue

				# Invalidate old copies involving dst
				_invalidate(copies, dst64)
				# Record the new copy
				copies[dst64] = resolved

				if resolved != src64:
					# Rewrite to use the resolved source
					result.append(f"\tmovq {resolved}, {dst_str}")
				else:
					result.append(line)
				continue
			elif src64 and dst64 and src64 == dst64:
				# Skip excluded registers (don't eliminate frame pointer self-moves)
				if src64 in _EXCLUDED_REGS:
					result.append(line)
					continue
				# Self-move already -- eliminate
				continue

		# For all other instructions, invalidate the written register
		written = _written_reg64(line)
		if written:
			_invalidate(copies, written)

		# pushq/popq implicitly modify %rsp
		if stripped.startswith(("pushq ", "pushl ", "pushw ")):
			_invalidate(copies, "%rsp")

		# Also handle single-operand writes: negq, notq, incq, decq, popq, etc.
		if stripped.startswith(("negq ", "notq ", "incq ", "decq ", "popq ",
			"salq ", "sarq ", "shrq ", "shlq ")):
			parts = stripped.split(None, 1)
			if len(parts) == 2:
				last_part = parts[1].split(",")[-1].strip()
				gm = _ANY_GP_REG_RE.search(last_part)
				if gm:
					r64 = _to_64(gm.group())
					if r64:
						_invalidate(copies, r64)

		# setCC writes to a sub-register, invalidate the parent
		if stripped.startswith("set"):
			parts = stripped.split(None, 1)
			if len(parts) == 2:
				gm = _ANY_GP_REG_RE.search(parts[1])
				if gm:
					r64 = _to_64(gm.group())
					if r64:
						_invalidate(copies, r64)

		# movzbq, movsbq, movzwq, movswq, movslq write to the destination
		if stripped.startswith(("movzbq ", "movsbq ", "movzwq ", "movswq ", "movslq ",
			"movzbl ", "movsbl ")):
			parts = stripped.split(",")
			if len(parts) == 2:
				gm = _ANY_GP_REG_RE.search(parts[1])
				if gm:
					r64 = _to_64(gm.group())
					if r64:
						_invalidate(copies, r64)

		result.append(line)

	return result


# ---------------------------------------------------------------------------
# Post-regalloc dead code elimination on emitted assembly
# ---------------------------------------------------------------------------

# Mnemonic prefixes where the destination register is PURELY written (not also read).
# In AT&T syntax, the last operand is the dest. For these, the dest is only written.
_PURE_DEST_PREFIXES = (
	"mov", "lea", "set", "cvt",
)

# Mnemonics that are NOT safe to eliminate (side effects beyond register writes)
_SIDE_EFFECT_MNEMONICS = frozenset({
	"call", "pushq", "pushl", "pushw", "popq", "popl", "popw",
	"ret", "leave", "syscall", "int",
	"cmpq", "cmpl", "cmpw", "cmpb", "testq", "testl", "testw", "testb",
	"ucomiss", "ucomisd",
	"cqto", "cqo", "cdq", "cwd", "cbw",
	"divq", "divl", "idivq", "idivl",
	"rep", "repne",
})

_IS_LABEL_RE = re.compile(r"^[A-Za-z_.][A-Za-z0-9_.]*:$")


def _is_label(stripped: str) -> bool:
	"""Check if a stripped line is a label (including .L* labels)."""
	return bool(_IS_LABEL_RE.match(stripped))


def _is_pure_dest_write(line: str) -> tuple[str | None, bool]:
	"""Determine if an instruction purely writes a GP register (no read of dest).

	Returns (reg64, can_eliminate) where reg64 is the 64-bit parent of the written
	register and can_eliminate indicates if the instruction is safe to remove when
	the dest is dead.
	"""
	stripped = line.lstrip("\t ")
	if not stripped or stripped.startswith(".") or stripped.startswith("#") or _is_label(stripped):
		return None, False

	parts = stripped.split(None, 1)
	if not parts or len(parts) < 2:
		return None, False

	mnemonic = parts[0]

	if mnemonic in _SIDE_EFFECT_MNEMONICS:
		return None, False

	# Check for pure-dest-write mnemonics
	if not mnemonic.startswith(_PURE_DEST_PREFIXES):
		# Handle xor zeroing idiom: xorq %rax, %rax
		if mnemonic.startswith("xor"):
			operands = parts[1].split(",")
			if len(operands) == 2:
				src_op = operands[0].strip()
				dst_op = operands[1].strip()
				src64 = _to_64(src_op)
				dst64 = _to_64(dst_op)
				if src64 and dst64 and src64 == dst64 and "(" not in dst_op:
					return dst64, False  # It's a pure write (zeroing), but we track it as "next writes dest"
		return None, False

	operands = parts[1].split(",")
	last_op = operands[-1].strip()

	# Destination must be a plain register (not memory)
	if "(" in last_op:
		return None, False

	m = _ANY_GP_REG_RE.match(last_op)
	if m and m.group() == last_op:
		reg64 = _to_64(last_op)
		if reg64:
			return reg64, True
	return None, False


def _next_instr_purely_writes(reachable: list[str], start: int, reg64: str) -> bool:
	"""Check if the next meaningful instruction purely overwrites reg64 without reading it."""
	idx = start
	while idx < len(reachable):
		next_line = reachable[idx].lstrip("\t ")
		if not next_line or next_line.startswith("#"):
			idx += 1
			continue
		break
	else:
		return False

	next_stripped = reachable[idx].lstrip("\t ")

	# Labels, jumps, ret, call -- not a pure overwrite
	if _is_label(next_stripped) or next_stripped.startswith(("jmp", "je ", "jne ", "jl ", "jg ",
		"jle ", "jge ", "ja ", "jb ", "jae ", "jbe ", "js ", "jns ", "ret", "call")):
		return False

	# Check if it's a pure write to the same register
	next_dest, is_pure = _is_pure_dest_write(reachable[idx])
	if next_dest != reg64:
		return False

	if is_pure:
		# Pure dest write (mov, lea, set, cvt) -- check source operands don't reference the reg
		parts = next_stripped.split(None, 1)
		if len(parts) >= 2:
			operands = parts[1].split(",")
			# All operands except last are sources
			source_str = ",".join(operands[:-1])
			for reg_name, parent in _SUB_TO_64.items():
				if parent == reg64 and reg_name in source_str:
					return False
			# Also check memory references in dest operand
			dest_op = operands[-1].strip()
			if "(" in dest_op:
				for reg_name, parent in _SUB_TO_64.items():
					if parent == reg64 and reg_name in dest_op:
						return False
		return True

	# xor zeroing idiom -- the next instruction purely writes without needing old value
	parts = next_stripped.split(None, 1)
	if parts and parts[0].startswith("xor") and len(parts) >= 2:
		operands = parts[1].split(",")
		if len(operands) == 2:
			src64 = _to_64(operands[0].strip())
			dst64 = _to_64(operands[1].strip())
			if src64 == dst64 == reg64:
				return True

	return False


def dead_code_eliminate_asm(lines: list[str]) -> list[str]:
	"""Eliminate dead stores and unreachable code from assembly lines.

	1. Remove instructions after unconditional jumps (jmp) until the next label.
	2. Remove register writes that are immediately overwritten by the next instruction
	   purely writing to the same register (without reading it first).
	"""
	# Pass 1: Remove unreachable code after unconditional jumps
	reachable: list[str] = []
	in_dead_zone = False

	for line in lines:
		stripped = line.lstrip("\t ")

		# Labels (including .L* labels) end dead zones
		if _is_label(stripped):
			in_dead_zone = False
			reachable.append(line)
			continue

		# Keep important directives even in dead zones
		if stripped.startswith("."):
			if in_dead_zone:
				if stripped.startswith((".size", ".globl", ".type", ".section")):
					in_dead_zone = False
					reachable.append(line)
				continue
			reachable.append(line)
			continue

		if in_dead_zone:
			continue

		reachable.append(line)

		# Unconditional jump starts a dead zone
		if stripped.startswith("jmp ") or stripped.startswith("jmp\t"):
			in_dead_zone = True

	# Pass 2: Eliminate dead register stores (pure write immediately overwritten)
	result: list[str] = []

	for i, line in enumerate(reachable):
		dest, can_eliminate = _is_pure_dest_write(line)
		if dest and can_eliminate and dest not in _EXCLUDED_REGS:
			if _next_instr_purely_writes(reachable, i + 1, dest):
				continue

		result.append(line)

	return result


class CodeGenerator:
	"""Generates x86-64 assembly (AT&T syntax) from an IRProgram."""

	def __init__(self, regalloc_maps: dict[str, dict[str, str]] | None = None) -> None:
		self._lines: list[str] = []
		self._stack_map: dict[str, int] = {}
		self._stack_size: int = 0
		self._float_consts: list[tuple[str, float, IRType]] = []
		self._float_const_counter: int = 0
		self._regalloc_maps: dict[str, dict[str, str]] = regalloc_maps or {}
		self._reg_map: dict[str, str] = {}
		self._callee_save_offsets: list[tuple[str, int]] = []
		self._va_label_counter: int = 0
		self._reg_save_area_offset: int = 0

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------

	def generate(self, program: IRProgram) -> str:
		"""Generate assembly for an entire IR program."""
		self._lines = []
		self._float_consts = []
		self._float_const_counter = 0

		# Emit .data section for initialized globals
		initialized = [
			g for g in program.globals
			if g.initializer is not None or g.initializer_values
			or g.float_initializer is not None or g.string_label is not None
			or g.symbol_initializer is not None
		]
		if initialized:
			self._emit(".section .data")
			for g in initialized:
				align = _global_alignment(g)
				if align > 1:
					self._emit_instr(f".align {align}")
				if g.storage_class != "static":
					self._emit(f".globl {g.name}")
				self._emit(f"{g.name}:")
				if g.initializer_values:
					directive = {
						IRType.BOOL: ".byte",
						IRType.CHAR: ".byte",
						IRType.SHORT: ".word",
						IRType.INT: ".long",
					}.get(g.ir_type, ".quad")
					for val in g.initializer_values:
						self._emit_instr(f"{directive} {val}")
				elif g.float_initializer is not None:
					if g.ir_type == IRType.FLOAT:
						bits = struct.unpack("<I", struct.pack("<f", g.float_initializer))[0]
						self._emit_instr(f".long {bits}")
					else:
						bits = struct.unpack("<Q", struct.pack("<d", g.float_initializer))[0]
						self._emit_instr(f".quad {bits}")
				elif g.string_label is not None:
					self._emit_instr(f".quad {g.string_label}")
				elif g.symbol_initializer is not None:
					if g.symbol_initializer_offset:
						self._emit_instr(f".quad {g.symbol_initializer}+{g.symbol_initializer_offset}")
					else:
						self._emit_instr(f".quad {g.symbol_initializer}")
				elif g.ir_type in (IRType.CHAR, IRType.BOOL):
					self._emit_instr(f".byte {g.initializer}")
				elif g.ir_type == IRType.SHORT:
					self._emit_instr(f".word {g.initializer}")
				elif g.ir_type == IRType.FLOAT:
					bits = struct.unpack("<I", struct.pack("<f", float(g.initializer or 0)))[0]
					self._emit_instr(f".long {bits}")
				elif g.ir_type == IRType.DOUBLE:
					bits = struct.unpack("<Q", struct.pack("<d", float(g.initializer or 0)))[0]
					self._emit_instr(f".quad {bits}")
				elif g.ir_type == IRType.INT:
					self._emit_instr(f".long {g.initializer}")
				else:
					self._emit_instr(f".quad {g.initializer}")

		# Emit .rodata section for string literals
		if program.string_data:
			self._emit(".section .rodata")
			for s in program.string_data:
				self._emit(f"{s.label}:")
				self._emit_instr(f'.asciz "{_escape_for_gas(s.value)}"')

		# Emit .bss section for uninitialized globals (skip extern-only declarations)
		uninitialized = [
			g for g in program.globals
			if g.initializer is None and not g.initializer_values
			and g.float_initializer is None and g.string_label is None
			and g.symbol_initializer is None
			and g.storage_class != "extern"
		]
		if uninitialized:
			self._emit(".section .bss")
			for g in uninitialized:
				align = _global_alignment(g)
				if align > 1:
					self._emit_instr(f".align {align}")
				if g.storage_class != "static":
					self._emit(f".globl {g.name}")
				self._emit(f"{g.name}:")
				if g.total_size > 0:
					self._emit_instr(f".zero {g.total_size}")
				elif g.ir_type in (IRType.CHAR, IRType.BOOL):
					self._emit_instr(".zero 1")
				elif g.ir_type == IRType.SHORT:
					self._emit_instr(".zero 2")
				elif g.ir_type in (IRType.INT, IRType.FLOAT):
					self._emit_instr(".zero 4")
				else:
					self._emit_instr(".zero 8")

		# Emit .text section for functions
		self._emit(".section .text")
		for func in program.functions:
			self._generate_function(func)

		# Emit float constant pool in .rodata
		if self._float_consts:
			self._emit(".section .rodata")
			for label, value, ir_type in self._float_consts:
				if ir_type == IRType.FLOAT:
					self._emit("\t.align 4")
					self._emit(f"{label}:")
					bits = struct.unpack("<I", struct.pack("<f", value))[0]
					self._emit_instr(f".long {bits}")
				else:
					self._emit("\t.align 8")
					self._emit(f"{label}:")
					bits = struct.unpack("<Q", struct.pack("<d", value))[0]
					self._emit_instr(f".quad {bits}")

		# Post-regalloc copy propagation to eliminate redundant mov instructions
		self._lines = copy_propagate_asm(self._lines)

		# Dead code elimination: dead stores + unreachable code after jumps
		self._lines = dead_code_eliminate_asm(self._lines)

		return "\n".join(self._lines) + "\n"

	# ------------------------------------------------------------------
	# Helpers
	# ------------------------------------------------------------------

	def _emit(self, line: str) -> None:
		self._lines.append(line)

	def _emit_instr(self, instr: str) -> None:
		self._lines.append(f"\t{instr}")

	@staticmethod
	def _align16(n: int) -> int:
		"""Round *n* up to the nearest multiple of 16."""
		return (n + 15) & ~15

	def _get_offset(self, name: str) -> int:
		"""Return the stack offset (relative to %rbp) for a temp."""
		return self._stack_map[name]

	def _alloc_float_const(self, value: float, ir_type: IRType) -> str:
		"""Allocate a label for a float constant in .rodata."""
		label = f".LFC{self._float_const_counter}"
		self._float_const_counter += 1
		self._float_consts.append((label, value, ir_type))
		return label

	def _load_value(self, value: IRValue, reg: str) -> None:
		"""Emit code to load an IRValue into an integer *reg*."""
		if isinstance(value, IRConst):
			self._emit_instr(f"movq ${value.value}, {reg}")
		elif isinstance(value, IRFloatConst):
			# Load float const into integer register via stack (unusual but handles edge cases)
			label = self._alloc_float_const(value.value, value.ir_type)
			self._emit_instr(f"leaq {label}(%rip), {reg}")
		elif isinstance(value, IRTemp):
			if value.name in self._reg_map:
				allocated = self._reg_map[value.name]
				if allocated != reg:
					self._emit_instr(f"movq {allocated}, {reg}")
			else:
				offset = self._get_offset(value.name)
				self._emit_instr(f"movq {offset}(%rbp), {reg}")
		elif isinstance(value, IRGlobalRef):
			self._emit_instr(f"leaq {value.name}(%rip), {reg}")
		else:
			raise ValueError(f"Unsupported IRValue type: {type(value).__name__}")

	def _load_float_value(self, value: IRValue, xmm: str, ir_type: IRType) -> None:
		"""Emit code to load an IRValue into an XMM register."""
		suffix = _ss_sd(ir_type)
		if isinstance(value, IRFloatConst):
			label = self._alloc_float_const(value.value, value.ir_type)
			self._emit_instr(f"mov{suffix} {label}(%rip), {xmm}")
		elif isinstance(value, IRTemp):
			offset = self._get_offset(value.name)
			self._emit_instr(f"mov{suffix} {offset}(%rbp), {xmm}")
		elif isinstance(value, IRConst):
			# Load integer constant as float: move to GP reg, then to XMM via stack
			label = self._alloc_float_const(float(value.value), ir_type)
			self._emit_instr(f"mov{suffix} {label}(%rip), {xmm}")
		else:
			raise ValueError(f"Cannot load {type(value).__name__} into XMM register")

	def _truncate_narrow(self, ir_type: IRType, is_unsigned: bool = False) -> None:
		"""Emit truncation for narrow types (BOOL/CHAR/SHORT) after arithmetic.

		Ensures the value in %rax is correctly narrowed to the type's bit width.
		"""
		if ir_type == IRType.BOOL:
			self._emit_instr("movzbq %al, %rax")
		elif ir_type == IRType.CHAR:
			if is_unsigned:
				self._emit_instr("movzbq %al, %rax")
			else:
				self._emit_instr("movsbq %al, %rax")
		elif ir_type == IRType.SHORT:
			if is_unsigned:
				self._emit_instr("movzwq %ax, %rax")
			else:
				self._emit_instr("movswq %ax, %rax")

	def _store_to_temp(self, reg: str, dest: IRTemp) -> None:
		"""Store integer *reg* into *dest*'s location (register or stack slot)."""
		if dest.name in self._reg_map:
			allocated = self._reg_map[dest.name]
			if allocated != reg:
				self._emit_instr(f"movq {reg}, {allocated}")
		else:
			offset = self._get_offset(dest.name)
			self._emit_instr(f"movq {reg}, {offset}(%rbp)")

	def _store_float_to_temp(self, xmm: str, dest: IRTemp, ir_type: IRType) -> None:
		"""Store XMM register into *dest*'s stack slot."""
		offset = self._get_offset(dest.name)
		suffix = _ss_sd(ir_type)
		self._emit_instr(f"mov{suffix} {xmm}, {offset}(%rbp)")

	# ------------------------------------------------------------------
	# Temp allocation (scan phase)
	# ------------------------------------------------------------------

	def _allocate_temps(self, func: IRFunction) -> None:
		"""Scan *func* and assign a stack slot (-8, -16, ...) to every non-register-allocated IRTemp."""
		temps: set[str] = set()
		for p in func.params:
			temps.add(p.name)
		for instr in func.body:
			self._collect_temps_from_instr(instr, temps)

		# Only allocate stack slots for temps not in registers
		stack_temps = sorted(t for t in temps if t not in self._reg_map)

		self._stack_size = 0
		self._stack_map = {}
		self._callee_save_offsets = []
		self._reg_save_area_offset = 0

		# Reserve slots for callee-saved register saves
		used_callee = set(self._reg_map.values())
		for reg in _CALLEE_SAVED_ALLOC_REGS:
			if reg in used_callee:
				self._stack_size += 8
				self._callee_save_offsets.append((reg, -self._stack_size))

		# Reserve register save area for variadic functions (6 GP regs * 8 bytes = 48)
		if func.is_variadic:
			self._stack_size += 48
			self._reg_save_area_offset = -self._stack_size

		# Allocate spill slots
		for name in stack_temps:
			self._stack_size += 8
			self._stack_map[name] = -self._stack_size

	def _collect_temps_from_instr(self, instr: object, temps: set[str]) -> None:
		if isinstance(instr, IRBinOp):
			temps.add(instr.dest.name)
			self._collect_value_temp(instr.left, temps)
			self._collect_value_temp(instr.right, temps)
		elif isinstance(instr, IRUnaryOp):
			temps.add(instr.dest.name)
			self._collect_value_temp(instr.operand, temps)
		elif isinstance(instr, IRCopy):
			temps.add(instr.dest.name)
			self._collect_value_temp(instr.source, temps)
		elif isinstance(instr, IRLoad):
			temps.add(instr.dest.name)
			self._collect_value_temp(instr.address, temps)
		elif isinstance(instr, IRStore):
			self._collect_value_temp(instr.address, temps)
			self._collect_value_temp(instr.value, temps)
		elif isinstance(instr, IRCall):
			if instr.dest is not None:
				temps.add(instr.dest.name)
			for arg in instr.args:
				self._collect_value_temp(arg, temps)
			if instr.func_value is not None:
				self._collect_value_temp(instr.func_value, temps)
		elif isinstance(instr, IRReturn):
			if instr.value is not None:
				self._collect_value_temp(instr.value, temps)
		elif isinstance(instr, IRParam):
			self._collect_value_temp(instr.value, temps)
		elif isinstance(instr, IRAddrOf):
			temps.add(instr.dest.name)
			temps.add(instr.source.name)
		elif isinstance(instr, IRAlloc):
			temps.add(instr.dest.name)
		elif isinstance(instr, IRBulkCopy):
			self._collect_value_temp(instr.dest_addr, temps)
			self._collect_value_temp(instr.src_addr, temps)
		elif isinstance(instr, IRCondJump):
			self._collect_value_temp(instr.condition, temps)
		elif isinstance(instr, IRConvert):
			temps.add(instr.dest.name)
			self._collect_value_temp(instr.source, temps)
		elif isinstance(instr, IRVaStart):
			self._collect_value_temp(instr.ap_addr, temps)
		elif isinstance(instr, IRVaArg):
			temps.add(instr.dest.name)
			self._collect_value_temp(instr.ap_addr, temps)
		elif isinstance(instr, IRVaEnd):
			self._collect_value_temp(instr.ap_addr, temps)
		elif isinstance(instr, IRVaCopy):
			self._collect_value_temp(instr.dest_addr, temps)
			self._collect_value_temp(instr.src_addr, temps)

	@staticmethod
	def _collect_value_temp(value: IRValue, temps: set[str]) -> None:
		if isinstance(value, IRTemp):
			temps.add(value.name)

	# ------------------------------------------------------------------
	# Function-level generation
	# ------------------------------------------------------------------

	def _emit_epilogue(self) -> None:
		"""Emit function epilogue: restore callee-saved regs, tear down frame, return."""
		for reg, offset in reversed(self._callee_save_offsets):
			self._emit_instr(f"movq {offset}(%rbp), {reg}")
		self._emit_instr("movq %rbp, %rsp")
		self._emit_instr("popq %rbp")
		self._emit_instr("ret")

	def _generate_function(self, func: IRFunction) -> None:
		# Skip prototype-only and extern declarations -- no body to emit
		if func.is_prototype or func.storage_class == "extern":
			return

		# Set register allocation map for this function
		self._reg_map = self._regalloc_maps.get(func.name, {})

		self._allocate_temps(func)

		frame_size = self._align16(self._stack_size)

		# Header: only emit .globl for non-static functions
		if func.storage_class != "static":
			self._emit(f".globl {func.name}")
		self._emit(f".type {func.name}, @function")
		self._emit(f"{func.name}:")

		# Prologue
		self._emit_instr("pushq %rbp")
		self._emit_instr("movq %rsp, %rbp")
		if frame_size > 0:
			self._emit_instr(f"subq ${frame_size}, %rsp")

		# Save callee-saved registers used by regalloc
		for reg, offset in self._callee_save_offsets:
			self._emit_instr(f"movq {reg}, {offset}(%rbp)")

		# Save all GP registers to register save area for variadic functions
		if func.is_variadic:
			for i, reg in enumerate(_ARG_REGS):
				offset = self._reg_save_area_offset + i * 8
				self._emit_instr(f"movq {reg}, {offset}(%rbp)")

		# Copy register-passed params to their destinations
		int_idx = 0
		float_idx = 0
		for i, param in enumerate(func.params):
			param_type = func.param_types[i] if i < len(func.param_types) else IRType.INT
			if _is_float(param_type):
				if float_idx < len(_FLOAT_ARG_REGS):
					offset = self._get_offset(param.name)
					suffix = _ss_sd(param_type)
					self._emit_instr(f"mov{suffix} {_FLOAT_ARG_REGS[float_idx]}, {offset}(%rbp)")
				float_idx += 1
			else:
				if param.name in self._reg_map:
					allocated = self._reg_map[param.name]
					if int_idx < len(_ARG_REGS):
						if _ARG_REGS[int_idx] != allocated:
							self._emit_instr(f"movq {_ARG_REGS[int_idx]}, {allocated}")
					else:
						src_offset = 16 + (int_idx - len(_ARG_REGS)) * 8
						self._emit_instr(f"movq {src_offset}(%rbp), {allocated}")
				else:
					offset = self._get_offset(param.name)
					if int_idx < len(_ARG_REGS):
						self._emit_instr(f"movq {_ARG_REGS[int_idx]}, {offset}(%rbp)")
					else:
						src_offset = 16 + (int_idx - len(_ARG_REGS)) * 8
						self._emit_instr(f"movq {src_offset}(%rbp), %rax")
						self._emit_instr(f"movq %rax, {offset}(%rbp)")
				int_idx += 1

		# Body
		for instr in func.body:
			self._generate_instruction(instr)

		# Implicit epilogue
		last_is_return = func.body and isinstance(func.body[-1], IRReturn)
		if not last_is_return:
			if func.return_type != IRType.VOID:
				if _is_float(func.return_type):
					self._emit_instr("xorps %xmm0, %xmm0")
				else:
					self._emit_instr("movq $0, %rax")
			self._emit_epilogue()

		# ELF .size directive
		self._emit(f".size {func.name}, .-{func.name}")

	# ------------------------------------------------------------------
	# Instruction dispatch
	# ------------------------------------------------------------------

	def _generate_instruction(self, instr: object) -> None:
		if isinstance(instr, IRBinOp):
			if _is_float(instr.ir_type):
				self._gen_float_binop(instr)
			else:
				self._gen_binop(instr)
		elif isinstance(instr, IRUnaryOp):
			if _is_float(instr.ir_type):
				self._gen_float_unaryop(instr)
			else:
				self._gen_unaryop(instr)
		elif isinstance(instr, IRCopy):
			if _is_float(instr.ir_type):
				self._gen_float_copy(instr)
			else:
				self._gen_copy(instr)
		elif isinstance(instr, IRLoad):
			self._gen_load(instr)
		elif isinstance(instr, IRStore):
			if _is_float(instr.ir_type):
				self._gen_float_store(instr)
			else:
				self._gen_store(instr)
		elif isinstance(instr, IRLabelInstr):
			self._emit(f"{instr.name}:")
		elif isinstance(instr, IRJump):
			self._emit_instr(f"jmp {instr.target}")
		elif isinstance(instr, IRCondJump):
			self._gen_condjump(instr)
		elif isinstance(instr, IRCall):
			self._gen_call(instr)
		elif isinstance(instr, IRReturn):
			if _is_float(instr.ir_type):
				self._gen_float_return(instr)
			else:
				self._gen_return(instr)
		elif isinstance(instr, IRAddrOf):
			self._gen_addr_of(instr)
		elif isinstance(instr, IRAlloc):
			self._gen_alloc(instr)
		elif isinstance(instr, IRBulkCopy):
			self._gen_bulk_copy(instr)
		elif isinstance(instr, IRConvert):
			self._gen_convert(instr)
		elif isinstance(instr, IRParam):
			pass  # args are conveyed via IRCall.args; IRParam is a no-op here
		elif isinstance(instr, IRVaStart):
			self._gen_va_start(instr)
		elif isinstance(instr, IRVaArg):
			self._gen_va_arg(instr)
		elif isinstance(instr, IRVaEnd):
			pass  # va_end is a no-op on x86-64
		elif isinstance(instr, IRVaCopy):
			self._gen_va_copy(instr)
		else:
			raise ValueError(f"Unknown instruction type: {type(instr).__name__}")

	# ------------------------------------------------------------------
	# Integer instruction generators
	# ------------------------------------------------------------------

	def _gen_binop(self, instr: IRBinOp) -> None:
		self._load_value(instr.left, "%rax")
		self._load_value(instr.right, "%rcx")

		op = instr.op
		unsigned = instr.is_unsigned
		if op == "+":
			self._emit_instr("addq %rcx, %rax")
		elif op == "-":
			self._emit_instr("subq %rcx, %rax")
		elif op == "*":
			self._emit_instr("imulq %rcx, %rax")
		elif op == "/":
			if unsigned:
				self._emit_instr("xorq %rdx, %rdx")
				self._emit_instr("divq %rcx")
			else:
				self._emit_instr("cqto")
				self._emit_instr("idivq %rcx")
		elif op == "%":
			if unsigned:
				self._emit_instr("xorq %rdx, %rdx")
				self._emit_instr("divq %rcx")
			else:
				self._emit_instr("cqto")
				self._emit_instr("idivq %rcx")
			self._emit_instr("movq %rdx, %rax")
		elif op in ("<", ">", "<=", ">=", "==", "!="):
			self._emit_instr("cmpq %rcx, %rax")
			if unsigned:
				setcc = {"<": "setb", ">": "seta", "<=": "setbe", ">=": "setae", "==": "sete", "!=": "setne"}[op]
			else:
				setcc = {"<": "setl", ">": "setg", "<=": "setle", ">=": "setge", "==": "sete", "!=": "setne"}[op]
			self._emit_instr(f"{setcc} %al")
			self._emit_instr("movzbq %al, %rax")
		elif op == "&":
			self._emit_instr("andq %rcx, %rax")
		elif op == "|":
			self._emit_instr("orq %rcx, %rax")
		elif op == "^":
			self._emit_instr("xorq %rcx, %rax")
		elif op == "<<":
			self._emit_instr("salq %cl, %rax")
		elif op == ">>":
			if unsigned:
				self._emit_instr("shrq %cl, %rax")
			else:
				self._emit_instr("sarq %cl, %rax")
		else:
			raise ValueError(f"Unknown binary operator: {op}")

		if op not in ("<", ">", "<=", ">=", "==", "!="):
			self._truncate_narrow(instr.ir_type, instr.is_unsigned)
		self._store_to_temp("%rax", instr.dest)

	def _gen_unaryop(self, instr: IRUnaryOp) -> None:
		self._load_value(instr.operand, "%rax")

		op = instr.op
		if op == "-":
			self._emit_instr("negq %rax")
		elif op == "~":
			self._emit_instr("notq %rax")
		elif op == "!":
			self._emit_instr("cmpq $0, %rax")
			self._emit_instr("sete %al")
			self._emit_instr("movzbq %al, %rax")
		else:
			raise ValueError(f"Unknown unary operator: {op}")

		if op != "!":
			self._truncate_narrow(instr.ir_type)
		self._store_to_temp("%rax", instr.dest)

	def _gen_copy(self, instr: IRCopy) -> None:
		# Optimize register-to-register copies: skip %rax intermediary
		if isinstance(instr.source, IRTemp) and instr.source.name in self._reg_map and instr.dest.name in self._reg_map:
			src_reg = self._reg_map[instr.source.name]
			dst_reg = self._reg_map[instr.dest.name]
			if src_reg == dst_reg:
				return  # Coalesced: same register, no-op
			self._emit_instr(f"movq {src_reg}, {dst_reg}")
			return
		self._load_value(instr.source, "%rax")
		self._store_to_temp("%rax", instr.dest)

	def _gen_load(self, instr: IRLoad) -> None:
		self._load_value(instr.address, "%rax")
		if _is_float(instr.ir_type):
			suffix = _ss_sd(instr.ir_type)
			self._emit_instr(f"mov{suffix} (%rax), %xmm0")
			self._store_float_to_temp("%xmm0", instr.dest, instr.ir_type)
		elif instr.ir_type == IRType.BOOL:
			self._emit_instr("movzbl (%rax), %eax")
			self._store_to_temp("%rax", instr.dest)
		elif instr.ir_type == IRType.CHAR:
			if instr.is_unsigned:
				self._emit_instr("movzbl (%rax), %eax")
			else:
				self._emit_instr("movsbl (%rax), %eax")
				self._emit_instr("movslq %eax, %rax")
			self._store_to_temp("%rax", instr.dest)
		elif instr.ir_type == IRType.SHORT:
			if instr.is_unsigned:
				self._emit_instr("movzwl (%rax), %eax")
			else:
				self._emit_instr("movswq (%rax), %rax")
			self._store_to_temp("%rax", instr.dest)
		elif instr.ir_type == IRType.INT:
			self._emit_instr("movl (%rax), %eax")
			if not instr.is_unsigned:
				self._emit_instr("movslq %eax, %rax")
			self._store_to_temp("%rax", instr.dest)
		else:
			self._emit_instr("movq (%rax), %rax")
			self._store_to_temp("%rax", instr.dest)

	def _gen_store(self, instr: IRStore) -> None:
		self._load_value(instr.value, "%rcx")
		self._load_value(instr.address, "%rax")
		if instr.ir_type in (IRType.CHAR, IRType.BOOL):
			self._emit_instr("movb %cl, (%rax)")
		elif instr.ir_type == IRType.SHORT:
			self._emit_instr("movw %cx, (%rax)")
		elif instr.ir_type == IRType.INT:
			self._emit_instr("movl %ecx, (%rax)")
		else:
			self._emit_instr("movq %rcx, (%rax)")

	def _gen_condjump(self, instr: IRCondJump) -> None:
		self._load_value(instr.condition, "%rax")
		self._emit_instr("cmpq $0, %rax")
		self._emit_instr(f"jne {instr.true_label}")
		self._emit_instr(f"jmp {instr.false_label}")

	def _gen_call(self, instr: IRCall) -> None:
		# Separate args into integer and float based on arg_types
		int_args: list[tuple[int, IRValue]] = []
		float_args: list[tuple[int, IRValue, IRType]] = []
		for i, arg in enumerate(instr.args):
			arg_type = instr.arg_types[i] if i < len(instr.arg_types) else IRType.INT
			if _is_float(arg_type):
				float_args.append((i, arg, arg_type))
			else:
				int_args.append((i, arg))

		# Handle stack args for integer args beyond register count
		stack_int_args = int_args[len(_ARG_REGS):]
		num_stack_args = len(stack_int_args)

		needs_padding = num_stack_args > 0 and num_stack_args % 2 != 0
		if needs_padding:
			self._emit_instr("subq $8, %rsp")

		for _, arg in reversed(stack_int_args):
			self._load_value(arg, "%rax")
			self._emit_instr("pushq %rax")

		# Load float register args
		for idx, (_, arg, arg_type) in enumerate(float_args):
			if idx < len(_FLOAT_ARG_REGS):
				self._load_float_value(arg, _FLOAT_ARG_REGS[idx], arg_type)

		# Load integer register args
		for idx, (_, arg) in enumerate(int_args[:len(_ARG_REGS)]):
			self._load_value(arg, _ARG_REGS[idx])

		# ABI requirement: %al = number of SSE register args (needed for variadic functions)
		num_sse_args = min(len(float_args), len(_FLOAT_ARG_REGS))
		self._emit_instr(f"movb ${num_sse_args}, %al")

		if instr.indirect and instr.func_value is not None:
			self._load_value(instr.func_value, "%r11")
			self._emit_instr("call *%r11")
		else:
			self._emit_instr(f"call {instr.function_name}")

		if num_stack_args > 0:
			cleanup = num_stack_args * 8
			if needs_padding:
				cleanup += 8
			self._emit_instr(f"addq ${cleanup}, %rsp")

		if instr.dest is not None:
			if _is_float(instr.return_type):
				self._store_float_to_temp("%xmm0", instr.dest, instr.return_type)
			else:
				self._store_to_temp("%rax", instr.dest)

	def _gen_return(self, instr: IRReturn) -> None:
		if instr.value is not None:
			self._load_value(instr.value, "%rax")
		self._emit_epilogue()

	def _gen_alloc(self, instr: IRAlloc) -> None:
		aligned = self._align16(instr.size)
		self._emit_instr(f"subq ${aligned}, %rsp")
		self._emit_instr("movq %rsp, %rax")
		self._store_to_temp("%rax", instr.dest)

	def _gen_bulk_copy(self, instr: IRBulkCopy) -> None:
		"""Emit bulk memory copy using rep movsb or unrolled movq/movb sequences."""
		size = instr.size
		if size <= 0:
			return
		# Load source into %rsi, dest into %rdi
		self._load_value(instr.src_addr, "%rsi")
		self._load_value(instr.dest_addr, "%rdi")
		if size >= 16:
			# Use rep movsb for larger copies
			self._emit_instr(f"movq ${size}, %rcx")
			self._emit_instr("rep movsb")
		else:
			# Unrolled copy: movq for 8-byte chunks, then movl/movw/movb for remainder
			offset = 0
			while offset + 8 <= size:
				self._emit_instr(f"movq {offset}(%rsi), %rax")
				self._emit_instr(f"movq %rax, {offset}(%rdi)")
				offset += 8
			if offset + 4 <= size:
				self._emit_instr(f"movl {offset}(%rsi), %eax")
				self._emit_instr(f"movl %eax, {offset}(%rdi)")
				offset += 4
			if offset + 2 <= size:
				self._emit_instr(f"movw {offset}(%rsi), %ax")
				self._emit_instr(f"movw %ax, {offset}(%rdi)")
				offset += 2
			if offset < size:
				self._emit_instr(f"movb {offset}(%rsi), %al")
				self._emit_instr(f"movb %al, {offset}(%rdi)")

	def _gen_addr_of(self, instr: IRAddrOf) -> None:
		offset = self._get_offset(instr.source.name)
		self._emit_instr(f"leaq {offset}(%rbp), %rax")
		self._store_to_temp("%rax", instr.dest)

	# ------------------------------------------------------------------
	# Float/SSE instruction generators
	# ------------------------------------------------------------------

	def _gen_float_binop(self, instr: IRBinOp) -> None:
		suffix = _ss_sd(instr.ir_type)
		self._load_float_value(instr.left, "%xmm0", instr.ir_type)
		self._load_float_value(instr.right, "%xmm1", instr.ir_type)

		op = instr.op
		if op == "+":
			self._emit_instr(f"add{suffix} %xmm1, %xmm0")
		elif op == "-":
			self._emit_instr(f"sub{suffix} %xmm1, %xmm0")
		elif op == "*":
			self._emit_instr(f"mul{suffix} %xmm1, %xmm0")
		elif op == "/":
			self._emit_instr(f"div{suffix} %xmm1, %xmm0")
		elif op in ("<", ">", "<=", ">=", "==", "!="):
			sd = _s_d(instr.ir_type)
			self._emit_instr(f"ucomis{sd} %xmm1, %xmm0")
			cmp_map = {
				"<": "setb", ">": "seta", "<=": "setbe", ">=": "setae",
				"==": "sete", "!=": "setne",
			}
			self._emit_instr(f"{cmp_map[op]} %al")
			self._emit_instr("movzbq %al, %rax")
			self._store_to_temp("%rax", instr.dest)
			return
		else:
			raise ValueError(f"Unsupported float binary operator: {op}")

		self._store_float_to_temp("%xmm0", instr.dest, instr.ir_type)

	def _gen_float_unaryop(self, instr: IRUnaryOp) -> None:
		suffix = _ss_sd(instr.ir_type)
		self._load_float_value(instr.operand, "%xmm0", instr.ir_type)

		if instr.op == "-":
			# Negate: 0.0 - x
			sd = _s_d(instr.ir_type)
			self._emit_instr(f"xorp{sd} %xmm1, %xmm1")
			self._emit_instr(f"sub{suffix} %xmm0, %xmm1")
			self._emit_instr(f"movap{sd} %xmm1, %xmm0")
			self._store_float_to_temp("%xmm0", instr.dest, instr.ir_type)
		elif instr.op == "!":
			# Logical not: compare against 0.0, produce integer 0 or 1
			sd = _s_d(instr.ir_type)
			self._emit_instr(f"xorp{sd} %xmm1, %xmm1")
			self._emit_instr(f"ucomis{sd} %xmm1, %xmm0")
			self._emit_instr("sete %al")
			self._emit_instr("setnp %cl")
			self._emit_instr("andb %cl, %al")
			self._emit_instr("movzbq %al, %rax")
			self._store_to_temp("%rax", instr.dest)
		else:
			raise ValueError(f"Unsupported float unary operator: {instr.op}")

	def _gen_float_copy(self, instr: IRCopy) -> None:
		self._load_float_value(instr.source, "%xmm0", instr.ir_type)
		self._store_float_to_temp("%xmm0", instr.dest, instr.ir_type)

	def _gen_float_store(self, instr: IRStore) -> None:
		suffix = _ss_sd(instr.ir_type)
		self._load_float_value(instr.value, "%xmm0", instr.ir_type)
		self._load_value(instr.address, "%rax")
		self._emit_instr(f"mov{suffix} %xmm0, (%rax)")

	def _gen_float_return(self, instr: IRReturn) -> None:
		if instr.value is not None:
			self._load_float_value(instr.value, "%xmm0", instr.ir_type)
		self._emit_epilogue()

	def _gen_convert(self, instr: IRConvert) -> None:
		"""Generate type conversion instructions."""
		from_type = instr.from_type
		to_type = instr.to_type

		if from_type == IRType.INT and to_type == IRType.FLOAT:
			self._load_value(instr.source, "%rax")
			self._emit_instr("cvtsi2ss %eax, %xmm0")
			self._store_float_to_temp("%xmm0", instr.dest, IRType.FLOAT)
		elif from_type == IRType.INT and to_type == IRType.DOUBLE:
			self._load_value(instr.source, "%rax")
			self._emit_instr("cvtsi2sd %eax, %xmm0")
			self._store_float_to_temp("%xmm0", instr.dest, IRType.DOUBLE)
		elif from_type == IRType.FLOAT and to_type == IRType.INT:
			self._load_float_value(instr.source, "%xmm0", IRType.FLOAT)
			self._emit_instr("cvttss2si %xmm0, %eax")
			self._emit_instr("movslq %eax, %rax")
			self._store_to_temp("%rax", instr.dest)
		elif from_type == IRType.DOUBLE and to_type == IRType.INT:
			self._load_float_value(instr.source, "%xmm0", IRType.DOUBLE)
			self._emit_instr("cvttsd2si %xmm0, %eax")
			self._emit_instr("movslq %eax, %rax")
			self._store_to_temp("%rax", instr.dest)
		elif from_type == IRType.FLOAT and to_type == IRType.DOUBLE:
			self._load_float_value(instr.source, "%xmm0", IRType.FLOAT)
			self._emit_instr("cvtss2sd %xmm0, %xmm0")
			self._store_float_to_temp("%xmm0", instr.dest, IRType.DOUBLE)
		elif from_type == IRType.DOUBLE and to_type == IRType.FLOAT:
			self._load_float_value(instr.source, "%xmm0", IRType.DOUBLE)
			self._emit_instr("cvtsd2ss %xmm0, %xmm0")
			self._store_float_to_temp("%xmm0", instr.dest, IRType.FLOAT)
		elif to_type in (IRType.BOOL, IRType.CHAR, IRType.SHORT) and from_type in (IRType.INT, IRType.LONG, IRType.POINTER, IRType.CHAR, IRType.SHORT, IRType.BOOL):
			# Integer narrowing: truncate to target width
			self._load_value(instr.source, "%rax")
			self._truncate_narrow(to_type, is_unsigned=instr.is_unsigned)
			self._store_to_temp("%rax", instr.dest)
		elif from_type in (IRType.BOOL, IRType.CHAR, IRType.SHORT) and to_type in (IRType.INT, IRType.LONG):
			# Integer widening: sign/zero-extend from narrow type
			self._load_value(instr.source, "%rax")
			self._truncate_narrow(from_type, is_unsigned=instr.is_unsigned)
			self._store_to_temp("%rax", instr.dest)
		else:
			# Fallback: just copy
			self._load_value(instr.source, "%rax")
			self._store_to_temp("%rax", instr.dest)

	# ------------------------------------------------------------------
	# Variadic argument instruction generators
	# ------------------------------------------------------------------

	def _gen_va_start(self, instr: IRVaStart) -> None:
		"""Initialize va_list struct: {gp_offset(4), fp_offset(4), overflow_arg_area(8), reg_save_area(8)}."""
		self._load_value(instr.ap_addr, "%rax")
		# gp_offset = num_named_gp * 8
		gp_offset = instr.num_named_gp * 8
		self._emit_instr(f"movl ${gp_offset}, (%rax)")
		# fp_offset = 48 (6 GP regs * 8 = 48, all FP slots "used")
		self._emit_instr("movl $48, 4(%rax)")
		# overflow_arg_area = first stack arg = 16(%rbp)
		self._emit_instr("leaq 16(%rbp), %rcx")
		self._emit_instr("movq %rcx, 8(%rax)")
		# reg_save_area = address of saved GP registers
		self._emit_instr(f"leaq {self._reg_save_area_offset}(%rbp), %rcx")
		self._emit_instr("movq %rcx, 16(%rax)")

	def _gen_va_arg(self, instr: IRVaArg) -> None:
		"""Fetch next variadic argument from va_list struct."""
		lbl = self._va_label_counter
		self._va_label_counter += 1
		overflow_label = f".Lva_overflow_{lbl}"
		done_label = f".Lva_done_{lbl}"

		# Load va_list struct address into %rax
		self._load_value(instr.ap_addr, "%rax")
		# Load gp_offset
		self._emit_instr("movl (%rax), %ecx")
		# If gp_offset >= 48, go to overflow path
		self._emit_instr("cmpl $48, %ecx")
		self._emit_instr(f"jae {overflow_label}")
		# Register path: load from reg_save_area + gp_offset
		self._emit_instr("movslq %ecx, %rcx")
		self._emit_instr("addq 16(%rax), %rcx")
		self._emit_instr("movq (%rcx), %rdx")
		# Increment gp_offset by 8
		self._emit_instr("addl $8, (%rax)")
		self._emit_instr(f"jmp {done_label}")
		# Overflow path: load from overflow_arg_area
		self._emit(f"{overflow_label}:")
		self._load_value(instr.ap_addr, "%rax")
		self._emit_instr("movq 8(%rax), %rcx")
		self._emit_instr("movq (%rcx), %rdx")
		# Increment overflow_arg_area by 8
		self._emit_instr("addq $8, 8(%rax)")
		self._emit(f"{done_label}:")
		self._store_to_temp("%rdx", instr.dest)

	def _gen_va_copy(self, instr: IRVaCopy) -> None:
		"""Copy 24-byte va_list struct from src to dest."""
		self._load_value(instr.src_addr, "%rsi")
		self._load_value(instr.dest_addr, "%rdi")
		# Copy 24 bytes (3 quadwords)
		self._emit_instr("movq (%rsi), %rax")
		self._emit_instr("movq %rax, (%rdi)")
		self._emit_instr("movq 8(%rsi), %rax")
		self._emit_instr("movq %rax, 8(%rdi)")
		self._emit_instr("movq 16(%rsi), %rax")
		self._emit_instr("movq %rax, 16(%rdi)")
