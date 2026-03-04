"""x86-64 code generator from three-address IR (AT&T syntax, System V AMD64 ABI)."""

from __future__ import annotations

import struct

from compiler.ir import (
	IRAlloc,
	IRBinOp,
	IRCall,
	IRCondJump,
	IRConst,
	IRConvert,
	IRCopy,
	IRFloatConst,
	IRFunction,
	IRGlobalRef,
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
		]
		if initialized:
			self._emit(".section .data")
			for g in initialized:
				if g.storage_class != "static":
					self._emit(f".globl {g.name}")
				self._emit(f"{g.name}:")
				if g.initializer_values:
					directive = {
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
				elif g.ir_type == IRType.CHAR:
					self._emit_instr(f".byte {g.initializer}")
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
			and g.storage_class != "extern"
		]
		if uninitialized:
			self._emit(".section .bss")
			for g in uninitialized:
				if g.storage_class != "static":
					self._emit(f".globl {g.name}")
				self._emit(f"{g.name}:")
				if g.ir_type == IRType.CHAR:
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

		# Reserve slots for callee-saved register saves
		used_callee = set(self._reg_map.values())
		for reg in _CALLEE_SAVED_ALLOC_REGS:
			if reg in used_callee:
				self._stack_size += 8
				self._callee_save_offsets.append((reg, -self._stack_size))

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
		elif isinstance(instr, IRAlloc):
			temps.add(instr.dest.name)
		elif isinstance(instr, IRCondJump):
			self._collect_value_temp(instr.condition, temps)
		elif isinstance(instr, IRConvert):
			temps.add(instr.dest.name)
			self._collect_value_temp(instr.source, temps)

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
		elif isinstance(instr, IRAlloc):
			self._gen_alloc(instr)
		elif isinstance(instr, IRConvert):
			self._gen_convert(instr)
		elif isinstance(instr, IRParam):
			pass  # args are conveyed via IRCall.args; IRParam is a no-op here
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

		self._store_to_temp("%rax", instr.dest)

	def _gen_copy(self, instr: IRCopy) -> None:
		self._load_value(instr.source, "%rax")
		self._store_to_temp("%rax", instr.dest)

	def _gen_load(self, instr: IRLoad) -> None:
		self._load_value(instr.address, "%rax")
		if _is_float(instr.ir_type):
			suffix = _ss_sd(instr.ir_type)
			self._emit_instr(f"mov{suffix} (%rax), %xmm0")
			self._store_float_to_temp("%xmm0", instr.dest, instr.ir_type)
		elif instr.ir_type == IRType.CHAR:
			self._emit_instr("movzbl (%rax), %eax")
			self._store_to_temp("%rax", instr.dest)
		elif instr.ir_type == IRType.SHORT:
			self._emit_instr("movswq (%rax), %rax")
			self._store_to_temp("%rax", instr.dest)
		elif instr.ir_type == IRType.INT:
			self._emit_instr("movl (%rax), %eax")
			self._emit_instr("movslq %eax, %rax")
			self._store_to_temp("%rax", instr.dest)
		else:
			self._emit_instr("movq (%rax), %rax")
			self._store_to_temp("%rax", instr.dest)

	def _gen_store(self, instr: IRStore) -> None:
		self._load_value(instr.value, "%rcx")
		self._load_value(instr.address, "%rax")
		if instr.ir_type == IRType.CHAR:
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
		else:
			raise ValueError(f"Unsupported float unary operator: {instr.op}")

		self._store_float_to_temp("%xmm0", instr.dest, instr.ir_type)

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
		else:
			# Fallback: just copy
			self._load_value(instr.source, "%rax")
			self._store_to_temp("%rax", instr.dest)
