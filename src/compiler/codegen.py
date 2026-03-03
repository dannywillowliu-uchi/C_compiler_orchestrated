"""x86-64 code generator from three-address IR (AT&T syntax, System V AMD64 ABI)."""

from __future__ import annotations

from compiler.ir import (
	IRAlloc,
	IRBinOp,
	IRCall,
	IRCondJump,
	IRConst,
	IRCopy,
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


class CodeGenerator:
	"""Generates x86-64 assembly (AT&T syntax) from an IRProgram."""

	def __init__(self) -> None:
		self._lines: list[str] = []
		self._stack_map: dict[str, int] = {}
		self._stack_size: int = 0

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------

	def generate(self, program: IRProgram) -> str:
		"""Generate assembly for an entire IR program."""
		self._lines = []

		# Emit .data section for initialized globals
		initialized = [g for g in program.globals if g.initializer is not None]
		if initialized:
			self._emit(".section .data")
			for g in initialized:
				self._emit(f".globl {g.name}")
				self._emit(f"{g.name}:")
				if g.ir_type == IRType.CHAR:
					self._emit_instr(f".byte {g.initializer}")
				else:
					self._emit_instr(f".quad {g.initializer}")

		# Emit .rodata section for string literals
		if program.string_data:
			self._emit(".section .rodata")
			for s in program.string_data:
				self._emit(f"{s.label}:")
				self._emit_instr(f'.asciz "{s.value}"')

		# Emit .bss section for uninitialized globals
		uninitialized = [g for g in program.globals if g.initializer is None]
		if uninitialized:
			self._emit(".section .bss")
			for g in uninitialized:
				self._emit(f".globl {g.name}")
				self._emit(f"{g.name}:")
				if g.ir_type == IRType.CHAR:
					self._emit_instr(".zero 1")
				else:
					self._emit_instr(".zero 8")

		# Emit .text section for functions
		self._emit(".section .text")
		for func in program.functions:
			self._generate_function(func)
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

	def _load_value(self, value: IRValue, reg: str) -> None:
		"""Emit code to load an IRValue into *reg*."""
		if isinstance(value, IRConst):
			self._emit_instr(f"movq ${value.value}, {reg}")
		elif isinstance(value, IRTemp):
			offset = self._get_offset(value.name)
			self._emit_instr(f"movq {offset}(%rbp), {reg}")
		elif isinstance(value, IRGlobalRef):
			self._emit_instr(f"leaq {value.name}(%rip), {reg}")
		else:
			raise ValueError(f"Unsupported IRValue type: {type(value).__name__}")

	def _store_to_temp(self, reg: str, dest: IRTemp) -> None:
		"""Store *reg* into *dest*'s stack slot."""
		offset = self._get_offset(dest.name)
		self._emit_instr(f"movq {reg}, {offset}(%rbp)")

	# ------------------------------------------------------------------
	# Temp allocation (scan phase)
	# ------------------------------------------------------------------

	def _allocate_temps(self, func: IRFunction) -> None:
		"""Scan *func* and assign a stack slot (-8, -16, ...) to every IRTemp."""
		temps: set[str] = set()
		for p in func.params:
			temps.add(p.name)
		for instr in func.body:
			self._collect_temps_from_instr(instr, temps)
		self._stack_size = 0
		self._stack_map = {}
		for name in sorted(temps):
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
		elif isinstance(instr, IRReturn):
			if instr.value is not None:
				self._collect_value_temp(instr.value, temps)
		elif isinstance(instr, IRParam):
			self._collect_value_temp(instr.value, temps)
		elif isinstance(instr, IRAlloc):
			temps.add(instr.dest.name)
		elif isinstance(instr, IRCondJump):
			self._collect_value_temp(instr.condition, temps)

	@staticmethod
	def _collect_value_temp(value: IRValue, temps: set[str]) -> None:
		if isinstance(value, IRTemp):
			temps.add(value.name)

	# ------------------------------------------------------------------
	# Function-level generation
	# ------------------------------------------------------------------

	def _generate_function(self, func: IRFunction) -> None:
		self._allocate_temps(func)

		frame_size = self._align16(self._stack_size)

		# Header
		self._emit(f".globl {func.name}")
		self._emit(f"{func.name}:")

		# Prologue
		self._emit_instr("pushq %rbp")
		self._emit_instr("movq %rsp, %rbp")
		if frame_size > 0:
			self._emit_instr(f"subq ${frame_size}, %rsp")

		# Copy register-passed params to their stack slots
		for i, param in enumerate(func.params):
			offset = self._get_offset(param.name)
			if i < len(_ARG_REGS):
				self._emit_instr(f"movq {_ARG_REGS[i]}, {offset}(%rbp)")
			else:
				# Params beyond the first 6 sit above the saved rbp/return addr
				src_offset = 16 + (i - len(_ARG_REGS)) * 8
				self._emit_instr(f"movq {src_offset}(%rbp), %rax")
				self._emit_instr(f"movq %rax, {offset}(%rbp)")

		# Body
		for instr in func.body:
			self._generate_instruction(instr)

	# ------------------------------------------------------------------
	# Instruction dispatch
	# ------------------------------------------------------------------

	def _generate_instruction(self, instr: object) -> None:
		if isinstance(instr, IRBinOp):
			self._gen_binop(instr)
		elif isinstance(instr, IRUnaryOp):
			self._gen_unaryop(instr)
		elif isinstance(instr, IRCopy):
			self._gen_copy(instr)
		elif isinstance(instr, IRLoad):
			self._gen_load(instr)
		elif isinstance(instr, IRStore):
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
			self._gen_return(instr)
		elif isinstance(instr, IRAlloc):
			self._gen_alloc(instr)
		elif isinstance(instr, IRParam):
			pass  # args are conveyed via IRCall.args; IRParam is a no-op here
		else:
			raise ValueError(f"Unknown instruction type: {type(instr).__name__}")

	# ------------------------------------------------------------------
	# Individual instruction generators
	# ------------------------------------------------------------------

	def _gen_binop(self, instr: IRBinOp) -> None:
		self._load_value(instr.left, "%rax")
		self._load_value(instr.right, "%rcx")

		op = instr.op
		if op == "+":
			self._emit_instr("addq %rcx, %rax")
		elif op == "-":
			self._emit_instr("subq %rcx, %rax")
		elif op == "*":
			self._emit_instr("imulq %rcx, %rax")
		elif op == "/":
			self._emit_instr("cqto")
			self._emit_instr("idivq %rcx")
		elif op == "%":
			self._emit_instr("cqto")
			self._emit_instr("idivq %rcx")
			self._emit_instr("movq %rdx, %rax")
		elif op in ("<", ">", "<=", ">=", "==", "!="):
			self._emit_instr("cmpq %rcx, %rax")
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
		self._emit_instr("movq (%rax), %rax")
		self._store_to_temp("%rax", instr.dest)

	def _gen_store(self, instr: IRStore) -> None:
		self._load_value(instr.value, "%rcx")
		self._load_value(instr.address, "%rax")
		self._emit_instr("movq %rcx, (%rax)")

	def _gen_condjump(self, instr: IRCondJump) -> None:
		self._load_value(instr.condition, "%rax")
		self._emit_instr("cmpq $0, %rax")
		self._emit_instr(f"jne {instr.true_label}")
		self._emit_instr(f"jmp {instr.false_label}")

	def _gen_call(self, instr: IRCall) -> None:
		stack_args = instr.args[len(_ARG_REGS):]
		num_stack_args = len(stack_args)

		# Pad for 16-byte alignment when an odd number of args are pushed
		needs_padding = num_stack_args > 0 and num_stack_args % 2 != 0
		if needs_padding:
			self._emit_instr("subq $8, %rsp")

		# Push stack args in reverse order (rightmost first)
		for arg in reversed(stack_args):
			self._load_value(arg, "%rax")
			self._emit_instr("pushq %rax")

		# Load register args
		for i, arg in enumerate(instr.args[: len(_ARG_REGS)]):
			self._load_value(arg, _ARG_REGS[i])

		self._emit_instr(f"call {instr.function_name}")

		# Clean up pushed stack args (+ alignment padding)
		if num_stack_args > 0:
			cleanup = num_stack_args * 8
			if needs_padding:
				cleanup += 8
			self._emit_instr(f"addq ${cleanup}, %rsp")

		if instr.dest is not None:
			self._store_to_temp("%rax", instr.dest)

	def _gen_return(self, instr: IRReturn) -> None:
		if instr.value is not None:
			self._load_value(instr.value, "%rax")
		# Epilogue
		self._emit_instr("movq %rbp, %rsp")
		self._emit_instr("popq %rbp")
		self._emit_instr("ret")

	def _gen_alloc(self, instr: IRAlloc) -> None:
		aligned = self._align16(instr.size)
		self._emit_instr(f"subq ${aligned}, %rsp")
		self._emit_instr("movq %rsp, %rax")
		self._store_to_temp("%rax", instr.dest)
