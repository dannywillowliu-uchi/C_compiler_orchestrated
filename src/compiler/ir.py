"""Intermediate representation as three-address code for the C compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class IRType(Enum):
	"""Primitive types in the IR."""
	BOOL = auto()
	INT = auto()
	CHAR = auto()
	SHORT = auto()
	LONG = auto()
	VOID = auto()
	POINTER = auto()
	FLOAT = auto()
	DOUBLE = auto()


def ir_type_byte_width(ir_type: IRType) -> int:
	"""Return the byte width for a given IR type."""
	_widths = {
		IRType.BOOL: 1,
		IRType.CHAR: 1,
		IRType.SHORT: 2,
		IRType.INT: 4,
		IRType.LONG: 8,
		IRType.POINTER: 8,
		IRType.FLOAT: 4,
		IRType.DOUBLE: 8,
		IRType.VOID: 0,
	}
	return _widths[ir_type]


def ir_type_is_integer(ir_type: IRType) -> bool:
	"""Return True if the IR type is an integer type."""
	return ir_type in (IRType.BOOL, IRType.CHAR, IRType.SHORT, IRType.INT, IRType.LONG)


def ir_type_asm_suffix(ir_type: IRType) -> str:
	"""Return the AT&T assembly suffix for a given IR type."""
	_suffixes = {
		IRType.BOOL: "b",
		IRType.CHAR: "b",
		IRType.SHORT: "w",
		IRType.INT: "l",
		IRType.LONG: "q",
		IRType.POINTER: "q",
	}
	return _suffixes[ir_type]


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IRValue:
	"""Base class for IR values (operands)."""

	def __str__(self) -> str:
		raise NotImplementedError


@dataclass(frozen=True)
class IRConst(IRValue):
	"""A constant integer or character value."""
	value: int
	ir_type: IRType = IRType.INT
	is_unsigned: bool = False

	def __str__(self) -> str:
		return str(self.value)


@dataclass(frozen=True)
class IRFloatConst(IRValue):
	"""A constant floating-point value."""
	value: float
	ir_type: IRType = IRType.FLOAT

	def __str__(self) -> str:
		return str(self.value)


@dataclass(frozen=True)
class IRTemp(IRValue):
	"""A temporary variable with a unique name."""
	name: str

	def __str__(self) -> str:
		return self.name


@dataclass(frozen=True)
class IRLabel(IRValue):
	"""A label name used as a jump target."""
	name: str

	def __str__(self) -> str:
		return self.name


@dataclass(frozen=True)
class IRGlobalRef(IRValue):
	"""A reference to a global symbol (variable or string label). Loads the address."""
	name: str

	def __str__(self) -> str:
		return f"&{self.name}"


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------

@dataclass
class IRInstruction:
	"""Base class for all IR instructions."""

	def __str__(self) -> str:
		raise NotImplementedError


@dataclass
class IRBinOp(IRInstruction):
	"""dest = left op right"""
	dest: IRTemp
	left: IRValue
	op: str
	right: IRValue
	ir_type: IRType = IRType.INT
	is_unsigned: bool = False

	def __str__(self) -> str:
		return f"{self.dest} = {self.left} {self.op} {self.right}"


@dataclass
class IRUnaryOp(IRInstruction):
	"""dest = op operand"""
	dest: IRTemp
	op: str
	operand: IRValue
	ir_type: IRType = IRType.INT

	def __str__(self) -> str:
		return f"{self.dest} = {self.op} {self.operand}"


@dataclass
class IRCopy(IRInstruction):
	"""dest = source"""
	dest: IRTemp
	source: IRValue
	ir_type: IRType = IRType.INT

	def __str__(self) -> str:
		return f"{self.dest} = {self.source}"


@dataclass
class IRAddrOf(IRInstruction):
	"""dest = &source (take address of source's stack slot)"""
	dest: IRTemp
	source: IRTemp

	def __str__(self) -> str:
		return f"{self.dest} = &{self.source}"


@dataclass
class IRLoad(IRInstruction):
	"""dest = *address"""
	dest: IRTemp
	address: IRValue
	ir_type: IRType = IRType.INT

	def __str__(self) -> str:
		return f"{self.dest} = *{self.address}"


@dataclass
class IRStore(IRInstruction):
	"""*address = value"""
	address: IRValue
	value: IRValue
	ir_type: IRType = IRType.INT

	def __str__(self) -> str:
		return f"*{self.address} = {self.value}"


@dataclass
class IRLabelInstr(IRInstruction):
	"""A label marker in the instruction stream."""
	name: str

	def __str__(self) -> str:
		return f"{self.name}:"


@dataclass
class IRJump(IRInstruction):
	"""Unconditional jump to target label."""
	target: str

	def __str__(self) -> str:
		return f"jump {self.target}"


@dataclass
class IRCondJump(IRInstruction):
	"""Conditional jump: if condition goto true_label else goto false_label."""
	condition: IRValue
	true_label: str
	false_label: str

	def __str__(self) -> str:
		return f"if {self.condition} goto {self.true_label} else goto {self.false_label}"


@dataclass
class IRCall(IRInstruction):
	"""dest = call function_name(args)"""
	dest: Optional[IRTemp]
	function_name: str
	args: list[IRValue] = field(default_factory=list)
	arg_types: list[IRType] = field(default_factory=list)
	return_type: IRType = IRType.INT
	indirect: bool = False
	func_value: Optional[IRValue] = None

	def __str__(self) -> str:
		args_str = ", ".join(str(a) for a in self.args)
		if self.indirect and self.func_value is not None:
			target = f"*{self.func_value}"
		else:
			target = self.function_name
		if self.dest is not None:
			return f"{self.dest} = call {target}({args_str})"
		return f"call {target}({args_str})"


@dataclass
class IRReturn(IRInstruction):
	"""Return from function with optional value."""
	value: Optional[IRValue] = None
	ir_type: IRType = IRType.INT

	def __str__(self) -> str:
		if self.value is not None:
			return f"return {self.value}"
		return "return"


@dataclass
class IRParam(IRInstruction):
	"""Push a parameter for an upcoming call."""
	value: IRValue

	def __str__(self) -> str:
		return f"param {self.value}"


@dataclass
class IRConvert(IRInstruction):
	"""Type conversion: dest = convert source from from_type to to_type."""
	dest: IRTemp
	source: IRValue
	from_type: IRType = IRType.INT
	to_type: IRType = IRType.FLOAT

	def __str__(self) -> str:
		return f"{self.dest} = convert {self.source} {self.from_type.name}->{self.to_type.name}"


@dataclass
class IRAlloc(IRInstruction):
	"""Stack allocation: dest = alloc size bytes."""
	dest: IRTemp
	size: int

	def __str__(self) -> str:
		return f"{self.dest} = alloc {self.size}"


# ---------------------------------------------------------------------------
# Program structure
# ---------------------------------------------------------------------------

@dataclass
class IRGlobalVar:
	"""A global variable declaration."""
	name: str
	ir_type: IRType
	initializer: Optional[int] = None  # None => uninitialized (.bss)
	initializer_values: list[int] = field(default_factory=list)
	total_size: int = 0  # Total allocation size for arrays/structs
	storage_class: Optional[str] = None  # "static", "extern", or None (default/global)
	float_initializer: Optional[float] = None  # For float/double global initializers
	string_label: Optional[str] = None  # For string pointer global initializers

	def __str__(self) -> str:
		if self.initializer_values:
			vals = ", ".join(str(v) for v in self.initializer_values)
			return f"global {self.ir_type.name} {self.name} = {{{vals}}}"
		init = f" = {self.initializer}" if self.initializer is not None else ""
		return f"global {self.ir_type.name} {self.name}{init}"


@dataclass
class IRStringData:
	"""A string literal stored in read-only data."""
	label: str
	value: str

	def __str__(self) -> str:
		return f'{self.label}: .string "{self.value}"'


@dataclass
class IRFunction:
	"""A function in the IR: name, parameters, body, and return type."""
	name: str
	params: list[IRTemp]
	body: list[IRInstruction]
	return_type: IRType
	param_types: list[IRType] = field(default_factory=list)
	storage_class: Optional[str] = None  # "static", "extern", or None (default/global)
	is_prototype: bool = False  # True if declaration-only (no body)

	def __str__(self) -> str:
		params_str = ", ".join(str(p) for p in self.params)
		header = f"function {self.name}({params_str}) -> {self.return_type.name}"
		lines = [header]
		for instr in self.body:
			lines.append(f"  {instr}")
		return "\n".join(lines)


@dataclass
class IRProgram:
	"""Top-level container: functions, globals, and string data."""
	functions: list[IRFunction] = field(default_factory=list)
	globals: list[IRGlobalVar] = field(default_factory=list)
	string_data: list[IRStringData] = field(default_factory=list)

	def __str__(self) -> str:
		parts: list[str] = []
		for g in self.globals:
			parts.append(str(g))
		for s in self.string_data:
			parts.append(str(s))
		for f in self.functions:
			parts.append(str(f))
		return "\n\n".join(parts)
