"""Tests for enum constants in switch case labels."""

import pytest

from compiler.ast_nodes import (
	BreakStmt,
	CaseClause,
	CompoundStmt,
	EnumConstant,
	EnumDecl,
	FunctionDecl,
	Identifier,
	IntLiteral,
	Program,
	SourceLocation,
	SwitchStmt,
	TypeSpec,
	VarDecl,
)
from compiler.semantic import SemanticAnalyzer, SemanticError


def loc(line: int = 1, col: int = 1) -> SourceLocation:
	return SourceLocation(line=line, col=col)


def int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def void_type() -> TypeSpec:
	return TypeSpec(base_type="void")


def _make_enum_decl(name: str, constants: list[tuple[str, int | None]]) -> EnumDecl:
	"""Build an EnumDecl node from a list of (name, optional_value) tuples."""
	enum_consts = []
	for cname, cval in constants:
		ec = EnumConstant(
			loc=loc(),
			name=cname,
			value=IntLiteral(loc=loc(), value=cval) if cval is not None else None,
		)
		enum_consts.append(ec)
	return EnumDecl(loc=loc(), name=name, constants=enum_consts)


def _make_switch_with_enum_cases(
	enum_decl: EnumDecl,
	case_values: list[str | int | None],
) -> Program:
	"""Build a program with an enum, a function, and a switch using given case values.

	case_values entries:
	  - str  -> Identifier (enum constant)
	  - int  -> IntLiteral
	  - None -> default case
	"""
	cases = []
	for cv in case_values:
		if cv is None:
			case_node = CaseClause(
				loc=loc(),
				value=None,
				statements=[BreakStmt(loc=loc())],
			)
		elif isinstance(cv, str):
			case_node = CaseClause(
				loc=loc(),
				value=Identifier(loc=loc(), name=cv),
				statements=[BreakStmt(loc=loc())],
			)
		else:
			case_node = CaseClause(
				loc=loc(),
				value=IntLiteral(loc=loc(), value=cv),
				statements=[BreakStmt(loc=loc())],
			)
		cases.append(case_node)

	var_decl = VarDecl(loc=loc(), name="x", type_spec=int_type(), initializer=IntLiteral(loc=loc(), value=0))
	switch = SwitchStmt(
		loc=loc(),
		expression=Identifier(loc=loc(), name="x"),
		cases=cases,
	)
	func = FunctionDecl(
		loc=loc(),
		name="test_func",
		return_type=void_type(),
		params=[],
		body=CompoundStmt(loc=loc(), statements=[var_decl, switch]),
	)
	return Program(loc=loc(), declarations=[enum_decl, func])


class TestEnumCaseLabels:
	"""Test that enum constants are accepted in switch case labels."""

	def test_basic_enum_cases(self) -> None:
		"""Switch on enum constants should pass semantic analysis."""
		enum = _make_enum_decl("Color", [("RED", 0), ("GREEN", 1), ("BLUE", 2)])
		prog = _make_switch_with_enum_cases(enum, ["RED", "GREEN", "BLUE"])
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)

	def test_enum_with_default(self) -> None:
		"""Enum cases mixed with default should work."""
		enum = _make_enum_decl("Color", [("RED", 0), ("GREEN", 1), ("BLUE", 2)])
		prog = _make_switch_with_enum_cases(enum, ["RED", None])
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)

	def test_enum_auto_values(self) -> None:
		"""Enum constants with auto-assigned values should work."""
		enum = _make_enum_decl("Dir", [("UP", None), ("DOWN", None), ("LEFT", None), ("RIGHT", None)])
		prog = _make_switch_with_enum_cases(enum, ["UP", "DOWN", "LEFT", "RIGHT"])
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)

	def test_duplicate_enum_case_detected(self) -> None:
		"""Using the same enum constant twice should raise an error."""
		enum = _make_enum_decl("Color", [("RED", 0), ("GREEN", 1)])
		prog = _make_switch_with_enum_cases(enum, ["RED", "RED"])
		analyzer = SemanticAnalyzer()
		with pytest.raises(SemanticError, match="duplicate case value"):
			analyzer.analyze(prog)

	def test_duplicate_enum_same_int_value(self) -> None:
		"""Two different enum constants with the same integer value should be detected as duplicates."""
		enum = _make_enum_decl("Status", [("OK", 0), ("SUCCESS", 0)])
		prog = _make_switch_with_enum_cases(enum, ["OK", "SUCCESS"])
		analyzer = SemanticAnalyzer()
		with pytest.raises(SemanticError, match="duplicate case value"):
			analyzer.analyze(prog)

	def test_enum_and_int_literal_mixed(self) -> None:
		"""Mix of enum constants and int literals in the same switch should work."""
		enum = _make_enum_decl("Code", [("A", 10), ("B", 20)])
		prog = _make_switch_with_enum_cases(enum, ["A", 5, "B"])
		analyzer = SemanticAnalyzer()
		analyzer.analyze(prog)

	def test_enum_and_int_literal_duplicate_value(self) -> None:
		"""Enum constant and int literal with the same value should be detected as duplicate."""
		enum = _make_enum_decl("Code", [("A", 5)])
		prog = _make_switch_with_enum_cases(enum, ["A", 5])
		analyzer = SemanticAnalyzer()
		with pytest.raises(SemanticError, match="duplicate case value"):
			analyzer.analyze(prog)

	def test_non_constant_identifier_rejected(self) -> None:
		"""A regular variable (not an enum constant) in a case label should be rejected."""
		enum = _make_enum_decl("Empty", [])
		# The switch references "y" which is a variable, not an enum constant
		var_y = VarDecl(loc=loc(), name="y", type_spec=int_type(), initializer=IntLiteral(loc=loc(), value=1))
		var_x = VarDecl(loc=loc(), name="x", type_spec=int_type(), initializer=IntLiteral(loc=loc(), value=0))
		bad_case = CaseClause(
			loc=loc(),
			value=Identifier(loc=loc(), name="y"),
			statements=[BreakStmt(loc=loc())],
		)
		switch = SwitchStmt(
			loc=loc(),
			expression=Identifier(loc=loc(), name="x"),
			cases=[bad_case],
		)
		func = FunctionDecl(
			loc=loc(),
			name="test_func",
			return_type=void_type(),
			params=[],
			body=CompoundStmt(loc=loc(), statements=[var_y, var_x, switch]),
		)
		prog = Program(loc=loc(), declarations=[enum, func])
		analyzer = SemanticAnalyzer()
		with pytest.raises(SemanticError, match="case expression must be a constant integer"):
			analyzer.analyze(prog)

	def test_undeclared_identifier_in_case_rejected(self) -> None:
		"""An undeclared identifier in a case label should be rejected."""
		enum = _make_enum_decl("Color", [("RED", 0)])
		prog = _make_switch_with_enum_cases(enum, ["UNKNOWN"])
		analyzer = SemanticAnalyzer()
		with pytest.raises(SemanticError, match="case expression must be a constant integer"):
			analyzer.analyze(prog)
